# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Construct Diffusers target components whose weights stay in their checkpoint.

A component built here has the native module structure, real buffers and one
``CheckpointParameter`` per parameter: a stride-0 placeholder of the final shape
and load dtype that owns no weight storage and records where its value lives.
``ParameterSource.materialize`` later reads only each rank's local block, so no
rank holds the full root on the host either.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
from accelerate import init_empty_weights
from diffusers.models.modeling_utils import no_init_weights
from safetensors import safe_open
from torch import nn

__all__ = [
    "CheckpointParameter",
    "CheckpointSlices",
    "SliceReader",
    "checkpoint_reader",
    "construct_from_checkpoint",
    "requests_component",
]

_WEIGHTS = "diffusion_pytorch_model"


class CheckpointSlices:
    """Lazy key-to-file view of one Diffusers safetensors component folder.

    Args:
        folder: Component directory holding a single safetensors file or a
            sharded index.
        variant: Optional Diffusers weight variant, for example ``"fp16"``.

    Raises:
        FileNotFoundError: If the folder holds no safetensors weights for the variant.
    """

    def __init__(self, folder: Union[str, Path], variant: Optional[str] = None) -> None:
        self.folder = Path(folder)
        stem = _WEIGHTS if variant is None else f"{_WEIGHTS}.{variant}"
        index = self.folder / f"{stem}.safetensors.index.json"
        single = self.folder / f"{stem}.safetensors"
        if index.is_file():
            weight_map = json.loads(index.read_text())["weight_map"]
            self._files = {key: self.folder / name for key, name in weight_map.items()}
        elif single.is_file():
            with safe_open(str(single), framework="pt") as handle:
                self._files = {key: single for key in handle.keys()}
        else:
            raise FileNotFoundError(
                f"expected {index.name} or {single.name} in component folder {self.folder}"
            )
        self._handles: Dict[Path, Any] = {}

    def keys(self) -> Sequence[str]:
        return tuple(self._files)

    def _handle(self, key: str):
        path = self._files[key]
        handle = self._handles.get(path)
        if handle is None:
            handle = self._handles[path] = safe_open(str(path), framework="pt")
        return handle

    def read(self, key: str, offset: Sequence[int], shape: Sequence[int]) -> torch.Tensor:
        """Read one block of a stored tensor in its checkpoint dtype."""
        index = tuple(slice(start, start + size) for start, size in zip(offset, shape))
        return self._handle(key).get_slice(key)[index]

    def read_full(self, key: str) -> torch.Tensor:
        return self._handle(key).get_tensor(key)

    def __deepcopy__(self, memo):
        return self


@dataclass(frozen=True)
class SliceReader:
    """Where one parameter's value lives and the dtype the native loader gives it."""

    slices: CheckpointSlices
    key: str
    load_dtype: torch.dtype

    def read(self, offset: Sequence[int], shape: Sequence[int]) -> torch.Tensor:
        return self.slices.read(self.key, offset, shape).to(self.load_dtype)


_METADATA_ATTRIBUTES = (
    "shape",
    "dtype",
    "device",
    "layout",
    "requires_grad",
    "is_leaf",
    "is_meta",
    "is_cuda",
    "is_sparse",
    "is_quantized",
    "grad",
    "grad_fn",
    "data",
    "names",
    "ndim",
)
_METADATA_METHODS = (
    "size",
    "dim",
    "numel",
    "element_size",
    "stride",
    "is_floating_point",
    "is_complex",
    "is_contiguous",
    "untyped_storage",
    "requires_grad_",
    "get_device",
    "__len__",
)


def _metadata_functions() -> frozenset:
    functions = set()
    for name in _METADATA_ATTRIBUTES:
        descriptor = getattr(torch.Tensor, name, None)
        for accessor in ("__get__", "__set__"):
            if hasattr(descriptor, accessor):
                functions.add(getattr(descriptor, accessor))
    for name in _METADATA_METHODS:
        method = getattr(torch.Tensor, name, None)
        if method is not None:
            functions.add(method)
    # Shape-only constructors, used to build meta replacements.
    functions.update((torch.empty_like, torch.Tensor.new_empty))
    return frozenset(functions)


_METADATA = _metadata_functions()


class CheckpointParameter(nn.Parameter):
    """A storage-free parameter whose value is read from its checkpoint when materialized.

    Copies share the reader, so a copied root stays checkpoint-backed. Only
    metadata (shape, dtype, device, trainability, ``.data`` replacement) is
    available before materialization: any operation that would read the
    placeholder's zeros raises instead of silently computing with them.
    """

    reader: SliceReader

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if func in _METADATA:
            with torch._C.DisableTorchFunctionSubclass():
                return func(*args, **(kwargs or {}))
        parameter = next(arg for arg in (*args, *(kwargs or {}).values()) if isinstance(arg, cls))
        raise RuntimeError(
            f"checkpoint-backed parameter {parameter.reader.key!r} was read through "
            f"{getattr(func, '__qualname__', func)!r} before sharded materialization; its value "
            "is still in the checkpoint. Set parallelism.target_materialization: replicated "
            "for an adapter that reads target weights while loading."
        )

    def __repr__(self) -> str:
        return (
            f"CheckpointParameter(key={self.reader.key!r}, shape={tuple(self.shape)}, "
            f"dtype={self.reader.load_dtype})"
        )

    def __new__(cls, shape: torch.Size, reader: SliceReader, requires_grad: bool = True):
        placeholder = torch.zeros((), dtype=reader.load_dtype).expand(shape)
        parameter = super().__new__(cls, placeholder, requires_grad)
        parameter.reader = reader
        return parameter

    def __deepcopy__(self, memo):
        result = memo.get(id(self))
        if result is None:
            result = memo[id(self)] = type(self)(self.shape, self.reader, self.requires_grad)
        return result

    def __reduce_ex__(self, protocol):
        raise TypeError(
            f"checkpoint-backed parameter for {self.reader.key!r} has no value to serialize "
            "before materialization"
        )


def requests_component(requested: Sequence[str], name: str) -> bool:
    """Whether a target-component request names ``name`` directly or by the transformer group."""
    return name in requested or ("transformers" in requested and "transformer" in name)


def checkpoint_reader(parameter: torch.Tensor) -> Optional[SliceReader]:
    """Return the reader of a checkpoint-backed parameter, or ``None``."""
    return parameter.reader if isinstance(parameter, CheckpointParameter) else None


def _load_dtype(
    name: str,
    native: torch.dtype,
    stored: torch.dtype,
    dtype: Optional[torch.dtype],
    keep_in_fp32: Sequence[str],
    assigns: Optional[bool] = None,
) -> torch.dtype:
    """Mirror the native loader's dtype selection for one stored tensor.

    ``assigns`` is ``None`` for ``low_cpu_mem_usage`` loading, which casts through
    ``load_model_dict_into_meta``. Eager loading either assigns every stored tensor
    (keeping its dtype) or copies into the constructed tensor (keeping that dtype).
    """
    if assigns is not None:
        return stored if assigns else native
    if not stored.is_floating_point:
        return native
    if dtype is None:
        return native
    if any(module in name.split(".") for module in keep_in_fp32):
        return torch.float32
    return dtype


def construct_from_checkpoint(
    component_cls: type,
    folder: Union[str, Path],
    dtype: Optional[torch.dtype],
    *,
    variant: Optional[str] = None,
    low_cpu_mem_usage: bool = True,
) -> Tuple[Optional[nn.Module], Optional[str]]:
    """Build a native Diffusers component without loading its weights.

    Args:
        component_cls: The ``ModelMixin`` subclass the pipeline would load.
        folder: The component's checkpoint directory.
        dtype: The ``dtype`` the native loader would receive, or ``None``.
        variant: Optional Diffusers weight variant.
        low_cpu_mem_usage: The native loader mode the adapter requests; it decides
            the dtype each stored tensor receives.

    Returns:
        ``(module, None)`` on success, or ``(None, reason)`` when the checkpoint
        needs native loading: missing keys, renamed legacy blocks, quantized or
        non-safetensors weights, or mismatched shapes.
    """
    folder = Path(folder)
    try:
        slices = CheckpointSlices(folder, variant)
    except FileNotFoundError as error:
        return None, str(error)
    config = component_cls.load_config(folder)
    if config.get("quantization_config") is not None:
        return None, "quantized checkpoints keep native loading"
    # The native loader instantiates under the requested default dtype and with
    # initialization disabled, so buffers and the global RNG match its path.
    previous = torch.get_default_dtype()
    if dtype is not None:
        torch.set_default_dtype(dtype)
    try:
        with no_init_weights(), init_empty_weights(include_buffers=False):
            module = component_cls.from_config(config)
    finally:
        torch.set_default_dtype(previous)
    if any(getattr(item, "_from_deprecated_attn_block", False) for item in module.modules()):
        return None, "legacy attention blocks rename keys on load"
    keep_in_fp32 = tuple(getattr(component_cls, "_keep_in_fp32_modules", None) or ())
    if keep_in_fp32 and not low_cpu_mem_usage:
        return None, "the native loader requires low_cpu_mem_usage for _keep_in_fp32_modules"
    keys = set(slices.keys())
    assigns = None
    if not low_cpu_mem_usage:
        # Diffusers ``check_support_param_buffer_assignment``: eager loading assigns
        # stored tensors only when the first state entry already has the stored dtype.
        state = module.state_dict()
        first = next(iter(state), None)
        assigns = bool(
            getattr(component_cls, "_supports_param_buffer_assignment", True)
            and first in keys
            and _safetensors_dtype(slices._handle(first).get_slice(first).get_dtype())
            == state[first].dtype
        )
    names = dict(module.named_parameters())
    missing = sorted(name for name in names if name not in keys)
    if missing:
        return None, f"checkpoint lacks parameters {missing[:5]} ({len(missing)} total)"
    for module_name, owner in module.named_modules():
        for leaf, parameter in list(owner._parameters.items()):
            if parameter is None:
                continue
            name = f"{module_name}.{leaf}" if module_name else leaf
            stored = slices._handle(name).get_slice(name)
            stored_shape = tuple(stored.get_shape())
            if stored_shape != tuple(parameter.shape):
                return None, (
                    f"checkpoint {name!r} has shape {stored_shape}, the module expects "
                    f"{tuple(parameter.shape)}"
                )
            stored_dtype = _safetensors_dtype(stored.get_dtype())
            reader = SliceReader(
                slices,
                name,
                _load_dtype(name, parameter.dtype, stored_dtype, dtype, keep_in_fp32, assigns),
            )
            owner._parameters[leaf] = CheckpointParameter(
                parameter.shape, reader, parameter.requires_grad
            )
    for module_name, owner in module.named_modules():
        for leaf, buffer in list(owner._buffers.items()):
            name = f"{module_name}.{leaf}" if module_name else leaf
            if buffer is None or name not in keys:
                continue
            stored = slices.read_full(name)
            target = _load_dtype(name, buffer.dtype, stored.dtype, dtype, keep_in_fp32, assigns)
            owner._buffers[leaf] = stored.to(target)
    module.register_to_config(_name_or_path=str(folder))
    module.eval()
    return module, None


_SAFETENSORS_DTYPES = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


def _safetensors_dtype(code: str) -> torch.dtype:
    if code not in _SAFETENSORS_DTYPES:
        raise ValueError(f"unsupported safetensors dtype {code!r}")
    return _SAFETENSORS_DTYPES[code]
