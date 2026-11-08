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

"""Role-aware distributed component loading and preparation."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Callable, Iterator, List, Optional, Sequence, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import DistributedType
from torch.distributed.tensor import DTensor

from ..utils.logger_utils import setup_logger
from .domain import ComponentRole, LoadPlan
from .materialization import ParameterSource, detach_parameters_to_meta

logger = setup_logger(__name__)


def _unshard_fsdp2_checkpoint_replay(module: nn.Module, _inputs: Any) -> None:
    """Backport the nested-FSDP multi-forward replay fix through public APIs."""
    unshard = getattr(module, "unshard", None)
    if not callable(unshard):
        module_type = type(module).__name__
        raise TypeError(f"FSDP2 checkpoint replay hook requires unshard(), received {module_type}")
    unshard()


def configure_backend_loading(accelerator: Accelerator, adapter_class: type) -> None:
    """Apply adapter capabilities before any pretrained component is loaded."""
    if accelerator.distributed_type != DistributedType.FSDP:
        return
    plugin = accelerator.state.fsdp_plugin
    if (
        getattr(plugin, "fsdp_version", 1) >= 2
        and plugin.cpu_ram_efficient_loading
        and not getattr(adapter_class, "supports_fsdp2_cpu_efficient_loading", False)
    ):
        plugin.cpu_ram_efficient_loading = False
        os.environ["FSDP_CPU_RAM_EFFICIENT_LOADING"] = "false"
        logger.info(
            "Disabled FSDP2 CPU-efficient loading for adapter=%s; "
            "its eager/mixed component source requires replicated construction.",
            adapter_class.__name__,
        )


class BackendLoadRuntime:
    """Backend-owned loading policy for one immutable component plan."""

    def __init__(self, accelerator: Accelerator, plan: LoadPlan, adapter: Any) -> None:
        self.accelerator = accelerator
        self.plan = plan
        self.adapter = adapter
        self._verified_roots: set[str] = set()
        self._target_source: Optional[ParameterSource] = None
        self._canonical_name: Optional[Callable[[str], str]] = None

    @contextmanager
    def load_scope(self, role: ComponentRole) -> Iterator[None]:
        """Enter the third-party loading scope for one component role."""
        yield

    def bootstrap_targets(self) -> None:
        """Prepare target state that must exist before distributed wrapping."""

    def synchronize_initial_state(self, modules: Sequence[nn.Module]) -> None:
        """Let the backend establish common weights before model-parallel transforms."""

    def components_loaded(
        self,
        components: Optional[Union[str, List[str]]],
    ) -> None:
        """Finalize newly resident non-target components."""

    def prepare(self, *objects: Any) -> Any:
        """Prepare target roots and related objects through Accelerate."""
        return self.accelerator.prepare(*objects)

    def detach_targets(self, bundle: nn.Module, canonical_name: Callable[[str], str]) -> None:
        """Retain loaded target values and move the bundle's parameters to meta."""
        raise TypeError(
            "sharded target materialization requires FSDP2; "
            f"received distributed_type={self.accelerator.distributed_type}"
        )

    def _logical_names(
        self,
        components: Optional[Union[str, List[str]]],
        *,
        role: ComponentRole,
    ) -> List[str]:
        names = self.adapter._resolve_component_names(components)
        selected = []
        seen_roots = set()
        for name in names:
            request = self.plan.request_for_component(name)
            if request.role is not role or request.root in seen_roots:
                continue
            seen_roots.add(request.root)
            selected.append(request.root)
        return selected


class FSDPBackendLoadRuntime(BackendLoadRuntime):
    """FSDP loading semantics isolated from trainer business logic."""

    @property
    def cpu_ram_efficient_loading(self) -> bool:
        plugin = self.accelerator.state.fsdp_plugin
        return bool(plugin.cpu_ram_efficient_loading)

    def synchronize_initial_state(self, modules: Sequence[nn.Module]) -> None:
        """Broadcast replicated FSDP2 roots before sharding random adapter/role weights.

        FSDP2's replicated-loading path does not synchronize module state. Data
        replicas deliberately have different RNG streams, so sharding their
        independently initialized LoRA matrices would assemble a different model
        for each topology. Efficient loading retains Accelerate's state broadcast.
        """
        plugin = self.accelerator.state.fsdp_plugin
        if (
            getattr(plugin, "fsdp_version", 1) != 2
            or self.cpu_ram_efficient_loading
            or self.accelerator.num_processes <= 1
        ):
            return
        tensors = {}
        for index, module in enumerate(modules):
            for name, tensor in (*module.named_parameters(), *module.named_buffers()):
                tensors.setdefault(id(tensor), (f"{index}.{name}", tensor))
        values = list(tensors.values())
        metadata = [
            (
                name,
                tuple(value.shape),
                str(value.dtype),
                value.requires_grad,
                value.is_meta,
                type(value).__name__,
            )
            for name, value in values
        ]
        peers = [None] * self.accelerator.num_processes
        dist.all_gather_object(peers, metadata)
        if any(peer != metadata for peer in peers) or any(
            value.is_meta or isinstance(value, DTensor) for _, value in values
        ):
            raise ValueError(
                "FSDP2 initial-state synchronization requires matching, materialized, "
                "unsharded parameters and buffers on every rank"
            )
        with torch.no_grad():
            for _, tensor in values:
                value = tensor.detach().to(self.accelerator.device).contiguous()
                dist.broadcast(value, src=0)
                tensor.copy_(value)

    @contextmanager
    def load_scope(self, role: ComponentRole) -> Iterator[None]:
        if role is ComponentRole.TARGET or not self.cpu_ram_efficient_loading:
            yield
            return

        previous = os.environ.get("FSDP_CPU_RAM_EFFICIENT_LOADING")
        os.environ["FSDP_CPU_RAM_EFFICIENT_LOADING"] = "false"
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop("FSDP_CPU_RAM_EFFICIENT_LOADING", None)
            else:
                os.environ["FSDP_CPU_RAM_EFFICIENT_LOADING"] = previous

    def detach_targets(self, bundle: nn.Module, canonical_name: Callable[[str], str]) -> None:
        """Retain the bundle's loaded values and move its parameters to the meta device.

        Under CPU-RAM-efficient loading only rank 0 is guaranteed to hold loaded
        values, so materialization broadcasts its source; otherwise initial-state
        synchronization already made every rank's values identical.

        Args:
            bundle: Unprepared target bundle after initial-state synchronization.
            canonical_name: Maps prepared parameter names to their unwrapped names.

        Raises:
            RuntimeError: If targets were already detached and not yet materialized.
        """
        if self._target_source is not None:
            raise RuntimeError("target parameters were already detached and await materialization")
        source_rank = (
            0 if self.cpu_ram_efficient_loading and self.accelerator.num_processes > 1 else None
        )
        self._target_source = detach_parameters_to_meta(
            bundle, canonical_name=canonical_name, source_rank=source_rank
        )
        self._canonical_name = canonical_name

    def prepare(self, *objects: Any) -> Any:
        """Apply adapter-owned FSDP2 wrap, checkpoint, replay, and communication policies.

        Args:
            *objects: Model roots and related objects forwarded to ``Accelerator.prepare``.

        Returns:
            The prepared object or tuple returned by ``Accelerator.prepare``.

        Raises:
            TypeError: If an adapter-requested FSDP2 policy cannot be installed on the supplied
                or prepared roots.
        """
        plugin = self.accelerator.state.fsdp_plugin
        use_in_forward_checkpointing = (
            (getattr(plugin, "fsdp_version", 1) or 1) >= 2
            and bool(getattr(plugin, "activation_checkpointing", False))
            and bool(
                getattr(
                    self.adapter,
                    "fsdp2_use_in_forward_activation_checkpointing",
                    False,
                )
            )
        )
        if use_in_forward_checkpointing:
            modules = [candidate for candidate in objects if isinstance(candidate, nn.Module)]
            configure = getattr(
                self.adapter,
                "configure_fsdp2_in_forward_activation_checkpointing",
                None,
            )
            if len(modules) != 1 or not callable(configure):
                raise TypeError(
                    "adapter-owned FSDP2 activation checkpointing requires exactly one "
                    "model root and a callable configuration hook"
                )
            configured = configure(modules[0])
            if not isinstance(configured, int) or isinstance(configured, bool) or configured < 1:
                raise TypeError(
                    "adapter-owned FSDP2 activation checkpointing expected a positive "
                    f"configured block count, received {configured!r}"
                )
            plugin.activation_checkpointing = False
            logger.info(
                "Delegated FSDP2 activation checkpointing to adapter-owned in-forward "
                "block boundaries: configured=%d",
                configured,
            )

        materialize = self._target_source is not None
        group_names = self._optimizer_group_names(objects) if materialize else None
        efficient_loading = plugin.cpu_ram_efficient_loading
        try:
            self._extend_fsdp2_wrap_policy(objects)
            if materialize:
                # The bundle is on meta and Flow-Factory owns its values, so Accelerate must
                # neither move it to meta again nor reload a full state dict.
                plugin.cpu_ram_efficient_loading = False
            prepared = super().prepare(*objects)
        finally:
            if use_in_forward_checkpointing:
                plugin.activation_checkpointing = True
            plugin.cpu_ram_efficient_loading = efficient_loading

        if (getattr(plugin, "fsdp_version", 1) or 1) < 2:
            if materialize:
                raise TypeError("sharded target materialization requires FSDP2 preparation")
            return prepared

        # Workaround for https://github.com/huggingface/accelerate/issues/4511: with
        # cpu_ram_efficient_loading=False, fsdp2_prepare_model keeps an unused
        # `original_sd = model.state_dict()` alive while `fully_shard` allocates the new shards.
        # The preparation peak is then the unsharded model plus every shard. Even when that fits,
        # the allocator keeps it reserved and cuBLAS/NCCL cannot allocate on the first step.
        # Once the minimum supported Accelerate captures `original_sd` only for efficient loading,
        # the peak stays near the unsharded model and this release can be removed.
        self._release_fsdp2_preparation_cache()
        if materialize:
            self._materialize_targets(prepared, group_names)

        use_default_stream = bool(getattr(self.adapter, "fsdp2_use_default_stream_unshard", False))
        disable_backward_prefetch = bool(
            getattr(self.adapter, "fsdp2_disable_backward_prefetch", False)
        )
        if (
            not use_in_forward_checkpointing
            and not use_default_stream
            and not disable_backward_prefetch
        ):
            return prepared

        prepared_objects = prepared if isinstance(prepared, (list, tuple)) else (prepared,)
        prepared_modules = []
        for original, candidate in zip(objects, prepared_objects):
            if not isinstance(original, nn.Module):
                continue
            prepared_modules.append(candidate)

        if not prepared_modules:
            raise TypeError("FSDP2 memory policy requested without a prepared module")

        if use_in_forward_checkpointing:
            configured = 0
            newly_registered = 0
            for root in prepared_modules:
                for module in root.modules():
                    if not callable(getattr(module, "unshard", None)):
                        continue
                    configured += 1
                    if any(
                        hook is _unshard_fsdp2_checkpoint_replay
                        for hook in module._forward_pre_hooks.values()
                    ):
                        continue
                    # PyTorch <=2.10 returns early from its FSDP pre-forward hook
                    # during activation-checkpoint replay. With two forward graphs,
                    # a preceding post-backward may already have resharded this unit.
                    # Appending this hook after FSDP's hook mirrors the upstream fix
                    # and makes the public call a no-op when parameters remain full.
                    module.register_forward_pre_hook(_unshard_fsdp2_checkpoint_replay)
                    newly_registered += 1
            if configured < 1:
                raise TypeError(
                    "adapter-owned FSDP2 checkpointing requires prepared FSDPModule roots"
                )
            logger.info(
                "Enabled nested FSDP2 multi-forward checkpoint replay unshard: "
                "configured=%d, newly_registered=%d",
                configured,
                newly_registered,
            )

        if use_default_stream:
            configured = []
            for candidate in prepared_modules:
                configure = getattr(candidate, "_set_unshard_async_op", None)
                if not callable(configure):
                    raise TypeError(
                        "FSDP2 default-stream unshard requires prepared modules to expose "
                        f"_set_unshard_async_op(), received {type(candidate).__name__}"
                    )
                configure(True)
                configured.append(type(candidate).__name__)
            logger.info("Enabled FSDP2 default-stream unshard for roots: %s", configured)

        if disable_backward_prefetch:
            configured = 0
            for root in prepared_modules:
                for module in root.modules():
                    configure = getattr(module, "set_modules_to_backward_prefetch", None)
                    if not callable(configure):
                        continue
                    # FSDP2 treats an empty list as "use default next-unit prefetch".
                    # Prefetching the current, already-unsharded unit is a public-API
                    # no-op that selects explicit mode and disables that default.
                    configure([module])
                    configured += 1
            if configured < 1:
                raise TypeError(
                    "FSDP2 backward-prefetch opt-out requires prepared FSDPModule roots"
                )
            logger.info(
                "Disabled default FSDP2 next-unit backward prefetch: configured=%d",
                configured,
            )
        return prepared

    def _optimizer_group_names(self, objects: Sequence[Any]) -> List[List[str]]:
        """Record each optimizer group's parameters by canonical name before prepare."""
        modules = [candidate for candidate in objects if isinstance(candidate, nn.Module)]
        optimizers = [
            candidate for candidate in objects if isinstance(candidate, torch.optim.Optimizer)
        ]
        if len(modules) != 1 or len(optimizers) != 1:
            raise TypeError(
                "sharded target materialization prepares exactly one model root and one "
                f"optimizer; received {len(modules)} modules and {len(optimizers)} optimizers"
            )
        names = {
            id(parameter): self._canonical_name(name)
            for name, parameter in modules[0].named_parameters()
        }
        groups = []
        for index, group in enumerate(optimizers[0].param_groups):
            outside = sum(id(parameter) not in names for parameter in group["params"])
            if outside:
                raise ValueError(
                    f"optimizer group {index} owns {outside} parameters outside the detached "
                    "target bundle"
                )
            groups.append([names[id(parameter)] for parameter in group["params"]])
        return groups

    def _materialize_targets(self, prepared: Any, group_names: List[List[str]]) -> None:
        """Fill local shards, then rebuild optimizer groups from the materialized parameters.

        Accelerate remaps optimizer parameters by ``data_ptr``, which is zero for every
        meta tensor, so its remap cannot tell the detached parameters apart.
        """
        prepared_objects = prepared if isinstance(prepared, (list, tuple)) else (prepared,)
        modules = [candidate for candidate in prepared_objects if isinstance(candidate, nn.Module)]
        optimizers = [
            candidate
            for candidate in prepared_objects
            if hasattr(candidate, "param_groups") and not isinstance(candidate, nn.Module)
        ]
        if len(modules) != 1 or len(optimizers) != 1:
            raise TypeError(
                "sharded target materialization expected one prepared model and optimizer; "
                f"received {len(modules)} modules and {len(optimizers)} optimizers"
            )
        source, canonical_name = self._target_source, self._canonical_name
        self._target_source = None
        root = self.accelerator.unwrap_model(modules[0])
        report = source.materialize(root, self.accelerator.device, canonical_name=canonical_name)
        parameters = {canonical_name(name): param for name, param in root.named_parameters()}
        param_groups = optimizers[0].param_groups
        if len(param_groups) != len(group_names):
            raise ValueError(
                f"prepared optimizer has {len(param_groups)} groups, expected {len(group_names)}"
            )
        owned: set[int] = set()
        for index, (group, names) in enumerate(zip(param_groups, group_names)):
            missing = [name for name in names if name not in parameters]
            if missing:
                raise ValueError(
                    f"optimizer group {index} parameters are absent after preparation: {missing[:5]}"
                )
            group["params"] = [parameters[name] for name in names]
            for parameter in group["params"]:
                if id(parameter) in owned:
                    raise ValueError(f"optimizer group {index} repeats a materialized parameter")
                owned.add(id(parameter))
        logger.info(
            "Materialized target parameters after FSDP2 preparation: parameters=%d, "
            "local=%.2f GiB, read=%.2f GiB, reserved=%.2f GiB",
            report.parameters,
            report.local_bytes / 2**30,
            report.source_bytes / 2**30,
            _reserved_bytes(self.accelerator.device) / 2**30,
        )

    def _release_fsdp2_preparation_cache(self) -> None:
        """Return blocks freed by FSDP2 sharding to the CUDA driver.

        Accelerate keeps the unsharded state alive while ``fully_shard`` allocates new shards, so
        the allocator retains the whole preparation peak as cached blocks. cuBLAS and NCCL
        allocate outside that pool and can fail on the first step unless the cache is released.
        """
        if self.accelerator.device.type != "cuda":
            return
        reserved = torch.cuda.memory_reserved(self.accelerator.device)
        torch.cuda.empty_cache()
        logger.info(
            "Released FSDP2 preparation cache: reserved %.2f GiB -> %.2f GiB",
            reserved / 2**30,
            torch.cuda.memory_reserved(self.accelerator.device) / 2**30,
        )

    def _extend_fsdp2_wrap_policy(self, objects: Sequence[Any]) -> None:
        plugin = self.accelerator.state.fsdp_plugin
        additional = tuple(getattr(self.adapter, "fsdp2_additional_wrap_module_names", ()))
        if (getattr(plugin, "fsdp_version", 1) or 1) < 2 or not additional:
            return
        configured = getattr(plugin, "transformer_cls_names_to_wrap", None)
        if configured is None:
            configured = [
                name
                for candidate in objects
                if isinstance(candidate, nn.Module)
                for name in (getattr(candidate, "_no_split_modules", None) or ())
            ]
        plugin.transformer_cls_names_to_wrap = list(dict.fromkeys([*configured, *additional]))
        logger.info(
            "Extended FSDP2 transformer wrap modules: %s",
            plugin.transformer_cls_names_to_wrap,
        )

    def bootstrap_targets(self) -> None:
        if not self.cpu_ram_efficient_loading or self.accelerator.num_processes <= 1:
            return
        synchronized = []
        for request in self.plan:
            if request.role is not ComponentRole.TARGET:
                continue
            component = self.adapter.get_component(request.root)
            self._raise_if_any_rank(
                not isinstance(component, nn.Module),
                f"target root={request.root!r} is not a module before FSDP prepare",
            )
            buffers = tuple(component.named_buffers())
            self._raise_if_any_rank(
                any(buffer.is_meta for _, buffer in buffers),
                f"target root={request.root!r} has a meta buffer before FSDP prepare",
            )
            for _, buffer in buffers:
                original_device = buffer.device
                value = buffer.detach().to(self.accelerator.device)
                dist.broadcast(value, src=0)
                buffer.data = value.to(original_device)
            synchronized.append(request.root)
        self.accelerator.wait_for_everyone()
        logger.info("FSDP target buffers synchronized: %s", synchronized)

    def components_loaded(
        self,
        components: Optional[Union[str, List[str]]],
    ) -> None:
        if self.accelerator.num_processes <= 1:
            return
        roots = [
            root
            for root in self._logical_names(components, role=ComponentRole.AUXILIARY)
            if root not in self._verified_roots
        ]
        if not roots:
            return
        self._check_replica_fingerprints(roots)
        self._verified_roots.update(roots)

    def _check_replica_fingerprints(self, roots: Sequence[str]) -> None:
        checked = []
        for root in roots:
            managed = self.adapter._should_manage_device(root)
            component = self.adapter.get_component(root)
            self._raise_if_any_rank(
                not managed or not isinstance(component, nn.Module),
                f"auxiliary root={root!r} is not a runtime-managed module",
            )
            self._check_fingerprint(root, component)
            checked.append(root)
        self.accelerator.wait_for_everyone()
        logger.info("FSDP replica sampled fingerprints matched: %s", checked)

    def _check_fingerprint(self, root: str, component: nn.Module) -> None:
        tensors = (*component.parameters(), *component.buffers())
        self._raise_if_any_rank(
            any(tensor.is_meta for tensor in tensors),
            f"replicated root={root!r} has a meta tensor after loading",
        )
        terms = []
        for tensor_index, tensor in enumerate(tensors):
            if not tensor.is_floating_point() or tensor.numel() == 0:
                continue
            flat = tensor.detach().reshape(-1)
            indices = torch.tensor(
                [0, flat.numel() // 2, flat.numel() - 1],
                device=flat.device,
                dtype=torch.long,
            )
            sampled = flat.index_select(0, indices).to(torch.float64)
            weight = float(tensor_index + 1)
            terms.append(
                torch.stack(
                    (
                        sampled.sum() * weight,
                        sampled.abs().sum() * weight,
                        sampled.square().sum() * weight,
                    )
                )
            )
        fingerprint = (
            torch.stack(terms).sum(dim=0)
            if terms
            else torch.zeros(3, device=self.accelerator.device, dtype=torch.float64)
        )
        gathered = [torch.empty_like(fingerprint) for _ in range(self.accelerator.num_processes)]
        dist.all_gather(gathered, fingerprint)
        mismatched = [
            rank
            for rank, candidate in enumerate(gathered)
            if not torch.equal(candidate, gathered[0])
        ]
        if mismatched:
            raise RuntimeError(
                f"replicated root={root!r} sampled fingerprint differs from rank 0; "
                f"mismatched_ranks={mismatched}"
            )

    def _raise_if_any_rank(self, local_failure: bool, message: str) -> None:
        failed = torch.tensor(
            int(local_failure),
            device=self.accelerator.device,
            dtype=torch.int32,
        )
        dist.all_reduce(failed, op=dist.ReduceOp.MAX)
        if failed.item():
            raise RuntimeError(message)


def _reserved_bytes(device: torch.device) -> int:
    return torch.cuda.memory_reserved(device) if device.type == "cuda" else 0


def build_backend_load_runtime(
    accelerator: Accelerator,
    plan: LoadPlan,
    adapter: Any,
) -> BackendLoadRuntime:
    """Build the backend strategy for one adapter load plan."""
    if accelerator.distributed_type == DistributedType.FSDP:
        return FSDPBackendLoadRuntime(accelerator, plan, adapter)
    return BackendLoadRuntime(accelerator, plan, adapter)
