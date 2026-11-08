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


"""Derive complete PEFT branch ownership from canonical projection styles."""

from __future__ import annotations

from typing import Dict, List, Mapping

from torch import nn
from torch.distributed.tensor import Shard
from torch.distributed.tensor.parallel import ColwiseParallel, ParallelStyle, RowwiseParallel

from .packed import PackedColwiseParallel, PackedRowwiseParallel
from .sequence import SequenceBroadcastParameter
from .tensor import ReplicateParallel, SharedInputColwiseParallel


def _lora_adapter_names(module: nn.Module) -> List[str]:
    """Return the adapter names PEFT injected into this module, if any."""
    lora_a = getattr(module, "lora_A", None)
    if lora_a is None or not hasattr(lora_a, "keys"):
        return []
    return [name for name in lora_a.keys() if any(True for _ in lora_a[name].parameters())]


def derive_lora_plan(
    root: nn.Module, resolved_plan: Mapping[str, ParallelStyle]
) -> Dict[str, ParallelStyle]:
    """Derive PEFT branch placements from each resolved base projection style.

    The original entry moves to ``base_layer``. A column-parallel base keeps
    ``lora_A`` replicated and shards ``lora_B`` over output features. A
    row-parallel base shards ``lora_A`` over input features, preserving the base
    input and output layouts, and keeps ``lora_B`` replicated. If that output is
    token-sharded, ``lora_B`` instead sums its token-partial weight gradient.

    ``SharedInputColwiseParallel`` leaves input reduction to its region: its
    LoRA B does the same, while LoRA A sums its partial weight gradient.
    Packed styles preserve the base block widths on column B or row A, so both
    branches exchange activations consistently without permuting saved weights.
    A plain ``SequenceBroadcastParameter`` without exclusions gives both LoRA
    factors the same token-partial gradient ownership as the base projection.

    Activation-only and other unsupported styles on a PEFT wrapper are rejected.
    Moving such a boundary onto ``base_layer`` alone would transform one arm of
    ``base_layer(x) + lora_B(lora_A(x))`` and break its shared output contract.
    """
    derived: Dict[str, ParallelStyle] = {}
    for path, style in resolved_plan.items():
        module = root.get_submodule(path)
        adapters = _lora_adapter_names(module)
        if not adapters:
            derived[path] = style
            continue
        derived[f"{path}.base_layer"] = style
        for adapter in adapters:
            if isinstance(style, RowwiseParallel):
                # `input_layouts` and `output_layouts` are stored as 1-tuples by
                # `__init__`, which would be wrapped again if handed back whole.
                inputs = getattr(style, "input_layouts", None)
                outputs = getattr(style, "output_layouts", None)
                derived[f"{path}.lora_A.{adapter}"] = (
                    PackedRowwiseParallel(
                        style.resolve_block_sizes(module.base_layer), output_layouts=outputs[0]
                    )
                    if isinstance(style, PackedRowwiseParallel)
                    else RowwiseParallel(
                        input_layouts=inputs[0] if inputs else None,
                        output_layouts=outputs[0] if outputs else None,
                    )
                )
                # Under sequence parallelism the base branch reduce-scatters its
                # output onto the token axis, so the adapter branch has to land
                # there too -- otherwise the two are added at different lengths
                # ("size of tensor a (160) must match tensor b (320)"). Carrying
                # the layout through `lora_A` does that, and then `lora_B` is
                # applied per token on a slice, so its gradient is a partial sum.
                token_sharded = bool(outputs) and any(isinstance(p, Shard) for p in outputs)
                derived[f"{path}.lora_B.{adapter}"] = (
                    SequenceBroadcastParameter() if token_sharded else ReplicateParallel()
                )
            elif isinstance(style, SharedInputColwiseParallel):
                # The base arm leaves its input gradient a partial sum for the
                # region entry to reduce, so the adapter arm has to as well:
                # `lora_B` does not reduce the gradient of its input, which
                # makes `lora_A`'s weight gradient a partial sum too -- the case
                # `SequenceBroadcastParameter`'s view declares.
                derived[f"{path}.lora_A.{adapter}"] = SequenceBroadcastParameter()
                derived[f"{path}.lora_B.{adapter}"] = SharedInputColwiseParallel(verify_input=False)
            elif type(style) is SequenceBroadcastParameter and not style.exclude:
                # Both LoRA factors consume token shards, just like the base
                # projection. This style changes parameter views only; layout
                # transforms must remain on the complete wrapper's boundary.
                derived[f"{path}.lora_A.{adapter}"] = SequenceBroadcastParameter()
                derived[f"{path}.lora_B.{adapter}"] = SequenceBroadcastParameter()
            else:
                if not isinstance(style, ColwiseParallel):
                    # Other styles have no declared adapter pairing. They may be an
                    # activation transform (`PrepareModuleInput/Output`, or a
                    # `ComposedParallelStyle` carrying one) or a sequence-parallel
                    # conditional parameter style, and both go wrong here in ways that are
                    # hard to read at the point of failure:
                    #
                    # * an activation transform relocated to `base_layer` hooks
                    #   the base branch only, so the gather runs on one arm of
                    #   `base_layer(x) + lora_B(lora_A(x))` and the add sees
                    #   `[B, S, out]` against `[B, S/tp, out]`;
                    # * `lora_B -> ColwiseParallel` halves the adapter arm's width
                    #   against a full-width base arm ("the size of tensor a (128)
                    #   must match the size of tensor b (64)");
                    # * and `lora_A -> ReplicateParallel` inside the
                    #   sequence-parallel region accumulates only this rank's
                    #   token share into a replicated weight, which is the silent
                    #   1/tp_size gradient `SequenceBroadcastParameter` exists to
                    #   prevent.
                    #
                    # The adapters document this as a limit; refuse it here so it
                    # names the module instead of surfacing mid-forward.
                    raise ValueError(
                        f"LoRA targets {path!r}, which the tensor-parallel plan covers with "
                        f"{type(style).__name__} -- a style whose adapter pairing is not "
                        f"derivable. Supported styles are ColwiseParallel, "
                        f"SharedInputColwiseParallel, RowwiseParallel and the plain "
                        f"SequenceBroadcastParameter; other styles can include "
                        f"activation transforms or conditional parameter views, "
                        f"which would hook the base branch alone or train the adapter on this "
                        f"rank's token share. Remove {path.rsplit('.', 1)[-1]!r} from "
                        f"model.target_modules; or set parallelism.sequence_parallel: false, "
                        f"which keeps tensor parallelism and drops the sequence-parallel "
                        f"overlay that puts such styles on a projection; or set "
                        f"parallelism.tp_size=1."
                    )
                derived[f"{path}.lora_A.{adapter}"] = ReplicateParallel()
                derived[f"{path}.lora_B.{adapter}"] = (
                    PackedColwiseParallel(style.resolve_block_sizes(module.base_layer))
                    if isinstance(style, PackedColwiseParallel)
                    else ColwiseParallel()
                )
    return derived


def peel_peft_wrapper(module: nn.Module) -> nn.Module:
    """Return the base transformer under a PEFT wrapper, else ``module``.

    A plan names paths in the architecture's own tree
    (``transformer_blocks.*.attn.to_q``). Once PEFT has injected LoRA the
    component is a ``PeftModel`` whose children sit under
    ``base_model.model.``, so every pattern would match nothing. The base model
    keeps its injected ``lora.Linear`` submodules -- they are what
    :func:`derive_lora_plan` then targets -- while being the tree the plan was
    written against.

    ``CompileAccelerator._peel_peft`` does the same thing for the same reason;
    the two are kept separate only because the acceleration layer runs after
    ``prepare`` and is a different review surface.
    """
    get_base_model = getattr(module, "get_base_model", None)
    if callable(get_base_model):
        return get_base_model()
    return module
