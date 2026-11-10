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

"""H3 text-refiner token padding, checkpoint arguments and native packing boundary."""

from inspect import signature

import torch
from diffusers.models.transformers.transformer_minimax_h3 import (
    MiniMaxH3TokenRefiner,
    MiniMaxH3TokenRefinerBlock,
)
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.parallel import ParallelStyle

from ..parallelism.sequence import SequencePadding
from ._context_parallel import (
    _argument,
    _pad_rows,
    _replace_argument,
    validate_h3_context_parallel_inputs,
)


class H3SequenceParallelTokenRefinerBlock(MiniMaxH3TokenRefinerBlock):
    """Preserve native block operations while carrying a padding mask explicitly."""

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Apply native residual operations with a complete text-key validity mask."""
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), attention_mask=attention_mask
        )
        return hidden_states + self.ff(self.norm2(hidden_states))


class H3SequenceParallelTokenRefiner(MiniMaxH3TokenRefiner):
    """Refine a padded token shard and restore native text rows before packing."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Pad and shard text, refine with explicit masks, then gather native rows."""
        if hidden_states.ndim != 3 or hidden_states.shape[1] < 1:
            raise ValueError(
                "MiniMax H3 refiner SP requires nonempty [batch,text,hidden] input; "
                f"received {tuple(hidden_states.shape)}"
            )
        mesh = self._ff_sequence_parallel_mesh
        rows = hidden_states.shape[1]
        padding = -rows % mesh.size()
        hidden_states = torch.nn.functional.pad(hidden_states, (0, 0, 0, padding))
        mask = (torch.arange(rows + padding, device=hidden_states.device) < rows)[
            None, None, None, :
        ]
        hidden_states = (
            DTensor.from_local(hidden_states, mesh, [Replicate()], run_check=False)
            .redistribute(placements=[Shard(1)])
            .to_local()
        )
        for block in self.refiner_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(block, hidden_states, mask)
            else:
                hidden_states = block(hidden_states, mask)
        hidden_states = self.final_norm(hidden_states)
        return (
            DTensor.from_local(hidden_states, mesh, [Shard(1)], run_check=False)
            .redistribute(placements=[Replicate()])
            .to_local()[:, :rows]
        )


class H3RefinerSequenceParallel(ParallelStyle):
    """Install explicit model-owned refiner types without replacing parameters.

    The key mask is a block argument, so both model-level and in-forward
    checkpoint replay retain their own padding metadata across later forwards.
    Refiner class names replace their native counterparts in wrapping metadata.
    """

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        refiner = module.token_refiner
        if type(refiner) is not MiniMaxH3TokenRefiner or not refiner.refiner_blocks:
            raise ValueError("MiniMax H3 SP requires the native nonempty token refiner")
        for block in refiner.refiner_blocks:
            if type(block) is not MiniMaxH3TokenRefinerBlock or "forward" in block.__dict__:
                raise ValueError("MiniMax H3 refiner SP must precede checkpoint/forward wrappers")
        refiner.__class__ = H3SequenceParallelTokenRefiner
        refiner._ff_sequence_parallel_mesh = device_mesh
        for block in refiner.refiner_blocks:
            block.__class__ = H3SequenceParallelTokenRefinerBlock
        for field in ("_no_split_modules", "_repeated_blocks"):
            setattr(
                module,
                field,
                [
                    (
                        H3SequenceParallelTokenRefinerBlock.__name__
                        if name == MiniMaxH3TokenRefinerBlock.__name__
                        else name
                    )
                    for name in getattr(module, field)
                ],
            )
        return module


class H3PackedSequencePadding(SequencePadding):
    """Pad the packed text/video/audio rows to the TP degree for SP without CP.

    The packed length follows the prompt, so SP alone would require it to divide
    ``tp_size``. These hooks append rows at the end of the packed sequence: zero
    hidden rows, identity rotary rows and modulation index 0. Every attention masks
    the padded keys, and the root's own ``index_select`` drops the padded output
    rows, so valid rows and every gradient match the unpadded model. CP installs its
    own padding to ``cp_size * tp_size`` and removes these hooks first.
    """

    def install(self, module: nn.Module, device_mesh: DeviceMesh) -> None:
        if hasattr(module, "_ff_h3_sp_padding_handles"):
            raise ValueError("MiniMax H3 SP padding must be installed exactly once")
        divisor = device_mesh.size()
        call_signature = signature(module.forward)

        def validate(target, args, kwargs):
            validate_h3_context_parallel_inputs(
                call_signature.bind(*args, **kwargs).arguments, divisor
            )

        def pad_rotary(rope, inputs, output):
            cos, sin = output
            return _pad_rows(cos, divisor, 0, 1), _pad_rows(sin, divisor, 0)

        def pad_output_norm(norm, args, kwargs):
            original = _argument(args, kwargs, "timestep_indices", 2)
            return _replace_argument(
                args, kwargs, "timestep_indices", 2, _pad_rows(original, divisor, 0)
            )

        handles = [
            module.register_forward_pre_hook(validate, with_kwargs=True, prepend=True),
            module.rope.register_forward_hook(pad_rotary),
            module.norm_out.register_forward_pre_hook(
                pad_output_norm, with_kwargs=True, prepend=True
            ),
        ]
        masks = []
        for index, block in enumerate(module.transformer_blocks):

            def pad_rows(target, args, kwargs, first=index == 0):
                # Runs inside checkpoint replay as well, so the recorded row count
                # always belongs to the forward being (re)computed.
                original = _argument(args, kwargs, "adaln_indices", 2)
                target._ff_h3_valid_rows = original.shape[0]
                args, kwargs = _replace_argument(
                    args, kwargs, "adaln_indices", 2, _pad_rows(original, divisor, 0)
                )
                if first:
                    hidden = _argument(args, kwargs, "hidden_states", 0)
                    args, kwargs = _replace_argument(
                        args, kwargs, "hidden_states", 0, _pad_rows(hidden, divisor, 1)
                    )
                return args, kwargs

            def key_mask(attention, args, kwargs, block=block):
                # Registered after the SP input gather, so the rows are complete.
                hidden = _argument(args, kwargs, "hidden_states", 0)
                rows, valid = hidden.shape[1], block._ff_h3_valid_rows
                if rows == valid:
                    return None
                native_mask = args[2] if len(args) > 2 else kwargs.get("attention_mask")
                if native_mask is not None:
                    raise ValueError("MiniMax H3 SP padding owns the native padding key mask")
                mask = (torch.arange(rows, device=hidden.device) < valid)[None, None, None, :]
                if len(args) > 2:
                    return (*args[:2], mask, *args[3:]), kwargs
                return args, {**kwargs, "attention_mask": mask}

            handles.append(
                block.register_forward_pre_hook(pad_rows, with_kwargs=True, prepend=True)
            )
            masks.append(block.attn.register_forward_pre_hook(key_mask, with_kwargs=True))
        module._ff_h3_sp_padding_mask_handles = tuple(masks)
        module._ff_h3_sp_padding_handles = (*handles, *masks)


def remove_h3_sequence_padding(module: nn.Module) -> None:
    """Remove SP-only packed padding so CP can install its own, larger divisor."""
    for handle in getattr(module, "_ff_h3_sp_padding_handles", ()):
        handle.remove()
    for name in ("_ff_h3_sp_padding_handles", "_ff_h3_sp_padding_mask_handles"):
        if hasattr(module, name):
            delattr(module, name)
