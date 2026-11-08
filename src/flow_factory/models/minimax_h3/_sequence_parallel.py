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

import torch
from diffusers.models.transformers.transformer_minimax_h3 import (
    MiniMaxH3TokenRefiner,
    MiniMaxH3TokenRefinerBlock,
)
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.parallel import ParallelStyle


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
