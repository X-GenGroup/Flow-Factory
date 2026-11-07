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

"""Flux-owned boundary between token-sharded dual and replicated single streams."""

from torch import Tensor, nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.parallel import ParallelStyle

from ..context_parallel import ContextParallelBoundary


class FluxSingleStreamInput(ParallelStyle):
    """Gather only the image stream before the first native single block concatenates it.

    Native Flux calls blocks by keyword eagerly and positionally during checkpoint
    recomputation. The same boundary handles both, using PyTorch redistribution
    for the forward gather and backward local slice. A sequence-region entry is
    inappropriate here: the downstream single stack is replicated, so its input
    gradient is already complete and must not be reduced across TP again.
    """

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        def gather(value: Tensor) -> Tensor:
            return (
                DTensor.from_local(value, device_mesh, [Shard(1)], run_check=False)
                .redistribute(placements=[Replicate()])
                .to_local()
                .clone()
            )

        def enter(_module: nn.Module, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
            if "hidden_states" in kwargs:
                return args, {**kwargs, "hidden_states": gather(kwargs["hidden_states"])}
            return (gather(args[0]), *args[1:]), kwargs

        module.register_forward_pre_hook(enter, with_kwargs=True)
        return module


def validate_flux_sequence_boundaries(
    module: nn.Module,
) -> tuple[ContextParallelBoundary, ...]:
    """Require native root CP splitting outside the bounded dual-stream SP region.

    CP splits before x_embedder, whose output starts SP. The first single block
    restores that CP-local image stream before final proj_out performs CP gather.
    Native callbacks therefore keep their original ownership and order.
    """
    from diffusers.hooks.context_parallel import ContextParallelGatherHook, ContextParallelSplitHook
    from diffusers.models._modeling_parallel import ContextParallelInput, ContextParallelOutput

    root_registry = getattr(module, "_diffusers_hook", None)
    output_registry = getattr(module.proj_out, "_diffusers_hook", None)
    split = root_registry.get_hook("cp_input---") if root_registry else None
    gather = output_registry.get_hook("cp_output---proj_out") if output_registry else None
    expected = {
        "hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        "encoder_hidden_states": ContextParallelInput(
            split_dim=1, expected_dims=3, split_output=False
        ),
        "img_ids": ContextParallelInput(split_dim=0, expected_dims=2, split_output=False),
        "txt_ids": ContextParallelInput(split_dim=0, expected_dims=2, split_output=False),
    }
    if (
        not isinstance(split, ContextParallelSplitHook)
        or split.metadata != expected
        or not isinstance(gather, ContextParallelGatherHook)
        or gather.metadata != [ContextParallelOutput(gather_dim=1, expected_dims=3)]
    ):
        raise ValueError("Flux TP/SP+CP requires native root image/text split and proj_out gather")
    return ()
