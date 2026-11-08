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

"""Token boundaries for Z-Image's independently padded native refiners."""

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.parallel import ParallelStyle, PrepareModuleOutput


class ZImageRefinerSequenceParallel(ParallelStyle):
    """Scatter a refiner stack once and gather before native joint assembly.

    Native masks and rotary tables stay complete across TP peers: attention
    gathers the token axis before consuming them. Each stack has its own padded
    length, so its divisibility check belongs at this boundary.
    """

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        if not isinstance(module, torch.nn.ModuleList) or not module:
            raise ValueError("Z-Image SP requires a nonempty native refiner stack")

        def scatter(block, args, kwargs):
            value = args[0] if args else kwargs["x"]
            if value.shape[1] % device_mesh.size():
                raise ValueError(
                    f"Z-Image refiner tokens={value.shape[1]} must be divisible by "
                    f"tp_size={device_mesh.size()}"
                )
            value = (
                DTensor.from_local(value, device_mesh, [Replicate()], run_check=False)
                .redistribute(placements=[Shard(1)])
                .to_local()
            )
            if args:
                return (value, *args[1:]), kwargs
            return args, {**kwargs, "x": value}

        module[0].register_forward_pre_hook(scatter, with_kwargs=True)
        PrepareModuleOutput(
            output_layouts=Shard(1),
            desired_output_layouts=Replicate(),
            use_local_output=True,
        )._apply(module[-1], device_mesh)
        return module
