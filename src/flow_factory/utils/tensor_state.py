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

"""Fixed-topology tensor metadata shared by parameter-state owners and storage."""

import json
from typing import Any

import torch
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.placement_types import _StridedShard


def dtensor_layout(tensor: DTensor) -> dict[str, Any]:
    """Describe a local shard without communicating or materializing full weights."""
    placements = []
    for placement in tensor.placements:
        if type(placement) is Replicate:
            placements.append({"kind": "replicate"})
        elif type(placement) is Shard:
            placements.append({"kind": "shard", "dim": placement.dim})
        elif type(placement) is _StridedShard:
            placements.append(
                {
                    "kind": "strided_shard",
                    "dim": placement.dim,
                    "split_factor": placement.split_factor,
                }
            )
        else:
            raise TypeError(f"unsupported checkpoint DTensor placement: {placement!r}")
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "mesh_device_type": tensor.device_mesh.device_type,
        "mesh": tensor.device_mesh.mesh.tolist(),
        "mesh_dim_names": list(tensor.device_mesh.mesh_dim_names or ()),
        "placements": placements,
        "local_shape": list(tensor.to_local().shape),
        "storage_device_type": tensor.device.type,
    }


def validate_tensor_state(received: torch.Tensor, expected: torch.Tensor, *, context: str) -> None:
    """Reject layout changes before a parameter-state owner copies any values."""
    if type(received) not in (torch.Tensor, DTensor):
        raise TypeError(
            f"{context} must be a plain torch.Tensor or DTensor, received {type(received).__name__}"
        )
    if (
        received.layout is not torch.strided
        or received.is_complex()
        or received.device.type == "meta"
    ):
        raise TypeError(
            f"{context} requires a materialized dense real strided tensor, received "
            f"layout={received.layout}, dtype={received.dtype}, device={received.device}"
        )
    if received.shape != expected.shape or received.dtype != expected.dtype:
        raise ValueError(
            f"{context} tensor metadata mismatch: expected shape={tuple(expected.shape)}, "
            f"dtype={expected.dtype}, received shape={tuple(received.shape)}, dtype={received.dtype}"
        )
    if isinstance(received, DTensor) != isinstance(expected, DTensor):
        raise ValueError(f"{context} distributed layout mismatch: replicated versus DTensor")
    if isinstance(received, DTensor) and dtensor_layout(received) != dtensor_layout(expected):
        raise ValueError(f"{context} DTensor layout mismatch (fixed topology is required)")


def restore_local_shard(
    local: torch.Tensor, layout: Any, template: Any, *, context: str
) -> DTensor:
    """Fill a clone on the existing mesh, including CPU-offloaded CUDA-mesh state."""
    if type(template) is not DTensor or json.dumps(layout, sort_keys=True) != json.dumps(
        dtensor_layout(template), sort_keys=True
    ):
        raise ValueError(f"{context} DTensor layout mismatch (fixed topology is required)")
    expected_local = template.to_local()
    validate_tensor_state(local, expected_local, context=f"{context} local shard")
    # from_local would move CPU-offloaded state back to the mesh's CUDA device.
    # A clone retains the prepared layout and storage without constructing groups.
    restored = template.detach().clone()
    restored.to_local().copy_(local.to(expected_local.device))
    return restored
