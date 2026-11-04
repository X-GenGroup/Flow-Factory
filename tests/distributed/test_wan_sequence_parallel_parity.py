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

"""Wan under tensor and sequence parallelism matches the unsharded model.

Both timestep layouts: one per sample (Wan2.1, the 14B Wan2.2 models) and one
per token (Wan2.2-TI2V-5B's ``expand_timesteps``), whose modulation has to be
narrowed with the sequence. Every parameter's gradient is compared, not the
loss: the overlay's failure modes leave the loss intact.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict

import torch
import torch.distributed as dist
from conftest import requires_gloo, run_gloo_ranks

WORLD_SIZE = 2


def _build() -> torch.nn.Module:
    from diffusers.models.transformers.transformer_wan import WanTransformer3DModel

    torch.manual_seed(0)
    model = WanTransformer3DModel(
        patch_size=(1, 2, 2),
        num_attention_heads=2,
        attention_head_dim=8,
        in_channels=4,
        out_channels=4,
        text_dim=16,
        freq_dim=16,
        ffn_dim=32,
        num_layers=2,
        rope_max_seq_len=32,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0, 0.2)
    return model.double()


def _forward_backward(model: torch.nn.Module, per_token: bool) -> Dict[str, torch.Tensor]:
    from torch.distributed.tensor import DTensor

    generator = torch.Generator().manual_seed(11)
    frames, height, width = 2, 4, 4
    tokens = frames * (height // 2) * (width // 2)
    latents = torch.randn(1, 4, frames, height, width, generator=generator).double()
    text = torch.randn(1, 5, 16, generator=generator).double()
    if per_token:
        timestep = torch.rand(1, tokens, generator=generator).double() * 1000
    else:
        timestep = torch.tensor([600.0], dtype=torch.float64)
    output = model(
        hidden_states=latents, timestep=timestep, encoder_hidden_states=text, return_dict=False
    )[0]
    output.pow(2).sum().backward()
    values = {"output": output.detach()}
    for name, parameter in model.named_parameters():
        grad = parameter.grad
        values[name] = (grad.full_tensor() if isinstance(grad, DTensor) else grad).detach()
    return values


def _parity_worker(rank: int, store_path: str, result_dir: str, options: Dict[str, Any]) -> None:
    store = dist.FileStore(store_path, WORLD_SIZE)
    dist.init_process_group("gloo", store=store, rank=rank, world_size=WORLD_SIZE)
    result: Dict[str, Any] = {"rank": rank}
    try:
        from torch.distributed.device_mesh import init_device_mesh

        from flow_factory.models.tensor_parallel import apply_tensor_parallel, resolve_parallel_plan
        from flow_factory.models.wan.wan2_t2v import Wan2_T2V_Adapter

        mesh = init_device_mesh("cpu", (WORLD_SIZE,), mesh_dim_names=("tp",))
        for per_token in (False, True):
            reference = _build()
            sharded = copy.deepcopy(reference)
            apply_tensor_parallel(
                sharded, mesh, resolve_parallel_plan(Wan2_T2V_Adapter, sequence_parallel=None)
            )
            expected = _forward_backward(reference, per_token)
            received = _forward_backward(sharded, per_token)
            relative = {
                name: ((received[name] - expected[name]).abs().max() / expected[name].abs().max())
                .clamp_max(1e3)
                .item()
                for name in expected
                if expected[name].abs().max() > 0
            }
            worst = max(relative, key=relative.get)
            result["per_token" if per_token else "per_sample"] = {
                "max_relative_diff": relative[worst],
                "worst": worst,
            }
    except Exception as error:  # reported to the parent, which asserts on it
        result["error"] = f"{type(error).__name__}: {error}"
    Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result), encoding="utf-8")
    dist.destroy_process_group()


@requires_gloo
def test_both_timestep_layouts_match_the_unsharded_model(tmp_path: Path) -> None:
    results = run_gloo_ranks(_parity_worker, tmp_path, WORLD_SIZE, {})

    for result in results:
        assert "error" not in result, result["error"]
        # Wan runs its norms on `hidden.float()`, so even a float64 model has a
        # float32 floor; a missing or doubled reduction is off by order one.
        for layout in ("per_sample", "per_token"):
            assert result[layout]["max_relative_diff"] < 1e-5, (layout, result)
