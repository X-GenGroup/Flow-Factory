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

"""LTX-2's I2AV timesteps and LTX-2.3's gates and prompt modulation, sharded.

I2AV hands the transformer one video timestep per token; LTX-2.3 adds per-head
attention gates and prompt modulation tables on the replicated text stream.
Every parameter's gradient is compared against the unsharded model, since each
failure here leaves the loss intact.
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
LTX23 = dict(gated_attn=True, audio_gated_attn=True, cross_attn_mod=True, audio_cross_attn_mod=True)


def _build(**flags: Any) -> torch.nn.Module:
    from diffusers.models.transformers.transformer_ltx2 import LTX2VideoTransformer3DModel

    torch.manual_seed(0)
    model = LTX2VideoTransformer3DModel(
        in_channels=8,
        out_channels=8,
        patch_size=1,
        patch_size_t=1,
        num_attention_heads=4,
        attention_head_dim=16,
        cross_attention_dim=64,
        audio_in_channels=8,
        audio_out_channels=8,
        audio_num_attention_heads=4,
        audio_attention_head_dim=8,
        audio_cross_attention_dim=32,
        num_layers=2,
        caption_channels=24,
        rope_type="split",
        **flags,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0, 0.2)
    return model.double()


def _forward_backward(model: torch.nn.Module, per_token: bool) -> Dict[str, torch.Tensor]:
    from torch.distributed.tensor import DTensor

    generator = torch.Generator().manual_seed(123)
    frames, height, width, audio_frames, text = 2, 2, 2, 4, 6
    tokens = frames * height * width
    if per_token:
        timestep = torch.rand(1, tokens, generator=generator).double() * 1000
        timestep[:, : height * width] = 0.0
    else:
        timestep = torch.tensor([500.0], dtype=torch.float64)
    video, audio = model(
        hidden_states=torch.randn(1, tokens, 8, generator=generator).double(),
        audio_hidden_states=torch.randn(1, audio_frames, 8, generator=generator).double(),
        encoder_hidden_states=torch.randn(1, text, 24, generator=generator).double(),
        audio_encoder_hidden_states=torch.randn(1, text, 24, generator=generator).double(),
        timestep=timestep,
        audio_timestep=torch.tensor([500.0], dtype=torch.float64),
        sigma=torch.tensor([500.0], dtype=torch.float64),
        audio_sigma=torch.tensor([500.0], dtype=torch.float64),
        num_frames=frames,
        height=height,
        width=width,
        audio_num_frames=audio_frames,
        fps=24.0,
        return_dict=False,
    )[:2]
    (video.pow(2).sum() + audio.pow(2).sum()).backward()
    values = {"video": video.detach(), "audio": audio.detach()}
    for name, parameter in model.named_parameters():
        grad = parameter.grad
        if grad is not None:
            values[name] = (grad.full_tensor() if isinstance(grad, DTensor) else grad).detach()
    return values


def _parity_worker(rank: int, store_path: str, result_dir: str, options: Dict[str, Any]) -> None:
    store = dist.FileStore(store_path, WORLD_SIZE)
    dist.init_process_group("gloo", store=store, rank=rank, world_size=WORLD_SIZE)
    result: Dict[str, Any] = {"rank": rank}
    try:
        from torch.distributed.device_mesh import init_device_mesh

        from flow_factory.models.ltx2.ltx2_i2av import LTX2_I2AV_Adapter
        from flow_factory.models.tensor_parallel import apply_tensor_parallel, resolve_parallel_plan

        mesh = init_device_mesh("cpu", (WORLD_SIZE,), mesh_dim_names=("tp",))
        cases = {
            "i2av per-token timestep, sp": ({}, True, True),
            "ltx2.3, tp": (LTX23, False, False),
            "ltx2.3, sp": (LTX23, True, False),
            "ltx2.3 i2av, sp": (LTX23, True, True),
        }
        for label, (flags, sequence_parallel, per_token) in cases.items():
            reference = _build(**flags)
            sharded = copy.deepcopy(reference)
            apply_tensor_parallel(
                sharded,
                mesh,
                resolve_parallel_plan(LTX2_I2AV_Adapter, sequence_parallel=sequence_parallel),
            )
            expected = _forward_backward(reference, per_token)
            received = _forward_backward(sharded, per_token)
            assert expected.keys() == received.keys(), sorted(expected.keys() ^ received.keys())
            relative = {
                name: (
                    (received[name] - expected[name]).abs().max() / expected[name].abs().max()
                ).item()
                for name in expected
                if expected[name].abs().max() > 0
            }
            worst = max(relative, key=relative.get)
            result[label] = {"relative": relative[worst], "worst": worst}
    except Exception as error:  # reported to the parent, which asserts on it
        result["error"] = f"{type(error).__name__}: {error}"
    Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result), encoding="utf-8")
    dist.destroy_process_group()


@requires_gloo
def test_per_token_timesteps_gates_and_prompt_modulation_match_unsharded(tmp_path: Path) -> None:
    results = run_gloo_ranks(_parity_worker, tmp_path, WORLD_SIZE, {})

    for result in results:
        assert "error" not in result, result["error"]
        for case, outcome in result.items():
            if case != "rank":
                assert outcome["relative"] < 1e-9, (case, outcome)
