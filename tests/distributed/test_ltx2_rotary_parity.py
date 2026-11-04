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

"""LTX-2 under tensor parallelism matches the unsharded model for both rotary layouts.

The published checkpoints use ``rope_type="split"``, whose tables carry a head
axis, while the class default is ``interleaved``. A slice on the wrong axis
leaves every shape consistent and the rotation wrong, so only a parity check
against the unsharded model can tell; the control below keeps that check honest.
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


def _build(rope_type: str) -> torch.nn.Module:
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
        rope_type=rope_type,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0, 0.2)
    return model.double().eval()


def _forward(model: torch.nn.Module) -> torch.Tensor:
    generator = torch.Generator().manual_seed(123)
    frames, height, width, audio_frames, text = 2, 2, 2, 4, 6
    video, audio = model(
        hidden_states=torch.randn(1, frames * height * width, 8, generator=generator).double(),
        audio_hidden_states=torch.randn(1, audio_frames, 8, generator=generator).double(),
        encoder_hidden_states=torch.randn(1, text, 24, generator=generator).double(),
        audio_encoder_hidden_states=torch.randn(1, text, 24, generator=generator).double(),
        timestep=torch.tensor([500.0], dtype=torch.float64),
        num_frames=frames,
        height=height,
        width=width,
        audio_num_frames=audio_frames,
        fps=24.0,
        return_dict=False,
    )[:2]
    return torch.cat([video.flatten(), audio.flatten()])


def _parity_worker(rank: int, store_path: str, result_dir: str, options: Dict[str, Any]) -> None:
    store = dist.FileStore(store_path, WORLD_SIZE)
    dist.init_process_group("gloo", store=store, rank=rank, world_size=WORLD_SIZE)
    result: Dict[str, Any] = {"rank": rank}
    try:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import Replicate, Shard
        from torch.distributed.tensor.parallel import PrepareModuleOutput

        from flow_factory.models.ltx2.ltx2_t2av import LTX2_T2AV_Adapter
        from flow_factory.models.tensor_parallel import apply_tensor_parallel, resolve_parallel_plan

        mesh = init_device_mesh("cpu", (WORLD_SIZE,), mesh_dim_names=("tp",))
        last_axis_slice = PrepareModuleOutput(
            output_layouts=(Replicate(), Replicate()),
            desired_output_layouts=(Shard(-1), Shard(-1)),
            use_local_output=True,
        )
        cases = {
            "interleaved": ("interleaved", False, False),
            "split": ("split", False, False),
            "split+sp": ("split", True, False),
            "split, last-axis control": ("split", False, True),
        }
        for label, (rope_type, sequence_parallel, control) in cases.items():
            reference = _build(rope_type)
            sharded = copy.deepcopy(reference)
            with torch.no_grad():
                expected = _forward(reference)
            plan = resolve_parallel_plan(LTX2_T2AV_Adapter, sequence_parallel=sequence_parallel)
            if control:
                for key in ("rope", "audio_rope", "cross_attn_rope", "cross_attn_audio_rope"):
                    plan[key] = last_axis_slice
            apply_tensor_parallel(sharded, mesh, plan)
            with torch.no_grad():
                received = _forward(sharded)
            result[label] = (received - expected).abs().max().item()
    except Exception as error:  # reported to the parent, which asserts on it
        result["error"] = f"{type(error).__name__}: {error}"
    Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result), encoding="utf-8")
    dist.destroy_process_group()


@requires_gloo
def test_both_rotary_layouts_match_the_unsharded_model(tmp_path: Path) -> None:
    results = run_gloo_ranks(_parity_worker, tmp_path, WORLD_SIZE, {})

    for result in results:
        assert "error" not in result, result["error"]
        for label in ("interleaved", "split", "split+sp"):
            assert result[label] < 1e-9, (label, result)
        assert result["split, last-axis control"] > 1e-2, result
