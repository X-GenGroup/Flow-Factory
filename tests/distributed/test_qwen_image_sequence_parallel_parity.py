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

"""Qwen-Image under tensor and sequence parallelism matches the unsharded model.

Including ``zero_cond_t`` (Qwen-Image-Edit-2511), whose blocks take a per-token
``modulate_index`` the overlay has to narrow with the image sequence, and with
the blocks called both by keyword and positionally (gradient checkpointing).
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


def _build(zero_cond_t: bool) -> torch.nn.Module:
    from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel

    torch.manual_seed(0)
    model = QwenImageTransformer2DModel(
        patch_size=2,
        in_channels=16,
        out_channels=4,
        num_layers=2,
        attention_head_dim=16,
        num_attention_heads=4,
        joint_attention_dim=32,
        axes_dims_rope=(4, 6, 6),
        zero_cond_t=zero_cond_t,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0, 0.2)
    return model.double()


def _forward_backward(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    from torch.distributed.tensor import DTensor

    generator = torch.Generator().manual_seed(1)
    target, reference = (1, 4, 4), (1, 4, 4)
    hidden = torch.randn(1, 32, 16, generator=generator).double()
    output = model(
        hidden_states=hidden,
        encoder_hidden_states=torch.randn(1, 7, 32, generator=generator).double(),
        encoder_hidden_states_mask=torch.ones(1, 7, dtype=torch.long),
        timestep=torch.tensor([0.5], dtype=torch.float64),
        img_shapes=[[target, reference]],
        return_dict=False,
    )[0]
    output.pow(2).sum().backward()
    values = {"output": output.detach()}
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

        from flow_factory.models.qwen_image.qwen_image_edit_plus import QwenImageEditPlusAdapter
        from flow_factory.models.tensor_parallel import apply_tensor_parallel, resolve_parallel_plan

        mesh = init_device_mesh("cpu", (WORLD_SIZE,), mesh_dim_names=("tp",))
        for zero_cond_t in (False, True):
            for checkpointing in (False, True):
                reference = _build(zero_cond_t)
                sharded = copy.deepcopy(reference)
                plan = resolve_parallel_plan(QwenImageEditPlusAdapter, sequence_parallel=True)
                apply_tensor_parallel(sharded, mesh, plan)
                if checkpointing:
                    sharded.enable_gradient_checkpointing()
                expected = _forward_backward(reference)
                received = _forward_backward(sharded)
                assert expected.keys() == received.keys(), sorted(expected.keys() ^ received.keys())
                result[f"zero_cond_t={zero_cond_t},checkpointing={checkpointing}"] = max(
                    ((received[name] - expected[name]).abs().max() / expected[name].abs().max()).item()
                    for name in expected
                    if expected[name].abs().max() > 0
                )
    except Exception as error:  # reported to the parent, which asserts on it
        result["error"] = f"{type(error).__name__}: {error}"
    Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result), encoding="utf-8")
    dist.destroy_process_group()


@requires_gloo
def test_qwen_image_with_and_without_zero_cond_t_matches_the_unsharded_model(
    tmp_path: Path,
) -> None:
    results = run_gloo_ranks(_parity_worker, tmp_path, WORLD_SIZE, {})

    for result in results:
        assert "error" not in result, result["error"]
        for case, relative in result.items():
            if case != "rank":
                assert relative < 1e-9, (case, result)
