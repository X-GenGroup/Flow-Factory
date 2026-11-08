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

"""SD3 AdaLN placement and one supervised flow update, without trainer lifecycle claims."""

import copy
import json
import traceback
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from conftest import run_gloo_ranks
from test_sd35_context_parallel import build_transformer, install_lora, model_inputs
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

from flow_factory.models.stable_diffusion.sd3_5 import SD3_5Adapter
from flow_factory.models.tensor_parallel import (
    ComposedParallelStyle,
    SequenceBroadcastOutput,
    SharedInputColwiseParallel,
    apply_tensor_parallel,
    build_total_plan,
    peel_peft_wrapper,
    resolve_parallel_plan,
)

ADALN = (
    "transformer_blocks.*.norm1.linear",
    "transformer_blocks.*.norm1_context.linear",
    "norm_out.linear",
)


def published_config(family):
    """Parameter geometry from the published Medium/Large/Large-Turbo configs."""
    medium = family == "medium"
    return dict(
        sample_size=128,
        patch_size=2,
        in_channels=16,
        out_channels=16,
        num_layers=24 if medium else 38,
        attention_head_dim=64,
        num_attention_heads=24 if medium else 38,
        joint_attention_dim=4096,
        caption_projection_dim=1536 if medium else 2432,
        pooled_projection_dim=2048,
        pos_embed_max_size=384 if medium else 192,
        dual_attention_layers=tuple(range(13)) if medium else (),
        qk_norm="rms_norm",
    )


def coverage(model, plan):
    """Count sharded parameter elements; rowwise biases remain replicated."""
    total = sum(parameter.numel() for parameter in model.parameters())
    sharded = 0
    for path, style in build_total_plan(model, plan).items():
        if isinstance(style, ComposedParallelStyle):
            style = style.styles[0]
        module = model.get_submodule(path)
        if isinstance(style, (ColwiseParallel, SharedInputColwiseParallel)):
            sharded += sum(parameter.numel() for parameter in module.parameters(recurse=False))
        elif isinstance(style, RowwiseParallel):
            sharded += module.weight.numel()
    return {"total_parameters": total, "sharded_parameters": sharded}


@pytest.mark.parametrize("family", ["medium", "large", "large-turbo"])
def test_sd35_adaln_meta_layout_and_parameter_coverage(family):
    from diffusers import SD3Transformer2DModel

    with torch.device("meta"):
        model = SD3Transformer2DModel(**published_config(family))
    plan = SD3_5Adapter.tensor_parallel_plan()
    old = {path: style for path, style in plan.items() if path not in ADALN}
    before, after = coverage(model, old), coverage(model, plan)
    width = model.config.caption_projection_dim
    norms = {
        path: module
        for path, module in model.named_modules()
        if path.endswith(("norm1.linear", "norm1_context.linear", "norm_out.linear"))
    }
    assert len(norms) == model.config.num_layers * 2 + 1
    for path, linear in norms.items():
        chunks = (
            2
            if path == "norm_out.linear"
            or path.endswith(f"{model.config.num_layers - 1}.norm1_context.linear")
            else 6
        )
        if family == "medium" and ".norm1.linear" in path:
            chunks = 9 if int(path.split(".")[1]) < 13 else 6
        assert linear.weight.shape == (chunks * width, width)
        assert linear.bias.shape == (chunks * width,)
    assert after["total_parameters"] == before["total_parameters"]
    assert after["sharded_parameters"] - before["sharded_parameters"] == sum(
        module.weight.numel() for module in norms.values()
    )
    expected = (
        (2_243_171_520, 1_460_974_080, 2_227_745_280)
        if family == "medium"
        else (8_056_627_520, 5_342_189_568, 8_027_428_864)
    )
    assert (
        after["total_parameters"],
        before["sharded_parameters"],
        after["sharded_parameters"],
    ) == expected
    for enabled in (False, True):
        resolved = build_total_plan(
            model, resolve_parallel_plan(SD3_5Adapter, sequence_parallel=enabled)
        )
        for path in norms:
            style = resolved[path]
            image = ".norm1.linear" in path or path == "norm_out.linear"
            if enabled and image:
                assert isinstance(style, ComposedParallelStyle)
                assert isinstance(style.styles[1], SequenceBroadcastOutput)
                style = style.styles[0]
            assert isinstance(style, RowwiseParallel)
            assert style.input_layouts == style.output_layouts == (Replicate(),)


@pytest.mark.parametrize("family", ["medium", "large"])
def test_sd35_explicit_adaln_lora_keeps_rowwise_input_and_sp_rejection(family):
    from peft import LoraConfig, get_peft_model

    model = get_peft_model(
        build_transformer(family),
        LoraConfig(
            r=2,
            lora_alpha=2,
            target_modules=["norm1.linear", "norm1_context.linear", "norm_out.linear"],
        ),
    )
    native = peel_peft_wrapper(model)
    plan = build_total_plan(native, resolve_parallel_plan(SD3_5Adapter, sequence_parallel=False))
    adapters = {path: style for path, style in plan.items() if ".lora_A." in path}
    assert len(adapters) == 5
    for style in adapters.values():
        assert isinstance(style, RowwiseParallel)
        assert style.input_layouts == style.output_layouts == (Replicate(),)
    with pytest.raises(ValueError, match="adapter pairing is not derivable"):
        build_total_plan(native, resolve_parallel_plan(SD3_5Adapter, sequence_parallel=True))


def _full(value):
    return value.full_tensor() if isinstance(value, DTensor) else value


def _worker(rank, store_path, result_dir, options):
    device = options["device"]
    if device == "cuda":
        torch.cuda.set_device(rank)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)
    dist.init_process_group(
        "nccl" if device == "cuda" else "gloo",
        store=dist.FileStore(store_path, 2),
        rank=rank,
        world_size=2,
    )
    result = {
        "rank": rank,
        "world_size": 2,
        "dp_size": 1,
        "tp_size": 2,
        "cp_size": 1,
        "device": device,
        "scenarios": [],
    }
    try:
        mesh = init_device_mesh(device, (2,), mesh_dim_names=("tp",))
        for family, mode, sequence_parallel in options["scenarios"]:
            module = build_transformer(family, head_dim=64 if device == "cuda" else 8)
            if mode == "lora":
                module = install_lora(module)
            module = module.to(device)
            reference = copy.deepcopy(module).to(device)
            expected_trainable = {name for name, p in module.named_parameters() if p.requires_grad}
            apply_tensor_parallel(
                peel_peft_wrapper(module),
                mesh,
                resolve_parallel_plan(SD3_5Adapter, sequence_parallel=sequence_parallel),
            )
            assert {
                name for name, p in module.named_parameters() if p.requires_grad
            } == expected_trainable
            actual_parameters = dict(module.named_parameters())
            reference_parameters = dict(reference.named_parameters())
            assert actual_parameters.keys() == reference_parameters.keys()
            adaln_count = 0
            for name, p in actual_parameters.items():
                if any(
                    part in name
                    for part in ("norm1.linear.", "norm1_context.linear.", "norm_out.linear.")
                ):
                    assert isinstance(p, DTensor)
                    assert p.placements == (
                        (Shard(1),) if name.endswith("weight") else (Replicate(),)
                    )
                    assert p.to_local().numel() == p.numel() // (
                        2 if name.endswith("weight") else 1
                    )
                    adaln_count += 1
                torch.testing.assert_close(_full(p), reference_parameters[name], rtol=0, atol=0)
            assert adaln_count == 10
            optimizer = torch.optim.AdamW(
                [p for p in module.parameters() if p.requires_grad], lr=0.001, eps=1e-4
            )
            reference_optimizer = torch.optim.AdamW(
                [p for p in reference.parameters() if p.requires_grad], lr=0.001, eps=1e-4
            )
            inputs = model_inputs(0)
            clean = inputs["hidden_states"]
            noise = torch.randn(clean.shape, generator=torch.Generator().manual_seed(901))
            target = (noise - clean).to(device)
            inputs["hidden_states"] = clean * 0.4 + noise * 0.6
            inputs["timestep"] = torch.tensor([600.0])
            inputs = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in inputs.items()
            }
            actual, expected = module(**inputs)[0], reference(**inputs)[0]
            torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-6)
            loss = torch.nn.functional.mse_loss(actual, target)
            reference_loss = torch.nn.functional.mse_loss(expected, target)
            torch.testing.assert_close(loss, reference_loss, rtol=3e-4, atol=3e-6)
            loss.backward()
            reference_loss.backward()
            active = 0
            for name, p in actual_parameters.items():
                expected_parameter = reference_parameters[name]
                if expected_parameter.grad is None:
                    assert p.grad is None, name
                    continue
                active += 1
                torch.testing.assert_close(
                    _full(p.grad), expected_parameter.grad, rtol=3e-4, atol=3e-6, msg=name
                )
            assert active > 0
            optimizer.step()
            reference_optimizer.step()
            for name, p in actual_parameters.items():
                q = reference_parameters[name]
                torch.testing.assert_close(_full(p), q, rtol=3e-4, atol=3e-6, msg=name)
                if q.grad is None:
                    assert p not in optimizer.state
                    continue
                for field in ("step", "exp_avg", "exp_avg_sq"):
                    torch.testing.assert_close(
                        _full(optimizer.state[p][field]),
                        reference_optimizer.state[q][field],
                        rtol=3e-4,
                        atol=3e-6,
                        msg=name + field,
                    )
            result["scenarios"].append(
                {
                    "family": family,
                    "mode": mode,
                    "sequence_parallel": sequence_parallel,
                    "heads": 24 if family == "medium" else 38,
                    "head_dim": 64 if device == "cuda" else 8,
                    "active_gradients": active,
                    "adaln_parameters_checked": adaln_count,
                    "loss": loss.item(),
                    "output_max_abs": (actual - expected).abs().max().item(),
                    "adam_steps": 1,
                    "supervision": "flow_velocity_noise_minus_clean",
                }
            )
            del (
                optimizer,
                reference_optimizer,
                module,
                reference,
                actual_parameters,
                reference_parameters,
            )
        result["status"] = "pass"
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result))
        dist.destroy_process_group()


@pytest.mark.parametrize("family", ["medium", "large"])
@pytest.mark.parametrize("mode", ["full", "lora"])
@pytest.mark.parametrize("sequence_parallel", [False, True], ids=["tp", "tp_sp"])
def test_cpu_sd35_adaln_tp_supervised_update(tmp_path, family, mode, sequence_parallel):
    rows = run_gloo_ranks(
        _worker, tmp_path, 2, {"device": "cpu", "scenarios": [(family, mode, sequence_parallel)]}
    )
    assert all(row.get("status") == "pass" for row in rows), rows


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two local CUDA devices")
def test_cuda_sd35_adaln_tp2_supervised_update(tmp_path):
    """One TP2/DP1 job covers Medium's nine-way and Large's six-way AdaLN."""
    rows = run_gloo_ranks(
        _worker,
        tmp_path,
        2,
        {"device": "cuda", "scenarios": [("medium", "full", True), ("large", "full", True)]},
        timeout=300,
    )
    assert all(row.get("status") == "pass" for row in rows), rows
