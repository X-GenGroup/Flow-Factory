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

"""Four-rank CUDA qualification of real Wan blocks through Accelerate FSDP2.

Random weights isolate distributed semantics; this is not pretrained-weight or
full-pipeline qualification. Both layouts process the same two logical samples.
Every gradient and actual AdamW update is compared, including frozen weights.
"""

from __future__ import annotations

import copy
import gc
import hashlib
import importlib.metadata
import json
import os
import traceback
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict

import pytest
import torch
import torch.distributed as dist
from conftest import run_gloo_ranks
from test_wan_sequence_parallel_parity import _build, _with_lora
from tp_qualification import full_tensor, parameters_by_name

WORLD_SIZE = 4


def _inputs(
    replica: int, step: int, device: torch.device, channels: int, text_dim: int
) -> Dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(100 + 2 * step + replica)
    return {
        "hidden_states": torch.randn(1, channels, 2, 4, 4, generator=generator).to(device),
        "encoder_hidden_states": torch.randn(1, 5, text_dim, generator=generator).to(device),
        "timestep": torch.tensor([500.0 + 10 * replica], device=device),
    }


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, name: str) -> None:
    torch.testing.assert_close(
        actual, expected, rtol=2e-4, atol=2e-6, msg=lambda detail: f"{name}: {detail}"
    )


def _worker(rank: int, store_path: str, result_dir: str, options: Dict[str, Any]) -> None:
    os.environ.update(
        RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(WORLD_SIZE), ACCELERATE_USE_FSDP="true"
    )
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group(
        "nccl",
        store=dist.FileStore(store_path, WORLD_SIZE),
        rank=rank,
        world_size=WORLD_SIZE,
        timeout=timedelta(seconds=120),
    )
    result: Dict[str, Any] = {
        "rank": rank,
        "world_size": WORLD_SIZE,
        "dp": 2,
        "tp": 2,
        **options,
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("torch", "diffusers", "accelerate", "peft")
        },
    }
    try:
        from accelerate import FullyShardedDataParallelPlugin
        from accelerate.parallelism_config import ParallelismConfig
        from torch.distributed.tensor import DTensor

        from flow_factory.models.tensor_parallel import (
            apply_tensor_parallel,
            peel_peft_wrapper,
            resolve_parallel_plan,
        )
        from flow_factory.models.wan.wan2_t2v import Wan2_T2V_Adapter
        from flow_factory.trainers.loader import Accelerator
        from flow_factory.utils.dist import data_parallel_geometry

        plugin = FullyShardedDataParallelPlugin(
            fsdp_version=2,
            auto_wrap_policy="TRANSFORMER_BASED_WRAP",
            transformer_cls_names_to_wrap=["WanTransformerBlock"],
            activation_checkpointing=options["checkpointing"],
            cpu_ram_efficient_loading=False,
            reshard_after_forward=True,
        )
        accelerator = Accelerator(
            mixed_precision="no",
            fsdp_plugin=plugin,
            parallelism_config=ParallelismConfig(dp_replicate_size=1, dp_shard_size=2, tp_size=2),
        )
        assert accelerator.state.distributed_type.name == "FSDP"
        assert accelerator.state.fsdp_plugin.fsdp_version == 2
        assert data_parallel_geometry(accelerator) == (2, rank // 2)
        result["device"] = str(accelerator.device)
        result["backend"] = "fsdp2"
        result["mesh_dim_names"] = list(accelerator.torch_device_mesh.mesh_dim_names)

        def make_model():
            weights = options.get("weights")
            if not weights:
                return (_with_lora(_build()) if options["lora"] else _build()).float()
            from peft import LoraConfig, get_peft_model

            from diffusers import WanTransformer3DModel

            base = WanTransformer3DModel.from_pretrained(
                weights, local_files_only=True, torch_dtype=torch.float32
            )
            torch.manual_seed(1)
            targets = Wan2_T2V_Adapter.default_target_modules.fget(object.__new__(Wan2_T2V_Adapter))
            return get_peft_model(
                base, LoraConfig(r=4, lora_alpha=4, target_modules=targets, init_lora_weights=False)
            )

        reference = make_model().to(accelerator.device)
        channels, text_dim = reference.config.in_channels, reference.config.text_dim
        model = make_model()
        apply_tensor_parallel(
            peel_peft_wrapper(model),
            accelerator.torch_device_mesh["tp"],
            resolve_parallel_plan(Wan2_T2V_Adapter, sequence_parallel=options["sequence_parallel"]),
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, eps=1e-4, weight_decay=0.01)
        model, optimizer = accelerator.prepare(model, optimizer)
        reference_optimizer = torch.optim.AdamW(
            reference.parameters(), lr=1e-3, eps=1e-4, weight_decay=0.01
        )
        expected_params = parameters_by_name(reference)
        actual_params = parameters_by_name(model)
        assert expected_params.keys() == actual_params.keys()
        assert all(
            p.requires_grad == actual_params[name].requires_grad
            for name, p in expected_params.items()
        )
        assert all(isinstance(p, DTensor) for p in actual_params.values())
        assert {id(p) for group in optimizer.param_groups for p in group["params"]} == {
            id(p) for p in model.parameters()
        }, "optimizer retained pre-prepare parameter identities"
        frozen = {name: full_tensor(p) for name, p in actual_params.items() if not p.requires_grad}
        if options["lora"]:
            assert frozen, "LoRA qualification did not contain frozen base weights"
        result["steps"] = []
        retained_bytes = []
        # Measured with the same FP32 weights: sequential versus batched
        # unsharded backward has a 4.9e-8 absolute noise floor in a small
        # gradient. AdamW trajectory drift reaches 1.4e-7 after a prior update.
        # Retain that control below; Q2 keeps the stricter no-extra-floor bound.
        gradient_floor = 2e-7 if options.get("weights") else 0.0
        result["gradient_absolute_floor"] = gradient_floor
        for step in range(4):
            reference_optimizer.zero_grad(set_to_none=True)
            optimizer.zero_grad(set_to_none=True)
            reference_outputs = []
            for replica in range(2):
                output = reference(
                    **_inputs(replica, step, accelerator.device, channels, text_dim),
                    return_dict=False,
                )[0]
                reference_outputs.append(output.detach().clone())
                (output.square().mean() / 2).backward()
            output = model(
                **_inputs(rank // 2, step, accelerator.device, channels, text_dim),
                return_dict=False,
            )[0]
            _assert_close(output, reference_outputs[rank // 2], "forward")
            accelerator.backward(output.square().mean())
            expected_names = {name for name, p in expected_params.items() if p.grad is not None}
            actual_names = {name for name, p in actual_params.items() if p.grad is not None}
            assert expected_names == actual_names, sorted(expected_names ^ actual_names)
            assert expected_names, "no gradients were produced"
            control_params = {}
            if options.get("weights"):
                control = copy.deepcopy(reference)
                control.zero_grad(set_to_none=True)
                rows = [
                    _inputs(replica, step, accelerator.device, channels, text_dim)
                    for replica in range(2)
                ]
                batched = {key: torch.cat([row[key] for row in rows]) for key in rows[0]}
                control_output = control(**batched, return_dict=False)[0]
                control_output.square().mean().backward()
                control_params = parameters_by_name(control)
            max_gradient_relative = 0.0
            noise_control = {"relative": 0.0}
            worst_gradient = {}
            for name in sorted(expected_names):
                expected_gradient = expected_params[name].grad
                actual_gradient = full_tensor(actual_params[name].grad)
                _assert_close(actual_gradient, expected_gradient, f"gradient: {name}")
                scale = expected_gradient.abs().max()
                error = (actual_gradient - expected_gradient).abs().max()
                assert error <= 5e-4 * scale + gradient_floor, f"gradient scale: {name}"
                if scale > 0:
                    relative = (error / scale).item()
                    if relative > max_gradient_relative:
                        worst_gradient = {
                            "name": name,
                            "absolute_error": error.item(),
                            "reference_max": scale.item(),
                        }
                        max_gradient_relative = relative
                    if control_params:
                        control_error = (control_params[name].grad - expected_gradient).abs().max()
                        control_relative = (control_error / scale).item()
                        if control_relative > noise_control["relative"]:
                            noise_control = {
                                "name": name,
                                "relative": control_relative,
                                "absolute_error": control_error.item(),
                                "reference_max": scale.item(),
                            }
            if control_params:
                del control, control_output, control_params, rows, batched
            for name in frozen:
                assert actual_params[name].grad is None, f"frozen gradient: {name}"
            torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.5)
            accelerator.clip_grad_norm_(model.parameters(), 0.5)
            before = {
                name: (p.detach().clone(), full_tensor(actual_params[name]))
                for name, p in expected_params.items()
            }
            reference_optimizer.step()
            optimizer.step()
            changed = 0
            for name, expected in expected_params.items():
                actual = full_tensor(actual_params[name])
                expected_before, actual_before = before[name]
                _assert_close(actual, expected, f"parameter: {name}")
                _assert_close(actual - actual_before, expected - expected_before, f"update: {name}")
                expected_delta = expected - expected_before
                actual_delta = actual - actual_before
                roundoff = 4 * torch.finfo(expected.dtype).eps * expected_before.abs().max()
                assert (actual_delta - expected_delta).abs().max() <= (
                    5e-4 * expected_delta.abs().max() + roundoff + gradient_floor
                ), f"update scale: {name}"
                if name == worst_gradient.get("name"):
                    worst_gradient["update_absolute_error"] = (
                        (actual_delta - expected_delta).abs().max().item()
                    )
                    worst_gradient["reference_update_max"] = expected_delta.abs().max().item()
                if name in frozen:
                    torch.testing.assert_close(actual, frozen[name], rtol=0, atol=0, msg=name)
                changed += int(bool(torch.count_nonzero(expected - expected_before)))
            assert changed > 0
            assert optimizer.state, "AdamW state was not populated"
            for name in sorted(expected_names):
                expected_state = reference_optimizer.state[expected_params[name]]
                actual_state = optimizer.state[actual_params[name]]
                for key in ("exp_avg", "exp_avg_sq", "step"):
                    _assert_close(
                        full_tensor(actual_state[key]),
                        expected_state[key],
                        f"optimizer {key}: {name}",
                    )
            result["steps"].append(
                {
                    "gradient_parameters": len(expected_names),
                    "max_gradient_relative": max_gradient_relative,
                    "worst_gradient": worst_gradient,
                    "unsharded_batching_control": noise_control,
                    "updated_parameters": changed,
                    "frozen_parameters": len(frozen),
                }
            )
            del before, output, reference_outputs
            reference_optimizer.zero_grad(set_to_none=True)
            optimizer.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.synchronize()
            retained_bytes.append(torch.cuda.memory_allocated())
        # First step initializes optimizer buffers. A retained activation graph
        # would add live allocations each iteration instead of reaching a plateau.
        assert max(retained_bytes[1:]) - min(retained_bytes[1:]) <= 1024 * 1024, retained_bytes
        result["retained_bytes"] = retained_bytes
        result["status"] = "pass"
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        Path(result_dir, f"rank-{rank}.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        dist.destroy_process_group()


@pytest.mark.skipif(
    torch.cuda.device_count() < WORLD_SIZE, reason="FSDP2 dp2 x tp2 requires four CUDA devices"
)
@pytest.mark.parametrize(
    "sequence_parallel,lora,checkpointing",
    [(False, False, False), (True, False, True), (False, True, False), (True, True, True)],
    ids=["tp-full", "sp-full-checkpointed", "tp-lora", "sp-lora-checkpointed"],
)
def test_wan_fsdp2_tp_updates_match_fixed_replica_reference(
    tmp_path: Path, sequence_parallel: bool, lora: bool, checkpointing: bool
) -> None:
    results = run_gloo_ranks(
        _worker,
        tmp_path,
        WORLD_SIZE,
        {"sequence_parallel": sequence_parallel, "lora": lora, "checkpointing": checkpointing},
        timeout=180,
    )
    for result in results:
        assert "error" not in result, result.get("error")
        assert result["status"] == "pass", result
        assert len(result["steps"]) == 4, result


@pytest.mark.skipif(
    torch.cuda.device_count() < WORLD_SIZE, reason="FSDP2 dp2 x tp2 requires four CUDA devices"
)
@pytest.mark.skipif(
    not os.environ.get("FF_WAN_TP_MODEL_PATH"),
    reason="set FF_WAN_TP_MODEL_PATH to a locally staged Wan transformer",
)
def test_pretrained_wan_lora_updates_match_fixed_replica_reference(tmp_path: Path) -> None:
    """Qualify pretrained denoiser weights without a VAE, text encoder or reward service."""
    weights = Path(os.environ["FF_WAN_TP_MODEL_PATH"]).resolve(strict=True)
    identities = {}
    for path in sorted(
        [
            weights / "config.json",
            *weights.glob("*.safetensors"),
            *weights.glob("*.safetensors.index.json"),
        ]
    ):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        identities[path.name] = {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}
    assert any(name.endswith(".safetensors") for name in identities), "no pretrained weights found"
    results = run_gloo_ranks(
        _worker,
        tmp_path,
        WORLD_SIZE,
        {
            "sequence_parallel": True,
            "lora": True,
            "checkpointing": True,
            "weights": str(weights),
            "weight_identity": identities,
        },
        timeout=300,
    )
    for result in results:
        assert "error" not in result, result.get("error")
        assert result["status"] == "pass", result
        assert len(result["steps"]) == 4, result
