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

"""Real Z-Image heads, learned padding and unified-token TP/SP/CP."""

import copy
import json
import traceback
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from conftest import run_gloo_ranks


def build_transformer(family=None, head_dim=8):
    from diffusers import ZImageTransformer2DModel

    torch.manual_seed(620)
    axes = [32, 48, 48] if head_dim == 128 else [2, 2, 4]
    return ZImageTransformer2DModel(
        all_patch_size=(2,),
        all_f_patch_size=(1,),
        in_channels=16,
        dim=30 * head_dim,
        n_layers=2,
        n_refiner_layers=1,
        n_heads=30,
        n_kv_heads=30,
        qk_norm=True,
        cap_feat_dim=32,
        axes_dims=axes,
        axes_lens=[256, 32, 32],
    )


def install_lora(module):
    from peft import LoraConfig, get_peft_model

    from flow_factory.models.z_image.z_image import ZImageAdapter

    module = get_peft_model(
        module,
        LoraConfig(
            r=2, lora_alpha=2, target_modules=ZImageAdapter.default_target_modules.fget(None)
        ),
    )
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            if "lora_B" in name:
                parameter.uniform_(-0.03, 0.03)
    return module


def model_inputs(step, dtype=torch.float64):
    generator = torch.Generator().manual_seed(917 + step)
    return dict(
        x=[torch.randn(16, 1, 8, 8, generator=generator, dtype=dtype) for _ in range(2)],
        cap_feats=[
            torch.randn(length, 32, generator=generator, dtype=dtype) for length in (19, 41)
        ],
        t=torch.tensor([0.8 - 0.103 * step] * 2, dtype=dtype),
        return_dict=False,
        patch_size=2,
        f_patch_size=1,
    )


def _cpu_worker(rank, store_path, result_dir, options):
    from diffusers.models._modeling_parallel import ContextParallelConfig
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.tensor import DTensor

    from flow_factory.models.tensor_parallel import (
        apply_tensor_parallel,
        peel_peft_wrapper,
        resolve_parallel_plan,
    )
    from flow_factory.models.z_image.z_image import ZImageAdapter

    dist.init_process_group("gloo", store=dist.FileStore(store_path, 4), rank=rank, world_size=4)
    result = {"rank": rank}
    try:
        mesh = DeviceMesh("cpu", torch.arange(4).reshape(2, 2), mesh_dim_names=("cp", "tp"))
        native_mesh = DeviceMesh(
            "cpu",
            mesh.mesh.T.reshape(1, 2, 1, 2),
            mesh_dim_names=("data", "tensor", "ring", "ulysses"),
        )
        module = build_transformer().double()
        if options["mode"] == "lora":
            module = install_lora(module)
        reference = copy.deepcopy(module)
        native = peel_peft_wrapper(module)
        ZImageAdapter.validate_context_parallel_module(native, 2, 2)
        apply_tensor_parallel(
            native, mesh["tp"], resolve_parallel_plan(ZImageAdapter, sequence_parallel=True)
        )
        ZImageAdapter.configure_context_parallel_module(native)
        native.set_attention_backend("native")
        ZImageAdapter.install_context_parallel_module(
            native, ContextParallelConfig(ulysses_degree=2, mesh=native_mesh)
        )
        ZImageAdapter.compose_context_parallel_sequence_boundaries(native)
        if options["checkpointing"]:
            native.enable_gradient_checkpointing()
        optimizer = torch.optim.AdamW(
            [p for p in module.parameters() if p.requires_grad], lr=0.001, eps=1e-4
        )
        reference_optimizer = torch.optim.AdamW(
            [p for p in reference.parameters() if p.requires_grad], lr=0.001, eps=1e-4
        )
        result["updates"] = []
        actual_parameters = dict(module.named_parameters())
        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
            expected = torch.stack(reference(**model_inputs(step))[0])
            output = torch.stack(module(**model_inputs(step))[0])
            torch.testing.assert_close(output, expected, rtol=3e-4, atol=3e-6)
            output.square().mean().backward()
            expected.square().mean().backward()
            nonzero = 0
            for name, parameter in reference.named_parameters():
                actual = actual_parameters[name]
                if parameter.grad is None:
                    assert actual.grad is None, name
                    continue
                # A native gather supplies each CP rank only its output slice.
                # Complete CP partial parameter gradients; FSDP owns this in GPU training.
                gradient = (
                    actual.grad.to_local() if isinstance(actual.grad, DTensor) else actual.grad
                )
                dist.all_reduce(gradient, group=mesh["cp"].get_group())
                full = (
                    actual.grad.full_tensor() if isinstance(actual.grad, DTensor) else actual.grad
                )
                if not torch.allclose(full, parameter.grad, rtol=3e-4, atol=3e-6):
                    torch.save(
                        {"actual": full, "expected": parameter.grad},
                        Path(result_dir, f"gradient-rank-{rank}.pt"),
                    )
                    result["gradient_diagnostic"] = {
                        "name": name,
                        "max_reference": parameter.grad.abs().max().item(),
                        "max_error": (full - parameter.grad).abs().max().item(),
                        "relative_l2": (
                            (full - parameter.grad).norm() / parameter.grad.norm()
                        ).item(),
                    }
                torch.testing.assert_close(
                    full,
                    parameter.grad,
                    rtol=3e-4,
                    atol=3e-6,
                    msg=lambda message: name + "\n" + message,
                )
                nonzero += int(torch.count_nonzero(parameter.grad) > 0)
            assert nonzero
            optimizer.step()
            reference_optimizer.step()
            for name, parameter in reference.named_parameters():
                actual = actual_parameters[name]
                full = actual.full_tensor() if isinstance(actual, DTensor) else actual
                torch.testing.assert_close(full, parameter, rtol=3e-4, atol=3e-6, msg=name)
                if not parameter.requires_grad:
                    continue
                for key in ("exp_avg", "exp_avg_sq"):
                    state = optimizer.state[actual][key]
                    value = state.full_tensor() if isinstance(state, DTensor) else state
                    torch.testing.assert_close(
                        value,
                        reference_optimizer.state[parameter][key],
                        rtol=3e-4,
                        atol=3e-6,
                        msg=name + key,
                    )
            result["updates"].append(
                {
                    "step": step,
                    "nonzero_gradients": nonzero,
                    "max_output_residual": (output - expected).abs().max().item(),
                }
            )
        result["status"] = "pass"
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result))
        dist.destroy_process_group()


@pytest.mark.parametrize("mode", ["full", "lora"])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_cpu_z_image_native_tp_sp_cp_matches_updates(tmp_path, mode, checkpointing):
    rows = run_gloo_ranks(_cpu_worker, tmp_path, 4, {"mode": mode, "checkpointing": checkpointing})
    assert all(row.get("status") == "pass" for row in rows), rows


def _gpu_worker(rank, store_path, result_dir, options):
    from accelerate import FullyShardedDataParallelPlugin
    from accelerate.parallelism_config import ParallelismConfig
    from test_model_preparation import (
        PreparationAdapter,
        assert_update_parity,
        full_value,
        initialize_cuda,
        optimizer_for,
        parameter_map,
        runtime_observation,
    )

    from flow_factory.acceleration.attention_backend import AttentionBackendAccelerator
    from flow_factory.hparams.parallelism_args import ParallelismArguments
    from flow_factory.loading.coordinator import ModelLoadCoordinator
    from flow_factory.models.model_bundle import ModelBundle
    from flow_factory.models.z_image.z_image import ZImageAdapter
    from flow_factory.trainers.common.model_preparation import ModelPreparation
    from flow_factory.trainers.loader import Accelerator

    initialize_cuda(rank, store_path, world_size=8)
    result = {
        "rank": rank,
        "execution_scope": "model_probe",
        "execution_capability": "z-image-tp-ulysses-fp32-probe-v1",
        **options,
    }
    try:

        class ZImagePreparationAdapter(PreparationAdapter):
            context_parallel_capability = ZImageAdapter.context_parallel_capability
            validate_context_parallel_module = ZImageAdapter.validate_context_parallel_module
            configure_context_parallel_module = ZImageAdapter.configure_context_parallel_module
            install_context_parallel_module = ZImageAdapter.install_context_parallel_module
            tensor_parallel_plan = ZImageAdapter.tensor_parallel_plan
            sequence_parallel_plan = ZImageAdapter.sequence_parallel_plan
            compose_context_parallel_sequence_boundaries = (
                ZImageAdapter.compose_context_parallel_sequence_boundaries
            )

        accelerator = Accelerator(
            mixed_precision="no",
            fsdp_plugin=FullyShardedDataParallelPlugin(
                fsdp_version=2,
                auto_wrap_policy="TRANSFORMER_BASED_WRAP",
                transformer_cls_names_to_wrap=["ZImageTransformerBlock"],
                cpu_ram_efficient_loading=False,
                reshard_after_forward=True,
                activation_checkpointing=True,
            ),
            parallelism_config=ParallelismConfig(
                dp_replicate_size=1, dp_shard_size=2, cp_size=2, tp_size=2, sp_size=1
            ),
        )
        result["runtime"] = runtime_observation(accelerator)
        module = build_transformer(head_dim=128).float()
        # Pretrained-like learned padding avoids the singular near-zero RMSNorm
        # conditioning. The separate FP64 CPU control retains native zero pads.
        with torch.no_grad():
            module.x_pad_token.normal_(0, 0.05)
            module.cap_pad_token.normal_(0, 0.05)
        if options["mode"] == "lora":
            module = install_lora(module).float()
        reference_module = copy.deepcopy(module).to(accelerator.device)
        adapter = ZImagePreparationAdapter(module, accelerator, options["mode"])
        adapter.declare_component_variants(("policy",))
        reference_adapter = PreparationAdapter(reference_module, accelerator, options["mode"])
        reference_adapter.declare_component_variants(("policy",))
        reference = ModelBundle(reference_adapter.component_variant_registry.bundle_members())

        # Keep the real-width random model on the same bounded trajectory as
        # the CPU oracle and native training lifecycle; generic fixture lr0.01
        # amplifies reduction-order error after the first update.
        def fixture_optimizer(value):
            result = optimizer_for(value)
            for group in result.param_groups:
                group["lr"] = 0.001
            return result

        reference_optimizer = fixture_optimizer(reference_adapter)
        owner = ModelPreparation(
            ModelLoadCoordinator(adapter, accelerator),
            ParallelismArguments(cp_size=2, tp_size=2, sequence_parallel=True),
        )
        owner.parallelize(AttentionBackendAccelerator(backend="native"))
        model, optimizer = owner.prepare(fixture_optimizer(adapter))
        owner.install_routes()
        result["updates"] = []
        result["geometry"] = {
            "heads": 30,
            "head_dim": 128,
            "text_tokens": [19, 41],
            "image_tokens": 16,
            "padded_unified_tokens": 96,
            "reduced_depth": 2,
            "learned_pad_initialization": "normal_0_0.05",
        }

        def inputs(replica, step):
            values = model_inputs(step + 11 * replica, dtype=torch.float32)
            return {
                name: (
                    [v.to(accelerator.device) for v in value]
                    if isinstance(value, list)
                    else value.to(accelerator.device) if isinstance(value, torch.Tensor) else value
                )
                for name, value in values.items()
            }

        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
            expected_outputs = []
            for replica in range(2):
                expected = torch.stack(reference("transformer", **inputs(replica, step))[0])
                expected_outputs.append(expected.detach())
                (expected.square().mean() / 2).backward()
            output = torch.stack(adapter.forward(**inputs(rank // 4, step))[0])
            torch.testing.assert_close(output, expected_outputs[rank // 4], rtol=3e-4, atol=2e-6)
            if step == 0:
                accelerator.backward(output.square().mean())
                actual_parameters = parameter_map(model)
                mismatches = 0
                for name, parameter in parameter_map(reference).items():
                    if parameter.grad is None:
                        assert actual_parameters[name].grad is None, name
                        continue
                    gradient = full_value(actual_parameters[name].grad)
                    torch.testing.assert_close(
                        gradient, parameter.grad / 2, rtol=3e-4, atol=2e-6, msg=name
                    )
                    mismatches += int(
                        not torch.allclose(gradient, parameter.grad, rtol=3e-4, atol=2e-6)
                    )
                assert mismatches
                result["unscaled_loss_control"] = {
                    "gradient_factor": 0.5,
                    "mismatched_parameters": mismatches,
                }
                optimizer.zero_grad(set_to_none=True)
                output = torch.stack(adapter.forward(**inputs(rank // 4, step))[0])
            accelerator.backward(output.square().mean() * 2)
            result["updates"].append(
                assert_update_parity(reference, model, reference_optimizer, optimizer)
            )
        result["status"] = "pass"
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result))
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 8, reason="requires DP2 TP2 CP2 on eight GPUs")
@pytest.mark.parametrize("mode", ["full", "lora"])
def test_cuda_z_image_tp_sp_cp_reference(tmp_path, mode):
    rows = run_gloo_ranks(_gpu_worker, tmp_path, 8, {"mode": mode})
    assert all(row.get("status") == "pass" for row in rows), rows


@pytest.mark.skipif(torch.cuda.device_count() < 8, reason="requires DP2 TP2 CP2 on eight GPUs")
def test_cuda_z_image_guard_tp_sp_cp_ordered_lora(tmp_path):
    from context_parallel_overlap_runner import qualify_overlap

    qualify_overlap(
        tmp_path,
        dict(
            case_module="z_image_coupled_fixture",
            model_family="z-image",
            algorithm="grpo-guard",
            dp_size=2,
            tp_size=2,
            cp_size=2,
            training_mode="lora",
            scheduling_mode="ordered",
            sampler_type="rank_local",
        ),
    )
