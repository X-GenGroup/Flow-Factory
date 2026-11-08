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

"""Independent Qwen image-only SP plus native TP/CP controls against an unsharded model."""

from __future__ import annotations

import copy
import json
import math
import traceback
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from conftest import requires_gloo, run_gloo_ranks
from qwen21_cp_fixture import model_inputs, reference_forward


def _cpu_worker(rank, store_path, result_dir, options):
    """Exercise real native Ulysses autograd on CPU, without standing in for FSDP evidence."""
    from types import SimpleNamespace

    from diffusers.models._modeling_parallel import ContextParallelConfig
    from qwen21_cp_fixture import build_transformer, install_lora
    from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
    from torch.distributed.tensor import DTensor

    from flow_factory.models.qwen_image_21.qwen_image_21 import QwenImage21Adapter
    from flow_factory.models.tensor_parallel import apply_tensor_parallel, peel_peft_wrapper

    result = {"rank": rank, **options}
    dist.init_process_group("gloo", init_method=f"file://{store_path}", rank=rank, world_size=4)
    try:
        whole = init_device_mesh("cpu", (2, 2), mesh_dim_names=("cp", "tp"))
        mesh = whole["tp"]
        module = build_transformer(heads=4)
        if options["mode"] == "lora":
            module = install_lora(module)
        reference = copy.deepcopy(module)
        raw = peel_peft_wrapper(module)
        apply_tensor_parallel(
            raw,
            mesh,
            {
                **QwenImage21Adapter.tensor_parallel_plan(),
                **QwenImage21Adapter.sequence_parallel_plan(),
            },
        )
        native_mesh = DeviceMesh(
            "cpu", whole.mesh.T.unsqueeze(1), mesh_dim_names=("tensor", "ring", "ulysses")
        )
        QwenImage21Adapter.configure_context_parallel_module(raw)
        QwenImage21Adapter.install_context_parallel_module(
            raw, ContextParallelConfig(ring_degree=1, ulysses_degree=2, mesh=native_mesh)
        )
        projection_rows = []

        def observe_output_projection(owner, args):
            state = raw._ff_qwen21_parallel_state
            if state.active:
                expected_rows = math.ceil(state.call.get().length / state.degree)
                assert args[0].shape[1] == expected_rows
                projection_rows.append(expected_rows)

        raw.proj_out.register_forward_pre_hook(observe_output_projection)
        adapter = object.__new__(QwenImage21Adapter)
        adapter.component_runtime = SimpleNamespace(get_component=lambda name: module)
        adapter._warned_cfg_without_negative = adapter._warned_negative_without_cfg = False
        result["cases"] = []
        for checkpoint in (False, True):
            if checkpoint:
                module.enable_gradient_checkpointing()
                reference.enable_gradient_checkpointing()
            for cached in (False, True):
                module.zero_grad(set_to_none=True)
                reference.zero_grad(set_to_none=True)
                inputs = model_inputs(0, 0, "cpu")
                projection_rows.clear()
                actual = adapter._predict_velocity_one(**inputs, use_kv_cache=cached)
                assert len(projection_rows) == 2  # Positive and negative CFG branches.
                expected = reference_forward(reference, inputs)
                torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-6)
                actual.square().mean().backward()
                expected.square().mean().backward()
                gradients = 0
                max_relative = 0.0
                for (name, parameter), (refname, refparam) in zip(
                    module.named_parameters(), reference.named_parameters()
                ):
                    assert name == refname
                    assert (parameter.grad is None) == (refparam.grad is None), name
                    if parameter.grad is None:
                        assert not parameter.requires_grad, name
                        continue
                    gradient = (
                        parameter.grad.full_tensor()
                        if isinstance(parameter.grad, DTensor)
                        else parameter.grad
                    )
                    gradient = gradient.clone()
                    dist.all_reduce(gradient, group=whole["cp"].get_group())
                    torch.testing.assert_close(
                        gradient, refparam.grad, rtol=3e-4, atol=2e-6, msg=name
                    )
                    error = (gradient - refparam.grad).norm() / refparam.grad.norm().clamp_min(
                        1e-15
                    )
                    assert error < 3e-5, (name, error)
                    max_relative = max(max_relative, float(error))
                    gradients += 1
                assert raw._ff_qwen21_parallel_state.call.get() is None
                assert raw._ff_qwen21_parallel_state.tokens.get() == ()
                result["cases"].append(
                    {
                        "cache": cached,
                        "checkpointing": checkpoint,
                        "gradients": gradients,
                        "relative_gradient_error": max_relative,
                        "output_projection_token_rows": list(projection_rows),
                    }
                )
        result["status"] = "pass"
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result, indent=2))
        dist.destroy_process_group()


@requires_gloo
@pytest.mark.parametrize("mode", ["full", "lora"])
def test_cpu_qwen21_prefix_and_joint_native_collectives(tmp_path, mode):
    results = run_gloo_ranks(_cpu_worker, tmp_path, 4, {"mode": mode})
    assert all(row.get("status") == "pass" and "error" not in row for row in results), results


from test_model_preparation import (
    PreparationAdapter,
    assert_update_parity,
    full_value,
    initialize_cuda,
    optimizer_for,
    parameter_map,
    runtime_observation,
)


def _qwen21_optimizer_for(adapter: PreparationAdapter) -> torch.optim.AdamW:
    """Match the conditioned Qwen Guard fixture's optimizer scale."""
    optimizer = optimizer_for(adapter)
    optimizer.defaults["lr"] = 1e-3
    for group in optimizer.param_groups:
        group["lr"] = 1e-3
    return optimizer


def _worker(rank: int, store_path: str, result_dir: str, options: dict) -> None:
    tp_size = options.get("tp_size", 1)
    initialize_cuda(rank, store_path, world_size=4 * tp_size)
    result = {"rank": rank, "execution_scope": "model_probe", **options}
    if tp_size > 1:
        result["execution_capability"] = "qwen21-prefix-tp-ulysses-fp32-probe-v1"
    try:
        from accelerate import FullyShardedDataParallelPlugin
        from accelerate.parallelism_config import ParallelismConfig
        from qwen21_cp_fixture import build_transformer, install_lora

        from flow_factory.acceleration.attention_backend import AttentionBackendAccelerator
        from flow_factory.hparams.parallelism_args import ParallelismArguments
        from flow_factory.loading.coordinator import ModelLoadCoordinator
        from flow_factory.models.model_bundle import ModelBundle
        from flow_factory.models.qwen_image_21.qwen_image_21 import QwenImage21Adapter
        from flow_factory.models.tensor_parallel import peel_peft_wrapper
        from flow_factory.models.variants import ComponentVariantRegistry
        from flow_factory.trainers.common.model_preparation import ModelPreparation
        from flow_factory.trainers.loader import Accelerator
        from flow_factory.utils.dist import (
            context_parallel_geometry,
            data_parallel_geometry,
            tensor_parallel_geometry,
        )

        class QwenPreparationAdapter(PreparationAdapter):
            configure_context_parallel_module = QwenImage21Adapter.configure_context_parallel_module
            install_context_parallel_module = QwenImage21Adapter.install_context_parallel_module
            _predict_velocity_one = QwenImage21Adapter._predict_velocity_one
            _prefill_kv_cache_one = QwenImage21Adapter._prefill_kv_cache_one
            _warned_cfg_without_negative = False
            _warned_negative_without_cfg = False
            sequence_parallel_by_default = False
            context_parallel_capability = QwenImage21Adapter.context_parallel_capability
            validate_context_parallel_module = QwenImage21Adapter.validate_context_parallel_module
            tensor_parallel_plan = QwenImage21Adapter.tensor_parallel_plan
            sequence_parallel_plan = QwenImage21Adapter.sequence_parallel_plan
            compose_context_parallel_sequence_boundaries = (
                QwenImage21Adapter.compose_context_parallel_sequence_boundaries
            )

        accelerator = Accelerator(
            mixed_precision="no",
            fsdp_plugin=FullyShardedDataParallelPlugin(
                fsdp_version=2,
                auto_wrap_policy="TRANSFORMER_BASED_WRAP",
                transformer_cls_names_to_wrap=["QwenImage21TransformerBlock"],
                cpu_ram_efficient_loading=False,
                reshard_after_forward=True,
                activation_checkpointing=options["checkpointing"],
            ),
            parallelism_config=ParallelismConfig(
                dp_replicate_size=1, dp_shard_size=2, cp_size=2, tp_size=tp_size, sp_size=1
            ),
        )
        result["runtime"] = runtime_observation(accelerator)
        assert data_parallel_geometry(accelerator) == (2, rank // (2 * tp_size))
        cp_geometry = context_parallel_geometry(accelerator)
        tp_geometry = tensor_parallel_geometry(accelerator)
        assert cp_geometry == (2, (rank // tp_size) % 2)
        assert tp_geometry == (2, rank % 2)
        result["runtime"].update(cp_size=cp_geometry[0], tp_size=tp_geometry[0])
        module = build_transformer().float()
        if options["mode"] == "lora":
            module = install_lora(module).float()
        reference_module = copy.deepcopy(module).to(accelerator.device)
        # Native SDPA supports autograd through Ulysses; native Ring needs LSE output.
        module.set_attention_backend("native")
        reference_module.set_attention_backend("native")
        adapter = QwenPreparationAdapter(module, accelerator, options["mode"])
        adapter.declare_component_variants(("policy",))
        reference_adapter = PreparationAdapter(reference_module, accelerator, options["mode"])
        reference_adapter.declare_component_variants(("policy",))
        reference = ModelBundle(reference_adapter.component_variant_registry.bundle_members())
        reference_optimizer = _qwen21_optimizer_for(reference_adapter)
        owner = ModelPreparation(
            ModelLoadCoordinator(adapter, accelerator),
            ParallelismArguments(
                cp_size=2, tp_size=tp_size, sequence_parallel=options.get("sequence_parallel")
            ),
        )
        owner.parallelize(AttentionBackendAccelerator(backend="native"))
        attention_shapes = []
        block_shapes = []

        def observe_attention(target, args, kwargs):
            image = kwargs.get("hidden_states", args[0] if args else None)
            kind = kwargs.get("kv_cache_mode")
            cache = kwargs.get("layer_cache")
            attention_shapes.append(
                {
                    "mode": kind,
                    "tokens": image.shape[1],
                    "cache_tokens": (
                        cache.k.shape[1] if cache is not None and cache.k is not None else None
                    ),
                }
            )

        def observe_sp_block(target, args, kwargs):
            image = kwargs.get("hidden_states", args[0] if args else None)
            kind = kwargs.get("kv_cache_mode", args[6] if len(args) > 6 else None)
            block_shapes.append({"mode": kind, "tokens": image.shape[1]})

        for block in peel_peft_wrapper(module).transformer_blocks:
            block.attn.register_forward_pre_hook(observe_attention, with_kwargs=True)
            block.register_forward_pre_hook(observe_sp_block, with_kwargs=True)
        model, optimizer = owner.prepare(_qwen21_optimizer_for(adapter))
        owner.install_routes()
        optimizer_fields = ("lr", "eps", "betas", "weight_decay", "amsgrad", "maximize")
        result["optimizer_hyperparameters"] = {
            "parallel": [
                {key: group[key] for key in optimizer_fields} for group in optimizer.param_groups
            ],
            "reference": [
                {key: group[key] for key in optimizer_fields}
                for group in reference_optimizer.param_groups
            ],
        }
        assert (
            result["optimizer_hyperparameters"]["parallel"]
            == result["optimizer_hyperparameters"]["reference"]
        )
        assert getattr(accelerator, "_cp_context", None) is None
        result["cp_preparation_owner"] = "diffusers"
        result["fsdp_mesh"] = list(accelerator.parallelism_config.fsdp_dim_names)
        result["updates"] = []
        result["target_tokens"] = 16
        result["ordered_reference_tokens"] = [4, 4]
        result["positive_prefix_tokens"] = 13
        result["negative_prefix_tokens"] = 11
        result["num_attention_heads"] = 32
        result["attention_head_dim"] = 128
        result["inactive_parameters"] = []
        expected_inactive = set()
        # Diffusers gather backward returns only this CP rank's output slice.
        # FSDP averages gradients over DP x CP. Multiplying by CP=2 restores the
        # mean over distinct logical samples rather than averaging token pieces.
        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
            expected_outputs = []
            for replica in range(2):
                output = reference_forward(
                    lambda **kwargs: reference("transformer", **kwargs),
                    model_inputs(replica, step, accelerator.device),
                )
                expected_outputs.append(output.detach())
                (output.square().mean() / 2).backward()
            output = adapter._predict_velocity_one(
                **model_inputs(rank // (2 * tp_size), step, accelerator.device),
                use_kv_cache=options["use_kv_cache"],
            )
            torch.testing.assert_close(
                output, expected_outputs[rank // (2 * tp_size)], rtol=3e-4, atol=2e-6
            )
            if step == 0:
                accelerator.backward(output.square().mean())
                expected_parameters, actual_parameters = parameter_map(reference), parameter_map(
                    model
                )
                mismatches = 0
                for name, parameter in expected_parameters.items():
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
                assert (
                    mismatches > 0
                ), "the unscaled-loss control must detect wrong gradient normalization"
                result["unscaled_loss_control"] = {
                    "gradient_factor": 0.5,
                    "mismatched_parameters": mismatches,
                }
                optimizer.zero_grad(set_to_none=True)
                output = adapter._predict_velocity_one(
                    **model_inputs(rank // (2 * tp_size), step, accelerator.device),
                    use_kv_cache=options["use_kv_cache"],
                )
            accelerator.backward(output.square().mean() * 2)
            for candidate in (reference, model):
                inactive = {
                    name
                    for name, parameter in parameter_map(candidate).items()
                    if parameter.requires_grad and parameter.grad is None
                }
                assert inactive == expected_inactive, (inactive, expected_inactive)
            result["updates"].append(
                assert_update_parity(reference, model, reference_optimizer, optimizer)
            )
        decode = [row for row in attention_shapes if row["mode"] == "cached"]
        extract = [row for row in attention_shapes if row["mode"] == "extract"]
        if options["use_kv_cache"]:
            assert decode and extract
            assert {row["tokens"] for row in decode} == {8}
            assert {row["cache_tokens"] for row in decode} == {11, 13}
            assert {row["tokens"] for row in extract} == {15, 17}
        expected_decode_tokens = 4 if options["sequence_parallel"] else 8
        if options["use_kv_cache"]:
            assert {row["tokens"] for row in block_shapes if row["mode"] == "cached"} == {
                expected_decode_tokens
            }
        else:
            expected_attention_tokens = {14, 16} if options["sequence_parallel"] else {14, 15}
            expected_block_tokens = {7, 8} if options["sequence_parallel"] else {14, 15}
            assert {row["tokens"] for row in attention_shapes} == expected_attention_tokens
            assert {row["tokens"] for row in block_shapes} == expected_block_tokens
        result["attention_stream_shapes"] = attention_shapes
        result["sp_block_stream_shapes"] = block_shapes
        result["checkpoint_recomputation"] = options["checkpointing"]
        result["status"] = "pass"
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result, indent=2))
        dist.destroy_process_group()


@pytest.mark.skipif(
    torch.cuda.device_count() < 8, reason="DP2 x CP2 x TP2 requires eight CUDA ranks"
)
@pytest.mark.parametrize("use_kv_cache", [False, True])
@pytest.mark.parametrize("sequence_parallel", [False, True])
@pytest.mark.parametrize("mode,checkpointing", [("full", False), ("lora", True)])
def test_cuda_qwen21_tp_sp_cp_reference(
    tmp_path, use_kv_cache, sequence_parallel, mode, checkpointing
):
    results = run_gloo_ranks(
        _worker,
        tmp_path,
        8,
        {
            "tp_size": 2,
            "use_kv_cache": use_kv_cache,
            "sequence_parallel": sequence_parallel,
            "checkpointing": checkpointing,
            "mode": mode,
        },
    )
    assert all(row.get("status") == "pass" and "error" not in row for row in results), results
    print(json.dumps({"probe": "qwen21_tp_sp_cp_fsdp2", "ranks": results}))


@pytest.mark.parametrize("use_kv_cache", [False, True])
def test_cuda_qwen21_guard_ordered_lora(tmp_path, use_kv_cache):
    from context_parallel_overlap_runner import qualify_overlap

    qualify_overlap(
        tmp_path,
        {
            "case_module": "qwen21_coupled_fixture",
            "model_family": "qwen-image-2.1",
            "algorithm": "grpo-guard",
            "training_mode": "lora",
            "scheduling_mode": "ordered",
            "dp_size": 2,
            "tp_size": 2,
            "cp_size": 2,
            "sequence_parallel": True,
            "use_kv_cache": use_kv_cache,
        },
    )


def test_native_vl_and_rgba_reference_fixture_preserves_cache_and_joint_predictions():
    from types import SimpleNamespace

    from PIL import Image
    from qwen21_cp_fixture import build_pipeline

    from flow_factory.models.qwen_image_21.qwen_image_21 import QwenImage21Adapter
    from flow_factory.models.runtime.classic import ClassicPipelineRuntime

    pipeline = build_pipeline(heads=4)
    adapter = object.__new__(QwenImage21Adapter)
    adapter.pipeline = pipeline
    adapter.component_runtime = ClassicPipelineRuntime(pipeline)
    adapter._components = adapter.component_runtime.override_components
    adapter.accelerator = SimpleNamespace(device=torch.device("cpu"))
    adapter._warned_cfg_without_negative = adapter._warned_negative_without_cfg = False
    images = [
        Image.new("RGBA", (32, 32), (20, 40, 80, 255)),
        Image.new("RGBA", (64, 32), (70, 30, 10, 255)),
    ]
    with torch.no_grad():
        condition = adapter._prepare_condition_images(images, condition_image_size=(32, 64))
        prompts = adapter.encode_prompt(
            "a red image",
            negative_prompt="blue",
            guidance_scale=2.0,
            images=condition["condition_images"],
        )
    assert condition["condition_img_shapes"] == [(1, 2, 2), (1, 2, 4)]
    assert condition["condition_image_latents"].shape == (1, 12, 64)
    assert int(prompts["image_pad_mask"].sum()) == 3
    assert prompts["prompt_embeds"].shape[1] != prompts["negative_prompt_embeds"].shape[1]
    inputs = {key: value for key, value in prompts.items() if "ids" not in key}
    inputs.update(
        t=torch.tensor([500.0]),
        latents=torch.randn(1, 16, 64),
        img_shapes=[*condition["condition_img_shapes"], (1, 4, 4)],
        condition_image_latents=condition["condition_image_latents"],
        guidance_scale=2.0,
        attention_kwargs=None,
    )
    with torch.no_grad():
        joint = adapter._predict_velocity_one(**inputs, use_kv_cache=False)
        cached = adapter._predict_velocity_one(**inputs, use_kv_cache=True)
    torch.testing.assert_close(joint, cached, rtol=3e-5, atol=3e-6)
    assert joint.shape == (1, 16, 64)
