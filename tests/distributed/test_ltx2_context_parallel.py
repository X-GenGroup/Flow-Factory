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

"""Native eight-rank LTX audio/video reference and conditioned lifecycle witnesses."""

import copy
import json
import traceback
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from conftest import run_gloo_ranks
from test_model_preparation import (
    PreparationAdapter,
    assert_update_parity,
    full_value,
    initialize_cuda,
    optimizer_for,
    parameter_map,
    runtime_observation,
)


def _ltx2_optimizer_for(adapter: PreparationAdapter) -> torch.optim.AdamW:
    """Match the conditioned LTX2 lifecycle fixture's learning-rate scale."""
    optimizer = optimizer_for(adapter)
    optimizer.defaults["lr"] = 1e-3
    for group in optimizer.param_groups:
        group["lr"] = 1e-3
    return optimizer


def _worker(rank, store_path, result_dir, options):
    initialize_cuda(rank, store_path, world_size=8)
    result = {
        "rank": rank,
        "execution_scope": "model_probe",
        "execution_capability": "ltx2-joint-av-tp-ulysses-fp32-probe-v1",
        **options,
    }
    try:
        from accelerate import FullyShardedDataParallelPlugin
        from accelerate.parallelism_config import ParallelismConfig
        from ltx2_cp_fixture import adapter_class, build_transformer, install_lora, model_inputs

        from flow_factory.acceleration.attention_backend import AttentionBackendAccelerator
        from flow_factory.hparams.parallelism_args import ParallelismArguments
        from flow_factory.loading.coordinator import ModelLoadCoordinator
        from flow_factory.models.model_bundle import ModelBundle
        from flow_factory.models.tensor_parallel import peel_peft_wrapper
        from flow_factory.trainers.common.model_preparation import ModelPreparation
        from flow_factory.trainers.loader import Accelerator
        from flow_factory.utils.dist import (
            context_parallel_geometry,
            data_parallel_geometry,
            tensor_parallel_geometry,
        )

        cls = adapter_class(options["per_token"])

        class LTXPreparationAdapter(PreparationAdapter):
            context_parallel_capability = cls.context_parallel_capability
            validate_context_parallel_module = cls.validate_context_parallel_module
            configure_context_parallel_module = cls.configure_context_parallel_module
            install_context_parallel_module = cls.install_context_parallel_module
            tensor_parallel_plan = cls.tensor_parallel_plan
            sequence_parallel_plan = cls.sequence_parallel_plan
            compose_context_parallel_sequence_boundaries = (
                cls.compose_context_parallel_sequence_boundaries
            )

        accelerator = Accelerator(
            mixed_precision="no",
            fsdp_plugin=FullyShardedDataParallelPlugin(
                fsdp_version=2,
                auto_wrap_policy="TRANSFORMER_BASED_WRAP",
                transformer_cls_names_to_wrap=["LTX2VideoTransformerBlock"],
                cpu_ram_efficient_loading=False,
                reshard_after_forward=True,
                activation_checkpointing=True,
            ),
            parallelism_config=ParallelismConfig(
                dp_replicate_size=1, dp_shard_size=2, cp_size=2, tp_size=2, sp_size=1
            ),
        )
        assert data_parallel_geometry(accelerator) == (2, rank // 4)
        assert context_parallel_geometry(accelerator) == (2, (rank // 2) % 2)
        assert tensor_parallel_geometry(accelerator) == (2, rank % 2)
        result["runtime"] = runtime_observation(accelerator) | {"cp_size": 2, "tp_size": 2}
        module = build_transformer(options["ltx23"]).float()
        if options["mode"] == "lora":
            module = install_lora(module, options["per_token"]).float()
        reference_module = copy.deepcopy(module).to(accelerator.device)
        module.set_attention_backend("native")
        reference_module.set_attention_backend("native")
        adapter = LTXPreparationAdapter(module, accelerator, options["mode"])
        adapter.declare_component_variants(("policy",))
        reference_adapter = LTXPreparationAdapter(reference_module, accelerator, options["mode"])
        reference_adapter.declare_component_variants(("policy",))
        reference = ModelBundle(reference_adapter.component_variant_registry.bundle_members())
        reference_optimizer = _ltx2_optimizer_for(reference_adapter)
        owner = ModelPreparation(
            ModelLoadCoordinator(adapter, accelerator),
            ParallelismArguments(cp_size=2, tp_size=2, sequence_parallel=True),
        )
        owner.parallelize(AttentionBackendAccelerator(backend="native"))
        native = peel_peft_wrapper(module)
        block_shapes = []
        cross_shapes = []
        time_values = []

        def observe_block(block, args, kwargs):
            video = kwargs.get("hidden_states", args[0] if args else None)
            audio = kwargs.get("audio_hidden_states", args[1] if len(args) > 1 else None)
            assert video.shape[1] == 3 and audio.shape[1] == 2, (video.shape, audio.shape)
            block_shapes.append([list(video.shape), list(audio.shape)])

        def observe_cross(attn, args, kwargs):
            query = kwargs.get("hidden_states", args[0] if args else None)
            source = kwargs.get("encoder_hidden_states", args[1] if len(args) > 1 else None)
            assert (query.shape[1], source.shape[1]) in ((6, 4), (4, 6))
            cross_shapes.append([list(query.shape), list(source.shape)])

        def observe_time(time, args, kwargs):
            value = kwargs.get("timestep", args[0] if args else None)
            if options["per_token"]:
                expected = torch.tensor([0.0] * 4 + [500.0] * 8, device=value.device)[
                    ((rank // 2) % 2) * 6 : ((rank // 2) % 2 + 1) * 6
                ]
                torch.testing.assert_close(value, expected, rtol=0, atol=0)
            else:
                assert value.shape == (1,)
            time_values.append(value.detach().cpu().tolist())

        native.time_embed.register_forward_pre_hook(observe_time, with_kwargs=True)
        for block in native.transformer_blocks:
            block.register_forward_pre_hook(observe_block, with_kwargs=True)
            block.audio_to_video_attn.register_forward_pre_hook(observe_cross, with_kwargs=True)
            block.video_to_audio_attn.register_forward_pre_hook(observe_cross, with_kwargs=True)
        model, optimizer = owner.prepare(_ltx2_optimizer_for(adapter))
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
        result.update(
            cp_preparation_owner="diffusers",
            fsdp_mesh=list(accelerator.parallelism_config.fsdp_dim_names),
            published_heads={"video": 32, "audio": 32},
            generated_tokens={"video": 12, "audio": 8},
            active_video_tokens=8 if options["per_token"] else 12,
            cp_tokens={"video": 6, "audio": 4},
            sp_tokens={"video": 3, "audio": 2},
            updates=[],
        )

        def inputs(replica, step):
            return model_inputs(
                options["ltx23"], options["per_token"], replica, step, accelerator.device
            )

        def objective(outputs):
            video, audio = outputs[:2]
            if options["per_token"]:
                video = video[:, 4:]
            return video.square().mean() * 0.75 + audio.square().mean() * 0.25

        def actual_loss(step):
            output = adapter.forward(**inputs(rank // 4, step))
            for actual, expected in zip(output[:2], expected_outputs[rank // 4]):
                torch.testing.assert_close(actual, expected, rtol=3e-4, atol=2e-6)
            return objective(output)

        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
            expected_outputs = {}
            for replica in range(2):
                output = reference("transformer", **inputs(replica, step))
                expected_outputs[replica] = [value.detach() for value in output[:2]]
                (objective(output) / 2).backward()
            if step == 0:
                accelerator.backward(actual_loss(step))
                expected = parameter_map(reference)
                actual = parameter_map(model)
                mismatches = 0
                for name, param in expected.items():
                    assert (param.grad is None) == (actual[name].grad is None), name
                    if param.requires_grad:
                        assert param.grad is not None, name
                        gradient = full_value(actual[name].grad)
                        torch.testing.assert_close(
                            gradient, param.grad / 2, rtol=3e-4, atol=2e-6, msg=name
                        )
                        mismatches += int(
                            not torch.allclose(gradient, param.grad, rtol=3e-4, atol=2e-6)
                        )
                assert mismatches > 0
                result["unscaled_loss_control"] = {
                    "gradient_factor": 0.5,
                    "mismatched_parameters": mismatches,
                }
                optimizer.zero_grad(set_to_none=True)
            accelerator.backward(actual_loss(step) * 2)
            result["updates"].append(
                assert_update_parity(reference, model, reference_optimizer, optimizer)
            )
        assert block_shapes and cross_shapes and time_values
        result.update(
            block_shapes=block_shapes,
            cross_attention_shapes=cross_shapes,
            timestep_values=time_values,
            status="pass",
        )
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result, indent=2))
        dist.destroy_process_group()


@pytest.mark.skipif(
    torch.cuda.device_count() < 8, reason="DP2 x TP2 x CP2 requires eight CUDA ranks"
)
@pytest.mark.parametrize("ltx23", (False, True), ids=("ltx20", "ltx23"))
@pytest.mark.parametrize("per_token", (False, True), ids=("t2av", "i2av"))
@pytest.mark.parametrize("mode", ("full", "lora"))
def test_cuda_ltx2_joint_av_tp_sp_cp_reference(tmp_path, ltx23, per_token, mode):
    results = run_gloo_ranks(
        _worker,
        tmp_path,
        8,
        {
            "ltx23": ltx23,
            "per_token": per_token,
            "mode": mode,
            "tp_size": 2,
            "sequence_parallel": True,
            "checkpointing": True,
        },
    )
    assert all(row.get("status") == "pass" and "error" not in row for row in results), results
    print(json.dumps({"probe": "ltx2_joint_av_tp_sp_cp_fsdp2", "ranks": results}))


@pytest.mark.parametrize("variant,mode", (("ltx20_t2av", "lora"), ("ltx23_i2av", "full")))
def test_cuda_ltx2_grpo_tp_sp_cp_ordered_lifecycle(tmp_path, variant, mode):
    from context_parallel_overlap_runner import qualify_overlap

    qualify_overlap(
        tmp_path,
        {
            "case_module": f"{variant}_coupled_fixture",
            "model_family": variant,
            "algorithm": "grpo",
            "training_mode": mode,
            "scheduling_mode": "ordered",
            "dp_size": 2,
            "tp_size": 2,
            "cp_size": 2,
            "sequence_parallel": True,
        },
    )
