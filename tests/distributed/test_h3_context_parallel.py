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

"""Native eight-rank H3 packed audio/video/text reference and conditioned lifecycle witnesses."""

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


def _parallelize_and_build_optimizers(owner, adapter, reference_adapter):
    """Bind optimizers to the live parameter identities created by TP."""
    from flow_factory.acceleration.attention_backend import AttentionBackendAccelerator

    owner.parallelize(AttentionBackendAccelerator(backend="native"))
    optimizers = tuple(optimizer_for(target) for target in (reference_adapter, adapter))
    # Keep the real-width native attention trajectory in a bounded regime.
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group["lr"] = 0.001
    return optimizers


def _worker(rank, store_path, result_dir, options):
    initialize_cuda(rank, store_path, world_size=8)
    result = {
        "rank": rank,
        "execution_scope": "model_probe",
        "execution_capability": "minimax-h3-packed-tp-ulysses-fp32-probe-v1",
        **options,
    }
    try:
        from accelerate import FullyShardedDataParallelPlugin
        from accelerate.parallelism_config import ParallelismConfig
        from h3_cp_fixture import adapter_class, build_transformer, install_lora, model_inputs

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

        cls = adapter_class(options["workflow"])
        component = cls.transformer_component_name

        class H3PreparationAdapter(PreparationAdapter):
            context_parallel_component_names = (component,)

            def __init__(self, module, accelerator, mode):
                from types import SimpleNamespace

                from flow_factory.models.runtime import PseudoPipelineRuntime

                super().__init__(module, accelerator, mode)
                self.model_args.target_components = [component]
                self.target_module_map = {component: "all"}
                self.component_runtime = PseudoPipelineRuntime(
                    SimpleNamespace(), {component: module}
                )
                self._components = self.component_runtime.override_components

            @property
            def trainable_component_names(self):
                return [component]

            def forward(self, **kwargs):
                return self.get_component(component)(**kwargs)

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
                transformer_cls_names_to_wrap=["MiniMaxH3TransformerBlock"],
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
        from flow_factory.models.minimax_h3._chunking import (
            install_h3_attention_norm_chunking,
            install_h3_feed_forward_chunking,
            install_h3_lora_projection_chunking,
            install_h3_rotary_chunking,
        )

        module = build_transformer().float()
        reference_module = copy.deepcopy(module)
        install_h3_feed_forward_chunking(module, max_tokens=3)
        install_h3_attention_norm_chunking(module, max_tokens=3)
        if options["workflow"] == "ref2va":
            install_h3_rotary_chunking(module, max_tokens=3)
        if options["mode"] == "lora":
            torch.manual_seed(73)
            reference_module = install_lora(reference_module).float()
            torch.manual_seed(73)
            module = install_lora(module).float()
            if options["workflow"] == "ref2va":
                install_h3_lora_projection_chunking(
                    peel_peft_wrapper(module), max_tokens=3, chunk_output_projection=False
                )
        reference_module = reference_module.to(accelerator.device)
        module.set_attention_backend("native")
        reference_module.set_attention_backend("native")
        adapter = H3PreparationAdapter(module, accelerator, options["mode"])
        adapter.declare_component_variants(("policy",))
        reference_adapter = H3PreparationAdapter(reference_module, accelerator, options["mode"])
        reference_adapter.declare_component_variants(("policy",))
        reference = ModelBundle(reference_adapter.component_variant_registry.bundle_members())
        owner = ModelPreparation(
            ModelLoadCoordinator(adapter, accelerator),
            ParallelismArguments(cp_size=2, tp_size=2, sequence_parallel=True),
        )
        reference_optimizer, actual_optimizer = _parallelize_and_build_optimizers(
            owner, adapter, reference_adapter
        )
        native = peel_peft_wrapper(module)
        block_shapes, mask_values = [], []
        cv, ca = {"t2va": (0, 0), "fl2va": (4, 0), "ref2va": (6, 2)}[options["workflow"]]
        packed_rows = 25 + cv + ca
        padded_rows = (packed_rows + 3) // 4 * 4

        def observe_block(block, args, kwargs):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            indices = kwargs.get("adaln_indices", args[2] if len(args) > 2 else None)
            assert hidden.shape[1] == indices.numel() == padded_rows // 4
            block_shapes.append([list(hidden.shape), indices.detach().cpu().tolist()])

        def observe_attention(attn, args, kwargs):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            mask = kwargs.get("attention_mask", args[2] if len(args) > 2 else None)
            assert hidden.shape[1] == padded_rows // 2
            expected = (
                torch.arange(padded_rows // 2, device=hidden.device)
                + ((rank // 2) % 2) * (padded_rows // 2)
                < packed_rows
            )[None, None, None, :]
            torch.testing.assert_close(mask, expected, rtol=0, atol=0)
            mask_values.append(mask.detach().cpu().flatten().tolist())

        observations = []
        for block in native.transformer_blocks:
            block.register_forward_pre_hook(observe_block, with_kwargs=True)
            observations.append(
                block.attn.register_forward_pre_hook(observe_attention, with_kwargs=True)
            )
        model, optimizer = owner.prepare(actual_optimizer)
        owner.install_routes()
        from torch.distributed.tensor import DTensor, Shard

        modulation_placements = {}
        for name, parameter in parameter_map(model).items():
            if not name.endswith(("adaln_proj.linear.weight", "norm_out.linear.weight")):
                continue
            assert isinstance(parameter, DTensor), name
            assert parameter.device_mesh.mesh_dim_names == ("dp_shard_cp", "tp"), name
            assert tuple(parameter.device_mesh.mesh.shape) == (4, 2), name
            assert parameter.placements == (Shard(0), Shard(1)), name
            assert parameter.numel() == parameter.to_local().numel() * 8, name
            modulation_placements[name] = {
                "mesh_names": list(parameter.device_mesh.mesh_dim_names),
                "mesh_shape": list(parameter.device_mesh.mesh.shape),
                "placements": [str(value) for value in parameter.placements],
                "global_shape": list(parameter.shape),
                "local_shape": list(parameter.to_local().shape),
            }
        assert len(modulation_placements) == 3
        result.update(
            cp_preparation_owner="diffusers",
            fsdp_mesh=list(accelerator.parallelism_config.fsdp_dim_names),
            canonical_component=component,
            published_heads=56,
            head_width=128,
            generated_tokens={"video": 12, "audio": 8},
            condition_tokens={"video": cv, "audio": ca},
            packed_rows=packed_rows,
            padded_rows=padded_rows,
            cp_tokens=padded_rows // 2,
            sp_tokens=padded_rows // 4,
            modulation_placements=modulation_placements,
            updates=[],
        )

        def inputs(replica, step):
            return {
                key: value.to(accelerator.device) if torch.is_tensor(value) else value
                for key, value in model_inputs(options["workflow"], replica, step, odd=True).items()
            }

        def objective(outputs):
            video, audio = outputs[:2]
            return video[:, cv:].square().mean() * 0.7 + audio[:, ca:].square().mean() * 0.3

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
                output = reference(component, **inputs(replica, step))
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
        assert block_shapes and mask_values
        for handle in observations + native._ff_h3_padding_mask_handles:
            handle.remove()
        with torch.no_grad():
            expected = reference(component, **inputs(rank // 4, 2))
            unmasked = adapter.forward(**inputs(rank // 4, 2))
        residual = max((left - right).abs().max().item() for left, right in zip(unmasked, expected))
        assert residual > 2e-6
        result.update(
            block_shapes=block_shapes,
            key_masks=mask_values,
            unmasked_padding_residual=residual,
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
@pytest.mark.parametrize("workflow", ("t2va", "fl2va", "ref2va"))
@pytest.mark.parametrize("mode", ("full", "lora"))
def test_cuda_h3_packed_tp_sp_cp_reference(tmp_path, workflow, mode):
    results = run_gloo_ranks(
        _worker,
        tmp_path,
        8,
        {
            "workflow": workflow,
            "mode": mode,
            "tp_size": 2,
            "sequence_parallel": True,
            "checkpointing": True,
        },
    )
    assert all(row.get("status") == "pass" and "error" not in row for row in results), results
    print(json.dumps({"probe": "minimax_h3_packed_tp_sp_cp_fsdp2", "ranks": results}))


@pytest.mark.parametrize("workflow,mode", (("fl2va", "lora"), ("ref2va", "full")))
def test_cuda_h3_grpo_tp_sp_cp_ordered_lifecycle(tmp_path, workflow, mode):
    from context_parallel_overlap_runner import qualify_overlap

    qualify_overlap(
        tmp_path,
        {
            "case_module": f"h3_{workflow}_coupled_fixture",
            "model_family": f"minimax_h3_{workflow}",
            "algorithm": "grpo",
            "training_mode": mode,
            "scheduling_mode": "ordered",
            "dp_size": 2,
            "tp_size": 2,
            "cp_size": 2,
            "sequence_parallel": True,
        },
    )
