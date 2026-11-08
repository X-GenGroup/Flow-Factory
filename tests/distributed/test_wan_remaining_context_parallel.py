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

"""Eight-rank remaining Wan full/LoRA native gradients and three Adam updates."""

import copy
import json
import traceback
from pathlib import Path
from types import SimpleNamespace

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
from wan_remaining_cp_fixture import VARIANTS


def _worker(rank, store_path, result_dir, options):
    initialize_cuda(rank, store_path, world_size=8)
    result = {
        "rank": rank,
        "execution_scope": "model_probe",
        **options,
        "execution_capability": "wan-remaining-tp-ulysses-fp32-probe-v1",
    }
    try:
        from accelerate import FullyShardedDataParallelPlugin
        from accelerate.parallelism_config import ParallelismConfig
        from wan_remaining_cp_fixture import (
            adapter_class,
            build_transformer,
            forward_inputs,
            install_lora,
            native_inputs,
        )

        from flow_factory.acceleration.attention_backend import AttentionBackendAccelerator
        from flow_factory.hparams.parallelism_args import ParallelismArguments
        from flow_factory.loading.coordinator import ModelLoadCoordinator
        from flow_factory.models.model_bundle import ModelBundle
        from flow_factory.models.runtime import PseudoPipelineRuntime
        from flow_factory.models.tensor_parallel import peel_peft_wrapper
        from flow_factory.trainers.common.model_preparation import ModelPreparation
        from flow_factory.trainers.loader import Accelerator
        from flow_factory.utils.dist import (
            context_parallel_geometry,
            data_parallel_geometry,
            tensor_parallel_geometry,
        )

        variant = options["variant"]
        cls = adapter_class(variant)

        class WanPreparationAdapter(PreparationAdapter):
            context_parallel_capability = cls.context_parallel_capability
            context_parallel_component_names = cls.context_parallel_component_names
            validate_context_parallel_module = cls.validate_context_parallel_module
            tensor_parallel_plan = cls.tensor_parallel_plan
            sequence_parallel_plan = cls.sequence_parallel_plan
            compose_context_parallel_sequence_boundaries = (
                cls.compose_context_parallel_sequence_boundaries
            )

            def __init__(self, modules, accelerator, mode):
                self.model_args = SimpleNamespace(
                    finetune_type=mode,
                    trainable_parameters_dtype=torch.float32,
                    target_components=list(modules),
                )
                self.accelerator = accelerator
                self.target_module_map = dict.fromkeys(modules, "all")
                self.component_runtime = PseudoPipelineRuntime(SimpleNamespace(), modules)
                self._components = self.component_runtime.override_components

            @property
            def trainable_component_names(self):
                return list(self.target_module_map)

            def forward(self, component, **kwargs):
                return self.get_component(component)(**kwargs)

        accelerator = Accelerator(
            mixed_precision="no",
            fsdp_plugin=FullyShardedDataParallelPlugin(
                fsdp_version=2,
                auto_wrap_policy="TRANSFORMER_BASED_WRAP",
                transformer_cls_names_to_wrap=["WanTransformerBlock"],
                cpu_ram_efficient_loading=False,
                reshard_after_forward=True,
                activation_checkpointing=True,
            ),
            parallelism_config=ParallelismConfig(
                dp_replicate_size=1,
                dp_shard_size=2,
                cp_size=2,
                tp_size=2,
                sp_size=1,
            ),
        )
        assert data_parallel_geometry(accelerator) == (2, rank // 4)
        assert context_parallel_geometry(accelerator) == (2, (rank // 2) % 2)
        assert tensor_parallel_geometry(accelerator) == (2, rank % 2)
        result["runtime"] = runtime_observation(accelerator) | {"cp_size": 2, "tp_size": 2}
        names = ("transformer", "transformer_2") if variant.endswith("dual") else ("transformer",)
        modules = {
            name: build_transformer(variant, seed=29 + index * 42).float()
            for index, name in enumerate(names)
        }
        if options["mode"] == "lora":
            modules = {
                name: install_lora(module, variant).float() for name, module in modules.items()
            }
        reference_modules = {
            name: copy.deepcopy(module).to(accelerator.device) for name, module in modules.items()
        }
        for module in (*modules.values(), *reference_modules.values()):
            module.set_attention_backend("native")
        adapter = WanPreparationAdapter(modules, accelerator, options["mode"])
        adapter.declare_component_variants(("policy",))
        reference_adapter = WanPreparationAdapter(reference_modules, accelerator, options["mode"])
        reference_adapter.declare_component_variants(("policy",))
        reference = ModelBundle(reference_adapter.component_variant_registry.bundle_members())
        reference_optimizer = optimizer_for(reference_adapter)
        owner = ModelPreparation(
            ModelLoadCoordinator(adapter, accelerator),
            ParallelismArguments(cp_size=2, tp_size=2, sequence_parallel=True),
        )
        owner.parallelize(AttentionBackendAccelerator(backend="native"))
        block_shapes, attention_shapes, times = [], [], []
        expanded = variant.endswith("expanded")
        cp_rank = (rank // 2) % 2

        def observe_time(module, args, kwargs):
            timestep = kwargs.get("timestep", args[0] if args else None)
            assert timestep.shape == ((6,) if expanded else (1,)), timestep.shape
            if variant == "i2v_expanded":
                expected = torch.tensor([0.0] * 4 + [600.0] * 8, device=timestep.device)
                expected = expected[cp_rank * 6 : (cp_rank + 1) * 6]
                torch.testing.assert_close(timestep, expected, rtol=0, atol=0)
            times.append(timestep.detach().cpu().tolist())

        def observe_block(module, args, kwargs):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            time = kwargs.get("temb", args[2] if len(args) > 2 else None)
            assert hidden.shape[1] == 3, hidden.shape
            assert time.ndim == (4 if expanded else 3), time.shape
            if expanded:
                assert time.shape[1:3] == (3, 6), time.shape
            block_shapes.append([list(hidden.shape), list(time.shape)])

        def observe_attention(module, args, kwargs):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            assert hidden.shape[1] == 6, hidden.shape
            attention_shapes.append(list(hidden.shape))

        for module in modules.values():
            native = peel_peft_wrapper(module)
            native.condition_embedder.register_forward_pre_hook(observe_time, with_kwargs=True)
            for block in native.blocks:
                block.register_forward_pre_hook(observe_block, with_kwargs=True)
                block.attn1.register_forward_pre_hook(observe_attention, with_kwargs=True)
        model, optimizer = owner.prepare(optimizer_for(adapter))
        owner.install_routes()
        assert len(owner._context_parallel.modules) == len(names)
        result.update(
            cp_preparation_owner="diffusers",
            fsdp_mesh=list(accelerator.parallelism_config.fsdp_dim_names),
            components=list(names),
            published_heads=24 if expanded else 40,
            image_tokens=514 if variant == "flf2v" else 0,
            text_tokens=512,
            generated_tokens=12,
            cp_tokens=6,
            sp_tokens=3,
            updates=[],
        )

        def inputs(replica, step, index):
            args = forward_inputs(
                variant,
                replica,
                step,
                accelerator.device,
                timestep=(960.0 if variant.endswith("dual") else 600.0) if index == 0 else 400.0,
            )
            return native_inputs(variant, args)

        def actual_loss(step):
            terms = []
            for index, name in enumerate(names):
                output = adapter.forward(name, **inputs(rank // 4, step, index))[0]
                torch.testing.assert_close(
                    output, expected_outputs[(rank // 4, name)], rtol=3e-4, atol=2e-6
                )
                terms.append(output.square().mean() / len(names))
            return sum(terms)

        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
            expected_outputs = {}
            for replica in range(2):
                for index, name in enumerate(names):
                    output = reference(name, **inputs(replica, step, index))[0]
                    expected_outputs[(replica, name)] = output.detach()
                    (output.square().mean() / (2 * len(names))).backward()
            if step == 0:
                accelerator.backward(actual_loss(step))
                expected_parameters, actual_parameters = parameter_map(reference), parameter_map(
                    model
                )
                mismatches = 0
                for name, parameter in expected_parameters.items():
                    if not parameter.requires_grad:
                        assert parameter.grad is None and actual_parameters[name].grad is None, name
                        continue
                    assert (
                        parameter.grad is not None and actual_parameters[name].grad is not None
                    ), name
                    gradient = full_value(actual_parameters[name].grad)
                    torch.testing.assert_close(
                        gradient, parameter.grad / 2, rtol=3e-4, atol=2e-6, msg=name
                    )
                    mismatches += int(
                        not torch.allclose(gradient, parameter.grad, rtol=3e-4, atol=2e-6)
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
        assert block_shapes and attention_shapes and times
        result.update(
            block_shapes=block_shapes,
            attention_shapes=attention_shapes,
            timestep_values=times,
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
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("mode", ("full", "lora"))
def test_cuda_wan_remaining_tp_sp_cp_reference(tmp_path, variant, mode):
    results = run_gloo_ranks(
        _worker,
        tmp_path,
        8,
        {
            "variant": variant,
            "mode": mode,
            "tp_size": 2,
            "sequence_parallel": True,
            "checkpointing": True,
        },
    )
    assert all(row.get("status") == "pass" and "error" not in row for row in results), results
    print(json.dumps({"probe": "wan_remaining_tp_sp_cp_fsdp2", "ranks": results}))


@pytest.mark.parametrize(
    "variant,mode", (("i2v_dual", "full"), ("i2v_expanded", "full"), ("flf2v", "lora"))
)
def test_cuda_wan_remaining_grpo_tp_sp_cp_ordered_lifecycle(tmp_path, variant, mode):
    from context_parallel_overlap_runner import qualify_overlap

    qualify_overlap(
        tmp_path,
        {
            "case_module": f"wan_{variant}_coupled_fixture",
            "model_family": f"wan-{variant}",
            "algorithm": "grpo",
            "training_mode": mode,
            "scheduling_mode": "ordered",
            "dp_size": 2,
            "tp_size": 2,
            "cp_size": 2,
        },
    )
