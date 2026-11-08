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

"""Independent Edit2509/2511 target-only TP/SP/CP gradients and Adam controls."""

from __future__ import annotations

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


def model_inputs(replica, step, device):
    """Six text positions stay CP-local length three and TP-replicated with masked CFG."""
    from qwen_edit_cp_fixture import model_inputs as native_inputs

    inputs = native_inputs(replica, step, device)
    inputs["encoder_hidden_states"] = inputs["encoder_hidden_states"][:, :6].contiguous()
    inputs["encoder_hidden_states_mask"] = inputs["encoder_hidden_states_mask"][:, :6].contiguous()
    return inputs


def _worker(rank: int, store_path: str, result_dir: str, options: dict) -> None:
    tp_size = options.get("tp_size", 1)
    initialize_cuda(rank, store_path, world_size=4 * tp_size)
    result = {"rank": rank, "execution_scope": "model_probe", **options}
    if tp_size > 1:
        result["execution_capability"] = "qwen-edit-tp-ulysses-fp32-probe-v1"
    try:
        from accelerate import FullyShardedDataParallelPlugin
        from accelerate.parallelism_config import ParallelismConfig
        from qwen_edit_cp_fixture import build_transformer, inactive_parameter_names, install_lora

        from flow_factory.acceleration.attention_backend import AttentionBackendAccelerator
        from flow_factory.hparams.parallelism_args import ParallelismArguments
        from flow_factory.loading.coordinator import ModelLoadCoordinator
        from flow_factory.models.model_bundle import ModelBundle
        from flow_factory.models.qwen_image.qwen_image_edit_plus import QwenImageEditPlusAdapter
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
            sequence_parallel_by_default = False
            context_parallel_capability = QwenImageEditPlusAdapter.context_parallel_capability
            validate_context_parallel_module = (
                QwenImageEditPlusAdapter.validate_context_parallel_module
            )
            tensor_parallel_plan = QwenImageEditPlusAdapter.tensor_parallel_plan
            sequence_parallel_plan = QwenImageEditPlusAdapter.sequence_parallel_plan
            compose_context_parallel_sequence_boundaries = (
                QwenImageEditPlusAdapter.compose_context_parallel_sequence_boundaries
            )

        accelerator = Accelerator(
            mixed_precision="no",
            fsdp_plugin=FullyShardedDataParallelPlugin(
                fsdp_version=2,
                auto_wrap_policy="TRANSFORMER_BASED_WRAP",
                transformer_cls_names_to_wrap=["QwenImageTransformerBlock"],
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
        module = build_transformer(options["zero_cond_t"]).float()
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
        reference_optimizer = optimizer_for(reference_adapter)
        owner = ModelPreparation(
            ModelLoadCoordinator(adapter, accelerator),
            ParallelismArguments(
                cp_size=2, tp_size=tp_size, sequence_parallel=options.get("sequence_parallel")
            ),
        )
        owner.parallelize(AttentionBackendAccelerator(backend="native"))
        attention_shapes = []
        block_shapes = []
        modulation_indices = []

        def observe_attention(target, args, kwargs):
            # Registered after SP's gather: attention sees CP-local image and text.
            image = kwargs.get("hidden_states", args[0] if args else None)
            text = kwargs.get("encoder_hidden_states", args[1] if len(args) > 1 else None)
            mask = kwargs.get("encoder_hidden_states_mask", args[2] if len(args) > 2 else None)
            shapes = [list(image.shape), list(text.shape), list(mask.shape)]
            assert shapes == [[2, 12, 64], [2, 4, 64], [2, 4]], shapes
            attention_shapes.append(shapes)

        def observe_sp_block(target, args, kwargs):
            image = kwargs.get("hidden_states", args[0] if args else None)
            text = kwargs.get("encoder_hidden_states", args[1] if len(args) > 1 else None)
            shapes = [list(image.shape), list(text.shape)]
            assert shapes == [[2, 6, 64], [2, 2, 64]], shapes
            index = kwargs.get("modulate_index", args[6] if len(args) > 6 else None)
            if options["zero_cond_t"]:
                start = ((rank // tp_size) % 2) * 12 + (rank % tp_size) * 6
                expected = torch.tensor([0] * 8 + [1] * 16, device=image.device)
                expected = expected[start : start + 6].repeat(2, 1)
                assert torch.equal(index, expected), (index, expected)
                modulation_indices.append(index.detach().cpu().tolist())
            else:
                assert index is None
            block_shapes.append(shapes)

        for block in peel_peft_wrapper(module).transformer_blocks:
            block.attn.register_forward_pre_hook(observe_attention, with_kwargs=True)
            if options["sequence_parallel"]:
                block.register_forward_pre_hook(observe_sp_block, with_kwargs=True)
        model, optimizer = owner.prepare(optimizer_for(adapter))
        owner.install_routes()
        assert getattr(accelerator, "_cp_context", None) is None
        result["cp_preparation_owner"] = "diffusers"
        result["fsdp_mesh"] = list(accelerator.parallelism_config.fsdp_dim_names)
        result["updates"] = []
        result["image_tokens"] = 24
        result["target_tokens"] = 8
        result["reference_tokens"] = [6, 10]
        result["text_tokens"] = 6
        result["cp_local_text_tokens"] = 3
        expected_inactive = {
            ComponentVariantRegistry._canonical_prepared_parameter_name(name)
            for name in inactive_parameter_names(reference)
        }
        result["inactive_parameters"] = sorted(expected_inactive)
        # Diffusers gather backward returns only this CP rank's output slice.
        # FSDP averages gradients over DP x CP. Multiplying by CP=2 restores the
        # mean over distinct logical samples rather than averaging token pieces.
        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
            expected_outputs = []
            for replica in range(2):
                output = reference(
                    "transformer",
                    **model_inputs(replica, step, accelerator.device),
                    return_dict=False,
                )[0][:, :8]
                expected_outputs.append(output.detach())
                (output.square().mean() / 2).backward()
            output = adapter.forward(
                **model_inputs(rank // (2 * tp_size), step, accelerator.device), return_dict=False
            )[0][:, :8]
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
                output = adapter.forward(
                    **model_inputs(rank // (2 * tp_size), step, accelerator.device),
                    return_dict=False,
                )[0][:, :8]
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
        assert len(attention_shapes) > 8 if options["checkpointing"] else len(attention_shapes) == 8
        assert bool(block_shapes) == options["sequence_parallel"]
        result["attention_stream_shapes"] = attention_shapes[0]
        result["attention_forward_calls"] = len(attention_shapes)
        result["sp_block_stream_shapes"] = block_shapes[0] if block_shapes else None
        result["sp_block_forward_calls"] = len(block_shapes)
        result["sp_block_modulate_index"] = modulation_indices[0] if modulation_indices else None
        result["status"] = "pass"
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result, indent=2))
        dist.destroy_process_group()


@pytest.mark.skipif(
    torch.cuda.device_count() < 8, reason="DP2 x CP2 x TP2 requires eight CUDA ranks"
)
@pytest.mark.parametrize("zero_cond_t", [False, True], ids=["2509", "2511"])
@pytest.mark.parametrize("mode", ["full", "lora"])
def test_cuda_qwen_edit_tp_sp_cp_reference(tmp_path, zero_cond_t, mode):
    results = run_gloo_ranks(
        _worker,
        tmp_path,
        8,
        {
            "tp_size": 2,
            "sequence_parallel": True,
            "zero_cond_t": zero_cond_t,
            "checkpointing": True,
            "mode": mode,
        },
    )
    assert all(row.get("status") == "pass" and "error" not in row for row in results), results
    print(json.dumps({"probe": "qwen_edit_tp_sp_cp_fsdp2", "ranks": results}))


def test_cuda_qwen_edit_guard_tp_sp_cp_ordered_full(tmp_path):
    from context_parallel_overlap_runner import qualify_overlap

    qualify_overlap(
        tmp_path,
        {
            "case_module": "qwen_edit_coupled_fixture",
            "model_family": "qwen-image-edit-plus",
            "algorithm": "grpo-guard",
            "training_mode": "full",
            "scheduling_mode": "ordered",
            "dp_size": 2,
            "cp_size": 2,
            "tp_size": 2,
        },
    )
