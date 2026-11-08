# Copyright 2026 Jayce-Ping
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Native Bagel TP/SP/CP with same-state arithmetic and own-trajectory Adam proofs."""

import copy
import json
import traceback
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from bagel_arithmetic_reference import ArithmeticReference
from bagel_own_adam_reference import own_adam, same_state_gradients, synchronize_shadow
from conftest import run_gloo_ranks
from native_adam_window_oracle import _audit_updates, _capture_update, _records
from test_model_preparation import (
    PreparationAdapter,
    full_value,
    initialize_cuda,
    parameter_map,
    runtime_observation,
)


def _loss(output, replica, step):
    target = torch.randn(
        output.shape, generator=torch.Generator().manual_seed(122 + 17 * replica + step)
    ).to(output.device)
    return (output.float() - target).square().mean()


def _frozen_digests(model):
    import hashlib

    return {
        name: hashlib.sha256(
            full_value(parameter).cpu().contiguous().view(torch.uint8).numpy().tobytes()
        ).hexdigest()
        for name, parameter in parameter_map(model).items()
        if not parameter.requires_grad
    }


def _native_ragged_control(model, input_adapter, replica):
    """Check native ragged decoding without changing its batch-sensitive pack.

    This unsharded smoke control preserves the original VAE/ViT operands and
    three CFG packs. Numerical parallel/reference comparisons use the same
    fixed-count B2 packs below; the separate FP64 attention oracle isolates
    projected-operand attention from whole-model packing effects.
    """
    from bagel_cp_fixture import forward_calls, model_call_inputs

    assert input_adapter.requires_preserved_replay_batch_composition is True
    model.zero_grad(set_to_none=True)
    calls, observation = model_call_inputs(input_adapter, replica, 0, ragged=True)
    assert len(calls) == 3 and observation["reference_counts"] == [1, 2]
    operands = [
        {name: value.clone() for name, value in call.items() if torch.is_tensor(value)}
        for call in calls
    ]
    caches = []
    for call in calls:
        assert call["query_lens"].tolist() == observation["query_lengths"]
        assert call["update_past_key_values"] is False
        cache = call["past_key_values"]
        caches.append(
            {
                (kind, layer): (value, value._version, value.clone())
                for kind in ("key_cache", "value_cache")
                for layer, value in getattr(cache, kind).items()
            }
        )
    packed = forward_calls(lambda **kwargs: model("transformer", **kwargs), calls)
    assert packed.shape[:2] == (3, sum(observation["query_lengths"]))
    assert torch.isfinite(packed).all() and torch.count_nonzero(packed)
    assert not torch.equal(packed[0], packed[1]) and not torch.equal(packed[0], packed[2])
    target = torch.randn(packed.shape, generator=torch.Generator().manual_seed(557)).to(
        packed.device
    )
    (packed.float() - target).square().mean().backward()
    gradients = [
        parameter.grad for parameter in parameter_map(model).values() if parameter.grad is not None
    ]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient) for gradient in gradients)
    for call, original_operands, original_cache in zip(calls, operands, caches):
        for name, expected in original_operands.items():
            torch.testing.assert_close(call[name], expected, rtol=0, atol=0)
        for (kind, layer), (original, version, expected) in original_cache.items():
            value = getattr(call["past_key_values"], kind)[layer]
            assert value is original and value._version == version and not value.requires_grad
            torch.testing.assert_close(value, expected, rtol=0, atol=0)
    model.zero_grad(set_to_none=True)
    return {
        **observation,
        "active_gradients": len(gradients),
        "finite_output": True,
        "nonzero_gradients": True,
        "requires_preserved_replay_batch_composition": True,
        "original_pack_forward_calls": len(calls),
        "original_pack_preserved": True,
        "original_pack_prefix_operands": True,
        "cache_unchanged": True,
        "parameter_sharding": False,
    }


def synchronized_check(function, label, result_dir, rank, operands):
    outcome, error = None, None
    try:
        outcome = function()
    except Exception:
        error = traceback.format_exc()
        torch.save(
            operands, Path(result_dir) / f"failure-{label.replace(chr(32), chr(45))}-rank-{rank}.pt"
        )
    errors = [None] * dist.get_world_size()
    dist.all_gather_object(errors, error)
    assert not any(errors), (label, errors)
    return outcome


def _worker(rank, store_path, result_dir, options):
    initialize_cuda(rank, store_path, world_size=8)
    result = {
        "rank": rank,
        "execution_scope": "model_probe",
        "execution_capability": "bagel-packed-prefix-tp-ulysses-bf16-probe-v1",
        "reference_contract": "bagel-canonical-arithmetic-global-signal-v1",
        "comparison": "same-state independent arithmetic shadow; actual autonomous 3-step Adam",
        **options,
    }
    try:
        from accelerate import FullyShardedDataParallelPlugin
        from accelerate.parallelism_config import ParallelismConfig
        from bagel_cp_fixture import (
            build_pipeline,
            build_tokenizer,
            forward_calls,
            install_lora,
            make_input_adapter,
            model_call_inputs,
            qualify_native_dependencies,
        )

        from flow_factory.acceleration.attention_backend import AttentionBackendAccelerator
        from flow_factory.hparams.parallelism_args import ParallelismArguments
        from flow_factory.loading.coordinator import ModelLoadCoordinator
        from flow_factory.models.bagel.bagel import BagelAdapter
        from flow_factory.models.model_bundle import ModelBundle
        from flow_factory.models.tensor_parallel import peel_peft_wrapper
        from flow_factory.trainers.common.model_preparation import ModelPreparation
        from flow_factory.trainers.loader import Accelerator
        from flow_factory.utils.dist import (
            context_parallel_geometry,
            data_parallel_geometry,
            tensor_parallel_geometry,
        )

        class BagelPreparationAdapter(PreparationAdapter):
            context_parallel_capability = BagelAdapter.context_parallel_capability
            context_parallel_diffusers_revision = BagelAdapter.context_parallel_diffusers_revision
            configure_context_parallel_module = BagelAdapter.configure_context_parallel_module
            install_context_parallel_module = BagelAdapter.install_context_parallel_module
            validate_context_parallel_module = BagelAdapter.validate_context_parallel_module
            compose_context_parallel_sequence_boundaries = (
                BagelAdapter.compose_context_parallel_sequence_boundaries
            )
            tensor_parallel_plan = BagelAdapter.tensor_parallel_plan
            sequence_parallel_plan = BagelAdapter.sequence_parallel_plan
            sequence_parallel_by_default = False

        result["qualified_native_dependencies"] = qualify_native_dependencies()
        accelerator = Accelerator(
            mixed_precision="no",
            fsdp_plugin=FullyShardedDataParallelPlugin(
                fsdp_version=2,
                auto_wrap_policy="TRANSFORMER_BASED_WRAP",
                transformer_cls_names_to_wrap=["Qwen2MoTDecoderLayer"],
                cpu_ram_efficient_loading=False,
                reshard_after_forward=True,
                activation_checkpointing=options["checkpointing"],
            ),
            parallelism_config=ParallelismConfig(
                dp_replicate_size=1, dp_shard_size=2, cp_size=2, tp_size=2, sp_size=1
            ),
        )
        assert data_parallel_geometry(accelerator) == (2, rank // 4)
        assert context_parallel_geometry(accelerator) == (2, (rank // 2) % 2)
        assert tensor_parallel_geometry(accelerator) == (2, rank % 2)
        result["runtime"] = {
            **runtime_observation(accelerator),
            "cp_size": context_parallel_geometry(accelerator)[0],
            "tp_size": tensor_parallel_geometry(accelerator)[0],
        }
        pipeline = build_pipeline()
        if options["mode"] == "lora":
            pipeline.bagel.language_model = install_lora(pipeline.bagel.language_model)
            pipeline.transformer = pipeline.bagel.language_model
        reference_pipeline = copy.deepcopy(pipeline)
        reference_pipeline.bagel.to(accelerator.device)
        reference_pipeline.vae.to(accelerator.device)
        # Auxiliary components are frozen and outside the language-model FSDP root.
        for name, child in pipeline.bagel.named_children():
            if name != "language_model":
                child.to(accelerator.device)
        pipeline.vae.to(accelerator.device)
        module = pipeline.transformer
        reference_module = reference_pipeline.transformer
        reference_adapter = PreparationAdapter(reference_module, accelerator, options["mode"])
        reference_adapter.declare_component_variants(("policy",))
        reference = ModelBundle(reference_adapter.component_variant_registry.bundle_members())
        adapter = BagelPreparationAdapter(module, accelerator, options["mode"])
        adapter.declare_component_variants(("policy",))
        owner = ModelPreparation(
            ModelLoadCoordinator(adapter, accelerator),
            ParallelismArguments(
                dp_shard_size=2,
                cp_size=2,
                tp_size=2,
                sequence_parallel=options["sequence_parallel"],
            ),
        )
        owner.parallelize(AttentionBackendAccelerator(backend="native"))
        optimizer = torch.optim.AdamW(
            adapter.component_variant_registry.parameters("policy"),
            lr=1e-3,
            eps=1e-4,
            weight_decay=1e-4,
        )
        from collections import Counter

        stream_shapes = Counter()
        raw = peel_peft_wrapper(module)
        result["observed_attention_heads"] = [
            dict(
                query=block.self_attn.num_heads,
                kv=block.self_attn.num_key_value_heads,
                width=block.self_attn.head_dim,
            )
            for block in raw.model.layers
        ]
        expected_heads = (14, 2) if options["sequence_parallel"] else (28, 4)
        assert all(
            (row["query"], row["kv"], row["width"]) == (*expected_heads, 128)
            for row in result["observed_attention_heads"]
        )

        def observe_stream(owner, args, kwargs):
            value = kwargs.get("packed_query_sequence", args[0] if args else None)
            stream_shapes[
                (
                    kwargs.get("mode", "und"),
                    kwargs.get("update_past_key_values", True),
                    value.shape[0],
                    int(kwargs["query_lens"].sum()),
                )
            ] += 1

        for block in raw.model.layers:
            block.register_forward_pre_hook(observe_stream, with_kwargs=True)
        model, optimizer = owner.prepare(optimizer)
        owner.install_routes()
        result["cp_preparation_owner"] = "diffusers"
        result["fsdp_mesh"] = list(accelerator.parallelism_config.fsdp_dim_names)
        fixed_actual, fixed_reference = _frozen_digests(model), _frozen_digests(reference)
        assert fixed_actual == fixed_reference

        tokenizer = build_tokenizer(Path(result_dir) / f"tokenizer-{rank}")
        actual_inputs = make_input_adapter(
            pipeline,
            tokenizer,
            accelerator.device,
            transformer=adapter.get_component("transformer"),
        )
        expected_inputs = make_input_adapter(
            reference_pipeline,
            tokenizer,
            accelerator.device,
            transformer=lambda **kwargs: reference("transformer", **kwargs),
        )
        result["ragged_native_control"] = _native_ragged_control(
            reference, expected_inputs, rank // 4
        )
        independent = ArithmeticReference(
            peel_peft_wrapper(reference_module),
            sequence_parallel=options["sequence_parallel"],
        ).install()
        parameters = {
            name: parameter
            for name, parameter in parameter_map(model).items()
            if parameter.requires_grad
        }
        refparameters = {
            name: parameter
            for name, parameter in parameter_map(reference).items()
            if parameter.requires_grad
        }
        assert parameters.keys() == refparameters.keys()
        actual_records, initial_shadow = _records(parameters), _records(refparameters)
        for name in parameters:
            torch.testing.assert_close(
                actual_records["initial_parameters"][name],
                initial_shadow["initial_parameters"][name],
                rtol=0,
                atol=0,
            )
        result["prefix_observations"] = []
        result["native_geometry"] = {"query_heads": 28, "kv_heads": 4, "head_dim": 128, "blocks": 2}
        result["checkpoint_recomputation"] = options["checkpointing"]
        result["same_state_steps"] = []
        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            reference.zero_grad(set_to_none=True)
            synchronization = synchronize_shadow(parameters, refparameters, optimizer)
            expected_outputs = []
            reference_loss = 0.0
            for replica in range(2):
                calls, observation = model_call_inputs(expected_inputs, replica, step)
                output = forward_calls(lambda **kwargs: reference("transformer", **kwargs), calls)
                expected_outputs.append(output.detach())
                loss = _loss(output, replica, step) / 2
                loss.backward()
                reference_loss += loss.detach()
            calls, observation = model_call_inputs(actual_inputs, rank // 4, step)
            result["prefix_observations"].append(observation)
            output = forward_calls(adapter.get_component("transformer"), calls)
            from test_context_parallel_training import _bf16_tensors

            output_residual = synchronized_check(
                lambda: _bf16_tensors(
                    {"output": output.detach()}, {"output": expected_outputs[rank // 4]}, atol=1e-3
                ),
                "same-state output",
                result_dir,
                rank,
                {"actual": output.detach(), "shadow": expected_outputs[rank // 4]},
            )
            if step == 0:
                accelerator.backward(_loss(output, rank // 4, step))
                unscaled = {
                    name: full_value(p.grad).cpu()
                    for name, p in parameters.items()
                    if p.grad is not None
                }
                reference_gradients = {
                    name: p.grad.detach().cpu()
                    for name, p in refparameters.items()
                    if p.grad is not None
                }
                _bf16_tensors(
                    unscaled,
                    {name: value / 2 for name, value in reference_gradients.items()},
                    atol=1e-3,
                )
                with pytest.raises(AssertionError):
                    _bf16_tensors(unscaled, reference_gradients, atol=1e-3)
                result["unscaled_loss_control"] = {
                    "gradient_factor": 0.5,
                    "mismatched_parameters": sum(
                        bool(value.norm() > 0)
                        and bool((unscaled[name] - value).norm() > 0.25 * value.norm())
                        for name, value in reference_gradients.items()
                    ),
                }
                optimizer.zero_grad(set_to_none=True)
                output = forward_calls(adapter.get_component("transformer"), calls)
            loss = _loss(output, rank // 4, step)
            accelerator.backward(loss * 2)
            inactive = {name for name, p in parameters.items() if p.grad is None}
            expected_inactive = {name for name, p in refparameters.items() if p.grad is None}
            assert inactive == expected_inactive
            assert all(name.endswith("lm_head.weight") for name in inactive), inactive
            result["inactive_parameters"] = sorted(inactive)
            global_loss = loss.detach().clone()
            dist.all_reduce(global_loss)
            global_loss /= 8
            actual_gradients = {
                name: full_value(p.grad).cpu()
                for name, p in parameters.items()
                if p.grad is not None
            }
            shadow_gradients = {
                name: p.grad.detach().cpu()
                for name, p in refparameters.items()
                if p.grad is not None
            }
            gradient_rows = synchronized_check(
                lambda: same_state_gradients(actual_gradients, shadow_gradients),
                "same-state gradients",
                result_dir,
                rank,
                {"actual": actual_gradients, "shadow": shadow_gradients},
            )
            result["same_state_steps"].append(
                {
                    "step": step + 1,
                    "output_relative_l2": output_residual,
                    "synchronization": synchronization,
                    "gradients": gradient_rows["parameters"],
                    "global_gradient_relative_l2": gradient_rows["global_relative_l2"],
                    "gradient_relative_l2_scope": gradient_rows["relative_l2_scope"],
                    "gradient_region_statistics": "diagnostic_only",
                    "independent_prefix": True,
                    "shadow_has_optimizer": False,
                }
            )
            _capture_update(
                parameters,
                optimizer,
                actual_records,
                global_loss,
                clipping_world_size=dist.get_world_size(),
            )
            assert independent.context.get() is None
        assert _frozen_digests(model) == fixed_actual
        assert _frozen_digests(reference) == fixed_reference
        result["frozen_parameter_count"] = len(fixed_actual)
        result["observed_streams"] = [
            dict(
                mode=mode,
                cache_update=update,
                local_tokens=tokens,
                original_tokens=original,
                calls=count,
            )
            for (mode, update, tokens, original), count in sorted(stream_shapes.items())
        ]
        generation = [
            row
            for row in result["observed_streams"]
            if row["mode"] == "gen" and not row["cache_update"]
        ]
        assert generation and {row["local_tokens"] for row in generation} == (
            {10} if options["sequence_parallel"] else {18}
        )
        assert all(row["original_tokens"] == 36 for row in generation)
        if options["checkpointing"]:
            assert sum(row["calls"] for row in generation) > 24
        result["updates"] = own_adam(actual_records, world_size=8)
        assert result["updates"]
        result["unclipped_global_gradient_norms"] = {
            name: [row["unclipped_global_gradient_norm"] for row in records["updates"]]
            for name, records in (("actual", actual_records),)
        }
        for name in result["inactive_parameters"]:
            torch.testing.assert_close(
                actual_records["initial_parameters"][name],
                full_value(parameters[name]).cpu(),
                rtol=0,
                atol=0,
            )
            assert not optimizer.state.get(parameters[name])
        assert peel_peft_wrapper(module)._ff_bagel_parallel_state.call.get() is None
        result.update(
            status="pass",
            native_flash_prefill=True,
            elementwise_comparison=True,
            adam_steps=3,
            same_state_reference=True,
            actual_history_never_reset=True,
            reference_observations=independent.observations,
        )
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result, indent=2))
        dist.destroy_process_group()


@pytest.mark.skipif(
    torch.cuda.device_count() < 8, reason="Bagel DP2×CP2×TP2 requires eight CUDA ranks"
)
@pytest.mark.parametrize("sequence_parallel", [False, True])
@pytest.mark.parametrize("mode,checkpointing", [("full", False), ("lora", True)])
def test_cuda_bagel_tp_sp_cp_native_reference(tmp_path, sequence_parallel, mode, checkpointing):
    rows = run_gloo_ranks(
        _worker,
        tmp_path,
        8,
        dict(mode=mode, checkpointing=checkpointing, sequence_parallel=sequence_parallel),
        timeout=1800,
    )
    assert all(row.get("status") == "pass" and "error" not in row for row in rows), rows
    print(json.dumps({"probe": "bagel_native_packed_tp_sp_cp", "ranks": rows}))


def test_bagel_adam_windows_keep_three_step_recurrence_and_reject_corruption():
    parameter = torch.nn.Parameter(torch.linspace(-0.1, 0.1, 262145))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3, eps=1e-4, weight_decay=1e-4)
    records = _records({"p": parameter})
    for step in range(3):
        parameter.grad = torch.linspace(-0.2, 0.3, 262145) * (1 - 0.3 * step)
        _capture_update(
            {"p": parameter}, optimizer, records, torch.tensor(1.0), clipping_world_size=1
        )
    assert _audit_updates(records, copy.deepcopy(records))["p"]["adam_steps"] == 3
    corrupt = copy.deepcopy(records)
    corrupt["updates"][1]["moments"]["p"]["exp_avg"][-1] += 0.01
    with pytest.raises(AssertionError):
        _audit_updates(corrupt, records)


def test_cuda_bagel_guard_ordered_lora(tmp_path):
    from context_parallel_overlap_runner import qualify_overlap

    qualify_overlap(
        tmp_path,
        dict(
            case_module="bagel_coupled_fixture",
            model_family="bagel",
            algorithm="grpo-guard",
            training_mode="lora",
            scheduling_mode="ordered",
            dp_size=2,
            tp_size=2,
            cp_size=2,
            sequence_parallel=True,
        ),
    )
