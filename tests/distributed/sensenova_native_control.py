# Copyright 2026 Jayce-Ping
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Independent native SenseNova prefix, full-gradient and three-step Adam control."""

import copy
import hashlib
import json
import traceback
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from context_parallel_model_control import _assert_fp32
from native_adam_window_oracle import _audit_updates, _capture_update, _records
from test_model_preparation import (
    PreparationAdapter,
    full_value,
    initialize_cuda,
    parameter_map,
    runtime_observation,
)


def _fixed(model):
    return {
        name: hashlib.sha256(
            full_value(p).cpu().contiguous().view(torch.uint8).numpy().tobytes()
        ).hexdigest()
        for name, p in parameter_map(model).items()
        if not p.requires_grad
    }


def _output(model, calls):
    return torch.stack([model(**call) for call in calls])


def _loss(output, replica, step):
    target = torch.randn(
        output.shape, generator=torch.Generator().manual_seed(182 + replica * 20 + step)
    ).to(output.device)
    return (output.float() - target).square().mean()


def _gradients(parameters):
    return {name: full_value(p.grad).cpu() for name, p in parameters.items() if p.grad is not None}


def _assert_output_parity(actual, expected):
    """Compare each FP32 CFG output at its reference scale; leave derivatives unscaled.

    The two-ppm cancellation floor is relative to max(1, reference full scale).
    Retain the original raw maximum-error cap and report the old pointwise rule
    separately. This is an output-only contract, not an end-to-end rounding bound.
    """
    assert actual.dtype == expected.dtype == torch.float32
    assert actual.shape == expected.shape and actual.ndim >= 2 and actual.shape[0] > 0
    assert actual.numel() > 0 and torch.isfinite(actual).all() and torch.isfinite(expected).all()
    metrics = []
    for cfg, (received, reference) in enumerate(zip(actual.detach(), expected.detach())):
        left, right = received.double(), reference.double()
        residual = left - right
        maximum = float(right.abs().max())
        scale = max(1.0, maximum)
        raw_cap = 8e-4 * maximum + 2e-8
        raw_maximum = float(residual.abs().max())
        raw_pointwise = torch.isclose(received, reference, rtol=3e-4, atol=2e-6)
        normalized_received, normalized_reference = received / scale, reference / scale
        torch.testing.assert_close(
            normalized_received, normalized_reference, rtol=3e-4, atol=2e-6, msg=f"CFG{cfg}"
        )
        assert raw_maximum <= raw_cap, (cfg, raw_maximum, raw_cap)
        metrics.append(
            dict(
                contract="sensenova-fp32-output-reference-scale-v1",
                cfg=cfg,
                reference_scale=scale,
                raw_max_abs=raw_maximum,
                raw_max_abs_bound=raw_cap,
                raw_pointwise_outliers=int((~raw_pointwise).sum()),
                raw_original_pass=bool(raw_pointwise.all()) and raw_maximum <= raw_cap,
                normalized_pointwise_pass=True,
                normalized_max_abs=float(
                    (normalized_received.double() - normalized_reference.double()).abs().max()
                ),
                relative_l2=float(residual.norm() / right.norm().clamp_min(1e-12)),
            )
        )
    return metrics


def _assert_inactive_state(parameters, optimizer, records, inactive):
    """Materialize inactive DTensors in the same canonical order on every rank."""
    for name in sorted(inactive):
        torch.testing.assert_close(
            full_value(parameters[name]).cpu(),
            records["initial_parameters"][name],
            rtol=0,
            atol=0,
        )
        assert not optimizer.state.get(parameters[name])


def worker(rank, store_path, result_dir, options):
    initialize_cuda(rank, store_path, world_size=8)
    result = {
        "rank": rank,
        "execution_scope": "model_probe",
        "execution_capability": "sensenova-tp-sp-ulysses-fp32-prefix-probe-v1",
        **options,
    }
    try:
        from accelerate import FullyShardedDataParallelPlugin
        from accelerate.parallelism_config import ParallelismConfig
        from sensenova_cp_fixture import (
            build_denoiser,
            build_tokenizer,
            conditioned_inputs,
            inactive_parameter_ids,
            install_lora,
            make_input_adapter,
            qualify_native_dependencies,
        )

        from flow_factory.acceleration.attention_backend import AttentionBackendAccelerator
        from flow_factory.hparams.parallelism_args import ParallelismArguments
        from flow_factory.loading.coordinator import ModelLoadCoordinator
        from flow_factory.models.model_bundle import ModelBundle
        from flow_factory.models.sensenova.sensenova import SenseNovaAdapter
        from flow_factory.models.tensor_parallel import peel_peft_wrapper
        from flow_factory.trainers.common.context_parallel import full_output_context_parallel_loss
        from flow_factory.trainers.common.model_preparation import ModelPreparation
        from flow_factory.trainers.loader import Accelerator
        from flow_factory.utils.dist import (
            context_parallel_geometry,
            data_parallel_geometry,
            tensor_parallel_geometry,
        )

        class NativeAdapter(PreparationAdapter):
            context_parallel_capability = SenseNovaAdapter.context_parallel_capability
            configure_context_parallel_module = SenseNovaAdapter.configure_context_parallel_module
            install_context_parallel_module = SenseNovaAdapter.install_context_parallel_module
            validate_context_parallel_module = SenseNovaAdapter.validate_context_parallel_module
            compose_context_parallel_sequence_boundaries = (
                SenseNovaAdapter.compose_context_parallel_sequence_boundaries
            )
            tensor_parallel_plan = SenseNovaAdapter.tensor_parallel_plan
            sequence_parallel_plan = SenseNovaAdapter.sequence_parallel_plan
            sequence_parallel_by_default = False

        result["qualified_native_dependencies"] = qualify_native_dependencies()
        accelerator = Accelerator(
            mixed_precision="no",
            fsdp_plugin=FullyShardedDataParallelPlugin(
                fsdp_version=2,
                auto_wrap_policy="TRANSFORMER_BASED_WRAP",
                transformer_cls_names_to_wrap=["Qwen3DecoderLayer", "NEOVisionModel"],
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
        module = build_denoiser(pixel_head=options["pixel"], native_heads=True).float()
        if options["mode"] == "lora":
            module = install_lora(module).float()
        reference_module = copy.deepcopy(module).to(accelerator.device)
        reference_adapter = PreparationAdapter(reference_module, accelerator, options["mode"])
        reference_adapter.declare_component_variants(("policy",))
        reference = ModelBundle(reference_adapter.component_variant_registry.bundle_members())
        adapter = NativeAdapter(module, accelerator, options["mode"])
        adapter.declare_component_variants(("policy",))
        owner = ModelPreparation(
            ModelLoadCoordinator(adapter, accelerator),
            ParallelismArguments(
                dp_shard_size=2, cp_size=2, tp_size=2, sequence_parallel=options["sp"]
            ),
        )
        owner.parallelize(AttentionBackendAccelerator(backend="native"))
        optimizer = torch.optim.AdamW(
            adapter.component_variant_registry.parameters("policy"),
            lr=1e-3,
            eps=1e-4,
            weight_decay=1e-4,
        )
        raw = peel_peft_wrapper(module)
        seen = []
        observed_query_heads = []
        for block in raw.model.language_model.model.layers:
            block.input_layernorm_mot_gen.register_forward_pre_hook(
                lambda owner, args: seen.append(args[0].shape[1])
            )
            block.self_attn.q_proj_mot_gen.register_forward_hook(
                lambda owner, args, output: observed_query_heads.append(output.shape[-1] // 128)
            )
        model, optimizer = owner.prepare(optimizer)
        owner.install_routes()
        result["cp_preparation_owner"] = "diffusers"
        result["fsdp_mesh"] = list(accelerator.parallelism_config.fsdp_dim_names)
        assert not any(getattr(m, "gradient_checkpointing", False) for m in raw.modules())
        parameters = {name: p for name, p in parameter_map(model).items() if p.requires_grad}
        refparameters = {name: p for name, p in parameter_map(reference).items() if p.requires_grad}
        assert parameters.keys() == refparameters.keys()
        frozen, frozen_ref = _fixed(model), _fixed(reference)
        assert frozen == frozen_ref
        reference_optimizer = torch.optim.AdamW(
            reference_adapter.component_variant_registry.parameters("policy"),
            lr=1e-3,
            eps=1e-4,
            weight_decay=1e-4,
        )
        actual_records, ref_records = _records(parameters), _records(refparameters)
        for name in parameters:
            torch.testing.assert_close(
                actual_records["initial_parameters"][name],
                ref_records["initial_parameters"][name],
                rtol=0,
                atol=0,
            )
        tokenizer = build_tokenizer(Path(result_dir) / f"tokenizer-{rank}")
        actual_input = make_input_adapter(
            module,
            tokenizer,
            accelerator.device,
            accelerator=accelerator,
            transformer=adapter.get_component("transformer"),
        )
        reference_input = make_input_adapter(reference_module, tokenizer, accelerator.device)
        # Reference prefix calls use its own real unsharded root. The prepared
        # policy routes every prefix/vision call through the actual FSDP bundle.
        result["conditioning"] = []
        result["output_parity"] = []
        metrics = []
        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
            outputs = []
            reference_loss = 0.0
            for replica in range(2):
                calls, observed = conditioned_inputs(reference_input, replica, step)
                output = _output(lambda **kwargs: reference("transformer", **kwargs), calls)
                outputs.append(output.detach())
                loss = _loss(output, replica, step) / 2
                loss.backward()
                reference_loss += loss.detach()
            calls, observed = conditioned_inputs(actual_input, rank // 4, step)
            result["conditioning"].append(observed)
            output = _output(adapter.get_component("transformer"), calls)
            assert observed_query_heads and set(observed_query_heads) == {16}
            result["observed_native_heads"] = {
                "query": observed_query_heads[-1],
                "kv": calls[0]["past_key_values"].layers[0].keys.shape[1],
                "width": calls[0]["past_key_values"].layers[0].keys.shape[-1],
            }
            assert (
                result["observed_native_heads"]["kv"] == 4
                and result["observed_native_heads"]["width"] == 128
            )
            result["output_parity"].append(
                dict(step=step, cfg=_assert_output_parity(output.detach(), outputs[rank // 4]))
            )
            expected_gradients = _gradients(refparameters)
            if step == 0:
                accelerator.backward(_loss(output, rank // 4, step))
                unscaled = _gradients(parameters)
                _assert_fp32(
                    unscaled, {name: value / 2 for name, value in expected_gradients.items()}
                )
                with pytest.raises(AssertionError):
                    _assert_fp32(unscaled, expected_gradients)
                result["unscaled_loss_control"] = {
                    "gradient_factor": 0.5,
                    "mismatched_parameters": sum(
                        float(value.norm()) > 0
                        and float((unscaled[name] - value).norm()) > 0.25 * float(value.norm())
                        for name, value in expected_gradients.items()
                    ),
                }
                optimizer.zero_grad(set_to_none=True)
                if options["pixel"]:
                    from diffusers.hooks.context_parallel import EquipartitionSharder

                    from flow_factory.models.sensenova import _context_parallel as cp_owner

                    class DroppedRemoteHeadDerivatives:
                        @staticmethod
                        def apply(value, mesh, dim):
                            return EquipartitionSharder.unshard(value, dim, mesh)

                    correct = cp_owner._GatherTokens
                    try:
                        cp_owner._GatherTokens = DroppedRemoteHeadDerivatives
                        bad_output = _output(adapter.get_component("transformer"), calls)
                        _assert_output_parity(bad_output.detach(), outputs[rank // 4])
                        accelerator.backward(
                            full_output_context_parallel_loss(
                                _loss(bad_output, rank // 4, step), accelerator
                            )
                        )
                        bad_gradients = _gradients(parameters)
                        _assert_fp32(
                            {
                                name: value
                                for name, value in bad_gradients.items()
                                if "fm_head" in name
                            },
                            {
                                name: value
                                for name, value in expected_gradients.items()
                                if "fm_head" in name
                            },
                        )
                        missing = [
                            name
                            for name, value in expected_gradients.items()
                            if "language_model" in name
                            and float((bad_gradients[name] - value).norm())
                            > 0.001 * float(value.norm().clamp_min(1e-12))
                        ]
                        assert missing
                        result["lost_remote_spatial_derivative_negative"] = missing
                    finally:
                        cp_owner._GatherTokens = correct
                    optimizer.zero_grad(set_to_none=True)
                output = _output(adapter.get_component("transformer"), calls)
            loss = _loss(output, rank // 4, step)
            accelerator.backward(full_output_context_parallel_loss(loss, accelerator))
            actual_gradients = _gradients(parameters)
            _assert_fp32(actual_gradients, expected_gradients)
            expected_inactive = {
                name for name, p in parameters.items() if id(p) in inactive_parameter_ids(raw)
            }
            inactive = {name for name, p in parameters.items() if p.grad is None}
            assert (
                inactive
                == expected_inactive
                == {name for name, p in refparameters.items() if p.grad is None}
            )
            result["inactive_parameters"] = sorted(inactive)
            errors = {
                name: float((actual_gradients[name] - value).norm() / value.norm().clamp_min(1e-12))
                for name, value in expected_gradients.items()
            }
            assert max(errors.values()) < 0.003, errors
            for factor in (0.0, 0.5):
                with pytest.raises(AssertionError):
                    _assert_fp32(
                        {name: value * factor for name, value in actual_gradients.items()},
                        expected_gradients,
                    )
            global_loss = loss.detach().clone()
            dist.all_reduce(global_loss)
            global_loss /= 8
            _capture_update(
                parameters,
                optimizer,
                actual_records,
                global_loss,
                clipping_world_size=dist.get_world_size(),
            )
            _capture_update(
                refparameters,
                reference_optimizer,
                ref_records,
                reference_loss,
                clipping_world_size=dist.get_world_size(),
            )
            _assert_fp32(
                actual_records["updates"][-1]["deltas"], ref_records["updates"][-1]["deltas"]
            )
            metrics.append({"step": step, "gradient_relative_l2": errors})
        assert _fixed(model) == frozen and _fixed(reference) == frozen_ref
        result["updates"] = _audit_updates(actual_records, ref_records)
        result["fp32_derivative_metrics"] = metrics
        result["unclipped_global_gradient_norms"] = {
            label: [row["unclipped_global_gradient_norm"] for row in records["updates"]]
            for label, records in (("actual", actual_records), ("reference", ref_records))
        }
        assert set(seen) == ({3} if options["sp"] else {5}), seen
        if options["checkpointing"]:
            assert len(seen) > 24
        result["observed_generation_tokens"] = seen
        result["native_geometry"] = {
            "query_heads": 32,
            "kv_heads": 8,
            "head_dim": 128,
            "blocks": 2,
            "image_tokens": 9,
        }
        _assert_inactive_state(parameters, optimizer, actual_records, inactive)
        result.update(
            status="pass", elementwise_comparison=True, nonzero_gradients=True, adam_steps=3
        )
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result, indent=2))
        dist.destroy_process_group()
