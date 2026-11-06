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

"""Fixed-operand native CP1/CP2 controls for explicit model fixture modules.

Both legs start from identical parameters and logical sample inputs. No rollout,
feedback, or schedule is regenerated here. The complete loop owns their evidence.
"""

import importlib
import json
import traceback
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from conftest import run_gloo_ranks
from test_context_parallel_training import (
    _bf16_tensors,
    _check_adam_trajectory,
    _compare,
    _compare_bf16_updates,
)
from test_model_preparation import (
    PreparationAdapter,
    full_value,
    initialize_cuda,
    parameter_map,
    runtime_observation,
)


def clip_prepared_model(accelerator, model):
    """Preserve Accelerate's prepared-root parameter identity and native order."""
    return accelerator.clip_grad_norm_(model.parameters(), 1.0)


def _worker(rank, store_path, result_dir, options):
    cp_size = options["cp_size"]
    initialize_cuda(rank, store_path, world_size=2 * cp_size)
    result = {"rank": rank, "execution_scope": "model_probe", **options}
    try:
        from accelerate import FullyShardedDataParallelPlugin
        from accelerate.parallelism_config import ParallelismConfig
        from torch.distributed.fsdp import FSDPModule

        from flow_factory.acceleration.attention_backend import AttentionBackendAccelerator
        from flow_factory.hparams.parallelism_args import ParallelismArguments
        from flow_factory.loading.coordinator import ModelLoadCoordinator
        from flow_factory.models.variants import ComponentVariantRegistry
        from flow_factory.trainers.common.context_parallel import full_output_context_parallel_loss
        from flow_factory.trainers.common.model_preparation import ModelPreparation
        from flow_factory.trainers.loader import Accelerator
        from flow_factory.utils.dist import context_parallel_geometry, data_parallel_geometry

        fixture = importlib.import_module(options["fixture_module"])
        adapter_class = fixture.adapter_class()

        class NativePreparationAdapter(PreparationAdapter):
            context_parallel_capability = adapter_class.context_parallel_capability
            validate_context_parallel_module = adapter_class.validate_context_parallel_module
            tensor_parallel_plan = adapter_class.tensor_parallel_plan
            sequence_parallel_plan = adapter_class.sequence_parallel_plan
            sequence_parallel_by_default = False

        accelerator = Accelerator(
            mixed_precision=options["precision"],
            fsdp_plugin=FullyShardedDataParallelPlugin(
                fsdp_version=2,
                auto_wrap_policy="TRANSFORMER_BASED_WRAP",
                transformer_cls_names_to_wrap=list(fixture.wrap_classes),
                cpu_ram_efficient_loading=False,
                reshard_after_forward=True,
                activation_checkpointing=True,
            ),
            parallelism_config=ParallelismConfig(
                dp_replicate_size=1, dp_shard_size=2, cp_size=cp_size, tp_size=1, sp_size=1
            ),
        )
        result["runtime"] = runtime_observation(accelerator)
        assert data_parallel_geometry(accelerator) == (2, rank // cp_size)
        assert context_parallel_geometry(accelerator) == (cp_size, rank % cp_size)
        module = fixture.build_transformer().float()
        if options["training_mode"] == "lora":
            module = fixture.install_lora(module).float()
        module.set_attention_backend("native")
        adapter = NativePreparationAdapter(module, accelerator, options["training_mode"])
        adapter.declare_component_variants(("policy",))
        owner = ModelPreparation(
            ModelLoadCoordinator(adapter, accelerator),
            ParallelismArguments(dp_shard_size=2, cp_size=cp_size, sequence_parallel=False),
        )
        owner.parallelize(AttentionBackendAccelerator(backend="native"))
        optimizer = torch.optim.AdamW(
            adapter.component_variant_registry.parameters("policy"),
            lr=1e-3,
            eps=1e-4,
            weight_decay=1e-4,
        )
        model, optimizer = owner.prepare(optimizer)
        owner.install_routes()
        expected_wraps = set(fixture.wrap_classes)
        wrapped = {
            cls.__name__
            for child in model.modules()
            if isinstance(child, FSDPModule)
            for cls in type(child).__mro__
            if cls.__name__ in expected_wraps
        }
        assert wrapped == expected_wraps
        result["wrapped_blocks"] = sorted(wrapped)
        if cp_size > 1:
            contract = accelerator.unwrap_model(model).context_parallel_contract
            assert contract.capability == adapter_class.context_parallel_capability
            assert contract.boundaries == (), "CP-only uses native model split/gather hooks"
        all_parameters = parameter_map(model)
        fixed_parameters = {name: full_value(p).cpu() for name, p in all_parameters.items()}
        parameters = {name: p for name, p in all_parameters.items() if p.requires_grad}
        assert parameters
        inactive = (
            frozenset(
                ComponentVariantRegistry._canonical_prepared_parameter_name(name)
                for name in fixture.inactive_parameter_names(model)
            )
            if hasattr(fixture, "inactive_parameter_names")
            else frozenset()
        )
        assert inactive <= parameters.keys()
        initial = {name: full_value(p).cpu() for name, p in parameters.items()}
        inputs = fixture.model_inputs(rank // cp_size, 0, accelerator.device)
        observations = {
            "initial_parameters": initial,
            "fixed_parameters": fixed_parameters,
            "inputs": fixture.serialize_inputs(inputs),
            "updates": [],
            "inactive_parameters": sorted(inactive),
        }
        if cp_size > 1:
            with accelerator.autocast():
                output = adapter.forward(**inputs, return_dict=False)[0]
                loss = output.float().square().mean()
            accelerator.backward(loss)
            observations["unscaled_gradients"] = {
                name: full_value(p.grad).cpu()
                for name, p in parameters.items()
                if p.grad is not None
            }
            assert observations["unscaled_gradients"].keys() == parameters.keys() - inactive
            optimizer.zero_grad(set_to_none=True)
        with accelerator.autocast():
            output = adapter.forward(**inputs, return_dict=False)[0]
            loss = output.float().square().mean()
        observations["output"] = output.detach().cpu()
        accelerator.backward(full_output_context_parallel_loss(loss, accelerator))
        assert {name for name, p in parameters.items() if p.grad is None} == inactive
        gradients = {
            name: full_value(p.grad).cpu() for name, p in parameters.items() if name not in inactive
        }
        assert any(torch.count_nonzero(value) for value in gradients.values())
        clip_prepared_model(accelerator, model)
        clipped = {
            name: full_value(p.grad).cpu() for name, p in parameters.items() if name not in inactive
        }
        settings = {
            key: optimizer.param_groups[0][key] for key in ("lr", "betas", "eps", "weight_decay")
        }
        settings["max_grad_norm"] = 1.0
        optimizer.step()
        following = {name: full_value(p).cpu() for name, p in parameters.items()}
        moments = {
            name: {key: full_value(value).cpu() for key, value in optimizer.state[p].items()}
            for name, p in parameters.items()
            if name not in inactive
        }
        update = {
            "before": initial,
            "gradients": gradients,
            "clipped_gradients": clipped,
            "parameters": following,
            "deltas": {
                name: following[name] - initial[name] for name in parameters if name not in inactive
            },
            "moments": moments,
            "optimizer": settings,
            "loss": loss.detach().cpu(),
        }
        assert any(torch.count_nonzero(value) for value in update["deltas"].values())
        observations["updates"].append(update)
        for name, parameter in all_parameters.items():
            if not parameter.requires_grad or name in inactive:
                assert parameter.grad is None
                assert not optimizer.state.get(parameter, {})
                torch.testing.assert_close(
                    full_value(parameter).cpu(), fixed_parameters[name], rtol=0, atol=0
                )
        torch.save(observations, Path(result_dir, f"observations-{rank}.pt"))
        result.update(status="pass", elementwise_comparison=True, nonzero_gradients=True)
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result, indent=2))
        dist.destroy_process_group()


def _assert_fp32(actual, expected):
    assert actual.keys() == expected.keys()
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=3e-4, atol=2e-6, msg=name)
        assert (actual[name] - expected[name]).abs().max() <= 8e-4 * expected[
            name
        ].abs().max() + 2e-8, name


def compare_model_controls(actual, expected, precision):
    """Apply frozen precision gates and independent clipping/Adam equations."""
    _compare(actual["initial_parameters"], expected["initial_parameters"], exact=True)
    _compare(actual["inputs"], expected["inputs"], exact=True)
    _compare(actual["fixed_parameters"], expected["fixed_parameters"], exact=True)
    assert actual.get("inactive_parameters", []) == expected.get("inactive_parameters", [])
    _check_adam_trajectory(actual)
    _check_adam_trajectory(expected)
    left, right = actual["updates"][0], expected["updates"][0]
    if precision == "bf16":
        # Shared fixed-state precision gates: 2/128 derivative norm, 4/128 update
        # norm, per-element 0.02 relative + 1e-3 absolute cancellation floor.
        # Adam moment/update budgets propagate admitted gradient errors through
        # independently checked recurrences; no model-specific exception applies.
        _bf16_tensors({"output": actual["output"]}, {"output": expected["output"]}, atol=1e-3)
        metrics = _compare_bf16_updates(actual, expected)
        for factor in (0.0, 0.5, 2.0, -1.0):
            with pytest.raises(AssertionError):
                _bf16_tensors(
                    {name: value * factor for name, value in left["gradients"].items()},
                    right["gradients"],
                    atol=1e-3,
                )
        _bf16_tensors(
            actual["unscaled_gradients"],
            {name: value / 2 for name, value in right["gradients"].items()},
            atol=1e-3,
        )
        with pytest.raises(AssertionError):
            _bf16_tensors(actual["unscaled_gradients"], right["gradients"], atol=1e-3)
    else:
        _assert_fp32({"output": actual["output"]}, {"output": expected["output"]})
        _assert_fp32(left["gradients"], right["gradients"])
        _assert_fp32(left["deltas"], right["deltas"])
        _assert_fp32(
            actual["unscaled_gradients"],
            {name: value / 2 for name, value in right["gradients"].items()},
        )
        for factor in (0.0, 0.5, 2.0, -1.0):
            with pytest.raises(AssertionError):
                _assert_fp32(
                    {name: value * factor for name, value in left["gradients"].items()},
                    right["gradients"],
                )
        with pytest.raises(AssertionError):
            _assert_fp32(actual["unscaled_gradients"], right["gradients"])
        metrics = [{"elementwise_comparison": True}]
    return metrics


def qualify_model_control(tmp_path, root, options):
    """Run fixed-operand controls with an explicit model recipe and common numerical gates."""
    del root
    assert options["tp_size"] == 1 and options["cp_size"] == options["dp_size"] == 2
    reports = []
    for precision in ("no", "bf16"):
        legs = {}
        for cp_size in (1, 2):
            directory = Path(tmp_path) / f"model-{options['training_mode']}-{precision}-cp{cp_size}"
            rows = run_gloo_ranks(
                _worker,
                directory,
                2 * cp_size,
                {
                    "fixture_module": options["fixture_module"],
                    "training_mode": options["training_mode"],
                    "precision": precision,
                    "cp_size": cp_size,
                },
                timeout=600,
            )
            assert all(row.get("status") == "pass" and "error" not in row for row in rows), rows
            legs[cp_size] = [
                torch.load(directory / f"observations-{rank}.pt", weights_only=False)
                for rank in range(2 * cp_size)
            ]
        for rank, actual in enumerate(legs[2]):
            metrics = compare_model_controls(actual, legs[1][rank // 2], precision)
            reports.append({"precision": precision, "rank": rank, "metrics": metrics})
    return reports
