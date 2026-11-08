# Copyright 2026 Jayce-Ping
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""SenseNova native prefix, CP/SP and trainable spatial-head derivative controls."""

import copy
import json
import traceback
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from conftest import requires_gloo, run_gloo_ranks


def _cpu_worker(rank, store_path, result_dir, options):
    result = {"rank": rank, **options}
    dist.init_process_group("gloo", init_method=f"file://{store_path}", rank=rank, world_size=4)
    try:
        from diffusers.models._modeling_parallel import ContextParallelConfig
        from sensenova_cp_fixture import build_denoiser, install_lora, prefix_and_inputs
        from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
        from torch.distributed.tensor import DTensor

        from flow_factory.models.sensenova.sensenova import SenseNovaAdapter
        from flow_factory.models.tensor_parallel import (
            apply_tensor_parallel,
            peel_peft_wrapper,
            resolve_parallel_plan,
        )

        module = build_denoiser(pixel_head=options["pixel"]).double()
        if options["mode"] == "lora":
            module = install_lora(module)
        reference = copy.deepcopy(module)
        whole = init_device_mesh("cpu", (2, 2), mesh_dim_names=("cp", "tp"))
        raw = peel_peft_wrapper(module)
        apply_tensor_parallel(
            raw,
            whole["tp"],
            resolve_parallel_plan(SenseNovaAdapter, sequence_parallel=options["sp"]),
        )
        SenseNovaAdapter.validate_context_parallel_module(raw, 2, 2)
        SenseNovaAdapter.configure_context_parallel_module(raw)
        raw.set_attention_backend("native")
        mesh = DeviceMesh(
            "cpu", whole.mesh.T.unsqueeze(1), mesh_dim_names=("tensor", "ring", "ulysses")
        )
        SenseNovaAdapter.install_context_parallel_module(
            raw, ContextParallelConfig(ring_degree=1, ulysses_degree=2, mesh=mesh)
        )
        counts = []
        head_inputs = []
        raw.model.fm_modules["fm_head"].register_forward_pre_hook(
            lambda owner, args: head_inputs.append(tuple(args[0].shape))
        )
        for block in raw.model.language_model.model.layers:
            block.input_layernorm_mot_gen.register_forward_pre_hook(
                lambda owner, args: counts.append(args[0].shape[1])
            )
        result["passes"] = []
        for checkpoint in (False, True):
            if checkpoint:
                raw.enable_gradient_checkpointing()
                peel_peft_wrapper(reference).enable_gradient_checkpointing()
            module.zero_grad(set_to_none=True)
            reference.zero_grad(set_to_none=True)
            inputs, prefix = prefix_and_inputs(module)
            refinputs, refprefix = prefix_and_inputs(reference)
            torch.testing.assert_close(prefix, refprefix, rtol=1e-6, atol=1e-7)
            original = [
                (layer.keys.clone(), layer.values.clone())
                for layer in inputs["past_key_values"].layers
            ]
            actual = module(**inputs)
            expected = reference(**refinputs)
            torch.testing.assert_close(actual, expected, rtol=3e-5, atol=1e-7)
            target = torch.randn(
                expected.shape, generator=torch.Generator().manual_seed(189)
            ).double()
            (actual - target).square().mean().backward()
            (expected - target).square().mean().backward()
            active = {}
            inactive = []
            for (name, p), (other, q) in zip(
                module.named_parameters(), reference.named_parameters()
            ):
                assert name == other and (p.grad is None) == (q.grad is None), name
                if p.grad is None:
                    if p.requires_grad:
                        inactive.append(name)
                    continue
                gradient = p.grad.full_tensor() if isinstance(p.grad, DTensor) else p.grad
                gradient = gradient.detach().clone()
                dist.all_reduce(gradient, group=whole["cp"].get_group())
                torch.testing.assert_close(gradient, q.grad, rtol=3e-4, atol=1e-7, msg=name)
                active[name] = float((gradient - q.grad).norm() / q.grad.norm().clamp_min(1e-15))
            for layer, (key, value) in zip(inputs["past_key_values"].layers, original):
                torch.testing.assert_close(layer.keys, key, rtol=0, atol=0)
                torch.testing.assert_close(layer.values, value, rtol=0, atol=0)
            result["passes"].append(dict(checkpoint=checkpoint, active=active, inactive=inactive))
        assert counts and set(counts) == ({3} if options["sp"] else {5}), counts
        assert len(counts) > 4, "native checkpoint backward did not re-enter decoder blocks"
        if not options["pixel"]:
            assert head_inputs and {shape[1] for shape in head_inputs} == (
                {3} if options["sp"] else {9}
            ), head_inputs
        if options["pixel"]:
            from diffusers.hooks.context_parallel import EquipartitionSharder

            from flow_factory.models.sensenova import _context_parallel as owner

            class DroppedRemoteHeadDerivatives:
                @staticmethod
                def apply(value, mesh, dim):
                    return EquipartitionSharder.unshard(value, dim, mesh)

            correct = owner._GatherTokens
            try:
                owner._GatherTokens = DroppedRemoteHeadDerivatives
                module.zero_grad(set_to_none=True)
                inputs, _ = prefix_and_inputs(module)
                bad = module(**inputs)
                torch.testing.assert_close(bad, expected, rtol=3e-5, atol=1e-7)
                (bad - target).square().mean().backward()
                mismatches = []
                for (name, p), (_, q) in zip(
                    module.named_parameters(), reference.named_parameters()
                ):
                    if q.grad is None:
                        continue
                    gradient = p.grad.full_tensor() if isinstance(p.grad, DTensor) else p.grad
                    gradient = gradient.detach().clone()
                    dist.all_reduce(gradient, group=whole["cp"].get_group())
                    error = float((gradient - q.grad).norm() / q.grad.norm().clamp_min(1e-15))
                    if "fm_head" in name:
                        torch.testing.assert_close(gradient, q.grad, rtol=3e-4, atol=1e-7, msg=name)
                    elif error > 1e-3:
                        mismatches.append(name)
                assert any("language_model" in name for name in mismatches), mismatches
                result["lost_remote_spatial_derivative_negative"] = mismatches
            finally:
                owner._GatherTokens = correct
        raw.disable_gradient_checkpointing()
        assert not any(getattr(m, "gradient_checkpointing", False) for m in raw.modules())
        result.update(status="pass", local_generation_tokens=counts, head_input_shapes=head_inputs)
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result, indent=2))
        dist.destroy_process_group()


@requires_gloo
@pytest.mark.parametrize("pixel", [False, True], ids=["u10", "u15"])
@pytest.mark.parametrize("mode", ["full", "lora"])
@pytest.mark.parametrize("sp", [False, True])
def test_sensenova_cp_sp_native_derivatives(tmp_path, pixel, mode, sp):
    results = run_gloo_ranks(_cpu_worker, tmp_path, 4, dict(pixel=pixel, mode=mode, sp=sp))
    assert all(row.get("status") == "pass" and "error" not in row for row in results), results


@pytest.mark.parametrize("pixel", [False, True], ids=["u10", "u15"])
def test_sensenova_real_ordered_reference_prefix_enters_component(tmp_path, pixel):
    from sensenova_cp_fixture import (
        build_denoiser,
        build_tokenizer,
        conditioned_inputs,
        make_input_adapter,
    )

    from flow_factory.models.sensenova.pipeline import SenseNovaImagePrefix

    module = build_denoiser(pixel_head=pixel)
    module.enable_gradient_checkpointing()
    calls = []
    module.register_forward_pre_hook(
        lambda owner, args, kwargs: calls.append(type(kwargs.get("prefix")).__name__),
        with_kwargs=True,
    )
    adapter = make_input_adapter(module, build_tokenizer(tmp_path / "tokenizer"), "cpu")
    operands, observed = conditioned_inputs(adapter, 0, 0, side=64)
    assert calls == [SenseNovaImagePrefix.__name__] * 3
    assert observed["reference_sizes"] == [(32, 32), (64, 32)]
    assert (
        observed["prefix_lengths"][0]
        > observed["prefix_lengths"][1]
        > observed["prefix_lengths"][2]
    )
    output = module(**operands[0])
    output.square().mean().backward()
    assert any(p.grad is not None for n, p in module.named_parameters() if "fm_head" in n)
    assert all(
        not layer.keys.requires_grad
        for inputs in operands
        for layer in inputs["past_key_values"].layers
    )
    module.disable_gradient_checkpointing()
    assert not any(getattr(layer, "gradient_checkpointing", False) for layer in module.modules())


@pytest.mark.skipif(
    torch.cuda.device_count() < 8, reason="SenseNova DP2×CP2×TP2 requires eight CUDA ranks"
)
@pytest.mark.parametrize("pixel", [False, True], ids=["u10", "u15"])
@pytest.mark.parametrize("mode,checkpointing", [("full", False), ("lora", True)])
@pytest.mark.parametrize("sp", [False, True])
def test_cuda_sensenova_tp_sp_cp_native_reference(tmp_path, pixel, mode, checkpointing, sp):
    from sensenova_native_control import worker

    results = run_gloo_ranks(
        worker,
        tmp_path,
        8,
        dict(pixel=pixel, mode=mode, checkpointing=checkpointing, sp=sp),
        timeout=2400,
    )
    assert all(row.get("status") == "pass" and "error" not in row for row in results), results
    print(json.dumps({"probe": "sensenova_native_prefix_spatial_flow", "ranks": results}))


@pytest.mark.parametrize("version", ["u10", "u15"])
def test_cuda_sensenova_guard_ordered_lora(tmp_path, version):
    from context_parallel_overlap_runner import qualify_overlap

    qualify_overlap(
        tmp_path,
        dict(
            case_module=f"sensenova_{version}_coupled_fixture",
            model_family="sensenova",
            algorithm="grpo-guard",
            training_mode="lora",
            scheduling_mode="ordered",
            dp_size=2,
            tp_size=2,
            cp_size=2,
            sequence_parallel=True,
        ),
    )


def _schedule_worker(rank, store_path, result_dir, options):
    from types import SimpleNamespace

    from flow_factory.models.sensenova.sensenova import SenseNovaAdapter

    dist.init_process_group("gloo", init_method=f"file://{store_path}", rank=rank, world_size=2)
    result = {"rank": rank}
    try:
        adapter = object.__new__(SenseNovaAdapter)
        adapter.accelerator = SimpleNamespace(device=torch.device("cpu"), num_processes=2)
        adapter._is_param_sharded = lambda: True
        with pytest.raises(ValueError, match="identical module-call schedules"):
            adapter._assert_sharded_schedule((rank, 1, 0), "prefix conditioning")
        adapter._assert_sharded_schedule((1, 1, 1), "prefix conditioning")
        result["status"] = "pass"
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result))
        dist.destroy_process_group()


@requires_gloo
def test_sensenova_sharded_conditioning_guard_fails_on_every_peer(tmp_path):
    results = run_gloo_ranks(_schedule_worker, tmp_path, 2, {})
    assert all(row.get("status") == "pass" and "error" not in row for row in results), results


def test_sensenova_native_recipe_passes_production_dependency_gate(monkeypatch, tmp_path):
    from sensenova_coupled_fixture import make_config
    from sensenova_cp_fixture import qualify_native_dependencies

    from flow_factory.models.sensenova.sensenova import SenseNovaAdapter
    from flow_factory.trainers.common.context_parallel import validate_context_parallel_config
    from flow_factory.trainers.registry import get_trainer_class

    monkeypatch.setenv("WORLD_SIZE", "8")
    versions = qualify_native_dependencies()
    assert versions["diffusers"].startswith("0.42.")
    config = make_config(
        tmp_path,
        tmp_path,
        dict(
            algorithm="grpo-guard",
            training_mode="lora",
            dp_size=2,
            tp_size=2,
            cp_size=2,
            sequence_parallel=True,
        ),
    )
    validate_context_parallel_config(config, SenseNovaAdapter, get_trainer_class("grpo-guard"))
    assert "negative_prompt" not in config.training_args.extra_kwargs
    assert config.training_args.extra_kwargs["image_guidance_scale"] == 1.5
    assert config.eval_args.extra_kwargs["image_guidance_scale"] == 1.5


def _forward_entry_worker(rank, store_path, result_dir, options):
    from types import SimpleNamespace

    from flow_factory.models.sensenova.sensenova import SenseNovaAdapter

    dist.init_process_group("gloo", init_method=f"file://{store_path}", rank=rank, world_size=2)
    result = {"rank": rank, "mismatch": options["mismatch"]}
    work = []
    try:

        class NoWorkAdapter(SenseNovaAdapter):
            def _build_context(self, *args, **kwargs):
                work.append("prefix")
                raise AssertionError("Prefix work entered before admission")

            @property
            def transformer(self):
                def forbidden(**kwargs):
                    work.append("denoising")
                    raise AssertionError("Denoising work entered before admission")

                return forbidden

        adapter = object.__new__(NoWorkAdapter)
        adapter.accelerator = SimpleNamespace(device=torch.device("cpu"), num_processes=2)
        adapter._is_param_sharded = lambda: True
        adapter._image_shape = lambda *args: (32, 32)
        adapter._noise_scale = lambda *args: 1.0
        cache = object() if options["mismatch"] == "cache_mode" and rank == 1 else None
        prompt = "" if options["mismatch"] != "cache_mode" else "sample"
        if rank == 1 and options["mismatch"] == "raw_prompt":
            prompt = None
        elif rank == 1 and options["mismatch"] == "empty_prompt_list":
            prompt = []
        call = dict(
            t=torch.tensor([500.0]),
            latents=torch.zeros(1, 3, 32, 32),
            prompt=prompt,
            past_key_values=cache,
        )
        if options["mismatch"] == "valid_empty_string":
            # Empty text is a valid unconditional prompt on every peer. It must
            # pass admission and reach the prefix route, unlike None or [].
            with pytest.raises(AssertionError, match="Prefix work entered"):
                adapter.forward(**call)
            assert work == ["prefix"]
        else:
            with pytest.raises(
                ValueError,
                match="forward entry.*rebuild_context.*required_raw_prompt_present",
            ):
                adapter.forward(**call)
            assert work == []
        adapter._assert_sharded_schedule((1, 0, 0), "forward entry after rejection")
        result.update(status="pass", module_work=work)
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result))
        dist.destroy_process_group()


@requires_gloo
@pytest.mark.parametrize(
    "mismatch", ["cache_mode", "raw_prompt", "empty_prompt_list", "valid_empty_string"]
)
def test_sensenova_forward_entry_rejects_divergence_before_any_model_work(tmp_path, mismatch):
    results = run_gloo_ranks(_forward_entry_worker, tmp_path, 2, {"mismatch": mismatch})
    expected_work = ["prefix"] if mismatch == "valid_empty_string" else []
    assert all(
        row.get("status") == "pass"
        and row.get("module_work") == expected_work
        and "error" not in row
        for row in results
    ), results


def _native_guard_inputs(pixel, mode="full"):
    from sensenova_cp_fixture import (
        build_denoiser,
        install_lora,
        make_input_adapter,
        prefix_and_inputs,
    )

    from flow_factory.scheduler import FlowMatchEulerDiscreteSDEScheduler, SchedulerGroup

    model = build_denoiser(pixel_head=pixel)
    if mode == "lora":
        model = install_lora(model)
    adapter = make_input_adapter(model, None, "cpu")
    adapter.pipeline.scheduler = FlowMatchEulerDiscreteSDEScheduler(noise_level=0.35)
    adapter.scheduler_group = SchedulerGroup({"latent": adapter.scheduler}, primary_name="latent")
    adapter.scheduler.set_timesteps(3, device="cpu")
    native_inputs, _ = prefix_and_inputs(model, side=64)
    latents = native_inputs["latents"]
    inputs = dict(
        t=torch.tensor([600.0]),
        t_next=torch.tensor([300.0]),
        latents=latents,
        next_latents=latents + 0.137,
        height=64,
        width=64,
        guidance_scale=1.0,
        past_key_values=native_inputs["past_key_values"],
        indexes_image=native_inputs["indexes_image"],
        attention_mask=native_inputs["attention_mask"],
        noise_level=0.35,
    )
    return model, adapter, inputs


@pytest.mark.parametrize("pixel", [False, True], ids=["u10", "u15"])
@pytest.mark.parametrize("mode", ["full", "lora"])
@pytest.mark.parametrize("fields", ["all", "velocity"])
def test_native_sensenova_singleton_preserves_guard_graph_and_parameter_gradients(
    pixel, mode, fields
):
    from types import SimpleNamespace

    from coupled_objective_reference import assert_objective

    from flow_factory.samples import ComponentTimes, LatentState, MultiModalStepOutput, ReplayStep
    from flow_factory.scheduler import FlowMatchEulerDiscreteSDESchedulerOutput
    from flow_factory.trainers.rl.grpo import GRPOGuardTrainer

    model, adapter, inputs = _native_guard_inputs(pixel, mode)
    if fields == "velocity":
        inputs.update(compute_log_prob=False, return_kwargs=["velocity"])
    native_outputs = []
    original_step = adapter.scheduler.step

    def observe_step(**kwargs):
        result = original_step(**kwargs)
        native_outputs.append(result)
        return result

    adapter.scheduler.step = observe_step
    output = adapter.forward(**inputs)
    (native,) = native_outputs
    native_fields, public_fields = native.to_dict(), output.to_dict()
    assert type(output) is FlowMatchEulerDiscreteSDESchedulerOutput
    assert public_fields.keys() == native_fields.keys()
    for name, expected in native_fields.items():
        actual = public_fields[name]
        if expected is None:
            assert actual is None
        else:
            assert actual is expected
            assert type(actual) is type(expected) and actual.dtype == expected.dtype
            torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=name)
    if fields == "velocity":
        assert output.next_latents_mean is None and output.log_prob is None
        return

    # The public mean and velocity must be ancestors of the public log probability,
    # including the contribution independently checked by the coupled-loss oracle.
    upstreams = torch.autograd.grad(
        output.log_prob.sum(), (output.next_latents_mean, output.velocity), retain_graph=True
    )
    assert all(torch.count_nonzero(value) for value in upstreams)
    concatenated = type(native).from_dict(
        {
            name: torch.cat([value], dim=0) if value is not None else None
            for name, value in native_fields.items()
        }
    )
    disconnected = torch.autograd.grad(
        concatenated.log_prob.sum(),
        (concatenated.next_latents_mean, concatenated.velocity),
        allow_unused=True,
        retain_graph=True,
    )
    assert disconnected == (None, None)
    trainer = object.__new__(GRPOGuardTrainer)
    trainer.adapter = adapter
    trainer.training_args = SimpleNamespace(kl_type="v-based")
    old_log_prob = native.log_prob.detach() - 0.02
    replay = ReplayStep(
        state=LatentState({"latent": inputs["latents"]}),
        next_state=LatentState({"latent": inputs["next_latents"]}),
        times=ComponentTimes(
            timestep={"latent": inputs["t"]}, next_timestep={"latent": inputs["t_next"]}
        ),
        log_prob=old_log_prob,
        component_log_probs={"latent": old_log_prob},
    )
    old_mean = LatentState({"latent": native.next_latents_mean.detach() + 0.01})
    reference = MultiModalStepOutput(
        velocity=LatentState({"latent": native.velocity.detach() + 0.03})
    )
    advantage, beta = torch.tensor([0.7]), 0.01

    def guard_loss(raw):
        step = MultiModalStepOutput(
            next_state_mean=LatentState({"latent": raw.next_latents_mean}),
            velocity=LatentState({"latent": raw.velocity}),
            std_dev_t={"latent": raw.std_dev_t},
            dt={"latent": raw.dt},
            log_prob=raw.log_prob,
            component_log_probs={"latent": raw.log_prob},
        )
        ratio = trainer._guard_ratio(step, replay, old_mean)
        policy = torch.maximum(-advantage * ratio, -advantage * ratio.clamp(0.8, 1.2)).mean()
        return policy + beta * trainer._reference_kl_divergence(step, reference, replay)

    loss, legacy_loss = guard_loss(output), guard_loss(concatenated)
    torch.testing.assert_close(loss, legacy_loss, rtol=0, atol=0)
    parameters = [value for value in model.parameters() if value.requires_grad]
    actual_gradients = torch.autograd.grad(loss, parameters, allow_unused=True, retain_graph=True)
    legacy_gradients = torch.autograd.grad(
        legacy_loss, parameters, allow_unused=True, retain_graph=True
    )
    assert any(value is not None and torch.count_nonzero(value) for value in actual_gradients)
    for actual, expected in zip(actual_gradients, legacy_gradients):
        if expected is None:
            assert actual is None
        else:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    mean_gradient, velocity_gradient = torch.autograd.grad(
        loss * 0.5, (output.next_latents_mean, output.velocity), retain_graph=True
    )
    assert_objective(
        {
            "algorithm": "grpo-guard",
            "mean": output.next_latents_mean.detach(),
            "velocity": output.velocity.detach(),
            "following": inputs["next_latents"],
            "old_mean": old_mean.components["latent"],
            "reference": reference.velocity.components["latent"],
            "log_prob": output.log_prob.detach(),
            "old_log_prob": old_log_prob,
            "std": output.std_dev_t,
            "dt": output.dt,
            "sigma": inputs["t"] / 1000,
            "advantage": advantage,
            "adv_clip_range": (-3, 3),
            "clip_range": (-0.2, 0.2),
            "kl_type": "v-based",
            "kl_beta": beta,
            "cp_size": 2,
            "gas": 4,
            "mean_gradient": mean_gradient,
            "velocity_gradient": velocity_gradient,
            "loss": loss.detach(),
        }
    )


@pytest.mark.parametrize("pixel", [False, True], ids=["u10", "u15"])
def test_native_sensenova_multi_sample_output_keeps_existing_aggregation(pixel):
    _, adapter, inputs = _native_guard_inputs(pixel)
    inputs.update(
        latents=torch.cat([inputs["latents"], inputs["latents"] + 0.1]),
        next_latents=torch.cat([inputs["next_latents"], inputs["next_latents"] - 0.2]),
        t=torch.tensor([600.0, 550.0]),
        t_next=torch.tensor([300.0, 250.0]),
    )
    native_outputs = []
    original_step = adapter.scheduler.step

    def observe_step(**kwargs):
        result = original_step(**kwargs)
        native_outputs.append(result)
        return result

    adapter.scheduler.step = observe_step
    output = adapter.forward(**inputs)
    assert len(native_outputs) == 2
    for name, actual in output.to_dict().items():
        values = [getattr(item, name) for item in native_outputs]
        if all(value is not None for value in values):
            torch.testing.assert_close(actual, torch.cat(values), rtol=0, atol=0, msg=name)
        else:
            assert actual is None


@pytest.mark.parametrize("scale", [1.0, 16.0, 1024.0])
def test_sensenova_output_parity_scales_cancellation_floor_and_retains_raw_verdict(scale):
    from sensenova_native_control import _assert_output_parity

    expected = torch.tensor([[0.0, 1.0, -0.5, 0.125]], dtype=torch.float32) * scale
    actual = expected.clone()
    actual[0, 0] = 1.5e-6 * scale
    before = actual.clone(), expected.clone()
    (row,) = _assert_output_parity(actual, expected)
    assert row["reference_scale"] == scale and row["normalized_pointwise_pass"]
    assert row["raw_original_pass"] is (scale == 1.0)
    assert row["raw_pointwise_outliers"] == int(scale > 1.0)
    assert row["relative_l2"] < 2e-6
    torch.testing.assert_close(actual, before[0], rtol=0, atol=0)
    torch.testing.assert_close(expected, before[1], rtol=0, atol=0)
    actual[0, 0] = 3e-6 * scale
    with pytest.raises(AssertionError):
        _assert_output_parity(actual, expected)


@pytest.mark.parametrize("corruption", ["zero", "half", "double", "permutation", "nan", "inf"])
def test_sensenova_output_parity_rejects_corrupt_values(corruption):
    from sensenova_native_control import _assert_output_parity

    expected = torch.tensor([[0.0, 16.0, -8.0, 2.0]], dtype=torch.float32)
    actual = expected.clone()
    if corruption in ("zero", "half", "double"):
        actual *= {"zero": 0.0, "half": 0.5, "double": 2.0}[corruption]
    elif corruption == "permutation":
        actual = actual.flip(-1)
    else:
        actual[0, 0] = float(corruption)
    with pytest.raises(AssertionError):
        _assert_output_parity(actual, expected)


def test_sensenova_output_parity_keeps_cfg_scales_separate_and_raw_cap():
    from sensenova_native_control import _assert_output_parity

    expected = torch.tensor([[0.0, 1.0], [0.0, 16.0], [0.0, 1024.0]], dtype=torch.float32)
    actual = expected.clone()
    actual[0, 0] = 3e-6
    with pytest.raises(AssertionError):
        _assert_output_parity(actual, expected)
    tiny = torch.tensor([[0.0, 1e-5]], dtype=torch.float32)
    received = tiny.clone()
    received[0, 0] = 1e-6
    # The normalized pointwise rule alone admits this; the original raw cap rejects it.
    torch.testing.assert_close(received, tiny, rtol=3e-4, atol=2e-6)
    with pytest.raises(AssertionError):
        _assert_output_parity(received, tiny)
    (zero,) = _assert_output_parity(torch.zeros_like(tiny), torch.zeros_like(tiny))
    assert zero["raw_original_pass"] and zero["relative_l2"] == 0


@pytest.mark.parametrize("mismatch", ["shape", "dtype", "reference_nan", "empty"])
def test_sensenova_output_parity_rejects_invalid_contract(mismatch):
    from sensenova_native_control import _assert_output_parity

    expected = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    actual = expected.clone()
    if mismatch == "shape":
        actual = actual.reshape(1, 1, 2)
    elif mismatch == "dtype":
        actual = actual.double()
    elif mismatch == "reference_nan":
        expected[0, 0] = float("nan")
    else:
        actual = expected = expected[:, :0]
    with pytest.raises(AssertionError):
        _assert_output_parity(actual, expected)


def _inactive_order_worker(rank, store_path, result_dir, options):
    from types import SimpleNamespace

    import sensenova_native_control as control
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, Shard

    result = {"rank": rank}
    dist.init_process_group("gloo", init_method=f"file://{store_path}", rank=rank, world_size=2)
    try:
        mesh = init_device_mesh("cpu", (2,))
        expected = {
            name: torch.arange(4, dtype=torch.float64) + 100 * (index + 1)
            for index, name in enumerate(sorted(options["orders"][0]))
        }
        parameters = {
            name: torch.nn.Parameter(
                DTensor.from_local(value.chunk(2)[rank].clone(), mesh, [Shard(0)], run_check=False),
                requires_grad=False,
            )
            for name, value in expected.items()
        }
        incoming = options["orders"][rank]
        # Same-shaped collectives all complete, but rank-dependent name order
        # reconstructs pieces of different parameters. Check after all calls.
        legacy = {name: control.full_value(parameters[name]) for name in incoming}
        assert any(not torch.equal(legacy[name], expected[name]) for name in expected)
        materialized = []
        names = {id(value): name for name, value in parameters.items()}
        native_full_value = control.full_value

        def observe_full_value(value):
            materialized.append(names[id(value)])
            return native_full_value(value)

        try:
            control.full_value = observe_full_value
            control._assert_inactive_state(
                parameters,
                SimpleNamespace(state={}),
                {"initial_parameters": expected},
                incoming,
            )
        finally:
            control.full_value = native_full_value
        assert materialized == sorted(expected)
        result.update(status="pass", incoming=incoming, materialized=materialized)
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result))
        dist.destroy_process_group()


@requires_gloo
def test_sensenova_inactive_state_materialization_orders_real_collectives_across_hash_seeds(
    tmp_path,
):
    import os
    import subprocess
    import sys

    script = "import json; print(json.dumps(list({'alpha', 'beta', 'gamma', 'delta'})))"
    orders = [
        json.loads(
            subprocess.check_output(
                [sys.executable, "-c", script],
                text=True,
                env={**os.environ, "PYTHONHASHSEED": str(seed)},
            )
        )
        for seed in (1, 2)
    ]
    assert orders[0] != orders[1] and set(orders[0]) == set(orders[1])
    results = run_gloo_ranks(_inactive_order_worker, tmp_path, 2, {"orders": orders})
    assert all(row.get("status") == "pass" and "error" not in row for row in results), results
    assert results[0]["incoming"] != results[1]["incoming"]
    assert results[0]["materialized"] == results[1]["materialized"] == sorted(orders[0])


@pytest.mark.parametrize("corruption", ["parameter", "optimizer_state"])
def test_sensenova_inactive_state_keeps_exact_values_and_empty_optimizer_state(corruption):
    from types import SimpleNamespace

    from sensenova_native_control import _assert_inactive_state

    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]), requires_grad=False)
    parameters = {"inactive": parameter}
    records = {"initial_parameters": {"inactive": parameter.detach().clone()}}
    optimizer = SimpleNamespace(state={})
    _assert_inactive_state(parameters, optimizer, records, {"inactive"})
    if corruption == "parameter":
        parameter[0] += 1e-6
    else:
        optimizer.state[parameter] = {"step": torch.tensor(1)}
    with pytest.raises(AssertionError):
        _assert_inactive_state(parameters, optimizer, records, {"inactive"})
