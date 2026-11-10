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

"""Native H3 packed SwiGLU and row-index CP/SP numerical controls."""

import copy
import json
import traceback
from itertools import product
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from conftest import requires_gloo, run_gloo_ranks
from h3_cp_fixture import adapter_class, build_transformer, install_lora, model_inputs


def _native_checkpoint_cast_control(module, native):
    """Compare identical CP/SP execution before and after the checkpoint owner."""
    from flow_factory.models.minimax_h3._chunking import install_h3_in_forward_block_checkpointing

    original = copy.deepcopy(module.state_dict())
    module.float()
    batches = [model_inputs("ref2va", odd=odd) for odd in (True, False)]
    direct = {}
    observations = []
    handles = [
        block.register_forward_pre_hook(
            lambda block, args, kwargs: observations.append(
                str((args[0] if args else kwargs["hidden_states"]).dtype)
            ),
            with_kwargs=True,
        )
        for block in native.transformer_blocks
    ]
    counts = {}
    for checkpointed in (False, True):
        if checkpointed:
            install_h3_in_forward_block_checkpointing(native)
        for dtype in (torch.float32, torch.bfloat16):
            module.zero_grad(set_to_none=True)
            observations.clear()
            with torch.autocast("cpu", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
                outputs = [module(**inputs) for inputs in batches]
                loss = sum(
                    video[:, 6:].float().square().mean() + audio[:, 2:].float().square().mean()
                    for video, audio in outputs
                )
            loss.backward()
            gradients = {
                name: (
                    value.grad.to_local() if hasattr(value.grad, "to_local") else value.grad
                ).clone()
                for name, value in module.named_parameters()
                if value.grad is not None
            }
            if not checkpointed:
                direct[dtype] = (outputs, gradients, tuple(observations))
                continue
            expected, expected_gradients, expected_dtypes = direct[dtype]
            assert tuple(observations) == expected_dtypes
            for actual_output, expected_output in zip(outputs, expected):
                for actual, target in zip(actual_output, expected_output):
                    torch.testing.assert_close(actual, target, rtol=0, atol=0)
            assert gradients.keys() == expected_gradients.keys()
            for name, value in gradients.items():
                torch.testing.assert_close(
                    value, expected_gradients[name], rtol=0, atol=0, msg=name
                )
            counts[str(dtype)] = len(gradients)
    for handle in handles:
        handle.remove()
    module.double().load_state_dict(original)
    module.zero_grad(set_to_none=True)
    return counts


def _worker(rank, store_path, result_dir, options):
    world = options["cp_size"] * options["tp_size"]
    dist.init_process_group(
        "gloo", store=dist.FileStore(store_path, world), rank=rank, world_size=world
    )
    result = {"rank": rank, "cases": []}
    try:
        from diffusers.models._modeling_parallel import ContextParallelConfig
        from torch.distributed.device_mesh import DeviceMesh
        from torch.distributed.tensor import DTensor

        from flow_factory.models.minimax_h3._chunking import (
            install_h3_attention_norm_chunking,
            install_h3_feed_forward_chunking,
            install_h3_rotary_chunking,
        )
        from flow_factory.models.minimax_h3._context_parallel import (
            compose_h3_sequence_boundaries,
            configure_h3_context_parallel,
            install_h3_context_parallel,
            validate_h3_context_parallel_module,
        )
        from flow_factory.models.tensor_parallel import (
            apply_tensor_parallel,
            peel_peft_wrapper,
            resolve_parallel_plan,
        )

        mesh = DeviceMesh(
            "cpu",
            torch.arange(world).reshape(options["cp_size"], options["tp_size"]).T.unsqueeze(1),
            mesh_dim_names=("tp", "ring", "ulysses"),
        )
        workflows = ("ref2va",) if options.get("in_forward") else ("t2va", "fl2va", "ref2va")
        for workflow, mode in product(workflows, ("full", "lora")):
            reference = build_transformer().double()
            module = copy.deepcopy(reference)
            native = peel_peft_wrapper(module)
            install_h3_feed_forward_chunking(native, max_tokens=3)
            install_h3_attention_norm_chunking(native, max_tokens=3)
            if workflow == "ref2va":
                install_h3_rotary_chunking(native, max_tokens=3)
            if mode == "lora":
                torch.manual_seed(73)
                reference = install_lora(reference)
                torch.manual_seed(73)
                module = install_lora(module)
                if workflow == "ref2va":
                    from flow_factory.models.minimax_h3._chunking import (
                        install_h3_lora_projection_chunking,
                    )

                    install_h3_lora_projection_chunking(
                        native, max_tokens=3, chunk_output_projection=False
                    )
            validate_h3_context_parallel_module(native, options["cp_size"], options["tp_size"])
            if options["cp_size"] > 1:
                configure_h3_context_parallel(native)
            module.set_attention_backend("native")
            reference.set_attention_backend("native")
            if options["tp_size"] > 1:
                apply_tensor_parallel(
                    native,
                    mesh["tp"],
                    resolve_parallel_plan(adapter_class(workflow), sequence_parallel=options["sp"]),
                )
            if options["cp_size"] > 1:
                install_h3_context_parallel(
                    native, ContextParallelConfig(ring_degree=1, ulysses_degree=2, mesh=mesh)
                )
                if options["sp"]:
                    boundaries = compose_h3_sequence_boundaries(native)
                    assert len(boundaries) == 5
                # CP replaces SP-only padding with its own cp_size * tp_size padding.
                assert not hasattr(native, "_ff_h3_sp_padding_handles")
            elif options["sp"]:
                assert len(native._ff_h3_sp_padding_handles) == 3 + 2 * len(
                    native.transformer_blocks
                )
            if options.get("in_forward"):
                from types import MethodType

                from flow_factory.models.minimax_h3._chunking import (
                    install_h3_in_forward_block_checkpointing,
                )

                for block in native.transformer_blocks:
                    registry = getattr(block, "_diffusers_hook", None)
                    assert registry is None or not registry.hooks
                    if registry is not None:
                        assert isinstance(block.forward, MethodType)
                        assert block.forward.__self__ is block
                        assert block.forward.__func__ is type(block).forward
                if options["cp_size"] > 1:
                    boundaries = compose_h3_sequence_boundaries(native)
                    assert compose_h3_sequence_boundaries(native) is boundaries
                cast_control = _native_checkpoint_cast_control(module, native)
                install_h3_in_forward_block_checkpointing(native)
            else:
                cast_control = None
                native.enable_gradient_checkpointing()
            reference.enable_gradient_checkpointing()
            optimizers = [
                torch.optim.AdamW(model.parameters(), lr=1e-3) for model in (reference, module)
            ]
            expected_parameters = dict(reference.named_parameters())
            cv, ca = {"t2va": (0, 0), "fl2va": (4, 0), "ref2va": (6, 2)}[workflow]

            def loss(output):
                return (
                    output[0][:, cv:].square().mean() * 0.7
                    + output[1][:, ca:].square().mean() * 0.3
                )

            def full(value):
                return value.full_tensor() if isinstance(value, DTensor) else value

            comparisons = missing_cp_factor = 0
            for step in range(3):
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
                input_batches = [
                    {
                        key: (
                            value.double()
                            if torch.is_tensor(value) and value.is_floating_point()
                            else value
                        )
                        for key, value in model_inputs(workflow, step=step, odd=odd).items()
                    }
                    for odd in ((True, False) if options.get("in_forward") else (True,))
                ]
                expected_outputs = [reference(**inputs) for inputs in input_batches]
                actual_outputs = [module(**inputs) for inputs in input_batches]
                for actual, expected in zip(actual_outputs, expected_outputs):
                    for value, target in zip(actual, expected):
                        torch.testing.assert_close(value, target, rtol=1e-8, atol=1e-10)
                sum(map(loss, expected_outputs)).backward()
                sum(map(loss, actual_outputs)).backward()
                inputs = input_batches[0]
                for name, parameter in module.named_parameters():
                    target = expected_parameters[name]
                    assert (parameter.grad is None) == (target.grad is None), name
                    if target.grad is None:
                        assert not parameter.requires_grad, name
                        continue
                    before = full(parameter.grad).detach().clone()
                    missing_cp_factor += int(
                        not torch.allclose(before, target.grad, rtol=1e-8, atol=1e-10)
                    )
                    local = (
                        parameter.grad.to_local()
                        if isinstance(parameter.grad, DTensor)
                        else parameter.grad
                    )
                    dist.all_reduce(local, group=mesh["ulysses"].get_group())
                    torch.testing.assert_close(
                        full(parameter.grad), target.grad, rtol=1e-8, atol=1e-10, msg=name
                    )
                    comparisons += 1
                for optimizer in optimizers:
                    optimizer.step()
                for name, parameter in module.named_parameters():
                    target = expected_parameters[name]
                    torch.testing.assert_close(
                        full(parameter), target, rtol=1e-8, atol=1e-10, msg=name
                    )
                    if parameter.requires_grad:
                        for field in ("exp_avg", "exp_avg_sq", "step"):
                            torch.testing.assert_close(
                                full(optimizers[1].state[parameter][field]),
                                optimizers[0].state[target][field],
                                rtol=1e-8,
                                atol=1e-10,
                                msg=name + field,
                            )
            assert comparisons > 0
            assert (missing_cp_factor > 0) == (options["cp_size"] > 1)
            wrong_snapshot_mismatches = None
            if options.get("in_forward"):
                from torch.utils.checkpoint import checkpoint

                # Deliberately omit the invocation snapshot while retaining the
                # same native body and both original CP input cuts.
                def unsafe_checkpoint(block, *args, **kwargs):
                    return checkpoint(
                        block.flow_factory_uncheckpointed_forward,
                        block,
                        *args,
                        use_reentrant=False,
                        preserve_rng_state=False,
                        **kwargs,
                    )

                for block in native.transformer_blocks:
                    block.forward = MethodType(unsafe_checkpoint, block)
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
                expected_outputs = [reference(**inputs) for inputs in input_batches]
                actual_outputs = [module(**inputs) for inputs in input_batches]
                sum(map(loss, expected_outputs)).backward()
                sum(map(loss, actual_outputs)).backward()
                wrong_snapshot_mismatches = 0
                for name, parameter in module.named_parameters():
                    target = expected_parameters[name]
                    if target.grad is None:
                        continue
                    local = (
                        parameter.grad.to_local()
                        if isinstance(parameter.grad, DTensor)
                        else parameter.grad
                    )
                    dist.all_reduce(local, group=mesh["ulysses"].get_group())
                    wrong_snapshot_mismatches += int(
                        not torch.allclose(full(parameter.grad), target.grad, rtol=1e-8, atol=1e-10)
                    )
                assert wrong_snapshot_mismatches > 0
                inputs = input_batches[0]
            padding_residual = None
            if options["cp_size"] > 1 or options["sp"]:
                # This deliberately broken mask still uses identical padded tensors,
                # projections and native CP/SP. Only allowing dummy keys changes output.
                for handle in (
                    native._ff_h3_padding_mask_handles
                    if options["cp_size"] > 1
                    else native._ff_h3_sp_padding_mask_handles
                ):
                    handle.remove()
                with torch.no_grad():
                    expected = reference(**inputs)
                    unmasked = module(**inputs)
                padding_residual = max(
                    (left - right).abs().max().item() for left, right in zip(unmasked, expected)
                )
                assert padding_residual > 1e-8, padding_residual
            if options.get("in_forward") and options["cp_size"] > 1:
                config = native._parallel_config.context_parallel_config
                with pytest.raises(ValueError, match="exactly once"):
                    install_h3_context_parallel(native, config)
                with pytest.raises(ValueError, match="exactly once"):
                    install_h3_context_parallel(native, object())
                boundaries[0].handle.remove()
                with pytest.raises(ValueError, match="boundary installation was modified"):
                    compose_h3_sequence_boundaries(native)
            result["cases"].append(
                {
                    "workflow": workflow,
                    "mode": mode,
                    "gradients": comparisons,
                    "optimizer_steps": 3,
                    "missing_cp_factor_mismatches": missing_cp_factor,
                    "unmasked_padding_residual": padding_residual,
                    "rows": inputs["position_ids"].shape[0],
                    "checkpoint_lengths": [
                        batch["position_ids"].shape[0] for batch in input_batches
                    ],
                    "wrong_snapshot_mismatches": wrong_snapshot_mismatches,
                    "checkpoint_cast_gradient_counts": cast_control,
                }
            )
        result["status"] = "pass"
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result, indent=2))
        dist.destroy_process_group()


@requires_gloo
@pytest.mark.parametrize(
    "cp_size,tp_size,sp",
    [(2, 1, False), (2, 2, False), (2, 2, True), (1, 2, True)],
    ids=("cp2-only", "cp2-tp2", "cp2-tp2-sp", "cp1-tp2-sp"),
)
def test_native_h3_packed_tp_sp_cp_all_parameter_gradients(tmp_path, cp_size, tp_size, sp):
    rows = run_gloo_ranks(
        _worker, tmp_path, cp_size * tp_size, {"cp_size": cp_size, "tp_size": tp_size, "sp": sp}
    )
    assert all(row.get("status") == "pass" for row in rows), rows


@requires_gloo
@pytest.mark.parametrize(
    "cp_size,tp_size,sp",
    [(2, 1, False), (2, 2, False), (2, 2, True), (1, 2, True)],
    ids=("cp2-only", "cp2-tp2", "cp2-tp2-sp", "cp1-tp2-sp"),
)
def test_native_h3_in_forward_checkpoint_all_parameter_gradients(tmp_path, cp_size, tp_size, sp):
    rows = run_gloo_ranks(
        _worker,
        tmp_path,
        cp_size * tp_size,
        {"cp_size": cp_size, "tp_size": tp_size, "sp": sp, "in_forward": True},
    )
    assert all(row.get("status") == "pass" for row in rows), rows


@requires_gloo
@pytest.mark.parametrize("mode", ["full", "lora"])
def test_native_h3_actual_width_modulation_all_parameter_gradients(tmp_path, mode):
    from h3_modulation_reference import run_native_modulation_reference

    rows = run_gloo_ranks(run_native_modulation_reference, tmp_path, 2, {"mode": mode})
    assert all(row.get("status") == "pass" for row in rows), rows


def test_native_h3_processor_admission_is_atomic():
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3AttnProcessor

    from flow_factory.models.minimax_h3._context_parallel import configure_h3_context_parallel

    class Foreign(MiniMaxH3AttnProcessor):
        pass

    module = build_transformer()
    first = module.transformer_blocks[0].attn.processor
    module.transformer_blocks[-1].attn.set_processor(Foreign())
    with pytest.raises(ValueError, match="native unfused attention"):
        configure_h3_context_parallel(module)
    assert module.transformer_blocks[0].attn.processor is first
    assert not hasattr(module.rope, "_diffusers_hook")


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_attention_heads", 55),
        ("ffn_dim", 127),
        ("time_embed_dim", 31),
        ("in_channels", 16),
        ("patch_size", (1, 1, 1)),
        ("attention_head_dim", 64),
    ],
)
def test_native_h3_config_rejects_unsupported_layout(field, value):
    from flow_factory.models.minimax_h3._context_parallel import validate_h3_context_parallel_config

    config = dict(build_transformer().config)
    config[field] = value
    with pytest.raises(ValueError, match="MiniMax H3"):
        validate_h3_context_parallel_config(config, 2, 2)


@pytest.mark.parametrize("field", ["partition", "tag", "time", "batch", "width"])
def test_native_h3_packed_metadata_fails_before_collectives(field):
    from flow_factory.models.minimax_h3._context_parallel import validate_h3_context_parallel_inputs

    values = model_inputs("ref2va", odd=True)
    if field == "partition":
        values["video_indices"][0] = values["audio_indices"][0]
    elif field == "tag":
        values["token_tags"][values["audio_indices"][0]] = 0
    elif field == "time":
        values["timestep_indices"][0] = 4
    elif field == "batch":
        values["hidden_states"] = values["hidden_states"].expand(2, -1, -1)
    else:
        values["audio_hidden_states"] = values["audio_hidden_states"][:, :, :-1]
    with pytest.raises(ValueError, match="MiniMax H3"):
        validate_h3_context_parallel_inputs(values, 4)


@pytest.mark.parametrize("workflow", ["t2va", "fl2va", "ref2va"])
def test_native_h3_encoding_order_and_saved_roundtrip(tmp_path, workflow):
    from h3_cp_fixture import build_pipeline, workflow_inputs

    from diffusers import ModularPipeline
    from flow_factory.models.minimax_h3.blocks import encode_h3_workflow_inputs

    pipeline = build_pipeline(workflow)
    values = workflow_inputs(workflow)
    with torch.no_grad():
        encoded = encode_h3_workflow_inputs(pipeline, values, workflow=workflow)
    pipeline.save_pretrained(str(tmp_path), overwrite_modular_index=True)
    restored = ModularPipeline.from_pretrained(str(tmp_path), workflow=workflow)
    restored.load_components(dtype=torch.float32)
    with torch.no_grad():
        actual = encode_h3_workflow_inputs(restored, values, workflow=workflow)
        varied = encode_h3_workflow_inputs(
            restored, {**values, "prompt": "a red video moves"}, workflow=workflow
        )
    for key in (
        "prompt_embeds",
        "position_ids",
        "token_tags",
        "video_indices",
        "audio_indices",
        "text_indices",
    ):
        torch.testing.assert_close(actual[key], encoded[key], rtol=0, atol=0)
    assert varied["prompt_embeds"].shape[1] != encoded["prompt_embeds"].shape[1]
    assert encoded["num_latent_frames"] == 37 and encoded["num_audio_latents"] == 207
    if workflow != "t2va":
        assert encoded["num_condition_video_rows"] > 0
        for actual_latents, expected_latents in zip(
            actual["condition_latents"], encoded["condition_latents"]
        ):
            torch.testing.assert_close(actual_latents, expected_latents, rtol=0, atol=0)
        reversed_values = dict(values)
        if workflow == "fl2va":
            reversed_values["image"], reversed_values["last_image"] = (
                values["last_image"],
                values["image"],
            )
        else:
            assert encoded["num_condition_audio_rows"] == 80
            reversed_values["references"] = list(reversed(values["references"]))
            for actual_latents, expected_latents in zip(
                actual["audio_condition_latents"], encoded["audio_condition_latents"]
            ):
                torch.testing.assert_close(actual_latents, expected_latents, rtol=0, atol=0)
        with torch.no_grad():
            reversed_encoding = encode_h3_workflow_inputs(
                restored, reversed_values, workflow=workflow
            )
        assert not torch.equal(reversed_encoding["prompt_embeds"], encoded["prompt_embeds"])
        assert not torch.equal(
            reversed_encoding["condition_latents"][0], encoded["condition_latents"][0]
        )
    with torch.no_grad():
        video = restored.vae.decode(torch.zeros(1, 24, 37, 2, 2), return_dict=False)[0]
        audio = restored.audio_vae.decode(torch.zeros(2, 32, 207), return_dict=False)[0]
    assert video.shape == (1, 3, 124, 32, 32) and torch.isfinite(video).all()
    assert audio.shape == (2, 1, 165600) and torch.isfinite(audio).all()


@pytest.mark.parametrize("workflow", ("fl2va", "ref2va"))
def test_native_h3_cached_rollout_preserves_conditions_and_clocks(tmp_path, workflow, monkeypatch):
    import h3_coupled_fixture as fixture
    from accelerate import Accelerator
    from h3_cp_fixture import workflow_inputs

    from flow_factory.models.minimax_h3.blocks import encode_h3_workflow_inputs
    from flow_factory.samples.references import canonicalize_reference_manifest

    fixture.build_fixture(tmp_path, 1, workflow)
    config = fixture.make_config(
        tmp_path,
        tmp_path / "result",
        {"dp_size": 1, "cp_size": 1, "tp_size": 1, "training_mode": "full", "algorithm": "grpo"},
        workflow,
    )
    adapter = adapter_class(workflow)(config, Accelerator(cpu=True))
    adapter.component_runtime.materialize_components(
        adapter.preprocessing_modules + adapter.inference_modules
    )
    values = workflow_inputs(workflow)
    with torch.no_grad():
        encoded = encode_h3_workflow_inputs(adapter.pipeline, values, workflow=workflow)
    extra = (
        {"images": [[values["image"], values["last_image"]]]}
        if workflow == "fl2va"
        else {
            "reference_manifest": [
                canonicalize_reference_manifest(
                    fixture.dataset_record(tmp_path, values["prompt"], 1, workflow)["references"],
                    row_index=0,
                )
            ]
        }
    )
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        samples = adapter.inference(
            prompt=[values["prompt"]],
            **encoded,
            frame_rate=24.0,
            num_inference_steps=3,
            guidance_scale=1.0,
            generator=torch.Generator().manual_seed(17),
            extra_call_back_kwargs=["velocity", "next_latents_mean"],
            **extra,
        )
    monkeypatch.setattr(fixture, "_audit_fresh_replay", lambda trainer, rows: rows)
    assert fixture.audit_samples(None, samples, workflow) is samples
    assert torch.isfinite(samples[0].video).all() and torch.isfinite(samples[0].audio).all()


@pytest.mark.parametrize("workflow", ("t2va", "fl2va", "ref2va"))
@pytest.mark.parametrize("mode", ("full", "lora"))
def test_native_h3_joint_grpo_dataward_scheduler_oracle(monkeypatch, workflow, mode):
    from contextlib import nullcontext
    from types import SimpleNamespace

    from h3_coupled_observations import assert_objectives, observe_objectives

    from flow_factory.samples import ComponentTimes
    from flow_factory.scheduler import MiniMaxH3SDEScheduler
    from flow_factory.trainers.common import context_parallel
    from flow_factory.trainers.rl.grpo import GRPOTrainer
    from tests.trainers.test_coupled_trainer_parity import _replay, _trainer

    module = build_transformer()
    if mode == "lora":
        module = install_lora(module)
    module.set_attention_backend("native")
    adapter = object.__new__(adapter_class(workflow))
    adapter.accelerator = SimpleNamespace(device=torch.device("cpu"))
    adapter.pipeline = SimpleNamespace(scheduler=MiniMaxH3SDEScheduler(shift=12.0))
    adapter.audio_scheduler = MiniMaxH3SDEScheduler(shift=3.0)
    for scheduler in (adapter.scheduler, adapter.audio_scheduler):
        scheduler.set_timesteps(3)
    adapter.component_runtime = SimpleNamespace(get_component=lambda _: module)
    inputs = model_inputs(workflow, odd=True)
    cv, ca = {"t2va": (0, 0), "fl2va": (4, 0), "ref2va": (6, 2)}[workflow]
    replayed = _replay(
        {"video": inputs["hidden_states"][:, cv:], "audio": inputs["audio_hidden_states"][:, ca:]}
    )
    sigma = {"video": torch.tensor([0.6]), "audio": torch.tensor([0.5])}
    following = {"video": torch.tensor([0.4]), "audio": torch.tensor([0.2])}
    replayed.times = ComponentTimes(
        timestep={k: v * 1000 for k, v in sigma.items()},
        next_timestep={k: v * 1000 for k, v in following.items()},
        sigma=sigma,
        next_sigma=following,
    )
    layout = {
        key: inputs[key]
        for key in ("position_ids", "token_tags", "video_indices", "audio_indices", "text_indices")
    }
    layout.update(num_condition_video_rows=cv, num_condition_audio_rows=ca)

    def forward(batch, replayed, fields):
        return adapter.forward(
            state=replayed.state,
            times=replayed.times,
            next_state=replayed.next_state,
            prompt_embeds=inputs["encoder_hidden_states"],
            condition_prefixes={
                "video": inputs["hidden_states"][:, :cv],
                "audio": inputs["audio_hidden_states"][:, :ca],
            },
            layout=layout,
            guidance_scale=1.0,
            noise_level=0.4,
            compute_log_prob=True,
            return_fields=fields,
        )

    with torch.no_grad():
        replayed.log_prob = (
            forward(None, replayed, ("log_prob",)).log_prob - torch.tensor(1.02).log()
        )
    trainer = _trainer(
        GRPOTrainer,
        adapter,
        trainer_type="grpo",
        per_device_batch_size=1,
        num_inner_epochs=1,
        adv_clip_range=(-3.0, 3.0),
        clip_range=(-0.2, 0.2),
        kl_beta=0.0,
        guidance_scale=1.0,
        gradient_accumulation_steps=2,
    )
    trainer.epoch = trainer.step = 0
    trainer.log_args = SimpleNamespace(verbose=False)
    trainer.logger = None
    trainer.config = SimpleNamespace(parallelism_args=SimpleNamespace(cp_size=2))
    trainer.autocast = trainer.accumulate_gradients = nullcontext
    trainer.accelerator = SimpleNamespace(
        sync_gradients=False, backward=lambda loss: (loss / 2).backward()
    )
    trainer._order_samples_for_optimize = lambda samples, inner_epoch: samples
    trainer._iter_prefetched_batches = lambda samples, size: iter(
        [{"advantage": torch.tensor([1.5])}]
    )
    trainer._replay_forward = forward
    adapter.train = lambda: module.train()
    adapter.get_train_step_indices = lambda: [0]
    adapter.get_replay_step = lambda batch, index: replayed
    monkeypatch.setattr(context_parallel, "context_parallel_geometry", lambda _: (2, 0))
    with observe_objectives(trainer) as records:
        trainer.optimize([object()])
    assert_objectives(records, 2)
    assert records[0]["oracle"]["active_elements"] == 12 * 96 + 8 * 32
    gradients = {name: p.grad for name, p in module.named_parameters() if p.requires_grad}
    assert gradients and all(
        value is not None and torch.isfinite(value).all() for value in gradients.values()
    )
    assert any(bool(value.count_nonzero()) for value in gradients.values())
    if mode == "full":
        assert any(
            bool(value.count_nonzero()) for name, value in gradients.items() if "audio" in name
        )


@pytest.mark.parametrize("workflow", ("t2va", "fl2va", "ref2va"))
def test_native_h3_checkpoint_metadata_rejects_before_loading_weights(monkeypatch, workflow):
    from types import SimpleNamespace

    import flow_factory.models.minimax_h3.adapters as adapters
    from diffusers import MiniMaxH3Transformer3DModel
    from flow_factory.hparams.parallelism_args import ParallelismArguments

    adapter = object.__new__(adapter_class(workflow))
    adapter.config = SimpleNamespace(parallelism_args=ParallelismArguments(cp_size=2, tp_size=2))
    adapter.model_args = SimpleNamespace(model_name_or_path="unused-native-checkpoint")
    config = dict(build_transformer().config)
    config["num_attention_heads"] = 55
    inspected = []

    def metadata(path, subfolder):
        inspected.append((path, subfolder))
        return config

    monkeypatch.setattr(MiniMaxH3Transformer3DModel, "load_config", metadata)

    def unexpected(*args, **kwargs):
        raise AssertionError("weights must not load before metadata validation")

    monkeypatch.setattr(adapters, "load_h3_workflow_pipeline", unexpected)
    with pytest.raises(ValueError, match="heads=55"):
        adapter.load_pipeline()
    assert inspected == [("unused-native-checkpoint", adapter.transformer_component_name)]


def test_h3_capacity_inventory_rejects_missing_shards_before_worker_launch(tmp_path):
    from test_h3_pretrained_capacity import checkpoint_identity

    folder = tmp_path / "transformer_ref"
    folder.mkdir()
    (folder / "config.json").write_text("{}")
    index = {"weight_map": {"native.weight": "native.safetensors"}}
    (folder / "diffusion_pytorch_model.safetensors.index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="every released transformer_ref shard"):
        checkpoint_identity(tmp_path)
    (folder / "native.safetensors").write_bytes(b"small inventory fixture")
    identity = checkpoint_identity(tmp_path)
    assert identity["native.safetensors"]["bytes"] == 23
    assert len(identity["native.safetensors"]["sha256"]) == 64


def test_h3_capacity_config_keeps_eight_rank_tp4_loading_scope(monkeypatch, tmp_path):
    from test_h3_pretrained_capacity import capacity_config, resolve_checkpoint

    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("FF_H3_MODEL_PATH", str(tmp_path))
    assert resolve_checkpoint() == tmp_path
    config = capacity_config(tmp_path, tmp_path / "output")
    assert config.parallelism_args.effective_context_parallel_size == 1
    assert config.parallelism_args.effective_tensor_parallel_size == 4
    assert config.parallelism_args.dp_shard_size == 2
    assert config.parallelism_args.sequence_parallel is True
    assert config.model_args.target_components == ["transformer_ref"]
    assert config.model_args.finetune_type == "lora"
    assert config.model_args.component_load_dtypes == torch.float32
    assert config.model_args.frozen_parameters_dtype is None
    assert config.mixed_precision == "bf16"


def test_h3_capacity_inputs_preserve_native_tp4_replica_geometry():
    from test_h3_pretrained_capacity import TP_SIZE, WORLD_SIZE, capacity_inputs

    values = [capacity_inputs(rank) for rank in range(WORLD_SIZE)]
    for rank, row in enumerate(values):
        assert row["hidden_states"].shape == (1, 18, 96)
        assert row["audio_hidden_states"].shape == (1, 10, 32)
        assert row["encoder_hidden_states"].shape == (1, 4, 5120)
        assert row["position_ids"].shape == (32, 3)
        assert row["position_ids"].shape[0] // TP_SIZE == 8
        for key, tensor in row.items():
            if torch.is_tensor(tensor):
                torch.testing.assert_close(
                    tensor, values[rank // TP_SIZE * TP_SIZE][key], atol=0, rtol=0
                )
    for key in ("hidden_states", "audio_hidden_states", "encoder_hidden_states"):
        assert not torch.equal(values[0][key], values[TP_SIZE][key])


def test_native_h3_reference_optimizer_owns_replaced_parameters(monkeypatch):
    from types import SimpleNamespace

    from accelerate.utils import DistributedType
    from test_h3_context_parallel import _parallelize_and_build_optimizers
    from test_model_preparation import PreparationAdapter, TinyLinear, optimizer_for

    from flow_factory.hparams.parallelism_args import ParallelismArguments
    from flow_factory.loading.coordinator import ModelLoadCoordinator
    from flow_factory.trainers.common.model_preparation import ModelPreparation

    accelerator = SimpleNamespace(
        num_processes=1,
        process_index=0,
        distributed_type=DistributedType.NO,
        state=SimpleNamespace(distributed_type=DistributedType.NO),
        unwrap_model=lambda model: model,
        prepare=lambda *objects: objects,
    )
    adapter = PreparationAdapter(TinyLinear(), accelerator, "full")
    reference = PreparationAdapter(TinyLinear(), accelerator, "full")
    for target in (adapter, reference):
        target.declare_component_variants(("policy",))
    owner = ModelPreparation(ModelLoadCoordinator(adapter, accelerator), ParallelismArguments())
    stale_optimizer = optimizer_for(adapter)

    def replace_parameters():
        registry = adapter.component_variant_registry
        for module in registry.bundle_members().values():
            for child in module.modules():
                for name, value in tuple(child.named_parameters(recurse=False)):
                    setattr(child, name, torch.nn.Parameter(value.detach().clone()))
        registry.rebind_parameters(registry.bundle_members())

    monkeypatch.setattr(owner, "_apply_tensor_parallel", replace_parameters)
    reference_optimizer, actual_optimizer = _parallelize_and_build_optimizers(
        owner, adapter, reference
    )
    with pytest.raises(ValueError, match="missing=2, unexpected=2"):
        owner.prepare(stale_optimizer)
    assert all(
        group["lr"] == 0.001
        for opt in (reference_optimizer, actual_optimizer)
        for group in opt.param_groups
    )
    model, prepared_optimizer = owner.prepare(actual_optimizer)
    assert prepared_optimizer is actual_optimizer
    assert {id(value) for group in actual_optimizer.param_groups for value in group["params"]} == {
        id(value) for value in model.parameters()
    }
