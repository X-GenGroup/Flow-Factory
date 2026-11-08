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

"""CPU native LTX CP/SP controls with both rectangular cross-modal directions."""

import copy
import json
import traceback
from itertools import product
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from conftest import requires_gloo, run_gloo_ranks
from ltx2_cp_fixture import build_transformer, model_inputs


def _worker(rank, store_path, result_dir, options):
    dist.init_process_group("gloo", store=dist.FileStore(store_path, 4), rank=rank, world_size=4)
    result = {"rank": rank}
    try:
        from diffusers.models._modeling_parallel import ContextParallelConfig
        from peft import LoraConfig, get_peft_model
        from torch.distributed.device_mesh import DeviceMesh
        from torch.distributed.tensor import DTensor
        from tp_qualification import assert_optimizer_step_parity

        from flow_factory.models.ltx2._context_parallel import (
            compose_ltx_sequence_boundaries,
            configure_ltx_context_parallel,
            install_ltx_context_parallel,
        )
        from flow_factory.models.ltx2.ltx2_t2av import LTX2_T2AV_Adapter
        from flow_factory.models.tensor_parallel import (
            apply_tensor_parallel,
            peel_peft_wrapper,
            resolve_parallel_plan,
        )

        mesh = DeviceMesh(
            "cpu",
            torch.arange(4).reshape(2, 2).T.unsqueeze(1),
            mesh_dim_names=("tp", "ring", "ulysses"),
        )
        result["cases"] = []
        for ltx23, per_token, mode in product((False, True), (False, True), ("full", "lora")):
            reference = build_transformer(ltx23).double()
            if mode == "lora":
                targets = LTX2_T2AV_Adapter.default_target_modules.fget(
                    object.__new__(LTX2_T2AV_Adapter)
                )
                reference = get_peft_model(
                    reference,
                    LoraConfig(r=2, lora_alpha=2, target_modules=targets, init_lora_weights=False),
                )
            module = copy.deepcopy(reference)
            native = peel_peft_wrapper(module)
            configure_ltx_context_parallel(native)
            module.set_attention_backend("native")
            reference.set_attention_backend("native")
            apply_tensor_parallel(
                native, mesh["tp"], resolve_parallel_plan(LTX2_T2AV_Adapter, sequence_parallel=True)
            )
            install_ltx_context_parallel(
                native, ContextParallelConfig(ring_degree=1, ulysses_degree=2, mesh=mesh)
            )
            boundaries = compose_ltx_sequence_boundaries(native)
            assert len(boundaries) == 2
            inputs = {
                key: (
                    value.double()
                    if torch.is_tensor(value) and value.is_floating_point()
                    else value
                )
                for key, value in model_inputs(ltx23, per_token).items()
            }
            # The adapter batches unconditional and conditional model operands.
            inputs = {
                name: (
                    torch.cat((value, value * 0.75), dim=0)
                    if torch.is_tensor(value) and value.is_floating_point()
                    else torch.cat((value, value), dim=0) if torch.is_tensor(value) else value
                )
                for name, value in inputs.items()
            }
            expected = reference(**inputs)[:2]
            actual = module(**inputs)[:2]
            for output, target in zip(actual, expected):
                torch.testing.assert_close(output, target, rtol=1e-8, atol=1e-10)
            expected_loss = expected[0].square().mean() * 0.75 + expected[1].square().mean() * 0.25
            actual_loss = actual[0].square().mean() * 0.75 + actual[1].square().mean() * 0.25
            # Preserve native self-attention perturbation and modality-isolation
            # branches while the local adaptation owns only cross-modal attention.
            for extra in ({"spatio_temporal_guidance_blocks": [0]}, {"isolate_modalities": True}):
                guided_expected = reference(**inputs, **extra)[:2]
                guided_actual = module(**inputs, **extra)[:2]
                for output, target in zip(guided_actual, guided_expected):
                    torch.testing.assert_close(output, target, rtol=1e-8, atol=1e-10)
                expected_loss = expected_loss + 0.1 * sum(
                    value.square().mean() for value in guided_expected
                )
                actual_loss = actual_loss + 0.1 * sum(
                    value.square().mean() for value in guided_actual
                )
            expected_loss.backward()
            actual_loss.backward()
            for parameter in module.parameters():
                if parameter.grad is not None:
                    gradient = (
                        parameter.grad.to_local()
                        if isinstance(parameter.grad, DTensor)
                        else parameter.grad
                    )
                    dist.all_reduce(gradient, group=mesh["ulysses"].get_group())
            outcome = assert_optimizer_step_parity(reference, module, rtol=1e-8, atol=1e-10)
            result["cases"].append(
                {"ltx23": ltx23, "per_token": per_token, "mode": mode, "updates": outcome}
            )
        result["status"] = "pass"
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result, default=str))
        dist.destroy_process_group()


@requires_gloo
def test_native_ltx_rectangular_crossmodal_cp_sp_parameter_gradients(tmp_path):
    results = run_gloo_ranks(_worker, tmp_path, 4, {})
    assert all(row.get("status") == "pass" for row in results), results


@pytest.mark.parametrize("ltx23", (False, True))
def test_ltx_native_text_connectors_first_frame_and_av_decoders_reload(tmp_path, ltx23):
    from types import SimpleNamespace

    from ltx2_cp_fixture import build_pipeline
    from PIL import Image

    from diffusers import LTX2ImageToVideoPipeline
    from flow_factory.models.ltx2.ltx2_i2av import LTX2_I2AV_Adapter

    pipeline = build_pipeline(tmp_path / "build", ltx23=ltx23, image_conditioned=True)
    pipeline.save_pretrained(tmp_path / "pipeline")
    loaded = LTX2ImageToVideoPipeline.from_pretrained(
        tmp_path / "pipeline", torch_dtype=torch.float32
    )
    observed = []
    for native in (pipeline, loaded):
        adapter = object.__new__(LTX2_I2AV_Adapter)
        adapter.pipeline = native
        adapter.accelerator = SimpleNamespace(device=torch.device("cpu"))
        adapter._has_warned_multi_image = False
        prompt = adapter.encode_prompt(
            "a red bird sings softly",
            negative_prompt="blue",
            guidance_scale=2.0,
            max_sequence_length=128,
        )
        assert prompt["connector_prompt_embeds"].shape == (1, 128, 512 if ltx23 else 60)
        assert prompt["connector_audio_prompt_embeds"].shape == (1, 128, 256 if ltx23 else 60)
        assert not torch.equal(
            prompt["connector_prompt_embeds"], prompt["negative_connector_prompt_embeds"]
        )
        images = adapter.encode_image([Image.new("RGB", (64, 64), "red")], height=64, width=64)[
            "condition_images"
        ]
        video, mask = native.prepare_latents(
            image=images * 2 - 1,
            batch_size=1,
            num_channels_latents=128,
            height=64,
            width=64,
            num_frames=17,
            dtype=torch.float32,
            device=torch.device("cpu"),
            generator=torch.Generator().manual_seed(41),
        )
        assert video.shape == (1, 12, 128)
        torch.testing.assert_close(mask, torch.tensor([[1.0] * 4 + [0.0] * 8]), rtol=0, atol=0)
        blue = adapter.encode_image([Image.new("RGB", (64, 64), "blue")], height=64, width=64)[
            "condition_images"
        ]
        other, _ = native.prepare_latents(
            image=blue * 2 - 1,
            batch_size=1,
            num_channels_latents=128,
            height=64,
            width=64,
            num_frames=17,
            dtype=torch.float32,
            device=torch.device("cpu"),
            generator=torch.Generator().manual_seed(41),
        )
        assert not torch.equal(video[:, :4], other[:, :4])
        torch.testing.assert_close(video[:, 4:], other[:, 4:], rtol=0, atol=0)
        decoded = native.vae.decode(
            native._unpack_latents(video, 3, 2, 2, 1, 1), return_dict=False
        )[0]
        mel = native.audio_vae.decode(torch.zeros(1, 8, 8, 16), return_dict=False)[0]
        audio = native.vocoder(mel)
        assert decoded.shape == (1, 3, 17, 64, 64) and audio.shape[:2] == (1, 2)
        assert audio.shape[-1] > 0 and torch.isfinite(audio).all() and torch.isfinite(decoded).all()
        observed.append((prompt, video, decoded, audio))
    for expected, actual in zip(observed[0], observed[1]):
        if isinstance(expected, dict):
            for key in expected:
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        else:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "field,value,message",
    (
        ("num_attention_heads", 30, "num_attention_heads"),
        ("audio_num_attention_heads", 30, "audio_num_attention_heads"),
        ("num_layers", 0, "blocks"),
        ("rope_type", "interleaved", "split rotary"),
        ("patch_size", 2, "unit spatial"),
        ("audio_in_channels", 64, "equal positive"),
    ),
)
def test_ltx_cp_metadata_rejects_unqualified_layouts(field, value, message):
    from flow_factory.models.ltx2._context_parallel import validate_ltx_context_parallel_config

    config = dict(build_transformer().config)
    config[field] = value
    with pytest.raises(ValueError, match=message):
        validate_ltx_context_parallel_config(config, 2, 2)


@pytest.mark.parametrize(
    "invalid", ("batch", "video_divisor", "audio_divisor", "packed", "coords", "mask")
)
def test_ltx_cp_rejects_packed_geometry_before_collectives(monkeypatch, invalid):
    from types import SimpleNamespace

    from flow_factory.models.ltx2 import _context_parallel as owner

    monkeypatch.setattr(owner, "context_parallel_geometry", lambda _: (2, 0))
    monkeypatch.setattr(owner, "tensor_parallel_geometry", lambda _: (2, 0))
    adapter = SimpleNamespace(
        accelerator=object(),
        config=SimpleNamespace(parallelism_args=SimpleNamespace(sequence_parallel=True)),
    )
    kwargs = dict(
        adapter=adapter,
        latents=torch.zeros(1, 20, 128),
        video_tokens=12,
        audio_tokens=8,
        video_shape=(3, 2, 2),
        conditioning_mask=torch.tensor([[1.0] * 4 + [0.0] * 8]),
    )
    owner.validate_ltx_context_parallel_inputs(**kwargs)
    if invalid == "batch":
        kwargs["latents"] = torch.zeros(2, 20, 128)
    elif invalid == "video_divisor":
        kwargs["video_tokens"] = 10
    elif invalid == "audio_divisor":
        kwargs["audio_tokens"] = 6
    elif invalid == "packed":
        kwargs["latents"] = torch.zeros(1, 24, 128)
    elif invalid == "coords":
        kwargs["video_coords"] = torch.zeros(1, 3, 12)
    else:
        kwargs["conditioning_mask"][0, 5] = 1
    with pytest.raises(ValueError, match="LTX2"):
        owner.validate_ltx_context_parallel_inputs(**kwargs)


@pytest.mark.parametrize("ltx23", (False, True))
@pytest.mark.parametrize("conditioned", (False, True))
@pytest.mark.parametrize("mode", ("full", "lora"))
def test_native_ltx_grpo_scheduler_endpoints_match_joint_av_oracle(
    monkeypatch, ltx23, conditioned, mode
):
    from contextlib import nullcontext
    from types import SimpleNamespace

    from ltx2_coupled_observations import assert_objectives, observe_objectives
    from ltx2_cp_fixture import adapter_class, install_lora

    from diffusers import LTX2ImageToVideoPipeline
    from flow_factory.scheduler import FlowMatchEulerDiscreteSDEScheduler
    from flow_factory.trainers.common import context_parallel
    from flow_factory.trainers.rl.grpo import GRPOTrainer
    from tests.trainers.test_coupled_trainer_parity import _replay, _trainer

    module = build_transformer(ltx23)
    if mode == "lora":
        module = install_lora(module, conditioned)
    module.set_attention_backend("native")
    adapter = object.__new__(adapter_class(conditioned))
    adapter.accelerator = SimpleNamespace(device=torch.device("cpu"))
    adapter.pipeline = SimpleNamespace(
        transformer=module,
        scheduler=FlowMatchEulerDiscreteSDEScheduler(),
        vae_spatial_compression_ratio=32,
        vae_temporal_compression_ratio=8,
        transformer_spatial_patch_size=1,
        transformer_temporal_patch_size=1,
        _pack_latents=LTX2ImageToVideoPipeline._pack_latents,
        _unpack_latents=LTX2ImageToVideoPipeline._unpack_latents,
    )
    adapter.audio_scheduler = FlowMatchEulerDiscreteSDEScheduler()
    adapter.component_runtime = SimpleNamespace(get_component=lambda _: module)
    inputs = model_inputs(ltx23, conditioned)
    replayed = _replay({"video": inputs["hidden_states"], "audio": inputs["audio_hidden_states"]})
    replayed.times.timestep = {name: torch.tensor([600.0]) for name in ("video", "audio")}
    replayed.times.next_timestep = {name: torch.tensor([400.0]) for name in ("video", "audio")}
    if conditioned:
        mask = torch.tensor([[1.0] * 4 + [0.0] * 8])
        replayed.state.active_masks = {
            "video": (~mask.bool()).unsqueeze(-1),
            "audio": torch.ones(1, 8, 1, dtype=torch.bool),
        }
        replayed.next_state.components["video"][:, :4] = replayed.state.components["video"][:, :4]

    def forward(batch, replayed, fields):
        return adapter.forward(
            t=torch.tensor([600.0]),
            t_next=torch.tensor([400.0]),
            latents=torch.cat(tuple(replayed.state.components.values()), dim=1),
            next_latents=torch.cat(tuple(replayed.next_state.components.values()), dim=1),
            video_seq_len=12,
            connector_prompt_embeds=inputs["encoder_hidden_states"],
            connector_audio_prompt_embeds=inputs["audio_encoder_hidden_states"],
            connector_attention_mask=inputs["encoder_attention_mask"],
            negative_connector_prompt_embeds=-inputs["encoder_hidden_states"],
            negative_connector_audio_prompt_embeds=-inputs["audio_encoder_hidden_states"],
            negative_connector_attention_mask=inputs["encoder_attention_mask"],
            guidance_scale=2.0,
            audio_guidance_scale=2.5,
            stg_scale=0.1,
            spatio_temporal_guidance_blocks=[0],
            modality_scale=1.1,
            height=64,
            width=64,
            num_frames=17,
            frame_rate=50.0,
            audio_num_frames=8,
            noise_level=0.4,
            compute_log_prob=True,
            return_kwargs=fields,
            use_cross_timestep=ltx23,
            _return_components=True,
            **({"conditioning_mask": mask} if conditioned else {}),
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
        guidance_scale=2.0,
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
    batch = {"advantage": torch.tensor([1.5])}
    trainer._iter_prefetched_batches = lambda samples, size: iter([batch])
    trainer._replay_forward = forward
    adapter.train = lambda: module.train()
    adapter.get_train_step_indices = lambda: [0]
    adapter.get_replay_step = lambda batch, index: replayed
    monkeypatch.setattr(context_parallel, "context_parallel_geometry", lambda _: (2, 0))
    with observe_objectives(trainer) as records:
        trainer.optimize([object()])
    assert_objectives(records, 2)
    assert records[0]["oracle"]["active_elements"] == (8 if conditioned else 12) * 128 + 8 * 128
    gradients = {name: p.grad for name, p in module.named_parameters() if p.requires_grad}
    assert gradients and all(
        value is not None and torch.isfinite(value).all() for value in gradients.values()
    )
    assert any(bool(value.count_nonzero()) for name, value in gradients.items() if "audio" in name)
    assert any(
        bool(value.count_nonzero()) for name, value in gradients.items() if "audio" not in name
    )


@pytest.mark.parametrize("variant", ("ltx20_t2av", "ltx23_i2av"))
def test_ltx_native_guided_rollout_preserves_structured_audio_video(tmp_path, variant, monkeypatch):
    from accelerate import Accelerator
    from ltx2_coupled_fixture import build_fixture, make_config
    from ltx2_cp_fixture import adapter_class
    from PIL import Image

    build_fixture(tmp_path, 2, variant)
    config = make_config(
        tmp_path,
        tmp_path / "result",
        {"dp_size": 1, "cp_size": 1, "tp_size": 1, "training_mode": "full", "algorithm": "grpo"},
        variant,
    )
    adapter = adapter_class(variant.endswith("i2av"))(config, Accelerator(cpu=True))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        samples = adapter.inference(
            prompt=["a red bird sings softly"],
            height=64,
            width=64,
            num_frames=17,
            frame_rate=50.0,
            num_inference_steps=3,
            guidance_scale=2.0,
            generator=torch.Generator().manual_seed(17),
            extra_call_back_kwargs=["velocity", "next_latents_mean"],
            **config.training_args.extra_kwargs,
            **(
                {"images": [Image.open(tmp_path / "data/1-first.png")]}
                if variant.endswith("i2av")
                else {}
            ),
        )
    import ltx2_coupled_fixture as fixture

    monkeypatch.setattr(fixture, "_audit_fresh_replay", lambda trainer, rows: rows)
    assert fixture.audit_samples(None, samples, variant) is samples
    (sample,) = samples
    assert sample.video.shape == (17, 3, 64, 64)
    assert sample.audio.shape == (2, 6960) and sample.audio_sample_rate == 24000
    assert torch.isfinite(sample.video).all() and torch.isfinite(sample.audio).all()
    assert sample.trajectory.component_names == ("video", "audio")
    video, audio = (sample.trajectory.components[name] for name in ("video", "audio"))
    assert video.states.shape == (4, 12, 128) and audio.states.shape == (4, 8, 128)
    assert torch.isfinite(video.states).all() and torch.isfinite(audio.states).all()
    torch.testing.assert_close(video.timesteps, audio.timesteps, rtol=0, atol=0)
    assert not torch.equal(sample.connector_prompt_embeds, sample.negative_connector_prompt_embeds)
    if variant.endswith("i2av"):
        torch.testing.assert_close(
            sample.conditioning_mask, torch.tensor([1.0] * 4 + [0.0] * 8), rtol=0, atol=0
        )
        torch.testing.assert_close(
            video.states[:, :4], video.states[:1, :4].expand_as(video.states[:, :4]), rtol=0, atol=0
        )
        assert torch.count_nonzero(video.active_mask) == 8


@pytest.mark.parametrize("boundary", ("admission", "configure"))
def test_ltx_cp_rejects_last_foreign_processor_before_mutation(boundary):
    from diffusers.models.transformers.transformer_ltx2 import (
        LTX2Attention,
        LTX2AudioVideoAttnProcessor,
    )

    from flow_factory.models.ltx2._context_parallel import (
        configure_ltx_context_parallel,
        validate_ltx_context_parallel_module,
    )

    class ForeignProcessor(LTX2AudioVideoAttnProcessor):
        pass

    module = build_transformer(True)
    module.transformer_blocks[-1].video_to_audio_attn.set_processor(ForeignProcessor())
    before = {
        name: attention.processor
        for name, attention in module.named_modules()
        if isinstance(attention, LTX2Attention)
    }
    with pytest.raises(ValueError, match="native LTX2PerturbedAttnProcessor.*video_to_audio_attn"):
        if boundary == "admission":
            validate_ltx_context_parallel_module(module, 2, 2)
        else:
            configure_ltx_context_parallel(module)
    assert all(
        attention.processor is before[name]
        for name, attention in module.named_modules()
        if isinstance(attention, LTX2Attention)
    )


@pytest.mark.parametrize("ltx23", (False, True))
def test_ltx_cp_admits_native_processors_and_preserves_self_attention(ltx23):
    from flow_factory.models.ltx2._context_parallel import (
        LTX2CrossModalCPProcessor,
        configure_ltx_context_parallel,
        validate_ltx_context_parallel_module,
    )

    module = build_transformer(ltx23)
    before = [block.attn1.processor for block in module.transformer_blocks]
    validate_ltx_context_parallel_module(module, 2, 2)
    configure_ltx_context_parallel(module)
    for block, processor in zip(module.transformer_blocks, before):
        assert block.attn1.processor is processor
        assert type(block.audio_to_video_attn.processor) is LTX2CrossModalCPProcessor
        assert type(block.video_to_audio_attn.processor) is LTX2CrossModalCPProcessor
