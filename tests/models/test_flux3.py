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

"""CPU contract, parity, and replay tests for the FLUX.3 adapter."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List

import numpy as np
import pytest
import torch
from PIL import Image

from diffusers import (
    AutoencoderKLFlux2,
    ClassifierFreeGuidance,
    FlowMatchEulerDiscreteScheduler,
    Flux3Transformer2DModel,
)
from diffusers.modular_pipelines import PipelineState
from diffusers.modular_pipelines.flux3 import (
    Flux3AutoBlocks,
    Flux3DistilledAutoBlocks,
    Flux3DistilledModularPipeline,
    Flux3ModularPipeline,
)
from diffusers.modular_pipelines.flux3.before_denoise import (
    Flux3DistilledTextRoPEInputsStep,
    Flux3PrepareImageCondLatentsStep,
    Flux3PrepareLatentsStep,
    Flux3SetTimestepsStep,
    Flux3TextRoPEInputsStep,
)
from diffusers.modular_pipelines.flux3.decoders import Flux3DecodeStep, Flux3UnpackLatentsStep
from diffusers.modular_pipelines.flux3.denoise import Flux3DenoiseStep, Flux3DistilledDenoiseStep
from diffusers.modular_pipelines.flux3.encoders import (
    Flux3ProcessImagesInputStep,
    Flux3TextEncoderStep,
    Flux3VaeEncoderStep,
)
from diffusers.modular_pipelines.flux3.image_processor import Flux3ImageProcessor
from diffusers.modular_pipelines.flux3.inputs import (
    Flux3DistilledTextInputStep,
    Flux3ImageCondInputStep,
    Flux3TextInputStep,
)
from flow_factory.contracts import (
    BatchCapability,
    GeometrySource,
    InputMediaOrder,
    NegativePromptPolicy,
)
from flow_factory.hparams import Arguments
from flow_factory.models.abc import BaseAdapter
from flow_factory.models.flux._output import encode_flux3_output_images
from flow_factory.models.flux.flux3 import (
    DEFAULT_MAX_SEQUENCE_LENGTH,
    IMAGE_ROPE_SCALE,
    TEXT_PAD_MULTIPLE,
    Flux3Adapter,
    Flux3Sample,
)
from flow_factory.models.registry import get_model_adapter_class
from flow_factory.models.runtime import ModularPipelineRuntime
from flow_factory.samples import BaseSample
from flow_factory.scheduler import FlowMatchEulerDiscreteSDEScheduler


class _FakeInnerTokenizer:
    def __call__(
        self,
        text: str,
        *,
        return_tensors: str = "pt",
        truncation: bool = True,
        max_length: int | None = None,
        padding: str | bool = False,
        padding_side: str = "right",
    ) -> dict[str, torch.Tensor]:
        assert return_tensors == "pt"
        assert truncation is True
        raw_tokens = [min(ord(ch), 250) for ch in text][:max_length]
        if not raw_tokens:
            raw_tokens = [1]
        if padding == "max_length":
            assert padding_side == "right"
            pad_count = max(0, max_length - len(raw_tokens))
            input_ids = raw_tokens + [0] * pad_count
            attention_mask = [1] * len(raw_tokens) + [0] * pad_count
        else:
            input_ids = raw_tokens
            attention_mask = [1] * len(raw_tokens)
        return {
            "input_ids": torch.tensor([input_ids], dtype=torch.long),
            "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
        }


class _FakeProcessor:
    def __init__(self) -> None:
        self.tokenizer = _FakeInnerTokenizer()
        self.chat_calls: list[str] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        content = messages[0]["content"]
        self.chat_calls.append(content)
        return f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"


class _FakeQwen3VL(torch.nn.Module):
    def __init__(self, hidden_dim: int = 4, num_layers: int = 33) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dummy_param = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.calls: list[dict[str, Any]] = []

    @property
    def dtype(self) -> torch.dtype:
        return self.dummy_param.dtype

    @property
    def device(self) -> torch.device:
        return self.dummy_param.device

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        output_hidden_states: bool = True,
        use_cache: bool = False,
    ) -> Any:
        assert output_hidden_states is True
        assert use_cache is False
        self.calls.append(
            {
                "input_ids": input_ids.clone(),
                "attention_mask": attention_mask.clone(),
                "position_ids": position_ids.clone(),
            }
        )
        batch_size, seq_len = input_ids.shape
        base = (
            input_ids.to(torch.float32) * attention_mask.to(torch.float32)
            + position_ids.to(torch.float32) * 0.01
        )
        base = base.unsqueeze(-1).repeat(1, 1, self.hidden_dim)
        hidden_states = tuple(base + float(layer_idx) * 0.1 for layer_idx in range(self.num_layers))
        return SimpleNamespace(hidden_states=hidden_states)


def _build_tiny_flux3_components(*, flow_map: bool = False) -> dict[str, Any]:
    torch.manual_seed(0)
    transformer = Flux3Transformer2DModel(
        in_channels=16,
        cond_in_channels=16,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=16,
        timestep_channels=32,
        axes_dims_rope=(2, 2, 2, 2),
        flow_map=flow_map,
    ).eval()
    vae = AutoencoderKLFlux2(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels=(8, 16),
        layers_per_block=1,
        latent_channels=4,
        norm_num_groups=4,
    ).eval()
    image_processor = Flux3ImageProcessor(vae_scale_factor=4, vae_latent_channels=4)
    scheduler = FlowMatchEulerDiscreteScheduler(shift=6.93, use_dynamic_shifting=False)
    text_encoder = _FakeQwen3VL(hidden_dim=2, num_layers=33).eval()
    tokenizer = _FakeProcessor()

    blocks_cls = Flux3DistilledAutoBlocks if flow_map else Flux3AutoBlocks
    pipeline: Flux3ModularPipeline = blocks_cls().init_pipeline()
    update_kwargs: dict[str, Any] = {
        "transformer": transformer,
        "vae": vae,
        "image_processor": image_processor,
        "scheduler": scheduler,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
    }
    if not flow_map:
        update_kwargs["guider"] = ClassifierFreeGuidance(guidance_scale=3.5)
    pipeline.update_components(**update_kwargs)
    return {
        "pipeline": pipeline,
        "transformer": transformer,
        "vae": vae,
        "image_processor": image_processor,
        "scheduler": scheduler,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
    }


def _build_flux3_adapter(*, flow_map: bool = False, noise_level: float = 0.0) -> Flux3Adapter:
    bundle = _build_tiny_flux3_components(flow_map=flow_map)
    adapter = object.__new__(Flux3Adapter)
    adapter.config = SimpleNamespace(
        mixed_precision="no",
        scheduler_args=SimpleNamespace(
            noise_level=noise_level,
            sde_steps=None,
            num_sde_steps=None,
            seed=42,
            dynamics_type="Flow-SDE" if noise_level > 0 else "ODE",
        ),
    )
    adapter.model_args = SimpleNamespace(
        model_name_or_path="tiny-flux3",
        target_components=["transformer"],
        target_modules="default",
        finetune_type="lora",
        trainable_parameters_dtype=torch.float32,
        component_load_dtypes=None,
        frozen_parameters_dtype=None,
    )
    adapter.training_args = SimpleNamespace(
        height=32,
        width=32,
        latent_storage_dtype=None,
        enable_gradient_checkpointing=False,
    )
    adapter.eval_args = SimpleNamespace()
    adapter.accelerator = SimpleNamespace(
        device=torch.device("cpu"),
        is_main_process=True,
        is_local_main_process=True,
        process_index=0,
        distributed_type=None,
        mixed_precision="no",
        unwrap_model=lambda model: model,
        wait_for_everyone=lambda: None,
    )
    adapter._mode = "eval" if noise_level == 0.0 else "rollout"
    adapter._component_load_dtype_manifest = Flux3Adapter.component_load_dtype_defaults
    adapter._component_load_dtype_overrides = None
    adapter.component_runtime = ModularPipelineRuntime.from_adapter(adapter, bundle["pipeline"])
    adapter.pipeline = bundle["pipeline"]
    adapter._components = adapter.component_runtime.override_components
    adapter.pipeline.scheduler = FlowMatchEulerDiscreteSDEScheduler(
        shift=6.93,
        use_dynamic_shifting=False,
        noise_level=noise_level,
        dynamics_type="Flow-SDE" if noise_level > 0 else "ODE",
        seed=42,
    )
    adapter.scheduler_group = adapter.build_scheduler_group()
    adapter._effective_pipeline_io_contract = adapter._resolve_pipeline_io_contract()
    adapter._condition_state_preparer = adapter._build_condition_state_preparer_declaration()
    adapter._condition_state_encoding_modules = (
        adapter._validate_condition_state_preparer_lifecycle()
    )
    adapter._output_state_codec = adapter._build_output_state_codec_declaration()
    adapter._output_state_encoding_modules = adapter._validate_output_state_codec_lifecycle()
    adapter._has_warned_inference_fallback = False
    adapter._has_warned_forward_fallback = False
    return adapter


def test_registry_inheritance_and_io_contracts() -> None:
    assert get_model_adapter_class("flux3") is Flux3Adapter
    Flux3Adapter.validate_offline_output_capability()
    assert BaseAdapter in Flux3Adapter.__bases__
    assert not any(
        base.__module__.startswith("flow_factory.models.flux.") and base is not Flux3Adapter
        for base in Flux3Adapter.__mro__
    )
    assert not Flux3Adapter.supports_diffusers_cache
    assert Flux3Adapter.supports_fsdp2_cpu_efficient_loading
    assert Flux3Adapter.ddp_find_unused_parameters

    base_adapter = _build_flux3_adapter(flow_map=False)
    assert not base_adapter.is_distilled
    base_contract = base_adapter.effective_pipeline_io_contract
    assert base_contract is not None
    assert base_contract.negative_prompt is NegativePromptPolicy.OPTIONAL
    assert base_contract.geometry_source is GeometrySource.CONFIGURED
    assert base_contract.batch_capability is BatchCapability.UNIFORM
    assert base_contract.input_media.order is InputMediaOrder.WITHIN_TYPE

    distilled_adapter = _build_flux3_adapter(flow_map=True)
    assert distilled_adapter.is_distilled
    distilled_contract = distilled_adapter.effective_pipeline_io_contract
    assert distilled_contract is not None
    assert distilled_contract.negative_prompt is NegativePromptPolicy.UNSUPPORTED
    with pytest.raises(ValueError, match="does not support negative_prompt"):
        distilled_adapter.encode_prompt("a fox", negative_prompt="blurry", guidance_scale=1.0)


def test_grpo_example_parses_through_production_config() -> None:
    root = Path(__file__).resolve().parents[2]
    config = Arguments.load_from_yaml(str(root / "examples/grpo/lora/flux3/default.yaml"))

    assert config.model_args.model_type == "flux3"
    assert config.model_args.model_name_or_path == "black-forest-labs/flux-3-image-klein-base"
    assert config.training_args.trainer_type == "grpo"
    assert config.training_args.guidance_scale == 3.5
    assert config.eval_args.guidance_scale == 3.5


def test_prompt_encoding_enforces_80_token_windows_and_pad_position_ids() -> None:
    adapter = _build_flux3_adapter(flow_map=False)
    encoded = adapter.encode_prompt(
        prompt="a red fox",
        guidance_scale=3.5,
        text_encoder_out_layers=(4, 8, 12, 16, 20, 24, 28, 32),
        pad_multiple=TEXT_PAD_MULTIPLE,
    )

    assert isinstance(encoded["prompt_ids"], torch.Tensor)
    assert isinstance(encoded["prompt_embeds"], torch.Tensor)
    assert isinstance(encoded["text_ids"], torch.Tensor)
    assert encoded["prompt_ids"].shape == (1, 80)
    assert encoded["prompt_embeds"].shape == (1, 80, 16)
    assert encoded["text_ids"].shape == (1, 80, 4)
    assert encoded["negative_prompt_embeds"].shape == (1, 80, 16)
    assert encoded["negative_text_ids"].shape == (1, 80, 4)

    text_encoder: _FakeQwen3VL = adapter.text_encoder  # type: ignore[assignment]
    first_call = text_encoder.calls[0]
    pad_positions = first_call["position_ids"][first_call["attention_mask"] == 0]
    assert pad_positions.numel() > 0
    assert torch.all(pad_positions == 1)

    # Variable-length prompts with different 80-token windows remain un-merged so shorter
    # prompts are never polluted with extra unmasked padding tokens.
    short_prompt = "fox"
    long_prompt = "word " * 15  # 75 chars + chat template -> (80, 160] window
    ragged = adapter.encode_prompt(
        prompt=[short_prompt, long_prompt],
        guidance_scale=1.0,
        text_encoder_out_layers=(4, 8, 12, 16, 20, 24, 28, 32),
    )
    assert isinstance(ragged["prompt_embeds"], list)
    assert ragged["prompt_embeds"][0].shape == (80, 16)
    assert ragged["prompt_embeds"][1].shape == (160, 16)


def test_default_max_sequence_length_matches_upstream_and_keeps_long_prompts() -> None:
    upstream_default = next(
        param.default
        for param in Flux3TextEncoderStep().inputs
        if param.name == "max_sequence_length"
    )
    assert DEFAULT_MAX_SEQUENCE_LENGTH == upstream_default == 8192

    adapter = _build_flux3_adapter(flow_map=False)
    # The fake tokenizer emits one token per character.
    encoded = adapter.encode_prompt(prompt="x" * 600, guidance_scale=1.0)

    text_encoder: _FakeQwen3VL = adapter.text_encoder  # type: ignore[assignment]
    real_length = int(text_encoder.calls[-1]["attention_mask"].sum())
    assert real_length > 600
    expected_window = math.ceil(real_length / TEXT_PAD_MULTIPLE) * TEXT_PAD_MULTIPLE
    assert encoded["prompt_embeds"].shape == (1, expected_window, 16)
    assert encoded["text_ids"].shape == (1, expected_window, 4)

    # Raw-prompt rollouts (e.g. enable_preprocess: false) encode through inference().
    adapter.inference(
        prompt=["x" * 600],
        height=16,
        width=16,
        num_inference_steps=1,
        guidance_scale=1.0,
        generator=torch.Generator().manual_seed(0),
        compute_log_prob=False,
    )
    assert int(text_encoder.calls[-1]["attention_mask"].sum()) == real_length


def test_condition_image_encoding_assigns_negative_l_coords_and_preserves_empty_slots() -> None:
    adapter = _build_flux3_adapter(flow_map=False)
    ref0 = Image.new("RGB", (16, 8), (120, 40, 200))
    ref1 = Image.new("RGB", (8, 8), (10, 180, 90))

    encoded = adapter.encode_image([[], [ref0, ref1]], condition_image_size=(64, 64))

    assert encoded["condition_images"][0] == []
    assert encoded["image_latents"][0] is None
    assert encoded["image_latent_ids"][0] is None

    # Tiny VAE has scale_factor=2, so 2x2 patchify grid is 4px:
    # ref0 (16x8) -> 4x2 = 8 tokens at l=-1; ref1 (8x8) -> 2x2 = 4 tokens at l=-2.
    latents_1 = encoded["image_latents"][1]
    ids_1 = encoded["image_latent_ids"][1]
    assert isinstance(latents_1, torch.Tensor)
    assert isinstance(ids_1, torch.Tensor)
    assert latents_1.shape == (12, 16)
    assert ids_1.shape == (12, 4)
    assert torch.all(ids_1[:8, 3] == -1.0)
    assert torch.all(ids_1[8:, 3] == -2.0)
    assert torch.allclose(ids_1[:8, 1].max(), torch.tensor(1.0 * IMAGE_ROPE_SCALE))
    assert torch.allclose(ids_1[:8, 2].max(), torch.tensor(3.0 * IMAGE_ROPE_SCALE))


def test_output_codec_and_decode_latents_round_trip() -> None:
    adapter = _build_flux3_adapter(flow_map=False)
    pixels = torch.zeros(2, 3, 16, 16)
    encoded = encode_flux3_output_images(
        adapter,
        pixels,
        torch.Generator().manual_seed(7),
        image_rope_scale=IMAGE_ROPE_SCALE,
    )
    assert encoded.latents.shape == (2, 16, 16)
    assert encoded.forward_context["latent_ids"].shape == (2, 16, 4)

    decoded_from_hw = adapter.decode_latents(
        encoded.latents,
        height=16,
        width=16,
        output_type="pt",
    )
    decoded_from_ids = adapter.decode_latents(
        encoded.latents,
        latent_ids=encoded.forward_context["latent_ids"],
        output_type="pt",
    )
    assert decoded_from_hw.shape == (2, 3, 16, 16)
    torch.testing.assert_close(decoded_from_hw, decoded_from_ids, rtol=0, atol=0)


@pytest.mark.parametrize("flow_map", [False, True])
def test_end_to_end_parity_against_diffusers_modular_blocks(flow_map: bool) -> None:
    """Verify exact numerical parity between Flux3Adapter.inference() and Diffusers blocks."""
    adapter = _build_flux3_adapter(flow_map=flow_map, noise_level=0.0)
    pipe = adapter.pipeline
    height, width = 16, 16
    num_inference_steps = 3
    guidance_scale = 1.0 if flow_map else 3.5
    layers = (4, 8, 12, 16, 20, 24, 28, 32)
    ref_images = [
        Image.fromarray(np.full((16, 8, 3), 80, dtype=np.uint8)),
        Image.fromarray(np.full((8, 8, 3), 160, dtype=np.uint8)),
    ]

    # 1. Encode prompts with adapter
    prompt_enc = adapter.encode_prompt(
        prompt="a snowy forest",
        negative_prompt="" if not flow_map else None,
        guidance_scale=guidance_scale,
        max_sequence_length=64,
        text_encoder_out_layers=layers,
    )

    # 2. Run official Diffusers modular blocks on the same components & seed
    ref_pipe = (Flux3DistilledAutoBlocks if flow_map else Flux3AutoBlocks)().init_pipeline()
    ref_update = {
        "transformer": adapter.transformer,
        "vae": adapter.vae,
        "image_processor": pipe.image_processor,
        "scheduler": FlowMatchEulerDiscreteScheduler(shift=6.93, use_dynamic_shifting=False),
        "text_encoder": adapter.text_encoder,
        "tokenizer": adapter.processor,
    }
    if not flow_map:
        ref_update["guider"] = ClassifierFreeGuidance(guidance_scale=guidance_scale)
    ref_pipe.update_components(**ref_update)

    state = PipelineState(
        values={
            "prompt_embeds": prompt_enc["prompt_embeds"].clone(),
            "negative_prompt_embeds": (
                None if flow_map else prompt_enc["negative_prompt_embeds"].clone()
            ),
            "image": ref_images,
            "height": height,
            "width": width,
            "num_images_per_prompt": 1,
            "generator": torch.Generator().manual_seed(123),
            "num_inference_steps": num_inference_steps,
            "timesteps": None,
            "sigmas": None,
            "latents": None,
            "attention_kwargs": None,
            "output_type": "pt",
        }
    )
    steps: List[Any] = [
        Flux3ProcessImagesInputStep(),
        Flux3VaeEncoderStep(),
        Flux3DistilledTextInputStep() if flow_map else Flux3TextInputStep(),
        Flux3ImageCondInputStep(default_height=height, default_width=width),
        Flux3PrepareLatentsStep(
            default_height=height,
            default_width=width,
            image_rope_scale=IMAGE_ROPE_SCALE,
        ),
        Flux3PrepareImageCondLatentsStep(image_rope_scale=IMAGE_ROPE_SCALE),
        Flux3SetTimestepsStep(shift=6.93, num_inference_steps=num_inference_steps),
        Flux3DistilledTextRoPEInputsStep() if flow_map else Flux3TextRoPEInputsStep(),
        Flux3DistilledDenoiseStep() if flow_map else Flux3DenoiseStep(),
        Flux3UnpackLatentsStep(),
        Flux3DecodeStep(),
    ]
    for step in steps:
        ref_pipe, state = step(ref_pipe, state)
    expected_images = state.values["images"]

    # 3. Run Flux3Adapter.inference() with the same seed and condition images
    samples = adapter.inference(
        images=[ref_images],
        prompt=["a snowy forest"],
        negative_prompt=[""] if not flow_map else None,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        condition_image_size=(64, 64),
        max_sequence_length=64,
        text_encoder_out_layers=layers,
        generator=torch.Generator().manual_seed(123),
        compute_log_prob=False,
    )

    assert len(samples) == 1
    torch.testing.assert_close(samples[0].image, expected_images[0], rtol=0, atol=1e-6)


def test_sde_rollout_replay_parity_and_gradient_flow() -> None:
    adapter = _build_flux3_adapter(flow_map=False, noise_level=0.7)
    adapter.transformer.train()
    ref_image = Image.fromarray(np.full((8, 8, 3), 100, dtype=np.uint8))

    samples = adapter.inference(
        images=[[ref_image]],
        prompt=["relight <ref_image_0>"],
        height=16,
        width=16,
        num_inference_steps=3,
        guidance_scale=2.0,
        condition_image_size=(64, 64),
        max_sequence_length=64,
        text_encoder_out_layers=(4, 8, 12, 16, 20, 24, 28, 32),
        generator=torch.Generator().manual_seed(99),
        compute_log_prob=True,
    )
    sample = samples[0]
    assert isinstance(sample, Flux3Sample)
    assert sample.all_latents is not None
    assert sample.log_probs is not None
    assert sample.all_latents.shape == (4, 16, 16)
    assert sample.log_probs.shape == (2,)

    batch = BaseSample.stack(samples)
    step_idx = 0
    t = batch["timesteps"][:, step_idx]
    t_next = batch["timesteps"][:, step_idx + 1]
    latents = batch["all_latents"][:, step_idx]
    next_latents = batch["all_latents"][:, step_idx + 1]
    noise_level = adapter.scheduler.get_noise_level_for_timestep(t[0])

    replay = adapter.forward(
        t=t,
        t_next=t_next,
        latents=latents,
        next_latents=next_latents,
        latent_ids=batch["latent_ids"],
        prompt_embeds=batch["prompt_embeds"],
        text_ids=batch["text_ids"],
        image_latents=batch["image_latents"],
        image_latent_ids=batch["image_latent_ids"],
        negative_prompt_embeds=batch["negative_prompt_embeds"],
        negative_text_ids=batch["negative_text_ids"],
        guidance_scale=2.0,
        noise_level=noise_level,
        compute_log_prob=True,
    )

    assert replay.log_prob is not None
    torch.testing.assert_close(
        replay.log_prob.detach(),
        batch["log_probs"][:, step_idx],
        rtol=0,
        atol=1e-6,
    )
    ratio = torch.exp(replay.log_prob - batch["log_probs"][:, step_idx])
    torch.testing.assert_close(ratio.detach(), torch.ones_like(ratio), rtol=0, atol=1e-6)

    loss = -replay.log_prob.mean()
    loss.backward()

    assert adapter.transformer.txt_mode_blocks[0].attn.to_q.weight.grad is not None
    assert adapter.transformer.mode_blocks["image"][0].attn.to_q.weight.grad is not None
    assert adapter.transformer.mode_blocks["image_cond"][0].attn.to_q.weight.grad is not None
    assert adapter.transformer.single_transformer_blocks[0].attn.to_q.weight.grad is not None


def test_gradient_checkpointing_order_and_lora_round_trip(tmp_path: Path) -> None:
    adapter = _build_flux3_adapter(flow_map=False, noise_level=0.0)
    units = adapter._gradient_checkpointing_units("transformer", adapter.transformer)
    unit_names = [name for name, _ in units]
    assert unit_names == [
        "txt_mode_blocks.0",
        "mode_blocks.image.0",
        "mode_blocks.image_cond.0",
        "single_transformer_blocks.0",
    ]

    adapter.model_args.lora_rank = 4
    adapter.model_args.lora_alpha = 8
    adapter.target_module_map = {"transformer": adapter.default_target_modules}
    adapter.apply_lora(
        target_modules=adapter.default_target_modules,
        components=["transformer"],
    )

    trainable_params = adapter.get_trainable_parameters()
    assert len(trainable_params) > 0
    with torch.no_grad():
        for param in trainable_params:
            param.add_(0.05)
    expected_snapshot = [param.detach().clone() for param in trainable_params]

    ckpt_dir = tmp_path / "flux3_lora_ckpt"
    adapter.save_checkpoint(str(ckpt_dir), save_ema=False, model_only=True)

    with torch.no_grad():
        for param in trainable_params:
            param.zero_()
    adapter.load_checkpoint(str(ckpt_dir), resume_type="lora")

    restored_params = adapter.get_trainable_parameters()
    for expected, restored in zip(expected_snapshot, restored_params, strict=True):
        torch.testing.assert_close(restored, expected, rtol=0, atol=0)


def test_offline_sft_forward_state_and_load_plan_contracts() -> None:
    from flow_factory.loading import ComponentRole, build_adapter_load_plan

    adapter = _build_flux3_adapter(flow_map=False, noise_level=0.0)
    adapter.training_args.height = 16
    adapter.training_args.width = 16

    # 1. Verify ModelLoadCoordinator plan partitions TARGET, HOST, and AUXILIARY
    plan = build_adapter_load_plan(adapter)
    assert plan.request_for_component("transformer").role is ComponentRole.TARGET
    assert plan.request_for_component("tokenizer").role is ComponentRole.HOST
    assert plan.request_for_component("scheduler").role is ComponentRole.HOST
    assert plan.request_for_component("text_encoder").role is ComponentRole.AUXILIARY
    assert plan.request_for_component("vae").role is ComponentRole.AUXILIARY
    assert plan.request_for_component("image_processor").role is ComponentRole.HOST

    # 2. Verify offline output-state codec + forward_state velocity matching
    target_image = Image.new("RGB", (16, 16), (60, 120, 180))
    media_batch = (
        (SimpleNamespace(type="image", payload=target_image, fps=None, sample_rate=None),),
    )
    prompt_enc = adapter.encode_prompt(
        prompt="a castle",
        guidance_scale=1.0,
        max_sequence_length=64,
        text_encoder_out_layers=(4, 8, 12, 16, 20, 24, 28, 32),
    )
    condition = {
        "prompt_embeds": prompt_enc["prompt_embeds"],
        "text_ids": prompt_enc["text_ids"],
    }
    prepared_condition = adapter.prepare_condition_state(condition)
    encoded_output = adapter.encode_output_state(
        media_batch,
        prepared_condition,
        torch.Generator().manual_seed(17),
    )

    times = adapter.build_training_component_times(torch.tensor([500.0]))
    noised = adapter.add_forward_process_noise(
        encoded_output.clean_state,
        times,
        generator=torch.Generator().manual_seed(23),
    )
    forward_kwargs = {
        **prepared_condition.condition,
        **prepared_condition.forward_context,
        **encoded_output.forward_context,
        **dict(adapter.offline_training_forward_overrides),
    }
    step_out = adapter.forward_state(
        batch=forward_kwargs,
        state=noised.state,
        times=times,
        compute_log_prob=False,
        return_fields=("velocity",),
    )
    assert step_out.velocity is not None
    per_elem_sq_error = {
        name: (step_out.velocity.components[name] - noised.target_velocity.components[name]).pow(2)
        for name in adapter.trajectory_component_order
    }
    reduced = adapter.reduce_flow_matching_objective_values(per_elem_sq_error, state=noised.state)
    assert reduced.shape == (1,)
    assert torch.isfinite(reduced).all()

    # 3. Verify Flux3Sample reconstruction_required_fields
    sample = Flux3Sample(
        prompt="a castle",
        image=torch.zeros(3, 16, 16),
        height=16,
        width=16,
        latent_ids=torch.zeros(16, 4),
        text_ids=torch.zeros(80, 4),
    )
    reconstructed = Flux3Sample(
        **{
            field_name: getattr(sample, field_name)
            for field_name in Flux3Sample.reconstruction_required_fields
            if hasattr(sample, field_name)
        }
    )
    assert isinstance(reconstructed, Flux3Sample)
