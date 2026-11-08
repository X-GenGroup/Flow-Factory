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

"""Native Flux3 base/flow-map components, packed weights and real image operands."""

from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import torch

from diffusers import Flux3Transformer2DModel
from flow_factory.models.flux.flux3 import Flux3Adapter
from flow_factory.models.runtime import ModularPipelineRuntime
from flow_factory.scheduler import FlowMatchEulerDiscreteSDEScheduler, set_scheduler_timesteps

VARIANTS = ("base", "distilled")
wrap_classes = ("Flux3ModeTransformerBlock", "Flux3SingleTransformerBlock")


def build_transformer(variant):
    """Retain published24 heads,128 packed channels, four axes and SwiGLU ratio3."""
    torch.manual_seed(813)
    return Flux3Transformer2DModel(
        in_channels=128,
        cond_in_channels=128,
        num_layers=2,
        num_single_layers=2,
        attention_head_dim=8,
        num_attention_heads=24,
        joint_attention_dim=64,
        timestep_channels=16,
        mlp_ratio=3.0,
        axes_dims_rope=(2, 2, 2, 2),
        flow_map=variant == "distilled",
    )


def install_lora(model, variant=None, *, extra_targets=()):
    from peft import LoraConfig, get_peft_model

    targets = [
        *Flux3Adapter.default_target_modules.fget(object.__new__(Flux3Adapter)),
        *extra_targets,
    ]
    torch.manual_seed(22)
    return get_peft_model(
        model, LoraConfig(r=2, lora_alpha=2, target_modules=targets, init_lora_weights=False)
    )


def build_pipeline(variant):
    from diffusers.modular_pipelines.flux3 import Flux3AutoBlocks, Flux3DistilledAutoBlocks
    from diffusers.modular_pipelines.flux3.image_processor import Flux3ImageProcessor
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import (
        PreTrainedTokenizerFast,
        Qwen2VLImageProcessor,
        Qwen3VLConfig,
        Qwen3VLForConditionalGeneration,
        Qwen3VLProcessor,
        Qwen3VLTextConfig,
        Qwen3VLVideoProcessor,
        Qwen3VLVisionConfig,
    )

    from diffusers import (
        AutoencoderKLFlux2,
        ClassifierFreeGuidance,
        FlowMatchEulerDiscreteScheduler,
    )

    torch.manual_seed(719)
    words = [
        "<pad>",
        "</s>",
        "<unk>",
        "a",
        "red",
        "blue",
        "image",
        "small",
        "<|image_pad|>",
        "<|video_pad|>",
        "<|vision_start|>",
        "<|vision_end|>",
    ]
    core = Tokenizer(
        WordLevel({word: index for index, word in enumerate(words)}, unk_token="<unk>")
    )
    core.pre_tokenizer = Whitespace()
    template = "{% for message in messages %}{{ message['content'] }}{% endfor %} </s>"
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=core,
        pad_token="<pad>",
        bos_token="</s>",
        eos_token="</s>",
        unk_token="<unk>",
        model_max_length=512,
        chat_template=template,
        additional_special_tokens=words[8:],
    )
    processor = Qwen3VLProcessor(
        tokenizer=tokenizer,
        image_processor=Qwen2VLImageProcessor(min_pixels=32 * 32, max_pixels=64 * 64),
        video_processor=Qwen3VLVideoProcessor(),
        chat_template=template,
    )
    text_encoder = Qwen3VLForConditionalGeneration(
        Qwen3VLConfig(
            text_config=Qwen3VLTextConfig(
                vocab_size=len(words),
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=32,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=4,
                max_position_embeddings=512,
                pad_token_id=0,
                rope_parameters={
                    "rope_type": "default",
                    "rope_theta": 10000.0,
                    "mrope_section": [1, 1, 0],
                },
            ),
            vision_config=Qwen3VLVisionConfig(
                depth=1,
                hidden_size=8,
                intermediate_size=16,
                num_heads=2,
                patch_size=8,
                spatial_merge_size=2,
                temporal_patch_size=2,
                out_hidden_size=8,
                num_position_embeddings=64,
                deepstack_visual_indexes=[],
            ),
            image_token_id=8,
            video_token_id=9,
            vision_start_token_id=10,
            vision_end_token_id=11,
        )
    ).eval()
    vae = AutoencoderKLFlux2(
        block_out_channels=(8, 8, 8, 8),
        layers_per_block=1,
        latent_channels=32,
        norm_num_groups=4,
        sample_size=64,
    ).eval()
    vae.bn.running_mean.copy_(torch.linspace(-0.2, 0.3, 128))
    vae.bn.running_var.copy_(torch.linspace(0.7, 1.3, 128))
    blocks = Flux3DistilledAutoBlocks if variant == "distilled" else Flux3AutoBlocks
    pipeline = blocks().init_pipeline()
    components = dict(
        transformer=build_transformer(variant),
        vae=vae,
        image_processor=Flux3ImageProcessor(vae_scale_factor=16, vae_latent_channels=32),
        scheduler=FlowMatchEulerDiscreteScheduler(shift=6.93, use_dynamic_shifting=False),
        text_encoder=text_encoder,
        tokenizer=processor,
    )
    if variant == "base":
        components["guider"] = ClassifierFreeGuidance(guidance_scale=2.0)
    pipeline.update_components(**components)
    return pipeline


def make_adapter(model, variant):
    from diffusers.modular_pipelines.flux3 import Flux3AutoBlocks, Flux3DistilledAutoBlocks

    blocks = Flux3DistilledAutoBlocks if variant == "distilled" else Flux3AutoBlocks
    pipeline = blocks().init_pipeline()
    pipeline.update_components(transformer=model)
    return pipeline_adapter(pipeline, variant)


def pipeline_adapter(pipeline, variant):
    adapter = object.__new__(Flux3Adapter)
    adapter.config = SimpleNamespace(
        mixed_precision="no",
        parallelism_args=SimpleNamespace(
            sequence_parallel=True,
            effective_context_parallel_size=1,
            effective_tensor_parallel_size=1,
        ),
        scheduler_args=SimpleNamespace(
            noise_level=0.35, sde_steps=None, num_sde_steps=None, seed=42, dynamics_type="Flow-SDE"
        ),
    )
    adapter.model_args = SimpleNamespace(
        model_name_or_path="tiny-flux3",
        target_components=["transformer"],
        target_modules="default",
        finetune_type="full",
        trainable_parameters_dtype=torch.float32,
        component_load_dtypes=None,
        frozen_parameters_dtype=None,
    )
    adapter.training_args = SimpleNamespace(
        height=64, width=64, latent_storage_dtype=None, enable_gradient_checkpointing=False
    )
    adapter.eval_args = SimpleNamespace()
    adapter.accelerator = SimpleNamespace(
        device=torch.device("cpu"),
        num_processes=1,
        process_index=0,
        is_main_process=True,
        is_local_main_process=True,
        distributed_type=None,
        mixed_precision="no",
        unwrap_model=lambda model: model,
        wait_for_everyone=lambda: None,
    )
    adapter._mode = "rollout"
    adapter._component_load_dtype_manifest = Flux3Adapter.component_load_dtype_defaults
    adapter._component_load_dtype_overrides = None
    adapter.pipeline = pipeline
    adapter.component_runtime = ModularPipelineRuntime.from_adapter(adapter, pipeline)
    adapter._components = adapter.component_runtime.override_components
    adapter.pipeline.scheduler = FlowMatchEulerDiscreteSDEScheduler(
        shift=6.93, use_dynamic_shifting=False, noise_level=0.35, dynamics_type="Flow-SDE", seed=42
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
    set_scheduler_timesteps(adapter.scheduler, num_inference_steps=3, seq_len=16, device="cpu")
    return adapter


def fixed_inputs(variant, replica=0, step=0, device="cpu", dtype=torch.float32):
    generator = torch.Generator().manual_seed(401 + step + replica * 31)

    def operand(*shape):
        return torch.randn(*shape, generator=generator, dtype=dtype).to(device)

    target = operand(1, 16, 128)
    values = dict(
        latents=target,
        image_latents=operand(1, 8, 128),
        latent_ids=operand(1, 16, 4),
        image_latent_ids=operand(1, 8, 4),
        prompt_embeds=operand(1, 80, 64),
        text_ids=operand(1, 80, 4),
        t=torch.tensor([600.0], dtype=dtype, device=device),
        t_next=torch.tensor([400.0], dtype=dtype, device=device),
        next_latents=target + 0.137,
        guidance_scale=2.0 if variant == "base" else 1.0,
    )
    if variant == "base":
        values.update(
            negative_prompt_embeds=operand(1, 80, 64), negative_text_ids=operand(1, 80, 4)
        )
    return values


def native_velocity(model, inputs, variant):
    """Independent native per-mode conditioning, flow-map target time and CFG."""
    args = dict(
        hidden_states=inputs["latents"],
        encoder_hidden_states=inputs["prompt_embeds"],
        img_ids=inputs["latent_ids"][0],
        txt_ids=inputs["text_ids"][0],
        timestep=inputs["t"] / 1000,
        image_cond_hidden_states=inputs.get("image_latents"),
        image_cond_ids=(
            inputs["image_latent_ids"][0] if inputs.get("image_latents") is not None else None
        ),
        target_timestep=inputs["t_next"] / 1000 if variant == "distilled" else None,
        return_dict=False,
    )
    conditional = model(**args)[0]
    if variant == "distilled":
        return conditional
    negative = model(
        **{
            **args,
            "encoder_hidden_states": inputs["negative_prompt_embeds"],
            "txt_ids": inputs["negative_text_ids"][0],
        }
    )[0]
    return negative + inputs["guidance_scale"] * (conditional - negative)


def build_fixture(root, variant):
    import numpy as np
    from PIL import Image

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    build_pipeline(variant).save_pretrained(str(root / "model"), overwrite_modular_index=True)
    if variant == "base":
        # Published base metadata omits flow_map; native loading must retain its false default.
        import json

        config_path = root / "model" / "transformer" / "config.json"
        config = json.loads(config_path.read_text())
        assert config.pop("flow_map") is False
        config_path.write_text(json.dumps(config, indent=2))
    data = root / "data"
    data.mkdir(exist_ok=True)
    for index in range(8):
        pixels = np.random.default_rng(index).integers(0, 256, (64, 128, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(data / f"{index}.png")


def load_pipeline(root, variant):
    from diffusers import ModularPipeline

    pipeline = ModularPipeline.from_pretrained(str(Path(root) / "model"))
    pipeline.load_components(torch_dtype=torch.float32)
    return pipeline


@lru_cache(maxsize=4)
def encoded_conditions(root, variant, replica):
    from PIL import Image

    pipeline = load_pipeline(root, variant)
    adapter = pipeline_adapter(pipeline, variant)
    with torch.no_grad():
        options = (
            {"negative_prompt": "blue", "guidance_scale": 2.0}
            if variant == "base"
            else {"guidance_scale": 1.0}
        )
        prompt = adapter.encode_prompt(
            "a red image" if replica else "a blue image", max_sequence_length=80, **options
        )
        image = adapter.encode_image(
            Image.open(Path(root) / "data" / f"{replica}.png"), condition_image_size=(32, 64)
        )
        _, ids = adapter._prepare_latents(
            1,
            128,
            64,
            64,
            vae_scale_factor=8,
            dtype=torch.float32,
            device=torch.device("cpu"),
            generator=torch.Generator().manual_seed(52),
        )
    return prompt, image, ids


def model_inputs(root, variant, replica, step, device):
    prompt, image, ids = encoded_conditions(str(root), variant, replica)
    values = fixed_inputs(variant, replica, step)
    values.update({key: value for key, value in prompt.items() if key in values})
    values.update(
        latent_ids=ids,
        image_latents=image["image_latents"][0].unsqueeze(0),
        image_latent_ids=image["image_latent_ids"][0].unsqueeze(0),
    )
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in values.items()
    }


if __name__ == "__main__":
    import sys

    build_fixture(Path(sys.argv[1]), sys.argv[2])
