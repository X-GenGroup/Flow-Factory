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

"""Native Flux2/Klein layouts with canonical packed weights and reference operands."""

from __future__ import annotations

from functools import lru_cache
from types import SimpleNamespace

import torch

from diffusers import Flux2Transformer2DModel
from flow_factory.models.flux.flux2 import Flux2Adapter
from flow_factory.models.flux.flux2_klein import Flux2KleinAdapter
from flow_factory.models.runtime.classic import ClassicPipelineRuntime
from flow_factory.scheduler import FlowMatchEulerDiscreteSDEScheduler, set_scheduler_timesteps

VARIANTS = {"dev": 48, "klein4b": 24, "klein9b": 32}
wrap_classes = ("Flux2TransformerBlock", "Flux2SingleTransformerBlock")


def adapter_class(variant):
    return Flux2Adapter if variant == "dev" else Flux2KleinAdapter


def build_transformer(variant):
    """Preserve published head counts,128 packed channels,four rotary axes and ratio3."""
    torch.manual_seed(813)
    return Flux2Transformer2DModel(
        in_channels=128,
        out_channels=128,
        num_layers=2,
        num_single_layers=2,
        attention_head_dim=8,
        num_attention_heads=VARIANTS[variant],
        joint_attention_dim=24,
        timestep_guidance_channels=16,
        mlp_ratio=3.0,
        axes_dims_rope=(2, 2, 2, 2),
        guidance_embeds=variant == "dev",
    )


def install_lora(model, variant, *, extra_targets=()):
    from peft import LoraConfig, get_peft_model

    cls = adapter_class(variant)
    targets = [*cls.default_target_modules.fget(object.__new__(cls)), *extra_targets]
    torch.manual_seed(22)
    return get_peft_model(
        model, LoraConfig(r=2, lora_alpha=2, target_modules=targets, init_lora_weights=False)
    )


def make_adapter(model, variant):
    cls = adapter_class(variant)
    adapter = object.__new__(cls)
    adapter.pipeline = SimpleNamespace(transformer=model)
    adapter.component_runtime = ClassicPipelineRuntime(adapter.pipeline)
    adapter._components = adapter.component_runtime.override_components
    adapter.accelerator = SimpleNamespace(
        device=torch.device("cpu"), num_processes=1, process_index=0
    )
    adapter.config = SimpleNamespace(parallelism_args=SimpleNamespace(sequence_parallel=True))
    adapter.training_args = SimpleNamespace(latent_storage_dtype="float32")
    adapter._has_warned_forward_fallback = False
    adapter._has_warned_inference_fallback = False
    adapter.scheduler = FlowMatchEulerDiscreteSDEScheduler(noise_level=0.35)
    set_scheduler_timesteps(adapter.scheduler, num_inference_steps=3, seq_len=16, device="cpu")
    return adapter


def fixed_inputs(variant, replica=0, step=0, device="cpu", dtype=torch.float32):
    generator = torch.Generator().manual_seed(401 + step + replica * 31)

    def operand(*shape):
        return torch.randn(*shape, generator=generator, dtype=dtype).to(device)

    # Target/reference boundary16 falls inside a12-token CP shard and a6-token SP half.
    target = operand(1, 16, 128)
    values = {
        "latents": target,
        "image_latents": operand(1, 8, 128),
        "latent_ids": operand(1, 16, 4),
        "image_latent_ids": operand(1, 8, 4),
        "prompt_embeds": operand(1, 8, 24),
        "text_ids": operand(1, 8, 4),
        "t": torch.tensor([600.0], dtype=dtype, device=device),
        "t_next": torch.tensor([400.0], dtype=dtype, device=device),
        "next_latents": target + 0.137,
        "guidance_scale": 2.0,
    }
    if variant != "dev":
        values.update(negative_prompt_embeds=operand(1, 8, 24), negative_text_ids=operand(1, 8, 4))
    return values


def native_velocity(model, inputs, variant):
    """Independent native conditioning, separate CFG calls and target-only crop."""
    target = inputs["latents"]
    arguments = {
        "hidden_states": torch.cat((target, inputs["image_latents"]), dim=1),
        "img_ids": torch.cat((inputs["latent_ids"], inputs["image_latent_ids"]), dim=1),
        "encoder_hidden_states": inputs["prompt_embeds"],
        "txt_ids": inputs["text_ids"],
        "timestep": inputs["t"] / 1000,
        "guidance": (
            torch.full_like(inputs["t"], inputs["guidance_scale"]) if variant == "dev" else None
        ),
        "return_dict": False,
    }
    conditional = model(**arguments)[0][:, : target.shape[1]]
    if variant == "dev":
        return conditional
    unconditional = model(
        **{
            **arguments,
            "encoder_hidden_states": inputs["negative_prompt_embeds"],
            "txt_ids": inputs["negative_text_ids"],
        }
    )[0][:, : target.shape[1]]
    return unconditional + inputs["guidance_scale"] * (conditional - unconditional)


def build_pipeline(variant):
    """Real native text and Flux2 VAE components, retaining default hidden-layer taps."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import (
        Mistral3Config,
        Mistral3ForConditionalGeneration,
        MistralConfig,
        PixtralImageProcessor,
        PixtralProcessor,
        PixtralVisionConfig,
        PreTrainedTokenizerFast,
        Qwen3Config,
        Qwen3ForCausalLM,
    )

    from diffusers import (
        AutoencoderKLFlux2,
        FlowMatchEulerDiscreteScheduler,
        Flux2KleinPipeline,
        Flux2Pipeline,
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
        "[IMG]",
        "[IMG_BREAK]",
        "[IMG_END]",
    ]
    core = Tokenizer(
        WordLevel({word: index for index, word in enumerate(words)}, unk_token="<unk>")
    )
    core.pre_tokenizer = Whitespace()
    template = "{% for message in messages %}{% if message['role']=='user' %}{% if message['content'] is string %}{{ message['content'] }}{% else %}{% for item in message['content'] %}{% if item['type']=='text' %}{{ item['text'] }}{% endif %}{% endfor %}{% endif %}{% endif %}{% endfor %} </s>"
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
    common = dict(
        vocab_size=len(words),
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=512,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=1,
    )
    if variant == "dev":
        text_encoder = Mistral3ForConditionalGeneration(
            Mistral3Config(
                text_config=MistralConfig(**common, num_hidden_layers=30),
                vision_config=PixtralVisionConfig(
                    hidden_size=8,
                    intermediate_size=16,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    head_dim=4,
                    image_size=32,
                    patch_size=8,
                ),
                image_token_index=8,
            )
        )
        tokenizer = PixtralProcessor(
            image_processor=PixtralImageProcessor(size={"longest_edge": 32}),
            tokenizer=tokenizer,
            patch_size=8,
            chat_template=template,
        )
        pipeline_class = Flux2Pipeline
    else:
        text_encoder = Qwen3ForCausalLM(Qwen3Config(**common, num_hidden_layers=27))
        pipeline_class = Flux2KleinPipeline
    vae = AutoencoderKLFlux2(
        block_out_channels=(8, 8, 8, 8),
        layers_per_block=1,
        latent_channels=32,
        norm_num_groups=4,
        sample_size=64,
    )
    # Nontrivial native BN state makes omission of latent normalization observable.
    vae.bn.running_mean.copy_(torch.linspace(-0.2, 0.3, 128))
    vae.bn.running_var.copy_(torch.linspace(0.7, 1.3, 128))
    return pipeline_class(
        transformer=build_transformer(variant),
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        vae=vae,
        scheduler=FlowMatchEulerDiscreteScheduler(),
    )


def build_fixture(root, variant):
    """Save native loadable components and actual64×128 reference PNGs resized to32×64."""
    from pathlib import Path

    import numpy as np
    from PIL import Image

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    build_pipeline(variant).save_pretrained(root / "model")
    data = root / "data"
    data.mkdir(exist_ok=True)
    for index in range(8):
        pixels = np.random.default_rng(index).integers(0, 256, (64, 128, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(data / f"{index}.png")


def load_pipeline(root, variant):
    from pathlib import Path

    from diffusers import Flux2KleinPipeline, Flux2Pipeline

    cls = Flux2Pipeline if variant == "dev" else Flux2KleinPipeline
    return cls.from_pretrained(Path(root) / "model", torch_dtype=torch.float32)


def pipeline_adapter(pipeline, variant):
    adapter = make_adapter(pipeline.transformer, variant)
    scheduler = adapter.scheduler
    adapter.pipeline = pipeline
    adapter.scheduler = scheduler
    adapter.component_runtime = ClassicPipelineRuntime(pipeline)
    adapter._components = adapter.component_runtime.override_components
    return adapter


@lru_cache(maxsize=6)
def encoded_conditions(root, variant, replica):
    """Native prompt/VAE encoders own the fixed reference operands on each rank."""
    from pathlib import Path

    from PIL import Image

    pipeline = load_pipeline(root, variant)
    adapter = pipeline_adapter(pipeline, variant)
    with torch.no_grad():
        options = {} if variant == "dev" else {"negative_prompt": "blue", "guidance_scale": 2.0}
        prompt = adapter.encode_prompt(
            "a red image" if replica else "a blue image", max_sequence_length=8, **options
        )
        image = adapter.encode_image(
            Image.open(Path(root) / "data" / f"{replica}.png"), condition_image_size=(32, 64)
        )
        ids = pipeline._prepare_latent_ids(torch.zeros(1, 128, 4, 4))
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

    build_fixture(sys.argv[1], sys.argv[2])
