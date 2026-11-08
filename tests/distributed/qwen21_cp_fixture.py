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

"""Native Qwen-2.1 ordered-prefix operands and an independent joint CFG reference."""

import torch

from diffusers import QwenImage21Transformer2DModel
from flow_factory.models.qwen_image_21.qwen_image_21 import QwenImage21Adapter


def build_transformer(*, heads=32):
    """Keep native 128-channel heads, three-axis RoPE and both cache-producing blocks."""
    torch.manual_seed(137)
    return QwenImage21Transformer2DModel(
        in_channels=64,
        out_channels=64,
        num_layers=2,
        attention_head_dim=128,
        num_attention_heads=heads,
        context_in_dim=64,
        mlp_ratio=3,
        axes_dims_rope=(16, 56, 56),
    )


def install_lora(module):
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(31)
    targets = QwenImage21Adapter.default_target_modules.fget(object.__new__(QwenImage21Adapter))
    return get_peft_model(
        module, LoraConfig(r=2, lora_alpha=2, target_modules=targets, init_lora_weights=False)
    )


def model_inputs(replica, step, device):
    """Odd positive/negative prefixes interleave text with two distinct ordered images."""
    generator = torch.Generator().manual_seed(737 + 100 * replica + step)

    def randn(*shape):
        return torch.randn(*shape, generator=generator).to(device)

    return {
        "t": torch.tensor([500.0 - 100 * step], device=device),
        "latents": randn(1, 16, 64),
        "prompt_embeds": randn(1, 7, 64),
        "prompt_embeds_mask": torch.ones(1, 7, dtype=torch.long, device=device),
        "image_pad_mask": torch.tensor(
            [[False, True, False, False, True, False, False]], device=device
        ),
        "img_shapes": [(1, 2, 2), (1, 2, 2), (1, 4, 4)],
        "condition_image_latents": randn(1, 8, 64),
        "negative_prompt_embeds": randn(1, 5, 64),
        "negative_prompt_embeds_mask": torch.ones(1, 5, dtype=torch.long, device=device),
        "negative_image_pad_mask": torch.tensor([[True, False, True, False, False]], device=device),
        "guidance_scale": 2.0,
        "attention_kwargs": None,
    }


def reference_forward(model, inputs):
    """Call the unmodified native joint model directly; never use the adapter cache path."""
    target = inputs["latents"]
    images = torch.cat([inputs["condition_image_latents"], target], dim=1)
    predictions = []
    for prefix in ("", "negative_"):
        embeddings = inputs[f"{prefix}prompt_embeds"]
        slots = inputs[f"{prefix}image_pad_mask"]
        mask = torch.cat([slots, slots.new_ones(1, target.shape[1] // 4)], dim=1)
        predictions.append(
            model(
                hidden_states=images,
                encoder_hidden_states=embeddings,
                encoder_hidden_states_mask=inputs[f"{prefix}prompt_embeds_mask"],
                timestep=inputs["t"] / 1000,
                img_shapes=[inputs["img_shapes"]],
                img_mask=mask,
                return_dict=False,
            )[0][:, -target.shape[1] :]
        )
    positive, negative = predictions
    return negative + inputs["guidance_scale"] * (positive - negative)


def build_pipeline(*, heads=32):
    """Use native VL processor/encoder and RGBA VAE with the real spatial compression."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import (
        Qwen2TokenizerFast,
        Qwen2VLImageProcessorFast,
        Qwen3VLConfig,
        Qwen3VLForConditionalGeneration,
        Qwen3VLProcessor,
        Qwen3VLVideoProcessor,
    )

    from diffusers import (
        AutoencoderKLQwenImage21,
        FlowMatchEulerDiscreteScheduler,
        QwenImage21Pipeline,
    )

    specials = [
        "<pad>",
        "<unk>",
        "<|im_start|>",
        "<|im_end|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|image_pad|>",
        "<|video_pad|>",
    ]
    words = [
        "system",
        "user",
        "assistant",
        "Comprehend",
        "and",
        "analyze",
        "the",
        "provided",
        "prompt",
        ".",
        "a",
        "red",
        "blue",
        "image",
        "small",
    ]
    vocab = {word: index for index, word in enumerate(specials + words)}
    core = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    core.pre_tokenizer = Whitespace()
    tokenizer = Qwen2TokenizerFast(
        tokenizer_object=core,
        pad_token="<pad>",
        unk_token="<unk>",
        eos_token="<|im_end|>",
        additional_special_tokens=specials[2:],
    )
    template = "{% for message in messages %}<|im_start|>{{ message['role'] }}\n{% for item in message['content'] %}{{ item['text'] }}{% endfor %}<|im_end|>\n{% endfor %}"
    image_processor = Qwen2VLImageProcessorFast(
        patch_size=16,
        temporal_patch_size=2,
        merge_size=2,
        size={"shortest_edge": 1024, "longest_edge": 2048},
        image_mean=[0.5] * 3,
        image_std=[0.5] * 3,
    )
    processor = Qwen3VLProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        video_processor=Qwen3VLVideoProcessor(),
        chat_template=template,
    )
    text_encoder = Qwen3VLForConditionalGeneration(
        Qwen3VLConfig(
            image_token_id=vocab["<|image_pad|>"],
            video_token_id=vocab["<|video_pad|>"],
            vision_start_token_id=vocab["<|vision_start|>"],
            vision_end_token_id=vocab["<|vision_end|>"],
            text_config={
                "vocab_size": len(vocab),
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 16,
                "max_position_embeddings": 1024,
                "pad_token_id": 0,
                "bos_token_id": 2,
                "eos_token_id": 3,
                "rope_parameters": {
                    "rope_type": "default",
                    "rope_theta": 10000.0,
                    "mrope_section": [2, 3, 3],
                    "mrope_interleaved": True,
                },
            },
            vision_config={
                "depth": 1,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_heads": 4,
                "out_hidden_size": 64,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2,
                "num_position_embeddings": 16,
                "deepstack_visual_indexes": [],
            },
        )
    )
    vae = AutoencoderKLQwenImage21(
        base_dim=4,
        decoder_base_dim=4,
        z_dim=64,
        dim_mult=[1, 2, 2, 2, 2],
        num_res_blocks=1,
        temperal_downsample=[False, True, True, True],
        latents_mean=[0.0] * 64,
        latents_std=[1.0] * 64,
    )
    return QwenImage21Pipeline(
        transformer=build_transformer(heads=heads),
        text_encoder=text_encoder,
        processor=processor,
        vae=vae,
        scheduler=FlowMatchEulerDiscreteScheduler(),
    )


def build_fixture(root, *, heads=32):
    from pathlib import Path

    import numpy as np
    from PIL import Image

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    build_pipeline(heads=heads).save_pretrained(root / "model")
    data = root / "data"
    data.mkdir(exist_ok=True)
    for index in range(8):
        for reference, width in enumerate((32, 64)):
            pixels = np.random.default_rng(100 * index + reference).integers(
                0, 256, (32, width, 4), dtype=np.uint8
            )
            pixels[:, :, 3] = 255
            Image.fromarray(pixels, mode="RGBA").save(data / f"{index}-{reference}.png")


if __name__ == "__main__":
    import sys

    build_fixture(sys.argv[1])
