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

"""Native Qwen Edit Plus VL, ordered reference and zero-timestep CP operands."""

from __future__ import annotations

import json
import string
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from transformers import (
    PreTrainedTokenizerFast,
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLImageProcessor,
    Qwen2VLProcessor,
    Qwen2VLVideoProcessor,
)

from diffusers import (
    AutoencoderKLQwenImage,
    FlowMatchEulerDiscreteScheduler,
    QwenImageEditPlusPipeline,
    QwenImageTransformer2DModel,
)

wrap_classes = ("QwenImageTransformerBlock",)


def adapter_class():
    """Resolve the editing adapter rather than substituting a T2I pipeline."""
    from flow_factory.models.qwen_image.qwen_image_edit_plus import QwenImageEditPlusAdapter

    return QwenImageEditPlusAdapter


def build_transformer(zero_cond_t=True):
    """Retain both checkpoint timestep layouts and native three-axis rotary blocks."""
    torch.manual_seed(11)
    return QwenImageTransformer2DModel(
        patch_size=2,
        in_channels=16,
        out_channels=4,
        num_layers=2,
        attention_head_dim=16,
        num_attention_heads=4,
        joint_attention_dim=32,
        axes_dims_rope=(4, 6, 6),
        guidance_embeds=False,
        zero_cond_t=zero_cond_t,
    )


def install_lora(module):
    """Use real image/text attention and MLP adapters with active A and B factors."""
    from peft import LoraConfig, get_peft_model

    cls = adapter_class()
    targets = cls.default_target_modules.fget(object.__new__(cls))
    torch.manual_seed(1)
    return get_peft_model(
        module, LoraConfig(r=2, lora_alpha=2, target_modules=targets, init_lora_weights=False)
    )


def model_inputs(replica, step, device):
    """Place target/reference boundaries inside CP/SP shards, with masked CFG text."""
    generator = torch.Generator(device=device).manual_seed(100 + 10 * replica + step)
    hidden = torch.randn(1, 24, 16, generator=generator, device=device)
    shapes = [(1, 2, 4), (1, 2, 3), (1, 2, 5)]
    return {
        "hidden_states": hidden.repeat(2, 1, 1),
        "encoder_hidden_states": torch.randn(2, 8, 32, generator=generator, device=device),
        "encoder_hidden_states_mask": torch.tensor(
            [[1, 0, 1, 1, 0, 1, 1, 1], [1, 1, 0, 1, 1, 0, 0, 0]],
            dtype=torch.bool,
            device=device,
        ),
        "timestep": torch.tensor([0.6, 0.6], device=device),
        "img_shapes": [list(shapes), list(shapes)],
    }


def serialize_inputs(inputs):
    """Keep ordered position metadata together with all observed tensors."""
    return {
        name: value.cpu() if isinstance(value, torch.Tensor) else value
        for name, value in inputs.items()
    }


def inactive_parameter_names(model):
    """The final text-output branches cannot reach an image-only target loss."""
    from qwen_cp_fixture import inactive_parameter_names as resolve

    return resolve(model)


def _tokenizer():
    """Keep real multimodal special tokens while bounding the text vocabulary."""
    specials = [
        "<pad>",
        "<eos>",
        "<unk>",
        "<|im_start|>",
        "<|im_end|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|image_pad|>",
        "<|video_pad|>",
    ]
    vocabulary = {token: index for index, token in enumerate(specials)}
    vocabulary.update(
        {character: index + len(specials) for index, character in enumerate(string.printable)}
    )
    core = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    core.pre_tokenizer = Split("", behavior="isolated")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=core,
        pad_token="<pad>",
        bos_token="<eos>",
        eos_token="<eos>",
        unk_token="<unk>",
        additional_special_tokens=specials[3:],
        model_max_length=1024,
    )
    return tokenizer


def build_pipeline(zero_cond_t=True):
    """Build actual VL vision/text, processor, VAE and editing transformer modules."""
    tokenizer = _tokenizer()
    processor_options = dict(
        size={"shortest_edge": 56 * 56, "longest_edge": 128 * 128},
        patch_size=14,
        temporal_patch_size=2,
        merge_size=2,
    )
    processor = Qwen2VLProcessor(
        tokenizer=tokenizer,
        image_processor=Qwen2VLImageProcessor(**processor_options),
        video_processor=Qwen2VLVideoProcessor(**processor_options),
    )
    torch.manual_seed(11)
    text_config = Qwen2_5_VLConfig(
        text_config={
            "vocab_size": len(tokenizer),
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 2048,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 1,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "mrope_section": [1, 1, 2],
            },
        },
        vision_config={
            "depth": 1,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 2,
            "out_hidden_size": 32,
            "patch_size": 14,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "window_size": 112,
            "fullatt_block_indexes": [0],
        },
        image_token_id=tokenizer.convert_tokens_to_ids("<|image_pad|>"),
        video_token_id=tokenizer.convert_tokens_to_ids("<|video_pad|>"),
        vision_start_token_id=tokenizer.convert_tokens_to_ids("<|vision_start|>"),
        vision_end_token_id=tokenizer.convert_tokens_to_ids("<|vision_end|>"),
    )
    return QwenImageEditPlusPipeline(
        transformer=build_transformer(zero_cond_t),
        text_encoder=Qwen2_5_VLForConditionalGeneration(text_config),
        tokenizer=tokenizer,
        processor=processor,
        vae=AutoencoderKLQwenImage(
            base_dim=8,
            z_dim=4,
            dim_mult=[1, 1, 1, 1],
            num_res_blocks=1,
            temperal_downsample=[False, True, True],
            latents_mean=[0.0] * 4,
            latents_std=[1.0] * 4,
        ),
        scheduler=FlowMatchEulerDiscreteScheduler(),
    )


def build_fixture(root, zero_cond_t=True):
    """Serialize a native editing pipeline plus ordered reference-image records."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    build_pipeline(zero_cond_t).save_pretrained(root / "model")
    for index in range(2):
        pixels = np.random.default_rng(31 + index).integers(0, 256, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(root / f"reference-{index}.png")
    prompts = root / "prompts"
    prompts.mkdir(exist_ok=True)
    (prompts / "train.jsonl").write_text(
        "".join(
            json.dumps(
                dataset_record(root, f"make picture {'red' if index % 2 else 'blue'}", index)
            )
            + "\n"
            for index in range(8)
        )
    )


def dataset_record(root, prompt, index):
    """Bind the actual ordered reference paths into every shared-runner work unit."""
    del index
    root = Path(root)
    return {"prompt": prompt, "images": [str(root / f"reference-{i}.png") for i in range(2)]}


if __name__ == "__main__":
    build_fixture(Path(sys.argv[1]))
