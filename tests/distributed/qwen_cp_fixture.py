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

"""Native Qwen-Image components and fixed masked operands for CP qualification."""

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
)

from diffusers import (
    AutoencoderKLQwenImage,
    FlowMatchEulerDiscreteScheduler,
    QwenImagePipeline,
    QwenImageTransformer2DModel,
)

wrap_classes = ("QwenImageTransformerBlock",)


def adapter_class():
    """Resolve the concrete T2I adapter used by production loading."""
    from flow_factory.models.qwen_image.qwen_image import QwenImageAdapter

    return QwenImageAdapter


def build_transformer():
    """Retain two real joint blocks and all three rotary axes."""
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
    )


def install_lora(module):
    """Train real image/text attention and MLP adapters with nonzero A and B."""
    from peft import LoraConfig, get_peft_model

    cls = adapter_class()
    targets = cls.default_target_modules.fget(object.__new__(cls))
    torch.manual_seed(1)
    return get_peft_model(
        module, LoraConfig(r=2, lora_alpha=2, target_modules=targets, init_lora_weights=False)
    )


def model_inputs(replica, step, device):
    """Use batched CFG-shaped conditions with holes and unequal valid text lengths."""
    generator = torch.Generator(device=device).manual_seed(100 + 10 * replica + step)
    hidden = torch.randn(1, 16, 16, generator=generator, device=device)
    return {
        "hidden_states": hidden.repeat(2, 1, 1),
        "encoder_hidden_states": torch.randn(2, 8, 32, generator=generator, device=device),
        "encoder_hidden_states_mask": torch.tensor(
            [[1, 0, 1, 1, 0, 1, 1, 1], [1, 1, 0, 1, 1, 0, 0, 0]],
            dtype=torch.bool,
            device=device,
        ),
        "timestep": torch.tensor([0.6, 0.6], device=device),
        "img_shapes": [[(1, 4, 4)], [(1, 4, 4)]],
    }


def serialize_inputs(inputs):
    """Retain tensors, masks and all non-tensor position metadata."""
    return {
        name: value.cpu() if isinstance(value, torch.Tensor) else value
        for name, value in inputs.items()
    }


def inactive_parameter_names(model):
    """Declare final text-output branches that cannot reach an image loss."""
    from flow_factory.models.variants import ComponentVariantRegistry

    roots = [
        module for module in model.modules() if isinstance(module, QwenImageTransformer2DModel)
    ]
    assert len(roots) == 1
    final = len(roots[0].transformer_blocks) - 1
    prefixes = (
        f"transformer_blocks.{final}.attn.to_add_out.",
        f"transformer_blocks.{final}.txt_mlp.",
    )
    names = frozenset(
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and any(
            prefix in ComponentVariantRegistry._canonical_prepared_parameter_name(name)
            for prefix in prefixes
        )
    )
    assert len(names) == 6, sorted(names)
    return names


def _tokenizer():
    # Character tokens exercise the actual native template and its 34-token crop
    # without downloading a pretrained tokenizer or giving the fixture NLP claims.
    vocabulary = {"<pad>": 0, "<eos>": 1, "<unk>": 2}
    vocabulary.update({character: index + 3 for index, character in enumerate(string.printable)})
    core = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    core.pre_tokenizer = Split("", behavior="isolated")
    return PreTrainedTokenizerFast(
        tokenizer_object=core,
        pad_token="<pad>",
        bos_token="<eos>",
        eos_token="<eos>",
        unk_token="<unk>",
        model_max_length=1024,
    )


def build_fixture(root):
    """Serialize native text, VAE and transformer components for full trainer execution."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    tokenizer = _tokenizer()
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
            "patch_size": 2,
            "temporal_patch_size": 1,
            "spatial_merge_size": 2,
            "window_size": 8,
            "fullatt_block_indexes": [0],
        },
        image_token_id=len(tokenizer) + 1,
        video_token_id=len(tokenizer) + 2,
        vision_start_token_id=len(tokenizer) + 3,
        vision_end_token_id=len(tokenizer) + 4,
    )
    pipeline = QwenImagePipeline(
        transformer=build_transformer(),
        text_encoder=Qwen2_5_VLForConditionalGeneration(text_config),
        tokenizer=tokenizer,
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
    pipeline.save_pretrained(root / "model")
    prompts = root / "prompts"
    prompts.mkdir(exist_ok=True)
    (prompts / "train.jsonl").write_text(
        "".join(
            json.dumps({"prompt": f"a {'red' if i % 2 else 'blue'} image"}) + "\n" for i in range(8)
        )
    )
    pixels = np.random.default_rng(3).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(root / "target.png")


if __name__ == "__main__":
    build_fixture(Path(sys.argv[1]))
