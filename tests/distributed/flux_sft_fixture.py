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

"""Native dual/single-stream Flux1 pipeline and image records for CP qualification."""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    CLIPTextConfig,
    CLIPTextModel,
    PreTrainedTokenizerFast,
    T5Config,
    T5EncoderModel,
)

from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline,
    FluxTransformer2DModel,
)


def flux_transformer():
    """Keep both native block types and all three rotary coordinate axes."""
    torch.manual_seed(11)
    return FluxTransformer2DModel(
        patch_size=1,
        in_channels=16,
        out_channels=16,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=8,
        num_attention_heads=4,
        joint_attention_dim=16,
        pooled_projection_dim=8,
        guidance_embeds=True,
        axes_dims_rope=(2, 2, 4),
    )


def flux_inputs(replica, step, device):
    """Distinct logical samples with nontrivial image positions for the component probe."""
    generator = torch.Generator(device=device).manual_seed(100 + 10 * replica + step)
    ids = torch.zeros(4, 4, 3, device=device)
    ids[:, :, 1] = torch.arange(4, device=device)[:, None]
    ids[:, :, 2] = torch.arange(4, device=device)[None, :]
    return {
        "hidden_states": torch.randn(1, 16, 16, generator=generator, device=device),
        "encoder_hidden_states": torch.randn(1, 8, 16, generator=generator, device=device),
        "pooled_projections": torch.randn(1, 8, generator=generator, device=device),
        "timestep": torch.tensor([0.6], device=device),
        "guidance": torch.tensor([3.5], device=device),
        "txt_ids": torch.zeros(8, 3, device=device),
        "img_ids": ids.reshape(16, 3),
    }


def flux_lora(module):
    """Use adapter targets across both streams, with nonzero A/B for gradient controls."""
    from peft import LoraConfig, get_peft_model

    from flow_factory.models.flux.flux1 import Flux1Adapter

    targets = Flux1Adapter.default_target_modules.fget(object.__new__(Flux1Adapter))
    torch.manual_seed(1)
    return get_peft_model(
        module, LoraConfig(r=2, lora_alpha=2, target_modules=targets, init_lora_weights=False)
    )


def _tokenizer(max_length):
    core = Tokenizer(
        WordLevel(
            {
                "<pad>": 0,
                "</s>": 1,
                "<unk>": 2,
                "a": 3,
                "red": 4,
                "blue": 5,
                "image": 6,
                "small": 7,
            },
            unk_token="<unk>",
        )
    )
    core.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=core,
        pad_token="<pad>",
        bos_token="</s>",
        eos_token="</s>",
        unk_token="<unk>",
        model_max_length=max_length,
    )


def build_flux_sft_fixture(root: Path) -> None:
    """Serialize native text encoders, VAE, denoiser and eight deterministic PNG records."""
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(11)
    pipeline = FluxPipeline(
        transformer=flux_transformer(),
        text_encoder=CLIPTextModel(
            CLIPTextConfig(
                vocab_size=8,
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                max_position_embeddings=8,
                pad_token_id=0,
                bos_token_id=1,
                eos_token_id=1,
            )
        ),
        text_encoder_2=T5EncoderModel(
            T5Config(
                vocab_size=8,
                d_model=16,
                d_ff=32,
                num_layers=1,
                num_heads=2,
                d_kv=8,
                dropout_rate=0.0,
                pad_token_id=0,
                eos_token_id=1,
            )
        ),
        tokenizer=_tokenizer(8),
        tokenizer_2=_tokenizer(512),
        vae=AutoencoderKL(
            in_channels=3,
            out_channels=3,
            down_block_types=("DownEncoderBlock2D",) * 4,
            up_block_types=("UpDecoderBlock2D",) * 4,
            block_out_channels=(8, 8, 8, 8),
            layers_per_block=1,
            latent_channels=4,
            norm_num_groups=4,
            sample_size=64,
            shift_factor=0.0,
            scaling_factor=1.0,
        ),
        scheduler=FlowMatchEulerDiscreteScheduler(),
    )
    pipeline.save_pretrained(root / "model")
    data = root / "data"
    data.mkdir()
    records = []
    for index in range(8):
        filename = f"{index}.png"
        pixels = np.random.default_rng(index).integers(0, 256, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(data / filename)
        records.append(
            {
                "schema_version": 2,
                "input": {"prompt": "a red image" if index % 2 else "a blue image", "media": []},
                "supervision": {
                    "type": "demonstration",
                    "target": {"media": [{"type": "image", "path": filename}]},
                },
            }
        )
    (data / "train.jsonl").write_text("\n".join(json.dumps(row) for row in records) + "\n")


if __name__ == "__main__":
    build_flux_sft_fixture(Path(sys.argv[1]))
