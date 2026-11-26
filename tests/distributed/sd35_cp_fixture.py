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

"""Native SD3.5 pipeline with real text packing and published head counts."""

import sys
from pathlib import Path

import torch
from flux_sft_fixture import _tokenizer
from transformers import CLIPTextConfig, CLIPTextModelWithProjection, T5Config, T5EncoderModel

from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, StableDiffusion3Pipeline


def build_transformer(family, head_dim=8):
    from diffusers import SD3Transformer2DModel

    heads = 24 if family == "medium" else 38
    torch.manual_seed(620)
    return SD3Transformer2DModel(
        sample_size=8,
        patch_size=2,
        in_channels=16,
        out_channels=16,
        num_layers=2,
        attention_head_dim=head_dim,
        num_attention_heads=heads,
        joint_attention_dim=32,
        caption_projection_dim=heads * head_dim,
        pooled_projection_dim=16,
        pos_embed_max_size=8,
        dual_attention_layers=(0,) if family == "medium" else (),
        qk_norm="rms_norm",
    )


def build_pipeline(family):
    torch.manual_seed(620)

    def clip():
        return CLIPTextModelWithProjection(
            CLIPTextConfig(
                vocab_size=8,
                hidden_size=8,
                projection_dim=8,
                intermediate_size=16,
                num_hidden_layers=2,
                num_attention_heads=2,
                max_position_embeddings=77,
                pad_token_id=0,
                bos_token_id=1,
                eos_token_id=1,
            )
        )

    return StableDiffusion3Pipeline(
        transformer=build_transformer(family),
        text_encoder=clip(),
        text_encoder_2=clip(),
        text_encoder_3=T5EncoderModel(
            T5Config(
                vocab_size=8,
                d_model=32,
                d_ff=64,
                num_layers=1,
                num_heads=4,
                d_kv=8,
                dropout_rate=0.0,
                pad_token_id=0,
                eos_token_id=1,
            )
        ),
        tokenizer=_tokenizer(77),
        tokenizer_2=_tokenizer(77),
        tokenizer_3=_tokenizer(512),
        vae=AutoencoderKL(
            in_channels=3,
            out_channels=3,
            down_block_types=("DownEncoderBlock2D",) * 4,
            up_block_types=("UpDecoderBlock2D",) * 4,
            block_out_channels=(8, 8, 8, 8),
            layers_per_block=1,
            latent_channels=16,
            norm_num_groups=4,
            sample_size=64,
            scaling_factor=1.5305,
            shift_factor=0.0609,
        ),
        scheduler=FlowMatchEulerDiscreteScheduler(shift=3.0),
    )


def build_fixture(root, family):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    build_pipeline(family).save_pretrained(root / "model")


if __name__ == "__main__":
    build_fixture(Path(sys.argv[1]), sys.argv[2])
