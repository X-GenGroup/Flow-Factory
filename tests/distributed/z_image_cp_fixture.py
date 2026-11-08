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

"""Native Z-Image pipeline with Qwen3 text and sixteen-channel VAE."""

import sys
from pathlib import Path

import torch
from flux_sft_fixture import _tokenizer
from test_z_image_context_parallel import build_transformer
from transformers import Qwen3Config, Qwen3ForCausalLM

from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, ZImagePipeline


def build_pipeline():
    torch.manual_seed(620)
    tokenizer = _tokenizer(512)
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['content'] }}{% endfor %} </s>"
    )
    transformer = build_transformer()
    with torch.no_grad():
        transformer.x_pad_token.normal_(0, 0.05)
        transformer.cap_pad_token.normal_(0, 0.05)
    return ZImagePipeline(
        transformer=transformer,
        text_encoder=Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size=8,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=8,
                max_position_embeddings=512,
                pad_token_id=0,
                bos_token_id=1,
                eos_token_id=1,
            )
        ),
        tokenizer=tokenizer,
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
            scaling_factor=0.3611,
            shift_factor=0.1159,
        ),
        scheduler=FlowMatchEulerDiscreteScheduler(shift=3.0),
    )


def build_fixture(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    build_pipeline().save_pretrained(root / "model")


if __name__ == "__main__":
    build_fixture(Path(sys.argv[1]))
