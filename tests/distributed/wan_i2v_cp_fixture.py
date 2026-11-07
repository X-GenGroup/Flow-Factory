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

"""Native Wan2.1 I2V fixture retaining 257 image and 512 text context tokens."""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel
from wan_sft_fixture import build_wan_sft_fixture

from diffusers import WanImageToVideoPipeline, WanPipeline, WanTransformer3DModel


def build_transformer(num_attention_heads=4):
    """Keep native image KV, condition projection and channel-concatenated latent input."""
    torch.manual_seed(29)
    return WanTransformer3DModel(
        patch_size=(1, 2, 2),
        num_attention_heads=num_attention_heads,
        attention_head_dim=8,
        in_channels=12,
        out_channels=4,
        text_dim=16,
        image_dim=16,
        added_kv_proj_dim=num_attention_heads * 8,
        freq_dim=16,
        ffn_dim=32,
        num_layers=2,
        rope_max_seq_len=32,
    )


def install_lora(model):
    """Use the adapter's real default target list with nonzero low-rank arms."""
    from peft import LoraConfig, get_peft_model

    from flow_factory.models.wan.wan2_i2v import Wan2_I2V_Adapter

    torch.manual_seed(31)
    targets = Wan2_I2V_Adapter.default_target_modules.fget(object.__new__(Wan2_I2V_Adapter))
    return get_peft_model(
        model, LoraConfig(r=2, lora_alpha=2, target_modules=targets, init_lora_weights=False)
    )


def model_inputs(replica, step, device):
    """Hold both conditioning streams and generated video fixed across compared layouts."""
    generator = torch.Generator().manual_seed(113 + replica * 17 + step)
    return {
        "hidden_states": torch.randn(1, 12, 2, 4, 4, generator=generator).to(device),
        "timestep": torch.tensor([600.0 - step * 100], device=device),
        "encoder_hidden_states": torch.randn(1, 512, 16, generator=generator).to(device),
        "encoder_hidden_states_image": torch.randn(1, 257, 16, generator=generator).to(device),
    }


def build_fixture(root, num_attention_heads=4):
    """Write native encoders/VAE plus images and strict finite-video demonstration rows."""
    root = Path(root)
    build_wan_sft_fixture(root, num_attention_heads)
    base = WanPipeline.from_pretrained(root / "model", torch_dtype=torch.float32)
    torch.manual_seed(37)
    pipeline = WanImageToVideoPipeline(
        tokenizer=base.tokenizer,
        text_encoder=base.text_encoder,
        vae=base.vae,
        transformer=build_transformer(num_attention_heads),
        scheduler=base.scheduler,
        image_encoder=CLIPVisionModel(
            CLIPVisionConfig(
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=2,
                image_size=32,
                patch_size=2,
            )
        ),
        image_processor=CLIPImageProcessor(
            size={"shortest_edge": 32}, crop_size={"height": 32, "width": 32}
        ),
    )
    pipeline.save_pretrained(root / "model")
    path = root / "data/train.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    prompts = root / "prompts"
    prompts.mkdir()
    online = []
    for index, row in enumerate(rows):
        pixels = np.random.default_rng(100 + index).integers(0, 256, (32, 32, 3), dtype=np.uint8)
        image_path = root / "data" / f"{index}.png"
        Image.fromarray(pixels).save(image_path)
        row["input"]["media"] = [{"type": "image", "path": image_path.name, "slot": "first_frame"}]
        online.append({"prompt": f"{row['input']['prompt']} {index}", "images": [str(image_path)]})
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    (prompts / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in online))
    (prompts / "test.jsonl").write_text(json.dumps(online[3]) + "\n")


if __name__ == "__main__":
    build_fixture(Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 4)
