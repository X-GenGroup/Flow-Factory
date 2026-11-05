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

"""A tiny native Wan pipeline and decoded-video dataset for complete SFT tests."""

import json
import sys
from pathlib import Path

import av
import numpy as np
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, UMT5Config, UMT5EncoderModel

from diffusers import (
    AutoencoderKLWan,
    FlowMatchEulerDiscreteScheduler,
    WanPipeline,
    WanTransformer3DModel,
)


def build_wan_sft_fixture(root: Path, num_attention_heads: int = 2) -> None:
    """Write deterministic native components and eight short demonstration videos."""
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(11)
    tokenizer = Tokenizer(
        WordLevel(
            {"<pad>": 0, "</s>": 1, "<unk>": 2, "a": 3, "video": 4, "red": 5, "blue": 6},
            unk_token="<unk>",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="<pad>", eos_token="</s>", unk_token="<unk>"
    )
    pipeline = WanPipeline(
        tokenizer=tokenizer,
        text_encoder=UMT5EncoderModel(
            UMT5Config(
                vocab_size=7,
                d_model=16,
                d_ff=32,
                num_layers=1,
                num_heads=2,
                d_kv=8,
                dropout_rate=0.0,
            )
        ),
        vae=AutoencoderKLWan(
            base_dim=4,
            z_dim=4,
            dim_mult=[1, 1, 1, 1],
            num_res_blocks=1,
            temperal_downsample=[False, True, True],
            latents_mean=[0.0] * 4,
            latents_std=[1.0] * 4,
        ),
        transformer=WanTransformer3DModel(
            patch_size=(1, 2, 2),
            num_attention_heads=num_attention_heads,
            attention_head_dim=8,
            in_channels=4,
            out_channels=4,
            text_dim=16,
            freq_dim=16,
            ffn_dim=32,
            num_layers=2,
            rope_max_seq_len=32,
        ),
        scheduler=FlowMatchEulerDiscreteScheduler(),
    )
    pipeline.save_pretrained(root / "model")
    data = root / "data"
    data.mkdir()
    records = []
    for index in range(8):
        filename = f"{index}.mp4"
        frames = np.random.default_rng(index).integers(0, 256, (5, 32, 32, 3), dtype=np.uint8)
        with av.open(str(data / filename), "w") as container:
            stream = container.add_stream("libx264", rate=4)
            stream.width = stream.height = 32
            stream.pix_fmt = "yuv420p"
            for pixels in frames:
                frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        records.append(
            {
                "schema_version": 2,
                "input": {"prompt": "a red video" if index % 2 else "a blue video", "media": []},
                "supervision": {
                    "type": "demonstration",
                    "target": {"media": [{"type": "video", "path": filename, "fps": 4.0}]},
                },
            }
        )
    (data / "train.jsonl").write_text("\n".join(json.dumps(row) for row in records) + "\n")


if __name__ == "__main__":
    build_wan_sft_fixture(Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 2)
