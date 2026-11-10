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

from pathlib import Path

import torch
import torchaudio

from flow_factory.utils.audio import save_audio


def test_audio_codec_roundtrip_and_feature_gradients(tmp_path: Path):
    """Compare eager/regional compile on a fixed native SD3 BF16 training workload.

    This is a one-GPU microbenchmark, not a pretrained or end-to-end throughput claim.
    Run in each candidate environment with the same GPU and without competing work.
    """


import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from diffusers import SD3Transformer2DModel
from flow_factory.utils.torch_runtime import configure_torch_runtime


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("eager", "compile"), default="eager")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    configure_torch_runtime(torch.device("cuda", 0))
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(1)
    model = SD3Transformer2DModel(
        sample_size=64,
        patch_size=2,
        in_channels=16,
        out_channels=16,
        num_layers=4,
        num_attention_heads=8,
        attention_head_dim=64,
        joint_attention_dim=512,
        caption_projection_dim=512,
        pooled_projection_dim=512,
        pos_embed_max_size=64,
    ).cuda()
    if args.mode == "compile":
        for block in model.transformer_blocks:
            block.compile()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, foreach=True)
    inputs = {
        "hidden_states": torch.randn(1, 16, 64, 64, device="cuda"),
        "encoder_hidden_states": torch.randn(1, 128, 512, device="cuda"),
        "pooled_projections": torch.randn(1, 512, device="cuda"),
        "timestep": torch.tensor([500.0], device="cuda"),
    }

    def step() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(**inputs, return_dict=False)[0]
            loss = output.float().square().mean()
        loss.backward()
        optimizer.step()
        return loss

    started = time.perf_counter()
    for _ in range(10):
        loss = step()
    torch.cuda.synchronize()
    warmup = time.perf_counter() - started
    torch.cuda.reset_peak_memory_stats()
    rounds, samples = [], []
    for _ in range(3):
        timings = []
        for _ in range(20):
            torch.cuda.synchronize()
            started = time.perf_counter()
            loss = step()
            torch.cuda.synchronize()
            timings.append(1000 * (time.perf_counter() - started))
        rounds.append(statistics.median(timings))
        samples.extend(timings)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    result = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cudnn_sdp_enabled": torch.backends.cuda.cudnn_sdp_enabled(),
        "device": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "mode": args.mode,
        "warmup_seconds": warmup,
        "model": "native SD3: 4 width-512 blocks, 1024 image + 128 text tokens, full AdamW, BF16 autocast",
        "median_ms": statistics.median(samples),
        "round_medians_ms": rounds,
        "p95_ms": sorted(samples)[int(len(samples) * 0.95) - 1],
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "final_loss": float(loss.detach()),
        "iterations": len(samples),
    }
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
