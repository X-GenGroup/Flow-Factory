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

"""Exercise the installed image without weights, network access or a training campaign."""

import argparse
import json
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path

import deepspeed  # noqa: F401
import torch
import torchcodec  # noqa: F401

from diffusers import SD3Transformer2DModel


def main() -> None:
    """Check entry points, dependency imports and one small denoiser update."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA smoke requested but no CUDA device is available")
    assert hasattr(torch.optim, "Muon")
    for executable in ("ff-train", "ff-prompt-enhance"):
        subprocess.run([executable, "--help"], check=True, stdout=subprocess.DEVNULL, timeout=60)
    torch.manual_seed(42)
    model = SD3Transformer2DModel(
        sample_size=4,
        in_channels=4,
        out_channels=4,
        num_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=16,
        caption_projection_dim=16,
        pooled_projection_dim=16,
        pos_embed_max_size=4,
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    result = model(
        hidden_states=torch.randn(1, 4, 4, 4, device=args.device),
        encoder_hidden_states=torch.randn(1, 2, 16, device=args.device),
        pooled_projections=torch.randn(1, 16, device=args.device),
        timestep=torch.ones(1, device=args.device),
    ).sample
    assert result.shape == (1, 4, 4, 4)
    loss = result.square().mean()
    assert torch.isfinite(loss)
    loss.backward()
    active = [p for p in model.parameters() if p.grad is not None]
    assert active and all(torch.isfinite(p.grad).all() for p in active)
    probe = next(p for p in active if p.grad.abs().sum() > 0)
    before = probe.detach().clone()
    optimizer.step()
    assert not torch.equal(before, probe.detach())
    report = {
        "device": args.device,
        "python": platform.python_version(),
        "cuda_build": torch.version.cuda,
        "device_name": torch.cuda.get_device_name() if args.device == "cuda" else "cpu",
        "loss": loss.item(),
        "gradient_tensors": len(active),
        "packages": {
            name: version(name)
            for name in (
                "flow-factory",
                "torch",
                "torchvision",
                "torchaudio",
                "torchcodec",
                "accelerate",
                "diffusers",
                "transformers",
                "peft",
                "deepspeed",
                "av",
                "numpy",
            )
        },
    }
    payload = json.dumps(report, indent=2) + "\n"
    print(payload)
    if args.report:
        args.report.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
