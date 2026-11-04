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

"""A deliberately narrow, local-weight Wan transformer canary; no model download."""

import hashlib
import json
import os
from pathlib import Path

import pytest
import torch


def test_pretrained_wan_lora_has_finite_gradients_and_real_updates():
    value = os.environ.get("FF_VALIDATION_WAN_TRANSFORMER")
    if not value:
        pytest.skip("set FF_VALIDATION_WAN_TRANSFORMER to a local Wan T2V transformer checkpoint")
    if not torch.cuda.is_available():
        pytest.skip("pretrained Wan canary needs one CUDA device")
    path = Path(value)
    config = json.loads((path / "config.json").read_text())
    assert (
        config["_class_name"] == "WanTransformer3DModel"
    ), "Animate and I2V are separate model contracts"
    assert config.get("image_dim") is None, "this canary qualifies the T2V transformer only"
    from peft import LoraConfig, get_peft_model

    from diffusers import WanTransformer3DModel

    torch.manual_seed(17)
    model = WanTransformer3DModel.from_pretrained(
        path, local_files_only=True, torch_dtype=torch.bfloat16
    ).cuda()
    model = get_peft_model(
        model,
        LoraConfig(
            r=2,
            lora_alpha=2,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            init_lora_weights=False,
        ),
    )
    model.eval()
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    frozen = {
        name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad
    }
    assert trainable and frozen
    assert all("lora_" in name for name in trainable)
    latents = torch.randn(1, config["in_channels"], 2, 4, 4, device="cuda", dtype=torch.bfloat16)
    text = torch.randn(1, 8, config["text_dim"], device="cuda", dtype=torch.bfloat16)
    timestep = torch.tensor([600.0], device="cuda", dtype=torch.bfloat16)
    before = {name: p.detach().clone() for name, p in trainable.items()}
    optimizer = torch.optim.SGD(trainable.values(), lr=0.01)
    output = model(
        hidden_states=latents, timestep=timestep, encoder_hidden_states=text, return_dict=False
    )[0]
    assert output.shape == latents.shape and torch.isfinite(output).all()
    output.float().square().mean().backward()
    actual = {name for name, p in model.named_parameters() if p.grad is not None}
    assert actual == set(trainable)
    for name, p in trainable.items():
        assert torch.isfinite(p.grad).all(), name
        assert torch.count_nonzero(p.grad).item() > 0, name
    optimizer.step()
    assert all(not torch.equal(before[name], p) for name, p in trainable.items())
    for name, p in model.named_parameters():
        if name in frozen:
            assert p.grad is None, name
            torch.testing.assert_close(p, frozen[name], rtol=0, atol=0)

    print(
        json.dumps(
            {
                "model_class": config["_class_name"],
                "model_path": str(path.resolve()),
                "config_sha256": hashlib.sha256((path / "config.json").read_bytes()).hexdigest(),
                "trainable_gradients_checked": len(trainable),
                "trainable_updates_checked": len(trainable),
                "frozen_parameters_checked": len(frozen),
                "device": str(next(model.parameters()).device),
            }
        )
    )
