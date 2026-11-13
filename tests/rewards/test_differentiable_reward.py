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

"""Input-gradient capability, routing and real PickScore qualification."""

import os
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from flow_factory.contracts.reward import RewardGradientCapability
from flow_factory.hparams import RewardArguments
from flow_factory.rewards.abc import PointwiseRewardModel, RewardModelOutput
from flow_factory.rewards.differentiable import (
    DifferentiableRewardProcessor,
    validate_differentiable_rewards,
)


class PixelReward(PointwiseRewardModel):
    gradient_capability = RewardGradientCapability(input_fields=("image",))

    def __init__(self, config, accelerator=None):
        super().__init__(config, accelerator)
        self.model = nn.Linear(3, 1, bias=False).double()
        with torch.no_grad():
            self.model.weight.copy_(torch.tensor([[0.2, 0.3, 0.5]]))

    def score_differentiable(self, *, prompt, image):
        return RewardModelOutput(rewards=self.model(image.mean((-1, -2))).squeeze(-1))

    @torch.no_grad()
    def __call__(self, **kwargs):
        return self.score_differentiable(**kwargs)


def config(**kwargs):
    return RewardArguments(
        name="pixel", reward_model=f"{__name__}.PixelReward", device="cpu", **kwargs
    )


def processor(cfg=None, model=None):
    cfg = cfg or config(batch_size=2)
    return DifferentiableRewardProcessor(
        {cfg.name: model or PixelReward(cfg)}, {cfg.name: cfg}, ("image",)
    )


def test_full_tail_routing_weights_and_derivatives():
    cfg = config(batch_size=2, applicable_datasets=["a"], weight={"a": 2.0})
    score = processor(cfg)
    pixels = torch.linspace(0.1, 0.9, 5 * 3 * 2 * 2).reshape(5, 3, 2, 2).double().requires_grad_()
    result = score(prompt=["x"] * 5, source_names=["a", "b", "a", "a", "b"], image=pixels)
    assert result.weights["pixel"].tolist() == [2, 0, 2, 2, 0]
    loss = (result.scores["pixel"] * result.weights["pixel"]).mean()
    gradient = torch.autograd.grad(loss, pixels)[0]
    expected = torch.tensor([0.2, 0.3, 0.5]).reshape(1, 3, 1, 1).expand_as(pixels) / 4
    expected = expected * result.weights["pixel"].reshape(-1, 1, 1, 1) / 5
    torch.testing.assert_close(gradient, expected.double())
    assert all(
        not p.requires_grad and p.grad is None for p in score.models["pixel"].model.parameters()
    )
    assert not score.models["pixel"].model.training
    assert result.scores["pixel"][[1, 4]].tolist() == [0, 0]


def test_finite_difference_and_ordinary_output():
    score = processor()
    x = torch.full((1, 3, 2, 2), 0.5, dtype=torch.float64, requires_grad=True)
    value = score(prompt=["x"], source_names=["a"], image=x).scores["pixel"]
    ordinary = score.models["pixel"](prompt=["x"], image=x).rewards
    torch.testing.assert_close(value.double(), ordinary)
    grad = torch.autograd.grad(value.sum(), x)[0]
    delta = torch.zeros_like(x)
    delta[0, 0, 0, 0] = 1e-3
    plus = score(prompt=["x"], source_names=["a"], image=x + delta).scores["pixel"]
    minus = score(prompt=["x"], source_names=["a"], image=x - delta).scores["pixel"]
    assert ((plus - minus) / 2e-3).item() == pytest.approx(grad[0, 0, 0, 0].item(), abs=1e-5)


def test_detach_negative_control():
    model = PixelReward(config())
    model.score_differentiable = lambda **kw: RewardModelOutput(
        rewards=kw["image"].mean((1, 2, 3)).detach()
    )
    with pytest.raises(ValueError, match="detached"):
        processor(model=model)(
            prompt=["x"], source_names=["a"], image=torch.ones(1, 3, 2, 2, requires_grad=True)
        )


@pytest.mark.parametrize("model", ["ocr", "pickscore_rank"])
def test_unsupported_reward_rejected_before_construction(model):
    with pytest.raises(ValueError, match="input gradients"):
        validate_differentiable_rewards([RewardArguments(reward_model=model)], ("image",))


def test_async_rejected_and_no_applicable_subset():
    with pytest.raises(ValueError, match="synchronous"):
        processor(config(async_reward=True))
    score = processor(config(applicable_datasets=[]))
    result = score(
        prompt=["x"],
        source_names=["a"],
        image=torch.ones(1, 3, 2, 2, dtype=torch.float64, requires_grad=True),
    )
    assert result.scores["pixel"].tolist() == [0]
    assert result.weights["pixel"].tolist() == [0]


def test_clip_tensor_preprocessing_geometry_and_gradient():
    from flow_factory.rewards.tensor_preprocessing import clip_image_preprocess

    cfg = SimpleNamespace(
        do_resize=True,
        size={"shortest_edge": 6},
        do_center_crop=True,
        crop_size={"height": 4, "width": 4},
        do_normalize=True,
        image_mean=[0.5] * 3,
        image_std=[0.25] * 3,
    )
    x = torch.full((2, 3, 5, 9), 0.75, requires_grad=True)
    y = clip_image_preprocess(x, cfg)
    assert y.shape == (2, 3, 4, 4)
    torch.testing.assert_close(y, torch.ones_like(y))
    assert torch.autograd.grad(y.sum(), x)[0].abs().sum() > 0


@pytest.mark.skipif(
    os.environ.get("FF_TEST_REAL_REWARD") != "1", reason="selected real-weight CUDA evidence"
)
def test_cuda_real_pickscore_input_gradient():
    from PIL import Image

    from flow_factory.rewards.pick_score import PickScoreRewardModel

    cfg = RewardArguments(
        name="pickscore", reward_model="pickscore", dtype="float32", device="cuda", batch_size=1
    )
    model = PickScoreRewardModel(cfg, SimpleNamespace(device=torch.device("cuda:0")))
    score = DifferentiableRewardProcessor({cfg.name: model}, {cfg.name: cfg}, ("image",))
    # Quantized source isolates tensor interpolation differences from input quantization.
    generator = torch.Generator().manual_seed(17)
    pixels = torch.randint(32, 224, (2, 3, 256, 320), generator=generator).float().cuda() / 255
    pixels.requires_grad_()
    prompt = ["a red bird", "a blue bird"]
    result = score(prompt=prompt, source_names=["a", "a"], image=pixels).scores[cfg.name]
    grad = torch.autograd.grad(result.sum(), pixels)[0]
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0
    assert all(p.grad is None and not p.requires_grad for p in model.model.parameters())
    images = [
        Image.fromarray((p.detach().cpu().permute(1, 2, 0).numpy() * 255).round().astype("uint8"))
        for p in pixels
    ]
    ordinary = model(prompt=prompt, image=images).rewards
    torch.testing.assert_close(result, ordinary, atol=0.02, rtol=0.02)
    print(
        {
            "scores": result.detach().tolist(),
            "ordinary": ordinary.tolist(),
            "gradient_norm": grad.norm().item(),
            "parity_max_abs": (result - ordinary).abs().max().item(),
        }
    )


def test_execution_admits_differentiable_feedback_without_advantages(monkeypatch):
    from dataclasses import dataclass
    from typing import ClassVar

    from flow_factory.contracts.execution import DIFFERENTIABLE_REWARD_EXECUTION_CONTRACT
    from flow_factory.hparams import Arguments
    from flow_factory.hparams.training_args import TrainingArguments, _registry

    @dataclass
    class GradientArguments(TrainingArguments):
        execution_contract = DIFFERENTIABLE_REWARD_EXECUTION_CONTRACT
        reward_gradient_fields: ClassVar[tuple[str, ...]] = ("image",)

    monkeypatch.setitem(_registry._TRAINING_ARGS_REGISTRY, "gradient-test", GradientArguments)
    values = {
        "model": {"model_type": "sd3_5", "model_name_or_path": "/tmp/gradient-test"},
        "train": {
            "trainer_type": "gradient-test",
            "group_size": 1,
            "unique_sample_num_per_epoch": 2,
        },
        "rewards": [{"reward_model": f"{__name__}.PixelReward"}],
        "log": {"save_dir": ".scratch/p19/test-saves"},
    }
    parsed = Arguments.from_dict(values)
    assert parsed.training_args.execution_contract is DIFFERENTIABLE_REWARD_EXECUTION_CONTRACT
    values["rewards"][0]["reward_model"] = "pickscore_rank"
    with pytest.raises(ValueError, match="input gradients"):
        Arguments.from_dict(values)
