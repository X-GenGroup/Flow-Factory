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

"""Selected real PickScore reward-to-native-policy CUDA integration."""

import os
from pathlib import Path

import pytest
import torch
import yaml


@pytest.mark.skipif(
    os.environ.get("FF_TEST_REAL_REWARD") != "1", reason="selected real-weight CUDA evidence"
)
def test_cuda_sd35_refl_real_pickscore(tmp_path):
    from test_refl_training import build_fixture

    from flow_factory.hparams import Arguments
    from flow_factory.trainers.loader import load_trainer

    fixture = tmp_path / "fixture"
    build_fixture(fixture, "sd3")
    raw = yaml.safe_load(Path("examples/refl/lora/sd3_5/pickscore.yaml").read_text())
    raw.update(num_processes=1)
    raw["model"].update(
        model_name_or_path=str(fixture / "model"), finetune_type="full", target_modules="all"
    )
    raw["data"].update(
        datasets=[
            {"name": "alignment", "dataset_dir": str(fixture / "prompts"), "train": {"weight": 1}}
        ],
        cache_dir=str(tmp_path / "cache"),
    )
    raw["train"].update(
        max_epochs=2,
        resolution=64,
        num_inference_steps=3,
        unique_sample_num_per_epoch=2,
        refl_reward_scale=1.0,
        refl_reward_target=None,
    )
    raw["scheduler"]["replay_steps"] = [1, 2]
    raw["log"].update(logging_backend="none", save_dir=str(tmp_path / "saves"), save_freq=0)
    raw["eval"]["eval_freq"] = 0
    trainer = load_trainer(Arguments.from_dict(raw))
    original_step = trainer._apply_optimizer_step
    gradients = []

    def observed_step(info):
        gradient = torch.cat(
            [
                p.grad.detach().float().flatten()
                for p in trainer.adapter.get_trainable_parameters()
                if p.grad is not None
            ]
        )
        assert torch.isfinite(gradient).all() and gradient.norm() > 0
        gradients.append(gradient.norm().item())
        return original_step(info)

    trainer._apply_optimizer_step = observed_step
    try:
        trainer.start()
        assert len(gradients) == 2
        scorer = trainer.reward_models["alignment"]
        assert all(not p.requires_grad and p.grad is None for p in scorer.model.parameters())
        print(
            {
                "policy_gradient_norms": gradients,
                "reward_revision": scorer.resolved_model_revision,
            }
        )
    finally:
        trainer.cleanup(wait=True)


@pytest.mark.skipif(
    os.environ.get("FF_TEST_REAL_REWARD") != "1", reason="selected real-weight CUDA evidence"
)
def test_cuda_wan_refl_real_pickscore(tmp_path):
    from test_refl_models import build_model_fixture, train_model

    torch.cuda.set_device(0)
    build_model_fixture(tmp_path / "fixture", "wan2_t2v")
    result = train_model(tmp_path, "wan2_t2v", real_reward=True)
    assert set(result["media_gradient_norms"]) == {"video"}
    assert result["reward_revision"]
