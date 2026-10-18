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

from typing import Optional

from flow_factory.hparams import Arguments


def _parse(train: dict, eval_args: Optional[dict] = None) -> Arguments:
    config = {
        "train": {
            "trainer_type": "grpo",
            "unique_sample_num_per_epoch": 8,
            "group_size": 2,
            "per_device_batch_size": 1,
            **train,
        },
        "scheduler": {"dynamics_type": "ODE"},
        "rewards": [{"name": "score", "reward_model": "clip"}],
    }
    if eval_args is not None:
        config["eval"] = eval_args
    return Arguments.from_dict(config)


def test_eval_inherits_the_training_clip_length_and_frame_rate() -> None:
    """Evaluate video at the training geometry when eval leaves it unset."""
    config = _parse({"num_frames": 9, "frame_rate": 24.0}, {"eval_freq": 5})

    assert config.eval_args.num_frames == 9
    assert config.eval_args.frame_rate == 24.0


def test_explicit_eval_video_geometry_is_kept() -> None:
    """Keep eval-specific clip length and frame rate."""
    config = _parse(
        {"num_frames": 9, "frame_rate": 24.0},
        {"num_frames": 17, "frame_rate": 12.0},
    )

    assert config.eval_args.num_frames == 17
    assert config.eval_args.frame_rate == 12.0


def test_image_configs_keep_no_eval_video_geometry() -> None:
    """Leave both fields unset when training has no video geometry either."""
    config = _parse({})

    assert config.eval_args.num_frames is None
    assert config.eval_args.frame_rate is None
