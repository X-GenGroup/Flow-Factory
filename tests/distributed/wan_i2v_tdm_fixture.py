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

"""Wan2.1 I2V TDM recipe reusing the established multirole continuation oracle."""

import subprocess
import sys
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

import torch
import wan_tdm_fixture as tdm

from flow_factory.hparams import Arguments

wrap_classes = tdm.wrap_classes
checkpoint_run_name = "wan-i2v-tdm"
execution_capability = "wan-i2v-tdm-cp-v1"
assert_worker = tdm.assert_worker


def build_fixture(root, tp_size):
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("wan_i2v_cp_fixture.py")),
            str(root),
            str(2 * tp_size),
        ],
        check=True,
    )


def make_config(root, result_dir, options):
    config = tdm.config_dict(root, result_dir, options)
    config["model"]["model_type"] = "wan2_i2v"
    config["log"]["run_name"] = checkpoint_run_name
    config["train"]["guidance_scale"] = 2.0
    config["eval"]["guidance_scale"] = 2.0
    return Arguments.from_dict(config)


@contextmanager
def observe_training(trainer, records):
    """Record actual I2V channels/CLIP contexts at every canonical adapter forward."""
    records["image_conditions"] = []
    forward = trainer.adapter.forward

    @wraps(forward)
    def observed_forward(*args, **kwargs):
        image = kwargs["image_embeds"]
        condition = kwargs["latent_condition"]
        assert image.shape == (1, 257, 16)
        assert condition.shape == (1, 8, 2, 4, 4)
        assert kwargs.get("first_frame_mask") is None
        assert torch.count_nonzero(image) and torch.count_nonzero(condition)
        records["image_conditions"].append(
            {"image": image.detach().cpu().clone(), "latent": condition.detach().cpu().clone()}
        )
        return forward(*args, **kwargs)

    trainer.adapter.forward = observed_forward
    with tdm.observe_training(trainer, records):
        try:
            yield
        finally:
            trainer.adapter.forward = forward


def assert_continuation(continuous, resumed, joined, options):
    tdm.assert_continuation(continuous, resumed, joined, options)
    tdm._compare(joined["image_conditions"], continuous["records"]["image_conditions"], exact=True)
