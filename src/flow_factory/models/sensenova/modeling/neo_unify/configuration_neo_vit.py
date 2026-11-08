# Copyright 2026 Jayce-Ping
#
# Vendored from OpenSenseNova/SenseNova-U1 (Apache-2.0). See the accompanying
# third-party notice in ``src/flow_factory/models/sensenova``.
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

# isort: skip_file

import os
from typing import Union

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class NEOVisionConfig(PretrainedConfig):

    model_type = "neo_vision"

    def __init__(
        self,
        num_channels=3,
        patch_size=16,
        hidden_size=1024,
        llm_hidden_size=2048,
        downsample_ratio=0.5,
        rope_theta_vision=10000.0,
        max_position_embeddings_vision=10000,
        min_pixels=65536,
        max_pixels=4194304,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.llm_hidden_size = self._singleton(llm_hidden_size, "llm_hidden_size")
        self.downsample_ratio = self._singleton(downsample_ratio, "downsample_ratio")
        self.rope_theta_vision = rope_theta_vision
        self.max_position_embeddings_vision = max_position_embeddings_vision
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

    @staticmethod
    def _singleton(value, name):
        # Native modules consume a singleton tuple. JSON serializes it as a list;
        # reloading that list must not add another nesting level.
        if isinstance(value, (list, tuple)):
            if len(value) != 1 or isinstance(value[0], (list, tuple)):
                raise ValueError(f"SenseNova vision {name} requires one scalar, received {value!r}")
            return (value[0],)
        return (value,)

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> "PretrainedConfig":
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        if "vision_config" in config_dict:
            config_dict = config_dict["vision_config"]

        if (
            "model_type" in config_dict
            and hasattr(cls, "model_type")
            and config_dict["model_type"] != cls.model_type
        ):
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)
