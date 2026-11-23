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

"""Shared identities for trainers using synchronous reward input gradients."""

from collections.abc import Mapping, Sequence

from ...hparams.data_args import DataArguments
from ...rewards.abc import BaseRewardModel
from ...samples import BaseSample


def reward_source_names(data_args: DataArguments, samples: Sequence[BaseSample]) -> list[str]:
    """Resolve sources without guessing the origin of a mixed-source sample."""
    sources = data_args.source_id_to_name
    names = []
    for sample in samples:
        if sample.source_id is None:
            training_sources = data_args.training_datasets
            if len(training_sources) != 1:
                raise ValueError(
                    "Differentiable reward sample lacks source identity; "
                    f"training sources={[source.name for source in training_sources]!r}"
                )
            names.append(training_sources[0].name)
        else:
            names.append(sources[int(sample.source_id)])
    return names


def scorer_identity(models: Mapping[str, BaseRewardModel]) -> dict:
    """Describe scorer revisions and transforms that affect continuation."""
    scorers = {}
    for name, scorer in models.items():
        config = getattr(scorer.model, "config", None)
        processor = getattr(scorer, "processor", None)
        image_processor = getattr(processor, "image_processor", None)
        scorers[name] = {
            "class": f"{type(scorer).__module__}.{type(scorer).__qualname__}",
            "gradient_fields": scorer.gradient_capability.input_fields,
            "gradient_alternatives": scorer.gradient_capability.alternative_input_fields,
            "model_revision": getattr(
                scorer, "resolved_model_revision", getattr(config, "_commit_hash", None)
            ),
            "processor_revision": getattr(scorer, "resolved_processor_revision", None),
            "image_preprocessing": (
                image_processor.to_dict() if image_processor is not None else None
            ),
        }
    return scorers
