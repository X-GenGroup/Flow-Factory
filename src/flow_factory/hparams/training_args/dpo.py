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

"""Training arguments for Diffusion-DPO (Direct Preference Optimization)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Tuple, Union

from ...contracts.sampler import (
    RANK_LOCAL_SAMPLER_SELECTION,
    SamplerSelectionContract,
)
from ...utils.dist import get_world_size
from ._base import TrainingArguments, _standardize_timestep_range


@dataclass
class DPOTrainingArguments(TrainingArguments):
    r"""Training arguments for Diffusion-DPO (Direct Preference Optimization).

    References:
    [1] Diffusion Model Alignment Using Direct Preference Optimization
        - https://arxiv.org/abs/2311.12908
    """

    def get_sampler_selection_contract(self) -> SamplerSelectionContract:
        """Keep transformed preference pairs rank-local during overlap."""

        if self.reward_optimization_overlap:
            return RANK_LOCAL_SAMPLER_SELECTION
        return super().get_sampler_selection_contract()

    # DPO core
    beta: float = field(
        default=2000.0,
        metadata={"help": "DPO temperature parameter controlling preference sharpness."},
    )
    # Advantage / pair formation
    global_std: bool = field(
        default=True,
        metadata={"help": "Whether to apply acquisition-wide std normalization."},
    )
    advantage_aggregation: Literal["sum", "gdpo"] = field(
        default="gdpo",
        metadata={
            "help": "Method to aggregate advantages within each group. Options: ['sum', 'gdpo']."
        },
    )

    # Timestep sampling
    weighting_scheme: Literal["logit_normal", "uniform"] = field(
        default="logit_normal",
        metadata={"help": "Timestep sampling distribution for DPO training."},
    )
    logit_mean: float = field(
        default=0.0,
        metadata={"help": "Mean for logit-normal timestep sampling."},
    )
    logit_std: float = field(
        default=1.0,
        metadata={"help": "Standard deviation for logit-normal timestep sampling."},
    )

    # Timestep control (multi-timestep training)
    num_train_timesteps: int = field(
        default=1,
        metadata={
            "help": "Total number of training timesteps per pair. 0 or None defaults to `int(num_inference_steps * (timestep_range[1] - timestep_range[0]))`."
        },
    )
    time_shift: float = field(
        default=1.0,
        metadata={"help": "Time shift for logit-normal timestep sampling. 1.0 = no shift."},
    )
    timestep_range: Union[float, Tuple[float, float]] = field(
        default=0.99,
        metadata={
            "help": "Timestep range for training. Float for [0, value], tuple for [start, end]."
        },
    )

    def __post_init__(self):
        super().__post_init__()
        self.timestep_range = _standardize_timestep_range(self.timestep_range)
        if not self.num_train_timesteps or self.num_train_timesteps <= 0:
            self.num_train_timesteps = max(
                1, int(self.num_inference_steps * (self.timestep_range[1] - self.timestep_range[0]))
            )

    @property
    def requires_ref_model(self) -> bool:
        """DPO always requires a reference model."""
        return True

    def compute_gradient_accumulation_steps(
        self,
        num_batches_per_epoch: int,
    ) -> int:
        """DPO forms M pairs from M*K samples, distributed evenly across ranks.

        The optimize loop iterates over M/world_size pairs (not M*K samples),
        because group_size (K) is consumed during pair formation.
        So the actual accumulate-batch count = (M / world_size) / batch_size,
        which differs from num_batches_per_epoch used for sampling.
        """
        world_size = get_world_size()
        pairs_per_rank = self.unique_sample_num_per_epoch // max(1, world_size)
        optimize_batches = pairs_per_rank // max(1, self.per_device_batch_size)
        return max(1, optimize_batches // self.gradient_step_per_epoch)

    def get_num_train_timesteps(self, args: Any) -> int:
        assert self.num_train_timesteps is not None
        return self.num_train_timesteps
