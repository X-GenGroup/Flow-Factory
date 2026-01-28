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

# src/flow_factory/hparams/training_args.py
import os
import math
import yaml
from datetime import datetime
from dataclasses import asdict, dataclass, field
from typing import Any, List, Literal, Union, Optional, Tuple, Dict
import logging
import torch.distributed as dist
from datetime import datetime

from .abc import ArgABC
from ..utils.logger_utils import setup_logger

logger = setup_logger(__name__, rank_zero_only=True)


def get_world_size() -> int:
    # Standard PyTorch/Accelerate/DDP variable
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    
    # OpenMPI / Horovod
    if "OMPI_COMM_WORLD_SIZE" in os.environ:
        return int(os.environ["OMPI_COMM_WORLD_SIZE"])
    
    # Intel MPI / Slurm (sometimes)
    if "PMI_SIZE" in os.environ:
        return int(os.environ["PMI_SIZE"])
    
    return 1

@dataclass
class EvaluationArguments(ArgABC):
    resolution: Union[int, tuple[int, int], list[int]] = field(
        default=(1024, 1024),
        metadata={"help": "Resolution for evaluation."},
    )
    height: Optional[int] = field(
        default=None,
        metadata={"help": "Height for evaluation. If None, use the first element of `resolution`."},
    )
    width: Optional[int] = field(
        default=None,
        metadata={"help": "Width for evaluation. If None, use the second element of `resolution`."},
    )
    per_device_batch_size: int = field(
        default=1,
        metadata={"help": "Batch size per device for evaluation."},
    )
    seed: Optional[int] = field(
        default=None,
        metadata={"help": "Random seed. Default to be the same as training."},
    )
    guidance_scale: float = field(
        default=3.5,
        metadata={"help": "Guidance scale for evaluation sampling."},
    )
    num_inference_steps: int = field(
        default=30,
        metadata={"help": "Number of timesteps for SDE."},
    )
    eval_freq: int = field(
        default=10,
        metadata={"help": "Evaluation frequency (in epochs). 0 for no evaluation."},
    )
    def __post_init__(self):
        if not self.resolution:
            logger.warning("`resolution` is not set, using default (512, 512).")
            self.resolution = (512, 512)
        elif isinstance(self.resolution, (list, tuple)):
            if len(self.resolution) == 1:
                self.resolution = (self.resolution[0], self.resolution[0])
            elif len(self.resolution) > 2:
                logger.warning(f"`resolution` has {len(self.resolution)} elements, only using the first two: ({self.resolution[0]}, {self.resolution[1]}).")
                self.resolution = (self.resolution[0], self.resolution[1])
            else:  # len == 2
                self.resolution = (self.resolution[0], self.resolution[1])
        else:  # int
            self.resolution = (self.resolution, self.resolution)
        
        # height/width override
        if self.height is not None and self.resolution[0] != self.height:
                logger.warning(
                    f"Both `resolution={self.resolution}` and `height={self.height}` are set. "
                    f"Using height to override: ({self.height}, {self.resolution[1]})."
                )
                self.resolution = (self.height, self.resolution[1])
        if self.width is not None and self.resolution[1] != self.width:
                logger.warning(
                    f"Both `resolution={self.resolution}` and `width={self.width}` are set. "
                    f"Using width to override: ({self.resolution[0]}, {self.width})."
                )
        
        # Final assignment
        self.height, self.width = self.resolution

    def to_dict(self) -> dict[str, Any]:
        return super().to_dict()


@dataclass
class TrainingArguments(ArgABC):
    r"""Arguments pertaining to training configuration."""
    resolution: Union[int, tuple[int, int], list[int]] = field(
        default=(512, 512),
        metadata={"help": "Resolution for sampling and training."},
    )
    height: Optional[int] = field(
        default=None,
        metadata={"help": "Height for sampling and training. If None, use the first element of `resolution`."},
    )
    width: Optional[int] = field(
        default=None,
        metadata={"help": "Width for sampling and training. If None, use the second element of `resolution`."},
    )
    # Sampling and training arguments
    per_device_batch_size: int = field(
        default=1,
        metadata={"help": "Batch size per device for sampling and training."},
    )
    gradient_step_per_epoch: int = field(
        default=2,
        metadata={"help": "Number of gradient steps per epoch."},
    )
    max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "Maximum gradient norm for clipping."},
    )
    num_batches_per_epoch : int = field(init=False)
    gradient_accumulation_steps : int = field(init=False)

    # GRPO arguments
    trainer_type: Literal["grpo", 'grpo_guard'] = field(
        default="grpo",
        metadata={"help": "Type of trainer to use."},
    )
    advantage_aggregation: Literal['sum', 'gdpo'] = field(
        default='gdpo',
        metadata={"help": "Method to aggregate advantages within each group for GRPO. Options: ['sum', 'gdpo']."},
    )
    group_size: int = field(
        default=16,
        metadata={"help": "Group size for GRPO sampling."},
    )
    unique_sample_num_per_epoch: int = field(
        default=8,
        metadata={"help": "Number of unique samples per group for GRPO sampling."},
    )
    global_std: bool = field(
        default=True,
        metadata={"help": "Whether to use global std for GRPO Advantage normalization."},
    )
    clip_range: Union[float, tuple[float, float]] = field(
        default=(-1e-4, 1e-4),
        metadata={"help": "Clipping range for PPO/GRPO."},
    )
    adv_clip_range: Union[float, tuple[float, float]] = field(
        default=(-5.0, 5.0),
        metadata={"help": "Clipping range for advantages in PPO/GRPO."},
    )
    kl_type: Literal['v-based', 'x-based'] = field(
        default='x-based',
        metadata={"help": """Type of KL divergence to use.
            'v-based': KL divergence in velocity space.
            'x-based': KL divergence in latent space."""},
    )
    kl_beta: float = field(
         default=0,
            metadata={"help": "KL penalty beta for PPO/GRPO."},
    )
    ref_param_device : Literal["cpu", "cuda"] = field(
        default="cuda",
        metadata={"help": "Device to store reference model parameters."},
    )

    # NFT arguments
    nft_beta: float = field(
        default=1,
        metadata={"help": "Beta parameter for NFT trainer."},
    )

    # AWM arguments
    ema_kl_beta: float = field(
        default=0,
        metadata={"help": "EMA KL penalty beta for AWM trainer."},
    )

    # AWM/NFT shared arguments - training steps, etc.
    num_train_timesteps: int = field(
        default=0,
        metadata={"help": "Total number of training timesteps. Default to `num_inference_steps`."},
    )
    time_sampling_strategy: Literal['uniform', 'logit_normal', 'discrete', 'discrete_with_init', 'discrete_wo_init'] = field(
        default='discrete',
        metadata={"help": "Time sampling strategy for training."},
    )
    time_shift: float = field(
        default=3.0,
        metadata={"help": "Time shift for logit normal time sampling."},
    )
    timestep_fraction: float = field(
        default=0.9,
        metadata={"help": "Timestep fraction for time sampling - first `timestep_fraction` portion of timesteps are used."},
    )

    # Sampling arguments
    num_inference_steps: int = field(
        default=10,
        metadata={"help": "Number of timesteps for SDE."},
    )
    guidance_scale: float = field(
        default=3.5,
        metadata={"help": "Guidance scale for sampling."},
    )

    # Environment arguments
    seed: int = field(
        default=42,
        metadata={"help": "Random seed. Default to be the same as training."},
    )

    # Optimization arguments
    learning_rate: Optional[float] = field(
        default=None,
        metadata={"help": "Initial learning rate. Default to 2e-4 for LoRA and 1e-5 for full fine-tuning."},
    )

    adam_weight_decay: float = field(
        default=1e-4,
        metadata={"help": "Weight decay for AdamW optimizer."},
    )

    adam_betas: tuple[float, float] = field(
        default=(0.9, 0.999),
        metadata={"help": "Betas for AdamW optimizer."},
    )
    adam_epsilon: float = field(
        default=1e-8,
        metadata={"help": "Epsilon for AdamW optimizer."},
    )
    enable_gradient_checkpointing:  bool = field(
        default=False,
        metadata={"help": "Whether to enable gradient checkpointing."},
    )

    # EMA arguments
    ema_decay: float = field(
        default=0.995,
        metadata={"help": "Decay for EMA model. Set to 0 to disable EMA."},
    )

    ema_update_interval: int = field(
        default=10,
        metadata={"help": "Update EMA every N epochs."},
    )

    ema_device: Literal["cpu", "cuda"] = field(
        default="cuda",
        metadata={"help": "Device to store EMA model."},
    )

    def __post_init__(self):
        if not self.resolution:
            logger.warning("`resolution` is not set, using default (512, 512).")
            self.resolution = (512, 512)
        elif isinstance(self.resolution, (list, tuple)):
            if len(self.resolution) == 1:
                self.resolution = (self.resolution[0], self.resolution[0])
            elif len(self.resolution) > 2:
                logger.warning(f"`resolution` has {len(self.resolution)} elements, only using the first two: ({self.resolution[0]}, {self.resolution[1]}).")
                self.resolution = (self.resolution[0], self.resolution[1])
            else:  # len == 2
                self.resolution = (self.resolution[0], self.resolution[1])
        else:  # int
            self.resolution = (self.resolution, self.resolution)
        
        # height/width override
        if self.height is not None and self.resolution[0] != self.height:
                logger.warning(
                    f"Both `resolution={self.resolution}` and `height={self.height}` are set. "
                    f"Using height to override: ({self.height}, {self.resolution[1]})."
                )
                self.resolution = (self.height, self.resolution[1])
        if self.width is not None and self.resolution[1] != self.width:
                logger.warning(
                    f"Both `resolution={self.resolution}` and `width={self.width}` are set. "
                    f"Using width to override: ({self.resolution[0]}, {self.width})."
                )

        # num_train_timesteps
        if self.num_train_timesteps <= 0:
            self.num_train_timesteps = self.num_inference_steps # Use same as inference steps
        
        # Final assignment
        self.height, self.width = self.resolution

        world_size = get_world_size()
        logger.info("World Size:" + str(world_size))

        # Adjust unique_sample_num for even distribution
        sample_num_per_iteration = world_size * self.per_device_batch_size
        step = sample_num_per_iteration // math.gcd(self.group_size, sample_num_per_iteration)
        new_m = (self.unique_sample_num_per_epoch + step - 1) // step * step
        if new_m != self.unique_sample_num_per_epoch:
            logger.warning(f"Adjusted `unique_sample_num` from {self.unique_sample_num_per_epoch} to {new_m} to make sure `unique_sample_num`*`group_size` is multiple of `batch_size`*`num_replicas` for even distribution.")
            self.unique_sample_num_per_epoch = new_m

        self.num_batches_per_epoch = (self.unique_sample_num_per_epoch * self.group_size) // sample_num_per_iteration
        self.gradient_accumulation_steps = max(1, self.num_batches_per_epoch // self.gradient_step_per_epoch)

        self.adam_betas : tuple[float, float] = tuple(self.adam_betas[:2]) # Ensure it's a tuple of two floats
        
        if not isinstance(self.clip_range, (tuple, list)):
            self.clip_range : tuple[int, int] = (-abs(self.clip_range), abs(self.clip_range))

        assert self.clip_range[0] < self.clip_range[1], "`clip_range` lower bound must be less than upper bound."

        if not isinstance(self.adv_clip_range, (tuple, list)):
            self.adv_clip_range : tuple[int, int] = (-abs(self.adv_clip_range), abs(self.adv_clip_range))

        assert self.adv_clip_range[0] < self.adv_clip_range[1], "`adv_clip_range` lower bound must be less than upper bound."

        if self.learning_rate is None:
            if 'lora' in self.trainer_type.lower():
                self.learning_rate = 2e-4
            else:
                self.learning_rate = 1e-5
            logger.info(f"`learning_rate` is not set, using default {self.learning_rate} for `{self.trainer_type}` training.")

    def to_dict(self) -> dict[str, Any]:
        return super().to_dict()

    def __str__(self) -> str:
        """Pretty print configuration as YAML."""
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False, indent=2)
    
    def __repr__(self) -> str:
        """Same as __str__ for consistency."""
        return self.__str__()