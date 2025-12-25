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

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s')
logger = logging.getLogger(__name__)


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
    clip_range: Union[float, tuple[float, float]] = field(
        default=(-1e-4, 1e-4),
        metadata={"help": "Clipping range for PPO/GRPO."},
    )
    adv_clip_range: Union[float, tuple[float, float]] = field(
        default=(-5.0, 5.0),
        metadata={"help": "Clipping range for advantages in PPO/GRPO."},
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

    # Sampling arguments
    dynamics_type: Literal["Flow-SDE", 'Dance-SDE', 'CPS', 'ODE'] = field(
        default="Flow-SDE",
        metadata={"help": "Type of SDE to use."},
    )
    num_inference_steps: int = field(
        default=10,
        metadata={"help": "Number of timesteps for SDE."},
    )
    noise_level: float = field(
        default=0.7,
        metadata={"help": "Noise level for SDE sampling."},
    )
    mix_ratio: float = field(
        default=0.01,
        metadata={"help": "Mix ratio between two initial latents for SDE sampling."},
    )
    num_train_steps : int = field(
        default=1,
        metadata={"help": "Number of train steps."},
    )
    train_steps: Optional[List[int]] = field(
        default=None,
        metadata={"help": (
            "Training steps for optimization. "
            "    `train_steps` will be randomly sampled from this list."
            "    If None, will use the first 1/2 of the timesteps."
        )
        },
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
        metadata={"help": "Update EMA every N steps."},
    )

    save_freq: int = field(
        default=10,
        metadata={"help": "Model saving frequency (in epochs). 0 for no saving."},
    )

    save_dir: str = field(
        default='save',
        metadata={"help": "Directory to save logs and checkpoints. None for no saving."},
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

        world_size = get_world_size()
        logger.info("World Size:" + str(world_size))

        if self.train_steps is None:
            self.train_steps = list(range(self.num_inference_steps // 2))

        # Adjust unique_sample_num for even distribution
        sample_num_per_iteration = world_size * self.per_device_batch_size
        step = sample_num_per_iteration // math.gcd(self.unique_sample_num_per_epoch, sample_num_per_iteration)
        new_m = (self.unique_sample_num_per_epoch + step - 1) // step * step
        if new_m != self.unique_sample_num_per_epoch:
            logger.warning(f"Adjusted `unique_sample_num` from {self.unique_sample_num_per_epoch} to {new_m} to make sure `unique_sample_num`*`group_size` is multiple of `batch_size`*`num_replicas` for even distribution.")
            self.unique_sample_num_per_epoch = new_m

        self.num_batches_per_epoch = (self.unique_sample_num_per_epoch * self.group_size) // sample_num_per_iteration
        self.gradient_accumulation_steps = max(1, self.num_batches_per_epoch // self.gradient_step_per_epoch)

        self.adam_betas = tuple(self.adam_betas)
        
        if not isinstance(self.clip_range, (tuple, list)):
            self.clip_range = (-abs(self.clip_range), abs(self.clip_range))

        assert self.clip_range[0] < self.clip_range[1], "`clip_range` lower bound must be less than upper bound."

        if not isinstance(self.adv_clip_range, (tuple, list)):
            self.adv_clip_range = (-abs(self.adv_clip_range), abs(self.adv_clip_range))

        assert self.adv_clip_range[0] < self.adv_clip_range[1], "`adv_clip_range` lower bound must be less than upper bound."

        if self.learning_rate is None:
            if 'lora' in self.trainer_type.lower():
                self.learning_rate = 2e-4
            else:
                self.learning_rate = 1e-5
            logger.info(f"`learning_rate` is not set, using default {self.learning_rate} for `{self.trainer_type}` training.")

        # Expand path to user's path
        self.save_dir = os.path.expanduser(self.save_dir)
        # If save_dir does not exist, create it
        os.makedirs(self.save_dir, exist_ok=True)

    def to_dict(self) -> dict[str, Any]:
        return super().to_dict()

    def __str__(self) -> str:
        """Pretty print configuration as YAML."""
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False, indent=2)
    
    def __repr__(self) -> str:
        """Same as __str__ for consistency."""
        return self.__str__()