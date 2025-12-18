import os
import math
from datetime import datetime
from dataclasses import asdict, dataclass, field
from typing import Any, List, Literal, Union, Optional, Tuple
from logging import getLogger
import torch.distributed as dist
from datetime import datetime

logger = getLogger(__name__)

def get_world_size() -> int:
    if dist.is_initialized():
        return dist.get_world_size()
    return 1

@dataclass
class EvaluationArguments:
    resolution: Union[int, tuple[int, int]] = field(
        default=(1024, 1024),
        metadata={"help": "Resolution for evaluation."},
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
    num_timesteps: int = field(
        default=30,
        metadata={"help": "Number of timesteps for SDE."},
    )
    eval_freq: int = field(
        default=10,
        metadata={"help": "Evaluation frequency (in epochs). 0 for no evaluation."},
    )
    def __post_init__(self):
        if isinstance(self.resolution, int):
            self.resolution = (self.resolution, self.resolution)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingArguments:
    r"""Arguments pertaining to training configuration."""
    resolution: Union[int, tuple[int, int]] = field(
        default=(512, 512),
        metadata={"help": "Resolution for sampling and training."},
    )
    # Sampling and training arguments
    per_device_batch_size: int = field(
        default=1,
        metadata={"help": "Batch size per device for sampling and training."},
    )
    group_size: int = field(
        default=16,
        metadata={"help": "Group size for GRPO sampling."},
    )
    global_std: bool = field(
        default=True,
        metadata={"help": "Whether to use global std for GRPO Advantage normalization."},
    )
    unique_sample_num_per_epoch: int = field(
        default=8,
        metadata={"help": "Number of unique samples per group for GRPO sampling."},
    )
    gradient_step_per_epoch: int = field(
        default=2,
        metadata={"help": "Number of gradient steps per epoch."},
    )

    # PPO/GRPO Clip arguments
    clip_range: Union[float, tuple[float, float]] = field(
        default=1e-4,
        metadata={"help": "Clipping range for PPO/GRPO."},
    )
    adv_clip_range: Union[float, tuple[float, float]] = field(
        default=5.0,
        metadata={"help": "Clipping range for advantages in PPO/GRPO."},
    )
    max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "Maximum gradient norm for clipping."},
    )

    # Denoising process arguments
    sde_type: Literal["Flow-SDE", 'Dance-SDE', 'CPS'] = field(
        default="Flow-SDE",
        metadata={"help": "Type of SDE to use."},
    )
    num_timesteps: int = field(
        default=10,
        metadata={"help": "Number of timesteps for SDE."},
    )
    noise_level: float = field(
        default=0.7,
        metadata={"help": "Noise level for SDE sampling."},
    )
    num_noise_steps : int = field(
        default=1,
        metadata={"help": "Number of noise steps for SDE sampling."},
    )
    noise_steps: Optional[List[int]] = field(
        default=None,
        metadata={"help": (
            "Noise window for SDE sampling. "
            "    `noise_step_num` steps will be randomly sampled from this list."
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
    learning_rate: float = field(
        default=3e-4,
        metadata={"help": "Initial learning rate."},
    )

    adam_weight_decay: float = field(
        default=1e-4,
        metadata={"help": "Weight decay for AdamW optimizer."},
    )

    adam_beta1: float = field(
        default=0.9,
        metadata={"help": "Beta1 for AdamW optimizer."},
    )
    adam_beta2: float = field(
        default=0.999,
        metadata={"help": "Beta2 for AdamW optimizer."},
    )
    adam_epsilon: float = field(
        default=1e-8,
        metadata={"help": "Epsilon for AdamW optimizer."},
    )

    # Precision arguments
    mixed_precision: Literal["no", "fp16", "bf16"] = field(
        default="bf16",
        metadata={"help": "Mixed precision training type."},
    )

    save_freq: int = field(
        default=10,
        metadata={"help": "Model saving frequency (in epochs). 0 for no saving."},
    )

    save_dir: str = field(
        default="logs",
        metadata={"help": "Directory to save logs and checkpoints."},
    )

    run_name : str = field(
        default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"),
        metadata={"help": "Name of the training run. Defaults to a timestamp."},
    )

    # Nested evaluation arguments
    eval_args: EvaluationArguments = field(
        default_factory=EvaluationArguments,
        metadata={"help": "Evaluation arguments."},
    )

    def __post_init__(self):
        world_size = get_world_size()

        if self.noise_steps is None:
            self.noise_steps = list(range(self.num_timesteps // 2))

        # Adjust unique_sample_num for even distribution
        sample_num_per_iteration = world_size * self.per_device_batch_size
        step = sample_num_per_iteration // math.gcd(self.unique_sample_num_per_epoch, sample_num_per_iteration)
        new_m = (self.unique_sample_num_per_epoch + step - 1) // step * step
        if new_m != self.unique_sample_num_per_epoch:
            logger.warning(f"Adjusted `unique_sample_num` from {self.unique_sample_num_per_epoch} to {new_m} to make sure `unique_sample_num`*`group_size` is multiple of `batch_size`*`num_replicas` for even distribution.")
            self.unique_sample_num_per_epoch = new_m

        self.num_batches_per_epoch = (self.unique_sample_num_per_epoch * self.group_size) // sample_num_per_iteration
        self.gradient_accumulation_steps = max(1, self.num_batches_per_epoch // self.gradient_step_per_epoch)

        if isinstance(self.resolution, int):
            self.resolution : Tuple[int, int]= (self.resolution, self.resolution)
        
        if not isinstance(self.clip_range, tuple):
            self.clip_range : Tuple[float, float] = (self.clip_range, self.clip_range)

        if not isinstance(self.adv_clip_range, tuple):
            self.adv_clip_range : Tuple[float, float] = (self.adv_clip_range, self.adv_clip_range)

        if self.eval_args.seed is None:
            self.eval_args.seed = self.seed

        # If run_name is not provided, set it to a timestamp
        if self.run_name is None:
            self.run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        # If save_dir does not exist, create it
        os.makedirs(self.save_dir, exist_ok=True)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d['eval_args'] = self.eval_args.to_dict()
        return d