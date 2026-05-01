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
from __future__ import annotations

import yaml
import importlib
from dataclasses import dataclass, field
from typing import Any, Type, Literal, Union, Optional, Tuple, Dict

from .abc import ArgABC
from ..utils.dist import get_world_size
from ..utils.logger_utils import setup_logger

logger = setup_logger(__name__, rank_zero_only=True)


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


# ============================================================================
# Training Arguments Base Class
# ============================================================================

@dataclass
class TrainingArguments(ArgABC):
    r"""Base training arguments shared across all algorithms."""

    # --- Trainer type ---
    trainer_type: str = field(
        default="grpo",
        metadata={"help": "Type of trainer to use."},
    )

    # --- Resolution ---
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

    # --- Sampling and training ---
    max_epochs: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Maximum number of outer training epochs (counter `epoch` runs 0 .. max_epochs-1). "
                "None or a negative value means no limit (train until interrupted)."
            ),
        },
    )
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
    num_batches_per_epoch: int = field(init=False)
    gradient_accumulation_steps: Union[int, Literal["auto"]] = field(
        default="auto",
        metadata={
            "help": (
                "Number of backward passes before each optimizer step. "
                "'auto' derives from `gradient_step_per_epoch`. "
                "When set to an integer, `gradient_step_per_epoch` is ignored "
                "and this value is passed directly to Accelerator."
            )
        },
    )
    num_inner_epochs: int = field(
        default=1,
        metadata={"help": "Number of epochs for each inner loop optimization."},
    )
    group_size: int = field(
        default=1,
        metadata={"help": "Group size for GRPO sampling."},
    )
    unique_sample_num_per_epoch: int = field(
        default=8,
        metadata={"help": "Number of unique samples per group."},
    )
    # --- Sampling ---
    num_inference_steps: int = field(
        default=10,
        metadata={"help": "Number of timesteps for inference/SDE."},
    )
    guidance_scale: float = field(
        default=3.5,
        metadata={"help": "Guidance scale for sampling."},
    )

    # --- Seed ---
    seed: int = field(
        default=42,
        metadata={"help": "Random seed."},
    )

    # --- Optimization ---
    learning_rate: float = field(
        default=1e-5,
        metadata={"help": "Initial learning rate."},
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
    enable_gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": "Whether to enable gradient checkpointing."},
    )
    offload_samples_to_cpu: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, sample tensor fields are moved to CPU at the end of each "
                "sample() iteration and lazily reloaded per micro-batch in optimize(). "
                "Reduces sample()/optimize() GPU peak by ~num_batches_per_epoch x "
                "per_batch_size at the cost of one D2H per sample plus per-reward H2D "
                "(~100ms/epoch total). Required for large per-sample tensors (video "
                "models such as Wan); recommended for higher resolutions or larger "
                "batch sizes; safe to leave off for moderate-VRAM image models. "
                "See .agents/knowledge/topics/sample_lifecycle.md for details."
            ),
        },
    )

    # --- EMA (accessed by models/abc.py for all algorithms) ---
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
    ema_decay_schedule: Literal["constant", "power", "linear", "piecewise_linear", "cosine", "warmup_cosine"] = field(
        default="power",
        metadata={"help": "Decay schedule for EMA."},
    )

    # --- Latent storage precision ---
    latent_storage_dtype: Optional[Literal['bf16', 'fp16', 'fp32']] = field(
        default='fp16',
        metadata={"help": (
            "Dtype for storing latents in trajectory. "
            "Default fp16 uses `float16`. It's recommended to use fp16 for both precision and memory efficiency. "
            "Options: bf16, fp16, fp32, None (use model-native dtype)."
        )},
    )

    def __post_init__(self):
        # --- Resolution standardization ---
        if not self.resolution:
            logger.warning("`resolution` is not set, using default (512, 512).")
            self.resolution = (512, 512)
        elif isinstance(self.resolution, (list, tuple)):
            if len(self.resolution) == 1:
                self.resolution = (self.resolution[0], self.resolution[0])
            elif len(self.resolution) > 2:
                logger.warning(f"`resolution` has {len(self.resolution)} elements, only using the first two: ({self.resolution[0]}, {self.resolution[1]}).")
                self.resolution = (self.resolution[0], self.resolution[1])
            else:
                self.resolution = (self.resolution[0], self.resolution[1])
        else:
            self.resolution = (self.resolution, self.resolution)

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

        self.height, self.width = self.resolution

        # --- Batch size calculation ---
        # NOTE: M alignment and derived quantities (num_batches_per_epoch,
        # gradient_accumulation_steps) are computed in Arguments._align_batch_geometry()
        # because the correct alignment strategy depends on the resolved sampler type,
        # which requires cross-component information (data_args, reward_args) only
        # available at the Arguments level.
        # Placeholder values are set here so the fields exist; they will be
        # overwritten by _align_batch_geometry() before any consumer reads them.
        world_size = get_world_size()
        logger.info("World Size:" + str(world_size))

        sample_num_per_iteration = world_size * self.per_device_batch_size
        self.num_batches_per_epoch = (
            (self.unique_sample_num_per_epoch * self.group_size)
            // max(1, sample_num_per_iteration)
        )
        if self.gradient_accumulation_steps == "auto":
            self._manual_gradient_accumulation_steps = False
            self.gradient_accumulation_steps = self.compute_gradient_accumulation_steps(
                self.num_batches_per_epoch,
            )
        else:
            self._manual_gradient_accumulation_steps = True
            self.gradient_accumulation_steps = int(self.gradient_accumulation_steps)
            if self.gradient_accumulation_steps < 1:
                raise ValueError(
                    f"`gradient_accumulation_steps` must be >= 1, "
                    f"got {self.gradient_accumulation_steps}."
                )

        # --- Optimizer defaults ---
        self.adam_betas = (self.adam_betas[0], self.adam_betas[1])

        if self.learning_rate is None:
            if 'lora' in self.trainer_type.lower():
                self.learning_rate = 1e-4
            else:
                self.learning_rate = 1e-5
            logger.info(f"`learning_rate` is not set, using default {self.learning_rate} for `{self.trainer_type}` training.")

    def compute_gradient_accumulation_steps(
        self, num_batches_per_epoch: int,
    ) -> int:
        """Compute gradient accumulation steps (before ×num_train_timesteps).

        Default: the optimize loop iterates over all ``num_batches_per_epoch``
        sample batches, so ``GAS = num_batches_per_epoch / gradient_step_per_epoch``.

        Subclasses may override when their optimize loop iterates over a
        different number of batches than the sampling loop (e.g. DPO consumes
        K during pair formation, reducing the batch count).
        """
        return max(1, num_batches_per_epoch // self.gradient_step_per_epoch)

    def get_num_train_timesteps(self, args: Any) -> int:
        """Return the gradient accumulation multiplier for per-timestep losses.

        Subclasses override this to provide algorithm-specific values.
        The `args` parameter is the parent `Arguments` object, giving access
        to sibling config groups like `scheduler_args` if needed.
        """
        return 1

    @property
    def requires_ref_model(self) -> bool:
        """Whether the algorithm requires maintaining reference model parameters.
        
        Defaults to True when ``kl_beta`` exists and is positive.
        Subclasses may override for custom semantics (e.g. always False for
        algorithms that never use a reference model, or always True for
        algorithms that need one regardless of KL).
        """
        return getattr(self, 'kl_beta', 0) > 0.0

    def get_preprocess_guidance_scale(self) -> float:
        """Return the guidance_scale for data preprocessing.

        The preprocessing stage uses this to decide whether to encode
        negative prompts.  Base implementation returns ``self.guidance_scale``.
        Subclasses may override to account for optimizer-time CFG needs
        (e.g., DGPO ``kl_cfg``), ensuring negative prompts are always
        encoded when any stage might require them.
        """
        return self.guidance_scale

    def to_dict(self) -> dict[str, Any]:
        return super().to_dict()

    def __str__(self) -> str:
        """Pretty print configuration as YAML."""
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False, indent=2)
    
    def __repr__(self) -> str:
        """Same as __str__ for consistency."""
        return self.__str__()


# ============================================================================
# Algorithm-Specific Subclasses
# ============================================================================

def _standardize_clip_range(value, name: str) -> tuple[float, float]:
    """Convert a scalar or sequence to a symmetric (lo, hi) tuple."""
    if not isinstance(value, (tuple, list)):
        return (-abs(value), abs(value))
    assert value[0] < value[1], f"`{name}` lower bound must be less than upper bound, got {value}."
    return (value[0], value[1])


def _standardize_timestep_range(value: Union[float, Tuple[float, float]]) -> Tuple[float, float]:
    """Convert float or tuple to ``(frac_lo, frac_hi)`` along denoising 1000→0.

    Fraction ``f`` maps to scheduler time ``1000 * (1 - f)``. Thus ``(0, 0.99)``
    corresponds to times from ``1000`` down to ``10``.
    """
    if not isinstance(value, (list, tuple)):
        result = (0.0, float(value))
    else:
        result = (float(value[0]), float(value[1]))
    assert 0 <= result[0] < result[1] <= 1.0, (
        f"`timestep_range` must satisfy 0 <= start < end <= 1, got {result}"
    )
    return result


@dataclass
class GRPOTrainingArguments(TrainingArguments):
    r"""Training arguments for GRPO / GRPO-Guard."""

    # Group-wise advantage normalization
    global_std: bool = field(
        default=True,
        metadata={"help": "Whether to use global std for advantage normalization."},
    )
    advantage_aggregation: Literal['sum', 'gdpo'] = field(
        default='gdpo',
        metadata={"help": "Method to aggregate advantages within each group. Options: ['sum', 'gdpo']."},
    )
    # Clipping / KL
    clip_range: tuple[float, float] = field(
        default=(-1e-4, 1e-4),
        metadata={"help": "Clipping range for PPO/GRPO ratio."},
    )
    adv_clip_range: tuple[float, float] = field(
        default=(-5.0, 5.0),
        metadata={"help": "Clipping range for advantages."},
    )
    kl_type: Literal['v-based', 'x-based'] = field(
        default='x-based',
        metadata={"help": "Type of KL divergence. 'v-based': velocity space, 'x-based': latent space."},
    )
    kl_beta: float = field(
        default=0,
        metadata={"help": "KL penalty beta. 0 to disable."},
    )
    ref_param_device: Literal["cpu", "cuda"] = field(
        default="cuda",
        metadata={"help": "Device to store reference model parameters."},
    )

    def __post_init__(self):
        super().__post_init__()
        self.clip_range = _standardize_clip_range(self.clip_range, 'clip_range')
        self.adv_clip_range = _standardize_clip_range(self.adv_clip_range, 'adv_clip_range')
        if self.kl_type not in ['v-based', 'x-based']:
            raise ValueError(f"Invalid KL type: {self.kl_type}. Valid options are: ['v-based', 'x-based'].")

    def get_num_train_timesteps(self, args: Any) -> int:
        return args.scheduler_args.num_sde_steps


@dataclass
class NFTTrainingArguments(TrainingArguments):
    r"""Training arguments for DiffusionNFT."""

    # Group-wise advantage normalization
    global_std: bool = field(
        default=True,
        metadata={"help": "Whether to use global std for advantage normalization."},
    )
    advantage_aggregation: Literal['sum', 'gdpo'] = field(
        default='gdpo',
        metadata={"help": "Method to aggregate advantages within each group. Options: ['sum', 'gdpo']."},
    )
    # NFT core
    nft_beta: float = field(
        default=1.0,
        metadata={"help": "Beta parameter for NFT trainer."},
    )
    off_policy: bool = field(
        default=False,
        metadata={"help": "Whether to use EMA parameters for sampling off-policy data."},
    )

    # Clipping / KL
    adv_clip_range: tuple[float, float] = field(
        default=(-5.0, 5.0),
        metadata={"help": "Clipping range for advantages."},
    )
    kl_type: Literal['v-based'] = field(
        default='v-based',
        metadata={"help": "Type of KL divergence. NFT defaults to 'v-based'."},
    )
    kl_beta: float = field(
        default=0,
        metadata={"help": "KL penalty beta. 0 to disable."},
    )
    ref_param_device: Literal["cpu", "cuda"] = field(
        default="cuda",
        metadata={"help": "Device to store reference model parameters."},
    )

    # Timestep control
    num_train_timesteps: int = field(
        default=0,
        metadata={"help": "Total number of training timesteps. 0 or None defaults to `int(num_inference_steps * (timestep_range[1] - timestep_range[0]))`."},
    )
    time_sampling_strategy: Literal['uniform', 'logit_normal', 'discrete', 'discrete_with_init', 'discrete_wo_init'] = field(
        default='discrete',
        metadata={"help": "Time sampling strategy for training."},
    )
    time_shift: float = field(
        default=3.0,
        metadata={"help": "Time shift for logit normal time sampling."},
    )
    timestep_range: Union[float, Tuple[float, float]] = field(
        default=0.9,
        metadata={
            "help": "Fraction range along denoise axis 1000→0; maps to scheduler times "
            "[1000*(1-end), 1000*(1-start)]. Float means [0, value]."
        },
    )

    def __post_init__(self):
        super().__post_init__()

        self.timestep_range = _standardize_timestep_range(self.timestep_range)

        if not self.num_train_timesteps or self.num_train_timesteps <= 0:
            self.num_train_timesteps = max(1, int(self.num_inference_steps * (self.timestep_range[1] - self.timestep_range[0])))

        self.adv_clip_range = _standardize_clip_range(self.adv_clip_range, 'adv_clip_range')
        if self.kl_type not in ['v-based']:
            raise ValueError(f"Invalid KL type: {self.kl_type}. Valid options are: ['v-based'].")

    def get_num_train_timesteps(self, args: Any) -> int:
        assert self.num_train_timesteps is not None
        return self.num_train_timesteps


@dataclass
class AWMTrainingArguments(TrainingArguments):
    r"""Training arguments for Advantage Weighted Matching (AWM)."""

    # Group-wise advantage normalization
    global_std: bool = field(
        default=True,
        metadata={"help": "Whether to use global std for advantage normalization."},
    )
    advantage_aggregation: Literal['sum', 'gdpo'] = field(
        default='gdpo',
        metadata={"help": "Method to aggregate advantages within each group. Options: ['sum', 'gdpo']."},
    )
    # AWM core
    ema_kl_beta: float = field(
        default=0,
        metadata={"help": "EMA KL penalty beta for AWM trainer."},
    )
    awm_weighting: str = field(
        default='Uniform',
        metadata={"help": "Weighting strategy for AWM."},
    )
    ghuber_power: float = field(
        default=0.25,
        metadata={"help": "Power parameter for generalized Huber loss."},
    )
    off_policy: bool = field(
        default=False,
        metadata={"help": "Whether to use EMA parameters for sampling off-policy data."},
    )

    # Clipping / KL
    clip_range: tuple[float, float] = field(
        default=(-1e-4, 1e-4),
        metadata={"help": "Clipping range for ratio."},
    )
    adv_clip_range: tuple[float, float] = field(
        default=(-5.0, 5.0),
        metadata={"help": "Clipping range for advantages."},
    )
    kl_type: Literal['v-based'] = field(
        default='v-based',
        metadata={"help": "Type of KL divergence. AWM defaults to 'v-based'."},
    )
    kl_beta: float = field(
        default=0,
        metadata={"help": "KL penalty beta. 0 to disable."},
    )
    ref_param_device: Literal["cpu", "cuda"] = field(
        default="cuda",
        metadata={"help": "Device to store reference model parameters."},
    )

    # Timestep control
    num_train_timesteps: int = field(
        default=0,
        metadata={"help": "Total number of training timesteps. 0 or None defaults to `int(num_inference_steps * (timestep_range[1] - timestep_range[0]))`."},
    )
    time_sampling_strategy: Literal['uniform', 'logit_normal', 'discrete', 'discrete_with_init', 'discrete_wo_init'] = field(
        default='discrete',
        metadata={"help": "Time sampling strategy for training."},
    )
    time_shift: float = field(
        default=3.0,
        metadata={"help": "Time shift for logit normal time sampling."},
    )
    timestep_range: Union[float, Tuple[float, float]] = field(
        default=0.9,
        metadata={
            "help": "Fraction range along denoise axis 1000→0; maps to scheduler times "
            "[1000*(1-end), 1000*(1-start)]. Float means [0, value]."
        },
    )

    def __post_init__(self):
        super().__post_init__()

        self.timestep_range = _standardize_timestep_range(self.timestep_range)

        if not self.num_train_timesteps or self.num_train_timesteps <= 0:
            self.num_train_timesteps = max(1, int(self.num_inference_steps * (self.timestep_range[1] - self.timestep_range[0])))

        self.clip_range = _standardize_clip_range(self.clip_range, 'clip_range')
        self.adv_clip_range = _standardize_clip_range(self.adv_clip_range, 'adv_clip_range')
        if self.kl_type not in ['v-based']:
            raise ValueError(f"Invalid KL type: {self.kl_type}. Valid options are: ['v-based'].")

    def get_num_train_timesteps(self, args: Any) -> int:
        assert self.num_train_timesteps is not None
        return self.num_train_timesteps


@dataclass
class DPOTrainingArguments(TrainingArguments):
    r"""Training arguments for Diffusion-DPO (Direct Preference Optimization).

    References:
    [1] Diffusion Model Alignment Using Direct Preference Optimization
        - https://arxiv.org/abs/2311.12908
    """

    # DPO core
    beta: float = field(
        default=2000.0,
        metadata={"help": "DPO temperature parameter controlling preference sharpness."},
    )
    ref_param_device: Literal["cpu", "cuda"] = field(
        default="cuda",
        metadata={"help": "Device to store reference model parameters."},
    )

    # Advantage / pair formation
    global_std: bool = field(
        default=True,
        metadata={"help": "Whether to use global std for advantage normalization."},
    )
    advantage_aggregation: Literal['sum', 'gdpo'] = field(
        default='gdpo',
        metadata={"help": "Method to aggregate advantages within each group. Options: ['sum', 'gdpo']."},
    )

    # Timestep sampling
    weighting_scheme: Literal['logit_normal', 'uniform'] = field(
        default='logit_normal',
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
        metadata={"help": "Total number of training timesteps per pair. 0 or None defaults to `int(num_inference_steps * (timestep_range[1] - timestep_range[0]))`."},
    )
    time_shift: float = field(
        default=1.0,
        metadata={"help": "Time shift for logit-normal timestep sampling. 1.0 = no shift."},
    )
    timestep_range: Union[float, Tuple[float, float]] = field(
        default=0.99,
        metadata={"help": "Timestep range for training. Float for [0, value], tuple for [start, end]."},
    )

    def __post_init__(self):
        super().__post_init__()
        self.timestep_range = _standardize_timestep_range(self.timestep_range)
        if not self.num_train_timesteps or self.num_train_timesteps <= 0:
            self.num_train_timesteps = max(1, int(
                self.num_inference_steps * (self.timestep_range[1] - self.timestep_range[0])
            ))

    @property
    def requires_ref_model(self) -> bool:
        """DPO always requires a reference model."""
        return True

    def compute_gradient_accumulation_steps(
        self, num_batches_per_epoch: int,
    ) -> int:
        """DPO forms M pairs from M×K samples, distributed evenly across ranks.

        The optimize loop iterates over M/world_size pairs (not M×K samples),
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


@dataclass
class DGPOTrainingArguments(GRPOTrainingArguments):
    r"""Training arguments for DGPO (Direct Group Preference Optimization).

    Extends GRPO with group-level DPO loss, shared noise, DSM clipping,
    and per-timestep training controls.
    """

    # DGPO core
    dpo_beta: float = field(
        default=100.0,
        metadata={"help": "DPO beta for group preference scaling."},
    )
    use_shared_noise: bool = field(
        default=True,
        metadata={"help": "Whether to share noise across samples within the same group."},
    )
    clip_dsm: bool = field(
        default=True,
        metadata={"help": "Whether to apply PPO-style DSM clipping using EMA old-policy predictions."},
    )
    clip_kl: bool = field(
        default=False,
        metadata={"help": "Whether to apply PPO-style clipping to the KL loss using the same ratio-based mask."},
    )
    switch_ema_ref: int = field(
        default=200,
        metadata={"help": "After this many optimizer steps, use EMA parameters for sampling instead of current params."},
    )
    off_policy: bool = field(
        default=False,
        metadata={"help": "Whether to use EMA parameters for sampling from the start (off-policy)."},
    )
    kl_cfg: float = field(
        default=1.0,
        metadata={"help": "CFG scale for reference model predictions. >1.0 enables CFG on the frozen ref model."},
    )
    use_ema_ref: bool = field(
        default=False,
        metadata={"help": "Use EMA (old policy) as DGPO loss reference instead of frozen pretrained. Dynamic ref from TDM-R1."},
    )

    # Old-policy EMA ref (ema_ref) — a fast-tracking EMA separate from the sampling EMA
    ema_ref_max_decay: float = field(
        default=0.3,
        metadata={"help": "Maximum decay for old-policy EMA ref. Actual decay is min(ema_ref_max_decay, ema_ref_ramp_rate * step)."},
    )
    ema_ref_ramp_rate: float = field(
        default=0.001,
        metadata={"help": "Linear ramp rate for old-policy EMA ref decay. decay(step) = min(max_decay, ramp_rate * step)."},
    )
    ema_ref_device: Literal["cpu", "cuda"] = field(
        default='cuda',
        metadata={"help": "Device for old-policy EMA ref parameters ('cuda' or 'cpu')."},
    )

    # Timestep control
    num_train_timesteps: int = field(
        default=0,
        metadata={"help": "Number of training timesteps per sample. 0 defaults to `int(num_inference_steps * (timestep_range[1] - timestep_range[0]))`."},
    )
    time_sampling_strategy: Literal['uniform', 'logit_normal', 'discrete', 'discrete_with_init', 'discrete_wo_init'] = field(
        default='discrete',
        metadata={"help": "Strategy for sampling training timesteps."},
    )
    time_shift: float = field(
        default=3.0,
        metadata={"help": "Shift parameter for logit-normal timestep sampling."},
    )
    timestep_range: Union[float, Tuple[float, float]] = field(
        default=0.6,
        metadata={"help": "Timestep range for discrete sampling. Float for [0, value], tuple for [start, end]."},
    )

    def __post_init__(self):
        super().__post_init__()
        self.timestep_range = _standardize_timestep_range(self.timestep_range)
        if not self.num_train_timesteps or self.num_train_timesteps <= 0:
            self.num_train_timesteps = max(1, int(self.num_inference_steps * (self.timestep_range[1] - self.timestep_range[0])))

    def get_num_train_timesteps(self, args: Any) -> int:
        assert self.num_train_timesteps is not None
        return self.num_train_timesteps

    @property
    def requires_ref_model(self) -> bool:
        """DGPO always requires a reference model for the group DPO loss."""
        return True

    def get_preprocess_guidance_scale(self) -> float:
        """Account for kl_cfg: ref model may need CFG even when sampling does not."""
        return max(self.guidance_scale, self.kl_cfg)


@dataclass
class CRDTrainingArguments(TrainingArguments):
    r"""Training arguments for Centered Reward Distillation (CRD).

    Reference:
        Diffusion Reinforcement Learning via Centered Reward Distillation
        https://arxiv.org/abs/2603.14128
    """

    # Group-wise advantage normalization
    global_std: bool = field(
        default=True,
        metadata={"help": "Whether to use global std for advantage normalization."},
    )
    advantage_aggregation: Literal['sum', 'gdpo'] = field(
        default='gdpo',
        metadata={"help": "Method to aggregate advantages within each group. Options: ['sum', 'gdpo']."},
    )

    # CRD core
    crd_beta: float = field(
        default=1.0,
        metadata={"help": "Beta scaling for CRD reward matching loss. Controls implicit vs external reward balance."},
    )
    crd_loss_type: Literal['mse', 'bce'] = field(
        default='mse',
        metadata={"help": "Loss type for CRD reward distillation. 'mse': squared error, 'bce': binary cross-entropy."},
    )
    use_old_for_loss: bool = field(
        default=True,
        metadata={"help": "Use 'old' model snapshot (instead of ref) for implicit reward estimation."},
    )
    adaptive_logp: bool = field(
        default=True,
        metadata={"help": "Adaptively weight implicit reward terms by prediction error magnitude."},
    )
    weight_temp: float = field(
        default=-1.0,
        metadata={"help": "Temperature for softmax weighting of advantages in CRD. Negative means uniform (inf temp)."},
    )
    # Decay schedules for model snapshots
    old_model_decay: str = field(
        default="0-0.25-0.005-0.999",
        metadata={"help": "Decay schedule for old model blending: 'start_step-start_value-slope-end_value' or preset name."},
    )
    sampling_model_decay: Union[str, int] = field(
        default="75-0.0-0.0075-0.999",
        metadata={"help": "Decay schedule for sampling model blending. Same format as old_model_decay, or int preset."},
    )

    # Clipping / KL
    adv_clip_range: tuple[float, float] = field(
        default=(-5.0, 5.0),
        metadata={"help": "Clipping range for advantages."},
    )
    kl_type: Literal['v-based'] = field(
        default='v-based',
        metadata={"help": "Type of KL divergence. CRD uses 'v-based' (velocity space)."},
    )
    kl_beta: float = field(
        default=0.1,
        metadata={"help": "KL penalty beta for regularization against the reference model."},
    )
    kl_cfg: float = field(
        default=4.5,
        metadata={
            "help": (
                "CFG scale for the teacher (reference) model during KL computation. "
                "If > 1.0, the reference forward pass uses classifier-free guidance: "
                "``noise_pred = uncond + kl_cfg * (cond - uncond)``. "
                "Set to 1.0 (default) to disable CFG on the teacher."
            )
        },
    )
    reward_adaptive_kl: bool = field(
        default=True,
        metadata={"help": "Dynamically adjust KL strength based on reward signal."},
    )
    ref_param_device: Literal["cpu", "cuda"] = field(
        default="cuda",
        metadata={"help": "Device to store reference model parameters."},
    )

    # Timestep control
    num_train_timesteps: int = field(
        default=0,
        metadata={"help": "Number of training timesteps. 0 = auto from num_inference_steps * timestep_range."},
    )
    time_sampling_strategy: Literal['uniform', 'logit_normal', 'discrete', 'discrete_with_init', 'discrete_wo_init'] = field(
        default='discrete',
        metadata={"help": "Time sampling strategy for training."},
    )
    time_shift: float = field(
        default=3.0,
        metadata={"help": "Time shift for logit normal time sampling."},
    )
    timestep_range: Union[float, Tuple[float, float]] = field(
        default=0.99,
        metadata={
            "help": "Fraction range along denoise axis 1000→0. Default 0.99 matches original CRD's timestep_fraction."
        },
    )

    def __post_init__(self):
        super().__post_init__()
        self.timestep_range = _standardize_timestep_range(self.timestep_range)
        if not self.num_train_timesteps or self.num_train_timesteps <= 0:
            self.num_train_timesteps = max(1, int(
                self.num_inference_steps * (self.timestep_range[1] - self.timestep_range[0])
            ))
        self.adv_clip_range = _standardize_clip_range(self.adv_clip_range, 'adv_clip_range')
        if self.kl_type not in ['v-based']:
            raise ValueError(f"Invalid KL type: {self.kl_type}. Valid options are: ['v-based'].")

    @property
    def requires_ref_model(self) -> bool:
        """CRD always needs a reference model for KL and implicit reward."""
        return True

    def get_num_train_timesteps(self, args: Any) -> int:
        assert self.num_train_timesteps is not None
        return self.num_train_timesteps

    def get_preprocess_guidance_scale(self) -> float:
        """Account for kl_cfg: ref model may need CFG even when sampling does not."""
        return max(self.guidance_scale, self.kl_cfg)

# ============================================================================
# Training Arguments Registry
# ============================================================================

_TRAINING_ARGS_REGISTRY: Dict[str, Type[TrainingArguments]] = {
    'grpo': GRPOTrainingArguments,
    'grpo-guard': GRPOTrainingArguments,
    'nft': NFTTrainingArguments,
    'awm': AWMTrainingArguments,
    'dgpo': DGPOTrainingArguments,
    'dpo': DPOTrainingArguments,
    'crd': CRDTrainingArguments,
}


def get_training_args_class(identifier: str) -> Type[TrainingArguments]:
    """
    Resolve the TrainingArguments subclass for a given trainer type.
    
    Supports:
    1. Registry lookup: 'grpo' -> GRPOTrainingArguments
    2. Direct python path: 'my_package.hparams.CustomTrainingArgs' -> CustomTrainingArgs
    
    Falls back to base TrainingArguments if lookup fails.
    """
    identifier_lower = identifier.lower()

    if identifier_lower in _TRAINING_ARGS_REGISTRY:
        return _TRAINING_ARGS_REGISTRY[identifier_lower]

    # Try dynamic import (python path like 'my_package.args.CustomArgs')
    try:
        module_path, class_name = identifier.rsplit('.', 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        if isinstance(cls, type) and issubclass(cls, TrainingArguments):
            return cls
        raise TypeError(
            f"'{identifier}' resolved to {cls}, which is not a TrainingArguments subclass."
        )
    except (ImportError, AttributeError, ValueError, TypeError) as e:
        raise ImportError(
            f"Could not resolve TrainingArguments for trainer_type='{identifier}'. "
            f"Ensure it is either:\n"
            f"  1. A registered trainer: {list(_TRAINING_ARGS_REGISTRY.keys())}\n"
            f"  2. A valid python path to a TrainingArguments subclass\n"
            f"Error: {e}"
        ) from e


def list_registered_training_args() -> Dict[str, Type[TrainingArguments]]:
    """Get all registered training argument classes."""
    return _TRAINING_ARGS_REGISTRY.copy()
