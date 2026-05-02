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

# src/flow_factory/hparams/args.py
"""
Main arguments class that encapsulates all configurations.

Supports loading from YAML files with nested structure.
"""
from __future__ import annotations
from dataclasses import dataclass, field, fields
from typing import Any, Literal, Optional
import yaml
from datetime import datetime
import math

from .abc import ArgABC
from .data_args import DataArguments
from .model_args import ModelArguments
from .scheduler_args import SchedulerArguments
from .training_args import TrainingArguments, EvaluationArguments, get_training_args_class
from .reward_args import RewardArguments, MultiRewardArguments
from .log_args import LogArguments
from ..utils.logger_utils import setup_logger
from ..utils.dist import get_world_size

logger = setup_logger(__name__, rank_zero_only=True)


@dataclass
class Arguments(ArgABC):
    """
    Main arguments class encapsulating all configurations.
    """
    
    launcher: Literal['accelerate'] = field(
        default='accelerate',
        metadata={"help": "Distributed launcher to use."},
    )
    config_file: str | None = field(
        default=None,
        metadata={"help": "Path to distributed configuration file."},
    )
    num_processes: int = field(
        default=1,
        metadata={"help": "Number of processes for distributed training."},
    )
    main_process_port: int = field(
        default=29500,
        metadata={"help": "Main process port for distributed training."},
    )
    mixed_precision: Optional[Literal['no', 'fp16', 'bf16']] = field(
        default='bf16',
        metadata={"help": "Mixed precision setting for training."},
    )
    # Nested argument groups
    data_args: DataArguments = field(
        default_factory=DataArguments,
        metadata={"help": "Arguments for data configuration."},
    )
    model_args: ModelArguments = field(
        default_factory=ModelArguments,
        metadata={"help": "Arguments for model configuration."},
    )
    scheduler_args: SchedulerArguments = field(
        default_factory=SchedulerArguments,
        metadata={"help": "Arguments for scheduler configuration."},
    )
    training_args: TrainingArguments = field(
        default_factory=TrainingArguments,
        metadata={"help": "Arguments for training configuration."},
    )
    eval_args: EvaluationArguments = field(
        default_factory=EvaluationArguments,
        metadata={"help": "Arguments for evaluation configuration."},
    )
    log_args: LogArguments = field(
        default_factory=LogArguments,
        metadata={"help": "Arguments for logging configuration."},
    )
    reward_args: MultiRewardArguments = field(
        default_factory=MultiRewardArguments,
        metadata={"help": "Arguments for multiple reward configurations."},
    )
    eval_reward_args: Optional[MultiRewardArguments] = field(
        default=None,
        metadata={"help": "Arguments for multiple evaluation reward configurations."},
    )

    def __post_init__(self):
        if self.log_args.run_name is None:
            time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_args.run_name = f"{self.model_args.model_type}_{self.model_args.finetune_type}_{self.training_args.trainer_type}_{time_stamp}"

        self._resolve_scheduler_sde_defaults()
        self._resolve_sampler_type()
        self._align_batch_geometry()
        self._adjust_gradient_accumulation()

    def _resolve_sampler_type(self) -> None:
        """Choose the distributed sampler strategy.

        Writes the resolved value back to ``data_args.sampler_type`` so all
        downstream consumers (``get_data_sampler``, ``RewardProcessor``,
        ``AdvantageProcessor``) read a concrete choice — never ``"auto"``.

        Rules:
        - DGPO is always forced to ``group_distributed``.
        - For non-DGPO trainers, explicit user choice is respected, unless
          ``distributed_k_repeat`` / ``group_distributed`` conflicts with async
          rewards (hard override to ``group_contiguous``).
        - ``"auto"`` prefers ``group_contiguous`` (minimal communication) and
          falls back to ``distributed_k_repeat`` only when the stricter
          geometric constraints cannot be satisfied without padding
          ``unique_sample_num_per_epoch``.
        """

        # 1. Detect async rewards
        all_configs = list(self.reward_args or [])
        if self.eval_reward_args:
            all_configs += list(self.eval_reward_args)

        self._has_async_rewards = any(getattr(cfg, 'async_reward', False) for cfg in all_configs)

        # 2. Resolve sampler type
        ta = self.training_args
        user_choice = self.data_args.sampler_type

        trainer_type = str(ta.trainer_type).lower()

        if (
            user_choice in {"distributed_k_repeat", "group_distributed"}
            and self._has_async_rewards
            and trainer_type != "dgpo"
        ):
            # Hard override to `group_contiguous` for async rewards
            # In fact, only group-wise async rewards require `group_contiguous` sampler
            # For pointwise async rewards, distributed_k_repeat is still valid
            # but for simplicity, we enforce `group_contiguous` for all async rewards
            logger.warning(
                "Async rewards require 'group_contiguous' sampler. "
                f"Overriding '{user_choice}' → 'group_contiguous'."
            )
            self.data_args.sampler_type = "group_contiguous"
        
        if user_choice == "auto" and trainer_type != "dgpo":
            # auto: prefer `group_contiguous` (all K copies on same rank → no cross-rank all-gather for rewards/advantages),
            # fall back to `distributed_k_repeat` (all K copies scattered across ranks → cross-rank all-gather for rewards/advantages)
            # There are two geometric constraints:
            #   - `groups_per_rank_ok`: unique_sample_num_per_epoch % num_replicas == 0
            #   - `local_batch_tiling_ok`: (unique_sample_num_per_epoch // num_replicas) * group_size % per_device_batch_size == 0
            world_size = get_world_size()
            m = ta.unique_sample_num_per_epoch
            groups_per_rank_ok = (m % world_size == 0)
            local_batch_tiling_ok = (m // world_size * ta.group_size % ta.per_device_batch_size == 0)
            # GroupContiguousSampler's requires both while DistributedKRepeatSampler's only requires the local batch tiling constraint.
            # If `groups_per_rank_ok` is not satisfied but `local_batch_tiling_ok` is satisfied,
            # use `distributed_k_repeat` to satisfy the constraint.
            if not groups_per_rank_ok and local_batch_tiling_ok:
                self.data_args.sampler_type = "distributed_k_repeat"
            else:
                # Otherwise, use `group_contiguous`
                # and later `_align_batch_geometry()` will adjust `unique_sample_num_per_epoch` to satisfy the geometric constraints above.
                self.data_args.sampler_type = "group_contiguous"

        if trainer_type == "dgpo" and self.data_args.sampler_type != "group_distributed":
            logger.warning(
                "DGPO requires sampler_type='group_distributed'. "
                f"Overriding '{self.data_args.sampler_type}' -> 'group_distributed'."
            )
            self.data_args.sampler_type = "group_distributed"


    def _align_batch_geometry(self) -> None:
        """Align ``unique_sample_num_per_epoch`` (and, for ``group_distributed``,
        ``group_size``) to sampler constraints, then recompute derived batch
        quantities.

        Must run after ``_resolve_sampler_type()`` so the sampler choice is
        finalised.  Overwrites placeholder values set in
        ``TrainingArguments.__post_init__``.

        Each sampler enforces slightly different divisibility constraints on
        ``unique_sample_num_per_epoch * group_size`` vs.
        ``num_replicas * per_device_batch_size`` (and, optionally,
        ``gradient_step_per_epoch``); the actual logic lives in the
        per-sampler helpers below.  This method only dispatches to the right
        one and then updates derived quantities (``num_batches_per_epoch`` +
        ``gradient_accumulation_steps``).
        """
        sampler_type = self.data_args.sampler_type
        if sampler_type == "distributed_k_repeat":
            self._align_for_distributed_k_repeat()
        elif sampler_type == "group_contiguous":
            self._align_for_group_contiguous()
        elif sampler_type == "group_distributed":
            self._align_for_group_distributed()
        else:
            raise ValueError(
                f"Unknown sampler_type={sampler_type!r}; "
                "expected one of {'distributed_k_repeat', 'group_contiguous', 'group_distributed'}."
            )
        self._recompute_derived_batch_quantities()

    # ---------------------------------------------------------------------
    # Shared alignment primitives (used by the three per-sampler helpers).
    # ---------------------------------------------------------------------
    @staticmethod
    def _round_up_to_step(value: int, step: int) -> int:
        """Smallest multiple of ``step`` that is ``>= value``."""
        return ((value + step - 1) // step) * step

    def _base_unique_sample_step(self) -> int:
        """Minimum alignment step for ``unique_sample_num_per_epoch``.

        Ensures::

            unique_sample_num_per_epoch * group_size
                ≡ 0  (mod  num_replicas * per_device_batch_size [* gradient_step_per_epoch])

        i.e. ``unique_sample_num_per_epoch`` must be a multiple of::

            (num_replicas * per_device_batch_size [* gradient_step_per_epoch])
            / gcd(group_size, num_replicas * per_device_batch_size)

        This same base step powers all three samplers; ``group_contiguous``
        further ``lcm``-s it with ``num_replicas``.
        """
        ta = self.training_args
        sample_num_per_iteration = get_world_size() * ta.per_device_batch_size
        base = sample_num_per_iteration // math.gcd(ta.group_size, sample_num_per_iteration)
        if not ta._manual_gradient_accumulation_steps:
            base *= ta.gradient_step_per_epoch
        return base

    def _warn_and_assign_unique_sample_num(
        self,
        new_unique_sample_num: int,
        sampler_name: str,
        extra_line: str = "",
    ) -> None:
        """Emit a standardised "adjusted unique_sample_num_per_epoch" warning
        and assign the new value.

        ``extra_line`` lets a caller prepend a sampler-specific constraint line
        (e.g. ``group_contiguous``'s ``unique_sample_num_per_epoch %
        num_replicas == 0`` requirement) above the shared constraint
        description.
        """
        ta = self.training_args
        world_size = get_world_size()
        constraint_suffix = (
            f" * gradient_step_per_epoch({ta.gradient_step_per_epoch}))"
            if not ta._manual_gradient_accumulation_steps
            else ")"
        )
        prefix = f"{extra_line}\n  " if extra_line else "  "
        logger.warning(
            f"{sampler_name}: adjusted `unique_sample_num_per_epoch` "
            f"from {ta.unique_sample_num_per_epoch} to {new_unique_sample_num} to satisfy:\n"
            f"{prefix}unique_sample_num_per_epoch({new_unique_sample_num}) "
            f"* group_size({ta.group_size}) "
            f"% (num_replicas({world_size}) "
            f"* per_device_batch_size({ta.per_device_batch_size})"
            + constraint_suffix
            + " == 0"
        )
        ta.unique_sample_num_per_epoch = new_unique_sample_num

    def _recompute_derived_batch_quantities(self) -> None:
        """Recompute ``num_batches_per_epoch`` and (in ``auto`` grad-accum
        mode) ``gradient_accumulation_steps`` from the now-aligned
        ``(unique_sample_num_per_epoch, group_size)`` pair.

        Called by the dispatcher after any per-sampler alignment; each of
        the three helpers thus only needs to adjust the inputs, not the
        outputs.
        """
        ta = self.training_args
        sample_num_per_iteration = get_world_size() * ta.per_device_batch_size
        ta.num_batches_per_epoch = (
            (ta.unique_sample_num_per_epoch * ta.group_size)
            // sample_num_per_iteration
        )
        if not ta._manual_gradient_accumulation_steps:
            ta.gradient_accumulation_steps = ta.compute_gradient_accumulation_steps(
                ta.num_batches_per_epoch,
            )

    # ---------------------------------------------------------------------
    # Per-sampler alignment (identical shape: adjust inputs, then unique_sample_num).
    # ---------------------------------------------------------------------
    def _align_for_distributed_k_repeat(self) -> None:
        """``DistributedKRepeatSampler``: only the base
        ``unique_sample_num_per_epoch * group_size`` divisibility constraint.
        """
        ta = self.training_args
        step = self._base_unique_sample_step()
        new_unique_sample_num = self._round_up_to_step(ta.unique_sample_num_per_epoch, step)
        if new_unique_sample_num != ta.unique_sample_num_per_epoch:
            self._warn_and_assign_unique_sample_num(new_unique_sample_num, "DistributedKRepeatSampler")

    def _align_for_group_contiguous(self) -> None:
        """``GroupContiguousSampler``: base constraint + ``unique_sample_num_per_epoch % num_replicas == 0``."""
        ta = self.training_args
        world_size = get_world_size()
        step = math.lcm(self._base_unique_sample_step(), world_size)
        new_unique_sample_num = self._round_up_to_step(ta.unique_sample_num_per_epoch, step)
        if new_unique_sample_num != ta.unique_sample_num_per_epoch:
            extra_line = (
                f"  1) unique_sample_num_per_epoch({new_unique_sample_num}) "
                f"% num_replicas({world_size}) == 0\n  2)"
            )
            self._warn_and_assign_unique_sample_num(
                new_unique_sample_num, "GroupContiguousSampler", extra_line,
            )

    def _align_for_group_distributed(self) -> None:
        """``GroupDistributedSampler``: first align ``group_size`` so that
        ``group_size % num_replicas == 0`` **and**
        ``(num_replicas * per_device_batch_size) % group_size == 0``; then
        do the base ``unique_sample_num_per_epoch`` alignment with the
        (possibly bumped) ``group_size``.

        **group_size alignment** — any valid ``group_size`` has the form
        ``num_replicas * d`` where ``d`` is a divisor of
        ``per_device_batch_size``, so we enumerate divisors of
        ``per_device_batch_size`` in O(√per_device_batch_size) and pick the
        smallest ``d`` with ``d >= ceil(group_size / num_replicas)``.
        When ``group_size > num_replicas * per_device_batch_size`` no
        solution exists.
        """
        ta = self.training_args
        if ta.group_size <= 0:
            raise ValueError(f"group_size must be positive, got {ta.group_size}.")

        world_size = get_world_size()
        per_device_batch_size = ta.per_device_batch_size
        sample_num_per_iteration = world_size * per_device_batch_size
        original_group_size = ta.group_size

        if original_group_size > sample_num_per_iteration:
            raise ValueError(
                "sampler_type='group_distributed' requires "
                "`group_size <= num_replicas * per_device_batch_size`; "
                f"got group_size={original_group_size}, "
                f"num_replicas * per_device_batch_size={sample_num_per_iteration}."
            )

        # Smallest new group_size = num_replicas * d  where
        # d divides per_device_batch_size  and  d >= ceil(group_size / num_replicas).
        min_copies_per_rank = -(-original_group_size // world_size)  # ceil division
        best_copies_per_rank = per_device_batch_size                 # fallback (always valid)
        i = 1
        while i * i <= per_device_batch_size:
            if per_device_batch_size % i == 0:
                for d in (i, per_device_batch_size // i):
                    if min_copies_per_rank <= d < best_copies_per_rank:
                        best_copies_per_rank = d
            i += 1
        new_group_size = world_size * best_copies_per_rank

        if new_group_size != original_group_size:
            logger.warning(
                "sampler_type='group_distributed' requires `group_size %% num_replicas == 0` "
                "and `(num_replicas * per_device_batch_size) %% group_size == 0`; "
                "auto-adjusting group_size from %d to %d "
                "(num_replicas=%d, per_device_batch_size=%d).",
                original_group_size,
                new_group_size,
                world_size,
                per_device_batch_size,
            )
            ta.group_size = new_group_size

        # Now do the shared unique_sample_num_per_epoch alignment with the aligned group_size.
        step = self._base_unique_sample_step()
        new_unique_sample_num = self._round_up_to_step(ta.unique_sample_num_per_epoch, step)
        if new_unique_sample_num != ta.unique_sample_num_per_epoch:
            self._warn_and_assign_unique_sample_num(new_unique_sample_num, "GroupDistributedSampler")


    def _adjust_gradient_accumulation(self) -> None:
        """Adjust gradient accumulation for per-timestep losses.

        Must run AFTER `_align_batch_geometry()` which finalises the base
        gradient_accumulation_steps from the aligned M.
        Skipped when gradient_accumulation_steps is manually set — the user
        value is treated as final.
        """
        if not self.training_args._manual_gradient_accumulation_steps:
            num_train_timesteps = self.training_args.get_num_train_timesteps(self)
            self.training_args.gradient_accumulation_steps *= num_train_timesteps
        else:
            logger.info(
                f"`gradient_accumulation_steps` manually set to "
                f"{self.training_args.gradient_accumulation_steps}. "
                f"`gradient_step_per_epoch` will not be used for "
                f"gradient accumulation computation."
            )

    def _resolve_scheduler_sde_defaults(self) -> None:
        """Fill `sde_steps` / `num_sde_steps` when YAML uses null.

        Matches runtime SDE schedulers: default step indices are
        ``0 .. num_inference_steps-2`` (all steps except the last). When
        ``num_sde_steps`` is null, use the full resolved pool (same as the
        scheduler property default).

        Skipped for ODE dynamics (no stochastic steps).
        """
        sched = self.scheduler_args
        if sched.dynamics_type == 'ODE':
            return

        n_inf = self.training_args.num_inference_steps
        if sched.sde_steps is None:
            sched.sde_steps = list(range(max(0, n_inf - 1)))
        if sched.num_sde_steps is None:
            sched.num_sde_steps = len(sched.sde_steps)
        if sched.num_sde_steps <= 0:
            raise ValueError(
                "scheduler.num_sde_steps must be positive after resolving nulls; "
                f"got `num_sde_steps`={sched.num_sde_steps!r}, `sde_steps`={sched.sde_steps!r}, "
                f"`num_inference_steps`={n_inf!r}."
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {}
        
        for f in fields(self):
            value = getattr(self, f.name)
            if value is None:
                continue
            if isinstance(value, ArgABC):
                # Remove '_args' suffix for nested configs
                key = f.name.replace('_args', '')
                result[key] = value.to_dict()
            else:
                result[f.name] = value

        extras = result.pop("extra_kwargs", {})
        result.update(extras)
        return result

    @classmethod
    def from_dict(cls, args_dict: dict[str, Any]) -> Arguments:
        """Create Arguments instance from dictionary."""

        # 1. Resolve TrainingArguments subclass based on trainer_type
        train_dict = args_dict.get('train', {})
        trainer_type = train_dict.get('trainer_type', 'grpo')
        training_args_cls = get_training_args_class(trainer_type)

        # 2. Nested arguments map
        nested_map = {
            'data': ('data_args', DataArguments),
            'model': ('model_args', ModelArguments),
            'scheduler': ('scheduler_args', SchedulerArguments),
            'train': ('training_args', training_args_cls),
            'eval': ('eval_args', EvaluationArguments),
            'log': ('log_args', LogArguments),
            'rewards': ('reward_args', MultiRewardArguments),
            'eval_rewards': ('eval_reward_args', MultiRewardArguments),
        }

        # 3. Build init kwargs
        init_kwargs = {}
        extras = {}
        
        valid_field_names = {f.name for f in fields(cls)}

        for k, v in args_dict.items():
            if k in nested_map:
                arg_name, arg_cls = nested_map[k]
                init_kwargs[arg_name] = arg_cls.from_dict(v)
            
            elif k in valid_field_names:
                init_kwargs[k] = v
            
            else:
                extras[k] = v

        if extras:
            expected_top_level_keys = sorted(
                set(nested_map.keys()) | (valid_field_names - {"extra_kwargs"})
            )
            logger.warning(
                f"{cls.__name__}.from_dict captured {len(extras)} unknown top-level key(s) into extra_kwargs: "
                f"{sorted(extras.keys())}. "
                "Verify these are intentional (expected top-level keys are "
                f"{expected_top_level_keys}); typos will be silently accepted otherwise."
            )

        # 4. Handle explicit 'extra_kwargs' if present in YAML and merge
        if "extra_kwargs" in init_kwargs:
            extras.update(init_kwargs["extra_kwargs"])
        
        init_kwargs["extra_kwargs"] = extras
        
        return cls(**init_kwargs)

    @classmethod
    def load_from_yaml(cls, yaml_file: str) -> Arguments:
        """
        Load Arguments from a YAML configuration file.
        Example: args = Arguments.load_from_yaml("config.yaml")
        """
        with open(yaml_file, 'r', encoding='utf-8') as f:
            args_dict = yaml.safe_load(f)
        
        return cls.from_dict(args_dict)
    
    def __str__(self) -> str:
        """Pretty print configuration as YAML."""
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False, indent=2)
    
    def __repr__(self) -> str:
        """Same as __str__ for consistency."""
        return self.__str__()