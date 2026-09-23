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

import copy
import math
import warnings
from dataclasses import dataclass, field, fields
from datetime import datetime
from typing import Any, Dict, Literal, Optional

import yaml

from ..contracts.execution import AcquisitionMode, ExecutionContract, FeedbackMode
from ..utils.dist import get_world_size
from ..utils.logger_utils import setup_logger
from .abc import ArgABC
from .acceleration_args import AccelerationArguments
from .data_args import DataArguments
from .log_args import LogArguments
from .model_args import ModelArguments
from .optimizer_args import AdamWOptimizerArguments, MultiOptimizerArguments
from .reward_args import MultiRewardArguments, RewardArguments
from .scheduler_args import SchedulerArguments
from .training_args import (
    DiffusionOPDTrainingArguments,
    DMD2TrainingArguments,
    EvaluationArguments,
    TDMR1TrainingArguments,
    TDMTrainingArguments,
    TrainingArguments,
    get_training_args_class,
)

logger = setup_logger(__name__, rank_zero_only=True)


def _json_safe(obj: Any) -> Any:
    """Recursively coerce a config dict into a JSON-serializable form.

    Defense-in-depth for ``Arguments.to_dict`` export sinks (wandb / SwanLab
    config, ``json.dumps``):

    - ``dict``: recurse into values.
    - ``list`` / ``tuple``: recurse into elements; both normalize to ``list``
      (so a ``set`` nested inside a tuple is still coerced).
    - ``set`` / ``frozenset`` (e.g. user values placed in a reward's
      ``extra_kwargs``): convert to a ``list`` sorted by ``repr`` for
      deterministic output.
    - any other value: returned unchanged.

    Internal ``_``-prefixed dataclass fields are already dropped by
    :meth:`ArgABC.to_dict`, so this is purely a safety net.
    """
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted((_json_safe(v) for v in obj), key=repr)
    return obj


@dataclass
class Arguments(ArgABC):
    """
    Main arguments class encapsulating all configurations.
    """

    launcher: Literal["accelerate"] = field(
        default="accelerate",
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
    mixed_precision: Optional[Literal["no", "fp16", "bf16"]] = field(
        default="bf16",
        metadata={"help": "Mixed precision setting for training."},
    )
    # Runtime distributed fields (populated by reconcile_config after accelerator creation)
    process_index: Optional[int] = field(
        default=None,
        metadata={"help": "Global process index (set at runtime by reconcile_config)."},
    )
    local_process_index: Optional[int] = field(
        default=None,
        metadata={"help": "Local process index on this node (set at runtime)."},
    )
    num_machines: Optional[int] = field(
        default=None,
        metadata={"help": "Number of machines detected from environment."},
    )
    gpus_per_node: Optional[int] = field(
        default=None,
        metadata={"help": "GPUs per node (num_processes // num_machines)."},
    )
    machine_rank: Optional[int] = field(
        default=None,
        metadata={"help": "Rank of the current machine (set at runtime)."},
    )
    main_process_ip: Optional[str] = field(
        default=None,
        metadata={"help": "IP of the master node (set at runtime from env vars)."},
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
    acceleration_args: AccelerationArguments = field(
        default_factory=AccelerationArguments,
        metadata={"help": "Arguments for the acceleration plugin layer."},
    )
    reward_args: MultiRewardArguments = field(
        default_factory=MultiRewardArguments,
        metadata={"help": "Arguments for multiple reward configurations."},
    )
    eval_reward_args: Optional[MultiRewardArguments] = field(
        default=None,
        metadata={"help": "Arguments for multiple evaluation reward configurations."},
    )
    optimizer_args: MultiOptimizerArguments = field(
        default_factory=MultiOptimizerArguments,
        metadata={"help": "Arguments for one optimizer per trainable variant."},
    )

    def _synthesize_default_optimizer_args(self) -> None:
        """Derive one optimizer configuration from the flat training arguments.

        A single-policy run configures its optimizer with `train.learning_rate` and
        friends, and every shipped example does. Rather than teach each reader both
        spellings, the flat fields are translated here, once, into the same
        configuration a multi-variant run writes out explicitly.
        """
        if len(self.optimizer_args) > 0:
            # `optimizers:` owns the clip norm. Mirror the primary entry's value onto
            # training_args so the shared gradient step reads one resolved number
            # instead of re-deriving it, and so the two can never disagree.
            self.training_args.max_grad_norm = self.optimizer_args[0].max_grad_norm
            return
        self.optimizer_args = MultiOptimizerArguments(
            optimizer_configs=[
                AdamWOptimizerArguments(
                    learning_rate=self.training_args.learning_rate,
                    weight_decay=self.training_args.adam_weight_decay,
                    max_grad_norm=self.training_args.max_grad_norm,
                    betas=self.training_args.adam_betas,
                    eps=self.training_args.adam_epsilon,
                )
            ]
        )

    def _validate_multirole_training_contract(self) -> None:
        """Validate reward and dynamics contracts before batch geometry."""
        training_args = self.training_args
        if isinstance(training_args, TDMR1TrainingArguments):
            if not self.reward_args:
                raise ValueError(
                    "trainer_type='tdm-r1' requires at least one training reward, "
                    "but rewards is empty."
                )
            if training_args.group_size < 2:
                raise ValueError(
                    "trainer_type='tdm-r1' requires complete reward groups with "
                    f"group_size >= 2, received group_size={training_args.group_size}."
                )
        if (
            isinstance(training_args, TDMTrainingArguments)
            and self.scheduler_args.dynamics_type != "ODE"
        ):
            raise ValueError(
                f"trainer_type={training_args.trainer_type!r} requires "
                "scheduler.dynamics_type='ODE', received "
                f"dynamics_type={self.scheduler_args.dynamics_type!r}."
            )

    def _get_execution_contract(self) -> ExecutionContract:
        """Return the immutable algorithm contract used by configuration gating."""
        contract = getattr(type(self.training_args), "execution_contract", None)
        if not isinstance(contract, ExecutionContract):
            raise TypeError(
                f"training arguments {type(self.training_args).__name__}.execution_contract "
                f"must be ExecutionContract, got {type(contract).__name__}: {contract!r}"
            )
        return contract

    def _validate_training_feedback_contract(self) -> None:
        """Reject runtime training rewards when the algorithm declares no feedback."""
        if self._get_execution_contract().feedback is FeedbackMode.NONE and self.reward_args:
            raise ValueError(
                f"trainer_type={self.training_args.trainer_type!r} does not accept training "
                f"rewards, but received {len(self.reward_args)} reward configuration(s). "
                "Use eval_rewards for evaluation-only monitoring."
            )

    def _validate_dataset_acquisition_contract(self) -> None:
        """Keep complete data epochs independent from grouped generation geometry."""
        if self._get_execution_contract().acquisition is not AcquisitionMode.DATASET:
            return
        if not self.data_args.training_datasets:
            raise ValueError(
                "dataset acquisition requires at least one enabled training source under "
                "data.datasets; the legacy prompt-only data.dataset_dir path cannot carry "
                "demonstration or preference supervision"
            )
        if not self.data_args.enable_preprocess:
            raise ValueError(
                "dataset acquisition requires data.enable_preprocess=True so model-input "
                "conditions use the input-only preprocessing cache"
            )
        if self.data_args.sampler_type != "auto":
            raise ValueError(
                "dataset acquisition uses torch.utils.data.DistributedSampler directly; "
                f"data.sampler_type must remain 'auto', received "
                f"{self.data_args.sampler_type!r}"
            )
        bad_weights = [
            (dataset.name, dataset.train.weight)
            for dataset in self.data_args.training_datasets
            if dataset.train is not None and dataset.train.weight != 1
        ]
        if bad_weights:
            raise ValueError(
                "dataset acquisition requires train.weight=1 so one epoch remains one "
                f"complete dataloader traversal; received {bad_weights!r}"
            )

    def _validate_dmd2_batch_geometry(self) -> None:
        """Require one generator step per distillation outer iteration."""
        training_args = self.training_args
        if not isinstance(training_args, DMD2TrainingArguments):
            return

        if training_args.gradient_step_per_epoch != 1:
            raise ValueError(
                f"{training_args.trainer_type} requires gradient_step_per_epoch=1; "
                f"received gradient_step_per_epoch={training_args.gradient_step_per_epoch}, "
                f"num_batches_per_epoch={training_args.num_batches_per_epoch}"
            )
        accumulation_steps = training_args.gradient_accumulation_steps
        if (
            not isinstance(accumulation_steps, int)
            or isinstance(accumulation_steps, bool)
            or accumulation_steps < 1
        ):
            raise ValueError(
                f"{training_args.trainer_type} expected gradient_accumulation_steps >= 1 "
                "as an int; "
                f"received gradient_accumulation_steps={accumulation_steps!r}"
            )
        if (
            isinstance(training_args, TDMTrainingArguments)
            and accumulation_steps % training_args.num_inference_steps
        ):
            raise ValueError(
                f"{training_args.trainer_type} requires gradient_accumulation_steps to "
                f"be divisible by num_inference_steps={training_args.num_inference_steps}; "
                f"received gradient_accumulation_steps={accumulation_steps}"
            )

    def _validate_distillation_manual_geometry(self) -> None:
        """Reject distillation configs that do not tile, without auto-aligning them."""
        ta = self.training_args
        world_size = get_world_size()
        sample_num_per_iteration = world_size * ta.per_device_batch_size
        total = ta.unique_sample_num_per_epoch * ta.group_size
        if total % sample_num_per_iteration != 0:
            raise ValueError(
                f"{ta.trainer_type} does not auto-align unique_sample_num_per_epoch or "
                "group_size; expected unique_sample_num_per_epoch"
                f"({ta.unique_sample_num_per_epoch}) * group_size({ta.group_size}) % "
                f"(num_replicas({world_size}) * per_device_batch_size"
                f"({ta.per_device_batch_size})) == 0, received remainder "
                f"{total % sample_num_per_iteration}"
            )
        if not isinstance(ta, TDMR1TrainingArguments):
            return
        # Which shapes tile is a property of the resolved sampler, so the geometry is
        # checked against that one rather than against the strictest of them.
        sampler_type = self.data_args.sampler_type
        reason = self._tdm_r1_sampler_support().get(sampler_type)
        if reason is not None:
            raise ValueError(
                f"tdm-r1 does not auto-align its batch geometry, and sampler_type="
                f"{sampler_type!r} cannot deliver whole reward groups here: {reason}"
            )

    def __post_init__(self):
        if self.log_args.run_name is None:
            time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_args.run_name = f"{self.model_args.model_type}_{self.model_args.finetune_type}_{self.training_args.trainer_type}_{time_stamp}"

        self._synthesize_default_optimizer_args()
        self._validate_training_feedback_contract()
        self._validate_dataset_routing()
        self._validate_dataset_acquisition_contract()
        # Resolve `RewardArguments.applicable_datasets is None` -> concrete
        # list of applicable dataset names. Must run AFTER validation (so the
        # unknown-name check is against the user's raw input, not the
        # expanded list) and BEFORE any consumer reads the field.
        self._resolve_reward_dataset_routing()
        # Normalize `weight` from scalar/dict -> fully-expanded dict keyed
        # by applicable dataset names. Must run AFTER routing resolution
        # (so `applicable_datasets` is concrete).
        self._resolve_reward_weights()
        # With routing concrete, fail-fast when a training source has NO
        # applicable training reward — the user almost certainly meant
        # to add the source name to some reward's `applicable_datasets` list.
        self._validate_every_source_has_a_reward()
        # Stamp every `data.datasets[*]` entry with a stable monotonic
        # `source_id` matching its list position. Read by the dataloader
        # / trainer / gate everywhere a transport-friendly form of the
        # source string is wanted. Populates `data_args.source_id_to_name`.
        self._assign_source_ids()
        # Translate every `RewardArguments.applicable_datasets: List[str]` into the
        # `_datasets_resolved: frozenset[int]` cache used by the hot-path
        # gate. Trivially deterministic on every rank because IDs come
        # from the same shared config.
        self._resolve_reward_dataset_ids()
        # DiffusionOPD teacher routing (no-op for other trainers): each teacher's
        # `applicable_datasets` must reference declared training datasets. The
        # one-teacher-per-dataset rule is enforced in DiffusionOPDTrainer, not here
        # (the config schema stays permissive for a future multi-teacher trainer).
        self._validate_teacher_sources()
        self._resolve_scheduler_sde_defaults()
        self._validate_multirole_training_contract()
        if self._get_execution_contract().acquisition is AcquisitionMode.DATASET:
            return
        self._resolve_sampler_type()
        self._align_batch_geometry()
        self._adjust_gradient_accumulation()
        self._validate_dmd2_batch_geometry()

    def _assign_source_ids(self) -> None:
        """Stamp each ``data.datasets[*]`` entry with a stable monotonic id.

        The id is just the entry's index in ``data.datasets``; with YAML
        being deterministic and every rank receiving the same config dict,
        this gives identical id assignments on every rank with no extra
        sync.

        No-op when ``data.datasets`` is unset (legacy single-source mode);
        consumers fall back to the legacy "source is None -> applies to
        all" path.
        """
        if not self.data_args.datasets:
            return
        for source_id, ds in enumerate(self.data_args.datasets):
            ds.source_id = source_id

    def _resolve_reward_dataset_ids(self) -> None:
        """Populate ``RewardArguments._datasets_resolved`` (``frozenset[int]``).

        Hot-path callers (the reward gate) do an ``int in frozenset[int]``
        membership check instead of an ``str in List[str]`` scan plus a
        string allocation for ``__source__``. Builds the cache once at
        config-load time so the runtime gate stays O(1).

        Consumers that bypass ``Arguments`` entirely (e.g. tests
        constructing a stand-alone ``RewardProcessor``) leave the cache
        as ``None`` and the gate falls back to the string form — same
        observable behavior.
        """
        if not self.data_args.datasets:
            return
        name_to_id = self.data_args.source_name_to_id
        for rc in list(self.reward_args) + list(self.eval_reward_args or []):
            # `applicable_datasets` was resolved to a concrete `List[str]` upstream by
            # `_resolve_reward_dataset_routing`; the names are guaranteed to
            # be in the registry because they passed `_validate_dataset_routing`.
            if rc.applicable_datasets is None:
                raise RuntimeError(
                    "Internal error: RewardArguments.applicable_datasets not resolved before "
                    "_resolve_reward_dataset_ids; check Arguments.__post_init__ ordering."
                )
            rc._datasets_resolved = frozenset(name_to_id[n] for n in rc.applicable_datasets)

    def _resolve_reward_dataset_routing(self) -> None:
        """Replace ``RewardArguments.applicable_datasets is None`` with the explicit list.

        After this method returns, every reward's ``datasets`` field is a
        concrete ``List[str]``:

        * ``datasets=None`` (user wrote ``null`` or omitted the field) is
          replaced with the full list of applicable dataset names for that
          reward's side (training rewards -> training-source names; eval
          rewards -> eval-source names).
        * ``datasets=[]`` (explicit empty list) is left as-is and a warning
          is emitted, because the reward will never fire — almost certainly
          a misconfiguration but the user is allowed to express it.
        * ``datasets=[name1, ...]`` (explicit non-empty list) is left as-is
          (validation already ran).

        Consumers (RewardProcessor, MultiRewardLoader,
        AdvantageProcessor, log builders) can therefore read ``datasets``
        as a concrete list without any ``is None`` short-circuit.
        """
        train_names = [d.name for d in self.data_args.training_datasets]
        eval_names = [d.name for d in self.data_args.eval_datasets]

        def _resolve_one_side(reward_args, applicable_universe, side: str) -> None:
            if not reward_args:
                return
            for rc in reward_args:
                if rc.applicable_datasets is None:
                    # Eager copy: rewards never share their `applicable_datasets` list
                    # so mutating one cannot bleed into another.
                    rc.applicable_datasets = list(applicable_universe)
                elif len(rc.applicable_datasets) == 0:
                    logger.warning(
                        f"{side} reward '{rc.name}' has `datasets: []` — "
                        "it will never fire. If you intended 'apply to every "
                        "dataset', omit the field or set it to null."
                    )
                # else: explicit non-empty list, leave alone (validation
                # already confirmed every name is in the universe).

        _resolve_one_side(self.reward_args, train_names, side="Training")
        _resolve_one_side(self.eval_reward_args, eval_names, side="Eval")

    def _resolve_reward_weights(self) -> None:
        """Normalize ``RewardArguments.weight`` to a fully-expanded dict.

        After this method returns, every reward's ``weight`` is a
        ``Dict[str, float]`` keyed by its applicable dataset names.

        * ``weight: 1.0`` (scalar) -> ``{ds1: 1.0, ds2: 1.0, ...}``
        * ``weight: {ds1: 2.0}`` (partial dict) -> ``{ds1: 2.0, ds2: 1.0, ...}``
          (missing keys filled with ``1.0``)

        Validates that every key in the dict form is a known applicable
        dataset name.  Skipped when ``applicable_datasets`` is ``None``
        (pre-resolution; should not happen after routing resolution).
        """
        all_rewards = list(self.reward_args) + list(self.eval_reward_args or [])
        for rc in all_rewards:
            if rc.applicable_datasets is None:
                continue
            if isinstance(rc.weight, dict):
                unknown = set(rc.weight.keys()) - set(rc.applicable_datasets)
                if unknown:
                    raise ValueError(
                        f"Reward '{rc.name}' has per-dataset weight keys "
                        f"{sorted(unknown)} that are not in its "
                        f"applicable_datasets={rc.applicable_datasets!r}."
                    )
                expanded = {ds: rc.weight.get(ds, 1.0) for ds in rc.applicable_datasets}
                rc.weight = expanded
            else:
                rc.weight = {ds: float(rc.weight) for ds in rc.applicable_datasets}

    def _validate_every_source_has_a_reward(self) -> None:
        """Inverse routing check: every training/eval source must be covered.

        ``_resolve_reward_dataset_routing`` already turned every reward's
        ``datasets`` into a concrete list.  Now, for each declared
        training source, check that AT LEAST ONE training reward routes
        to it; same for the eval side.  A source with no applicable
        reward would silently produce all-NaN advantages at runtime
        (caught only later by ``AdvantageProcessor``'s ``weight_sum == 0``
        guard) — much better to fail at config-load time with a clear
        message naming the missing source.

        Skipped when no rewards are configured on the relevant side
        (legitimate for eval-only or sample-only runs).
        """
        train_names = [d.name for d in self.data_args.training_datasets]
        eval_names = [d.name for d in self.data_args.eval_datasets]

        def _check_side(reward_args, source_names, side: str) -> None:
            if not reward_args or not source_names:
                return
            covered: set[str] = set()
            for rc in reward_args:
                # `rc.applicable_datasets` is concrete post-resolver.
                covered.update(rc.applicable_datasets or [])
            uncovered = [n for n in source_names if n not in covered]
            if uncovered:
                raise ValueError(
                    f"{side} source(s) {uncovered!r} have NO applicable {side.lower()} "
                    f"reward — every reward's `applicable_datasets` field excludes them. Either "
                    f"add at least one of these names to a reward's `applicable_datasets`, set "
                    f"that reward's `applicable_datasets` to `null` (= apply to every source), "
                    f"or drop the dataset entry from `data.datasets`."
                )

        if self._get_execution_contract().feedback is FeedbackMode.RUNTIME_REWARD:
            _check_side(self.reward_args, train_names, side="Training")
        _check_side(self.eval_reward_args, eval_names, side="Eval")

    def _validate_teacher_sources(self) -> None:
        """Validate DiffusionOPD teacher routing (OPD trainer only).

        Every ``TeacherConfig.applicable_datasets`` entry must reference a
        declared training dataset (``data.datasets[*].name`` with training
        enabled). The schema deliberately permits several teachers to share a
        dataset; the one-teacher-per-dataset constraint is enforced by
        ``DiffusionOPDTrainer`` (not here), so a future multi-teacher/ensemble
        trainer can reuse this config unchanged.

        Gated on ``isinstance(DiffusionOPDTrainingArguments)`` so it is a no-op
        for non-OPD trainers. (A plain ``getattr(ta, "teachers", ...)`` would be
        unsafe: ``ArgABC.__getattr__`` falls back to ``extra_kwargs``, so a stray
        ``teachers:`` key in a non-OPD YAML would be picked up here and raise a
        confusing error.)
        """
        ta = self.training_args
        if not isinstance(ta, DiffusionOPDTrainingArguments):
            return
        teachers = ta.teachers
        if not teachers:
            return

        train_names = {d.name for d in self.data_args.training_datasets}
        if not train_names:
            raise ValueError(
                "DiffusionOPD requires `data.datasets` with at least one training dataset, "
                "but none were found. Declare each teacher's dataset under `data.datasets`."
            )

        for i, teacher in enumerate(teachers):
            unknown = [d for d in teacher.applicable_datasets if d not in train_names]
            if unknown:
                raise ValueError(
                    f"DiffusionOPD teacher[{i}] (name={teacher.name!r}, path={teacher.path!r}) references "
                    f"dataset(s) {unknown} not in training datasets {sorted(train_names)}. "
                    "Each `applicable_datasets` entry must match a `data.datasets[*].name` whose `train` is enabled."
                )

        # Reverse check: every active training dataset must be claimed by at
        # least one teacher (mirrors `_validate_every_source_has_a_reward`).
        # The one-teacher-per-dataset (no-overlap) rule stays in
        # DiffusionOPDTrainer; here we only require full coverage so an
        # uncovered dataset fails at config parse, not mid-rollout.
        covered = {ds for t in teachers for ds in t.applicable_datasets}
        uncovered = sorted(train_names - covered)
        if uncovered:
            raise ValueError(
                f"DiffusionOPD training dataset(s) {uncovered} are not distilled by any teacher. "
                f"Each `data.datasets[*].name` with `train.enabled` must appear in exactly one "
                f"teacher's `applicable_datasets`. Declared teachers cover: {sorted(covered)}."
            )

    def _validate_dataset_routing(self) -> None:
        """Validate the unified ``data.datasets`` schema and reward routing.

        Single source of truth for cross-cutting checks involving the new
        :class:`DatasetArguments` list.  Covers:

        * Mutual exclusion with the legacy single ``data.dataset_dir`` path
          when the user has actually customized that field (we cannot
          reject the dataclass default ``"data"`` itself, as many configs
          set it explicitly).
        * Uniqueness of dataset names across the ``data.datasets`` list.
        * ``train.weight > 0`` for every training participant.
        * Cross-validation of every ``RewardArguments.applicable_datasets`` entry
          against the union of declared training / eval dataset names.
          Training rewards must reference training-source names; eval
          rewards must reference eval-source names.
        """
        tds_unified = self.data_args.datasets or []
        if not tds_unified:
            return

        # Mutual exclusion: `data.dataset_dir` (non-default) conflicts
        # with `data.datasets`. Users must migrate to the unified schema.
        default_dataset_dir = next(
            (f.default for f in fields(self.data_args.__class__) if f.name == "dataset_dir"),
            None,
        )
        if default_dataset_dir is not None and self.data_args.dataset_dir != default_dataset_dir:
            raise ValueError(
                "Both `data.dataset_dir` (custom value: "
                f"{self.data_args.dataset_dir!r}) and `data.datasets` are set. "
                "These are mutually exclusive. Use `data.datasets` exclusively — "
                "move `dataset_dir` into each dataset entry's `dataset_dir` field."
            )

        # Unique names within the unified list.
        names = [d.name for d in tds_unified]
        if len(names) != len(set(names)):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(
                f"Duplicate `data.datasets` names: {dupes}. "
                "Each dataset entry must have a unique name."
            )

        # Weight > 0 for every training participant.
        bad_weights: list[tuple[str, float | None]] = []
        for d in tds_unified:
            if not d.is_training_source:
                continue
            if d.train is None:
                raise RuntimeError(
                    f"Internal error: dataset '{d.name}' reports is_training_source=True "
                    "but has train=None."
                )
            w = d.train.weight
            if w <= 0:
                bad_weights.append((d.name, w))
        if bad_weights:
            raise ValueError(
                f"All training datasets must have train.weight > 0; got: {bad_weights}. "
                "Set `train.weight: 1.0` for uniform mixing."
            )

        # Cross-validate reward `applicable_datasets` references.
        train_names = {d.name for d in tds_unified if d.is_training_source}
        eval_names = {d.name for d in tds_unified if d.is_eval_source}

        # Training rewards: `applicable_datasets` must reference TRAINING-source names.
        for rc in self.reward_args:
            if rc.applicable_datasets is None:
                continue
            if not train_names:
                raise ValueError(
                    f"Reward '{rc.name}' has applicable_datasets={rc.applicable_datasets!r} but no training "
                    "dataset is configured under `data.datasets`. Either remove "
                    "`applicable_datasets` from this reward or define training datasets it can route to."
                )
            unknown = set(rc.applicable_datasets) - train_names
            if unknown:
                raise ValueError(
                    f"Reward '{rc.name}' references unknown training dataset(s): "
                    f"{sorted(unknown)}. Valid training dataset names: {sorted(train_names)}."
                )

        # Eval rewards: `applicable_datasets` must reference EVAL-source names.
        if self.eval_reward_args:
            for rc in self.eval_reward_args:
                if rc.applicable_datasets is None:
                    continue
                if not eval_names:
                    raise ValueError(
                        f"Eval reward '{rc.name}' has applicable_datasets={rc.applicable_datasets!r} but no eval "
                        "dataset is configured under `data.datasets`. Either remove "
                        "`applicable_datasets` from this eval reward or define eval datasets it can route to."
                    )
                unknown = set(rc.applicable_datasets) - eval_names
                if unknown:
                    raise ValueError(
                        f"Eval reward '{rc.name}' references unknown eval dataset(s): "
                        f"{sorted(unknown)}. Valid eval dataset names: {sorted(eval_names)}."
                    )

    def _resolve_sampler_type(self) -> None:
        """Choose the distributed sampler strategy.

        Writes the resolved value back to ``data_args.sampler_type`` so all
        downstream consumers (``get_data_sampler``, ``RewardProcessor``,
        ``AdvantageProcessor``) read a concrete choice — never ``"auto"``.

        Rules:
        - DGPO is always forced to ``group_distributed``.
        - TDM-R1 resolves between ``group_contiguous`` and
          ``group_distributed`` according to group geometry and overlap mode.
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

        self._has_async_rewards = any(getattr(cfg, "async_reward", False) for cfg in all_configs)

        # 2. Resolve sampler type
        ta = self.training_args
        user_choice = self.data_args.sampler_type

        trainer_type = str(ta.trainer_type).lower()

        if trainer_type == "tdm-r1":
            self.data_args.sampler_type = self._resolve_tdm_r1_sampler_type(user_choice)
            return

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
            groups_per_rank_ok = m % world_size == 0
            local_batch_tiling_ok = m // world_size * ta.group_size % ta.per_device_batch_size == 0
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

    def _tdm_r1_sampler_support(self) -> Dict[str, Optional[str]]:
        """Report why each sampler can or cannot serve TDM-R1's group preference.

        Returns:
            Sampler name to the reason it is unusable, or ``None`` when usable.
        """
        ta = self.training_args
        world_size = get_world_size()
        group_size = ta.group_size
        batch_size = ta.per_device_batch_size
        unique_num = ta.unique_sample_num_per_epoch

        contiguous_reason: Optional[str] = None
        if batch_size % group_size != 0:
            contiguous_reason = (
                f"per_device_batch_size({batch_size}) % group_size({group_size}) != 0, so a "
                "rank-local microbatch would straddle a group boundary"
            )
        elif unique_num % world_size != 0:
            contiguous_reason = (
                f"unique_sample_num_per_epoch({unique_num}) % num_replicas({world_size}) != 0, "
                "so complete groups cannot be dealt one rank at a time"
            )

        distributed_reason: Optional[str] = None
        if group_size > world_size * batch_size:
            distributed_reason = (
                f"group_size({group_size}) exceeds one global microbatch "
                f"({world_size * batch_size})"
            )
        elif (world_size * batch_size) % group_size != 0:
            distributed_reason = (
                f"num_replicas({world_size}) * per_device_batch_size({batch_size}) % "
                f"group_size({group_size}) != 0, so a global microbatch would not hold whole groups"
            )

        return {
            "group_contiguous": contiguous_reason,
            "group_distributed": distributed_reason,
        }

    def _resolve_tdm_r1_sampler_type(self, user_choice: str) -> str:
        """Pick the sampler whose group layout TDM-R1's preference loss can read.

        TDM-R1 needs each preference microbatch to carry whole reward groups, but that
        is a property of the sampler rather than a constraint the batch shape has to
        satisfy: ``group_contiguous`` keeps a whole group on one rank, while
        ``group_distributed`` packs complete groups into a global microbatch and sums
        partial group logits across ranks. Requiring only the former forced
        ``per_device_batch_size % group_size == 0`` on every run, which rules out the
        common case of one group per global step.

        Args:
            user_choice: Configured ``data.sampler_type``, possibly ``"auto"``.

        Returns:
            The resolved sampler name.

        Raises:
            ValueError: If the choice cannot deliver whole groups, or if no sampler can.
        """
        support = self._tdm_r1_sampler_support()

        # Group-wise async rewards are scored on one rank, so they need the whole group
        # there regardless of what the preference loss could otherwise handle.
        if self._has_async_rewards and not getattr(
            self.training_args,
            "reward_optimization_overlap",
            False,
        ):
            reason = support["group_contiguous"]
            if reason is not None:
                raise ValueError(
                    "tdm-r1 with async rewards requires sampler_type='group_contiguous', "
                    f"which this geometry cannot satisfy: {reason}"
                )
            if user_choice not in {"auto", "group_contiguous"}:
                logger.warning(
                    f"Async rewards require 'group_contiguous' sampler. Overriding "
                    f"'{user_choice}' -> 'group_contiguous'."
                )
            return "group_contiguous"

        if user_choice == "distributed_k_repeat":
            logger.warning(
                "TDM-R1 cannot use sampler_type='distributed_k_repeat': it scatters a group's "
                "members over arbitrary ranks, leaving each rank a different set of partial "
                "groups and no shared group-id space to sum the preference logits in. "
                "Selecting a group-preserving sampler instead."
            )
            user_choice = "auto"

        if user_choice in support:
            reason = support[user_choice]
            if reason is not None:
                raise ValueError(
                    f"tdm-r1 cannot use sampler_type={user_choice!r} with this batch geometry: "
                    f"{reason}"
                )
            return user_choice

        # Prefer the rank-local layout: it reads its group logits without a collective.
        if support["group_contiguous"] is None:
            return "group_contiguous"
        if support["group_distributed"] is None:
            return "group_distributed"
        raise ValueError(
            "tdm-r1 needs each microbatch to carry whole reward groups and no sampler can "
            "deliver that here. 'group_contiguous' is unusable because "
            f"{support['group_contiguous']}; 'group_distributed' is unusable because "
            f"{support['group_distributed']}. Set per_device_batch_size to a multiple of "
            "group_size, or choose a group_size that divides the global microbatch."
        )

    def _align_batch_geometry(self) -> None:
        """Align ``unique_sample_num_per_epoch`` to sampler constraints.

        Recompute derived batch quantities afterward. Group-distributed geometry
        validates and preserves the configured ``group_size``.

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

        if isinstance(self.training_args, DMD2TrainingArguments):
            self._validate_distillation_manual_geometry()
            # Refusing to auto-align the total is not the same as refusing to say how it
            # divides between sources. The multi-source loader reads the per-source count
            # off each training spec, so skipping the write-back left it None and the
            # dataloader raised "per-source unique_sample_num_per_epoch is missing".
            # `step=1` makes the shared allocator a pure partition: it distributes the
            # total the config already validated instead of rounding it up.
            self._align_unique_sample_num(
                sampler_name=self.training_args.trainer_type,
                base_step_func=lambda: 1,
            )
            self._recompute_derived_batch_quantities()
            return
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
            f"* per_device_batch_size({ta.per_device_batch_size})" + constraint_suffix + " == 0"
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
            ta.unique_sample_num_per_epoch * ta.group_size
        ) // sample_num_per_iteration
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
        self._align_unique_sample_num(
            sampler_name="DistributedKRepeatSampler",
            base_step_func=self._base_unique_sample_step,
        )

    def _align_for_group_contiguous(self) -> None:
        """``GroupContiguousSampler``: base constraint + ``unique_sample_num_per_epoch % num_replicas == 0``."""
        world_size = get_world_size()

        def _step() -> int:
            return math.lcm(self._base_unique_sample_step(), world_size)

        # Per-sampler `extra_line` so the warning explains why the alignment
        # bumps `M` further than the base step.  Lazily formatted inside the
        # primitive so the per-sampler-specific text only fires on the
        # legacy (single-source) path; the multi-source path includes a
        # source-breakdown line of its own.
        def _extra(new_M: int) -> str:
            return (
                f"  1) unique_sample_num_per_epoch({new_M}) "
                f"% num_replicas({world_size}) == 0\n  2)"
            )

        self._align_unique_sample_num(
            sampler_name="GroupContiguousSampler",
            base_step_func=_step,
            extra_line_for_legacy=_extra,
        )

    def _align_for_group_distributed(self) -> None:
        """Validate global group packing, then align the unique prompt count.

        A group may be smaller than the world size. The sampler packs complete
        groups into the rank-major global microbatch, so only ``K <= W*B`` and
        ``(W*B) % K == 0`` are required; the configured group size is never
        silently changed.
        """
        ta = self.training_args
        if ta.group_size <= 0:
            raise ValueError(f"group_size must be positive, got {ta.group_size}.")

        world_size = get_world_size()
        per_device_batch_size = ta.per_device_batch_size
        sample_num_per_iteration = world_size * per_device_batch_size
        group_size = ta.group_size

        if group_size > sample_num_per_iteration:
            raise ValueError(
                "sampler_type='group_distributed' requires "
                "`group_size <= num_replicas * per_device_batch_size`; "
                f"got group_size={group_size}, "
                f"num_replicas * per_device_batch_size={sample_num_per_iteration}."
            )
        if sample_num_per_iteration % group_size:
            raise ValueError(
                "sampler_type='group_distributed' requires "
                "`(num_replicas * per_device_batch_size) % group_size == 0`; "
                f"got {world_size} * {per_device_batch_size} % {group_size} != 0."
            )

        self._align_unique_sample_num(
            sampler_name="GroupDistributedSampler",
            base_step_func=self._base_unique_sample_step,
        )

    # ---------------------------------------------------------------------
    # Shared alignment primitive (legacy + multi-source dispatch)
    # ---------------------------------------------------------------------
    def _align_unique_sample_num(
        self,
        *,
        sampler_name: str,
        base_step_func,
        extra_line_for_legacy=None,
    ) -> None:
        """Shared core for the three per-sampler ``_align_for_*`` helpers.

        Dispatches on whether ``data.datasets`` declares more than one
        training-eligible source:

        * **Legacy / single-source** — round ``unique_sample_num_per_epoch``
          up to the next multiple of ``base_step_func()``; emit the
          standard "adjusted" warning when the value changes.  Behaviour
          is byte-identical to the pre-refactor code paths.

        * **Multi-source (N >= 2)** — exact allocator: round the total
          up to a multiple of ``step * sum(weights)`` so per-source
          ``M_i = M_total * w_i / sum(weights)`` is itself a positive
          integer multiple of ``step``.  Geometric consequence: every
          batch comes from a single source (no per-batch homogeneity
          contract to defend at runtime; it falls out of the math).

        ``extra_line_for_legacy`` is a one-shot text producer for the
        legacy path's per-sampler constraint description; it's only
        invoked on the single-source code path so the multi-source
        warning can supply its own per-source breakdown without
        duplicating the sampler-specific blurb.
        """
        ta = self.training_args
        step = base_step_func()
        tds = self.data_args.training_datasets  # property; List[DatasetArguments]
        N = len(tds)

        if N <= 1:
            new_M = self._round_up_to_step(ta.unique_sample_num_per_epoch, step)
            if new_M != ta.unique_sample_num_per_epoch:
                extra = extra_line_for_legacy(new_M) if extra_line_for_legacy else ""
                self._warn_and_assign_unique_sample_num(new_M, sampler_name, extra)
            # Stamp the resolved M_i onto each training spec so
            # `print(config)` reflects the final geometry.
            if N == 1:
                tds[0].train.unique_sample_num_per_epoch = ta.unique_sample_num_per_epoch  # type: ignore[union-attr]
            return

        # Multi-source partition (N >= 2).
        # Step 1: target total = M rounded up so that
        #   (a) M_total is a multiple of `step` (per-source sampler constraint), and
        #   (b) M_total is a multiple of `step * sum(weights)` so the per-source
        #       allocation `M_i = M_total * w_i / sum(weights)` is itself a positive
        #       integer multiple of `step` -- no remainder loop, no rounding drift,
        #       and every batch is geometrically guaranteed to come from a single
        #       source (`num_batches_per_epoch % sum(weights) == 0` -> the scheduler
        #       can place each source's quota exactly).
        original_M = ta.unique_sample_num_per_epoch
        weights = [int(d.train.weight) for d in tds]  # type: ignore[union-attr]
        W_sum = sum(weights)
        if W_sum <= 0:
            # _validate_dataset_routing should have caught this; assert defensively.
            raise ValueError(f"sum(train.weight) must be > 0; got weights={weights}.")
        partition_step = step * W_sum
        target_total = max(self._round_up_to_step(original_M, partition_step), partition_step)
        # Exact allocation: M_i = (target_total / sum(w)) * w_i = step * j * w_i.
        per_source_unit = target_total // W_sum  # multiple of `step` by construction
        partition = {d.name: per_source_unit * w for d, w in zip(tds, weights)}
        final_total = sum(partition.values())
        if final_total != target_total:
            raise RuntimeError(
                f"Internal error: partition sum ({final_total}) != target_total "
                f"({target_total}). partition={partition}."
            )

        if final_total != original_M:
            breakdown = ", ".join(f"{n}={v}" for n, v in sorted(partition.items()))
            extra = (
                f"  multi-source partition ({len(tds)} sources, "
                f"sum(weight)={W_sum}): {breakdown}\n  "
            )
            self._warn_and_assign_unique_sample_num(final_total, sampler_name, extra)
        # Stamp resolved M_i onto each training spec so the printed
        # config shows the final geometry per-source.
        for d in tds:
            if d.train is None:
                raise RuntimeError(
                    f"Internal error: training dataset '{d.name}' has train=None "
                    "during partition writeback."
                )
            d.train.unique_sample_num_per_epoch = partition[d.name]

    def _adjust_gradient_accumulation(self) -> None:
        """Adjust gradient accumulation for per-timestep losses.

        Must run AFTER `_align_batch_geometry()` which finalises the base
        gradient_accumulation_steps from the aligned M.
        Skipped when gradient_accumulation_steps is manually set — the user
        value is treated as final.
        """
        if not self.training_args._manual_gradient_accumulation_steps:
            if isinstance(self.training_args, DMD2TrainingArguments) and not isinstance(
                self.training_args,
                TDMTrainingArguments,
            ):
                self.training_args.gradient_accumulation_steps = 1
            else:
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
        if sched.dynamics_type == "ODE":
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
            # Internal/runtime fields (e.g. distributed cache) never export.
            if f.name.startswith("_"):
                continue
            value = getattr(self, f.name)
            if value is None:
                continue
            if isinstance(value, ArgABC):
                # Remove '_args' suffix for nested configs
                key = f.name.replace("_args", "")
                result[key] = value.to_dict()
            elif isinstance(value, list) and value and isinstance(value[0], ArgABC):
                # List of ArgABC instances (e.g. eval_datasets)
                result[f.name] = [item.to_dict() for item in value]
            else:
                result[f.name] = value

        extras = result.pop("extra_kwargs", {})
        result.update(extras)
        # Final JSON-safe pass: coerce any nested set/frozenset (e.g. from
        # extra_kwargs) so the exported config survives json.dumps and any
        # logging backend that serializes it.
        return _json_safe(result)

    @classmethod
    def from_dict(cls, args_dict: dict[str, Any]) -> Arguments:
        """Create Arguments instance from dictionary.

        Side effects: a top-level ``eval_datasets:`` key (legacy schema
        introduced earlier on this branch) is migrated into
        ``data.datasets[*].eval`` with a :class:`DeprecationWarning`.
        The migration shim is removed in a future release; in the
        meantime callers should prefer the unified ``data.datasets:``
        list (each entry opts into training and/or evaluation via its
        ``train`` / ``eval`` sub-blocks).
        """
        # 0. Top-level eval_datasets deprecation shim
        # ------------------------------------------------
        # Migrate `eval_datasets: [...]` (top level) into
        # `data.datasets: [{name, dataset_dir, ..., eval: {...}}]`.
        # We keep the migrated dict shape consistent with the new schema
        # so the rest of `from_dict` (and `Arguments.__post_init__`) does
        # not need to know about the legacy location.
        args_dict = cls._migrate_legacy_eval_datasets(args_dict)

        # 1. Resolve TrainingArguments subclass based on trainer_type
        train_dict = args_dict.get("train", {})
        trainer_type = train_dict.get("trainer_type", "grpo")
        training_args_cls = get_training_args_class(trainer_type)

        # 2. Nested arguments map
        nested_map = {
            "data": ("data_args", DataArguments),
            "model": ("model_args", ModelArguments),
            "scheduler": ("scheduler_args", SchedulerArguments),
            "train": ("training_args", training_args_cls),
            "eval": ("eval_args", EvaluationArguments),
            "log": ("log_args", LogArguments),
            "acceleration": ("acceleration_args", AccelerationArguments),
            "optimizers": ("optimizer_args", MultiOptimizerArguments),
            "rewards": ("reward_args", MultiRewardArguments),
            "eval_rewards": ("eval_reward_args", MultiRewardArguments),
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
        with open(yaml_file, "r", encoding="utf-8") as f:
            args_dict = yaml.safe_load(f)

        return cls.from_dict(args_dict)

    @staticmethod
    def _migrate_legacy_eval_datasets(args_dict: dict[str, Any]) -> dict[str, Any]:
        """Auto-migrate the legacy top-level ``eval_datasets:`` YAML key.

        Converts each legacy top-level ``eval_datasets:`` entry dict into a
        ``DatasetArguments`` entry under ``data.datasets`` whose
        ``eval:`` sub-block carries the same overrides.  Top-level
        ``eval_datasets`` is removed from the input dict so the rest of
        ``from_dict`` sees the canonical schema only.

        Idempotent for fully-migrated configs.  We deep-copy the input
        before mutating so the caller's dict (e.g. a YAML round-trip
        cache held by a test harness) is never modified — defensive, since
        we mutate nested ``data.datasets`` lists below.

        Notes:
        - Eval-only datasets (no ``train:`` block) are perfectly valid
          under the unified schema, so no fabricated ``train:`` field is
          inserted.
        - When a ``data.datasets`` entry with the same ``name`` is
          already present, the migration MERGES the legacy ``eval:``
          fields into it (rather than creating a duplicate, which
          would trip the duplicate-name validator).
        """
        if "eval_datasets" not in args_dict:
            return args_dict

        legacy = args_dict["eval_datasets"]
        if not isinstance(legacy, list) or not legacy:
            # Empty list or weird shape — drop and let the validator complain.
            new_dict = dict(args_dict)
            new_dict.pop("eval_datasets", None)
            return new_dict

        warnings.warn(
            "Top-level `eval_datasets:` is deprecated and will be removed in a "
            "future release. Move each entry under `data.datasets:` and put "
            "its eval-specific fields (split / num_inference_steps / "
            "guidance_scale / resolution / max_dataset_size) inside an `eval:` "
            "sub-block. The shared `dataset_dir` / `image_dir` / `video_dir` / "
            "`audio_dir` move to the parent dataset entry.",
            DeprecationWarning,
            stacklevel=3,
        )

        # Deep-copy so we never mutate the caller's dict (or its
        # nested data/datasets list, which we may extend in place below).
        new_dict = copy.deepcopy(args_dict)
        data_dict = new_dict.get("data") or {}
        if not isinstance(data_dict, dict):
            data_dict = {}
        existing_datasets: list = list(data_dict.get("datasets") or [])
        # Index existing entries by name for in-place merge below.
        by_name: dict[str, dict] = {}
        for i, d in enumerate(existing_datasets):
            if isinstance(d, dict) and "name" in d:
                by_name[d["name"]] = d

        # Field categorisation: a legacy top-level eval_datasets entry
        # carries both parent-level fields (name / dataset_dir / media
        # roots) and DatasetEvalSpec-level fields. Split accordingly.
        _PARENT_KEYS = {"name", "dataset_dir", "image_dir", "video_dir", "audio_dir"}
        _EVAL_SPEC_KEYS = {
            "split",
            "max_dataset_size",
            "resolution",
            "num_inference_steps",
            "guidance_scale",
        }

        for entry in legacy:
            if not isinstance(entry, dict):
                raise TypeError(
                    f"Legacy `eval_datasets:` entries must be dicts, got "
                    f"{type(entry).__name__}."
                )
            name = entry.get("name")
            if not name:
                raise ValueError(
                    "Legacy `eval_datasets:` entry is missing a `name`. "
                    "Migrate manually to the unified `data.datasets:` schema."
                )

            parent_fields = {k: v for k, v in entry.items() if k in _PARENT_KEYS}
            eval_fields = {k: v for k, v in entry.items() if k in _EVAL_SPEC_KEYS}
            unknown = set(entry) - _PARENT_KEYS - _EVAL_SPEC_KEYS
            if unknown:
                # Surface unknown keys as DatasetArguments extras (extra_kwargs)
                # rather than silently dropping them.  Same shape as ArgABC.
                parent_fields.setdefault("extra_kwargs", {}).update({k: entry[k] for k in unknown})

            if name in by_name:
                # Merge eval-spec fields into existing entry's `eval` block.
                target = by_name[name]
                eval_block = target.get("eval")
                if eval_block is None:
                    target["eval"] = eval_fields
                elif isinstance(eval_block, dict):
                    # Newer schema wins on conflict (defensive — should be rare).
                    eval_block_merged = {**eval_fields, **eval_block}
                    target["eval"] = eval_block_merged
                # Don't overwrite parent fields when merging — the unified
                # entry is authoritative.
            else:
                # Create a fresh dataset entry from the legacy fields.
                migrated = {**parent_fields, "eval": eval_fields}
                existing_datasets.append(migrated)
                by_name[name] = migrated

        data_dict["datasets"] = existing_datasets
        new_dict["data"] = data_dict
        new_dict.pop("eval_datasets", None)
        return new_dict

    def __str__(self) -> str:
        """Pretty print configuration as YAML."""
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False, indent=2)

    def __repr__(self) -> str:
        """Same as __str__ for consistency."""
        return self.__str__()
