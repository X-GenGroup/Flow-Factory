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

# src/flow_factory/trainers/distillation/opd/trainer.py
"""DiffusionOPD on-policy distillation trainer.

Distills several task-specialized LoRA teachers into a single student along
the student's own rollout trajectories. The target space is configurable as
the one-step transition mean (``xt``), velocity (``v``), or predicted clean
latent (``x0``), with optional detached self-normalization. The transition-mean
target supports ODE and SDE dynamics; velocity-derived targets are ODE-only.

The distilled denoising steps are selected by ``train.timestep_range`` (a
fraction band of the trajectory step indices; default 0.99 = upstream
``timestep_fraction``), NOT the SDE-only ``scheduler.train_timesteps`` (which is
empty under ODE). See ``_select_train_step_indices``.

Design (2-pass, per epoch):
  sample()    -> student rolls out on-policy trajectories (tagged by source),
                 reusing the standard ``generate_samples`` pipeline.
  optimize()  -> PASS 1 (no_grad): for each teacher (ONE weight swap), forward
                 over its routed samples' stored states x_j and cache the teacher
                 target on each sample.
              -> PASS 2 (student params only): standard gradient loop that
                 forwards the student at the same x_j and matches its projected
                 target to the cached teacher target.

This keeps teacher swaps to M-per-epoch, runs the gradient loop with student
params only (no autocast-cache disable, no DDP bypass), and reuses proven FF
trajectory-replay primitives shared with GRPO.

References:
    [1] On-Policy Distillation of Diffusion Models (https://github.com/ali-vilab/DiffusionOPD)
"""

from __future__ import annotations

import math
import os
from collections import defaultdict
from functools import partial
from typing import Any, ClassVar, Dict, List, Mapping, Optional, Tuple, Union, cast

import torch
import tqdm as tqdm_

tqdm = partial(tqdm_.tqdm, dynamic_ncols=True)

from ....contracts.execution import (
    ONLINE_NO_FEEDBACK_EXECUTION_CONTRACT,
    ExecutionContract,
)
from ....hparams import DiffusionOPDTrainingArguments
from ....hparams.training_args.opd import resolve_distill_step_band
from ....samples import (
    BaseSample,
    LatentState,
    MultiModalStepOutput,
    ReplayStep,
    StackedSampleBatch,
)
from ....utils.logger_utils import setup_logger
from ....utils.trajectory_collector import compute_trajectory_indices
from ...abc import BaseTrainer
from ...common.forward_kwargs import replay_forward_kwargs
from .common import (
    TARGET_REQUEST_FIELDS,
    compute_structured_distillation_loss,
    load_teachers,
    project_distillation_target_state,
    resolve_scheduler_group_dynamics,
)

logger = setup_logger(__name__)


class DiffusionOPDTrainer(BaseTrainer):
    """Multi-teacher on-policy distillation trainer (ODE + SDE)."""

    # Distillation paradigm: no reward/advantage stage and rollout log-probs do not
    # enter the loss, so lossy rollout acceleration is permitted (constraints.md #7).
    paradigm = "distillation"
    execution_contract: ClassVar[ExecutionContract] = ONLINE_NO_FEEDBACK_EXECUTION_CONTRACT

    def _algorithm_runtime_child_names(self) -> Tuple[str, ...]:
        """Declare teacher snapshot names directly from validated configuration."""
        return tuple(
            teacher.name or f"opd_teacher_{index}"
            for index, teacher in enumerate(self.training_args.teachers)
        )

    def _initialize_snapshots(self) -> None:
        """Realize every teacher snapshot before exact-state compatibility checks.

        Exact resume creates shape-compatible placeholders rather than loading the
        external teacher checkpoints and mutating the prepared student before the
        runtime manifest has passed preflight. The checkpoint payload replaces those
        placeholders after Accelerator restores the core state.
        """
        teacher_names = self._algorithm_runtime_child_names()
        state_resume = bool(self.model_args.resume_path and self.model_args.resume_type == "state")
        if state_resume:
            if self.model_args.finetune_type != "lora":
                raise ValueError(
                    "DiffusionOPD teacher snapshots require LoRA finetuning, but "
                    f"model_args.finetune_type={self.model_args.finetune_type!r}."
                )
            target_components = [
                component
                for component, modules in self.adapter.target_module_map.items()
                if modules
            ]
            if not target_components:
                raise ValueError(
                    "DiffusionOPD adapter has no trainable LoRA components for "
                    "teacher snapshot placeholders."
                )
            snapshot_device = (
                self.accelerator.device
                if self.training_args.teacher_param_device == "cuda"
                else torch.device("cpu")
            )
            for teacher_name in teacher_names:
                self.adapter.add_named_parameters(
                    teacher_name,
                    target_components=target_components,
                    device=snapshot_device,
                    overwrite=True,
                )
            loaded_names = list(teacher_names)
        else:
            teachers = self.training_args.teachers
            loaded_names = load_teachers(
                self.adapter,
                [teacher.path for teacher in teachers],
                self.training_args.teacher_param_device,
                [teacher.name for teacher in teachers],
            )
            if tuple(loaded_names) != teacher_names:
                raise RuntimeError(
                    "DiffusionOPD teacher snapshot declaration drifted during loading: "
                    f"declared={teacher_names!r}, loaded={tuple(loaded_names)!r}"
                )

        self._teacher_names = loaded_names
        for teacher_name in teacher_names:
            self._register_named_parameter_runtime_child(teacher_name)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.training_args: DiffusionOPDTrainingArguments

        scheduler_group = self.adapter.scheduler_group
        # Every component scheduler must agree on the dynamics family: the KL
        # denominator is 1 under ODE and a transition variance under SDE.
        self._is_sde = resolve_scheduler_group_dynamics(
            self.adapter, self.training_args.loss_target
        )
        # Teacher and student targets are computed at the SAME stored state x_j
        # with the SAME noise_level so their projections are comparable.
        self._student_noise_level = (
            float(scheduler_group.primary.noise_level) if self._is_sde else 0.0
        )

        # Teacher snapshots are initialized by the BaseTrainer lifecycle before
        # exact-state compatibility preflight.
        teachers = self.training_args.teachers
        student_gs = float(self.training_args.guidance_scale)
        self._teacher_gs: List[float] = [
            float(teacher.guidance_scale) if teacher.guidance_scale is not None else student_gs
            for teacher in teachers
        ]

        # Dataset -> teacher routing
        # The config schema permits several teachers to share a dataset, but
        # DiffusionOPDTrainer distills exactly one teacher per dataset and
        # rejects any overlap below. Routing is keyed on ``BaseSample.source``,
        # which is exactly the dataset name.
        self._source_to_teacher: Dict[str, int] = {}
        for teacher_idx, teacher in enumerate(teachers):
            for dataset in teacher.applicable_datasets:
                if dataset in self._source_to_teacher:
                    raise ValueError(
                        f"Dataset {dataset!r} is claimed by multiple teachers "
                        f"({self._teacher_names[self._source_to_teacher[dataset]]!r} and "
                        f"{self._teacher_names[teacher_idx]!r}). The DiffusionOPD config schema "
                        "permits this for a future multi-teacher/ensemble trainer, but the current "
                        "DiffusionOPDTrainer distills exactly one teacher per dataset."
                    )
                self._source_to_teacher[dataset] = teacher_idx

        # Runtime cross-check against the actually-built per-dataset dataloaders.
        self._available_sources = set(self.train_dataloaders_by_source.keys())
        for teacher_idx, teacher in enumerate(teachers):
            for dataset in teacher.applicable_datasets:
                if dataset not in self._available_sources:
                    raise ValueError(
                        f"Teacher {self._teacher_names[teacher_idx]!r} references dataset {dataset!r} "
                        f"that has no training dataloader. Available datasets: "
                        f"{sorted(self._available_sources)}. Check that `data.datasets` has an entry "
                        "with this `name` and `train.enabled: true`."
                    )

        self._teacher_target_store_device = (
            "cpu" if self.training_args.offload_samples_to_cpu else self.accelerator.device
        )

        logger.info(
            f"DiffusionOPDTrainer initialized: {len(self._teacher_names)} teacher(s) "
            f"{self._teacher_names}, dynamics={scheduler_group.primary.dynamics_type!r} "
            f"(is_sde={self._is_sde}, student_noise_level={self._student_noise_level}), "
            f"datasets={sorted(self._available_sources)}, "
            f"student_gs={student_gs}, teacher_gs={self._teacher_gs}, "
            f"loss_target={self.training_args.loss_target!r}, "
            f"self_normalize={self.training_args.self_normalize}."
        )

    # =============================== Lifecycle ===============================
    def sample(self) -> List[BaseSample]:
        """Roll out on-policy student trajectories over the multi-source dataloader.

        Stores only the trajectory positions needed for the distilled step band
        (``timestep_range``): current + next latents for each step in the band.
        Rewards are not used by the distillation loss, so no reward buffer is
        attached; reward monitoring is left to :meth:`evaluate`.
        """
        train_step_indices = self._select_train_step_indices(
            self.training_args.num_inference_steps, self.training_args.timestep_range
        )
        trajectory_indices = compute_trajectory_indices(
            train_timestep_indices=train_step_indices,
            num_inference_steps=self.training_args.num_inference_steps,
        )
        return self.generate_samples(
            reward_buffer=None,
            compute_log_prob=False,
            trajectory_indices=trajectory_indices,
        )

    def prepare_feedback(self, samples: List[BaseSample]) -> None:
        """No-op: DiffusionOPD has no reward/advantage stage."""
        return

    # =============================== Optimization ===============================
    def optimize(self, samples: List[BaseSample]) -> None:
        """Cache teacher targets, then optimize the student in the selected target space."""
        if not samples:
            logger.warning("DiffusionOPD optimize() received no samples; skipping epoch.")
            return

        # Train-mode dynamics for BOTH passes (scheduler.is_eval=False) so the SDE
        # transition means are computed consistently for teacher and student.
        self.adapter.train()
        # Distilled denoising-step band from `timestep_range` (see `_select_train_step_indices`).
        # Same indices the rollout stored in `sample()`, so the replay aligns.
        train_timesteps = self._select_train_step_indices(
            self.training_args.num_inference_steps, self.training_args.timestep_range
        )

        self._precompute_teacher_targets(samples, train_timesteps)
        self._distill(samples, train_timesteps)

    @torch.no_grad()
    def _precompute_teacher_targets(
        self,
        samples: List[BaseSample],
        train_timesteps: torch.Tensor,
    ) -> None:
        """PASS 1: cache each teacher's projected target on its routed samples.

        One ``use_named_parameters`` swap per teacher (performed OUTSIDE the
        autocast block); a per-teacher ``autocast`` scope gives each teacher a
        fresh weight cache (so no stale-cast across swaps), with an explicit
        ``clear_autocast_cache`` as a belt-and-suspenders guard.
        """
        per_device_batch_size = self.training_args.per_device_batch_size

        samples_by_teacher: Dict[int, List[BaseSample]] = defaultdict(list)
        for sample in samples:
            samples_by_teacher[self._teacher_index_for_sample(sample)].append(sample)

        for teacher_idx, teacher_samples in samples_by_teacher.items():
            teacher_name = self._teacher_names[teacher_idx]
            teacher_gs = self._teacher_gs[teacher_idx]
            num_batches = math.ceil(len(teacher_samples) / per_device_batch_size)

            # Swap teacher weights in OUTSIDE the autocast context.
            with self.adapter.use_named_parameters(teacher_name):
                with self.autocast():
                    for batch in tqdm(
                        self._iter_prefetched_batches(teacher_samples, per_device_batch_size),
                        total=num_batches,
                        desc=f"Epoch {self.epoch} Teacher[{teacher_name}] targets",
                        disable=not self.show_progress_bar,
                    ):
                        # Teacher target per component at each training step.
                        teacher_target_steps = [
                            self._forward_step(
                                batch,
                                int(timestep_index),
                                guidance_scale=teacher_gs,
                                context=f"teacher {teacher_name!r}",
                            )[1]
                            for timestep_index in train_timesteps
                        ]
                        # {component: (B, num_train_steps, *latent)}
                        teacher_target_stacked = {
                            name: torch.stack(
                                [step.components[name].detach() for step in teacher_target_steps],
                                dim=1,
                            )
                            for name in self.adapter.trajectory_component_order
                        }
                        for j, sample in enumerate(batch.samples):
                            sample.extra_kwargs["teacher_target"] = {
                                name: values[j].to(self._teacher_target_store_device).clone()
                                for name, values in teacher_target_stacked.items()
                            }
            # Belt-and-suspenders guard against a nested-autocast cache edge case.
            torch.clear_autocast_cache()

    def _distill(
        self,
        samples: List[BaseSample],
        train_timesteps: torch.Tensor,
    ) -> None:
        """PASS 2: match student targets to cached teacher targets."""
        device = self.accelerator.device
        per_device_batch_size = self.training_args.per_device_batch_size
        num_batches = math.ceil(len(samples) / per_device_batch_size)

        for inner_epoch in range(self.training_args.num_inner_epochs):
            # Shuffle unless disabled for pack-composition-dependent adapters.
            shuffled_samples = self._order_samples_for_optimize(samples, inner_epoch)

            self.adapter.train()
            # Per-teacher loss accumulators over the current gradient-accumulation window.
            # Fixed (num_teachers,) shape so the cross-rank reduce in `_log_distill_metrics`
            # is collective-safe regardless of which teachers each rank's micro-batches held.
            num_teachers = len(self._teacher_names)
            teacher_loss_sum = torch.zeros(num_teachers, device=device)
            teacher_loss_count = torch.zeros(num_teachers, device=device)
            grad_norm = None

            for batch in tqdm(
                self._iter_prefetched_batches(shuffled_samples, per_device_batch_size),
                total=num_batches,
                desc=f"Epoch {self.epoch} Distill",
                position=0,
                disable=not self.show_progress_bar,
            ):
                # Teacher index per sample in this (possibly source-mixed) micro-batch.
                teacher_idx = torch.tensor(
                    [self._teacher_index_for_sample(s) for s in batch.samples],
                    device=device,
                    dtype=torch.long,
                )  # (B,)
                # teacher_target rides BaseSample.to() with the sample; ensure on device.
                teacher_target_all = self._require_teacher_target_cache(
                    batch, num_steps=len(train_timesteps)
                )

                for idx, timestep_index in enumerate(
                    tqdm(
                        train_timesteps,
                        desc=f"Epoch {self.epoch} Timestep",
                        position=1,
                        leave=False,
                        disable=not self.show_progress_bar,
                    )
                ):
                    step_index = int(timestep_index)
                    with self.accumulate_gradients():
                        with self.autocast():
                            replay, student_target, output = self._forward_step(
                                batch,
                                step_index,
                                guidance_scale=self.training_args.guidance_scale,
                                context="student",
                                include_transition_stats=self._is_sde,
                            )
                            # Each sample is matched to its own routed teacher target.
                            teacher_target = LatentState(
                                {
                                    name: values[:, idx]
                                    for name, values in teacher_target_all.items()
                                }
                            )
                            # Validation guarantees SDE uses the xt transition-mean target.
                            denominators = (
                                self._component_kl_denominators(output, replay, step_index)
                                if self._is_sde
                                else None
                            )
                            per_sample_distill_loss = compute_structured_distillation_loss(
                                self.adapter,
                                student_target=student_target,
                                teacher_target=teacher_target,
                                state=replay.state,
                                self_normalize=self.training_args.self_normalize,
                                denominators=denominators,
                            )
                            loss = per_sample_distill_loss.mean()

                        self.accelerator.backward(loss)

                        # Accumulate per-teacher loss sums/counts for logging (detached).
                        with torch.no_grad():
                            teacher_loss_sum.index_add_(
                                0, teacher_idx, per_sample_distill_loss.detach()
                            )
                            teacher_loss_count.index_add_(
                                0, teacher_idx, torch.ones_like(per_sample_distill_loss)
                            )

                        if self.accelerator.sync_gradients:
                            grad_norm = self.accelerator.clip_grad_norm_(
                                self.adapter.get_trainable_parameters(),
                                self.training_args.max_grad_norm,
                            )
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            self._log_distill_metrics(
                                teacher_loss_sum, teacher_loss_count, grad_norm
                            )
                            self.step += 1
                            teacher_loss_sum.zero_()
                            teacher_loss_count.zero_()

    # =============================== Helpers ===============================
    def _log_distill_metrics(
        self,
        teacher_loss_sum: torch.Tensor,
        teacher_loss_count: torch.Tensor,
        grad_norm: Optional[torch.Tensor],
    ) -> None:
        """Globally reduce per-teacher loss sums/counts and log their means.

        Logs one ``train/distill_loss_{teacher_name}`` per teacher seen this
        window plus the overall ``train/distill_loss``. The reduce
        operates on fixed ``(num_teachers,)`` tensors, identical on every rank,
        so it is collective-safe even when teachers are unevenly distributed
        across ranks/micro-batches.
        """
        # Pack sum + count into one tensor so the cross-rank reduction is a single
        # collective (the pack-and-reduce idiom used across utils/dist.py).
        packed = torch.stack([teacher_loss_sum, teacher_loss_count])
        packed = cast(torch.Tensor, self.accelerator.reduce(packed, reduction="sum"))
        g_sum, g_count = packed[0], packed[1]

        metrics: Dict[str, Any] = {}
        total_count = g_count.sum()
        if total_count > 0:
            metrics["distill_loss"] = g_sum.sum() / total_count
        for teacher_idx, name in enumerate(self._teacher_names):
            if g_count[teacher_idx] > 0:
                metrics[f"distill_loss_{name}"] = g_sum[teacher_idx] / g_count[teacher_idx]
        if grad_norm is not None:
            metrics["grad_norm"] = grad_norm

        self.log_data({f"train/{k}": v for k, v in metrics.items()}, step=self.step)

    @staticmethod
    def _select_train_step_indices(
        num_inference_steps: int,
        timestep_range: Union[float, Tuple[float, float]],
    ) -> torch.Tensor:
        """Trajectory step indices to distill on, from ``timestep_range``.

        ``timestep_range=(frac_lo, frac_hi)`` (a bare float ``f`` is treated as
        ``(0, f)``) selects the contiguous band of denoising transitions
        ``[int(T*frac_lo), int(T*frac_hi))`` where ``T = num_inference_steps``.
        Default ``0.99`` reproduces upstream DiffusionOPD's ``timestep_fraction``
        (distill the first 99% of steps, ``int(10*0.99)=9`` -> indices ``[0..8]``,
        skipping the near-clean tail). Deterministic and dynamics-agnostic, so it
        does NOT use the SDE-only ``scheduler.train_timesteps`` (empty under ODE),
        and gives identical indices in ``sample()`` and ``optimize()``. The band
        comes from :func:`resolve_distill_step_band`, the same resolver
        ``get_num_train_timesteps`` uses for the gradient-accumulation count.
        """
        lo, hi = resolve_distill_step_band(num_inference_steps, timestep_range)
        return torch.arange(lo, hi, dtype=torch.long)

    def _teacher_index_for_sample(self, sample: BaseSample) -> int:
        """Resolve a sample's teacher index from its dataset (``sample.source``).

        Single-dataset configs use a bare DataLoader that does not inject
        ``__source__`` (so ``sample.source`` is None); route those to the sole
        teacher of the only available dataset.
        """
        dataset = sample.source
        if dataset is None:
            if len(self._available_sources) == 1:
                dataset = next(iter(self._available_sources))
            else:
                raise RuntimeError(
                    f"DiffusionOPD sample is missing `source` but {len(self._available_sources)} "
                    "datasets are active; cannot route to a teacher. Multi-dataset rollouts must "
                    "carry `source` (set by MultiSourceTrainDataLoader)."
                )
        if dataset not in self._source_to_teacher:
            raise RuntimeError(
                f"Sample dataset {dataset!r} is not routed to any teacher. "
                f"Routing: {self._source_to_teacher}."
            )
        return self._source_to_teacher[dataset]

    def _replay_forward_kwargs(self, batch: StackedSampleBatch) -> Dict[str, Any]:
        """Return training arguments the batch does not already carry.

        Batch-level values win on shared keys.
        """
        return replay_forward_kwargs(self, batch)

    def _forward_step(
        self,
        batch: StackedSampleBatch,
        step_index: int,
        guidance_scale: float,
        context: str,
        include_transition_stats: bool = False,
    ) -> Tuple[ReplayStep, LatentState, MultiModalStepOutput]:
        """Forward one stored step and return its configured target projection.

        Replays the rollout transition ``x_j -> x_{j+1}`` through
        :meth:`BaseAdapter.get_replay_step` / :meth:`BaseAdapter.forward_state`,
        so states, times and per-component sigmas come from the trajectory
        rather than from legacy index maps. The transition statistics are
        requested only for the stochastic student pass.

        Args:
            batch: Collated micro-batch replayed at this step.
            step_index: Global rollout transition index.
            guidance_scale: Scale overriding the configured training value.
            context: Which pass is running (``"teacher 'name'"`` or
                ``"student"``); projection errors are raised with this and the
                step index so a two-pass failure is attributable.
            include_transition_stats: Whether to request ``std_dev_t``/``dt``.

        Returns:
            ``(replay, target, output)``: the replayed step, its target-space
            projection, and the raw step output.
        """
        replay = self.adapter.get_replay_step(batch, step_index)
        return_fields = (TARGET_REQUEST_FIELDS[self.training_args.loss_target],)
        if include_transition_stats:
            return_fields += ("std_dev_t", "dt")

        forward_kwargs = self._replay_forward_kwargs(batch)
        # guidance_scale overrides the training_args value; adapters whose
        # forward() does not accept it drop it in the bridge's filter step.
        forward_kwargs["guidance_scale"] = guidance_scale
        output = self.adapter.forward_state(
            batch=batch,
            state=replay.state,
            times=replay.times,
            next_state=replay.next_state,
            compute_log_prob=False,
            return_fields=return_fields,
            noise_level=self._student_noise_level,
            **forward_kwargs,
        )
        target = project_distillation_target_state(
            self.adapter,
            loss_target=self.training_args.loss_target,
            state=replay.state,
            output=output,
            times=replay.times,
            context=f"{context} pass at step_index={step_index}",
        )
        return replay, target, output

    def _require_teacher_target_cache(
        self,
        batch: StackedSampleBatch,
        num_steps: int,
    ) -> Dict[str, torch.Tensor]:
        """Return the PASS 1 teacher cache for this micro-batch, on the compute device.

        The cache is an ordered component mapping of stacked per-step targets
        (``{component: (B, num_steps, *latent)}``) that rides
        :meth:`BaseSample.to` with the sample.
        """
        expected_names = self.adapter.trajectory_component_order
        cached = batch.get("teacher_target")
        if not isinstance(cached, Mapping):
            received = "None" if cached is None else type(cached).__name__
            raise ValueError(
                "expected DiffusionOPDTrainer cached 'teacher_target' as an ordered component "
                f"mapping in component order {expected_names}, received {received}; PASS 1 "
                "(_precompute_teacher_targets) must run before PASS 2"
            )
        if tuple(cached) != expected_names:
            raise ValueError(
                "expected DiffusionOPDTrainer cached 'teacher_target' in component order "
                f"{expected_names}, received {tuple(cached)}"
            )
        batch_size = len(batch.samples)
        device = self.accelerator.device
        resolved: Dict[str, torch.Tensor] = {}
        for name in expected_names:
            values = cached[name]
            if not isinstance(values, torch.Tensor):
                raise TypeError(
                    f"expected a torch.Tensor DiffusionOPDTrainer cached 'teacher_target' for "
                    f"component {name!r}, received {type(values).__name__}"
                )
            if values.ndim < 3 or values.shape[:2] != (batch_size, num_steps):
                raise ValueError(
                    f"expected DiffusionOPDTrainer cached 'teacher_target' component {name!r} "
                    f"with {num_steps} stored distillation steps and shape "
                    f"({batch_size}, {num_steps}, ...), received {tuple(values.shape)}"
                )
            resolved[name] = values.to(device)
        return resolved

    def _require_component_statistic(
        self,
        values: Optional[Mapping[str, torch.Tensor]],
        field: str,
        step_index: int,
    ) -> Mapping[str, torch.Tensor]:
        """Return one transition statistic mapping required by stochastic dynamics."""
        expected_names = self.adapter.trajectory_component_order
        if not isinstance(values, Mapping) or tuple(values) != expected_names:
            received = (
                "None"
                if values is None
                else (str(tuple(values)) if isinstance(values, Mapping) else type(values).__name__)
            )
            raise ValueError(
                f"expected DiffusionOPDTrainer replay at step_index={step_index} to carry "
                f"{field} in component order {expected_names}, received {received}; request "
                "'std_dev_t' and 'dt' through return_fields"
            )
        return values

    def _component_kl_denominators(
        self,
        output: MultiModalStepOutput,
        replay: ReplayStep,
        step_index: int,
    ) -> Dict[str, torch.Tensor]:
        """Per-component SDE KL denominators for one replayed transition.

        Each component is normalized by its OWN scheduler's transition
        variance, so a heterogeneous multi-modal group never borrows the
        primary scheduler's noise schedule.
        """
        expected_names = self.adapter.trajectory_component_order
        std_dev_t = self._require_component_statistic(output.std_dev_t, "std_dev_t", step_index)
        dt = self._require_component_statistic(output.dt, "dt", step_index)
        batch_size = replay.state.components[expected_names[0]].shape[0]

        denominators: Dict[str, torch.Tensor] = {}
        for name in expected_names:
            scheduler = self.adapter.scheduler_group[name]
            denominator = scheduler.get_kl_divergence_denominator(std_dev_t[name], dt[name])
            if not isinstance(denominator, torch.Tensor):
                raise TypeError(
                    f"expected a torch.Tensor KL denominator from the {name!r} scheduler "
                    f"({type(scheduler).__name__}, dynamics_type="
                    f"{scheduler.dynamics_type!r}) at step_index={step_index}, received "
                    f"{type(denominator).__name__}: {denominator!r}"
                )
            # A scalar-like (B, 1, ...) denominator carries one value per sample;
            # anything else would silently average distinct per-element scales.
            if (
                denominator.ndim == 0
                or denominator.shape[0] != batch_size
                or denominator.numel() != batch_size
            ):
                raise ValueError(
                    f"expected DiffusionOPDTrainer KL denominator at step_index={step_index} "
                    f"for component {name!r} to hold one value per sample with shape "
                    f"({batch_size},), received {tuple(denominator.shape)}"
                )
            reference = replay.state.components[name]
            if not denominator.is_floating_point() or denominator.device != reference.device:
                raise ValueError(
                    f"expected DiffusionOPDTrainer KL denominator at step_index={step_index} "
                    f"for component {name!r} as a floating tensor on the replay device "
                    f"{reference.device}, received {denominator.dtype} on {denominator.device}"
                )
            # The shared loss helper re-checks these values; validating here too
            # attributes an unusable transition variance to its replay step.
            if not bool(torch.isfinite(denominator).all()):
                raise ValueError(
                    f"expected DiffusionOPDTrainer KL denominator at step_index={step_index} "
                    f"for component {name!r} to be finite, received {denominator}"
                )
            if not bool((denominator > 0).all()):
                raise ValueError(
                    f"expected DiffusionOPDTrainer KL denominator at step_index={step_index} "
                    f"for component {name!r} to be strictly positive, received {denominator}"
                )
            denominators[name] = denominator.reshape(batch_size)
        return denominators
