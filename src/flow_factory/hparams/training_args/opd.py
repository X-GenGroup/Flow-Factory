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

"""Training arguments for DiffusionOPD (on-policy distillation).

DiffusionOPD distills several task-specialized teachers into a single
student along the student's own rollout trajectories. Each teacher is a
LoRA checkpoint routed to one or more training datasets (``data.datasets``
entries) via :class:`TeacherConfig.applicable_datasets`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, List, Literal, Optional, Tuple, Union

from ...contracts.execution import (
    ONLINE_NO_FEEDBACK_EXECUTION_CONTRACT,
    ExecutionContract,
)
from ..abc import ArgABC
from ._base import TrainingArguments, _standardize_timestep_range


def resolve_distill_step_band(
    num_inference_steps: int,
    timestep_range: Union[float, Tuple[float, float]],
) -> Tuple[int, int]:
    """Resolve ``timestep_range`` to the ``[lo, hi)`` denoising-step index band.

    Single source of truth for which trajectory steps DiffusionOPD distills:
    a bare float ``f`` means ``(0, f)``; ``(frac_lo, frac_hi)`` maps to
    ``[int(T*frac_lo), int(T*frac_hi))`` with ``T = num_inference_steps``,
    clamped to at least one step. Used both by
    :meth:`DiffusionOPDTrainingArguments.get_num_train_timesteps` (gradient-
    accumulation math) and the trainer's per-step loop, so the two never drift.
    """
    if isinstance(timestep_range, (int, float)):
        frac_lo, frac_hi = 0.0, float(timestep_range)
    else:
        frac_lo, frac_hi = timestep_range
    lo = int(num_inference_steps * frac_lo)
    hi = max(lo + 1, int(num_inference_steps * frac_hi))
    return lo, hi


@dataclass
class TeacherConfig(ArgABC):
    """Configuration for a single DiffusionOPD teacher.

    Attributes:
        path: Path or HuggingFace repo id of the teacher LoRA checkpoint.
            Must share the student's LoRA architecture (target modules;
            rank/alpha compatible) since it is loaded into the student's
            active adapter slot.
        name: Unique teacher identifier (used for the named-parameter
            snapshot and log keys). Defaults to ``opd_teacher_{i}``.
        applicable_datasets: Dataset names (matching ``data.datasets[*].name``)
            whose student rollouts are distilled from this teacher's per-step
            KL. Required and non-empty. NOTE: the schema intentionally permits
            several teachers to list the same dataset (so a future
            multi-teacher/ensemble trainer can reuse this config), but the
            current ``DiffusionOPDTrainer`` distills exactly one teacher per
            dataset and raises if a dataset is claimed by more than one teacher.
        guidance_scale: Optional per-teacher CFG override for the teacher
            forward. ``None`` falls back to the trainer-global student
            ``guidance_scale`` (e.g. a GenEval teacher queried at its
            no-CFG training distribution while the student rolls out at a
            higher scale).
    """

    path: str = field(
        metadata={"help": "Path or HF repo id of the teacher LoRA checkpoint."},
    )
    name: Optional[str] = field(
        default=None,
        metadata={"help": "Unique teacher name (defaults to opd_teacher_{i})."},
    )
    applicable_datasets: List[str] = field(
        default_factory=list,
        metadata={
            "help": "Dataset names this teacher is distilled on (matches data.datasets[*].name)."
        },
    )
    guidance_scale: Optional[float] = field(
        default=None,
        metadata={
            "help": "Per-teacher CFG override for the teacher forward (None = student guidance_scale)."
        },
    )


@dataclass
class DiffusionOPDTrainingArguments(TrainingArguments):
    r"""Training arguments for the DiffusionOPD distillation trainer.

    Selects the distillation space with ``loss_target`` and independently
    enables detached error self-normalization with ``self_normalize``.
    ``xt`` matches transition means and supports ODE or SDE dynamics; ``v``
    and ``x0`` are ODE-only velocity-derived targets. Under SDE, the ``xt``
    loss remains divided by the scheduler's transition variance.
    """

    execution_contract: ClassVar[ExecutionContract] = ONLINE_NO_FEEDBACK_EXECUTION_CONTRACT

    teachers: List[TeacherConfig] = field(
        default_factory=list,
        metadata={"help": "List of teacher configs; each maps a LoRA checkpoint -> dataset(s)."},
    )
    teacher_param_device: Literal["cpu", "cuda"] = field(
        default="cuda",
        metadata={
            "help": "Device to store teacher LoRA snapshots ('cuda' = fast swaps, 'cpu' = lower VRAM)."
        },
    )
    timestep_range: Union[float, Tuple[float, float]] = field(
        default=0.99,
        metadata={
            "help": (
                "Fraction band of denoising transitions to distill, along the denoise axis "
                "1000->0. A float ``f`` means the band ``[0, f]`` (the first ``f``-fraction of "
                "steps, skipping the near-clean tail); a tuple is an explicit ``[lo, hi]`` band. "
                "Default 0.99 matches upstream DiffusionOPD's ``timestep_fraction``. "
                "Dynamics-agnostic (does not use ``scheduler.replay_steps``)."
            )
        },
    )
    loss_target: Literal["xt", "v", "x0"] = field(
        default="xt",
        metadata={
            "help": (
                "Distillation target space: 'xt' matches the one-step transition mean; "
                "'v' matches velocity; 'x0' matches the predicted clean latent. "
                "The 'v' and 'x0' targets require ODE dynamics."
            )
        },
    )
    self_normalize: bool = field(
        default=False,
        metadata={
            "help": (
                "Divide each sample's target-space MSE by its detached mean absolute "
                "student-teacher error plus 1e-8."
            )
        },
    )

    def __post_init__(self):
        super().__post_init__()

        valid_loss_targets = ("xt", "v", "x0")
        if self.loss_target not in valid_loss_targets:
            raise ValueError(
                "train.loss_target must be one of "
                f"{valid_loss_targets}, got {self.loss_target!r}."
            )
        if not isinstance(self.self_normalize, bool):
            raise TypeError(
                "train.self_normalize must be a bool, "
                f"got {type(self.self_normalize).__name__}: {self.self_normalize!r}."
            )

        # Standardize to (frac_lo, frac_hi); a float f -> (0.0, f). Mirrors NFT's
        # `timestep_range` convention so the trainer can derive distillation-step
        # indices from a fraction band.
        self.timestep_range = _standardize_timestep_range(self.timestep_range)

        # Coerce dict entries -> TeacherConfig (ArgABC.from_dict does not recurse
        # into nested list-of-ArgABC fields).
        coerced: List[TeacherConfig] = []
        for i, teacher in enumerate(self.teachers):
            if isinstance(teacher, TeacherConfig):
                coerced.append(teacher)
            elif isinstance(teacher, dict):
                coerced.append(TeacherConfig.from_dict(teacher))
            else:
                raise TypeError(
                    f"train.teachers[{i}] must be a dict or TeacherConfig, "
                    f"got {type(teacher).__name__}."
                )
        self.teachers = coerced

        if not self.teachers:
            raise ValueError(
                "DiffusionOPDTrainingArguments requires at least one teacher; "
                "got an empty `train.teachers` list."
            )

        seen_names: set[str] = set()
        for i, teacher in enumerate(self.teachers):
            if not teacher.path:
                raise ValueError(f"train.teachers[{i}] is missing a `path`.")
            if not teacher.applicable_datasets:
                raise ValueError(
                    f"DiffusionOPD requires each teacher to specify non-empty `applicable_datasets`; "
                    f"teacher[{i}] (path={teacher.path!r}) has none. Each teacher must declare "
                    "which `data.datasets[*].name` dataset(s) it is distilled on."
                )
            if teacher.name is None:
                teacher.name = f"opd_teacher_{i}"
            if teacher.name in seen_names:
                raise ValueError(
                    f"Duplicate teacher name {teacher.name!r}; teacher names must be unique."
                )
            seen_names.add(teacher.name)
            if teacher.guidance_scale is not None:
                teacher.guidance_scale = float(teacher.guidance_scale)

    def get_preprocess_guidance_scale(self) -> float:
        """Encode negative prompts if the student OR any teacher uses CFG > 1.

        DiffusionOPD rolls out / forwards at the student ``guidance_scale``
        but may query a teacher at its own (possibly higher) scale, so the
        preprocessing stage must cover the max of both.
        """
        scales = [self.guidance_scale]
        scales += [
            teacher.guidance_scale
            for teacher in self.teachers
            if teacher.guidance_scale is not None
        ]
        return max(scales)

    def get_num_train_timesteps(self, args: Any) -> int:
        """Per-micro-batch backward count = number of distilled denoising steps.

        The optimize loop runs one ``accelerator.backward`` per distilled step
        (``timestep_range`` band), so the gradient-accumulation math must scale
        by this count for ``gradient_step_per_epoch`` to hold (mirrors GRPO
        returning ``num_sde_steps``). Without this override the base returns 1
        and the trainer takes ``num_distill_steps``x too many optimizer steps.
        """
        lo, hi = resolve_distill_step_band(self.num_inference_steps, self.timestep_range)
        return hi - lo
