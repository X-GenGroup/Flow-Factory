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

# src/flow_factory/acceleration/validator.py
"""Paradigm-gated safety validation for accelerators (fail-fast, ``constraints.md`` #26).

The correctness contract (``constraints.md`` #7) hinges on **symmetric
application**, encoded by ``stage``, not on numerical bit-exactness:

* ``stage='both'`` accelerators mutate the transformer persistently, so the same
  transform runs in both rollout ``inference()`` and training ``forward()``. When that
  transform is identical across the two stages (exact, or symmetric-approximate like
  Sage int8 attention) rollout and training stay CONSISTENT — safe for any algorithm.
  They belong in the ``shared`` slot and are never rejected. ``safety`` is only used to
  *warn*: a ``stage='both'`` + ``lossy`` accelerator (e.g. ``torch.compile``, which is
  applied symmetrically but is not bit-exact across stages due to its grad/no-grad
  graph split) is still allowed, but on a **coupled** trainer the on-policy PPO ratio
  will be ≈1, not exactly 1, so the validator logs a warning.
* ``stage='rollout'`` accelerators run only during Stage-3 rollout. If such an
  accelerator changes outputs (``safety='lossy'``, e.g. feature caching), rollout
  diverges from the training forward, which cannot replicate it (that needs
  full gradient through every block). That is only safe when the rollout
  trajectory's log-prob never feeds the loss — i.e. **decoupled** / **distillation**
  algorithms. For **coupled** algorithms (GRPO / GRPO-Guard / DPPO) the rollout
  log-prob becomes the PPO "old log-prob", so a divergent rollout biases the
  importance ratio -> silently wrong gradients. A bit-identical rollout accelerator
  (``safety='lossless'``) is safe for any paradigm.
"""

from typing import Optional

from ..utils.logger_utils import setup_logger
from .abc import BaseAccelerator

logger = setup_logger(__name__)

# RL paradigms that tolerate a lossy (distribution-shifting) rollout: the rollout
# trajectory's per-step log-prob does not enter the training loss.
_LOSSY_SAFE_PARADIGMS = frozenset({"decoupled", "distillation"})


def validate_accelerator(
    accelerator: BaseAccelerator,
    *,
    slot: str,
    paradigm: Optional[str],
    trainer_name: str,
) -> None:
    """Validate one accelerator against its config slot and the trainer paradigm.

    Args:
        accelerator: The constructed accelerator instance.
        slot: The config slot it was placed in — ``"shared"`` (applied to both
            rollout and training via :meth:`~BaseAccelerator.setup`) or
            ``"rollout"`` (applied only during Stage 3 via
            :meth:`~BaseAccelerator.rollout_context`).
        paradigm: The trainer's RL paradigm
            (``"coupled"`` / ``"decoupled"`` / ``"distillation"``), or ``None`` if
            the trainer did not declare one.
        trainer_name: Trainer class name, for error messages.

    Raises:
        ValueError: If the accelerator is unsafe for the given slot / paradigm.
    """
    name = type(accelerator).__name__

    if slot not in ("shared", "rollout"):
        raise ValueError(
            f"Unknown acceleration slot {slot!r} for '{name}'; expected 'shared' or 'rollout'."
        )

    # The slot must match the accelerator's `stage`: `shared` accelerators mutate the
    # transformer persistently (applied to both rollout and training via `setup`);
    # `rollout` accelerators are a per-epoch context (`rollout_context`). A mismatch
    # would silently no-op (e.g. a stage='both' accelerator in the rollout slot has
    # no rollout_context), so reject it.
    if slot == "shared":
        if accelerator.stage != "both":
            raise ValueError(
                f"Accelerator '{name}' (stage='{accelerator.stage}') cannot occupy the shared "
                "slot, which applies a persistent transform to both rollout and training. Put a "
                "stage='both' accelerator here, or move this one to the `acceleration.rollout` list."
            )
        # A stage='both' transform is applied to the SAME module in rollout
        # `inference()` and training `forward()`. For a transform that is identical
        # across the two stages (exact, or symmetric-approximate like Sage int8) this is
        # consistent by construction — safe for any paradigm, no gate. A `lossy`
        # stage='both' accelerator (e.g. torch.compile, whose grad/no-grad compiled-graph
        # split leaves a ~1e-5 residual) is applied symmetrically but is NOT bit-exact
        # across stages; it stays within clip_range so it is allowed, but on a coupled
        # trainer the on-policy ratio will be ~1, not exactly 1 — warn so the user can
        # pick eager / an exact attention backend if strict ratio==1 is required.
        if accelerator.safety == "lossy" and paradigm == "coupled":
            logger.warning(
                "Accelerator '%s' (stage='both', safety='lossy') is applied symmetrically "
                "but is not bit-exact across rollout and training (e.g. torch.compile's "
                "grad/no-grad graph split). On the coupled trainer '%s' the on-policy PPO "
                "ratio will be ~1 but NOT exactly 1 (within clip_range). Use eager or an "
                "exact attention backend if a strictly bit-exact ratio is required.",
                name,
                trainer_name,
            )
        return

    # slot == "rollout".
    if accelerator.stage != "rollout":
        raise ValueError(
            f"Accelerator '{name}' (stage='{accelerator.stage}') cannot occupy the rollout slot, "
            "which only runs a per-epoch `rollout_context`. Put a stage='both' accelerator in the "
            "`acceleration.shared` list instead."
        )

    # A rollout-only accelerator that changes outputs (`lossy`) makes rollout diverge
    # from the training forward, so it is only safe when the rollout trajectory's
    # log-prob never feeds the loss (decoupled / distillation). A `lossless`
    # rollout accelerator (bit-identical) is safe for any paradigm.
    if accelerator.safety == "lossy":
        if paradigm is None:
            raise ValueError(
                f"Trainer '{trainer_name}' did not declare a `paradigm`, so a lossy rollout "
                f"accelerator ('{name}') cannot be validated for correctness. Set the trainer's "
                "`paradigm` class attribute to one of 'coupled' / 'decoupled' / 'distillation'."
            )
        if paradigm not in _LOSSY_SAFE_PARADIGMS:
            raise ValueError(
                f"Lossy rollout accelerator '{name}' is unsafe for the '{paradigm}' trainer "
                f"'{trainer_name}'. Lossy acceleration changes `velocity` during rollout but "
                "cannot be replicated in the training forward; for coupled algorithms "
                "(GRPO / GRPO-Guard / DPPO) this biases the PPO importance ratio and silently "
                f"corrupts gradients. Allowed paradigms: {sorted(_LOSSY_SAFE_PARADIGMS)}. Use a "
                "stage='both' accelerator (e.g. 'torch_compile') instead, or switch to a decoupled "
                "algorithm (NFT / AWM / DGPO / DPO / CRD)."
            )
        logger.warning(
            "Lossy rollout accelerator '%s' enabled for '%s' (paradigm='%s'). This shifts the "
            "generated-sample distribution; monitor the reward mean/std for regression.",
            name,
            trainer_name,
            paradigm,
        )
