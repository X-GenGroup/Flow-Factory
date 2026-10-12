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

"""Pure pairwise DPO objective shared by online and offline trainers."""

from __future__ import annotations

import math
from numbers import Real
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def dpo_objective(
    policy_chosen_loss: torch.Tensor,
    policy_rejected_loss: torch.Tensor,
    reference_chosen_loss: torch.Tensor,
    reference_rejected_loss: torch.Tensor,
    beta: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute pairwise diffusion-DPO from aligned per-sample losses.

    Lower per-sample losses correspond to higher implicit rewards. Online and
    offline DPO share these exact loss and metric definitions.
    """
    _validate_dpo_inputs(
        policy_chosen_loss=policy_chosen_loss,
        policy_rejected_loss=policy_rejected_loss,
        reference_chosen_loss=reference_chosen_loss,
        reference_rejected_loss=reference_rejected_loss,
        beta=beta,
    )

    chosen_delta = policy_chosen_loss - reference_chosen_loss
    rejected_delta = policy_rejected_loss - reference_rejected_loss
    preference_delta = chosen_delta - rejected_delta
    loss = -F.logsigmoid(-0.5 * beta * preference_delta).mean()
    with torch.no_grad():
        implicit_reward_chosen = -0.5 * beta * chosen_delta
        implicit_reward_rejected = -0.5 * beta * rejected_delta
        metrics = {
            "implicit_reward_chosen": implicit_reward_chosen,
            "implicit_reward_rejected": implicit_reward_rejected,
            "implicit_accuracy": (implicit_reward_chosen > implicit_reward_rejected).float().mean(),
        }
    return loss, metrics


def _validate_dpo_inputs(
    *,
    policy_chosen_loss: torch.Tensor,
    policy_rejected_loss: torch.Tensor,
    reference_chosen_loss: torch.Tensor,
    reference_rejected_loss: torch.Tensor,
    beta: float,
) -> None:
    named_losses = {
        "policy_chosen_loss": policy_chosen_loss,
        "policy_rejected_loss": policy_rejected_loss,
        "reference_chosen_loss": reference_chosen_loss,
        "reference_rejected_loss": reference_rejected_loss,
    }
    for name, values in named_losses.items():
        if not isinstance(values, torch.Tensor):
            raise TypeError(
                f"expected {name} for DPO objective to be a torch.Tensor, "
                f"received {type(values).__name__}: {values!r}"
            )
        if values.ndim != 1 or values.shape[0] == 0:
            raise ValueError(
                f"expected {name} for DPO objective to have non-empty shape (B,), "
                f"received shape {tuple(values.shape)}"
            )
        if not values.is_floating_point():
            raise TypeError(
                f"expected {name} for DPO objective to use a floating dtype, "
                f"received dtype {values.dtype}"
            )

    expected_shape = policy_chosen_loss.shape
    expected_dtype = policy_chosen_loss.dtype
    expected_device = policy_chosen_loss.device
    for name, values in tuple(named_losses.items())[1:]:
        if values.shape != expected_shape:
            raise ValueError(
                "expected all DPO per-sample losses to have the same shape, "
                f"but policy_chosen_loss has {tuple(expected_shape)} and {name} has "
                f"{tuple(values.shape)}"
            )
        if values.dtype != expected_dtype:
            raise TypeError(
                "expected all DPO per-sample losses to have the same dtype, "
                f"but policy_chosen_loss uses {expected_dtype} and {name} uses "
                f"{values.dtype}"
            )
        if values.device != expected_device:
            raise ValueError(
                "expected all DPO per-sample losses on the same device, "
                f"but policy_chosen_loss is on {expected_device} and {name} is on "
                f"{values.device}"
            )

    if isinstance(beta, bool) or not isinstance(beta, Real):
        raise TypeError(
            "expected beta for DPO objective to be a finite real number, "
            f"received {type(beta).__name__}: {beta!r}"
        )
    if not math.isfinite(float(beta)):
        raise ValueError(f"expected finite beta for DPO objective, received beta={beta!r}")


__all__ = ["dpo_objective"]
