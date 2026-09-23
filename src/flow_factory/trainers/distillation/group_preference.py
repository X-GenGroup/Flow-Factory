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

"""Shared cross-rank group-preference objective."""

import math
from dataclasses import dataclass
from numbers import Real

import torch
import torch.nn.functional as F
from accelerate import Accelerator


@dataclass(frozen=True)
class GroupPreferenceBatch:
    """Describe local samples in a dense group-id space.

    ``reduce_across_ranks=True`` (DGPO) treats ``num_groups`` as a global
    dense id space that every rank must cover. ``False`` (TDM-R1) keeps
    complete groups rank-local and skips the cross-rank sum.
    """

    local_group_indices: torch.Tensor
    num_groups: int
    group_size: int
    advantages: torch.Tensor
    reduce_across_ranks: bool = True
    allow_sparse_local_groups: bool = False


def _validate_group_layout(
    local_values: torch.Tensor,
    group_indices: torch.Tensor,
    num_groups: int,
    *,
    require_local_coverage: bool = True,
) -> None:
    """Validate the dense local group layout used by scatter-add."""
    if not isinstance(local_values, torch.Tensor):
        raise TypeError(
            "expected local_values for group reduction to be a torch.Tensor, "
            f"received {type(local_values).__name__}: {local_values!r}"
        )
    if local_values.ndim < 1 or local_values.shape[0] == 0:
        raise ValueError(
            "expected local_values for group reduction to have non-empty shape (B, ...), "
            f"received shape {tuple(local_values.shape)}"
        )
    if not isinstance(group_indices, torch.Tensor):
        raise TypeError(
            "expected group_indices for group reduction to be a torch.Tensor, "
            f"received {type(group_indices).__name__}: {group_indices!r}"
        )
    if group_indices.dtype != torch.int64:
        raise TypeError(
            "expected group_indices for group reduction to use torch.int64, "
            f"received dtype {group_indices.dtype}"
        )
    if group_indices.ndim != 1:
        raise ValueError(
            "expected group_indices for group reduction to be one-dimensional with shape (B,), "
            f"received shape {tuple(group_indices.shape)}"
        )
    if group_indices.shape[0] != local_values.shape[0]:
        raise ValueError(
            "expected group_indices and local_values to share batch size, received "
            f"{group_indices.shape[0]} and {local_values.shape[0]}"
        )
    if group_indices.device != local_values.device:
        raise ValueError(
            "expected group_indices and local_values on the same device, received "
            f"{group_indices.device} and {local_values.device}"
        )
    if isinstance(num_groups, bool) or not isinstance(num_groups, int):
        raise TypeError(
            "expected num_groups for group reduction to be an int, "
            f"received {type(num_groups).__name__}: {num_groups!r}"
        )
    if num_groups <= 0:
        raise ValueError(
            "expected positive num_groups for group reduction, " f"received num_groups={num_groups}"
        )

    observed_indices = torch.unique(group_indices, sorted=True)
    if int(observed_indices[0].item()) < 0 or int(observed_indices[-1].item()) >= num_groups:
        raise ValueError(
            "expected local group indices inside [0, num_groups), received "
            f"indices={observed_indices.tolist()} and num_groups={num_groups}"
        )
    dense_indices = torch.arange(num_groups, device=group_indices.device, dtype=torch.int64)
    if require_local_coverage and not torch.equal(observed_indices, dense_indices):
        raise ValueError(
            "expected dense local group indices covering [0, num_groups), received "
            f"indices={observed_indices.tolist()} and num_groups={num_groups}"
        )


def reduce_group_sums(
    accelerator: Accelerator,
    local_values: torch.Tensor,
    group_indices: torch.Tensor,
    num_groups: int,
    *,
    reduce_across_ranks: bool = True,
    allow_sparse_local_groups: bool = False,
) -> torch.Tensor:
    """Scatter-add local values and optionally sum matching groups across ranks.

    Args:
        accelerator: Accelerator providing the cross-rank sum collective.
        local_values: Per-sample values with shape ``(B, ...)``.
        group_indices: Dense int64 group indices with shape ``(B,)``.
        num_groups: Number of groups represented on each rank.
        reduce_across_ranks: Whether to ``accelerator.reduce`` group sums.
            TDM-R1 keeps complete groups rank-local under ``group_contiguous``.
        allow_sparse_local_groups: Whether a rank may omit groups that other
            ranks contribute to the shared dense group-id space.

    Returns:
        Detached group sums with shape ``(num_groups, ...)``.
    """
    if not isinstance(reduce_across_ranks, bool):
        raise TypeError(
            "expected reduce_across_ranks for group reduction to be bool, "
            f"received {type(reduce_across_ranks).__name__}: {reduce_across_ranks!r}"
        )
    _validate_group_layout(
        local_values,
        group_indices,
        num_groups,
        require_local_coverage=not allow_sparse_local_groups,
    )

    detached_values = local_values.detach()
    group_sums = torch.zeros(
        (num_groups, *detached_values.shape[1:]),
        device=detached_values.device,
        dtype=detached_values.dtype,
    )
    scatter_indices = group_indices.reshape(
        (group_indices.shape[0],) + (1,) * (detached_values.ndim - 1)
    ).expand_as(detached_values)
    group_sums.scatter_add_(0, scatter_indices, detached_values)

    if not reduce_across_ranks or accelerator.num_processes <= 1:
        return group_sums

    reduced_sums = accelerator.reduce(group_sums, reduction="sum")
    if not isinstance(reduced_sums, torch.Tensor):
        raise TypeError(
            "expected cross-rank reduction to return a torch.Tensor, "
            f"received {type(reduced_sums).__name__}: {reduced_sums!r}"
        )
    if reduced_sums.shape != group_sums.shape:
        raise ValueError(
            "expected cross-rank reduction to preserve group-sum shape "
            f"{tuple(group_sums.shape)}, received shape {tuple(reduced_sums.shape)}"
        )
    if reduced_sums.dtype != group_sums.dtype:
        raise TypeError(
            "expected cross-rank reduction to preserve group-sum dtype "
            f"{group_sums.dtype}, received dtype {reduced_sums.dtype}"
        )
    if reduced_sums.device != group_sums.device:
        raise ValueError(
            "expected cross-rank reduction to preserve group-sum device "
            f"{group_sums.device}, received device {reduced_sums.device}"
        )
    return reduced_sums.detach()


def _validate_preference_values(
    batch: GroupPreferenceBatch,
    trainable_values: torch.Tensor,
    reference_values: torch.Tensor,
) -> None:
    """Validate aligned scalar values at the preference boundary."""
    if not isinstance(batch, GroupPreferenceBatch):
        raise TypeError(
            "expected batch for group preference to be GroupPreferenceBatch, "
            f"received {type(batch).__name__}: {batch!r}"
        )

    named_values = {
        "advantages": batch.advantages,
        "trainable_values": trainable_values,
        "reference_values": reference_values,
    }
    for name, values in named_values.items():
        if not isinstance(values, torch.Tensor):
            raise TypeError(
                f"expected {name} for group preference to be a torch.Tensor, "
                f"received {type(values).__name__}: {values!r}"
            )
        if values.ndim != 1 or values.shape[0] == 0:
            raise ValueError(
                f"expected {name} for group preference to have non-empty shape (B,), "
                f"received shape {tuple(values.shape)}"
            )
        if not values.is_floating_point():
            raise TypeError(
                f"expected {name} for group preference to use a floating dtype, "
                f"received dtype {values.dtype}"
            )

    expected_shape = trainable_values.shape
    if batch.advantages.shape != expected_shape or reference_values.shape != expected_shape:
        raise ValueError(
            "expected advantages, trainable_values, and reference_values to have the same shape, "
            f"received {tuple(batch.advantages.shape)}, {tuple(trainable_values.shape)}, and "
            f"{tuple(reference_values.shape)}"
        )
    expected_dtype = trainable_values.dtype
    if batch.advantages.dtype != expected_dtype or reference_values.dtype != expected_dtype:
        raise TypeError(
            "expected advantages, trainable_values, and reference_values to have the same dtype, "
            f"received {batch.advantages.dtype}, {trainable_values.dtype}, and "
            f"{reference_values.dtype}"
        )
    expected_device = trainable_values.device
    if batch.advantages.device != expected_device or reference_values.device != expected_device:
        raise ValueError(
            "expected advantages, trainable_values, and reference_values on the same device, "
            f"received {batch.advantages.device}, {trainable_values.device}, and "
            f"{reference_values.device}"
        )


def _validate_sign_partitions(
    group_membership: torch.Tensor,
) -> None:
    """Require positive and non-positive members in every global group."""
    if group_membership.ndim != 2 or group_membership.shape[1] != 2:
        raise ValueError(
            "expected reduced group membership with shape (num_groups, 2), "
            f"received shape {tuple(group_membership.shape)}"
        )
    missing = (group_membership[:, 0] == 0) | (group_membership[:, 1] == 0)
    if torch.any(missing):
        missing_groups = torch.nonzero(missing, as_tuple=False).flatten().tolist()
        raise ValueError(
            "expected every required group preference partition to contain both positive "
            "(advantage > 0) and negative (advantage <= 0) members; "
            f"group {missing_groups[0]} is missing a positive or negative partition "
            f"(all missing groups={missing_groups})"
        )


def group_preference_loss(
    accelerator: Accelerator,
    batch: GroupPreferenceBatch,
    trainable_values: torch.Tensor,
    reference_values: torch.Tensor,
    beta: float,
    *,
    require_both_signs: bool = False,
) -> torch.Tensor:
    """Compute the legacy or strict paper-valued group-preference loss.

    Args:
        accelerator: Accelerator providing cross-rank group reductions.
        batch: Dense group indices and normalized per-sample advantages.
        trainable_values: Per-sample values carrying the output gradient.
        reference_values: Per-sample frozen-reference values.
        beta: Finite scale applied to trainable-reference deltas.
        require_both_signs: Whether every global group must contain an
            ``advantage > 0`` and an ``advantage <= 0`` member.

    Returns:
        Legacy detached-weight surrogate when ``require_both_signs=False``.
        Strict mode returns ``mean(softplus(-group_logit))`` exactly while
        preserving the detached-weight preference gradient through
        ``trainable_values`` only.

    Note:
        Strict ``group_logit`` is positive when the trainable density improves
        preferred samples and worsens rejected samples relative to the reference.
    """
    _validate_preference_values(batch, trainable_values, reference_values)
    if isinstance(beta, bool) or not isinstance(beta, Real):
        raise TypeError(
            "expected beta for group preference to be a finite real number, "
            f"received {type(beta).__name__}: {beta!r}"
        )
    beta_value = float(beta)
    if not math.isfinite(beta_value):
        raise ValueError(f"expected finite beta for group preference, received beta={beta_value!r}")
    if not isinstance(require_both_signs, bool):
        raise TypeError(
            "expected require_both_signs for group preference to be bool, "
            f"received {type(require_both_signs).__name__}: {require_both_signs!r}"
        )
    if isinstance(batch.group_size, bool) or not isinstance(batch.group_size, int):
        raise TypeError(
            "expected positive non-bool int for group_size in group preference, "
            f"received {type(batch.group_size).__name__}: {batch.group_size!r}"
        )
    if batch.group_size <= 0:
        raise ValueError(
            "expected positive non-bool int for group_size in group preference, "
            f"received group_size={batch.group_size}"
        )

    advantages = batch.advantages.detach()
    _validate_group_layout(
        trainable_values,
        batch.local_group_indices,
        batch.num_groups,
        require_local_coverage=not batch.allow_sparse_local_groups,
    )
    delta = trainable_values.detach() - reference_values.detach()
    local_preference = advantages * beta_value * delta / batch.group_size
    if require_both_signs:
        local_membership = torch.stack(
            ((advantages > 0).to(advantages.dtype), (advantages <= 0).to(advantages.dtype)),
            dim=1,
        )
        local_reduction = torch.cat((local_preference.unsqueeze(1), local_membership), dim=1)
        group_reduction = reduce_group_sums(
            accelerator,
            local_reduction,
            batch.local_group_indices,
            batch.num_groups,
            reduce_across_ranks=batch.reduce_across_ranks,
            allow_sparse_local_groups=batch.allow_sparse_local_groups,
        )
        group_logits = group_reduction[:, 0]
        _validate_sign_partitions(group_reduction[:, 1:])
    else:
        group_logits = reduce_group_sums(
            accelerator,
            local_preference,
            batch.local_group_indices,
            batch.num_groups,
            reduce_across_ranks=batch.reduce_across_ranks,
            allow_sparse_local_groups=batch.allow_sparse_local_groups,
        )
    group_weights = torch.sigmoid(group_logits)[batch.local_group_indices].detach()
    if not require_both_signs:
        return (group_weights * advantages * trainable_values).mean()

    # Positive means the trainable density improved preferred samples and
    # worsened rejected samples relative to the reference.
    paper_group_logits = -group_logits
    paper_value = F.softplus(-paper_group_logits).mean()
    gradient_surrogate = (group_weights * advantages * trainable_values).mean()
    return paper_value + (gradient_surrogate - gradient_surrogate.detach())
