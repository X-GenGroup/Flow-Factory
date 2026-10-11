"""Forward-argument construction shared by trainer runtimes."""

from collections.abc import Mapping
from typing import Any


def _batch_preferred_kwargs(
    configured: Mapping[str, Any],
    batch: Mapping[str, Any],
) -> dict[str, Any]:
    """Return configured values for keys not already carried by ``batch``.

    Collated batch values take precedence over configured defaults; that precedence
    is part of the replay contract.
    """
    return {key: value for key, value in configured.items() if key not in batch}


def training_forward_kwargs(trainer: Any, batch: Mapping[str, Any]) -> dict[str, Any]:
    """Return training defaults while preserving batch-key precedence."""
    return _batch_preferred_kwargs({**trainer.training_args}, batch)


def replay_forward_kwargs(trainer: Any, batch: Mapping[str, Any]) -> dict[str, Any]:
    """Return replay defaults while preserving batch-key precedence."""
    return training_forward_kwargs(trainer, batch)


def reference_forward_kwargs(
    trainer: Any,
    batch: Mapping[str, Any],
    **overrides: Any,
) -> dict[str, Any]:
    """Return replay defaults with explicit reference-pass overrides."""
    kwargs = replay_forward_kwargs(trainer, batch)
    kwargs.update(overrides)
    return kwargs
