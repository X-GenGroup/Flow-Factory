import pytest

from flow_factory.hparams import GRPOTrainingArguments
from flow_factory.utils.base import filter_kwargs


def _adapter_inference(prompt=None, max_sequence_length: int = 1024):
    """Stand-in for an adapter whose own sequence-length default is not 512."""
    return max_sequence_length


def test_unset_max_sequence_length_leaves_the_adapter_default_alone() -> None:
    """A declared field would be forwarded always and silently halve Qwen/LTX2."""
    training_args = GRPOTrainingArguments.from_dict({"seed": 7})

    forwarded = filter_kwargs(_adapter_inference, **dict(training_args))

    assert "max_sequence_length" not in forwarded


def test_configured_max_sequence_length_reaches_the_adapter() -> None:
    """Undeclared keys still travel to the adapter through extra_kwargs."""
    training_args = GRPOTrainingArguments.from_dict({"max_sequence_length": 2048})

    forwarded = filter_kwargs(_adapter_inference, **dict(training_args))

    assert forwarded["max_sequence_length"] == 2048


@pytest.mark.parametrize("value", [0, -0.1, float("inf"), float("nan")])
def test_reward_overlap_poll_interval_must_be_finite_and_positive(value: float) -> None:
    with pytest.raises(ValueError, match="finite and > 0"):
        GRPOTrainingArguments.from_dict({"reward_optimization_overlap_poll_interval": value})


def test_reward_overlap_mode_is_validated_even_when_overlap_is_disabled() -> None:
    with pytest.raises(ValueError, match="must be 'ordered' or 'ready'"):
        GRPOTrainingArguments.from_dict({"reward_optimization_overlap_mode": "random"})


def test_reward_overlap_mode_defaults_to_ready() -> None:
    training_args = GRPOTrainingArguments.from_dict({})

    assert training_args.reward_optimization_overlap_mode == "ready"
