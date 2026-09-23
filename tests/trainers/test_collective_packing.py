from types import SimpleNamespace

import numpy as np
import pytest
import torch

import flow_factory.utils.dist as dist_utils
from flow_factory.advantage import AdvantageProcessor
from flow_factory.samples import BaseSample, MiniMaxH3Ref2VASample
from flow_factory.utils.dist import gather_aligned_floating_tensors, gather_samples


class GatherRecorder:
    def __init__(self) -> None:
        self.calls: list[torch.Tensor] = []
        self.device = torch.device("cpu")
        self.num_processes = 2

    def gather(self, tensor: torch.Tensor) -> torch.Tensor:
        self.calls.append(tensor.detach().clone())
        return torch.cat((tensor, tensor), dim=0)


def test_aligned_floating_tensors_use_one_gather_and_round_trip_named_columns():
    accelerator = GatherRecorder()

    gathered = gather_aligned_floating_tensors(
        accelerator,
        {
            "reward": torch.tensor([3.0, 4.0], dtype=torch.float32),
            "advantage": torch.tensor([1.0, 2.0], dtype=torch.float32),
        },
    )

    assert len(accelerator.calls) == 1
    assert accelerator.calls[0].shape == (2, 2)
    assert accelerator.calls[0].dtype == torch.float32
    torch.testing.assert_close(
        gathered["advantage"],
        torch.tensor([1.0, 2.0, 1.0, 2.0], dtype=torch.float32),
    )
    torch.testing.assert_close(
        gathered["reward"],
        torch.tensor([3.0, 4.0, 3.0, 4.0], dtype=torch.float32),
    )


def test_aligned_floating_tensors_preserve_mixed_dtypes_with_separate_gathers():
    accelerator = GatherRecorder()

    gathered = gather_aligned_floating_tensors(
        accelerator,
        {
            "reward": torch.tensor([3.0, 4.0], dtype=torch.float64),
            "advantage": torch.tensor([1.0, 2.0], dtype=torch.float32),
        },
    )

    assert len(accelerator.calls) == 2
    assert gathered["advantage"].dtype == torch.float32
    assert gathered["reward"].dtype == torch.float64


def test_aligned_floating_tensors_reject_misaligned_or_nonfloating_fields():
    accelerator = GatherRecorder()

    with pytest.raises(ValueError, match="expected tensor 'reward' shape"):
        gather_aligned_floating_tensors(
            accelerator,
            {
                "advantage": torch.ones(2),
                "reward": torch.ones(3),
            },
        )
    with pytest.raises(TypeError, match="floating dtype"):
        gather_aligned_floating_tensors(
            accelerator,
            {
                "advantage": torch.ones(2),
                "reward": torch.ones(2, dtype=torch.int64),
            },
        )


def test_gather_samples_packs_same_dtype_fields_and_preserves_other_fields():
    accelerator = GatherRecorder()
    samples = [
        BaseSample(
            prompt_embeds=torch.tensor([1.0, 2.0]),
            negative_prompt_embeds=torch.tensor([3.0, 4.0]),
            prompt_ids=torch.tensor([1, 2, 3]),
        ),
        BaseSample(
            prompt_embeds=torch.tensor([5.0, 6.0]),
            negative_prompt_embeds=torch.tensor([7.0, 8.0]),
            prompt_ids=torch.tensor([4, 5, 6]),
        ),
    ]

    gathered = gather_samples(
        accelerator,
        samples,
        ["prompt_embeds", "negative_prompt_embeds", "prompt_ids"],
        device=torch.device("cpu"),
    )

    assert len(accelerator.calls) == 2
    assert accelerator.calls[0].shape == (2, 4)
    assert accelerator.calls[1].shape == (2, 3)
    assert len(gathered) == 4
    torch.testing.assert_close(gathered[0].prompt_embeds, samples[0].prompt_embeds)
    torch.testing.assert_close(
        gathered[3].negative_prompt_embeds, samples[1].negative_prompt_embeds
    )
    torch.testing.assert_close(gathered[2].prompt_ids, samples[0].prompt_ids)


def test_gather_samples_preserves_concrete_reconstruction_fields() -> None:
    accelerator = GatherRecorder()
    manifest = '[{"path":"condition.png","type":"image"}]'
    sample = MiniMaxH3Ref2VASample(
        prompt="A reference-conditioned prompt",
        reference_manifest=manifest,
    )

    gathered = gather_samples(accelerator, [sample], ["prompt"])

    assert len(gathered) == 1
    assert isinstance(gathered[0], MiniMaxH3Ref2VASample)
    assert gathered[0].prompt == sample.prompt
    assert gathered[0].reference_manifest == manifest


def test_gather_samples_keeps_large_cpu_fields_on_separate_paths(monkeypatch):
    monkeypatch.setattr(dist_utils, "_CPU_PACKED_GATHER_MAX_BYTES", 1)
    accelerator = GatherRecorder()
    samples = [
        BaseSample(
            prompt_embeds=torch.tensor([1.0, 2.0]),
            negative_prompt_embeds=torch.tensor([3.0, 4.0]),
        )
    ]

    gather_samples(
        accelerator,
        samples,
        ["prompt_embeds", "negative_prompt_embeds"],
        device=torch.device("cpu"),
    )

    assert len(accelerator.calls) == 2
    assert all(call.shape == (1, 2) for call in accelerator.calls)


def test_zero_std_ratio_rides_the_existing_batched_stats_reduction():
    reductions: list[torch.Tensor] = []

    def reduce(tensor: torch.Tensor, reduction: str = "sum") -> torch.Tensor:
        assert reduction == "sum"
        reductions.append(tensor.detach().clone())
        return tensor

    accelerator = SimpleNamespace(
        device=torch.device("cpu"),
        reduce=reduce,
    )
    processor = AdvantageProcessor(
        accelerator=accelerator,
        reward_weights={"ocr": {"default": 1.0}},
        group_size=2,
        sampler_type="group_contiguous",
    )
    rewards = np.array([1.0, 1.0, 2.0, 2.0])
    group_indices = np.array([0, 0, 1, 1])

    metrics = processor._build_weighted_sum_log_data(
        gathered_rewards={"ocr": rewards},
        group_indices=group_indices,
        aggregated_rewards=rewards,
        advantages=np.zeros(4),
        samples=[BaseSample(prompt=str(index)) for index in range(4)],
    )

    assert len(reductions) == 1
    assert metrics["train/reward_zero_std_ratio"] == 1.0


def test_distributed_advantage_payload_is_prepared_before_its_gather():
    gather_calls: list[torch.Tensor] = []

    def gather(tensor: torch.Tensor) -> torch.Tensor:
        gather_calls.append(tensor.detach().clone())
        return tensor

    accelerator = SimpleNamespace(
        device=torch.device("cpu"),
        num_processes=1,
        process_index=0,
        gather=gather,
    )
    processor = AdvantageProcessor(
        accelerator=accelerator,
        reward_weights={"ocr": {"default": 1.0}},
        group_size=1,
        sampler_type="group_distributed",
    )
    samples = [BaseSample(prompt="one", _unique_id=7, source_id=0)]
    rewards = {"ocr": torch.tensor([0.25])}

    prepared = processor.prepare_group_reward_collection(
        samples,
        rewards,
        require_all_rewards=True,
    )

    assert gather_calls == []
    collected, groups, sources = processor.collect_group_rewards(
        samples,
        rewards,
        prepared_collection=prepared,
    )
    assert len(gather_calls) == 1
    torch.testing.assert_close(gather_calls[0], torch.tensor([[0.25, 7.0, 0.0]]))
    np.testing.assert_array_equal(collected["ocr"], np.array([0.25], dtype=np.float32))
    np.testing.assert_array_equal(groups, np.array([0]))
    np.testing.assert_array_equal(sources, np.array([0]))


def test_distributed_advantage_shape_failure_happens_before_gather():
    accelerator = SimpleNamespace(
        device=torch.device("cpu"),
        gather=lambda _tensor: (_ for _ in ()).throw(
            AssertionError("invalid local payload must not enter gather")
        ),
    )
    processor = AdvantageProcessor(
        accelerator=accelerator,
        reward_weights={"ocr": {"default": 1.0}},
        group_size=1,
        sampler_type="group_distributed",
    )
    samples = [BaseSample(prompt="one", _unique_id=7)]

    with pytest.raises(ValueError, match="2 values for 1 local samples"):
        processor.prepare_group_reward_collection(
            samples,
            {"ocr": torch.tensor([0.25, 0.5])},
            require_all_rewards=True,
        )
