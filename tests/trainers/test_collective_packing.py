from types import SimpleNamespace

import numpy as np
import pytest
import torch

import flow_factory.utils.dist as dist_utils
from flow_factory.advantage import AdvantageProcessor, CollectedGroupLayout
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
    assert accelerator.calls[0].shape == (2, 3)
    assert accelerator.calls[0].dtype == torch.int64
    assert accelerator.calls[1].shape == (2, 4)
    assert accelerator.calls[1].dtype == torch.float32
    assert len(gathered) == 4
    torch.testing.assert_close(gathered[0].prompt_embeds, samples[0].prompt_embeds)
    torch.testing.assert_close(
        gathered[3].negative_prompt_embeds, samples[1].negative_prompt_embeds
    )
    torch.testing.assert_close(gathered[2].prompt_ids, samples[0].prompt_ids)


def test_gather_samples_preserves_concrete_reconstruction_fields(monkeypatch) -> None:
    monkeypatch.setattr(
        dist_utils,
        "gather_object",
        lambda _values: (_ for _ in ()).throw(
            AssertionError("annotated string fields must use tensor collectives")
        ),
    )
    accelerator = GatherRecorder()
    manifest = '[{"path":"condition.png","type":"image"}]'
    sample = MiniMaxH3Ref2VASample(
        prompt="参考图像 prompt 🌏",
        reference_manifest=manifest,
    )

    gathered = gather_samples(accelerator, [sample], ["prompt"])

    assert len(accelerator.calls) == 2
    assert accelerator.calls[0].dtype == torch.int64
    assert accelerator.calls[1].dtype == torch.uint8
    assert len(gathered) == 2
    assert all(isinstance(item, MiniMaxH3Ref2VASample) for item in gathered)
    assert [item.prompt for item in gathered] == [sample.prompt, sample.prompt]
    assert [item.reference_manifest for item in gathered] == [manifest, manifest]


def test_gather_samples_can_select_extra_fields_without_pickle(monkeypatch) -> None:
    monkeypatch.setattr(
        dist_utils,
        "gather_object",
        lambda _values: (_ for _ in ()).throw(
            AssertionError("selected tensor extras must not use pickle collectives")
        ),
    )
    accelerator = GatherRecorder()
    sample = BaseSample(
        prompt_ids=torch.tensor([1, 2, 3]),
        extra_kwargs={
            "advantage": torch.tensor(0.5),
            "rewards": {"ocr": torch.tensor(1.0)},
        },
    )

    gathered = gather_samples(
        accelerator,
        [sample],
        ["prompt_ids"],
        extra_field_names=["advantage"],
    )

    assert len(gathered) == 2
    assert all(set(item.extra_kwargs) == {"advantage"} for item in gathered)
    torch.testing.assert_close(gathered[1].extra_kwargs["advantage"], torch.tensor(0.5))


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


def test_gather_samples_packs_optional_scalar_metadata_without_pickle(monkeypatch) -> None:
    monkeypatch.setattr(
        dist_utils,
        "gather_object",
        lambda _values: (_ for _ in ()).throw(
            AssertionError("optional scalar metadata must use the tensor collective")
        ),
    )
    accelerator = GatherRecorder()
    samples = [
        BaseSample(
            source_id=3,
            sampling_group_id=10 + index,
            sampling_group_member_id=index,
            sampling_sample_id=100 + index,
        )
        for index in range(2)
    ]

    gathered = gather_samples(
        accelerator,
        samples,
        [
            "source",
            "source_id",
            "sampling_group_id",
            "sampling_group_member_id",
            "sampling_sample_id",
            "_unique_id",
        ],
    )

    assert len(accelerator.calls) == 1
    assert accelerator.calls[0].device.type == "cpu"
    assert accelerator.calls[0].dtype == torch.int64
    assert [sample.source for sample in gathered] == [None] * 4
    assert [sample.source_id for sample in gathered] == [3, 3, 3, 3]
    assert [sample.unique_id for sample in gathered] == [10, 11, 10, 11]
    assert [sample.sampling_sample_id for sample in gathered] == [100, 101, 100, 101]


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
    assert len(gather_calls) == 2
    torch.testing.assert_close(gather_calls[0], torch.tensor([[0.25]]))
    torch.testing.assert_close(gather_calls[1], torch.tensor([[0, 7]], dtype=torch.int64))
    np.testing.assert_array_equal(collected["ocr"], np.array([0.25], dtype=np.float32))
    np.testing.assert_array_equal(groups, np.array([0]))
    np.testing.assert_array_equal(sources, np.array([0]))


def test_distributed_advantage_preserves_int64_group_identity() -> None:
    gather_calls: list[torch.Tensor] = []

    def gather(tensor: torch.Tensor) -> torch.Tensor:
        gather_calls.append(tensor.detach().clone())
        return tensor

    accelerator = SimpleNamespace(device=torch.device("cpu"), gather=gather)
    processor = AdvantageProcessor(
        accelerator=accelerator,
        reward_weights={"ocr": {"default": 1.0}},
        group_size=1,
        sampler_type="group_distributed",
    )
    large_id = 2**62 + 123
    samples = [BaseSample(prompt="one", _unique_id=large_id, source_id=4)]
    rewards = {"ocr": torch.tensor([0.25])}

    prepared = processor.prepare_group_reward_collection(samples, rewards)
    processor.collect_group_rewards(samples, rewards, prepared_collection=prepared)

    assert gather_calls[1].dtype == torch.int64
    assert gather_calls[1].tolist() == [[4, large_id]]


def test_distributed_advantage_reuses_precollected_group_layout() -> None:
    gather_calls: list[torch.Tensor] = []

    def gather(tensor: torch.Tensor) -> torch.Tensor:
        gather_calls.append(tensor.detach().clone())
        return tensor

    accelerator = SimpleNamespace(
        device=torch.device("cpu"),
        num_processes=1,
        gather=gather,
    )
    processor = AdvantageProcessor(
        accelerator=accelerator,
        reward_weights={"ocr": {"default": 1.0}},
        group_size=1,
        sampler_type="group_distributed",
    )
    samples = [BaseSample(prompt="one", _unique_id=2**62 + 123, source_id=4)]
    rewards = {"ocr": torch.tensor([0.25])}
    layout = CollectedGroupLayout(
        group_indices=np.array([0], dtype=np.int64),
        source_ids=np.array([4], dtype=np.int64),
        local_sample_count=1,
        num_processes=1,
    )

    prepared = processor.prepare_group_reward_collection(
        samples,
        rewards,
        collected_layout=layout,
    )
    collected, groups, sources = processor.collect_group_rewards(
        samples,
        rewards,
        prepared_collection=prepared,
    )

    assert len(gather_calls) == 1
    assert gather_calls[0].dtype == torch.float32
    np.testing.assert_array_equal(collected["ocr"], np.array([0.25], dtype=np.float32))
    np.testing.assert_array_equal(groups, np.array([0]))
    np.testing.assert_array_equal(sources, np.array([4]))


def test_streaming_weighted_sum_reduces_group_stats_without_reward_gather() -> None:
    reductions: list[torch.Tensor] = []

    def reduce(tensor: torch.Tensor, reduction: str) -> torch.Tensor:
        assert reduction == "sum"
        reductions.append(tensor.detach().clone())
        peer_stats = torch.tensor([0.0, 2.0, 7.0, 25.0], dtype=torch.float64)
        return tensor + peer_stats

    accelerator = SimpleNamespace(
        device=torch.device("cpu"),
        num_processes=2,
        process_index=0,
        gather=lambda _tensor: (_ for _ in ()).throw(
            AssertionError("streaming group statistics must not gather rewards")
        ),
        reduce=reduce,
    )
    processor = AdvantageProcessor(
        accelerator=accelerator,
        reward_weights={"ocr": {"default": 1.0}},
        group_size=4,
        global_std=False,
        sampler_type="group_tiled",
    )
    samples = [BaseSample(prompt="same"), BaseSample(prompt="same")]
    rewards = {"ocr": torch.tensor([1.0, 2.0])}
    layout = CollectedGroupLayout(
        group_indices=np.array([0, 0, 0, 0], dtype=np.int64),
        source_ids=np.array([-1, -1, -1, -1], dtype=np.int64),
        local_sample_count=2,
        num_processes=2,
    )
    prepared = processor.prepare_group_reward_collection(
        samples,
        rewards,
        collected_layout=layout,
    )

    advantages = processor._compute_advantages(
        samples,
        rewards,
        aggregation_func="sum",
        build_metrics=False,
        prepared_collection=prepared,
    )

    assert len(reductions) == 1
    torch.testing.assert_close(
        advantages,
        torch.tensor([-1.3416407865, -0.4472135955], dtype=torch.float64),
    )


def test_streaming_gdpo_normalizes_each_reward_before_weighted_aggregation() -> None:
    accelerator = SimpleNamespace(
        device=torch.device("cpu"),
        num_processes=1,
        process_index=0,
        gather=lambda _tensor: (_ for _ in ()).throw(
            AssertionError("streaming GDPO must not gather rewards")
        ),
    )
    processor = AdvantageProcessor(
        accelerator=accelerator,
        reward_weights={
            "aesthetic": {"default": 2.0},
            "ocr": {"default": 0.5},
        },
        group_size=2,
        global_std=False,
        sampler_type="group_tiled",
    )
    samples = [BaseSample(prompt="same"), BaseSample(prompt="same")]
    rewards = {
        "aesthetic": torch.tensor([1.0, 3.0]),
        "ocr": torch.tensor([4.0, 2.0]),
    }
    layout = CollectedGroupLayout(
        group_indices=np.array([0, 0], dtype=np.int64),
        source_ids=np.array([-1, -1], dtype=np.int64),
        local_sample_count=2,
        num_processes=1,
    )
    prepared = processor.prepare_group_reward_collection(
        samples,
        rewards,
        collected_layout=layout,
    )

    advantages = processor._compute_advantages(
        samples,
        rewards,
        aggregation_func="gdpo",
        build_metrics=False,
        prepared_collection=prepared,
    )

    torch.testing.assert_close(
        advantages,
        torch.tensor([-1.5, 1.5], dtype=torch.float64),
    )


def test_advantage_groups_equal_prompt_ids_separately_per_source() -> None:
    processor = AdvantageProcessor(
        accelerator=SimpleNamespace(device=torch.device("cpu")),
        reward_weights={"ocr": {"a": 1.0, "b": 1.0}},
        group_size=1,
        sampler_type="group_contiguous",
    )
    samples = [
        BaseSample(prompt="same", _unique_id=17, source_id=0),
        BaseSample(prompt="same", _unique_id=17, source_id=1),
    ]

    _rewards, groups, sources = processor.collect_group_rewards(
        samples,
        {"ocr": torch.tensor([0.25, 0.75])},
    )

    np.testing.assert_array_equal(groups, np.array([0, 1]))
    np.testing.assert_array_equal(sources, np.array([0, 1]))


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


@pytest.mark.parametrize(
    ("global_std", "expected"),
    [
        (False, torch.tensor([-2.0, 2.0, 0.0, 0.0], dtype=torch.float64)),
        (
            True,
            torch.tensor(
                [-(2**0.5), 2**0.5, 0.0, 0.0],
                dtype=torch.float64,
            ),
        ),
    ],
)
def test_gdpo_respects_global_std_for_final_combined_advantages(
    global_std: bool,
    expected: torch.Tensor,
) -> None:
    accelerator = SimpleNamespace(
        device=torch.device("cpu"),
        reduce=lambda value, reduction: value,
    )
    processor = AdvantageProcessor(
        accelerator=accelerator,
        reward_weights={
            "aesthetic": {"default": 1.0},
            "ocr": {"default": 1.0},
        },
        group_size=2,
        global_std=global_std,
        sampler_type="group_contiguous",
    )
    samples = [BaseSample(prompt=str(index), _unique_id=index // 2) for index in range(4)]

    advantages = processor.compute_gdpo(
        samples,
        {
            "aesthetic": torch.tensor([0.0, 2.0, 10.0, 14.0]),
            "ocr": torch.tensor([0.0, 4.0, 9.0, 5.0]),
        },
        store_to_samples=False,
        build_metrics=False,
    )

    torch.testing.assert_close(advantages, expected)
