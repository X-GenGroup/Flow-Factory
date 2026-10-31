# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest
import torch
from PIL import Image

from flow_factory.data_utils.offline_condition_cache import build_offline_condition_cache
from flow_factory.data_utils.schema import normalize_v2_record
from flow_factory.models.minimax_h3.adapters import (
    MiniMaxH3FL2VAAdapter,
    MiniMaxH3Ref2VAAdapter,
    MiniMaxH3T2VAAdapter,
)
from flow_factory.samples import (
    ComponentTimes,
    LatentState,
    MiniMaxH3T2VASample,
    StackedSampleBatch,
    StructuredTrajectory,
)


def _adapter(adapter_class: type, transformer: Any = None) -> Any:
    adapter = object.__new__(adapter_class)
    adapter.pipeline = SimpleNamespace()
    adapter.accelerator = SimpleNamespace(device=torch.device("cpu"))
    adapter.scheduler = SimpleNamespace(shift=12.0)
    adapter.audio_scheduler = SimpleNamespace(shift=3.0)
    adapter.training_args = SimpleNamespace(latent_storage_dtype=None)
    adapter.get_component = lambda name: transformer or SimpleNamespace()
    adapter.on_load_components = lambda components, device=None: None
    return adapter


@pytest.mark.parametrize(
    ("adapter_class", "kwargs", "expected"),
    [
        (
            MiniMaxH3T2VAAdapter,
            {},
            {"prompt": "describe", "height": 64, "width": 96, "num_frames": 5},
        ),
        (
            MiniMaxH3FL2VAAdapter,
            {"images": [["first", "last"]]},
            {
                "prompt": "describe",
                "image": "first",
                "last_image": "last",
                "height": 64,
                "width": 96,
                "num_frames": 5,
            },
        ),
        (
            MiniMaxH3FL2VAAdapter,
            {
                "images": [["ending"]],
                "image_slots": [["last_frame"]],
            },
            {
                "prompt": "describe",
                "last_image": "ending",
                "height": 64,
                "width": 96,
                "num_frames": 5,
            },
        ),
    ],
)
def test_preprocess_uses_exact_workflow_inputs_and_b1(
    monkeypatch, adapter_class: type, kwargs: Dict[str, Any], expected: Dict[str, Any]
) -> None:
    calls: List[Any] = []
    monkeypatch.setattr(
        "flow_factory.models.minimax_h3.workflow.encode_h3_workflow_inputs",
        lambda pipeline, values, workflow: calls.append((workflow, values))
        or {"prompt_embeds": torch.zeros(1, 2, 4)},
        raising=False,
    )
    adapter = _adapter(adapter_class)

    result = adapter.preprocess_func(
        prompt=["describe"], height=64, width=96, num_frames=5, **kwargs
    )

    assert calls == [(adapter.workflow, expected)]
    assert result["prompt_embeds"].shape[0] == 1
    with pytest.raises(ValueError, match=r"workflow=.*B=1.*received"):
        adapter.preprocess_func(prompt=["one", "two"], height=64, width=96, num_frames=5)


def test_v2_last_only_condition_reaches_h3_preprocess_through_arrow(tmp_path, monkeypatch) -> None:
    """The complete public-schema/cache path preserves a sparse last-frame binding."""
    ending_path = tmp_path / "ending.png"
    Image.new("RGB", (16, 16), color=(12, 34, 56)).save(ending_path)
    record = normalize_v2_record(
        {
            "schema_version": 2,
            "input": {
                "prompt": "Reveal what led to this ending.",
                "media": [
                    {
                        "type": "image",
                        "path": ending_path.name,
                        "slot": "last_frame",
                    }
                ],
            },
            "supervision": {
                "type": "demonstration",
                "target": {
                    "media": [
                        {"type": "video", "path": "target.mp4", "fps": 24.0},
                        {
                            "type": "audio",
                            "path": "target.wav",
                            "sample_rate": 32000,
                        },
                    ]
                },
            },
            "metadata": {},
        },
        dataset_dir=tmp_path,
    )
    calls: List[Any] = []
    monkeypatch.setattr(
        "flow_factory.models.minimax_h3.workflow.encode_h3_workflow_inputs",
        lambda pipeline, values, workflow: calls.append((workflow, values))
        or {"prompt_embeds": torch.zeros(1, 2, 4)},
    )
    adapter = _adapter(MiniMaxH3FL2VAAdapter)

    cache = build_offline_condition_cache(
        [record],
        source_name="h3-last-only",
        dataset_dir=tmp_path,
        cache_dir=tmp_path / "cache",
        preprocess_func=adapter.preprocess_func,
        preprocess_kwargs={
            "height": 64,
            "width": 96,
            "num_frames": 124,
        },
        pipeline_io_contract=adapter.pipeline_io_contract,
        preprocessing_batch_size=1,
    )

    assert len(cache) == 1
    assert len(calls) == 1
    workflow, values = calls[0]
    assert workflow == "fl2va"
    assert "image" not in values
    assert isinstance(values["last_image"], Image.Image)
    assert values["last_image"].size == (16, 16)


def test_preprocess_adds_outer_batch_to_arrow_cache_fields(monkeypatch) -> None:
    monkeypatch.setattr(
        "flow_factory.models.minimax_h3.workflow.encode_h3_workflow_inputs",
        lambda *args, **kwargs: {
            "prompt_embeds": torch.zeros(1, 2, 4),
            "token_tags": torch.tensor([1, 1]),
            "height": 64,
        },
    )
    adapter = _adapter(MiniMaxH3T2VAAdapter)

    result = adapter.preprocess_func(
        prompt=["describe"],
        height=64,
        width=96,
        num_frames=124,
    )

    assert result["prompt_embeds"].shape == (1, 2, 4)
    assert len(result["token_tags"]) == 1
    torch.testing.assert_close(result["token_tags"][0], torch.tensor([1, 1]))
    assert result["height"] == [64]


def test_ref_preprocess_builds_ordered_pinned_objects_without_returning_them(monkeypatch) -> None:
    constructed: List[Any] = []

    def reference_type(media_type: str):
        return lambda **kwargs: constructed.append((media_type, kwargs)) or SimpleNamespace(
            media_type=media_type
        )

    monkeypatch.setattr(
        "flow_factory.models.minimax_h3.workflow.require_minimax_h3_support",
        lambda: SimpleNamespace(
            ImageReference=reference_type("image"),
            VideoReference=reference_type("video"),
            AudioReference=reference_type("audio"),
        ),
    )
    encoded_inputs: Dict[str, Any] = {}
    monkeypatch.setattr(
        "flow_factory.models.minimax_h3.workflow.encode_h3_workflow_inputs",
        lambda pipeline, values, workflow: encoded_inputs.update(values)
        or {"prompt_embeds": torch.zeros(1, 2, 4)},
        raising=False,
    )
    adapter = _adapter(MiniMaxH3Ref2VAAdapter)
    references = [
        {"type": "image", "path": "i.png", "media": "image"},
        {
            "type": "video",
            "path": "v.mp4",
            "frames": "frames",
            "fps": 24.0,
            "audio": torch.zeros(2, 8),
            "sample_rate": 32000,
        },
        {"type": "audio", "path": "a.wav", "media": torch.ones(1, 8), "sample_rate": 16000},
    ]

    result = adapter.preprocess_func(
        prompt=["describe"],
        references=[references],
        reference_manifest=["manifest"],
        height=64,
        width=96,
        num_frames=5,
    )

    assert [media_type for media_type, _ in constructed] == ["image", "video", "audio"]
    assert constructed[1][1]["frames"] == "frames"
    assert "video" not in constructed[1][1]
    assert [ref.media_type for ref in encoded_inputs["references"]] == [
        "image",
        "video",
        "audio",
    ]
    assert result["reference_manifest"] == ["manifest"]
    assert "references" not in result
    assert all(not isinstance(value, SimpleNamespace) for value in result.values())


def _state(value: float = 0.0) -> LatentState:
    return LatentState(
        {
            "video": torch.full((1, 2, 96), value, dtype=torch.float32),
            "audio": torch.full((1, 3, 32), value, dtype=torch.float32),
        }
    )


def _times(value: float = 1.0, next_value: float = 0.0) -> ComponentTimes:
    return ComponentTimes(
        timestep={name: torch.tensor([value * 1000]) for name in ("video", "audio")},
        next_timestep={name: torch.tensor([next_value * 1000]) for name in ("video", "audio")},
        sigma={name: torch.tensor([value]) for name in ("video", "audio")},
        next_sigma={name: torch.tensor([next_value]) for name in ("video", "audio")},
    )


@pytest.mark.parametrize(
    "adapter_class",
    [MiniMaxH3T2VAAdapter, MiniMaxH3FL2VAAdapter, MiniMaxH3Ref2VAAdapter],
)
def test_training_times_map_primary_video_coordinate_to_audio_shift(
    adapter_class: type,
) -> None:
    adapter = _adapter(adapter_class)

    times = adapter.build_training_component_times(torch.tensor([500.0]))

    assert tuple(times.timestep) == ("video", "audio")
    assert times.timestep["video"].item() == pytest.approx(500.0)
    assert times.timestep["audio"].item() != pytest.approx(500.0)
    assert times.next_timestep["video"].item() == 0
    assert times.next_timestep["audio"].item() == 0


def test_inference_collects_structured_target_only_trajectory(monkeypatch) -> None:
    calls: List[Any] = []
    cache_events: List[Any] = []
    prepared_values: List[Dict[str, Any]] = []
    prefixes = {
        "video": torch.ones(1, 1, 96),
        "audio": torch.ones(1, 1, 32),
    }
    schedules = {
        "video": (torch.tensor([1000.0, 500.0, 0.0]), torch.tensor([1.0, 0.5, 0.0])),
        "audio": (torch.tensor([1000.0, 300.0, 0.0]), torch.tensor([1.0, 0.3, 0.0])),
    }

    def prepare(pipeline, values, **kwargs):
        del pipeline, kwargs
        prepared_values.append(values)
        return _state(), prefixes

    monkeypatch.setattr(
        "flow_factory.models.minimax_h3.workflow.prepare_h3_rollout_state",
        prepare,
        raising=False,
    )
    monkeypatch.setattr(
        "flow_factory.models.minimax_h3.workflow.build_h3_schedule_plan",
        lambda *args, **kwargs: SimpleNamespace(schedules=schedules),
        raising=False,
    )

    def forward(*args, **kwargs):
        cache_events.append(("step", len(calls)))
        calls.append((args[1], kwargs))
        value = float(len(calls))
        return SimpleNamespace(
            next_state=_state(value),
            next_state_mean=_state(value + 20),
            log_prob=torch.tensor([value]),
            component_log_probs={
                "video": torch.tensor([value]),
                "audio": torch.tensor([value]),
            },
            velocity=_state(value + 10),
        )

    monkeypatch.setattr(
        "flow_factory.models.minimax_h3.workflow.forward_h3_state", forward, raising=False
    )
    monkeypatch.setattr(
        "flow_factory.models.minimax_h3.workflow.decode_h3_targets",
        lambda *args, **kwargs: (torch.zeros(1, 2, 3, 4, 4), torch.zeros(1, 2, 16), 32000),
        raising=False,
    )

    class CacheAwareTransformer(torch.nn.Module):
        is_cache_enabled = True

        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def _reset_stateful_cache(self) -> None:
            cache_events.append(("reset", len(calls)))

        @contextmanager
        def cache_context(self, name: str):
            cache_events.append(("enter", name, len(calls)))
            try:
                yield
            finally:
                cache_events.append(("exit", name, len(calls)))

    adapter = _adapter(MiniMaxH3T2VAAdapter, transformer=CacheAwareTransformer())
    adapter_forward = adapter.forward
    adapter_forward_calls: List[Dict[str, Any]] = []
    adapter_decode = adapter.decode_latents
    adapter_decode_calls: List[Dict[str, Any]] = []
    adapter.forward = lambda **kwargs: adapter_forward_calls.append(kwargs) or adapter_forward(
        **kwargs
    )
    adapter.decode_latents = lambda latents, **kwargs: adapter_decode_calls.append(
        {"latents": latents, **kwargs}
    ) or adapter_decode(latents, **kwargs)

    samples = adapter.inference(
        prompt=["describe"],
        prompt_embeds=torch.zeros(1, 2, 4),
        layout={
            "video_indices": torch.arange(3),
            "audio_indices": torch.arange(3, 7),
            "text_indices": torch.arange(7, 9),
            "num_condition_video_rows": 1,
            "num_condition_audio_rows": 1,
        },
        geometry={
            "num_latent_frames": 1,
            "latent_height": 2,
            "latent_width": 2,
            "num_audio_latents": 3,
        },
        num_inference_steps=2,
        trajectory_indices=[0, 2],
        compute_log_prob=True,
        extra_call_back_kwargs=["velocity", "next_latents", "next_latents_mean"],
    )

    assert len(calls) == 2
    assert prepared_values[0]["num_condition_video_rows"] == 1
    assert prepared_values[0]["num_condition_audio_rows"] == 1
    assert prepared_values[0]["num_latent_frames"] == 1
    assert len(adapter_forward_calls) == 2
    assert len(adapter_decode_calls) == 1
    assert adapter_decode_calls[0]["geometry"]["num_latent_frames"] == 1
    sample = samples[0]
    assert isinstance(sample.trajectory, StructuredTrajectory)
    assert sample.trajectory.components["video"].state_index_map.tolist() == [0, -1, 1]
    assert sample.trajectory.log_prob_index_map.tolist() == [0, -1]
    assert sample.timesteps is sample.all_latents is sample.latent_index_map is None
    assert sample.log_probs is sample.log_prob_index_map is None
    assert sample.audio.shape == (2, 16)
    assert sample.audio_sample_rate == 32000
    assert prefixes["video"].shape[1] == 1
    assert sample.trajectory.components["video"].states.shape[-1] == 96
    assert sample.trajectory.components["audio"].states.shape[-1] == 32
    assert sample.trajectory.callbacks["velocity"]["video"].index_map.tolist() == [0, -1]
    assert sample.trajectory.callbacks["velocity"]["audio"].values.shape[-1] == 32
    assert torch.equal(
        sample.trajectory.callbacks["next_latents"]["video"].values[0],
        _state(1).components["video"][0],
    )
    assert torch.equal(
        sample.trajectory.callbacks["next_latents_mean"]["audio"].values[0],
        _state(21).components["audio"][0],
    )

    final_only = adapter.inference(
        prompt=["describe"],
        prompt_embeds=torch.zeros(1, 2, 4),
        layout={
            "video_indices": torch.arange(3),
            "audio_indices": torch.arange(3, 7),
            "text_indices": torch.arange(7, 9),
            "num_condition_video_rows": 1,
            "num_condition_audio_rows": 1,
        },
        geometry={
            "num_latent_frames": 1,
            "latent_height": 2,
            "latent_width": 2,
            "num_audio_latents": 3,
        },
        num_inference_steps=2,
        trajectory_indices=[2],
        compute_log_prob=True,
    )[0]
    assert final_only.trajectory is not None
    assert final_only.trajectory.log_probs is None
    assert final_only.trajectory.log_prob_index_map is None
    assert [event for event in cache_events if event[0] == "reset"] == [
        ("reset", 0),
        ("reset", 2),
    ]
    assert [event for event in cache_events if event[0] == "step"] == [
        ("step", 0),
        ("step", 1),
        ("step", 2),
        ("step", 3),
    ]
    assert [event[1] for event in cache_events if event[0] == "enter"] == ["minimax_h3_t2va"] * 4


def test_inference_generator_seeds_only_the_initial_latent(monkeypatch) -> None:
    """A request generator must not be forwarded into H3 denoising steps."""
    schedules = {
        "video": (torch.tensor([1000.0, 0.0]), torch.tensor([1.0, 0.0])),
        "audio": (torch.tensor([1000.0, 0.0]), torch.tensor([1.0, 0.0])),
    }
    prepared: Dict[str, Any] = {}
    stepped: List[Any] = []

    def prepare(pipeline, values, **kwargs):
        del pipeline, values
        prepared["generator"] = kwargs.get("generator")
        return _state(), {"video": torch.ones(1, 1, 96), "audio": torch.ones(1, 1, 32)}

    def forward(*args, **kwargs):
        del args
        stepped.append(kwargs.get("generator"))
        return SimpleNamespace(
            next_state=_state(1.0),
            log_prob=torch.tensor([1.0]),
            component_log_probs={"video": torch.tensor([1.0]), "audio": torch.tensor([1.0])},
        )

    monkeypatch.setattr(
        "flow_factory.models.minimax_h3.workflow.prepare_h3_rollout_state", prepare, raising=False
    )
    monkeypatch.setattr(
        "flow_factory.models.minimax_h3.workflow.build_h3_schedule_plan",
        lambda *args, **kwargs: SimpleNamespace(schedules=schedules),
        raising=False,
    )
    monkeypatch.setattr(
        "flow_factory.models.minimax_h3.workflow.forward_h3_state", forward, raising=False
    )
    monkeypatch.setattr(
        "flow_factory.models.minimax_h3.workflow.decode_h3_targets",
        lambda *args, **kwargs: (torch.zeros(1, 2, 3, 4, 4), torch.zeros(1, 2, 16), 32000),
        raising=False,
    )

    class _Transformer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def _reset_stateful_cache(self) -> None:
            return None

        @contextmanager
        def cache_context(self, name: str):
            del name
            yield

    adapter = _adapter(MiniMaxH3T2VAAdapter, transformer=_Transformer())
    adapter.decode_latents = lambda latents, **kwargs: (
        torch.zeros(1, 2, 3, 4, 4),
        torch.zeros(1, 2, 16),
        32000,
    )
    sentinel = torch.Generator()
    adapter.inference(
        prompt=["describe"],
        prompt_embeds=torch.zeros(1, 2, 4),
        generator=sentinel,
        layout={
            "video_indices": torch.arange(3),
            "audio_indices": torch.arange(3, 7),
            "text_indices": torch.arange(7, 9),
            "num_condition_video_rows": 0,
            "num_condition_audio_rows": 0,
        },
        geometry={
            "num_latent_frames": 1,
            "latent_height": 2,
            "latent_width": 2,
            "num_audio_latents": 3,
        },
        num_inference_steps=1,
        trajectory_indices=None,
        compute_log_prob=False,
    )
    assert prepared["generator"] is sentinel
    assert stepped == [None]


def test_ref_forward_cache_context_owns_inner_while_forward_uses_prepared_route(
    monkeypatch,
) -> None:
    from flow_factory.models.model_bundle import RoutedComponentProxy

    events: List[Any] = []

    class CacheAwareInner(torch.nn.Module):
        is_cache_enabled = True

        @contextmanager
        def cache_context(self, name: str):
            events.append(("enter", name))
            try:
                yield
            finally:
                events.append(("exit", name))

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            events.append(("inner",))
            return value

    inner = CacheAwareInner()

    class PreparedBundle(torch.nn.Module):
        def forward(self, name: str, *args: Any, **kwargs: Any) -> Any:
            events.append(("bundle", name))
            return inner(*args, **kwargs)

    proxy = RoutedComponentProxy(
        PreparedBundle(),
        "transformer_ref",
        inner,
    )

    def forward(transformer, state, *args, **kwargs):
        transformer(torch.ones(1))
        return state

    monkeypatch.setattr("flow_factory.models.minimax_h3.workflow.forward_h3_state", forward)
    adapter = _adapter(MiniMaxH3Ref2VAAdapter, transformer=proxy)

    adapter.forward(
        state=_state(),
        times=_times(),
        condition_prefixes={},
        prompt_embeds=torch.zeros(1, 2, 4),
        layout={},
        return_fields=("velocity",),
    )

    assert events == [
        ("enter", "minimax_h3_ref2va"),
        ("bundle", "transformer_ref"),
        ("inner",),
        ("exit", "minimax_h3_ref2va"),
    ]


def test_forward_state_uses_prepared_component_and_forward_parity(monkeypatch) -> None:
    weight = torch.nn.Parameter(torch.tensor(2.0))

    class Transformer:
        pass

    prepared = Transformer()
    prepared.weight = weight
    calls: List[Any] = []

    def forward(transformer, state, *args, **kwargs):
        calls.append((transformer, kwargs))
        scaled = LatentState(
            {name: values * transformer.weight for name, values in state.components.items()}
        )
        return SimpleNamespace(
            next_state=scaled,
            log_prob=torch.ones(1),
            component_log_probs={"video": torch.ones(1), "audio": torch.ones(1)},
            velocity=scaled,
        )

    monkeypatch.setattr(
        "flow_factory.models.minimax_h3.workflow.forward_h3_state", forward, raising=False
    )
    adapter = _adapter(MiniMaxH3T2VAAdapter, transformer=prepared)
    batch = StackedSampleBatch(
        [
            MiniMaxH3T2VASample(
                prompt="describe",
                prompt_embeds=torch.zeros(2, 4),
                extra_kwargs={
                    "condition_prefixes": {
                        "video": torch.zeros(1, 96),
                        "audio": torch.zeros(1, 32),
                    },
                    "layout": {},
                    "advantage": torch.tensor(9),
                },
            )
        ]
    )
    state = _state(1)
    times = _times()

    bridged = adapter.forward_state(
        batch=batch,
        state=state,
        times=times,
        compute_log_prob=True,
        trainer_metadata="drop",
    )
    direct = adapter.forward(
        state=state,
        times=times,
        compute_log_prob=True,
        prompt_embeds=batch["prompt_embeds"],
        condition_prefixes=batch["condition_prefixes"],
        layout=batch["layout"],
    )
    bridged.next_state.components["video"].sum().backward()

    assert torch.equal(
        bridged.next_state.components["video"], direct.next_state.components["video"]
    )
    assert calls[0][0] is prepared
    assert weight.grad is not None
    assert "advantage" not in calls[0][1]
    assert "trainer_metadata" not in calls[0][1]


def test_forward_rejects_replay_batch_argument() -> None:
    adapter = _adapter(MiniMaxH3T2VAAdapter)

    with pytest.raises(
        ValueError,
        match=r"workflow='t2va' forward received unsupported arguments=\('batch',\)",
    ):
        adapter.forward(batch=SimpleNamespace(), state=_state(1), times=_times())
