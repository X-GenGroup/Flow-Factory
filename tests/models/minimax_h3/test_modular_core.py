# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from flow_factory.models.minimax_h3 import blocks
from flow_factory.models.minimax_h3._common import MINIMAX_H3_COMPONENT_ORDER
from flow_factory.samples import ComponentTimes, LatentState
from flow_factory.scheduler import SDESchedulerOutput


class FakePipelineState:
    def __init__(self, values):
        self.values = values


class RecordingBlock:
    def __init__(self, name, output=None, error=None):
        self.name = name
        self.output = output or {}
        self.error = error

    def __call__(self, pipeline, state):
        pipeline.calls.append(self.name)
        if self.error is not None:
            raise self.error
        state.values.update(self.output)
        return pipeline, state


class FakeWorkflowBlock:
    def __call__(self, pipeline, state):
        name = type(self).__name__
        pipeline.calls.append(name)
        values = state.values
        if name in {"TextEncoderStep", "FL2VATextEncoderStep", "Ref2VATextEncoderStep"}:
            values["prompt_embeds"] = torch.ones(1, 2, 8)
            values["text_token_tags"] = torch.tensor([1, 1])
        elif name == "NoKeyframeAnchorsStep":
            values["keyframe_anchors"] = ()
        elif name == "ResizeStep":
            values["keyframes"] = ["resized"]
            values["keyframe_anchors"] = ("first",)
        elif name == "RefSetupStep":
            values["normalized_references"] = ["normalized"]
        elif name == "KeyframeEncoderStep":
            values["condition_latents"] = [torch.ones(1, 24, 1, 2, 2)]
        elif name == "ReferenceEncoderStep":
            values["condition_latents"] = [torch.ones(1, 24, 1, 2, 2)]
            values["audio_condition_latents"] = [torch.ones(1, 32)]
        elif name in {"PrepareLayoutStep", "RefPrepareLayoutStep"}:
            values.update(_fake_layout_cache())
        elif name == "PrepareConditionLatentsStep":
            pipeline.draws.append(
                ("condition", torch.rand((), generator=values["generator"]).item())
            )
            count = values["num_condition_video_rows"]
            values["condition_rows"] = torch.full((count, 96), 3.0)
        elif name == "PrepareLatentsStep":
            pipeline.draws.append(("video", torch.rand((), generator=values["generator"]).item()))
            pipeline.draws.append(("audio", torch.rand((), generator=values["generator"]).item()))
            values["latents"] = torch.ones(2, 96)
            values["audio_latents"] = torch.ones(3, 32)
        elif name == "FL2VAPrepareLatentsStep":
            values["latents"] = torch.cat([values["condition_rows"], values["latents"]])
        elif name == "Ref2VAPrepareLatentsStep":
            values["latents"] = torch.cat([values["condition_rows"], values["latents"]])
            values["audio_latents"] = torch.cat(
                [*values["audio_condition_latents"], values["audio_latents"]]
            )
        elif name == "AfterDenoiseStep":
            pipeline.decode_condition_counts = (
                values["num_condition_video_rows"],
                values["num_condition_audio_rows"],
            )
        elif name == "VideoDecodeStep":
            if values["output_type"] not in ("pil", "np", "pt"):
                raise ValueError("output_type must be one of pil, np, or pt")
            values["videos"] = ["video"]
        elif name == "AudioDecodeStep":
            values["audio"] = torch.ones(1, 2, 16)
            values["sampling_rate"] = 32000
        return pipeline, state


class ResizeStep(FakeWorkflowBlock):
    pass


class RefSetupStep(FakeWorkflowBlock):
    pass


class TextEncoderStep(FakeWorkflowBlock):
    pass


class FL2VATextEncoderStep(FakeWorkflowBlock):
    pass


class Ref2VATextEncoderStep(FakeWorkflowBlock):
    pass


class NoKeyframeAnchorsStep(FakeWorkflowBlock):
    pass


class KeyframeEncoderStep(FakeWorkflowBlock):
    pass


class ReferenceEncoderStep(FakeWorkflowBlock):
    pass


class PrepareLayoutStep(FakeWorkflowBlock):
    pass


class RefPrepareLayoutStep(FakeWorkflowBlock):
    pass


class PrepareConditionLatentsStep(FakeWorkflowBlock):
    pass


class PrepareLatentsStep(FakeWorkflowBlock):
    pass


class FL2VAPrepareLatentsStep(FakeWorkflowBlock):
    pass


class Ref2VAPrepareLatentsStep(FakeWorkflowBlock):
    pass


class SetTimestepsStep(FakeWorkflowBlock):
    @staticmethod
    def build_row_timesteps(
        video_indices,
        audio_indices,
        num_condition_video_rows,
        num_condition_audio_rows,
        num_text_tokens,
        video_timestep,
        audio_timestep,
        condition_video_timestep,
        condition_audio_timestep,
    ):
        length = video_indices.numel() + audio_indices.numel() + num_text_tokens
        values = torch.full((length,), video_timestep)
        values[video_indices[:num_condition_video_rows]] = condition_video_timestep
        values[audio_indices[num_condition_audio_rows:]] = audio_timestep
        values[audio_indices[:num_condition_audio_rows]] = condition_audio_timestep
        return torch.unique(values, sorted=True, return_inverse=True)


class AfterDenoiseStep(FakeWorkflowBlock):
    pass


class VideoDecodeStep(FakeWorkflowBlock):
    pass


class AudioDecodeStep(FakeWorkflowBlock):
    pass


class FakeModularPipeline:
    pass


class FakeMiniMaxH3Blocks:
    _workflow_map = {
        "t2va": {"prompt": True},
        "fl2va": (
            {"prompt": True, "image": True},
            {"prompt": True, "last_image": True},
        ),
        "ref2va": {"prompt": True, "references": True},
    }


class FakeMiniMaxH3AttnProcessor:
    def __call__(self, *args, **kwargs):
        return None


def fake_dispatch_attention_fn(*args, **kwargs):
    return None


class FakeReference:
    pass


@dataclasses.dataclass(frozen=True)
class FakeSymbols:
    MiniMaxH3ModularPipeline: type = FakeModularPipeline
    MiniMaxH3Blocks: type = FakeMiniMaxH3Blocks
    MiniMaxH3AttnProcessor: type = FakeMiniMaxH3AttnProcessor
    dispatch_attention_fn: object = fake_dispatch_attention_fn
    PipelineState: type = FakePipelineState
    ResizeStep: type = ResizeStep
    RefSetupStep: type = RefSetupStep
    TextEncoderStep: type = TextEncoderStep
    FL2VATextEncoderStep: type = FL2VATextEncoderStep
    Ref2VATextEncoderStep: type = Ref2VATextEncoderStep
    NoKeyframeAnchorsStep: type = NoKeyframeAnchorsStep
    KeyframeEncoderStep: type = KeyframeEncoderStep
    ReferenceEncoderStep: type = ReferenceEncoderStep
    PrepareLayoutStep: type = PrepareLayoutStep
    RefPrepareLayoutStep: type = RefPrepareLayoutStep
    PrepareConditionLatentsStep: type = PrepareConditionLatentsStep
    PrepareLatentsStep: type = PrepareLatentsStep
    FL2VAPrepareLatentsStep: type = FL2VAPrepareLatentsStep
    Ref2VAPrepareLatentsStep: type = Ref2VAPrepareLatentsStep
    SetTimestepsStep: type = SetTimestepsStep
    AfterDenoiseStep: type = AfterDenoiseStep
    VideoDecodeStep: type = VideoDecodeStep
    AudioDecodeStep: type = AudioDecodeStep
    ImageReference: type = FakeReference
    VideoReference: type = FakeReference
    AudioReference: type = FakeReference


def _fake_layout_cache(video_conditions=0, audio_conditions=0):
    return {
        "height": 32,
        "width": 32,
        "num_frames": 5,
        "num_latent_frames": 1,
        "latent_height": 2,
        "latent_width": 2,
        "num_audio_latents": 1,
        "position_ids": torch.zeros(7 + video_conditions + audio_conditions, 3),
        "token_tags": torch.zeros(7 + video_conditions + audio_conditions, dtype=torch.long),
        "video_indices": torch.arange(5, 7 + video_conditions),
        "audio_indices": torch.arange(2, 5 + audio_conditions),
        "text_indices": torch.arange(2),
        "num_condition_video_rows": video_conditions,
        "num_condition_audio_rows": audio_conditions,
    }


def _fake_pipeline():
    return SimpleNamespace(calls=[], draws=[], decode_condition_counts=None)


def test_block_executor_propagates_shared_state_and_selects_outputs(monkeypatch):
    monkeypatch.setattr(blocks, "require_minimax_h3_support", lambda: FakeSymbols())
    pipeline = SimpleNamespace(calls=[])
    first = RecordingBlock("first", {"tensor": torch.tensor([1.0])})
    second = RecordingBlock("second", {"result": "ok"})

    selected = blocks.run_h3_blocks(
        pipeline, [first, second], {"input": 3}, requested_outputs=("tensor", "result")
    )

    assert pipeline.calls == ["first", "second"]
    assert selected["tensor"] is first.output["tensor"]
    assert selected["result"] == "ok"


def test_block_executor_reports_missing_output_and_preserves_exception(monkeypatch):
    monkeypatch.setattr(blocks, "require_minimax_h3_support", lambda: FakeSymbols())
    with pytest.raises(ValueError, match="workflow='t2va'.*field='missing'"):
        blocks.run_h3_blocks(
            SimpleNamespace(calls=[]),
            [RecordingBlock("first")],
            {},
            requested_outputs=("missing",),
            workflow="t2va",
        )
    cause = RuntimeError("upstream")
    with pytest.raises(RuntimeError, match="upstream") as raised:
        blocks.run_h3_blocks(
            SimpleNamespace(calls=[]),
            [RecordingBlock("first", error=cause)],
            {},
            requested_outputs=(),
        )
    assert raised.value is cause


@pytest.mark.parametrize(
    ("workflow", "expected_order", "required_fields"),
    [
        (
            "t2va",
            ["TextEncoderStep", "NoKeyframeAnchorsStep", "PrepareLayoutStep"],
            {"prompt_embeds", "video_indices", "audio_indices"},
        ),
        (
            "fl2va",
            [
                "ResizeStep",
                "FL2VATextEncoderStep",
                "KeyframeEncoderStep",
                "PrepareLayoutStep",
            ],
            {"prompt_embeds", "condition_latents", "video_indices"},
        ),
        (
            "ref2va",
            [
                "RefSetupStep",
                "Ref2VATextEncoderStep",
                "ReferenceEncoderStep",
                "RefPrepareLayoutStep",
            ],
            {
                "prompt_embeds",
                "condition_latents",
                "audio_condition_latents",
            },
        ),
    ],
)
def test_encoding_workflows_use_pinned_block_order_and_cache_contract(
    monkeypatch, workflow, expected_order, required_fields
):
    monkeypatch.setattr(blocks, "require_minimax_h3_support", lambda: FakeSymbols())
    pipeline = _fake_pipeline()
    cache = blocks.encode_h3_workflow_inputs(pipeline, {"prompt": "test"}, workflow=workflow)

    assert pipeline.calls == expected_order
    assert required_fields <= cache.keys()
    assert set(cache) == set(blocks.ENCODE_WORKFLOW_FIELDS[workflow])
    assert "generator" not in cache
    assert "text_encoder" not in cache
    assert not {
        "keyframes",
        "keyframe_anchors",
        "normalized_references",
        "text_token_tags",
    }.intersection(cache)


@pytest.mark.parametrize(
    ("workflow", "video_conditions", "audio_conditions", "expected_order"),
    [
        ("t2va", 0, 0, ["PrepareLatentsStep"]),
        (
            "fl2va",
            1,
            0,
            [
                "PrepareConditionLatentsStep",
                "PrepareLatentsStep",
                "FL2VAPrepareLatentsStep",
            ],
        ),
        (
            "ref2va",
            1,
            1,
            [
                "PrepareConditionLatentsStep",
                "PrepareLatentsStep",
                "Ref2VAPrepareLatentsStep",
            ],
        ),
    ],
)
def test_rollout_uses_condition_video_audio_rng_order_and_exact_split(
    monkeypatch, workflow, video_conditions, audio_conditions, expected_order
):
    monkeypatch.setattr(blocks, "require_minimax_h3_support", lambda: FakeSymbols())
    pipeline = _fake_pipeline()
    cache = _fake_layout_cache(video_conditions, audio_conditions)
    cache["condition_latents"] = [torch.ones(1, 24, 1, 2, 2)]
    cache["audio_condition_latents"] = [torch.full((1, 32), 4.0)] if audio_conditions else []
    generator = torch.Generator().manual_seed(7)

    targets, prefixes = blocks.prepare_h3_rollout_state(
        pipeline, cache, workflow=workflow, generator=generator
    )

    assert pipeline.calls == expected_order
    assert [name for name, _ in pipeline.draws] == (
        ["video", "audio"] if workflow == "t2va" else ["condition", "video", "audio"]
    )
    assert targets.components["video"].shape == (1, 2, 96)
    assert targets.components["audio"].shape == (1, 3, 32)
    assert prefixes["video"].shape[1] == video_conditions
    assert prefixes["audio"].shape[1] == audio_conditions


def test_condition_prefix_helper_accepts_collated_b1_tensor_containers(monkeypatch):
    monkeypatch.setattr(blocks, "require_minimax_h3_support", lambda: FakeSymbols())
    pipeline = _fake_pipeline()
    cache = _fake_layout_cache(video_conditions=1, audio_conditions=2)
    cache["video_indices"] = cache["video_indices"].unsqueeze(0)
    cache["audio_indices"] = cache["audio_indices"].unsqueeze(0)
    cache["num_condition_video_rows"] = torch.tensor([1])
    cache["num_condition_audio_rows"] = [2]
    cache["condition_latents"] = torch.ones(1, 1, 1, 24, 1, 2, 2)
    cache["audio_condition_latents"] = torch.stack(
        [torch.full((1, 32), 4.0), torch.full((1, 32), 5.0)]
    ).unsqueeze(0)

    prefixes = blocks.prepare_h3_condition_prefixes(
        pipeline,
        cache,
        workflow="ref2va",
        generator=torch.Generator().manual_seed(7),
    )

    assert pipeline.calls == ["PrepareConditionLatentsStep"]
    assert [name for name, _ in pipeline.draws] == ["condition"]
    assert prefixes["video"].shape == (1, 1, 96)
    assert prefixes["audio"].shape == (1, 2, 32)
    torch.testing.assert_close(prefixes["audio"][0, :, 0], torch.tensor([4.0, 5.0]))


def test_rollout_rejects_full_rows_mismatching_layout_before_split(monkeypatch):
    monkeypatch.setattr(blocks, "require_minimax_h3_support", lambda: FakeSymbols())
    cache = _fake_layout_cache(video_conditions=1)
    cache["video_indices"] = torch.arange(4)
    cache["condition_latents"] = [torch.ones(1, 24, 1, 2, 2)]
    cache["audio_condition_latents"] = []

    with pytest.raises(
        ValueError,
        match=(
            "workflow='fl2va'.*component='video'.*field='full_row_count'.*" "actual=3.*expected=4"
        ),
    ):
        blocks.prepare_h3_rollout_state(
            _fake_pipeline(),
            cache,
            workflow="fl2va",
            generator=torch.Generator().manual_seed(1),
        )


def test_rollout_rejects_zero_target_rows_with_exact_diagnostic(monkeypatch):
    class EmptyTargetPrepareLatentsStep(FakeWorkflowBlock):
        def __call__(self, pipeline, state):
            pipeline.calls.append(type(self).__name__)
            state.values["latents"] = torch.empty(0, 96)
            state.values["audio_latents"] = torch.ones(3, 32)
            return pipeline, state

    bundle = dataclasses.replace(FakeSymbols(), PrepareLatentsStep=EmptyTargetPrepareLatentsStep)
    monkeypatch.setattr(blocks, "require_minimax_h3_support", lambda: bundle)
    cache = _fake_layout_cache(video_conditions=2)
    cache["video_indices"] = torch.arange(2)
    cache["condition_latents"] = [torch.ones(1, 24, 1, 2, 2)]
    cache["audio_condition_latents"] = []

    with pytest.raises(
        ValueError,
        match=(
            "workflow='fl2va'.*component='video'.*field='target_row_count'.*" "actual=0.*expected=0"
        ),
    ):
        blocks.prepare_h3_rollout_state(
            _fake_pipeline(),
            cache,
            workflow="fl2va",
            generator=torch.Generator().manual_seed(1),
        )


def test_row_timestep_oracle_and_transition_schedules():
    from flow_factory.models.minimax_h3.layout import build_h3_schedule_plan, build_row_timesteps
    from flow_factory.scheduler import MiniMaxH3SDEScheduler

    video_scheduler = MiniMaxH3SDEScheduler(shift=12.0, dynamics_type="ODE")
    audio_scheduler = MiniMaxH3SDEScheduler(shift=3.0, dynamics_type="ODE")
    layout = {
        "video_indices": torch.tensor([2, 5, 6]),
        "audio_indices": torch.tensor([3, 4]),
        "text_indices": torch.tensor([0, 1]),
        "num_condition_video_rows": 1,
        "num_condition_audio_rows": 1,
    }
    plan = build_h3_schedule_plan(video_scheduler, audio_scheduler, 2, layout, torch.device("cpu"))

    assert [field.name for field in dataclasses.fields(plan)] == ["schedules"]
    assert len(plan.schedules["video"][1]) == 3
    assert plan.schedules["video"][1][-1].item() == 0
    unique, inverse = build_row_timesteps(layout, 0.2, 0.4, 0.999)
    oracle = torch.tensor([0.2, 0.2, 0.999, 1.0, 0.4, 0.2, 0.2])
    expected_unique, expected_inverse = torch.unique(oracle, sorted=True, return_inverse=True)
    torch.testing.assert_close(unique, expected_unique)
    torch.testing.assert_close(inverse, expected_inverse)


def test_layout_permutation_validation_runs_once_outside_step_loop(monkeypatch):
    from flow_factory.models.minimax_h3 import layout as layout_module
    from flow_factory.scheduler import MiniMaxH3SDEScheduler

    calls = []
    original_validate = layout_module._validate_layout

    def record_validate(values, *, validate_permutation=True):
        calls.append(validate_permutation)
        return original_validate(
            values,
            validate_permutation=validate_permutation,
        )

    monkeypatch.setattr(layout_module, "_validate_layout", record_validate)
    values = {
        "video_indices": torch.tensor([2, 5, 6]),
        "audio_indices": torch.tensor([3, 4]),
        "text_indices": torch.tensor([0, 1]),
        "num_condition_video_rows": 1,
        "num_condition_audio_rows": 1,
    }
    layout_module.build_h3_schedule_plan(
        MiniMaxH3SDEScheduler(shift=12.0, dynamics_type="ODE"),
        MiniMaxH3SDEScheduler(shift=3.0, dynamics_type="ODE"),
        2,
        values,
        torch.device("cpu"),
    )
    layout_module.build_row_timesteps(values, 0.2, 0.4, 0.999)

    assert calls == [True, False]


class FakeTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0))
        self.calls = []

    def forward(
        self,
        hidden_states,
        audio_hidden_states,
        encoder_hidden_states,
        timestep,
        timestep_indices,
        attention_kwargs=None,
        token_tags=None,
        position_ids=None,
        video_indices=None,
        audio_indices=None,
        text_indices=None,
    ):
        self.calls.append(
            {
                "hidden_states": hidden_states,
                "audio_hidden_states": audio_hidden_states,
                "token_tags": token_tags,
            }
        )
        return hidden_states * self.weight, audio_hidden_states * self.weight


def _state(video_rows=2, audio_rows=3):
    return LatentState(
        {
            "video": torch.ones(1, video_rows, 96),
            "audio": torch.ones(1, audio_rows, 32),
        }
    )


def test_transformer_uses_supplied_component_keeps_grad_and_slices_prefixes():
    from flow_factory.models.minimax_h3.denoise import run_h3_joint_transformer

    transformer = FakeTransformer()
    prefixes = {
        "video": torch.full((1, 1, 96), 3.0),
        "audio": torch.full((1, 2, 32), 4.0),
    }
    layout = {
        "video_indices": torch.tensor([2, 5, 6]),
        "audio_indices": torch.tensor([3, 4, 7, 8, 9]),
        "text_indices": torch.tensor([0, 1]),
        "num_condition_video_rows": 1,
        "num_condition_audio_rows": 2,
        "token_tags": torch.arange(10),
        "position_ids": torch.zeros(10, 3),
    }
    times = ComponentTimes(
        timestep={"video": torch.tensor([800.0]), "audio": torch.tensor([600.0])},
        next_timestep={"video": torch.tensor([0.0]), "audio": torch.tensor([0.0])},
        sigma={"video": torch.tensor([0.8]), "audio": torch.tensor([0.6])},
        next_sigma={"video": torch.tensor([0.0]), "audio": torch.tensor([0.0])},
    )
    velocity = run_h3_joint_transformer(
        transformer,
        _state(),
        prefixes,
        torch.ones(1, 2, 8),
        times,
        layout,
    )

    assert len(transformer.calls) == 1
    assert transformer.calls[0]["hidden_states"].shape == (1, 3, 96)
    assert velocity.components["video"].shape == (1, 2, 96)
    assert velocity.components["audio"].shape == (1, 3, 32)
    velocity.components["video"].sum().backward()
    assert transformer.weight.grad is not None


def test_transformer_dispatches_through_real_prepared_component_proxy():
    from flow_factory.models.minimax_h3.denoise import run_h3_joint_transformer
    from flow_factory.models.model_bundle import RoutedComponentProxy

    inner = FakeTransformer()

    class PreparedBundle(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []

        def forward(self, name, *args, **kwargs):
            self.calls.append(name)
            return inner(*args, **kwargs)

    bundle = PreparedBundle()
    proxy = RoutedComponentProxy(bundle, "transformer", inner)
    prefixes = {"video": torch.zeros(1, 1, 96), "audio": torch.zeros(1, 2, 32)}
    layout = {
        "video_indices": torch.tensor([2, 5, 6]),
        "audio_indices": torch.tensor([3, 4, 7, 8, 9]),
        "text_indices": torch.tensor([0, 1]),
        "num_condition_video_rows": 1,
        "num_condition_audio_rows": 2,
        "token_tags": torch.arange(10),
        "position_ids": torch.zeros(10, 3),
    }
    times = ComponentTimes(
        timestep={"video": torch.tensor([800.0]), "audio": torch.tensor([600.0])},
        next_timestep={"video": torch.tensor([0.0]), "audio": torch.tensor([0.0])},
        sigma={"video": torch.tensor([0.8]), "audio": torch.tensor([0.6])},
        next_sigma={"video": torch.tensor([0.0]), "audio": torch.tensor([0.0])},
    )

    run_h3_joint_transformer(proxy, _state(), prefixes, torch.ones(1, 2, 8), times, layout)

    assert bundle.calls == ["transformer"]
    assert len(inner.calls) == 1


def test_transformer_reads_layout_signature_through_peft_wrapper():
    from flow_factory.models.minimax_h3.denoise import run_h3_joint_transformer

    inner = FakeTransformer()

    class PeftWrapper(torch.nn.Module):
        def get_base_model(self):
            return inner

        def forward(self, *args, **kwargs):
            return inner(*args, **kwargs)

    layout = {
        "video_indices": torch.tensor([2, 5, 6]),
        "audio_indices": torch.tensor([3, 4, 7, 8, 9]),
        "text_indices": torch.tensor([0, 1]),
        "num_condition_video_rows": 1,
        "num_condition_audio_rows": 2,
        "token_tags": torch.arange(10),
        "position_ids": torch.zeros(10, 3),
    }
    times = ComponentTimes(
        timestep={"video": torch.tensor([800.0]), "audio": torch.tensor([600.0])},
        next_timestep={"video": torch.tensor([0.0]), "audio": torch.tensor([0.0])},
        sigma={"video": torch.tensor([0.8]), "audio": torch.tensor([0.6])},
        next_sigma={"video": torch.tensor([0.0]), "audio": torch.tensor([0.0])},
    )

    run_h3_joint_transformer(
        PeftWrapper(),
        _state(),
        {"video": torch.zeros(1, 1, 96), "audio": torch.zeros(1, 2, 32)},
        torch.ones(1, 2, 8),
        times,
        layout,
    )

    torch.testing.assert_close(inner.calls[0]["token_tags"], layout["token_tags"])


def test_transformer_signature_is_cached_by_real_module_type(monkeypatch):
    from flow_factory.models.minimax_h3 import denoise

    denoise._forward_parameter_names.cache_clear()
    signature_calls = []
    original_signature = denoise.inspect.signature

    def record_signature(value):
        signature_calls.append(value)
        return original_signature(value)

    monkeypatch.setattr(denoise.inspect, "signature", record_signature)
    transformer = FakeTransformer()
    layout = {
        "video_indices": torch.tensor([2, 5, 6]),
        "audio_indices": torch.tensor([3, 4, 7, 8, 9]),
        "text_indices": torch.tensor([0, 1]),
        "num_condition_video_rows": 1,
        "num_condition_audio_rows": 2,
        "token_tags": torch.arange(10),
        "position_ids": torch.zeros(10, 3),
    }
    times = ComponentTimes(
        timestep={"video": torch.tensor([800.0]), "audio": torch.tensor([600.0])},
        next_timestep={"video": torch.tensor([0.0]), "audio": torch.tensor([0.0])},
        sigma={"video": torch.tensor([0.8]), "audio": torch.tensor([0.6])},
        next_sigma={"video": torch.tensor([0.0]), "audio": torch.tensor([0.0])},
    )
    prefixes = {"video": torch.zeros(1, 1, 96), "audio": torch.zeros(1, 2, 32)}

    for _ in range(2):
        denoise.run_h3_joint_transformer(
            transformer,
            _state(),
            prefixes,
            torch.ones(1, 2, 8),
            times,
            layout,
        )

    assert signature_calls == [FakeTransformer.forward]


def test_transformer_rejects_batch_greater_than_one():
    from flow_factory.models.minimax_h3.denoise import run_h3_joint_transformer

    bad = LatentState(
        {
            "video": torch.ones(2, 2, 96),
            "audio": torch.ones(2, 3, 32),
        }
    )
    with pytest.raises(ValueError, match="workflow='t2va'.*B=1.*received B=2"):
        run_h3_joint_transformer(
            FakeTransformer(),
            bad,
            {"video": torch.empty(2, 0, 96), "audio": torch.empty(2, 0, 32)},
            torch.ones(2, 1, 8),
            None,
            {},
            workflow="t2va",
        )


class FakeScheduler:
    def __init__(self, name, calls):
        self.name = name
        self.calls = calls

    def step(self, velocity, timestep, state, **kwargs):
        self.calls.append(
            (self.name, kwargs["sigma"], kwargs["sigma_next"], kwargs["next_latents"])
        )
        return SDESchedulerOutput(
            next_latents=state - velocity,
            next_latents_mean=state - velocity,
            std_dev_t=torch.ones(1),
            dt=torch.ones(1),
            log_prob=torch.ones(1),
            velocity=velocity,
        )


def test_component_step_is_video_then_audio_and_target_only():
    from flow_factory.models.minimax_h3.denoise import step_h3_components

    calls = []
    times = ComponentTimes(
        timestep={"video": torch.tensor([800.0]), "audio": torch.tensor([600.0])},
        next_timestep={"video": torch.tensor([400.0]), "audio": torch.tensor([300.0])},
        sigma={"video": torch.tensor([0.8]), "audio": torch.tensor([0.6])},
        next_sigma={"video": torch.tensor([0.4]), "audio": torch.tensor([0.3])},
    )
    output = step_h3_components(
        _state(),
        _state(),
        times,
        FakeScheduler("video", calls),
        FakeScheduler("audio", calls),
        next_state=_state(),
        compute_log_prob=True,
    )
    assert [call[0] for call in calls] == list(MINIMAX_H3_COMPONENT_ORDER)
    assert output.next_state.components["video"].shape == (1, 2, 96)
    assert output.log_prob.shape == (1,)


def test_component_step_replay_preserves_rng_and_all_return_fields():
    from flow_factory.models.minimax_h3.denoise import step_h3_components

    class DrawingScheduler(FakeScheduler):
        def step(self, velocity, timestep, state, **kwargs):
            torch.rand((), generator=kwargs["generator"])
            return super().step(velocity, timestep, state, **kwargs)

    times = ComponentTimes(
        timestep={"video": torch.tensor([800.0]), "audio": torch.tensor([600.0])},
        next_timestep={"video": torch.tensor([400.0]), "audio": torch.tensor([300.0])},
        sigma={"video": torch.tensor([0.8]), "audio": torch.tensor([0.6])},
        next_sigma={"video": torch.tensor([0.4]), "audio": torch.tensor([0.3])},
    )
    rollout_generator = torch.Generator().manual_seed(11)
    replay_generator = torch.Generator().manual_seed(11)
    rollout = step_h3_components(
        _state(),
        _state(),
        times,
        DrawingScheduler("video", []),
        DrawingScheduler("audio", []),
        generator=rollout_generator,
        return_kwargs=("next_latents", "log_prob", "velocity"),
    )
    replay = step_h3_components(
        _state(),
        _state(),
        times,
        DrawingScheduler("video", []),
        DrawingScheduler("audio", []),
        next_state=rollout.next_state,
        generator=replay_generator,
        return_kwargs=("next_latents", "log_prob", "velocity"),
    )

    assert torch.equal(rollout_generator.get_state(), replay_generator.get_state())
    assert replay.next_state is not None
    assert replay.next_state_mean is not None
    assert replay.std_dev_t is not None
    assert replay.dt is not None
    assert replay.log_prob is not None
    assert replay.component_log_probs is not None
    assert replay.velocity is not None


def test_forward_velocity_only_skips_both_schedulers():
    from flow_factory.models.minimax_h3.denoise import forward_h3_state

    calls = []
    output = forward_h3_state(
        FakeTransformer(),
        _state(),
        {"video": torch.zeros(1, 0, 96), "audio": torch.zeros(1, 0, 32)},
        torch.ones(1, 2, 8),
        ComponentTimes(
            timestep={"video": torch.tensor([800.0]), "audio": torch.tensor([600.0])},
            next_timestep={"video": torch.tensor([0.0]), "audio": torch.tensor([0.0])},
            sigma={"video": torch.tensor([0.8]), "audio": torch.tensor([0.6])},
            next_sigma={"video": torch.tensor([0.0]), "audio": torch.tensor([0.0])},
        ),
        {
            "video_indices": torch.tensor([5, 6]),
            "audio_indices": torch.tensor([2, 3, 4]),
            "text_indices": torch.tensor([0, 1]),
            "num_condition_video_rows": 0,
            "num_condition_audio_rows": 0,
            "token_tags": torch.arange(7),
            "position_ids": torch.zeros(7, 3),
        },
        FakeScheduler("video", calls),
        FakeScheduler("audio", calls),
        velocity_only=True,
    )

    assert isinstance(output, LatentState)
    assert calls == []


@pytest.mark.parametrize("malformation", ["prefix", "layout", "times"])
def test_transformer_fails_fast_on_malformed_prefix_layout_or_times(malformation):
    from flow_factory.models.minimax_h3.denoise import run_h3_joint_transformer

    prefixes = {"video": torch.zeros(1, 0, 96), "audio": torch.zeros(1, 0, 32)}
    layout = {
        "video_indices": torch.tensor([5, 6]),
        "audio_indices": torch.tensor([2, 3, 4]),
        "text_indices": torch.tensor([0, 1]),
        "num_condition_video_rows": 0,
        "num_condition_audio_rows": 0,
        "token_tags": torch.arange(7),
        "position_ids": torch.zeros(7, 3),
    }
    times = ComponentTimes(
        timestep={"video": torch.tensor([800.0]), "audio": torch.tensor([600.0])},
        next_timestep={"video": torch.tensor([0.0]), "audio": torch.tensor([0.0])},
        sigma={"video": torch.tensor([0.8]), "audio": torch.tensor([0.6])},
        next_sigma={"video": torch.tensor([0.0]), "audio": torch.tensor([0.0])},
    )
    if malformation == "prefix":
        layout["num_condition_video_rows"] = 1
    elif malformation == "layout":
        layout.pop("audio_indices")
    else:
        times = None

    with pytest.raises((TypeError, ValueError), match="MiniMax H3"):
        run_h3_joint_transformer(
            FakeTransformer(), _state(), prefixes, torch.ones(1, 2, 8), times, layout
        )


def test_decode_uses_target_only_upstream_order_and_zero_condition_counts(monkeypatch):
    from flow_factory.models.minimax_h3 import blocks, decoding

    monkeypatch.setattr(decoding, "require_minimax_h3_support", lambda: FakeSymbols())
    monkeypatch.setattr(blocks, "require_minimax_h3_support", lambda: FakeSymbols())
    pipeline = _fake_pipeline()

    videos, audio, sampling_rate = decoding.decode_h3_targets(
        pipeline,
        _state(),
        {
            "num_latent_frames": 1,
            "latent_height": 2,
            "latent_width": 4,
            "num_audio_latents": 1,
        },
        output_type="pt",
    )

    assert pipeline.calls == ["AfterDenoiseStep", "VideoDecodeStep", "AudioDecodeStep"]
    assert pipeline.decode_condition_counts == (0, 0)
    assert videos == ["video"]
    assert audio.shape == (1, 2, 16)
    assert sampling_rate == 32000


def test_decode_uses_vae_device_when_preprocessing_module_is_offloaded():
    from flow_factory.models.minimax_h3.decoding import _decoder_execution_device

    class Pipeline:
        def __init__(self):
            self.text_encoder = torch.nn.Linear(1, 1, device="meta")
            self.vae = torch.nn.Linear(1, 1)
            self.audio_vae = torch.nn.Linear(1, 1)

        @property
        def components(self):
            return {
                name: value
                for name in ("text_encoder", "vae", "audio_vae")
                if (value := getattr(self, name)) is not None
            }

        @property
        def _execution_device(self):
            return next(iter(self.components.values())).weight.device

    pipeline = Pipeline()
    text_encoder = pipeline.text_encoder

    with _decoder_execution_device(pipeline):
        assert pipeline.text_encoder is None
        assert pipeline._execution_device == torch.device("cpu")

    assert pipeline.text_encoder is text_encoder


def test_decode_preserves_upstream_output_type_validation(monkeypatch):
    from flow_factory.models.minimax_h3 import blocks, decoding

    monkeypatch.setattr(decoding, "require_minimax_h3_support", lambda: FakeSymbols())
    monkeypatch.setattr(blocks, "require_minimax_h3_support", lambda: FakeSymbols())

    with pytest.raises(ValueError, match="output_type"):
        decoding.decode_h3_targets(
            _fake_pipeline(),
            _state(),
            {
                "num_latent_frames": 1,
                "latent_height": 2,
                "latent_width": 4,
                "num_audio_latents": 1,
            },
            output_type="latent",
        )


def test_dependency_probe_error_names_minimum_release(monkeypatch):
    from flow_factory.models.minimax_h3 import dependency

    monkeypatch.setattr(dependency, "_SYMBOLS", None)
    import_error = ImportError("missing")
    monkeypatch.setattr(dependency, "_IMPORT_ERROR", import_error)
    with pytest.raises(ImportError, match=r"diffusers>=0\.40\.0") as raised:
        dependency.require_minimax_h3_support()
    assert raised.value.__cause__ is import_error


def test_dependency_probe_validates_complete_pinned_compatible_bundle(monkeypatch):
    from flow_factory.models.minimax_h3 import dependency

    bundle = FakeSymbols()
    monkeypatch.setattr(dependency, "_SYMBOLS", bundle)
    monkeypatch.setattr(dependency, "_IMPORT_ERROR", None)
    assert dependency.require_minimax_h3_support() is bundle


@pytest.mark.parametrize(
    ("workflow", "triggers"),
    [
        ("t2va", {}),
        ("fl2va", ({"prompt": True, "image": True},)),
        ("ref2va", {"prompt": True}),
    ],
)
def test_dependency_probe_rejects_incompatible_workflow_triggers(monkeypatch, workflow, triggers):
    from flow_factory.models.minimax_h3 import dependency

    workflow_map = dict(FakeMiniMaxH3Blocks._workflow_map)
    workflow_map[workflow] = triggers
    blocks_class = type("IncompatibleMiniMaxH3Blocks", (), {"_workflow_map": workflow_map})
    bundle = dataclasses.replace(FakeSymbols(), MiniMaxH3Blocks=blocks_class)
    monkeypatch.setattr(dependency, "_SYMBOLS", bundle)
    monkeypatch.setattr(dependency, "_IMPORT_ERROR", None)

    with pytest.raises(ImportError, match=rf"MiniMaxH3Blocks.*{workflow}") as raised:
        dependency.require_minimax_h3_support()
    assert dependency.MINIMAX_H3_DIFFUSERS_MIN_VERSION in str(raised.value)
    assert dependency.MINIMAX_H3_INSTALL in str(raised.value)


def test_dependency_probe_rejects_block_without_pipeline_state_call_contract(monkeypatch):
    from flow_factory.models.minimax_h3 import dependency

    class NoArgumentBlock:
        def __call__(self):
            return None

    bundle = dataclasses.replace(FakeSymbols(), PrepareLatentsStep=NoArgumentBlock)
    monkeypatch.setattr(dependency, "_SYMBOLS", bundle)
    monkeypatch.setattr(dependency, "_IMPORT_ERROR", None)

    with pytest.raises(ImportError, match="PrepareLatentsStep.*pipeline, state"):
        dependency.require_minimax_h3_support()


@pytest.mark.parametrize(
    "broken_field",
    [
        "PipelineState",
        "MiniMaxH3Blocks",
        "MiniMaxH3AttnProcessor",
        "PrepareLatentsStep",
        "Ref2VATextEncoderStep",
        "SetTimestepsStep",
    ],
)
def test_dependency_probe_rejects_incompatible_api_with_actionable_requirement(
    monkeypatch, broken_field
):
    from flow_factory.models.minimax_h3 import dependency

    bundle = dataclasses.replace(FakeSymbols(), **{broken_field: object})
    monkeypatch.setattr(dependency, "_SYMBOLS", bundle)
    monkeypatch.setattr(dependency, "_IMPORT_ERROR", None)

    with pytest.raises(ImportError) as raised:
        dependency.require_minimax_h3_support()

    message = str(raised.value)
    assert "diffusers>=0.40.0" in message
    assert "pip install 'diffusers>=0.40.0'" in message
    assert broken_field in message


def test_dependency_probe_rejects_non_callable_attention_dispatch(monkeypatch):
    from flow_factory.models.minimax_h3 import dependency

    bundle = dataclasses.replace(FakeSymbols(), dispatch_attention_fn=object())
    monkeypatch.setattr(dependency, "_SYMBOLS", bundle)
    monkeypatch.setattr(dependency, "_IMPORT_ERROR", None)

    with pytest.raises(ImportError, match="dispatch_attention_fn must be callable"):
        dependency.require_minimax_h3_support()


def test_pyproject_requires_released_h3_diffusers():
    import re

    from packaging.version import Version

    from flow_factory.models.minimax_h3.dependency import MINIMAX_H3_DIFFUSERS_MIN_VERSION

    text = Path("pyproject.toml").read_text()
    floors = re.findall(r'"diffusers>=([0-9][0-9.]*)"', text)
    assert len(floors) == 1, floors
    # The project floor may rise above H3's own minimum, never below it.
    assert Version(floors[0]) >= Version(MINIMAX_H3_DIFFUSERS_MIN_VERSION), floors
    assert "git+https://github.com/huggingface/diffusers.git@" not in text
