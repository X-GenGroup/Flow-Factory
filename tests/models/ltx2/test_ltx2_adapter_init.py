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

"""Constructor-path tests for the LTX2 adapters.

Only the heavyweight boundaries are patched (pipeline loading, target-module
parsing, freezing and precision casting). ``BaseAdapter.__init__`` itself runs
for real, so it is the constructor that installs the canonical scheduler and
calls ``build_scheduler_group``.
"""

from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import pytest
import torch

from diffusers import FlowMatchEulerDiscreteScheduler
from flow_factory.hparams import SchedulerArguments
from flow_factory.models import abc as adapter_abc
from flow_factory.models.abc import BaseAdapter
from flow_factory.models.ltx2.ltx2_i2av import LTX2_I2AV_Adapter
from flow_factory.models.ltx2.ltx2_t2av import LTX2_T2AV_Adapter
from flow_factory.scheduler import FlowMatchEulerDiscreteSDEScheduler, SchedulerGroup

ADAPTER_CLASSES = [LTX2_T2AV_Adapter, LTX2_I2AV_Adapter]


class TransformerStub(torch.nn.Module):
    """Minimal module so the pipeline exposes a transformer component."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)


class PipelineStub:
    """Pipeline stand-in exposing only what ``BaseAdapter.__init__`` reads."""

    def __init__(self) -> None:
        self.scheduler = FlowMatchEulerDiscreteScheduler(shift=3.0)
        self.transformer = TransformerStub()
        self.vae = torch.nn.Identity()
        self.audio_vae = torch.nn.Identity()

    @property
    def components(self) -> Dict[str, Any]:
        """Return the eager component declaration the classic runtime reads."""
        return {
            "scheduler": self.scheduler,
            "transformer": self.transformer,
            "vae": self.vae,
            "audio_vae": self.audio_vae,
        }


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        model_args=SimpleNamespace(
            model_name_or_path="stub",
            resume_path=None,
            resume_type="lora",
            finetune_type="full",
            target_modules="default",
            target_components=["transformer"],
            trainable_parameters_dtype="fp32",
            frozen_parameters_dtype=None,
        ),
        training_args=SimpleNamespace(enable_gradient_checkpointing=False),
        eval_args=SimpleNamespace(),
        scheduler_args=SchedulerArguments(noise_level=0.3, seed=1234),
        mixed_precision="no",
    )


@pytest.fixture()
def scheduler_calls(monkeypatch: pytest.MonkeyPatch) -> List[Dict[str, Any]]:
    """Record every real scheduler construction performed during init."""
    calls: List[Dict[str, Any]] = []
    real_loader = adapter_abc._load_scheduler

    def spy(*, pipeline_scheduler: Any, scheduler_args: Any) -> Any:
        scheduler = real_loader(
            pipeline_scheduler=pipeline_scheduler, scheduler_args=scheduler_args
        )
        calls.append(
            {
                "source": pipeline_scheduler,
                "source_type": type(pipeline_scheduler).__name__,
                "scheduler_args": scheduler_args,
                "created": scheduler,
            }
        )
        return scheduler

    monkeypatch.setattr(adapter_abc, "_load_scheduler", spy)
    return calls


@pytest.fixture(autouse=True)
def patched_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch only the heavyweight loading/runtime boundaries."""
    monkeypatch.setattr(LTX2_T2AV_Adapter, "load_pipeline", lambda self: PipelineStub())
    monkeypatch.setattr(LTX2_I2AV_Adapter, "load_pipeline", lambda self: PipelineStub())
    monkeypatch.setattr(BaseAdapter, "_init_target_module_map", lambda self: {"transformer": []})
    monkeypatch.setattr(BaseAdapter, "_freeze_components", lambda self: None)
    monkeypatch.setattr(BaseAdapter, "_mix_precision", lambda self: None)


def _build(cls: type) -> Any:
    return cls(_config(), SimpleNamespace(device=torch.device("cpu"), mixed_precision="no"))


@pytest.mark.parametrize("cls", ADAPTER_CLASSES)
def test_constructor_builds_the_scheduler_group_with_the_video_primary(cls: type) -> None:
    adapter = _build(cls)

    assert isinstance(adapter.scheduler_group, SchedulerGroup)
    assert adapter.scheduler_group.names == ("video", "audio")
    assert adapter.scheduler_group.primary_name == "video"
    assert adapter.scheduler_group.primary is adapter.pipeline.scheduler
    assert adapter.scheduler_group["video"] is adapter.scheduler
    assert adapter.scheduler_group["audio"] is adapter.audio_scheduler


@pytest.mark.parametrize("cls", ADAPTER_CLASSES)
def test_constructor_creates_twin_schedulers_in_video_then_audio_order(
    cls: type, scheduler_calls: List[Dict[str, Any]]
) -> None:
    adapter = _build(cls)

    assert len(scheduler_calls) == 2
    video_call, audio_call = scheduler_calls
    assert video_call["source"] is not None
    assert video_call["source_type"] == "FlowMatchEulerDiscreteScheduler"
    # The audio twin is rebuilt from the already installed video scheduler, which is
    # what keeps both schedules numerically identical.
    assert audio_call["source"] is video_call["created"]
    assert audio_call["scheduler_args"] is video_call["scheduler_args"]
    assert video_call["created"] is adapter.scheduler
    assert audio_call["created"] is adapter.audio_scheduler


@pytest.mark.parametrize("cls", ADAPTER_CLASSES)
def test_constructor_twins_share_the_configuration_but_not_the_instance(cls: type) -> None:
    adapter = _build(cls)

    assert isinstance(adapter.scheduler, FlowMatchEulerDiscreteSDEScheduler)
    assert isinstance(adapter.audio_scheduler, FlowMatchEulerDiscreteSDEScheduler)
    assert adapter.audio_scheduler is not adapter.scheduler
    assert dict(adapter.audio_scheduler.config) == dict(adapter.scheduler.config)
    assert adapter.scheduler.config.shift == 3.0
    for attribute in ("noise_level", "seed", "dynamics_type"):
        assert getattr(adapter.audio_scheduler, attribute) == getattr(adapter.scheduler, attribute)
    assert adapter.scheduler.noise_level == 0.3


@pytest.mark.parametrize("cls", ADAPTER_CLASSES)
def test_constructor_does_not_rebuild_the_audio_twin_after_super_init(
    cls: type, scheduler_calls: List[Dict[str, Any]]
) -> None:
    adapter = _build(cls)

    # Exactly two constructions total proves no post-super duplicate audio scheduler.
    assert len(scheduler_calls) == 2
    assert not hasattr(adapter, "_create_audio_scheduler")


@pytest.mark.parametrize("cls", ADAPTER_CLASSES)
def test_scheduler_setter_refreshes_the_primary_and_keeps_the_audio_twin(cls: type) -> None:
    adapter = _build(cls)
    audio = adapter.audio_scheduler
    replacement = adapter.load_scheduler()

    adapter.scheduler = replacement

    assert adapter.scheduler is replacement
    assert adapter.pipeline.scheduler is replacement
    assert adapter.scheduler_group.primary is replacement
    assert adapter.scheduler_group.names == ("video", "audio")
    assert adapter.scheduler_group["audio"] is audio


@pytest.mark.parametrize("cls", ADAPTER_CLASSES)
def test_constructor_dispatches_lifecycle_calls_in_component_order(cls: type) -> None:
    adapter = _build(cls)
    seeded: List[Tuple[str, int]] = []
    for name, scheduler in zip(adapter.scheduler_group.names, adapter.scheduler_group.values()):
        scheduler.set_seed = lambda seed, name=name: seeded.append((name, seed))

    adapter.set_trajectory_seed(21)

    assert seeded == [("video", 21), ("audio", 21)]


def test_i2av_preprocess_forwards_cached_negative_prompt_to_text_encoder() -> None:
    adapter = object.__new__(LTX2_I2AV_Adapter)
    captured: Dict[str, Any] = {}

    def encode_prompt(**kwargs: Any) -> Dict[str, Any]:
        captured.update(kwargs)
        return {"connector_prompt_embeds": torch.zeros(1, 1, 1)}

    adapter.encode_prompt = encode_prompt

    adapter.preprocess_func(
        prompt=["describe"],
        negative_prompt=["avoid blur"],
        images=None,
        guidance_scale=4.0,
    )

    assert captured["negative_prompt"] == ["avoid blur"]


@pytest.mark.parametrize("adapter_class", ADAPTER_CLASSES)
def test_caption_dropout_rebuilds_both_ltx2_connector_streams(adapter_class):
    def encode_text(texts, _length, device, dtype):
        ids = torch.tensor([[1, len(text)] for text in texts], device=device)
        return ids, ids.unsqueeze(-1).to(dtype), ids.ne(0)

    encoder = SimpleNamespace(
        pipeline=SimpleNamespace(
            connectors=lambda embeddings, mask: (embeddings, embeddings + 1, mask)
        ),
        _encode_text=encode_text,
    )
    kwargs = dict(guidance_scale=4.0, dtype=torch.float32, device=torch.device("cpu"))
    positive = adapter_class.encode_prompt(encoder, ["the scene"], **kwargs)
    dropped = adapter_class.encode_prompt(
        encoder, [adapter_class.sft_condition_dropout_policy.empty_prompt], **kwargs
    )
    for field in (
        "prompt_ids",
        "connector_prompt_embeds",
        "connector_audio_prompt_embeds",
        "connector_attention_mask",
    ):
        torch.testing.assert_close(dropped[field], positive[f"negative_{field}"])
        assert not torch.equal(dropped[field], positive[field])
