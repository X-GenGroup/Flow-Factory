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

"""Tests for one-shot MiniMax H3 offline condition realization."""

from types import SimpleNamespace

import pytest
import torch

from flow_factory.models.minimax_h3._condition import MiniMaxH3ConditionStatePreparer
from flow_factory.models.minimax_h3.adapters import (
    MiniMaxH3FL2VAAdapter,
    MiniMaxH3Ref2VAAdapter,
    MiniMaxH3T2VAAdapter,
)


def _cached_condition() -> dict[str, object]:
    return {
        "prompt_embeds": torch.zeros(1, 2, 4),
        "position_ids": torch.zeros(1, 9, 3, dtype=torch.float64),
        "token_tags": torch.zeros(1, 9, dtype=torch.long),
        "text_indices": torch.tensor([[0, 1]], dtype=torch.long),
        "audio_indices": torch.tensor([[2, 3, 4]], dtype=torch.long),
        "video_indices": torch.tensor([[5, 6, 7, 8]], dtype=torch.long),
        "num_condition_video_rows": torch.tensor([2]),
        "num_condition_audio_rows": torch.tensor([1]),
        "condition_latents": [[torch.ones(1, 24, 1, 2, 2)]],
        "audio_condition_latents": [[torch.ones(1, 32)]],
        "height": torch.tensor([32]),
        "width": torch.tensor([32]),
        "num_frames": torch.tensor([22]),
        "reference_manifest": "ordered-manifest",
    }


def test_conditioned_adapters_declare_preparer_but_t2va_keeps_identity() -> None:
    t2va = object.__new__(MiniMaxH3T2VAAdapter)
    fl2va = object.__new__(MiniMaxH3FL2VAAdapter)
    ref2va = object.__new__(MiniMaxH3Ref2VAAdapter)

    assert t2va.build_condition_state_preparer() is None
    assert isinstance(fl2va.build_condition_state_preparer(), MiniMaxH3ConditionStatePreparer)
    assert isinstance(ref2va.build_condition_state_preparer(), MiniMaxH3ConditionStatePreparer)
    assert MiniMaxH3ConditionStatePreparer.required_components == ("scheduler",)
    assert MiniMaxH3T2VAAdapter.preprocess_cache_version == "minimax-h3-t2va-eos-v4"
    assert {
        MiniMaxH3FL2VAAdapter.preprocess_cache_version,
        MiniMaxH3Ref2VAAdapter.preprocess_cache_version,
    } == {"minimax-h3-v3"}


def test_ref2va_preparer_realizes_prefix_once_and_routes_owned_contexts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = SimpleNamespace(workflow="ref2va", pipeline=object())
    preparer = MiniMaxH3ConditionStatePreparer(adapter)
    condition = _cached_condition()
    prefixes = {
        "video": torch.randn(1, 2, 96),
        "audio": torch.randn(1, 1, 32),
    }
    calls: list[object] = []

    def prepare(pipeline, cached, *, workflow, generator):
        calls.append((pipeline, cached, workflow, generator))
        return prefixes

    monkeypatch.setattr(
        "flow_factory.models.minimax_h3._condition.prepare_h3_condition_prefixes",
        prepare,
    )
    generator = torch.Generator().manual_seed(11)

    prepared = preparer.prepare_condition_state(condition, generator)

    assert calls == [(adapter.pipeline, condition, "ref2va", generator)]
    assert "condition_latents" not in prepared.condition
    assert "audio_condition_latents" not in prepared.condition
    assert "position_ids" not in prepared.condition
    assert "num_condition_video_rows" not in prepared.condition
    assert prepared.condition["reference_manifest"] == "ordered-manifest"
    forward = prepared.model_forward_condition()
    codec = prepared.output_codec_condition()
    assert forward["condition_prefixes"] is prefixes
    assert forward["layout"]["video_indices"].shape == (4,)
    assert forward["layout"]["num_condition_video_rows"] == 2
    assert codec["layout"] is forward["layout"]
    assert "condition_prefixes" not in codec
    assert prepared.model_forward_condition()["condition_prefixes"]["video"] is prefixes["video"]


def test_preparer_rejects_wrong_workflow_and_stale_runtime_prefix() -> None:
    with pytest.raises(ValueError, match="requires workflow"):
        MiniMaxH3ConditionStatePreparer(
            SimpleNamespace(workflow="t2va", pipeline=object())
        ).prepare_condition_state(_cached_condition())

    condition = _cached_condition()
    condition["condition_prefixes"] = {"video": torch.empty(1, 0, 96)}
    with pytest.raises(ValueError, match="must not contain already-realized"):
        MiniMaxH3ConditionStatePreparer(
            SimpleNamespace(workflow="fl2va", pipeline=object())
        ).prepare_condition_state(condition)
