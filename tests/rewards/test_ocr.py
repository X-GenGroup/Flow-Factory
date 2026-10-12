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

import json
from pathlib import Path

import numpy as np
import pytest

from flow_factory.rewards.ocr import OCRRewardModel


class _FakeOCR:
    def __init__(self, recognized_texts: list[str]) -> None:
        self.recognized_texts = recognized_texts

    def predict(self, image: np.ndarray) -> list[dict[str, list[str]]]:
        assert isinstance(image, np.ndarray)
        return [{"rec_texts": self.recognized_texts}]


def _reward(recognized_texts: list[str]) -> OCRRewardModel:
    reward = object.__new__(OCRRewardModel)
    reward.model = _FakeOCR(recognized_texts)
    return reward


def test_ocr_scores_every_visible_text_target() -> None:
    reward = _reward(["OPEN", "EX1T"])
    metadata = json.dumps({"visible_texts": json.dumps(["OPEN", "EXIT"])})
    scores = reward._compute_scores_batch(
        prompt=['signs reading "OPEN" and "EXIT"'],
        image=[np.zeros((8, 8, 3), dtype=np.uint8)],
        metadata=[metadata],
    )
    assert scores == pytest.approx([0.875])


def test_ocr_video_scores_without_metadata_across_batches() -> None:
    reward = _reward(["OPEN"])
    frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(3)]
    scores = reward._compute_video_scores(
        prompt=['a sign reading "OPEN"'],
        video=[frames],
        metadata=None,
        batch_size=2,
    )
    assert scores.tolist() == pytest.approx([1.0])


def test_ocr_accepts_native_visible_texts_list() -> None:
    metadata = json.dumps({"visible_texts": ["Federal Bank", "Policy 2025"]})
    assert OCRRewardModel._targets_from_metadata(metadata, sample_index=3) == [
        "Federal Bank",
        "Policy 2025",
    ]


def test_ocr_extracts_targets_from_plain_text_dataset_prompt() -> None:
    reward = _reward(["OPEN", "EX1T"])
    scores = reward._compute_scores_batch(
        prompt=['signs reading "OPEN" and "EXIT"'],
        image=[np.zeros((8, 8, 3), dtype=np.uint8)],
    )
    assert scores == pytest.approx([0.875])


def test_ocr_falls_back_to_prompt_when_metadata_has_no_targets() -> None:
    assert OCRRewardModel._targets_for_sample(
        prompt='a card reading “OPEN”',
        metadata=json.dumps({"__source__": "ocr"}),
        sample_index=2,
    ) == ["OPEN"]


def test_ocr_rejects_prompt_without_quoted_target() -> None:
    with pytest.raises(ValueError, match="quoted OCR target.*sample 4"):
        OCRRewardModel._targets_from_prompt("a blank sign", sample_index=4)


def test_ocr_rejects_invalid_json_metadata_with_sample_context() -> None:
    with pytest.raises(ValueError, match="JSON object metadata.*sample 5.*invalid JSON"):
        OCRRewardModel._parse_metadata("{", sample_index=5)


@pytest.mark.parametrize("split", ("train", "test"))
def test_bundled_ocr_prompts_have_extractable_targets(split: str) -> None:
    path = Path(__file__).parents[2] / "dataset" / "ocr" / f"{split}.txt"
    prompts = path.read_text(encoding="utf-8").splitlines()
    assert prompts
    for sample_index, prompt in enumerate(prompts):
        assert OCRRewardModel._targets_from_prompt(prompt, sample_index)


@pytest.mark.parametrize(
    ("metadata", "match"),
    [
        (json.dumps({}), "visible_texts"),
        (json.dumps({"visible_texts": []}), "at least one"),
        (json.dumps({"visible_texts": [""]}), "nonempty"),
        (json.dumps({"visible_texts": [1]}), "list\\[str\\]"),
    ],
)
def test_ocr_rejects_invalid_visible_texts(metadata: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        OCRRewardModel._targets_from_metadata(metadata, sample_index=7)


def test_ocr_rejects_mismatched_batch_lengths() -> None:
    reward = _reward(["OPEN"])
    with pytest.raises(ValueError, match="prompt=1, image=0, metadata=1"):
        reward._compute_scores_batch(
            prompt=["prompt"],
            image=[],
            metadata=[json.dumps({"visible_texts": ["OPEN"]})],
        )
