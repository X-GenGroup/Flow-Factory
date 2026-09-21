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

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_main_readme_lists_sensenova_as_t2i_and_multi_reference_i2i() -> None:
    readme = _text("README.md")
    t2i_start = readme.index('<tr><td rowspan="6">Text-to-Image</td>')
    combined_start = readme.index('<tr><td rowspan="9">Text-to-Image & Image(s)-to-Image</td>')
    video_start = readme.index('<tr><td rowspan="4">Text-to-Video</td>')

    t2i_only_rows = readme[t2i_start:combined_start]
    combined_rows = readme[combined_start:video_start]
    assert "SenseNova" not in t2i_only_rows
    assert "SenseNova-U1 1.0" in combined_rows
    assert "SenseNova-U1 1.5" in combined_rows
    assert combined_rows.count("<td>sensenova</td>") == 2
    assert combined_rows.count("<td>16B</td>") == 2
    assert "BAGEL-7B-MoT" in combined_rows

    dataset_docs = readme[readme.index("## Image-to-Image & Image-to-Video") :]
    assert "SenseNova-U1" in dataset_docs
    assert "ordered list of image paths" in dataset_docs


def test_examples_readme_distinguishes_sensenova_t2i_and_i2i_recipes() -> None:
    examples = _text("examples/README.md")
    required = (
        "examples/grpo/lora/sensenova/default.yaml",
        "examples/grpo/lora/sensenova/multi_reference_image.yaml",
    )
    for path in required:
        assert f"../{path}" in examples
        assert (ROOT / path).is_file()
    assert "T2I + OCR GRPO" in examples
    assert "ordered multi-reference I2I + PickScore GRPO" in examples
    assert "python dataset/multi_ref_image/prepare.py" in examples


def test_dataset_guide_documents_sensenova_ordered_references() -> None:
    datasets = _text("guidance/datasets.md")
    assert "## SenseNova-U1 datasets" in datasets
    assert '"images":["first.png","second.png"]' in datasets
    assert "python dataset/multi_ref_image/prepare.py" in datasets
    assert "NaViT-pack independent samples like Bagel" in datasets


def test_internal_docs_distinguish_sensenova_from_bagel_packing() -> None:
    architecture = _text(".agents/knowledge/architecture.md")
    conventions = _text(".agents/knowledge/topics/adapter_conventions.md")

    assert "ordered variable-count references" in architecture
    assert "rather than Bagel-style NaViT packing" in architecture
    assert "SenseNova ragged I2I is per-sample, not NaViT-packed" in conventions
    assert "SenseNovaI2ISample" in conventions
    assert "all 14 adapters" in conventions
    assert "SD3.5, Z-Image, SenseNova" in conventions


def test_project_guides_include_sensenova_components_and_examples() -> None:
    agents = _text("AGENTS.md")
    new_model = _text("guidance/new_model.md")
    examples = _text("examples/README.md")
    changelog = _text("CHANGELOG.md")

    assert "SenseNova-U1 (1.0/1.5; T2I + ordered multi-reference I2I)" in agents
    assert "src/flow_factory/models/sensenova/" in new_model
    assert "no standalone Flow-Factory VAE or text encoder" in new_model
    assert "BaseAdapter` resolves model components through `ComponentRuntime" in new_model
    for asset in (
        "sensenova-u15-ocr-train-reward-ocr-mean.png",
        "sensenova-u15-ocr-eval-reward-ocr-mean.png",
        "sensenova-u15-ocr-train-ratio-mean.png",
    ):
        assert asset in examples
        assert (ROOT / "docs/assets" / asset).is_file()
    assert "PR #217" in changelog
