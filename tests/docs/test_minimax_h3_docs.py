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
EXAMPLE_LINKS = (
    "examples/grpo/lora/minimax_h3_t2va/debug.yaml",
    "examples/grpo/lora/minimax_h3_fl2va/default.yaml",
    "examples/grpo/lora/minimax_h3_ref2va/default.yaml",
)


def _text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_readme_documents_h3_links_dependency_and_limits() -> None:
    text = _text("README.md")
    assert (
        "The configurations under `examples/` have been verified to yield measurable "
        "performance gains."
    ) not in text
    assert "all 120 model/backend/algorithm cells" in text
    assert "smoke completion is not a claim of" in text
    assert "all 36 real-weight smoke cells" in text
    assert "FL2VA first-plus-last" in text
    assert "T2VA is real-weight validated on 1 and 16 GPUs" not in text
    assert text.count("<td>33B</td>") == 3

    for model_type, link in zip(
        ("minimax-h3-t2va", "minimax-h3-fl2va", "minimax-h3-ref2va"),
        EXAMPLE_LINKS,
    ):
        assert model_type in text
        assert link in text
        assert (ROOT / link).is_file()

    for required in (
        "diffusers>=0.40.0",
        "pip install -e .",
        "PyAV >=17.0.0",
        "B=1",
        "no CFG",
        "shift 12",
        "shift 3",
        "data-ward velocity",
        "N transitions",
        "N + 1 states",
        "33B",
        "do not claim a completed long-run reward trend",
        "[Datasets](guidance/datasets.md)",
    ):
        assert required in text


def test_examples_readme_links_h3_and_separates_validation_levels() -> None:
    text = _text("examples/README.md")
    for root_link in EXAMPLE_LINKS:
        relative_link = f"../{root_link}"
        assert relative_link in text
        assert (ROOT / root_link).is_file()
    assert "all 36 H3 main cells" in text
    assert "GPU validation matrix" in text
    assert "execution coverage" in text
    assert "reward" in text
    assert "61 GB" in text
    assert "ImageBind" in text
    assert "facebookresearch/ImageBind.git" in text
    assert "NonCommercial" in text


def test_h3_first_block_cache_docs_preserve_support_and_validation_boundaries() -> None:
    acceleration = _text("guidance/acceleration.md")
    knowledge = _text(".agents/knowledge/topics/minimax_h3.md")
    example = _text("examples/tdm/lora/minimax_h3_t2va/default.yaml")

    for required in (
        "supported_diffusers_cache_policies",
        "prepare_diffusers_cache(policy, component_name, transformer)",
        "MiniMax H3 T2VA/FL2VA/Ref2VA",
        "`first_block` only",
        "H3 FirstBlockCache cannot be",
        "forced-grad rollout path",
        "threshold is workload-dependent",
    ):
        assert required in acceleration
    for required in (
        "actual unwrapped main-block class",
        "transformer_ref",
        "CPU cache-hit/reset test",
        "remain workload-dependent",
    ):
        assert required in knowledge
    assert "# acceleration:" in example
    assert "#     - name: diffusers_cache" in example
    assert "#       params: { policy: first_block, threshold: 0.05 }" in example


def test_gpu_validation_matrix_declares_scope_and_completed_result() -> None:
    text = _text("guidance/gpu_validation.md")

    assert "10 x 3 x 4 = 120 jobs" in text
    for result in ("144/144", "120/120", "22/22", "2/2", "132/132"):
        assert result in text
    assert "Four positive Muon jobs passed" in text
    for mode in (
        "sd35-t2i",
        "bagel-mri2i",
        "wan-t2v",
        "wan-i2v-first",
        "wan-flf2v",
        "ltx2-t2av",
        "ltx2-i2av",
        "h3-t2va",
        "h3-fl2va",
        "h3-ref2va",
    ):
        assert f"`{mode}`" in text
    for backend in ("ddp", "zero2", "fsdp2"):
        assert f"`{backend}`" in text
    for algorithm in ("grpo", "sft", "offline-dpo", "tdm"):
        assert f"`{algorithm}`" in text
    assert "exactly two rank-local dataloader batches" in text
    assert "two training epochs" in text
    assert "eval.eval_freq: 0" in text
    assert "DistributedSampler" in text
    assert "Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers" in text
    assert "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers` | ordered first and last" not in text
    assert "| Wan first/last | `Wan2.2-I2V-A14B-Diffusers` |" in text


def test_install_docs_distinguish_released_and_pinned_diffusers_runtimes() -> None:
    readme = _text("README.md")
    dockerfile = _text("docker/docker-cuda/Dockerfile")
    docker_readme = _text("docker/README.md")
    qwen_news = readme[readme.index("* **[2026-09-20]**") : readme.index("* **[2026-08-21]**")]
    h3_news = readme[readme.index("* **[2026-08-21]**") : readme.index("* **[2026-04-25]**")]

    assert "diffusers>=0.40.0" in readme
    assert "pip install -e ./diffusers" in qwen_news
    assert "pip install -e ./diffusers" not in h3_news
    assert "pip install -e ./diffusers" not in dockerfile
    assert "submodule (required)" not in docker_readme


def test_new_model_guide_documents_component_runtime_boundaries() -> None:
    text = _text("guidance/new_model.md")
    for required in (
        "ClassicPipelineRuntime",
        "ModularPipelineRuntime",
        "PseudoPipelineRuntime",
        "build_component_runtime()",
        "canonical lookup",
        "prepared/replacement override",
        "declared specs",
        "materialized modules",
        "materialize_components(None)",
        "adapter.pipeline",
        "adapter.scheduler",
        "scheduler_group",
        "trajectory_component_order",
        "ModelBundle",
        "RoutedComponentProxy",
    ):
        assert required in text


def test_workflow_guide_documents_structured_trajectory_contract_and_algorithms() -> None:
    text = _text("guidance/workflow.md")
    for required in (
        "StructuredTrajectory",
        "trajectory is None",
        "structured trajectories only",
        "`-1` means",
        "represented by `None`",
        "T + 1",
        "log-probability and callback maps have length T",
        '("video", "audio")',
        "GRPO",
        "GRPO-Guard",
        "DPPO",
        "DiffusionNFT",
        "AWM",
        "DPO",
        "DGPO",
        "CRD",
        "DiffusionOPD",
        "neutral guidance `1.0`",
        "framework-interface compatibility",
        "numerical parity",
    ):
        assert required in text
