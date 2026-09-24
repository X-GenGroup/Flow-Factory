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

"""Protect the mandatory framework-upgrade GPU campaign from silent drift."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from typing import Any

import pytest

from flow_factory.rewards.pick_score import PickScoreRewardModel

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts/validate_gpu_campaign.py"
_SPEC = importlib.util.spec_from_file_location("validate_gpu_campaign", _SCRIPT)
validate = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(validate)
_MANIFEST_PATH = _REPO_ROOT / "config/gpu_validation/framework_upgrade.yaml"


def _manifest() -> dict[str, Any]:
    return validate.load_mapping(_MANIFEST_PATH)


def _passing_results(manifest: dict[str, Any]) -> dict[str, Any]:
    jobs = validate.validate_manifest(manifest, repo_root=_REPO_ROOT)
    result_jobs = []
    for job in jobs:
        result_jobs.append(
            {
                "id": job["id"],
                "status": "passed",
                "observations": {
                    "all_ranks_completed": True,
                    "finite_metrics": True,
                    "parameter_update": True,
                    "world_size": job["world_size"],
                    "backend": {"id": job["backend"], **job["runtime_assertions"]},
                    "cycle": job["cycle"],
                    "run_contract": job["run_contract"],
                },
                "artifacts": {
                    "command": f"artifacts/{job['id']}/command.txt",
                    "resolved_config": f"artifacts/{job['id']}/resolved.yaml",
                    "environment": f"artifacts/{job['id']}/environment.json",
                    "logs": f"artifacts/{job['id']}/train.log",
                    "metrics": f"artifacts/{job['id']}/metrics.json",
                },
            }
        )
    return {
        "gate_id": manifest["gate"]["id"],
        "manifest_sha256": validate.manifest_sha256(_MANIFEST_PATH),
        "commit_sha": "a" * 40,
        "jobs": result_jobs,
    }


def test_canonical_manifest_materializes_the_exact_54_job_gate() -> None:
    manifest = _manifest()
    jobs = validate.validate_manifest(manifest, repo_root=_REPO_ROOT)

    assert len(jobs) == 54
    assert len({job["id"] for job in jobs}) == 54
    assert {job["backend"] for job in jobs} == {"ddp", "zero2", "fsdp2"}
    assert {job["algorithm"] for job in jobs} == {
        "grpo",
        "nft",
        "sft",
        "offline-dpo",
        "online-dpo",
        "tdm",
    }
    expected_pairs = {
        ("qwen-image-2.1-anchor", "grpo"),
        ("sd3.5-text-to-image-anchor", "nft"),
        ("sd3.5-text-to-image-anchor", "sft"),
        ("sd3.5-text-to-image-anchor", "offline-dpo"),
        ("sd3.5-text-to-image-anchor", "online-dpo"),
        ("bagel-packed-tdm-anchor", "tdm"),
        *{
            ("flux2-klein-base-4b-multi-reference-edit", algorithm)
            for algorithm in validate.REQUIRED_ALGORITHMS
        },
        *{
            ("minimax-h3-text-to-audio-video", algorithm)
            for algorithm in validate.REQUIRED_ALGORITHMS
        },
    }
    assert {(job["profile"], job["algorithm"]) for job in jobs} == expected_pairs


def test_campaign_keeps_production_image_shape_and_explicit_expensive_media_exception() -> None:
    manifest = _manifest()
    profiles = {profile["id"]: profile for profile in manifest["profiles"]}
    workloads = manifest["workloads"]

    assert manifest["cluster"] == {
        "world_size": 32,
        "num_machines": 4,
        "processes_per_machine": 8,
        "mixed_precision": "bf16",
        "image_long_edge": 1024,
    }
    assert manifest["common_overrides"] == {
        "launcher": "accelerate",
        "log": {"logging_backend": "none", "save_freq": 0},
        "eval": {"eval_freq": 0},
        "train": {"seed": 42},
    }
    assert manifest["reward_profiles"] == {
        "pickscore-cpu-async": {
            "reward_model": "PickScore",
            "device": "cpu",
            "dtype": "float32",
            "batch_size": 16,
            "async_reward": True,
            "num_workers": 1,
        }
    }
    for name in (
        "image-reward-subgroup-overlap",
        "image-reward-global-batch-overlap",
        "image-reward-rank-local-overlap",
    ):
        assert workloads[name]["group_size"] == 16
        assert workloads[name]["unique_sample_num_per_epoch"] == 96
    assert workloads["image-tdm-packed"]["per_device_batch_size"] == 2
    assert workloads["image-tdm-packed"]["unique_sample_num_per_epoch"] == 128

    qwen = profiles["qwen-image-2.1-anchor"]
    assert qwen["task"] == "text_to_image"
    assert qwen["dataset_profile"] == "ocr-prompts"

    flux = profiles["flux2-klein-base-4b-multi-reference-edit"]
    assert flux["checkpoint"] == "black-forest-labs/FLUX.2-klein-base-4B"
    assert flux["task"] == "multi_image_to_image"
    assert flux["geometry"]["condition_image_count"] == 2
    assert set(flux["runs"]) == validate.REQUIRED_ALGORITHMS

    h3 = profiles["minimax-h3-text-to-audio-video"]
    assert h3["geometry"]["resolution"] == [576, 1024]
    assert h3["geometry"]["num_frames"] / h3["geometry"]["frame_rate"] >= 5.0
    assert set(h3["runs"]) == validate.REQUIRED_ALGORITHMS
    for name in ("av-reward-subgroup", "av-reward-rank-local"):
        assert workloads[name]["group_size"] == 2
        assert workloads[name]["unique_sample_num_per_epoch"] == 32
        assert workloads[name]["reward_optimization_overlap"] is False


def test_every_runtime_reward_job_uses_pickscore_and_reward_free_jobs_load_none() -> None:
    manifest = _manifest()
    jobs = validate.validate_manifest(manifest, repo_root=_REPO_ROOT)

    assert {"prompt", "image", "video"} <= set(PickScoreRewardModel.required_fields)
    for job in jobs:
        reward_profile = job["run_contract"]["reward_profile"]
        if job["run_contract"]["feedback"] == "runtime_reward":
            assert reward_profile == {
                "id": "pickscore-cpu-async",
                "reward_model": "PickScore",
                "device": "cpu",
                "dtype": "float32",
                "batch_size": 16,
                "async_reward": True,
                "num_workers": 1,
            }
        else:
            assert reward_profile is None


def test_runtime_backend_assertions_cannot_be_replaced_by_launcher_labels() -> None:
    manifest = _manifest()

    assert manifest["backends"]["ddp"]["runtime_assertions"] == {"distributed_type": "MULTI_GPU"}
    assert manifest["backends"]["zero2"]["runtime_assertions"] == {
        "distributed_type": "DEEPSPEED",
        "zero_stage": 2,
    }
    assert manifest["backends"]["fsdp2"]["runtime_assertions"] == {
        "distributed_type": "FSDP",
        "fsdp_version": 2,
    }


def test_complete_exact_commit_evidence_passes() -> None:
    manifest = _manifest()
    results = _passing_results(manifest)

    validate.validate_results(
        manifest,
        results,
        expected_manifest_sha256=validate.manifest_sha256(_MANIFEST_PATH),
        expected_commit_sha="a" * 40,
    )


@pytest.mark.parametrize(
    "failure",
    ["missing", "skipped", "wrong_backend", "wrong_workload", "stale_manifest", "stale_commit"],
)
def test_incomplete_or_masquerading_evidence_never_passes(failure: str) -> None:
    manifest = _manifest()
    results = _passing_results(manifest)
    if failure == "missing":
        results["jobs"].pop()
    elif failure == "skipped":
        results["jobs"][0]["status"] = "infrastructure"
    elif failure == "wrong_backend":
        zero2 = next(
            job for job in results["jobs"] if job["observations"]["backend"]["id"] == "zero2"
        )
        zero2["observations"]["backend"] = {
            "id": "zero2",
            "distributed_type": "MULTI_GPU",
            "zero_stage": 2,
        }
    elif failure == "wrong_workload":
        results["jobs"][0]["observations"]["run_contract"]["geometry"]["resolution"] = [
            512,
            512,
        ]
    elif failure == "stale_manifest":
        results["manifest_sha256"] = "0" * 64
    else:
        results["commit_sha"] = "b" * 40

    with pytest.raises(validate.CampaignValidationError):
        validate.validate_results(
            manifest,
            results,
            expected_manifest_sha256=validate.manifest_sha256(_MANIFEST_PATH),
            expected_commit_sha="a" * 40,
        )


def test_manifest_rejects_geometry_that_cannot_close_rank_batches() -> None:
    manifest = copy.deepcopy(_manifest())
    manifest["workloads"]["image-reward-subgroup-overlap"]["unique_sample_num_per_epoch"] = 95

    with pytest.raises(validate.CampaignValidationError, match="does not close"):
        validate.validate_manifest(manifest, repo_root=_REPO_ROOT)


@pytest.mark.parametrize(
    ("workload", "unique_samples", "message"),
    [
        ("image-reward-subgroup-overlap", 98, "subgroup_tile.*multiple of 4"),
        ("image-reward-rank-local-overlap", 94, "rank_local.*multiple of 32"),
    ],
)
def test_manifest_enforces_sampler_specific_group_alignment(
    workload: str, unique_samples: int, message: str
) -> None:
    manifest = copy.deepcopy(_manifest())
    manifest["workloads"][workload]["unique_sample_num_per_epoch"] = unique_samples

    with pytest.raises(validate.CampaignValidationError, match=message):
        validate.validate_manifest(manifest, repo_root=_REPO_ROOT)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("reward_model", "CLIP", "must use PickScore"),
        ("device", "cuda", "requires an async CPU reward profile"),
        ("async_reward", False, "requires an async CPU reward profile"),
    ],
)
def test_manifest_rejects_non_pickscore_or_overlap_incompatible_reward_profiles(
    field: str, value: object, message: str
) -> None:
    manifest = copy.deepcopy(_manifest())
    manifest["reward_profiles"]["pickscore-cpu-async"][field] = value

    with pytest.raises(validate.CampaignValidationError, match=message):
        validate.validate_manifest(manifest, repo_root=_REPO_ROOT)


def test_hard_constraint_and_agent_workflows_route_to_the_manifest_gate() -> None:
    expected_path = "config/gpu_validation/framework_upgrade.yaml"
    constraints = (_REPO_ROOT / ".agents/knowledge/constraints.md").read_text(encoding="utf-8")
    guidance = (_REPO_ROOT / "guidance/gpu_validation.md").read_text(encoding="utf-8")
    develop = (_REPO_ROOT / ".agents/skills/ff-develop/SKILL.md").read_text(encoding="utf-8")
    review = (_REPO_ROOT / ".agents/skills/ff-review/SKILL.md").read_text(encoding="utf-8")

    assert "### 30. Framework-Upgrade GPU Merge Gate" in constraints
    assert expected_path in constraints
    assert expected_path in guidance
    assert "54 mandatory jobs" in guidance
    assert "guidance/gpu_validation.md" in develop
    assert "guidance/gpu_validation.md" in review
