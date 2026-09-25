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
    minimum_work_units = manifest["overlap_critical_paths"]["minimum_work_units"]
    result_jobs = []
    for job in jobs:
        run_contract = job["run_contract"]
        observations = {
            "all_ranks_completed": True,
            "finite_metrics": True,
            "parameter_update": True,
            "world_size": job["world_size"],
            "backend": {"id": job["backend"], **job["runtime_assertions"]},
            "cycle": job["cycle"],
            "run_contract": run_contract,
        }
        artifacts = {
            "command": f"artifacts/{job['id']}/command.txt",
            "resolved_config": f"artifacts/{job['id']}/resolved.yaml",
            "environment": f"artifacts/{job['id']}/environment.json",
            "logs": f"artifacts/{job['id']}/train.log",
            "metrics": f"artifacts/{job['id']}/metrics.json",
        }
        if run_contract["feedback"] == "runtime_reward":
            reward_profile = run_contract["reward_profile"]
            source_ids = reward_profile["source_services"]
            workload = run_contract["workload"]
            sample_count = workload["group_size"] * workload["unique_sample_num_per_epoch"]
            observations["reward_runtime"] = {
                "source_services": source_ids,
                "deployment_kind": reward_profile["deployment_kind"],
                "all_sources_ready": True,
                "service_topology": {
                    service["id"]: (
                        service["server"]
                        if service["placement"] == "remote_service"
                        else {
                            "placement": "in_process",
                            "reward_model": service["reward_model"],
                        }
                    )
                    for service in reward_profile["services"]
                },
                "endpoint_env_bindings": {
                    service["id"]: service["endpoint_env"]
                    for service in reward_profile["services"]
                    if service["placement"] == "remote_service"
                },
                "scored_samples": sample_count,
                "source_sample_counts": {source_id: sample_count for source_id in source_ids},
                "source_latency_ms": {
                    source_id: {"p50": 10.0, "p95": 20.0, "max": 30.0} for source_id in source_ids
                },
            }
            overlap_enabled = workload.get("reward_optimization_overlap", False)
            observations["reward_overlap"] = {"enabled": overlap_enabled}
            if overlap_enabled:
                observations["reward_overlap"].update(
                    {
                        "mode": workload["reward_optimization_overlap_mode"],
                        "work_units": minimum_work_units,
                        "poll_count": 3,
                        "optimization_overlap_seconds": 1.0,
                        "reward_wait_seconds": 0.5,
                        "out_of_order_work_units": 0,
                        "optimization_started_while_reward_pending": True,
                    }
                )
            artifacts.update(
                {
                    "reward_service_probe": f"artifacts/{job['id']}/reward-probe.json",
                    "reward_timing": f"artifacts/{job['id']}/reward-timing.json",
                }
            )
        result_jobs.append(
            {
                "id": job["id"],
                "status": "passed",
                "observations": observations,
                "artifacts": artifacts,
            }
        )
    return {
        "gate_id": manifest["gate"]["id"],
        "manifest_sha256": validate.manifest_sha256(_MANIFEST_PATH),
        "commit_sha": "a" * 40,
        "jobs": result_jobs,
    }


def test_canonical_manifest_materializes_36_core_and_9_overlap_jobs() -> None:
    manifest = _manifest()
    jobs = validate.validate_manifest(manifest, repo_root=_REPO_ROOT)
    core_jobs = [job for job in jobs if job["suite"] == "core"]
    overlap_jobs = [job for job in jobs if job["suite"] == "overlap_supplemental"]

    assert len(jobs) == 45
    assert len(core_jobs) == 36
    assert len(overlap_jobs) == 9
    assert len({job["id"] for job in jobs}) == 45
    assert {job["backend"] for job in jobs} == {"ddp", "zero2", "fsdp2"}
    assert {job["algorithm"] for job in core_jobs} == validate.REQUIRED_CORE_ALGORITHMS
    assert validate.REQUIRED_OVERLAP_ALGORITHMS <= {job["algorithm"] for job in jobs}
    expected_pairs = {
        ("qwen-image-2.1-anchor", "grpo"),
        ("sd3.5-text-to-image-anchor", "nft"),
        ("sd3.5-text-to-image-anchor", "sft"),
        ("sd3.5-text-to-image-anchor", "offline-dpo"),
        ("sd3.5-text-to-image-anchor", "online-dpo"),
        ("bagel-packed-tdm-anchor", "tdm"),
        *{
            ("minimax-h3-text-to-audio-video", algorithm)
            for algorithm in validate.REQUIRED_CORE_ALGORITHMS
        },
    }
    assert {(job["profile"], job["algorithm"]) for job in core_jobs} == expected_pairs
    jobs_by_id = {job["id"]: job for job in jobs}
    for backend in ("ddp", "zero2", "fsdp2"):
        assert jobs_by_id[f"sd3.5-text-to-image-anchor__{backend}__nft"]["cycle"][
            "optimizer_steps"
        ] == {"default": 2}
        assert jobs_by_id[f"sd3.5-text-to-image-anchor__{backend}__online-dpo"]["cycle"][
            "optimizer_steps"
        ] == {"default": 3}
        assert jobs_by_id[f"minimax-h3-text-to-audio-video__{backend}__offline-dpo"][
            "run_contract"
        ]["model_overrides"] == {
            "lora_rank": 16,
            "lora_alpha": 16,
        }
    assert jobs_by_id["overlap__sd35__ddp__awm__subgroup-ready__aes"]["cycle"][
        "optimizer_steps"
    ] == {"default": 2}
    assert jobs_by_id["overlap__sd35__ddp__dgpo__global-batch-ready__multi-gdpo"]["cycle"][
        "optimizer_steps"
    ] == {"default": 2}
    assert jobs_by_id["overlap__sd35__ddp__tdm-r1__global-batch-ready__multi"]["cycle"][
        "optimizer_steps"
    ] == {"generator": 1, "fake": 1, "surrogate": 1}
    dppo = jobs_by_id["overlap__sd35__ddp__dppo__subgroup-ready__aes"]
    assert dppo["run_contract"]["reward_profile"]["id"] == "remote-aes-async"
    assert dppo["run_contract"]["overlap_observation"] == "required"
    bagel = jobs_by_id["overlap__bagel-b2__ddp__grpo__subgroup-ready__hy-ocr"]
    assert bagel["run_contract"]["reward_profile"]["id"] == "remote-hy-ocr-async"
    assert bagel["run_contract"]["overlap_observation"] == "observe_only"


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
    assert set(manifest["reward_profiles"]) == {
        "remote-aes-async",
        "remote-hy-ocr-async",
        "remote-aes-plus-hy-ocr-async",
        "local-clap-plus-imagebind-sync",
    }
    assert manifest["reward_services"]["aes-v3"]["server"] == {
        "gpu_count": 72,
        "replica_count": 9,
        "tensor_parallel_size": 8,
    }
    assert manifest["reward_services"]["hy-ocr-1.5"]["server"] == {
        "gpu_count": 8,
        "data_parallel_size": 8,
    }
    for name in (
        "image-reward-subgroup-ready",
        "image-reward-global-batch-ready",
        "image-reward-rank-local-ready",
    ):
        assert workloads[name]["group_size"] == 16
        assert workloads[name]["unique_sample_num_per_epoch"] == 96
    assert workloads["image-tdm-packed"]["per_device_batch_size"] == 2
    assert workloads["image-tdm-packed"]["unique_sample_num_per_epoch"] == 128

    qwen = profiles["qwen-image-2.1-anchor"]
    assert qwen["task"] == "text_to_image"
    assert qwen["dataset_profile"] == "ocr-prompts"
    assert qwen["reward_profile"] == "remote-aes-plus-hy-ocr-async"
    assert qwen["advantage_aggregation"] == "gdpo"

    h3 = profiles["minimax-h3-text-to-audio-video"]
    assert h3["geometry"]["resolution"] == [576, 1024]
    assert h3["geometry"]["num_frames"] / h3["geometry"]["frame_rate"] >= 5.0
    assert set(h3["runs"]) == validate.REQUIRED_CORE_ALGORITHMS
    assert h3["reward_profile"] == "local-clap-plus-imagebind-sync"
    for name in ("av-reward-subgroup-sync", "av-reward-rank-local-sync"):
        assert workloads[name]["group_size"] == 2
        assert workloads[name]["unique_sample_num_per_epoch"] == 32
        assert workloads[name]["reward_optimization_overlap"] is False


def test_runtime_reward_jobs_use_task_appropriate_deployments() -> None:
    manifest = _manifest()
    jobs = validate.validate_manifest(manifest, repo_root=_REPO_ROOT)

    for job in jobs:
        reward_profile = job["run_contract"]["reward_profile"]
        if job["run_contract"]["feedback"] == "runtime_reward":
            assert reward_profile is not None
            if job["run_contract"]["workload"].get("reward_optimization_overlap"):
                assert reward_profile["deployment_kind"] == "remote_service"
                assert reward_profile["client_device"] == "cpu"
                assert reward_profile["async_reward"] is True
                assert all(
                    service["output_scope"] == "pointwise" for service in reward_profile["services"]
                )
            else:
                assert reward_profile["id"] == "local-clap-plus-imagebind-sync"
                assert reward_profile["async_reward"] is False
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
    manifest["workloads"]["image-reward-subgroup-ready"]["unique_sample_num_per_epoch"] = 95

    with pytest.raises(validate.CampaignValidationError, match="does not close"):
        validate.validate_manifest(manifest, repo_root=_REPO_ROOT)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"lora_rank": 16}, "must contain exactly"),
        ({"lora_rank": 0, "lora_alpha": 16}, "positive integer"),
    ],
)
def test_manifest_rejects_incomplete_or_nonpositive_model_overrides(
    overrides: dict[str, int], message: str
) -> None:
    manifest = copy.deepcopy(_manifest())
    h3 = next(
        profile
        for profile in manifest["profiles"]
        if profile["id"] == "minimax-h3-text-to-audio-video"
    )
    h3["runs"]["offline-dpo"]["model_overrides"] = overrides

    with pytest.raises(validate.CampaignValidationError, match=message):
        validate.validate_manifest(manifest, repo_root=_REPO_ROOT)


@pytest.mark.parametrize(
    ("workload", "unique_samples", "message"),
    [
        ("image-reward-subgroup-ready", 98, "subgroup_tile.*multiple of 4"),
        ("image-reward-rank-local-ready", 94, "rank_local.*multiple of 32"),
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
        ("client_device", "cuda", "must use remote pointwise services through CPU clients"),
        ("async_reward", False, "requires an async remote reward profile"),
    ],
)
def test_manifest_rejects_overlap_incompatible_reward_profiles(
    field: str, value: object, message: str
) -> None:
    manifest = copy.deepcopy(_manifest())
    manifest["reward_profiles"]["remote-aes-async"][field] = value

    with pytest.raises(validate.CampaignValidationError, match=message):
        validate.validate_manifest(manifest, repo_root=_REPO_ROOT)


def test_manifest_rejects_drift_in_pairwise_overlap_coverage() -> None:
    manifest = copy.deepcopy(_manifest())
    manifest["workloads"]["image-reward-subgroup-ordered"][
        "reward_optimization_overlap_mode"
    ] = "ready"

    with pytest.raises(validate.CampaignValidationError, match="required_scheduling_modes"):
        validate.validate_manifest(manifest, repo_root=_REPO_ROOT)


def test_manifest_requires_both_overlap_observation_policies() -> None:
    manifest = copy.deepcopy(_manifest())
    manifest["overlap_critical_paths"]["supplemental_jobs"][-1].pop("overlap_observation")

    with pytest.raises(validate.CampaignValidationError, match="required and observe-only"):
        validate.validate_manifest(manifest, repo_root=_REPO_ROOT)


def test_manifest_rejects_overlap_cycle_with_only_one_work_unit() -> None:
    manifest = copy.deepcopy(_manifest())
    manifest["profiles"][1]["runs"]["nft"].pop("optimizer_steps")

    with pytest.raises(validate.CampaignValidationError, match="at least 2 optimizer work units"):
        validate.validate_manifest(manifest, repo_root=_REPO_ROOT)


def test_manifest_cannot_weaken_overlap_minimum_to_one_work_unit() -> None:
    manifest = copy.deepcopy(_manifest())
    manifest["overlap_critical_paths"]["minimum_work_units"] = 1

    with pytest.raises(validate.CampaignValidationError, match="must be at least 2"):
        validate.validate_manifest(manifest, repo_root=_REPO_ROOT)


def test_manifest_rejects_cycle_override_that_changes_optimizer_roles() -> None:
    manifest = copy.deepcopy(_manifest())
    manifest["profiles"][1]["runs"]["nft"]["optimizer_steps"] = {"generator": 2}

    with pytest.raises(validate.CampaignValidationError, match="must preserve algorithm roles"):
        validate.validate_manifest(manifest, repo_root=_REPO_ROOT)


def test_manifest_rejects_missing_tdm_r1_surrogate_optimizer_role() -> None:
    manifest = copy.deepcopy(_manifest())
    manifest["algorithms"]["tdm-r1"]["cycle"]["optimizer_steps"].pop("surrogate")

    with pytest.raises(validate.CampaignValidationError, match="tdm-r1.*surrogate"):
        validate.validate_manifest(manifest, repo_root=_REPO_ROOT)


@pytest.mark.parametrize("failure", ["unhealthy", "missing_source", "no_concurrency"])
def test_runtime_reward_evidence_must_prove_real_async_execution(failure: str) -> None:
    manifest = _manifest()
    results = _passing_results(manifest)
    result = next(job for job in results["jobs"] if job["id"] == "qwen-image-2.1-anchor__ddp__grpo")
    if failure == "unhealthy":
        result["observations"]["reward_runtime"]["all_sources_ready"] = False
    elif failure == "missing_source":
        result["observations"]["reward_runtime"]["source_services"].pop()
    else:
        result["observations"]["reward_overlap"]["optimization_overlap_seconds"] = 0.0

    with pytest.raises(validate.CampaignValidationError):
        validate.validate_results(
            manifest,
            results,
            expected_manifest_sha256=validate.manifest_sha256(_MANIFEST_PATH),
            expected_commit_sha="a" * 40,
        )


def test_observe_only_fast_reward_boundary_allows_measured_zero_overlap() -> None:
    manifest = _manifest()
    results = _passing_results(manifest)
    result = next(
        job
        for job in results["jobs"]
        if job["id"] == "overlap__bagel-b2__ddp__grpo__subgroup-ready__hy-ocr"
    )
    result["observations"]["reward_overlap"].update(
        {
            "optimization_overlap_seconds": 0.0,
            "optimization_started_while_reward_pending": False,
        }
    )

    validate.validate_results(
        manifest,
        results,
        expected_manifest_sha256=validate.manifest_sha256(_MANIFEST_PATH),
        expected_commit_sha="a" * 40,
    )


def test_hard_constraint_and_agent_workflows_route_to_the_manifest_gate() -> None:
    expected_path = "config/gpu_validation/framework_upgrade.yaml"
    constraints = (_REPO_ROOT / ".agents/knowledge/constraints.md").read_text(encoding="utf-8")
    guidance = (_REPO_ROOT / "guidance/gpu_validation.md").read_text(encoding="utf-8")
    develop = (_REPO_ROOT / ".agents/skills/ff-develop/SKILL.md").read_text(encoding="utf-8")
    review = (_REPO_ROOT / ".agents/skills/ff-review/SKILL.md").read_text(encoding="utf-8")

    assert "### 30. Framework-Upgrade GPU Merge Gate" in constraints
    assert expected_path in constraints
    assert expected_path in guidance
    assert "45 mandatory jobs" in guidance
    assert "guidance/gpu_validation.md" in develop
    assert "guidance/gpu_validation.md" in review
