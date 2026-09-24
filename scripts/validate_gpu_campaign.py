#!/usr/bin/env python3
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

"""Validate the framework-upgrade GPU campaign and its exact-commit evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections.abc import Mapping, Sequence
from math import gcd
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "config/gpu_validation/framework_upgrade.yaml"
REQUIRED_ALGORITHMS = {
    "grpo",
    "nft",
    "sft",
    "offline-dpo",
    "online-dpo",
    "tdm",
}
REQUIRED_BACKENDS = {"ddp", "zero2", "fsdp2"}
REQUIRED_JOB_OBSERVATIONS = {
    "all_ranks_completed",
    "finite_metrics",
    "parameter_update",
    "world_size",
    "backend",
    "cycle",
    "run_contract",
}
_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")


class CampaignValidationError(ValueError):
    """Report a malformed campaign contract or incomplete result bundle."""


def load_mapping(path: Path) -> dict[str, Any]:
    """Load a YAML or JSON mapping from disk.

    Args:
        path: Input file path.

    Returns:
        Mutable top-level mapping.

    Raises:
        CampaignValidationError: If the document is not a mapping.
    """
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() == ".json":
            value = json.load(handle)
        else:
            value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise CampaignValidationError(
            f"expected a mapping in {path}, received {type(value).__name__}"
        )
    return value


def manifest_sha256(path: Path) -> str:
    """Return the digest that binds evidence to an exact campaign manifest.

    Args:
        path: Manifest path.

    Returns:
        Lowercase SHA-256 hexadecimal digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head(repo_root: Path) -> str:
    """Return the exact commit checked out in a repository or worktree.

    Args:
        repo_root: Git repository or worktree root.

    Returns:
        Full lowercase commit SHA.

    Raises:
        CampaignValidationError: If Git cannot resolve a full commit SHA.
    """
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    commit_sha = process.stdout.strip()
    if process.returncode or _COMMIT_SHA.fullmatch(commit_sha) is None:
        detail = process.stderr.strip() or commit_sha or "no output"
        raise CampaignValidationError(f"could not resolve the tested commit SHA: {detail}")
    return commit_sha


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CampaignValidationError(
            f"expected {location} to be a mapping, received {type(value).__name__}"
        )
    return value


def _sequence(value: Any, location: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise CampaignValidationError(
            f"expected {location} to be a sequence, received {type(value).__name__}"
        )
    return value


def _positive_int(value: Any, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CampaignValidationError(
            f"expected {location} to be a positive integer, received {value!r}"
        )
    return value


def _nonempty_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CampaignValidationError(f"expected {location} to be a non-empty string")
    return value


def _validate_backend_contracts(
    manifest: Mapping[str, Any], repo_root: Path | None
) -> Mapping[str, Any]:
    gate = _mapping(manifest.get("gate"), "gate")
    backends = _mapping(manifest.get("backends"), "backends")
    if set(backends) != REQUIRED_BACKENDS:
        raise CampaignValidationError(
            f"framework-upgrade backends must be {sorted(REQUIRED_BACKENDS)}, received {sorted(backends)}"
        )
    expected_count = _positive_int(
        gate.get("expected_backend_count"), "gate.expected_backend_count"
    )
    if len(backends) != expected_count:
        raise CampaignValidationError(
            f"expected {expected_count} backends, received {len(backends)}"
        )
    for backend_id, value in backends.items():
        backend = _mapping(value, f"backends.{backend_id}")
        config_file = _nonempty_string(
            backend.get("config_file"), f"backends.{backend_id}.config_file"
        )
        runtime = _mapping(
            backend.get("runtime_assertions"), f"backends.{backend_id}.runtime_assertions"
        )
        _nonempty_string(
            runtime.get("distributed_type"),
            f"backends.{backend_id}.runtime_assertions.distributed_type",
        )
        if repo_root is not None and not (repo_root / config_file).is_file():
            raise CampaignValidationError(
                f"backend {backend_id!r} references missing config file {config_file!r}"
            )
    return backends


def _validate_algorithm_contracts(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    algorithms = _mapping(manifest.get("algorithms"), "algorithms")
    if set(algorithms) != REQUIRED_ALGORITHMS:
        raise CampaignValidationError(
            "framework-upgrade algorithms must be "
            f"{sorted(REQUIRED_ALGORITHMS)}, received {sorted(algorithms)}"
        )
    for algorithm_id, value in algorithms.items():
        algorithm = _mapping(value, f"algorithms.{algorithm_id}")
        _nonempty_string(algorithm.get("trainer_type"), f"algorithms.{algorithm_id}.trainer_type")
        acquisition = _nonempty_string(
            algorithm.get("acquisition"), f"algorithms.{algorithm_id}.acquisition"
        )
        feedback = _nonempty_string(
            algorithm.get("feedback"), f"algorithms.{algorithm_id}.feedback"
        )
        if acquisition not in {"generation", "dataset"}:
            raise CampaignValidationError(
                f"algorithm {algorithm_id!r} has unsupported acquisition {acquisition!r}"
            )
        if feedback not in {"runtime_reward", "none"}:
            raise CampaignValidationError(
                f"algorithm {algorithm_id!r} has unsupported feedback {feedback!r}"
            )
        cycle = _mapping(algorithm.get("cycle"), f"algorithms.{algorithm_id}.cycle")
        units = _positive_int(
            cycle.get("acquisition_units"),
            f"algorithms.{algorithm_id}.cycle.acquisition_units",
        )
        if units != 1:
            raise CampaignValidationError(
                f"algorithm {algorithm_id!r} must exercise exactly one acquisition unit"
            )
        _nonempty_string(
            cycle.get("acquisition_unit"),
            f"algorithms.{algorithm_id}.cycle.acquisition_unit",
        )
        optimizer_steps = _mapping(
            cycle.get("optimizer_steps"),
            f"algorithms.{algorithm_id}.cycle.optimizer_steps",
        )
        if not optimizer_steps:
            raise CampaignValidationError(
                f"algorithm {algorithm_id!r} must declare at least one optimizer role"
            )
        for role, count in optimizer_steps.items():
            _positive_int(count, f"algorithms.{algorithm_id}.cycle.optimizer_steps.{role}")
    return algorithms


def _validate_workload(
    workload_id: str,
    workload: Mapping[str, Any],
    *,
    algorithm: Mapping[str, Any],
    world_size: int,
) -> None:
    batch_size = _positive_int(
        workload.get("per_device_batch_size"),
        f"workloads.{workload_id}.per_device_batch_size",
    )
    acquisition = algorithm["acquisition"]
    if acquisition == "dataset":
        accumulation = _positive_int(
            workload.get("gradient_accumulation_steps"),
            f"workloads.{workload_id}.gradient_accumulation_steps",
        )
        records = _positive_int(
            workload.get("dataset_records"), f"workloads.{workload_id}.dataset_records"
        )
        expected_records = world_size * batch_size * accumulation
        if records != expected_records:
            raise CampaignValidationError(
                f"offline workload {workload_id!r} must contain exactly {expected_records} "
                f"records for one optimizer step, received {records}"
            )
        return

    group_size = _positive_int(workload.get("group_size"), f"workloads.{workload_id}.group_size")
    unique_samples = _positive_int(
        workload.get("unique_sample_num_per_epoch"),
        f"workloads.{workload_id}.unique_sample_num_per_epoch",
    )
    total_samples = group_size * unique_samples
    global_batch_size = world_size * batch_size
    if total_samples % global_batch_size:
        raise CampaignValidationError(
            f"workload {workload_id!r} has U*K={total_samples}, which does not close "
            f"W*B={global_batch_size} rank batches"
        )
    sampler_type = _nonempty_string(
        workload.get("sampler_type"), f"workloads.{workload_id}.sampler_type"
    )
    if sampler_type == "global_batch":
        if group_size > global_batch_size or global_batch_size % group_size:
            raise CampaignValidationError(
                f"global_batch workload {workload_id!r} requires K <= W*B and (W*B) % K == 0"
            )
    elif sampler_type == "subgroup_tile":
        subgroup_size = _positive_int(
            workload.get("sampler_subgroup_size"),
            f"workloads.{workload_id}.sampler_subgroup_size",
        )
        if world_size % subgroup_size:
            raise CampaignValidationError(
                f"subgroup_tile workload {workload_id!r} requires subgroup size to divide world size"
            )
        alignment = (world_size // subgroup_size) * (
            subgroup_size * batch_size // gcd(subgroup_size * batch_size, group_size)
        )
        if unique_samples % alignment:
            raise CampaignValidationError(
                f"subgroup_tile workload {workload_id!r} requires U to be a multiple of {alignment}"
            )
    elif sampler_type == "rank_local":
        alignment = world_size * batch_size // gcd(batch_size, group_size)
        if unique_samples % alignment:
            raise CampaignValidationError(
                f"rank_local workload {workload_id!r} requires U to be a multiple of {alignment}"
            )
    else:
        raise CampaignValidationError(
            f"campaign workload {workload_id!r} uses unsupported sampler_type {sampler_type!r}"
        )
    if algorithm["feedback"] == "runtime_reward" and group_size < 2:
        raise CampaignValidationError(
            f"runtime-reward workload {workload_id!r} requires a non-degenerate group_size"
        )
    if algorithm["feedback"] != "runtime_reward" and workload.get(
        "reward_optimization_overlap", False
    ):
        raise CampaignValidationError(
            f"reward-free workload {workload_id!r} cannot enable reward/optimization overlap"
        )
    if workload.get("reward_optimization_overlap", False):
        if workload.get("reward_optimization_overlap_mode") != "ready":
            raise CampaignValidationError(
                f"overlap workload {workload_id!r} must exercise ready-order scheduling"
            )
        if workload.get("global_std") is not False:
            raise CampaignValidationError(
                f"overlap workload {workload_id!r} must keep global_std=false"
            )


def validate_manifest(
    manifest: Mapping[str, Any], *, repo_root: Path | None = None
) -> list[dict[str, Any]]:
    """Validate the campaign contract and enumerate its exact job matrix.

    Args:
        manifest: Parsed campaign manifest.
        repo_root: Optional repository root used to validate referenced files.

    Returns:
        Ordered concrete job records.

    Raises:
        CampaignValidationError: If a contract or matrix invariant is invalid.
    """
    if manifest.get("schema_version") != 1:
        raise CampaignValidationError(
            f"expected schema_version 1, received {manifest.get('schema_version')!r}"
        )
    gate = _mapping(manifest.get("gate"), "gate")
    _nonempty_string(gate.get("id"), "gate.id")
    triggers = _sequence(gate.get("trigger_scopes"), "gate.trigger_scopes")
    if not triggers or any(not isinstance(value, str) or not value for value in triggers):
        raise CampaignValidationError("gate.trigger_scopes must contain non-empty strings")
    if gate.get("exact_commit_required") is not True or gate.get("all_jobs_must_pass") is not True:
        raise CampaignValidationError(
            "framework-upgrade gate must require exact-commit evidence and all jobs to pass"
        )

    cluster = _mapping(manifest.get("cluster"), "cluster")
    world_size = _positive_int(cluster.get("world_size"), "cluster.world_size")
    num_machines = _positive_int(cluster.get("num_machines"), "cluster.num_machines")
    per_machine = _positive_int(
        cluster.get("processes_per_machine"), "cluster.processes_per_machine"
    )
    if num_machines * per_machine != world_size:
        raise CampaignValidationError(
            "cluster.num_machines * cluster.processes_per_machine must equal cluster.world_size"
        )
    image_long_edge = _positive_int(cluster.get("image_long_edge"), "cluster.image_long_edge")
    common_overrides = _mapping(manifest.get("common_overrides"), "common_overrides")
    if common_overrides.get("launcher") != "accelerate":
        raise CampaignValidationError("common_overrides.launcher must be 'accelerate'")
    common_log = _mapping(common_overrides.get("log"), "common_overrides.log")
    common_eval = _mapping(common_overrides.get("eval"), "common_overrides.eval")
    if common_log.get("logging_backend") != "none" or common_log.get("save_freq") != 0:
        raise CampaignValidationError(
            "framework-upgrade jobs must disable external logging and checkpoint saving"
        )
    if common_eval.get("eval_freq") != 0:
        raise CampaignValidationError("framework-upgrade jobs must disable evaluation")

    backends = _validate_backend_contracts(manifest, repo_root)
    algorithms = _validate_algorithm_contracts(manifest)
    workloads = _mapping(manifest.get("workloads"), "workloads")
    profiles = _sequence(manifest.get("profiles"), "profiles")
    if not profiles:
        raise CampaignValidationError("profiles must not be empty")

    seen_profiles: set[str] = set()
    jobs: list[dict[str, Any]] = []
    pair_count = 0
    for profile_index, value in enumerate(profiles):
        profile = _mapping(value, f"profiles[{profile_index}]")
        profile_id = _nonempty_string(profile.get("id"), f"profiles[{profile_index}].id")
        if profile_id in seen_profiles:
            raise CampaignValidationError(f"duplicate profile id {profile_id!r}")
        seen_profiles.add(profile_id)
        for field in ("task", "dataset_profile", "model_type", "checkpoint"):
            _nonempty_string(profile.get(field), f"profiles.{profile_id}.{field}")
        geometry = _mapping(profile.get("geometry"), f"profiles.{profile_id}.geometry")
        resolution = _sequence(
            geometry.get("resolution"), f"profiles.{profile_id}.geometry.resolution"
        )
        if len(resolution) != 2 or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in resolution
        ):
            raise CampaignValidationError(
                f"profile {profile_id!r} resolution must be [positive height, positive width]"
            )
        if max(resolution) != image_long_edge:
            raise CampaignValidationError(
                f"profile {profile_id!r} must use the campaign's {image_long_edge}px long edge"
            )
        if profile.get("task") == "text_to_audio_video":
            frames = _positive_int(
                geometry.get("num_frames"), f"profiles.{profile_id}.geometry.num_frames"
            )
            frame_rate = geometry.get("frame_rate")
            minimum_duration = geometry.get("minimum_duration_seconds")
            if not isinstance(frame_rate, (int, float)) or frame_rate <= 0:
                raise CampaignValidationError(
                    f"profile {profile_id!r} requires a positive frame_rate"
                )
            if not isinstance(minimum_duration, (int, float)) or minimum_duration <= 0:
                raise CampaignValidationError(
                    f"profile {profile_id!r} requires a positive minimum_duration_seconds"
                )
            if frames / frame_rate < minimum_duration:
                raise CampaignValidationError(
                    f"profile {profile_id!r} is shorter than its minimum duration"
                )
            if profile.get("model_type") == "minimax-h3-t2va":
                if (frames - 5) % 17:
                    raise CampaignValidationError(
                        f"MiniMax H3 profile {profile_id!r} requires num_frames=17*n+5"
                    )
                if any(dimension % 64 for dimension in resolution):
                    raise CampaignValidationError(
                        f"MiniMax H3 profile {profile_id!r} requires dimensions divisible by 64"
                    )
                if geometry.get("guidance_scale") != 1.0:
                    raise CampaignValidationError(
                        f"MiniMax H3 profile {profile_id!r} requires neutral guidance_scale=1.0"
                    )

        runs = _mapping(profile.get("runs"), f"profiles.{profile_id}.runs")
        if not runs:
            raise CampaignValidationError(f"profile {profile_id!r} must declare at least one run")
        for algorithm_id, run_value in runs.items():
            if algorithm_id not in algorithms:
                raise CampaignValidationError(
                    f"profile {profile_id!r} references unknown algorithm {algorithm_id!r}"
                )
            run = _mapping(run_value, f"profiles.{profile_id}.runs.{algorithm_id}")
            recipe = _nonempty_string(
                run.get("recipe"), f"profiles.{profile_id}.runs.{algorithm_id}.recipe"
            )
            workload_id = _nonempty_string(
                run.get("workload"), f"profiles.{profile_id}.runs.{algorithm_id}.workload"
            )
            if workload_id not in workloads:
                raise CampaignValidationError(
                    f"profile {profile_id!r} references unknown workload {workload_id!r}"
                )
            if repo_root is not None and not (repo_root / recipe).is_file():
                raise CampaignValidationError(
                    f"profile {profile_id!r} references missing recipe {recipe!r}"
                )
            algorithm = _mapping(algorithms[algorithm_id], f"algorithms.{algorithm_id}")
            workload = _mapping(workloads[workload_id], f"workloads.{workload_id}")
            _validate_workload(
                workload_id,
                workload,
                algorithm=algorithm,
                world_size=world_size,
            )
            if algorithm_id == "online-dpo" and workload.get("sampler_type") != "rank_local":
                raise CampaignValidationError(
                    f"online-dpo profile {profile_id!r} must use rank_local sampler placement"
                )
            pair_count += 1
            for backend_id, backend_value in backends.items():
                backend = _mapping(backend_value, f"backends.{backend_id}")
                jobs.append(
                    {
                        "id": f"{profile_id}__{backend_id}__{algorithm_id}",
                        "profile": profile_id,
                        "model_type": profile["model_type"],
                        "algorithm": algorithm_id,
                        "backend": backend_id,
                        "world_size": world_size,
                        "runtime_assertions": dict(backend["runtime_assertions"]),
                        "cycle": dict(algorithm["cycle"]),
                        "run_contract": {
                            "task": profile["task"],
                            "dataset_profile": profile["dataset_profile"],
                            "model_type": profile["model_type"],
                            "checkpoint": profile["checkpoint"],
                            "geometry": dict(geometry),
                            "trainer_type": algorithm["trainer_type"],
                            "acquisition": algorithm["acquisition"],
                            "feedback": algorithm["feedback"],
                            "recipe": recipe,
                            "workload": {**dict(workload), "id": workload_id},
                            "reward_profile": profile.get("reward_profile"),
                            "advantage_aggregation": profile.get("advantage_aggregation"),
                            "offline_profile": run.get("offline_profile"),
                            "common_overrides": dict(common_overrides),
                        },
                    }
                )

    expected_pairs = _positive_int(
        gate.get("expected_algorithm_profile_pairs"),
        "gate.expected_algorithm_profile_pairs",
    )
    if pair_count != expected_pairs:
        raise CampaignValidationError(
            f"expected {expected_pairs} algorithm/profile pairs, received {pair_count}"
        )
    expected_jobs = _positive_int(gate.get("expected_job_count"), "gate.expected_job_count")
    if len(jobs) != expected_jobs:
        raise CampaignValidationError(f"expected {expected_jobs} jobs, received {len(jobs)}")
    job_ids = [job["id"] for job in jobs]
    if len(job_ids) != len(set(job_ids)):
        raise CampaignValidationError("campaign generated duplicate job ids")
    return jobs


def validate_results(
    manifest: Mapping[str, Any],
    results: Mapping[str, Any],
    *,
    expected_manifest_sha256: str,
    expected_commit_sha: str,
) -> None:
    """Validate that an evidence bundle passes every exact campaign job.

    Args:
        manifest: Parsed campaign manifest.
        results: Parsed result bundle.
        expected_manifest_sha256: Digest of the manifest used to launch the jobs.
        expected_commit_sha: Full commit SHA whose code must have produced the evidence.

    Raises:
        CampaignValidationError: If evidence is missing, stale, or failed.
    """
    expected_jobs = {job["id"]: job for job in validate_manifest(manifest)}
    gate = _mapping(manifest["gate"], "gate")
    evidence = _mapping(manifest.get("evidence"), "evidence")
    observation_values = _sequence(
        evidence.get("required_job_observations"),
        "evidence.required_job_observations",
    )
    required_observations = {
        _nonempty_string(value, f"evidence.required_job_observations[{index}]")
        for index, value in enumerate(observation_values)
    }
    if required_observations != REQUIRED_JOB_OBSERVATIONS:
        raise CampaignValidationError(
            "evidence.required_job_observations must be "
            f"{sorted(REQUIRED_JOB_OBSERVATIONS)}, received {sorted(required_observations)}"
        )
    if results.get("gate_id") != gate["id"]:
        raise CampaignValidationError(
            f"result gate_id must be {gate['id']!r}, received {results.get('gate_id')!r}"
        )
    if results.get("manifest_sha256") != expected_manifest_sha256:
        raise CampaignValidationError("results were not produced from this exact campaign manifest")
    commit_sha = results.get("commit_sha")
    if not isinstance(commit_sha, str) or _COMMIT_SHA.fullmatch(commit_sha) is None:
        raise CampaignValidationError(
            "results.commit_sha must be a full lowercase 40-character SHA"
        )
    if commit_sha != expected_commit_sha:
        raise CampaignValidationError(
            f"results belong to commit {commit_sha}, but the tested commit is {expected_commit_sha}"
        )
    result_jobs = _sequence(results.get("jobs"), "results.jobs")
    indexed: dict[str, Mapping[str, Any]] = {}
    for index, value in enumerate(result_jobs):
        result = _mapping(value, f"results.jobs[{index}]")
        job_id = _nonempty_string(result.get("id"), f"results.jobs[{index}].id")
        if job_id in indexed:
            raise CampaignValidationError(f"duplicate result for job {job_id!r}")
        indexed[job_id] = result
    missing = sorted(set(expected_jobs) - set(indexed))
    extra = sorted(set(indexed) - set(expected_jobs))
    if missing or extra:
        raise CampaignValidationError(
            f"result job set differs from the manifest; missing={missing}, extra={extra}"
        )

    artifact_values = _sequence(evidence.get("required_artifacts"), "evidence.required_artifacts")
    required_artifacts = [
        _nonempty_string(value, f"evidence.required_artifacts[{index}]")
        for index, value in enumerate(artifact_values)
    ]
    accepted_status = _nonempty_string(evidence.get("accepted_status"), "evidence.accepted_status")
    for job_id, expected in expected_jobs.items():
        result = indexed[job_id]
        if result.get("status") != accepted_status:
            raise CampaignValidationError(
                f"job {job_id!r} has status {result.get('status')!r}, expected {accepted_status!r}"
            )
        observations = _mapping(result.get("observations"), f"results.{job_id}.observations")
        if observations.get("world_size") != expected["world_size"]:
            raise CampaignValidationError(
                f"job {job_id!r} did not run at world_size={expected['world_size']}"
            )
        for flag in ("all_ranks_completed", "finite_metrics", "parameter_update"):
            if observations.get(flag) is not True:
                raise CampaignValidationError(f"job {job_id!r} requires observation {flag}=true")
        backend = _mapping(observations.get("backend"), f"results.{job_id}.backend")
        if backend.get("id") != expected["backend"]:
            raise CampaignValidationError(
                f"job {job_id!r} reported backend id {backend.get('id')!r}"
            )
        for key, value in expected["runtime_assertions"].items():
            if backend.get(key) != value:
                raise CampaignValidationError(
                    f"job {job_id!r} backend assertion {key!r} must be {value!r}, "
                    f"received {backend.get(key)!r}"
                )
        cycle = _mapping(observations.get("cycle"), f"results.{job_id}.cycle")
        if dict(cycle) != expected["cycle"]:
            raise CampaignValidationError(
                f"job {job_id!r} cycle evidence does not match the algorithm contract"
            )
        run_contract = _mapping(observations.get("run_contract"), f"results.{job_id}.run_contract")
        if dict(run_contract) != expected["run_contract"]:
            raise CampaignValidationError(
                f"job {job_id!r} model, geometry, recipe, or workload differs from the manifest"
            )
        artifacts = _mapping(result.get("artifacts"), f"results.{job_id}.artifacts")
        for artifact in required_artifacts:
            _nonempty_string(artifacts.get(artifact), f"results.{job_id}.artifacts.{artifact}")


def main() -> int:
    """Run manifest and optional result validation from the command line.

    Returns:
        Zero after successful validation.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--list-jobs", action="store_true")
    parser.add_argument(
        "--commit-sha",
        help="Expected tested commit (defaults to the checked-out worktree HEAD).",
    )
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = load_mapping(manifest_path)
    jobs = validate_manifest(manifest, repo_root=REPO_ROOT)
    if args.list_jobs:
        for job in jobs:
            print(job["id"])
    if args.results is not None:
        results = load_mapping(args.results.resolve())
        validate_results(
            manifest,
            results,
            expected_manifest_sha256=manifest_sha256(manifest_path),
            expected_commit_sha=args.commit_sha or git_head(REPO_ROOT),
        )
        print(f"validated {len(jobs)} passing jobs for commit {results['commit_sha']}")
    elif not args.list_jobs:
        print(f"validated campaign {manifest['gate']['id']} with {len(jobs)} jobs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
