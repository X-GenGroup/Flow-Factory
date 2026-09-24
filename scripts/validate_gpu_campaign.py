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
from math import gcd, isfinite
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "config/gpu_validation/framework_upgrade.yaml"
REQUIRED_CORE_ALGORITHMS = {
    "grpo",
    "nft",
    "sft",
    "offline-dpo",
    "online-dpo",
    "tdm",
}
REQUIRED_OVERLAP_ALGORITHMS = {
    "grpo",
    "grpo-guard",
    "dppo",
    "nft",
    "awm",
    "crd",
    "dgpo",
    "online-dpo",
    "tdm-r1",
}
REQUIRED_ALGORITHMS = REQUIRED_CORE_ALGORITHMS | REQUIRED_OVERLAP_ALGORITHMS
REQUIRED_BACKENDS = {"ddp", "zero2", "fsdp2"}
REQUIRED_OPTIMIZER_ROLES = {
    "grpo": {"default"},
    "grpo-guard": {"default"},
    "dppo": {"default"},
    "nft": {"default"},
    "awm": {"default"},
    "crd": {"default"},
    "dgpo": {"default"},
    "sft": {"default"},
    "offline-dpo": {"default"},
    "online-dpo": {"default"},
    "tdm": {"generator", "fake"},
    "tdm-r1": {"generator", "fake", "surrogate"},
}
REQUIRED_OVERLAP_SAMPLERS = {"subgroup_tile", "global_batch", "global_tile", "rank_local"}
REQUIRED_OVERLAP_MODES = {"ready", "ordered"}
REQUIRED_OVERLAP_OBSERVATIONS = {"required", "observe_only"}
REQUIRED_REWARD_CARDINALITIES = {"single", "multi"}
REQUIRED_ADVANTAGE_AGGREGATIONS = {"sum", "gdpo"}
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


def _nonnegative_number(value: Any, location: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or value < 0
    ):
        raise CampaignValidationError(
            f"expected {location} to be a non-negative number, received {value!r}"
        )
    return float(value)


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


def _validate_reward_contracts(
    manifest: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    services = _mapping(manifest.get("reward_services"), "reward_services")
    profiles = _mapping(manifest.get("reward_profiles"), "reward_profiles")
    if not services or not profiles:
        raise CampaignValidationError("reward_services and reward_profiles must not be empty")

    for service_id, value in services.items():
        service = _mapping(value, f"reward_services.{service_id}")
        placement = _nonempty_string(
            service.get("placement"), f"reward_services.{service_id}.placement"
        )
        if placement not in {"remote_service", "in_process"}:
            raise CampaignValidationError(
                f"reward service {service_id!r} has unsupported placement {placement!r}"
            )
        _nonempty_string(service.get("transport"), f"reward_services.{service_id}.transport")
        if service.get("output_scope") != "pointwise":
            raise CampaignValidationError(
                f"reward service {service_id!r} must provide pointwise output"
            )
        if placement == "remote_service":
            _nonempty_string(
                service.get("endpoint_env"), f"reward_services.{service_id}.endpoint_env"
            )
            server = _mapping(service.get("server"), f"reward_services.{service_id}.server")
            gpu_count = _positive_int(
                server.get("gpu_count"), f"reward_services.{service_id}.server.gpu_count"
            )
            if "replica_count" in server or "tensor_parallel_size" in server:
                replicas = _positive_int(
                    server.get("replica_count"),
                    f"reward_services.{service_id}.server.replica_count",
                )
                tensor_parallel = _positive_int(
                    server.get("tensor_parallel_size"),
                    f"reward_services.{service_id}.server.tensor_parallel_size",
                )
                if replicas * tensor_parallel != gpu_count:
                    raise CampaignValidationError(
                        f"reward service {service_id!r} replica_count * tensor_parallel_size "
                        "must equal gpu_count"
                    )
            if "data_parallel_size" in server:
                data_parallel = _positive_int(
                    server.get("data_parallel_size"),
                    f"reward_services.{service_id}.server.data_parallel_size",
                )
                if data_parallel != gpu_count:
                    raise CampaignValidationError(
                        f"reward service {service_id!r} data_parallel_size must equal gpu_count"
                    )
        else:
            _nonempty_string(
                service.get("reward_model"), f"reward_services.{service_id}.reward_model"
            )

    for profile_id, value in profiles.items():
        profile = _mapping(value, f"reward_profiles.{profile_id}")
        deployment = _nonempty_string(
            profile.get("deployment_kind"),
            f"reward_profiles.{profile_id}.deployment_kind",
        )
        source_values = _sequence(
            profile.get("source_services"), f"reward_profiles.{profile_id}.source_services"
        )
        source_ids = [
            _nonempty_string(source, f"reward_profiles.{profile_id}.source_services[{index}]")
            for index, source in enumerate(source_values)
        ]
        if not source_ids or len(source_ids) != len(set(source_ids)):
            raise CampaignValidationError(
                f"reward profile {profile_id!r} must contain unique source services"
            )
        unknown = sorted(set(source_ids) - set(services))
        if unknown:
            raise CampaignValidationError(
                f"reward profile {profile_id!r} references unknown services {unknown}"
            )
        placements = {services[source_id]["placement"] for source_id in source_ids}
        if placements != {deployment}:
            raise CampaignValidationError(
                f"reward profile {profile_id!r} deployment {deployment!r} does not match "
                f"source placements {sorted(placements)}"
            )
        device = _nonempty_string(
            profile.get("client_device"), f"reward_profiles.{profile_id}.client_device"
        )
        if not isinstance(profile.get("async_reward"), bool):
            raise CampaignValidationError(
                f"reward_profiles.{profile_id}.async_reward must be a boolean"
            )
        _positive_int(profile.get("batch_size"), f"reward_profiles.{profile_id}.batch_size")
        _positive_int(
            profile.get("workers_per_source"),
            f"reward_profiles.{profile_id}.workers_per_source",
        )
        if profile["async_reward"] and (deployment != "remote_service" or device != "cpu"):
            raise CampaignValidationError(
                f"async reward profile {profile_id!r} must use remote pointwise services "
                "through CPU clients"
            )
    return services, profiles


def _expanded_reward_profile(
    profile_id: str,
    reward_profiles: Mapping[str, Any],
    reward_services: Mapping[str, Any],
) -> dict[str, Any]:
    profile = _mapping(reward_profiles[profile_id], f"reward_profiles.{profile_id}")
    source_ids = list(profile["source_services"])
    return {
        "id": profile_id,
        **dict(profile),
        "services": [
            {"id": source_id, **dict(reward_services[source_id])} for source_id in source_ids
        ],
    }


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
        expected_roles = REQUIRED_OPTIMIZER_ROLES[algorithm_id]
        if set(optimizer_steps) != expected_roles:
            raise CampaignValidationError(
                f"algorithm {algorithm_id!r} optimizer roles must be "
                f"{sorted(expected_roles)}, received {sorted(optimizer_steps)}"
            )
        for role, count in optimizer_steps.items():
            _positive_int(count, f"algorithms.{algorithm_id}.cycle.optimizer_steps.{role}")
    return algorithms


def _resolve_job_cycle(
    algorithm_id: str,
    algorithm: Mapping[str, Any],
    run: Mapping[str, Any],
    *,
    location: str,
) -> dict[str, Any]:
    """Resolve profile-specific optimizer counts without changing algorithm roles."""

    cycle = dict(_mapping(algorithm["cycle"], f"algorithms.{algorithm_id}.cycle"))
    default_steps = _mapping(
        cycle["optimizer_steps"],
        f"algorithms.{algorithm_id}.cycle.optimizer_steps",
    )
    override = run.get("optimizer_steps")
    if override is None:
        cycle["optimizer_steps"] = dict(default_steps)
        return cycle

    optimizer_steps = _mapping(override, f"{location}.optimizer_steps")
    if set(optimizer_steps) != set(default_steps):
        raise CampaignValidationError(
            f"{location}.optimizer_steps must preserve algorithm roles "
            f"{sorted(default_steps)}, received {sorted(optimizer_steps)}"
        )
    cycle["optimizer_steps"] = {
        role: _positive_int(count, f"{location}.optimizer_steps.{role}")
        for role, count in optimizer_steps.items()
    }
    return cycle


def _resolve_overlap_observation(
    run: Mapping[str, Any],
    workload: Mapping[str, Any],
    algorithm: Mapping[str, Any],
    *,
    location: str,
) -> str | None:
    """Resolve whether runtime evidence must observe physical reward/optimizer overlap."""

    configured = run.get("overlap_observation")
    if algorithm["feedback"] != "runtime_reward":
        if configured is not None:
            raise CampaignValidationError(
                f"{location}.overlap_observation is only valid for runtime-reward jobs"
            )
        return None
    if not workload.get("reward_optimization_overlap", False):
        if configured is not None:
            raise CampaignValidationError(
                f"{location}.overlap_observation cannot be set when overlap is disabled"
            )
        return "disabled"

    observation = configured or "required"
    if observation not in REQUIRED_OVERLAP_OBSERVATIONS:
        raise CampaignValidationError(
            f"{location}.overlap_observation must be one of "
            f"{sorted(REQUIRED_OVERLAP_OBSERVATIONS)}, received {observation!r}"
        )
    return observation


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
    elif sampler_type == "global_tile":
        groups_per_tile = global_batch_size // gcd(global_batch_size, group_size)
        if unique_samples % groups_per_tile:
            raise CampaignValidationError(
                f"global_tile workload {workload_id!r} requires U to be a multiple of "
                f"{groups_per_tile}"
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
        overlap_mode = workload.get("reward_optimization_overlap_mode")
        if overlap_mode not in REQUIRED_OVERLAP_MODES:
            raise CampaignValidationError(
                f"overlap workload {workload_id!r} must use one of "
                f"{sorted(REQUIRED_OVERLAP_MODES)}"
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
    if manifest.get("schema_version") != 2:
        raise CampaignValidationError(
            f"expected schema_version 2, received {manifest.get('schema_version')!r}"
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
    reward_services, reward_profiles = _validate_reward_contracts(manifest)

    backends = _validate_backend_contracts(manifest, repo_root)
    algorithms = _validate_algorithm_contracts(manifest)
    workloads = _mapping(manifest.get("workloads"), "workloads")
    profiles = _sequence(manifest.get("profiles"), "profiles")
    if not profiles:
        raise CampaignValidationError("profiles must not be empty")

    seen_profiles: set[str] = set()
    profile_records: dict[str, Mapping[str, Any]] = {}
    jobs: list[dict[str, Any]] = []
    pair_count = 0
    for profile_index, value in enumerate(profiles):
        profile = _mapping(value, f"profiles[{profile_index}]")
        profile_id = _nonempty_string(profile.get("id"), f"profiles[{profile_index}].id")
        if profile_id in seen_profiles:
            raise CampaignValidationError(f"duplicate profile id {profile_id!r}")
        seen_profiles.add(profile_id)
        profile_records[profile_id] = profile
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
            if algorithm_id not in REQUIRED_CORE_ALGORITHMS:
                raise CampaignValidationError(
                    f"core profile {profile_id!r} may only use algorithms "
                    f"{sorted(REQUIRED_CORE_ALGORITHMS)}, received {algorithm_id!r}"
                )
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
            cycle = _resolve_job_cycle(
                algorithm_id,
                algorithm,
                run,
                location=f"profiles.{profile_id}.runs.{algorithm_id}",
            )
            overlap_observation = _resolve_overlap_observation(
                run,
                workload,
                algorithm,
                location=f"profiles.{profile_id}.runs.{algorithm_id}",
            )
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
            reward_profile = None
            if algorithm["feedback"] == "runtime_reward":
                reward_profile_id = _nonempty_string(
                    run.get("reward_profile", profile.get("reward_profile")),
                    f"profiles.{profile_id}.reward_profile",
                )
                if reward_profile_id not in reward_profiles:
                    raise CampaignValidationError(
                        f"profile {profile_id!r} references unknown reward profile "
                        f"{reward_profile_id!r}"
                    )
                reward_profile_config = _mapping(
                    reward_profiles[reward_profile_id],
                    f"reward_profiles.{reward_profile_id}",
                )
                if workload.get("reward_optimization_overlap", False) and (
                    reward_profile_config.get("async_reward") is not True
                    or reward_profile_config.get("client_device") != "cpu"
                    or reward_profile_config.get("deployment_kind") != "remote_service"
                ):
                    raise CampaignValidationError(
                        f"overlap workload {workload_id!r} requires an async remote reward "
                        "profile with CPU clients"
                    )
                reward_profile = _expanded_reward_profile(
                    reward_profile_id, reward_profiles, reward_services
                )
            pair_count += 1
            for backend_id, backend_value in backends.items():
                backend = _mapping(backend_value, f"backends.{backend_id}")
                jobs.append(
                    {
                        "id": f"{profile_id}__{backend_id}__{algorithm_id}",
                        "suite": "core",
                        "profile": profile_id,
                        "model_type": profile["model_type"],
                        "algorithm": algorithm_id,
                        "backend": backend_id,
                        "world_size": world_size,
                        "runtime_assertions": dict(backend["runtime_assertions"]),
                        "cycle": cycle,
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
                            "reward_profile": reward_profile,
                            "overlap_observation": overlap_observation,
                            "advantage_aggregation": profile.get("advantage_aggregation"),
                            "offline_profile": run.get("offline_profile"),
                            "common_overrides": dict(common_overrides),
                        },
                    }
                )

    expected_pairs = _positive_int(
        gate.get("expected_core_algorithm_profile_pairs"),
        "gate.expected_core_algorithm_profile_pairs",
    )
    if pair_count != expected_pairs:
        raise CampaignValidationError(
            f"expected {expected_pairs} core algorithm/profile pairs, received {pair_count}"
        )
    expected_core_jobs = _positive_int(
        gate.get("expected_core_job_count"), "gate.expected_core_job_count"
    )
    if len(jobs) != expected_core_jobs:
        raise CampaignValidationError(
            f"expected {expected_core_jobs} core jobs, received {len(jobs)}"
        )

    core_jobs = {job["id"]: job for job in jobs}
    critical_paths = _mapping(manifest.get("overlap_critical_paths"), "overlap_critical_paths")
    if critical_paths.get("strategy") != "constrained_pairwise":
        raise CampaignValidationError(
            "overlap_critical_paths.strategy must be 'constrained_pairwise'"
        )
    minimum_work_units = _positive_int(
        critical_paths.get("minimum_work_units"),
        "overlap_critical_paths.minimum_work_units",
    )
    if minimum_work_units < 2:
        raise CampaignValidationError(
            "overlap_critical_paths.minimum_work_units must be at least 2"
        )
    reused_values = _sequence(
        critical_paths.get("reused_core_jobs"),
        "overlap_critical_paths.reused_core_jobs",
    )
    reused_ids = [
        _nonempty_string(value, f"overlap_critical_paths.reused_core_jobs[{index}]")
        for index, value in enumerate(reused_values)
    ]
    expected_reused = _positive_int(
        critical_paths.get("expected_reused_core_job_count"),
        "overlap_critical_paths.expected_reused_core_job_count",
    )
    if len(reused_ids) != expected_reused or len(reused_ids) != len(set(reused_ids)):
        raise CampaignValidationError(
            f"expected {expected_reused} unique reused overlap jobs, received {reused_ids}"
        )
    missing_reused = sorted(set(reused_ids) - set(core_jobs))
    if missing_reused:
        raise CampaignValidationError(
            f"overlap critical paths reference unknown core jobs {missing_reused}"
        )

    supplemental_values = _sequence(
        critical_paths.get("supplemental_jobs"),
        "overlap_critical_paths.supplemental_jobs",
    )
    expected_supplemental = _positive_int(
        gate.get("expected_overlap_supplemental_job_count"),
        "gate.expected_overlap_supplemental_job_count",
    )
    if len(supplemental_values) != expected_supplemental:
        raise CampaignValidationError(
            f"expected {expected_supplemental} overlap supplemental jobs, "
            f"received {len(supplemental_values)}"
        )

    supplemental_jobs: list[dict[str, Any]] = []
    for case_index, case_value in enumerate(supplemental_values):
        case = _mapping(case_value, f"overlap_critical_paths.supplemental_jobs[{case_index}]")
        case_id = _nonempty_string(
            case.get("id"), f"overlap_critical_paths.supplemental_jobs[{case_index}].id"
        )
        profile_id = _nonempty_string(case.get("profile"), f"{case_id}.profile")
        algorithm_id = _nonempty_string(case.get("algorithm"), f"{case_id}.algorithm")
        backend_id = _nonempty_string(case.get("backend"), f"{case_id}.backend")
        recipe = _nonempty_string(case.get("recipe"), f"{case_id}.recipe")
        workload_id = _nonempty_string(case.get("workload"), f"{case_id}.workload")
        reward_profile_id = _nonempty_string(
            case.get("reward_profile"), f"{case_id}.reward_profile"
        )
        if profile_id not in profile_records:
            raise CampaignValidationError(
                f"supplemental job {case_id!r} references unknown profile {profile_id!r}"
            )
        if algorithm_id not in REQUIRED_OVERLAP_ALGORITHMS or algorithm_id not in algorithms:
            raise CampaignValidationError(
                f"supplemental job {case_id!r} uses unsupported overlap algorithm "
                f"{algorithm_id!r}"
            )
        if backend_id not in backends:
            raise CampaignValidationError(
                f"supplemental job {case_id!r} references unknown backend {backend_id!r}"
            )
        if workload_id not in workloads:
            raise CampaignValidationError(
                f"supplemental job {case_id!r} references unknown workload {workload_id!r}"
            )
        if reward_profile_id not in reward_profiles:
            raise CampaignValidationError(
                f"supplemental job {case_id!r} references unknown reward profile "
                f"{reward_profile_id!r}"
            )
        if repo_root is not None and not (repo_root / recipe).is_file():
            raise CampaignValidationError(
                f"supplemental job {case_id!r} references missing recipe {recipe!r}"
            )
        profile = profile_records[profile_id]
        geometry = _mapping(profile["geometry"], f"profiles.{profile_id}.geometry")
        algorithm = _mapping(algorithms[algorithm_id], f"algorithms.{algorithm_id}")
        workload = _mapping(workloads[workload_id], f"workloads.{workload_id}")
        cycle = _resolve_job_cycle(
            algorithm_id,
            algorithm,
            case,
            location=f"overlap_critical_paths.supplemental_jobs[{case_index}]",
        )
        overlap_observation = _resolve_overlap_observation(
            case,
            workload,
            algorithm,
            location=f"overlap_critical_paths.supplemental_jobs[{case_index}]",
        )
        _validate_workload(
            workload_id,
            workload,
            algorithm=algorithm,
            world_size=world_size,
        )
        if algorithm_id == "online-dpo" and workload.get("sampler_type") != "rank_local":
            raise CampaignValidationError(
                f"online-dpo supplemental job {case_id!r} must use rank_local placement"
            )
        reward_profile_config = _mapping(
            reward_profiles[reward_profile_id], f"reward_profiles.{reward_profile_id}"
        )
        if (
            workload.get("reward_optimization_overlap") is not True
            or reward_profile_config.get("async_reward") is not True
            or reward_profile_config.get("client_device") != "cpu"
            or reward_profile_config.get("deployment_kind") != "remote_service"
        ):
            raise CampaignValidationError(
                f"overlap supplemental job {case_id!r} requires an enabled overlap workload "
                "and an async remote reward profile with CPU clients"
            )
        backend = _mapping(backends[backend_id], f"backends.{backend_id}")
        supplemental_jobs.append(
            {
                "id": case_id,
                "suite": "overlap_supplemental",
                "profile": profile_id,
                "model_type": profile["model_type"],
                "algorithm": algorithm_id,
                "backend": backend_id,
                "world_size": world_size,
                "runtime_assertions": dict(backend["runtime_assertions"]),
                "cycle": cycle,
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
                    "reward_profile": _expanded_reward_profile(
                        reward_profile_id, reward_profiles, reward_services
                    ),
                    "overlap_observation": overlap_observation,
                    "advantage_aggregation": case.get("advantage_aggregation"),
                    "offline_profile": None,
                    "common_overrides": dict(common_overrides),
                },
            }
        )

    jobs.extend(supplemental_jobs)
    critical_jobs = [core_jobs[job_id] for job_id in reused_ids] + supplemental_jobs
    overlap_jobs = [
        job
        for job in critical_jobs
        if job["run_contract"]["workload"].get("reward_optimization_overlap") is True
    ]
    required_overlap_jobs = [
        job for job in overlap_jobs if job["run_contract"]["overlap_observation"] == "required"
    ]
    boundary_jobs = [job for job in critical_jobs if job not in overlap_jobs]
    if not boundary_jobs or not any(
        job["run_contract"]["reward_profile"]["deployment_kind"] == "in_process"
        and job["run_contract"]["reward_profile"]["async_reward"] is False
        for job in boundary_jobs
    ):
        raise CampaignValidationError(
            "overlap critical paths must retain an in-process synchronous reward boundary job"
        )
    for job in overlap_jobs:
        reward_profile = job["run_contract"]["reward_profile"]
        if (
            reward_profile["deployment_kind"] != "remote_service"
            or reward_profile["async_reward"] is not True
            or reward_profile["client_device"] != "cpu"
        ):
            raise CampaignValidationError(
                f"critical overlap job {job['id']!r} does not use async remote CPU clients"
            )
        optimizer_steps = _mapping(
            job["cycle"]["optimizer_steps"],
            f"critical overlap job {job['id']!r}.cycle.optimizer_steps",
        )
        if sum(optimizer_steps.values()) < minimum_work_units:
            raise CampaignValidationError(
                f"critical overlap job {job['id']!r} must schedule at least "
                f"{minimum_work_units} optimizer work units"
            )

    declared_observations = {
        _nonempty_string(
            value,
            f"overlap_critical_paths.required_observation_policies[{index}]",
        )
        for index, value in enumerate(
            _sequence(
                critical_paths.get("required_observation_policies"),
                "overlap_critical_paths.required_observation_policies",
            )
        )
    }
    if declared_observations != REQUIRED_OVERLAP_OBSERVATIONS:
        raise CampaignValidationError(
            "overlap_critical_paths.required_observation_policies must be "
            f"{sorted(REQUIRED_OVERLAP_OBSERVATIONS)}, received "
            f"{sorted(declared_observations)}"
        )
    observed_observations = {job["run_contract"]["overlap_observation"] for job in overlap_jobs}
    if observed_observations != REQUIRED_OVERLAP_OBSERVATIONS:
        raise CampaignValidationError(
            "critical jobs must cover required and observe-only overlap evidence policies"
        )

    required_dimensions = {
        "required_overlap_algorithms": REQUIRED_OVERLAP_ALGORITHMS,
        "required_sampler_types": REQUIRED_OVERLAP_SAMPLERS,
        "required_scheduling_modes": REQUIRED_OVERLAP_MODES,
        "required_source_cardinalities": REQUIRED_REWARD_CARDINALITIES,
        "required_advantage_aggregations": REQUIRED_ADVANTAGE_AGGREGATIONS,
    }
    observed_dimensions = {
        "required_overlap_algorithms": {job["algorithm"] for job in required_overlap_jobs},
        "required_sampler_types": {
            job["run_contract"]["workload"]["sampler_type"] for job in required_overlap_jobs
        },
        "required_scheduling_modes": {
            job["run_contract"]["workload"]["reward_optimization_overlap_mode"]
            for job in required_overlap_jobs
        },
        "required_source_cardinalities": {
            (
                "single"
                if len(job["run_contract"]["reward_profile"]["source_services"]) == 1
                else "multi"
            )
            for job in required_overlap_jobs
        },
        "required_advantage_aggregations": {
            job["run_contract"]["advantage_aggregation"] for job in required_overlap_jobs
        },
    }
    for field, required in required_dimensions.items():
        declared = {
            _nonempty_string(value, f"overlap_critical_paths.{field}[{index}]")
            for index, value in enumerate(
                _sequence(critical_paths.get(field), f"overlap_critical_paths.{field}")
            )
        }
        if declared != required:
            raise CampaignValidationError(
                f"overlap_critical_paths.{field} must be {sorted(required)}, "
                f"received {sorted(declared)}"
            )
        if observed_dimensions[field] != required:
            raise CampaignValidationError(
                f"critical jobs cover {field}="
                f"{sorted(repr(value) for value in observed_dimensions[field])}, "
                f"expected {sorted(required)}"
            )
    if not any(
        job["run_contract"]["workload"]["per_device_batch_size"] > 1
        and job["run_contract"]["workload"].get("preserve_acquisition_microbatches") is True
        for job in overlap_jobs
    ):
        raise CampaignValidationError(
            "overlap critical paths must include a packed B>1 acquisition-manifest job"
        )

    expected_total = _positive_int(
        gate.get("expected_total_job_count"), "gate.expected_total_job_count"
    )
    if len(jobs) != expected_total:
        raise CampaignValidationError(f"expected {expected_total} total jobs, received {len(jobs)}")
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
    critical_paths = _mapping(manifest["overlap_critical_paths"], "overlap_critical_paths")
    minimum_work_units = _positive_int(
        critical_paths.get("minimum_work_units"),
        "overlap_critical_paths.minimum_work_units",
    )
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
    reward_artifact_values = _sequence(
        evidence.get("runtime_reward_artifacts"), "evidence.runtime_reward_artifacts"
    )
    runtime_reward_artifacts = [
        _nonempty_string(value, f"evidence.runtime_reward_artifacts[{index}]")
        for index, value in enumerate(reward_artifact_values)
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
        expected_contract = expected["run_contract"]
        if expected_contract["feedback"] != "runtime_reward":
            if observations.get("reward_runtime") is not None:
                raise CampaignValidationError(
                    f"reward-free job {job_id!r} must not report reward_runtime evidence"
                )
            continue

        for artifact in runtime_reward_artifacts:
            _nonempty_string(artifacts.get(artifact), f"results.{job_id}.artifacts.{artifact}")
        expected_profile = expected_contract["reward_profile"]
        reward_runtime = _mapping(
            observations.get("reward_runtime"), f"results.{job_id}.reward_runtime"
        )
        source_ids = list(expected_profile["source_services"])
        if reward_runtime.get("source_services") != source_ids:
            raise CampaignValidationError(
                f"job {job_id!r} did not exercise the expected reward sources {source_ids}"
            )
        if reward_runtime.get("deployment_kind") != expected_profile["deployment_kind"]:
            raise CampaignValidationError(
                f"job {job_id!r} reported the wrong reward deployment kind"
            )
        if reward_runtime.get("all_sources_ready") is not True:
            raise CampaignValidationError(
                f"job {job_id!r} must prove every reward source passed a real probe"
            )
        expected_topology = {}
        expected_endpoint_bindings = {}
        for service in expected_profile["services"]:
            source_id = service["id"]
            if service["placement"] == "remote_service":
                expected_topology[source_id] = dict(service["server"])
                expected_endpoint_bindings[source_id] = service["endpoint_env"]
            else:
                expected_topology[source_id] = {
                    "placement": "in_process",
                    "reward_model": service["reward_model"],
                }
        observed_topology = _mapping(
            reward_runtime.get("service_topology"),
            f"results.{job_id}.reward_runtime.service_topology",
        )
        if dict(observed_topology) != expected_topology:
            raise CampaignValidationError(
                f"job {job_id!r} did not prove the expected reward service topology"
            )
        endpoint_bindings = _mapping(
            reward_runtime.get("endpoint_env_bindings"),
            f"results.{job_id}.reward_runtime.endpoint_env_bindings",
        )
        if dict(endpoint_bindings) != expected_endpoint_bindings:
            raise CampaignValidationError(
                f"job {job_id!r} did not bind the expected reward endpoint aliases"
            )
        workload = expected_contract["workload"]
        expected_samples = workload["group_size"] * workload["unique_sample_num_per_epoch"]
        if reward_runtime.get("scored_samples") != expected_samples:
            raise CampaignValidationError(
                f"job {job_id!r} must score exactly {expected_samples} samples"
            )
        source_sample_counts = _mapping(
            reward_runtime.get("source_sample_counts"),
            f"results.{job_id}.reward_runtime.source_sample_counts",
        )
        if dict(source_sample_counts) != {source_id: expected_samples for source_id in source_ids}:
            raise CampaignValidationError(
                f"job {job_id!r} has incomplete per-source reward sample counts"
            )
        source_latency = _mapping(
            reward_runtime.get("source_latency_ms"),
            f"results.{job_id}.reward_runtime.source_latency_ms",
        )
        if set(source_latency) != set(source_ids):
            raise CampaignValidationError(
                f"job {job_id!r} must report latency for every reward source"
            )
        for source_id in source_ids:
            latency = _mapping(
                source_latency[source_id],
                f"results.{job_id}.reward_runtime.source_latency_ms.{source_id}",
            )
            p50 = _nonnegative_number(
                latency.get("p50"),
                f"results.{job_id}.reward_runtime.source_latency_ms.{source_id}.p50",
            )
            p95 = _nonnegative_number(
                latency.get("p95"),
                f"results.{job_id}.reward_runtime.source_latency_ms.{source_id}.p95",
            )
            maximum = _nonnegative_number(
                latency.get("max"),
                f"results.{job_id}.reward_runtime.source_latency_ms.{source_id}.max",
            )
            if not p50 <= p95 <= maximum:
                raise CampaignValidationError(
                    f"job {job_id!r} has invalid latency quantiles for {source_id!r}"
                )

        expected_overlap = workload.get("reward_optimization_overlap", False)
        reward_overlap = _mapping(
            observations.get("reward_overlap"), f"results.{job_id}.reward_overlap"
        )
        if reward_overlap.get("enabled") is not expected_overlap:
            raise CampaignValidationError(
                f"job {job_id!r} reported the wrong overlap enabled state"
            )
        if not expected_overlap:
            continue
        overlap_observation = expected_contract["overlap_observation"]
        if reward_overlap.get("mode") != workload["reward_optimization_overlap_mode"]:
            raise CampaignValidationError(f"job {job_id!r} reported the wrong overlap mode")
        work_units = _positive_int(
            reward_overlap.get("work_units"), f"results.{job_id}.reward_overlap.work_units"
        )
        if work_units < minimum_work_units:
            raise CampaignValidationError(
                f"job {job_id!r} must exercise at least {minimum_work_units} "
                "independently ready work units"
            )
        _positive_int(
            reward_overlap.get("poll_count"), f"results.{job_id}.reward_overlap.poll_count"
        )
        overlap_seconds = _nonnegative_number(
            reward_overlap.get("optimization_overlap_seconds"),
            f"results.{job_id}.reward_overlap.optimization_overlap_seconds",
        )
        started_while_pending = reward_overlap.get("optimization_started_while_reward_pending")
        if not isinstance(started_while_pending, bool):
            raise CampaignValidationError(
                f"job {job_id!r} must report whether optimization started while reward "
                "work was pending"
            )
        if overlap_observation == "required" and overlap_seconds == 0:
            raise CampaignValidationError(
                f"job {job_id!r} did not prove reward/optimization concurrency"
            )
        _nonnegative_number(
            reward_overlap.get("reward_wait_seconds"),
            f"results.{job_id}.reward_overlap.reward_wait_seconds",
        )
        _nonnegative_number(
            reward_overlap.get("out_of_order_work_units"),
            f"results.{job_id}.reward_overlap.out_of_order_work_units",
        )
        if overlap_observation == "required" and started_while_pending is not True:
            raise CampaignValidationError(
                f"job {job_id!r} must start optimization while reward work is still pending"
            )
        if overlap_observation == "observe_only" and started_while_pending != (overlap_seconds > 0):
            raise CampaignValidationError(
                f"job {job_id!r} reported inconsistent observe-only overlap timing"
            )


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
