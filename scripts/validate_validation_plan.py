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

"""Check an explicitly selected validation plan and its evidence; never launch tests.

This checks declared coverage and recorded identities. Reviewers still determine whether the
obligations cover the diff and whether logs support the recorded observations.
"""

import argparse
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
AXES = ("dp_replicate", "dp_shard", "tp", "cp", "ep")
TOPOLOGY_KEYS = {"nodes", "ranks_per_node", *AXES}
BACKENDS = {"cpu", "gloo", "nccl", "cuda", "ddp", "zero2", "fsdp2", "tp"}
ORACLES = {"assertions", "numerical", "overlap_order", "overlap_timing"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def text(value: Any, where: str) -> None:
    require(isinstance(value, str) and bool(value.strip()), f"{where}: expected nonempty text")


def mapping(value: Any, where: str) -> None:
    require(isinstance(value, dict), f"{where}: expected mapping")


def strings(value: Any, where: str) -> None:
    require(isinstance(value, list) and bool(value), f"{where}: expected nonempty list")
    for item in value:
        text(item, where)
    require(len(set(value)) == len(value), f"{where}: duplicate entries")


def digest(value: Any, length: int, where: str) -> None:
    require(
        isinstance(value, str) and re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is not None,
        f"{where}: expected {length}-character lowercase hex identity",
    )


def artifacts(value: Any, where: str) -> None:
    mapping(value, where)
    require(bool(value), f"{where}: include test and config identities")
    for path, sha in value.items():
        text(path, where)
        require(
            not PurePosixPath(path).is_absolute() and ".." not in PurePosixPath(path).parts,
            f"{where}: expected relative artifact path, got {path!r}",
        )
        digest(sha, 64, f"{where}.{path}")


def indexed(items: Any, where: str, *, allow_empty: bool = False) -> dict:
    require(isinstance(items, list) and (bool(items) or allow_empty), f"{where}: expected list")
    result = {}
    for item in items:
        mapping(item, where)
        name = item.get("id")
        text(name, f"{where}.id")
        require(name not in result, f"{where}: duplicate id {name!r}")
        result[name] = item
    return result


def validate_topology(case: dict) -> None:
    name, backend, qa = case["id"], case.get("backend"), case.get("qa")
    require(
        isinstance(backend, str) and backend in BACKENDS, f"{name}: unknown backend {backend!r}"
    )
    require(
        isinstance(qa, str) and qa in {"Q0", "Q1", "Q2", "Q3"}, f"{name}: unknown QA level {qa!r}"
    )
    topo = case.get("topology")
    mapping(topo, f"{name}.topology")
    require(set(topo) == TOPOLOGY_KEYS, f"{name}: topology keys must be {sorted(TOPOLOGY_KEYS)}")
    require(
        all(type(value) is int and value >= 1 for value in topo.values()),
        f"{name}: topology dimensions must be positive integers",
    )
    world = topo["nodes"] * topo["ranks_per_node"]
    require(
        world == math.prod(topo[axis] for axis in AXES), f"{name}: mesh does not match placement"
    )
    execution_scope = case.get("execution_scope", "runtime")
    require(
        isinstance(execution_scope, str)
        and execution_scope
        in {"runtime", "topology_only", "model_probe", "feedback_probe", "attention_probe"},
        f"{name}: unknown execution_scope",
    )
    if execution_scope == "topology_only":
        require(
            backend in {"gloo", "nccl"},
            f"{name}: topology_only requires a raw Gloo or NCCL process group",
        )
        require(topo["ep"] == 1, f"{name}: topology_only does not define EP ownership")
        require(qa in {"Q0", "Q2"}, f"{name}: topology_only cannot qualify a real model")
        oracle = case.get("oracle", {})
        require(
            isinstance(oracle, dict)
            and oracle.get("kind") == "assertions"
            and oracle.get("reward") == "none",
            f"{name}: topology_only requires assertions without a reward/model execution claim",
        )
    elif execution_scope == "feedback_probe":
        require(
            backend == "gloo" and qa == "Q0" and topo["ep"] == 1 and topo["tp"] * topo["cp"] >= 2,
            f"{name}: feedback_probe requires CPU/Gloo model-parallel peers without EP",
        )
        oracle = case.get("oracle", {})
        require(
            isinstance(oracle, dict)
            and oracle.get("kind") == "assertions"
            and oracle.get("reward") == "varying",
            f"{name}: feedback_probe requires varying reward assertions, not a model update claim",
        )
    elif execution_scope == "attention_probe":
        require(
            backend == "nccl"
            and qa == "Q2"
            and topo["cp"] == 2
            and topo["dp_shard"] == topo["dp_replicate"] == topo["tp"] == topo["ep"] == 1
            and case.get("execution_capability") == "native-ring-bf16-attention-v1",
            f"{name}: attention_probe requires the bounded two-rank native Ring primitive",
        )
        oracle = case.get("oracle", {})
        require(
            isinstance(oracle, dict)
            and oracle.get("kind") == "numerical"
            and oracle.get("reward") == "none",
            f"{name}: attention_probe requires numerical derivatives without model updates or rewards",
        )
    elif execution_scope == "model_probe":
        require(
            backend == "fsdp2" and qa == "Q2",
            f"{name}: model_probe requires bounded FSDP2 Q2 evidence",
        )
        require(
            topo["cp"] >= 2 and topo["ep"] == topo["dp_replicate"] == 1,
            f"{name}: model_probe requires a CP x FSDP-shard mesh without EP/replication",
        )
        if topo["tp"] > 1:
            require(
                case.get("execution_capability")
                in {
                    "wan-tp-ulysses-fp32-probe-v1",
                    "ltx2-joint-av-tp-ulysses-fp32-probe-v1",
                    "wan-i2v-tp-ulysses-fp32-probe-v1",
                    "wan-remaining-tp-ulysses-fp32-probe-v1",
                    "qwen-image-tp-ulysses-fp32-probe-v1",
                    "flux-kontext-tp-cp-fp32-probe-v1",
                    "flux-dual-sp-cp-fp32-probe-v1",
                    "flux2-packed-tp-sp-cp-fp32-probe-v1",
                    "qwen-edit-tp-ulysses-fp32-probe-v1",
                    "z-image-tp-ulysses-fp32-probe-v1",
                    "sd35-tp-ulysses-fp32-probe-v1",
                    "qwen21-prefix-tp-ulysses-fp32-probe-v1",
                },
                f"{name}: TP+CP model_probe requires its declared FP32 capability",
            )
        if case.get("execution_capability") == "wan-i2v-tp-ulysses-fp32-probe-v1":
            require(
                topo["tp"] == topo["cp"] == topo["dp_shard"] == 2,
                f"{name}: Wan I2V reference evidence requires DP2 TP2 CP2",
            )
        if case.get("execution_capability") == "wan-remaining-tp-ulysses-fp32-probe-v1":
            require(
                topo["tp"] == topo["cp"] == topo["dp_shard"] == 2,
                f"{name}: Wan remaining reference evidence requires DP2 TP2 CP2",
            )
        if case.get("execution_capability") == "qwen-image-tp-ulysses-fp32-probe-v1":
            require(
                topo["tp"] == topo["cp"] == topo["dp_shard"] == 2,
                f"{name}: Qwen reference evidence requires DP2 TP2 CP2",
            )
        if case.get("execution_capability") in {
            "flux-kontext-tp-cp-fp32-probe-v1",
            "flux-dual-sp-cp-fp32-probe-v1",
            "flux2-packed-tp-sp-cp-fp32-probe-v1",
        }:
            require(
                topo["tp"] == topo["cp"] == topo["dp_shard"] == 2,
                (
                    f"{name}: Flux Kontext reference evidence requires DP2 TP2 CP2"
                    if case.get("execution_capability") == "flux-kontext-tp-cp-fp32-probe-v1"
                    else f"{name}: Flux SP reference evidence requires DP2 TP2 CP2"
                ),
            )
        if case.get("execution_capability") in {
            "qwen-edit-tp-ulysses-fp32-probe-v1",
            "qwen21-prefix-tp-ulysses-fp32-probe-v1",
        }:
            require(
                topo["tp"] == topo["cp"] == topo["dp_shard"] == 2,
                f"{name}: Qwen prefix/edit reference evidence requires DP2 TP2 CP2",
            )
        if case.get("execution_capability") == "ltx2-joint-av-tp-ulysses-fp32-probe-v1":
            require(
                topo["tp"] == topo["cp"] == topo["dp_shard"] == 2,
                f"{name}: LTX2 reference evidence requires DP2 TP2 CP2",
            )
        if case.get("execution_capability") == "sd35-tp-ulysses-fp32-probe-v1":
            require(
                topo["tp"] == topo["cp"] == topo["dp_shard"] == 2,
                f"{name}: SD3.5 reference evidence requires DP2 TP2 CP2",
            )
        if case.get("execution_capability") == "z-image-tp-ulysses-fp32-probe-v1":
            require(
                topo["tp"] == topo["cp"] == topo["dp_shard"] == 2,
                f"{name}: Z-Image reference evidence requires DP2 TP2 CP2",
            )
        oracle = case.get("oracle", {})
        require(
            isinstance(oracle, dict)
            and oracle.get("kind") == "numerical"
            and oracle.get("reward") == "none",
            f"{name}: model_probe requires a numerical oracle without reward execution",
        )
    else:
        require(
            topo["ep"] == 1, f"{name}: CP/EP execution requires a declared supported capability"
        )
        if topo["cp"] > 1:
            require(
                case.get("execution_capability")
                in {
                    "wan-sft-ulysses-bf16-v1",
                    "wan-sft-tp-ulysses-bf16-fp32-sdpa-v1",
                    "wan-cp-state-v1",
                    "wan-offline-dpo-cp-v1",
                    "flux-offline-dpo-cp-v1",
                    "wan-grpo-cp-v1",
                    "wan-i2v_dual-grpo-cp-v1",
                    "wan-i2v_expanded-grpo-cp-v1",
                    "wan-flf2v-grpo-cp-v1",
                    "ltx20_t2av-grpo-cp-v1",
                    "ltx23_i2av-grpo-cp-v1",
                    "wan-dmd2-cp-v1",
                    "wan-nft-cp-v1",
                    "wan-dgpo-cp-v1",
                    "wan-opd-cp-v1",
                    "wan-tdm-cp-v1",
                    "wan-i2v-tdm-cp-v1",
                    "wan-tdm-r1-cp-v1",
                    "wan-cp-evaluation-v1",
                    "flux-grpo-guard-cp-v1",
                    "flux-kontext-grpo-guard-cp-v1",
                    "flux2-grpo-guard-cp-v1",
                    "flux2-klein-dppo-cp-v1",
                    "qwen-image-edit-plus-grpo-guard-cp-v1",
                    "z-image-grpo-guard-cp-v1",
                    "sd3-5-dppo-cp-v1",
                    "qwen-image-2.1-grpo-guard-cp-v1",
                    "flux-dppo-cp-v1",
                    "flux-nft-cp-v1",
                    "flux-dpo-cp-v1",
                    "qwen-image-grpo-guard-cp-v1",
                    "qwen-image-awm-cp-v1",
                    "qwen-image-crd-cp-v1",
                },
                f"{name}: CP/EP execution requires a declared supported capability",
            )
            require(
                backend == "fsdp2" and qa == "Q2" and topo["dp_replicate"] == 1,
                f"{name}: CP requires FSDP2 Q2 without replication",
            )
            if case["execution_capability"] == "wan-sft-ulysses-bf16-v1":
                require(topo["tp"] == 1, f"{name}: CP-only capability runs without TP/replication")
            elif case["execution_capability"] == "wan-sft-tp-ulysses-bf16-fp32-sdpa-v1":
                require(topo["tp"] >= 2, f"{name}: composed capability requires TP>=2")
            elif case["execution_capability"] in {"ltx20_t2av-grpo-cp-v1", "ltx23_i2av-grpo-cp-v1"}:
                require(
                    topo["tp"] == topo["cp"] == topo["dp_shard"] == 2,
                    f"{name}: LTX2 lifecycle evidence requires DP2 TP2 CP2",
                )
            elif case["execution_capability"] == "sd3-5-dppo-cp-v1":
                require(
                    topo["tp"] == topo["cp"] == topo["dp_shard"] == 2,
                    f"{name}: SD3.5 lifecycle evidence requires DP2 TP2 CP2",
                )
            elif case["execution_capability"] == "z-image-grpo-guard-cp-v1":
                require(
                    topo["tp"] == topo["cp"] == topo["dp_shard"] == 2,
                    f"{name}: Z-Image lifecycle evidence requires DP2 TP2 CP2",
                )
            elif case["execution_capability"] == "wan-i2v-tdm-cp-v1":
                require(
                    topo["tp"] == topo["cp"] == topo["dp_shard"] == 2,
                    f"{name}: Wan I2V TDM evidence requires DP2 TP2 CP2",
                )
            elif case["execution_capability"] in {
                "wan-i2v_dual-grpo-cp-v1",
                "wan-i2v_expanded-grpo-cp-v1",
                "wan-flf2v-grpo-cp-v1",
            }:
                require(
                    topo["tp"] == topo["cp"] == topo["dp_shard"] == 2,
                    f"{name}: Wan remaining lifecycle evidence requires DP2 TP2 CP2",
                )
            elif case["execution_capability"] in {
                "qwen-image-edit-plus-grpo-guard-cp-v1",
                "qwen-image-2.1-grpo-guard-cp-v1",
            }:
                require(
                    topo["tp"] == topo["cp"] == topo["dp_shard"] == 2,
                    f"{name}: Qwen prefix/edit Guard evidence requires DP2 TP2 CP2",
                )
            elif case["execution_capability"] == "flux-kontext-grpo-guard-cp-v1":
                require(
                    topo["tp"] == topo["cp"] == topo["dp_shard"] == 2,
                    f"{name}: Flux Kontext lifecycle evidence requires DP2 TP2 CP2",
                )
            elif case["execution_capability"] in {
                "flux2-grpo-guard-cp-v1",
                "flux2-klein-dppo-cp-v1",
            }:
                require(
                    topo["tp"] == topo["cp"] == topo["dp_shard"] == 2,
                    f"{name}: Flux2 lifecycle evidence requires DP2 TP2 CP2",
                )
            elif case["execution_capability"] in {
                "flux-grpo-guard-cp-v1",
                "flux-dppo-cp-v1",
            }:
                require(
                    topo["tp"] in (1, 2), f"{name}: Flux coupled CP evidence requires TP=1 or 2"
                )
            elif case["execution_capability"] in {
                "flux-offline-dpo-cp-v1",
                "flux-nft-cp-v1",
                "flux-dpo-cp-v1",
            }:
                require(topo["tp"] == 1, f"{name}: Flux CP evidence requires TP=1")
            elif case["execution_capability"] == "qwen-image-grpo-guard-cp-v1":
                require(topo["tp"] in (1, 2), f"{name}: Qwen Guard CP evidence requires TP=1 or 2")
            elif case["execution_capability"] in {
                "qwen-image-awm-cp-v1",
                "qwen-image-crd-cp-v1",
            }:
                require(topo["tp"] == 1, f"{name}: Qwen CP evidence requires TP=1")
            oracle = case.get("oracle", {})
            expected_reward = (
                "varying"
                if case["execution_capability"]
                in {
                    "wan-grpo-cp-v1",
                    "wan-i2v_dual-grpo-cp-v1",
                    "wan-i2v_expanded-grpo-cp-v1",
                    "wan-flf2v-grpo-cp-v1",
                    "ltx20_t2av-grpo-cp-v1",
                    "ltx23_i2av-grpo-cp-v1",
                    "wan-cp-evaluation-v1",
                    "wan-nft-cp-v1",
                    "wan-dgpo-cp-v1",
                    "wan-tdm-r1-cp-v1",
                    "flux-grpo-guard-cp-v1",
                    "flux-kontext-grpo-guard-cp-v1",
                    "flux2-grpo-guard-cp-v1",
                    "flux2-klein-dppo-cp-v1",
                    "qwen-image-edit-plus-grpo-guard-cp-v1",
                    "z-image-grpo-guard-cp-v1",
                    "sd3-5-dppo-cp-v1",
                    "qwen-image-2.1-grpo-guard-cp-v1",
                    "flux-dppo-cp-v1",
                    "flux-nft-cp-v1",
                    "flux-dpo-cp-v1",
                    "qwen-image-grpo-guard-cp-v1",
                    "qwen-image-awm-cp-v1",
                    "qwen-image-crd-cp-v1",
                }
                else "none"
            )
            require(
                isinstance(oracle, dict)
                and oracle.get("kind") == "numerical"
                and oracle.get("reward") == expected_reward,
                f"{name}: CP requires a numerical oracle with reward={expected_reward}",
            )
        require(backend != "nccl", f"{name}: raw NCCL requires execution_scope: topology_only")
    if backend in {"cpu", "gloo"}:
        require(qa == "Q0", f"{name}: CPU/Gloo evidence is Q0")
    else:
        require(qa != "Q0", f"{name}: GPU evidence cannot be Q0")
    if qa == "Q1":
        require(backend == "cuda" and world == 1, f"{name}: Q1 requires one CUDA rank")
    if qa == "Q2":
        require(world >= 2, f"{name}: Q2 requires a nontrivial distributed group")
    if backend in {"cpu", "cuda"}:
        require(world == 1, f"{name}: local backend requires one rank")
    if backend == "ddp":
        require(
            topo["dp_replicate"] >= 2 and topo["dp_shard"] == topo["tp"] == 1,
            f"{name}: DDP requires replicated data ranks and no TP/sharding",
        )
    if backend == "fsdp2":
        require(topo["dp_shard"] >= 2, f"{name}: sharded training requires dp_shard >= 2")
    if backend == "zero2":
        require(
            topo["dp_replicate"] >= 2 and topo["tp"] == topo["dp_shard"] == 1,
            f"{name}: ZeRO-2 requires replicated data ranks; TP/model sharding is unsupported",
        )
    if backend == "tp":
        require(
            topo["tp"] >= 2 and topo["dp_replicate"] == topo["dp_shard"] == 1,
            f"{name}: TP primitive requires only a TP group; production training uses FSDP2",
        )


def validate_plan(plan: dict, *, require_coverage: bool = True) -> None:
    mapping(plan, "plan")
    require(
        type(plan.get("schema_version")) is int and plan["schema_version"] == 1,
        "plan: unsupported schema_version",
    )
    for field in ("base_commit", "tested_commit"):
        digest(plan.get(field), 40, field)
    require(
        isinstance(plan.get("risk"), str) and plan["risk"] in {"R0", "R1", "R2", "R3", "R4"},
        "plan: unknown risk",
    )
    strings(plan.get("owners"), "owners")
    strings(plan.get("facets"), "facets")
    routes = yaml.safe_load((ROOT / ".agents/harness/test_routes.yaml").read_text())
    require(set(plan["facets"]) <= set(routes["facets"]), "plan: unknown risk facet")
    artifacts(plan.get("artifacts"), "artifacts")
    obligations = indexed(plan.get("obligations"), "obligations")
    for name, obligation in obligations.items():
        text(obligation.get("description"), f"obligation {name}")
        require(type(obligation.get("required")) is bool, f"{name}: required must be boolean")
    required = {name for name, item in obligations.items() if item["required"]}
    require(bool(required), "plan: declare at least one required obligation")
    covered = set()
    for name, case in indexed(plan.get("cases"), "cases").items():
        require(type(case.get("selected")) is bool, f"{name}: selected must be boolean")
        strings(case.get("covers"), f"{name}.covers")
        require(set(case["covers"]) <= set(obligations), f"{name}: unknown obligation")
        for field in ("reason", "fixture"):
            text(case.get(field), f"{name}.{field}")
        command = case.get("command")
        require(
            isinstance(command, list) and bool(command), f"{name}: command must be an argv list"
        )
        for arg in command:
            text(arg, f"{name}.command")
        validate_topology(case)
        oracle = case.get("oracle")
        mapping(oracle, f"{name}.oracle")
        require(
            isinstance(oracle.get("kind"), str) and oracle["kind"] in ORACLES,
            f"{name}: unknown oracle kind",
        )
        text(oracle.get("description"), f"{name}.oracle.description")
        require(
            isinstance(oracle.get("reward"), str)
            and oracle["reward"] in {"none", "constant", "varying", "real"},
            f"{name}: unknown reward mode",
        )
        require(
            not (
                oracle["reward"] == "constant" and oracle["kind"] in {"numerical", "overlap_timing"}
            ),
            f"{name}: constant rewards cannot prove a learning update or measured reward performance",
        )
        if case["selected"]:
            covered.update(case["covers"])
    if require_coverage:
        require(
            required <= covered, f"uncovered required obligations: {sorted(required - covered)}"
        )


def plan_sha256(plan: dict) -> str:
    """Hash the parsed plan, independent of YAML formatting."""
    return hashlib.sha256(
        json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_overlap(observations: dict, name: str) -> None:
    events = observations.get("events")
    require(isinstance(events, list) and bool(events), f"{name}: overlap needs ordered events")
    states, overlapped = {}, False
    phases = ["submit", "ready", "optimize_start", "optimize_end"]
    for event in events:
        mapping(event, f"{name}.event")
        unit, phase = event.get("unit"), event.get("event")
        text(unit, f"{name}.event.unit")
        require(phase in phases, f"{name}: unknown overlap event")
        index = phases.index(phase)
        require(states.get(unit, -1) + 1 == index, f"{name}: invalid event order for {unit}")
        if phase == "optimize_start" and any(value == 0 for value in states.values()):
            overlapped = True
        states[unit] = index
    require(
        len(states) >= 2 and all(value == 3 for value in states.values()),
        f"{name}: overlap needs at least two completed work units and no pending work",
    )
    require(overlapped, f"{name}: optimization never started while another reward was pending")


def validate_observations(case: dict, result: dict) -> None:
    name = case["id"]
    observed = result.get("observations")
    mapping(observed, f"{name}.observations")
    if case.get("execution_capability") is not None:
        require(
            observed.get("execution_capability") == case["execution_capability"],
            f"{name}: observed execution_capability disagrees with plan",
        )
    require(
        type(observed.get("exit_code")) is int and observed["exit_code"] == 0,
        f"{name}: passing evidence needs exit_code 0",
    )
    oracle = case["oracle"]
    if oracle["kind"] == "numerical":
        require(
            observed.get("elementwise_comparison") is True
            and observed.get("nonzero_gradients") is True,
            f"{name}: numerical evidence needs elementwise comparison and nonzero gradients",
        )
        if oracle["reward"] != "none":
            require(observed.get("varying_rewards") is True, f"{name}: missing reward variation")
    if oracle["kind"].startswith("overlap_"):
        validate_overlap(observed, name)
    if oracle["kind"] == "overlap_timing":
        require(
            observed.get("artificial_delay") is False,
            f"{name}: controlled delays do not measure hidden reward latency",
        )
        for field in ("timing_method", "workload"):
            text(observed.get(field), f"{name}.{field}")
        seconds = observed.get("overlap_seconds")
        require(
            type(seconds) in (float, int) and math.isfinite(seconds) and seconds > 0,
            f"{name}: expected positive measured overlap_seconds",
        )


def validate_runtime(case: dict, result: dict) -> None:
    """Require observed plugin identity and completion of every rank, not a backend label."""
    name, backend = case["id"], case["backend"]
    runtime = result.get("runtime")
    mapping(runtime, f"{name}.runtime")
    world = case["topology"]["nodes"] * case["topology"]["ranks_per_node"]
    require(
        type(runtime.get("world_size")) is int and runtime["world_size"] == world,
        f"{name}: observed world_size disagrees with plan",
    )
    ranks = runtime.get("ranks_completed")
    require(
        isinstance(ranks, list)
        and all(type(rank) is int for rank in ranks)
        and sorted(ranks) == list(range(world)),
        f"{name}: not every rank completed",
    )
    expected = {
        "cpu": {"device": "cpu", "distributed_type": "NO"},
        "cuda": {"device": "cuda", "distributed_type": "NO"},
        "gloo": {
            "device": "cpu",
            "distributed_type": "PROCESS_GROUP",
            "process_group_backend": "gloo",
        },
        "tp": {
            "device": "cuda",
            "distributed_type": "PROCESS_GROUP",
            "process_group_backend": "nccl",
        },
        "nccl": {
            "device": "cuda",
            "distributed_type": "PROCESS_GROUP",
            "process_group_backend": "nccl",
        },
        "ddp": {"device": "cuda", "distributed_type": "MULTI_GPU"},
        "zero2": {"device": "cuda", "distributed_type": "DEEPSPEED", "zero_stage": 2},
        "fsdp2": {"device": "cuda", "distributed_type": "FSDP", "fsdp_version": 2},
    }[backend]
    for field, value in expected.items():
        require(
            type(runtime.get(field)) is type(value) and runtime[field] == value,
            f"{name}: observed runtime {field} does not establish {backend}",
        )


def validate_results(
    plan: dict, results: dict, *, expected_commit: str, require_coverage: bool = True
) -> dict:
    """Return all non-passing selections explicitly; malformed or stale evidence raises."""
    validate_plan(plan, require_coverage=require_coverage)
    digest(expected_commit, 40, "expected commit")
    require(plan["tested_commit"] == expected_commit, "plan: stale tested commit")
    mapping(results, "results")
    require(
        type(results.get("schema_version")) is int and results["schema_version"] == 1,
        "results: unsupported schema_version",
    )
    require(results.get("tested_commit") == plan["tested_commit"], "results: stale tested commit")
    require(results.get("plan_sha256") == plan_sha256(plan), "results: stale plan digest")
    require(results.get("artifacts") == plan["artifacts"], "results: stale test/config artifacts")
    environment = results.get("environment")
    mapping(environment, "environment")
    for field in ("python", "hardware"):
        text(environment.get(field), f"environment.{field}")
    packages = environment.get("packages")
    mapping(packages, "environment.packages")
    require(bool(packages), "environment: record dependency versions")
    for package, version in packages.items():
        text(package, "package name")
        text(version, f"package {package}")
    cases = indexed(plan["cases"], "cases")
    actual = indexed(results.get("cases"), "results.cases", allow_empty=True)
    require(set(actual) <= set(cases), f"unknown result IDs: {sorted(set(actual) - set(cases))}")
    statuses = {}
    for name, case in cases.items():
        result = actual.get(name)
        if result is None:
            statuses[name] = "blocked" if case["selected"] else "not_selected"
            continue
        status = result.get("status")
        require(
            isinstance(status, str) and status in {"pass", "fail", "blocked", "not_selected"},
            f"{name}: unknown status",
        )
        require(
            (status == "not_selected") == (not case["selected"]),
            f"{name}: result status disagrees with selected plan",
        )
        statuses[name] = status
        if status != "pass":
            text(result.get("reason"), f"{name}.reason")
            continue
        for field in ("backend", "topology", "command"):
            require(
                result.get(field) == case[field], f"{name}: observed {field} disagrees with plan"
            )
        case_environment = result.get("environment", environment)
        mapping(case_environment, f"{name}.environment")
        for field in ("python", "hardware"):
            text(case_environment.get(field), f"{name}.environment.{field}")
        observed_packages = case_environment.get("packages")
        mapping(observed_packages, f"{name}.environment.packages")
        for package, version in observed_packages.items():
            text(package, f"{name}.package name")
            text(version, f"{name}.package {package}")
        normalize = lambda value: re.sub(r"[-_.]+", "-", value).lower()
        required_packages = case.get("resources", {}).get("packages", [])
        observed_names = {normalize(package) for package in observed_packages}
        missing_packages = [
            package for package in required_packages if normalize(package) not in observed_names
        ]
        require(not missing_packages, f"{name}: missing dependency identities {missing_packages}")
        require(
            result.get("execution_scope", "runtime") == case.get("execution_scope", "runtime"),
            f"{name}: observed execution_scope disagrees with plan",
        )
        validate_topology({**case, "topology": result["topology"]})
        validate_runtime(case, result)
        artifacts(result.get("evidence"), f"{name}.evidence")
        validate_observations(case, result)
    pending = {
        name: status
        for name, status in statuses.items()
        if cases[name]["selected"] and status != "pass"
    }
    covered = {
        obligation
        for name, status in statuses.items()
        if status == "pass"
        for obligation in cases[name]["covers"]
    }
    missing = sorted(
        item["id"] for item in plan["obligations"] if item["required"] and item["id"] not in covered
    )
    return {
        "status": (
            "fail" if "fail" in pending.values() else "blocked" if pending or missing else "pass"
        ),
        "cases": statuses,
        "uncovered_obligations": missing,
    }


def verify_source(plan: dict, root: Path) -> None:
    """Verify immutable source and test/config blobs, independently of the result's echo."""

    def git(*args: str) -> bytes:
        return subprocess.check_output(["git", *args], cwd=root, stderr=subprocess.PIPE)

    for field in ("base_commit", "tested_commit"):
        git("cat-file", "-e", f"{plan[field]}^{{commit}}")
    for path, expected in plan["artifacts"].items():
        blob = f"{plan['tested_commit']}:{path}"
        size = int(git("cat-file", "-s", blob))
        require(size <= 12 * 1024 * 1024, f"artifact too large for source verification: {path}")
        actual = hashlib.sha256(git("cat-file", "blob", blob)).hexdigest()
        require(actual == expected, f"artifact differs from tested commit: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path, help="Manually selected YAML or JSON plan")
    parser.add_argument("--results", type=Path, help="Recorded YAML or JSON results")
    parser.add_argument("--expected-commit", help="Full reviewed commit; required with --results")
    parser.add_argument(
        "--repo", type=Path, default=ROOT, help="Repository containing tested source"
    )
    args = parser.parse_args()
    try:
        plan = yaml.safe_load(args.plan.read_text())
        validate_plan(plan)
        if args.results:
            require(args.expected_commit is not None, "--results requires --expected-commit")
            verify_source(plan, args.repo)
            report = validate_results(
                plan, yaml.safe_load(args.results.read_text()), expected_commit=args.expected_commit
            )
            print(json.dumps(report, indent=2))
            return 0 if report["status"] == "pass" else 1
        print(
            json.dumps(
                {
                    "status": "plan_valid",
                    "plan_sha256": plan_sha256(plan),
                    "note": "Plan structure only; no execution evidence has been checked.",
                }
            )
        )
        return 0
    except (OSError, ValueError, yaml.YAMLError, subprocess.CalledProcessError) as error:
        parser.exit(2, f"validation error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
