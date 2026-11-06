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

"""Exercise coverage selection, exact-source gates, and bounded evidence transport."""

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import agent_scope
import validate_validation_plan as validator
import validation_evidence as transport
import validation_plan as planner


def case(name, covers, cost=1, backend="cpu"):
    return {
        "id": name,
        "covers": covers,
        "cost_seconds": cost,
        "backend": backend,
        "selected": True,
        "reason": "focused test",
        "fixture": "controlled assertions",
        "fixture_kind": "synthetic",
        "facets": ["scope"],
        "owners": ["src/**"],
        "command": ["python3", "-c", "assert True"],
        "qa": "Q0",
        "artifacts": [],
        "topology": {
            "nodes": 1,
            "ranks_per_node": 1,
            "dp_replicate": 1,
            "dp_shard": 1,
            "tp": 1,
            "cp": 1,
            "ep": 1,
        },
        "oracle": {"kind": "assertions", "description": "assertions", "reward": "none"},
        "resources": {"gpus": 0, "packages": [], "environment": {}, "model_paths": []},
    }


@pytest.fixture
def plan():
    return {
        "schema_version": 1,
        "base_commit": "b" * 40,
        "tested_commit": "a" * 40,
        "risk": "R2",
        "owners": ["tests/contracts.py"],
        "facets": ["harness"],
        "artifacts": {"tests/contracts.py": "d" * 64},
        "obligations": [{"id": "contract", "description": "assert the contract", "required": True}],
        "cases": [case("contract-test", ["contract"])],
        "selection": {"explicit_cases": [], "explicit_obligations": []},
    }


def passed(plan, root):
    result = planner.empty_results(plan)
    log = root / ".scratch/probe.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("one assertion passed\n")
    selected = plan["cases"][0]
    result["cases"] = [
        {
            "id": selected["id"],
            "status": "pass",
            "backend": "cpu",
            "topology": selected["topology"],
            "command": selected["command"],
            "runtime": {
                "world_size": 1,
                "ranks_completed": [0],
                "device": "cpu",
                "distributed_type": "NO",
            },
            "observations": {"exit_code": 0},
            "evidence": {".scratch/probe.log": hashlib.sha256(log.read_bytes()).hexdigest()},
        }
    ]
    return result


def test_mandatory_interaction_precedes_cost_selection():
    catalog = {"cases": [case("cheap", ["role"], 1), case("fsdp", ["role"], 50)]}
    requirements = {
        "obligations": {"role": ["shared state"]},
        "mandatory": {"fsdp": ["FSDP interaction"]},
    }
    cases, missing = planner.select_cases(catalog, requirements, [])
    assert not missing
    assert [item["id"] for item in cases if item["selected"]] == ["fsdp"]
    assert cases[1]["reason"] == "FSDP interaction"


def test_missing_coverage_is_retained_and_explicit_additions_are_checked():
    requirements = {"obligations": {"unknown": ["new owner"]}, "mandatory": {}}
    catalog = {"cases": [case("known", ["known"])]}
    cases, missing = planner.select_cases(catalog, requirements, ["known"])
    assert cases[0]["selected"] and missing == ["unknown"]
    with pytest.raises(ValueError, match="Unknown"):
        planner.select_cases(catalog, requirements, ["typo"])


def test_literal_policy_owners_resolve_to_existing_files():
    policy = yaml.safe_load((ROOT / planner.POLICY).read_text())
    for rule in policy["owners"]:
        for path in rule["paths"]:
            if not any(char in path for char in "*?["):
                assert (ROOT / path).is_file(), f"Stale literal policy owner: {path}"


def test_cost_selection_is_deterministic_and_covers_every_obligation():
    catalog = {"cases": [case("b", ["x"], 2), case("a", ["x", "y"], 2), case("c", ["z"], 1)]}
    requirements = {"obligations": {x: [x] for x in ("x", "y", "z")}, "mandatory": {}}
    cases, missing = planner.select_cases(catalog, requirements, [])
    assert [item["id"] for item in cases if item["selected"]] == ["a", "c"]
    assert not missing


@pytest.mark.parametrize(
    "path,delta,expected,forbidden",
    [
        (
            "src/flow_factory/utils/dist.py",
            "rank // tp_size",
            "geometry.logical_samples",
            "roles.isolation.fsdp2",
        ),
        (
            "src/flow_factory/trainers/common/dpo_objective.py",
            "loss = beta * margin",
            "objective.dpo",
            "roles.isolation.fsdp2",
        ),
        (
            "src/flow_factory/trainers/offline/offline_dpo.py",
            "loss = beta * margin",
            "objective.dpo",
            "roles.isolation.fsdp2",
        ),
        (
            "src/flow_factory/trainers/role_optimization.py",
            "optimizer.step()",
            "roles.isolation.fsdp2",
            "reward.worker_device",
        ),
        (".agents/knowledge/README.md", "wording", "harness.contracts", "parallel.tp_parity"),
        (
            "src/flow_factory/contracts/topology.py",
            "replicas = dp_replicate * dp_shard",
            "geometry.topology",
            "roles.isolation.fsdp2",
        ),
    ],
)
def test_historical_owner_patterns(path, delta, expected, forbidden):
    rules = yaml.safe_load((ROOT / ".agents/harness/risk_rules.yaml").read_text())
    policy = yaml.safe_load((ROOT / planner.POLICY).read_text())
    scope = agent_scope.classify_changes([path], {path: delta}, rules, intent="develop")
    scope["scope_complete"] = True
    required = planner.infer_requirements(scope, policy, {})["obligations"]
    assert expected in required
    assert forbidden not in required


def test_proven_logging_change_does_not_inherit_backend_sweep():
    path = "src/flow_factory/trainers/abc.py"
    policy = yaml.safe_load((ROOT / planner.POLICY).read_text())
    scope = {
        "paths": [path],
        "facets": ["observability"],
        "logging_only_paths": [path],
        "scope_complete": True,
    }
    assert set(planner.infer_requirements(scope, policy, {})["obligations"]) == {
        "repository.consistency"
    }


@pytest.mark.parametrize(
    "path",
    ["src/flow_factory/utils/dist.py", "src/flow_factory/utils/group_coordinator.py"],
)
def test_collective_scope_changes_select_feedback_reconstruction(path):
    policy = yaml.safe_load((ROOT / planner.POLICY).read_text())
    catalog = yaml.safe_load((ROOT / "config/validation/catalog.yaml").read_text())
    scope = {"paths": [path], "facets": ["distributed", "geometry"], "scope_complete": True}
    required = planner.infer_requirements(scope, policy, {})
    assert "feedback.reconstruction" in required["obligations"]
    assert not any(name.startswith("roles.isolation.") for name in required["obligations"])
    cases, missing = planner.select_cases(catalog, required, [])
    assert not missing
    assert "feedback-reconstruction" in {case["id"] for case in cases if case["selected"]}


def test_subgroup_selection_requires_distinct_compositional_evidence_with_eight_cuda_ranks():
    policy = yaml.safe_load((ROOT / planner.POLICY).read_text())
    catalog = yaml.safe_load((ROOT / "config/validation/catalog.yaml").read_text())
    scope = {
        "paths": ["src/flow_factory/utils/group_coordinator.py"],
        "facets": ["distributed", "geometry"],
        "scope_complete": True,
    }
    required = planner.infer_requirements(scope, policy, {})
    cases, missing = planner.select_cases(catalog, required, [])
    assert not missing
    selected = {case["id"]: case for case in cases if case["selected"]}
    distinct_proofs = {
        "logical-subgroups-nccl-tp2-cp1-dp4": ("geometry.subgroups.nccl.tp2.cp1.dp4", 2, 1, 4),
        "logical-subgroups-nccl-tp1-cp2-dp4": ("geometry.subgroups.nccl.tp1.cp2.dp4", 1, 2, 4),
        "wan-cp-subgroup-grpo-tp1-dp4": ("geometry.subgroups.native.tp1.cp2.dp4", 1, 2, 4),
        "wan-tp-cp-subgroup-grpo-dp2": ("geometry.subgroups.native.tp2.cp2.dp2", 2, 2, 2),
    }
    for case_id, (obligation, tp, cp, dp) in distinct_proofs.items():
        case = selected[case_id]
        assert obligation in required["obligations"]
        assert case["covers"] == [obligation]
        assert (case["topology"]["tp"], case["topology"]["cp"], case["topology"]["dp_shard"]) == (
            tp,
            cp,
            dp,
        )
        assert case["topology"]["nodes"] == 1
        assert case["resources"]["gpus"] == case["topology"]["ranks_per_node"] == 8
        if case["backend"] == "nccl":
            assert case["execution_scope"] == "topology_only"
            assert case["oracle"]["reward"] == "none"
        else:
            assert case["backend"] == "fsdp2"
            assert case["execution_capability"] == "wan-grpo-cp-v1"
            assert all(
                any(f"-{mode}]" in arg for arg in case["command"]) for mode in ("full", "lora")
            )
        # Neither a smaller DP mesh nor a CPU/backend substitute may erase one
        # of the independently required isolation or training proofs.
        reduced = {"cases": [entry for entry in catalog["cases"] if entry["id"] != case_id]}
        _, uncovered = planner.select_cases(reduced, required, [])
        assert obligation in uncovered, (case_id, uncovered)

    all_axes = selected["logical-subgroups-tp2-cp2-replicate1"]
    assert all_axes["backend"] == "gloo"
    assert all_axes["topology"]["ranks_per_node"] == 16
    assert all_axes["resources"]["gpus"] == 0
    assert all(case["resources"]["gpus"] <= 8 for case in selected.values())


def test_configuration_geometry_methods_select_contracts_without_cuda_sweep():
    path = "src/flow_factory/hparams/args.py"
    policy = yaml.safe_load((ROOT / planner.POLICY).read_text())
    scope = {"paths": [path], "facets": ["geometry"], "scope_complete": True}
    required = planner.infer_requirements(
        scope, policy, {path: ["Arguments.data_parallel_world_size"]}
    )["obligations"]
    assert {"geometry.topology", "geometry.logical_samples"} <= set(required)
    assert not any(name.startswith(("geometry.nccl.", "roles.isolation.")) for name in required)


@pytest.mark.parametrize("method", ["_preprocess_dataset", "_write_empty_part"])
def test_empty_source_cache_methods_select_the_evaluation_contract(method):
    path = "src/flow_factory/data_utils/dataset.py"
    policy = yaml.safe_load((ROOT / planner.POLICY).read_text())
    scope = {"paths": [path], "facets": ["data_schema"], "scope_complete": True}
    required = planner.infer_requirements(scope, policy, {path: [f"GeneralDataset.{method}"]})
    assert set(required["obligations"]) == {"repository.consistency", "evaluation.contracts"}
    unrelated = planner.infer_requirements(
        scope, policy, {path: ["GeneralDataset._load_raw_dataset"]}
    )
    assert f"review.owner:{path}" in unrelated["obligations"]
    assert "evaluation.contracts" not in unrelated["obligations"]


@pytest.mark.parametrize(
    "symbol", ["_evaluation_loader_schema", "_loader_schema", "_loader_length"]
)
def test_loader_identity_methods_select_the_evaluation_contract(symbol):
    path = "src/flow_factory/trainers/common/runtime_identity.py"
    policy = yaml.safe_load((ROOT / planner.POLICY).read_text())
    scope = {"paths": [path], "facets": ["runtime_state"], "scope_complete": True}
    required = planner.infer_requirements(scope, policy, {path: [symbol]})
    assert {"evaluation.contracts", "runtime.identity", "checkpoint.contracts"} <= set(
        required["obligations"]
    )
    unrelated = planner.infer_requirements(scope, policy, {path: ["_backend_schema"]})
    assert "evaluation.contracts" not in unrelated["obligations"]


def test_changed_trainer_methods_select_their_specific_contracts():
    path = "src/flow_factory/trainers/abc.py"
    policy = yaml.safe_load((ROOT / planner.POLICY).read_text())
    rules = yaml.safe_load((ROOT / ".agents/harness/risk_rules.yaml").read_text())
    scope = agent_scope.classify_changes([path], {path: "changed method"}, rules, intent="develop")
    scope["scope_complete"] = True
    required = planner.infer_requirements(
        scope, policy, {path: ["BaseTrainer._run_reward_optimization_overlap"]}
    )["obligations"]
    assert "feedback.tp_agreement" in required
    assert "distributed.collective_order" in required
    assert "runtime.identity" not in required
    assert "roles.isolation.fsdp2" not in required
    required = planner.infer_requirements(
        scope, policy, {path: ["BaseTrainer._save_exact_training_state"]}
    )["obligations"]
    assert "runtime.identity" in required
    assert "checkpoint.contracts" in required
    assert "feedback.tp_agreement" not in required


@pytest.mark.parametrize(
    "path,symbol",
    [
        ("src/flow_factory/rewards/reward_processor.py", "RewardBuffer._prepare_streaming_tile"),
        ("src/flow_factory/rewards/reward_processor.py", "RewardBuffer.resolve_streaming_tile"),
        ("src/flow_factory/rewards/reward_processor.py", "RewardBuffer._collect_async_future"),
        ("src/flow_factory/trainers/abc.py", "BaseTrainer._resolve_reward_overlap_tile_feedback"),
        ("src/flow_factory/trainers/abc.py", "BaseTrainer._synchronize_reward_overlap_error"),
    ],
)
def test_streaming_error_owners_require_distinct_gloo_geometries(path, symbol):
    policy = yaml.safe_load((ROOT / planner.POLICY).read_text())
    catalog = yaml.safe_load((ROOT / "config/validation/catalog.yaml").read_text())
    scope = {"paths": [path], "facets": [], "scope_complete": True}
    required = planner.infer_requirements(scope, policy, {path: [symbol]})
    cases, missing = planner.select_cases(catalog, required, [])
    assert not missing
    selected = {case["id"]: case for case in cases if case["selected"]}
    for tp, cp in ((1, 2), (2, 1), (2, 2)):
        obligation = f"overlap.failure_order.tp{tp}.cp{cp}"
        case_id = f"overlap-failure-order-tp{tp}-cp{cp}"
        probe = selected[case_id]
        assert obligation in required["obligations"]
        assert probe["covers"] == [obligation]
        assert probe["backend"] == "gloo" and probe["qa"] == "Q0"
        assert probe["execution_scope"] == "feedback_probe"
        assert probe["topology"] == {
            "nodes": 1,
            "ranks_per_node": tp * cp,
            "dp_replicate": 1,
            "dp_shard": 1,
            "tp": tp,
            "cp": cp,
            "ep": 1,
        }
        assert probe["resources"]["gpus"] == 0
        nodes = [arg for arg in probe["command"] if "::" in arg]
        assert nodes and all(f"[{tp}-{cp}-" in node for node in nodes)
        validator.validate_topology(probe)
        reduced = copy.deepcopy(catalog)
        reduced["cases"] = [case for case in reduced["cases"] if case["id"] != case_id]
        _, absent = planner.select_cases(reduced, required, [])
        assert obligation in absent


@pytest.mark.parametrize(
    "artifact",
    ["tests/distributed/test_reward_overlap_failure_order.py", "tests/distributed/conftest.py"],
)
def test_streaming_failure_fixture_artifacts_make_all_geometries_mandatory(artifact):
    policy = yaml.safe_load((ROOT / planner.POLICY).read_text())
    catalog = yaml.safe_load((ROOT / "config/validation/catalog.yaml").read_text())
    scope = {
        "paths": [artifact],
        "facets": [],
        "scope_complete": True,
        "docs_only_paths": [],
        "logging_only_paths": [],
    }
    required = planner.requirements_for(scope, {}, catalog, policy)
    assert {f"overlap-failure-order-tp{tp}-cp{cp}" for tp, cp in ((1, 2), (2, 1), (2, 2))} <= set(
        required["mandatory"]
    )


@pytest.mark.parametrize(
    "path,symbol,native",
    [
        (
            "src/flow_factory/trainers/abc.py",
            "BaseTrainer._backward_full_output_loss",
            {
                "parallel.cp_sft",
                "parallel.tp_cp_sft",
                "algorithm.cp.grpo.tp1",
                "algorithm.cp.grpo.tp2",
                "algorithm.cp.dmd2.tp1",
                "algorithm.cp.dmd2.tp2",
            },
        ),
        (
            "src/flow_factory/trainers/common/context_parallel.py",
            "full_output_context_parallel_loss",
            {
                "parallel.cp_sft",
                "parallel.tp_cp_sft",
                "algorithm.cp.grpo.tp1",
                "algorithm.cp.dmd2.tp2",
            },
        ),
        (
            "src/flow_factory/trainers/distillation/distillation_runtime.py",
            "run_role_phase",
            {"algorithm.cp.dmd2.tp1", "algorithm.cp.dmd2.tp2"},
        ),
        (
            "src/flow_factory/trainers/distillation/tdm_r1.py",
            "TDMR1Trainer._optimize_reward_overlap_tile",
            set(),
        ),
    ],
)
def test_full_output_backward_owners_select_all_affected_backends(path, symbol, native):
    policy = yaml.safe_load((ROOT / planner.POLICY).read_text())
    catalog = yaml.safe_load((ROOT / "config/validation/catalog.yaml").read_text())
    # The semantic owner itself must supply the obligations, without an incidental
    # roles+distributed facet interaction to hide an incomplete mapping.
    scope = {"paths": [path], "facets": [], "scope_complete": True}
    required = planner.infer_requirements(scope, policy, {path: [symbol]})
    assert {
        "optimization.full_output_backward",
        "roles.isolation.ddp",
        "roles.isolation.zero2",
        "roles.isolation.fsdp2",
        *native,
    } <= set(required["obligations"])
    cases, missing = planner.select_cases(catalog, required, [])
    assert not missing
    selected = {case["id"] for case in cases if case["selected"]}
    assert {
        "full-output-backward-contracts",
        "roles-ddp",
        "roles-zero2",
        "roles-fsdp2",
    } <= selected


@pytest.mark.parametrize(
    "path,symbol",
    [
        ("src/flow_factory/rewards/reward_processor.py", "RewardBuffer._prepare_streaming_tile"),
        ("src/flow_factory/trainers/abc.py", "BaseTrainer._resolve_reward_overlap_tile_feedback"),
        ("src/flow_factory/trainers/abc.py", "BaseTrainer._run_reward_optimization_overlap"),
        ("src/flow_factory/trainers/abc.py", "BaseTrainer.cleanup"),
        ("src/flow_factory/rewards/tile_plan.py", "build_reward_tile_plan"),
        ("src/flow_factory/utils/group_coordinator.py", "GroupCoordinator.gather"),
        (
            "src/flow_factory/advantage/advantage_processor.py",
            "AdvantageProcessor._compute_advantages",
        ),
    ],
)
def test_complete_overlap_owners_require_real_gloo_loop_geometries(path, symbol):
    policy = yaml.safe_load((ROOT / planner.POLICY).read_text())
    catalog = yaml.safe_load((ROOT / "config/validation/catalog.yaml").read_text())
    scope = {"paths": [path], "facets": [], "scope_complete": True}
    required = planner.infer_requirements(scope, policy, {path: [symbol]})
    cases, missing = planner.select_cases(catalog, required, [])
    assert not missing
    selected = {case["id"]: case for case in cases if case["selected"]}
    for tp, dp, count in ((1, 2, 5), (1, 4, 2), (2, 2, 2)):
        obligation = f"overlap.loop.tp{tp}.cp2.dp{dp}"
        case_id = f"overlap-loop-tp{tp}-cp2-dp{dp}"
        case = selected[case_id]
        assert obligation in required["obligations"] and case["covers"] == [obligation]
        assert case["backend"] == "gloo" and case["qa"] == "Q0"
        assert case["execution_scope"] == "feedback_probe"
        assert case["topology"] == {
            "nodes": 1,
            "ranks_per_node": dp * tp * 2,
            "dp_replicate": 1,
            "dp_shard": dp,
            "tp": tp,
            "cp": 2,
            "ep": 1,
        }
        assert case["resources"]["gpus"] == 0
        nodes = [arg for arg in case["command"] if "::" in arg]
        assert len(nodes) == count
        assert all(f"[{dp}-{tp}-" in node or "failed_next_tile" in node for node in nodes)
        validator.validate_topology(case)
        reduced = copy.deepcopy(catalog)
        reduced["cases"] = [item for item in reduced["cases"] if item["id"] != case_id]
        assert obligation in planner.select_cases(reduced, required, [])[1]


@pytest.mark.parametrize(
    "artifact", ["tests/distributed/test_reward_overlap_loop.py", "tests/distributed/conftest.py"]
)
def test_complete_overlap_artifacts_bind_every_loop_geometry(artifact):
    policy = yaml.safe_load((ROOT / planner.POLICY).read_text())
    catalog = yaml.safe_load((ROOT / "config/validation/catalog.yaml").read_text())
    scope = {
        "paths": [artifact],
        "facets": [],
        "scope_complete": True,
        "docs_only_paths": [],
        "logging_only_paths": [],
    }
    required = planner.requirements_for(scope, {}, catalog, policy)
    assert {f"overlap-loop-tp{tp}-cp2-dp{dp}" for tp, dp in ((1, 2), (1, 4), (2, 2))} <= set(
        required["mandatory"]
    )


def test_full_output_objective_owner_does_not_select_complete_overlap_loop():
    policy = yaml.safe_load((ROOT / planner.POLICY).read_text())
    path = "src/flow_factory/trainers/abc.py"
    scope = {"paths": [path], "facets": [], "scope_complete": True}
    required = planner.infer_requirements(
        scope, policy, {path: ["BaseTrainer._backward_full_output_loss"]}
    )
    assert not any(name.startswith("overlap.loop.") for name in required["obligations"])


@pytest.mark.parametrize(
    "path,symbol",
    [
        (
            "src/flow_factory/trainers/common/context_parallel.py",
            "validate_context_parallel_config",
        ),
        ("src/flow_factory/trainers/abc.py", "BaseTrainer._resolve_reward_overlap_tile_feedback"),
        ("src/flow_factory/rewards/reward_processor.py", "RewardBuffer.resolve_streaming_tile"),
        ("src/flow_factory/trainers/rl/grpo.py", "GRPOTrainer.optimize"),
    ],
)
def test_native_overlap_owners_select_three_distinct_bounded_scenarios(path, symbol):
    policy = yaml.safe_load((ROOT / planner.POLICY).read_text())
    catalog = yaml.safe_load((ROOT / "config/validation/catalog.yaml").read_text())
    required = planner.infer_requirements(
        {"paths": [path], "facets": [], "scope_complete": True}, policy, {path: [symbol]}
    )
    selected, missing = planner.select_cases(catalog, required, [])
    assert not missing
    selected = {case["id"]: case for case in selected if case["selected"]}
    for label, dp, tp in (("a", 2, 1), ("b", 2, 2), ("c", 4, 1)):
        case = selected[f"wan-overlap-native-{label}"]
        assert case["topology"]["dp_shard"] == dp and case["topology"]["tp"] == tp
        assert case["topology"]["cp"] == 2 and case["resources"]["gpus"] == dp * tp * 2
        assert case["oracle"]["kind"] == "numerical" and case["oracle"]["reward"] == "varying"
        assert len([node for node in case["command"] if "::" in node]) == 1
        assert "tests/distributed/conftest.py" in case["artifacts"]
        assert "tests/distributed/attention_precision_reference.py" in case["artifacts"]
        validator.validate_topology(case)


@pytest.mark.parametrize(
    "artifact",
    [
        "tests/distributed/context_parallel_overlap_runner.py",
        "tests/distributed/online_optimizer_observations.py",
        "tests/distributed/coupled_objective_reference.py",
        "tests/distributed/wan_overlap_fixture.py",
    ],
)
def test_native_overlap_artifacts_keep_all_actual_consumers_mandatory(artifact):
    policy = yaml.safe_load((ROOT / planner.POLICY).read_text())
    catalog = yaml.safe_load((ROOT / "config/validation/catalog.yaml").read_text())
    scope = {
        "paths": [artifact],
        "facets": [],
        "scope_complete": True,
        "docs_only_paths": [],
        "logging_only_paths": [],
    }
    required = planner.requirements_for(scope, {}, catalog, policy)
    assert {f"wan-overlap-native-{label}" for label in ("a", "b", "c")} <= set(
        required["mandatory"]
    )


def test_unrelated_grpo_guard_objective_does_not_require_wan_overlap_matrix():
    path = "src/flow_factory/trainers/rl/grpo.py"
    policy = yaml.safe_load((ROOT / planner.POLICY).read_text())
    scope = {"paths": [path], "facets": [], "scope_complete": True}
    required = planner.infer_requirements(scope, policy, {path: ["GRPOGuardTrainer._guard_ratio"]})
    assert not any(name.startswith("overlap.native.wan.") for name in required["obligations"])


def test_cp_admission_only_does_not_recertify_unchanged_algorithm_math():
    path = "src/flow_factory/trainers/common/context_parallel.py"
    policy = yaml.safe_load((ROOT / planner.POLICY).read_text())
    required = planner.infer_requirements(
        {"paths": [path], "facets": [], "scope_complete": True},
        policy,
        {path: ["validate_context_parallel_config"]},
    )["obligations"]
    assert "parallel.cp_contracts" in required and "overlap.cp_admission" in required
    assert {
        f"overlap.native.wan.{name}" for name in ("ordered", "tp_ready", "subgroup_ready")
    } <= set(required)
    assert not any(
        name.startswith(
            (
                "algorithm.cp.",
                "checkpoint.cp_state.",
                "preparation.cp_probe",
                "preparation.tp_cp_probe",
            )
        )
        for name in required
    )
    assert "parallel.cp_sft" not in required and "parallel.tp_cp_sft" not in required


@pytest.mark.parametrize(
    "symbol",
    [
        "ContextParallelTrainingCapability",
        "ContextParallelTrainingCapability.__post_init__",
        "context_parallel_training_capability",
        "validate_context_parallel_backend",
        "full_output_context_parallel_loss",
    ],
)
def test_cp_runtime_capability_backend_and_normalization_keep_native_evidence(symbol):
    path = "src/flow_factory/trainers/common/context_parallel.py"
    policy = yaml.safe_load((ROOT / planner.POLICY).read_text())
    required = planner.infer_requirements(
        {"paths": [path], "facets": [], "scope_complete": True}, policy, {path: [symbol]}
    )["obligations"]
    assert {
        "parallel.cp_sft",
        "parallel.tp_cp_sft",
        "preparation.cp_probe",
        "preparation.tp_cp_probe",
        "checkpoint.cp_state.tp1",
        "checkpoint.cp_state.tp2",
        "algorithm.cp.grpo.tp1",
        "algorithm.cp.grpo.tp2",
        "algorithm.cp.dmd2.tp1",
        "algorithm.cp.dmd2.tp2",
    } <= set(required)
    if symbol == "full_output_context_parallel_loss":
        assert {
            "optimization.full_output_backward",
            "roles.isolation.ddp",
            "roles.isolation.zero2",
            "roles.isolation.fsdp2",
        } <= set(required)


@pytest.mark.parametrize(
    "symbol",
    [
        "BaseTrainer.__init__",
        "BaseTrainer.start",
        "BaseTrainer._save_exact_training_state",
        "BaseTrainer._restore_runtime_state",
        "BaseTrainer._run_periodic_cycle_boundaries",
        "BaseTrainer._after_acquisition_cycle",
        "BaseTrainer.progress",
        "BaseTrainer._get_progress",
        "BaseTrainer.epoch",
        "BaseTrainer.step",
        "BaseTrainer.cycle_index",
        "BaseTrainer.should_continue_training",
        "BaseTrainer._suspend_adapter_state_resume",
        "BaseTrainer._backward_full_output_loss",
        "BaseTrainer._apply_optimizer_step",
        "BaseTrainer._finish_role_microbatch",
    ],
)
def test_actual_trainer_state_owners_keep_both_cp_continuation_cases(symbol):
    path = "src/flow_factory/trainers/abc.py"
    policy = yaml.safe_load((ROOT / planner.POLICY).read_text())
    required = planner.infer_requirements(
        {"paths": [path], "facets": [], "scope_complete": True}, policy, {path: [symbol]}
    )["obligations"]
    assert {"checkpoint.cp_state.tp1", "checkpoint.cp_state.tp2"} <= set(required)


def test_overlap_feedback_change_does_not_claim_a_checkpoint_format_change():
    path = "src/flow_factory/trainers/abc.py"
    policy = yaml.safe_load((ROOT / planner.POLICY).read_text())
    required = planner.infer_requirements(
        {"paths": [path], "facets": [], "scope_complete": True},
        policy,
        {path: ["BaseTrainer", "BaseTrainer._resolve_reward_overlap_tile_feedback"]},
    )["obligations"]
    assert "overlap.native.wan.tp_ready" in required
    assert not any(name.startswith("checkpoint.cp_state.") for name in required)


@pytest.mark.parametrize("symbol", ["<unparsed>", "new_unreviewed_cp_runtime"])
def test_unknown_cp_owner_remains_an_uncovered_review_obligation(symbol):
    path = "src/flow_factory/trainers/common/context_parallel.py"
    policy = yaml.safe_load((ROOT / planner.POLICY).read_text())
    required = planner.infer_requirements(
        {"paths": [path], "facets": [], "scope_complete": True}, policy, {path: [symbol]}
    )["obligations"]
    assert f"review.owner:{path}" in required


@pytest.mark.parametrize(
    "addition,unknown",
    [
        ("\ndef new_runtime_helper():\n    return 1\n", "new_runtime_helper"),
        ("\nNATIVE_BACKEND = 'new'\n", "<module>"),
        ("\nimport new_backend\n", "<module>"),
    ],
)
def test_cp_admission_plus_unknown_definition_or_module_change_fails_visible(addition, unknown):
    path = "src/flow_factory/trainers/common/context_parallel.py"
    policy = yaml.safe_load((ROOT / planner.POLICY).read_text())
    before = "def validate_context_parallel_config(config):\n    return False\n"
    after = before.replace("False", "True") + addition
    symbols = planner.changed_symbols(before, after)
    assert "validate_context_parallel_config" in symbols and unknown in symbols
    required = planner.infer_requirements(
        {"paths": [path], "facets": [], "scope_complete": True}, policy, {path: symbols}
    )
    assert f"review.symbol:{path}:{unknown}" in required["obligations"]
    catalog = yaml.safe_load((ROOT / "config/validation/catalog.yaml").read_text())
    _, missing = planner.select_cases(catalog, required, [])
    assert f"review.symbol:{path}:{unknown}" in missing


def test_property_getter_change_keeps_its_state_owner_visible():
    before = (
        "class BaseTrainer:\n    @property\n    def progress(self):\n        return self._progress\n"
        "    @progress.setter\n    def progress(self, value):\n        self._progress = value\n"
    )
    after = before.replace("return self._progress", "return self.other_progress")
    assert "BaseTrainer.progress" in planner.changed_symbols(before, after)


def test_full_output_suite_and_its_fixture_dependencies_are_mandatory_artifacts():
    path = "tests/trainers/test_full_output_backward.py"
    policy = yaml.safe_load((ROOT / planner.POLICY).read_text())
    catalog = yaml.safe_load((ROOT / "config/validation/catalog.yaml").read_text())
    case = next(case for case in catalog["cases"] if case["id"] == "full-output-backward-contracts")
    assert {
        path,
        "tests/models/minimax_h3/test_no_weight_production_acceptance.py",
        "tests/trainers/test_dgpo_crd_production_paths.py",
        "tests/trainers/test_multirole_production_paths.py",
        "tests/trainers/test_offline_trainers.py",
        "tests/trainers/test_opd_production_paths.py",
        "tests/trainers/test_tdm.py",
    } <= set(case["artifacts"])
    for artifact in case["artifacts"]:
        scope = {
            "paths": [artifact],
            "facets": [],
            "scope_complete": True,
            "docs_only_paths": [],
            "logging_only_paths": [],
        }
        required = planner.requirements_for(scope, {}, catalog, policy)
        assert "optimization.full_output_backward" in required["obligations"]
        assert "full-output-backward-contracts" in required["mandatory"]
        _, missing = planner.select_cases(catalog, required, [])
        assert not missing


@pytest.mark.parametrize(
    "path,trainer,method,expected",
    [
        (
            "src/flow_factory/trainers/distillation/tdm_r1.py",
            "TDMR1Trainer",
            "_surrogate_boundary_loss",
            "lifecycle.tdm",
        ),
        (
            "src/flow_factory/trainers/rl/dppo.py",
            "DPPOTrainer",
            "optimize",
            "optimization.full_output_backward",
        ),
        (
            "src/flow_factory/trainers/rl/nft.py",
            "DiffusionNFTTrainer",
            "optimize",
            "optimization.full_output_backward",
        ),
    ],
)
def test_objective_leaf_does_not_select_shared_backward_backends(path, trainer, method, expected):
    policy = yaml.safe_load((ROOT / planner.POLICY).read_text())
    catalog = yaml.safe_load((ROOT / "config/validation/catalog.yaml").read_text())
    before = f"class {trainer}:\n def {method}(self): return 1\n"
    after = f"class {trainer}:\n def {method}(self): return 2\n"
    scope = {"paths": [path], "facets": ["objective"], "scope_complete": True}
    required = planner.infer_requirements(
        scope, policy, {path: planner.changed_symbols(before, after)}
    )
    assert expected in required["obligations"]
    assert not any(name.startswith("roles.isolation.") for name in required["obligations"])
    cases, missing = planner.select_cases(catalog, required, [])
    assert not missing
    native = {case["id"] for case in cases if case["selected"] and case["resources"]["gpus"]}
    assert native == ({"flux-overlap-native-e"} if trainer == "DPPOTrainer" else set())


def test_unknown_owner_and_incomplete_scope_cannot_pass_as_static():
    scope = {"paths": ["src/flow_factory/new_owner.py"], "facets": [], "scope_complete": False}
    policy = yaml.safe_load((ROOT / planner.POLICY).read_text())
    required = planner.infer_requirements(scope, policy, {})["obligations"]
    assert "review.owner:src/flow_factory/new_owner.py" in required
    assert "review.incomplete_scope" in required


def test_changed_symbols_include_deletions_and_nested_methods():
    before = "class A:\n def loss(self): return 1\ndef removed(): return 0\n"
    after = "class A:\n def loss(self): return 2\n"
    assert planner.changed_symbols(before, after) == ["A", "A.loss", "removed"]


def test_missing_result_and_uncovered_obligation_are_blocked(plan):
    summary, _ = planner.report(plan, [], plan["tested_commit"])
    assert summary["status"] == "blocked"
    assert summary["uncovered_obligations"] == ["contract"]
    plan["cases"][0]["selected"] = False
    summary, _ = planner.report(plan, [], plan["tested_commit"])
    assert summary["status"] == "blocked"
    assert summary["cases"]["contract-test"] == "not_selected"
    with pytest.raises(ValueError, match="uncovered required"):
        validator.validate_plan(plan)


def test_passing_evidence_stale_revision_and_plan_tampering(plan, tmp_path):
    result = passed(plan, tmp_path)
    assert planner.report(plan, [result], plan["tested_commit"])[0]["status"] == "pass"
    with pytest.raises(ValueError, match="commit"):
        planner.report(plan, [result], "c" * 40)
    plan["cases"][0]["command"] = ["false"]
    with pytest.raises(ValueError, match="stale plan"):
        planner.report(plan, [result], plan["tested_commit"])


def test_conflicting_evidence_fails_but_missing_dependency_can_be_filled(plan, tmp_path):
    result = passed(plan, tmp_path)
    missing = planner.empty_results(plan)
    missing["cases"] = [{"id": "contract-test", "status": "blocked", "reason": "no torch"}]
    assert planner.report(plan, [missing, result], plan["tested_commit"])[0]["status"] == "pass"
    conflicting = copy.deepcopy(missing)
    conflicting["cases"][0]["status"] = "fail"
    with pytest.raises(ValueError, match="Conflicting"):
        planner.report(plan, [result, conflicting], plan["tested_commit"])


def test_canonical_plan_cannot_drop_required_gpu_case(plan, monkeypatch):
    original = copy.deepcopy(plan)
    monkeypatch.setattr(planner, "generate", lambda *args: original)
    planner.verify_generated_plan(ROOT, plan)
    plan["obligations"][0]["required"] = False
    with pytest.raises(ValueError, match="canonical"):
        planner.verify_generated_plan(ROOT, plan)


def test_bundle_verifies_bytes_and_source_before_import(plan, tmp_path):
    result = passed(plan, tmp_path)
    bundle = transport.make_bundle(tmp_path, plan, result)
    imported = transport.import_bundle(tmp_path, plan, bundle)
    transport.verify_files(tmp_path, imported)
    bundle["files"][".scratch/probe.log"] = "forged pass"
    with pytest.raises(ValueError, match="content mismatch"):
        transport.import_bundle(tmp_path, plan, bundle)
    bundle["tested_commit"] = "c" * 40
    with pytest.raises(ValueError, match="source identity"):
        transport.import_bundle(tmp_path, plan, bundle)


def test_bundle_rejects_external_paths_and_oversize_payload(plan, tmp_path):
    result = passed(plan, tmp_path)
    bundle = transport.make_bundle(tmp_path, plan, result)
    bundle["files"]["extra"] = "x" * transport.MAX_BUNDLE_BYTES
    with pytest.raises(ValueError, match="exceeds"):
        transport.import_bundle(tmp_path, plan, bundle)
    result["cases"][0]["evidence"] = {"../outside.log": "d" * 64}
    with pytest.raises(ValueError, match="relative artifact path"):
        validator.validate_results(plan, result, expected_commit=plan["tested_commit"])


@pytest.mark.parametrize(
    "body,expected",
    [
        ("def test_probe(): assert 2 + 2 == 4\n", "pass"),
        ("import pytest\ndef test_probe(): pytest.skip('no fixture')\n", "blocked"),
        ("def test_probe(): assert False\n", "fail"),
    ],
)
def test_cpu_runner_does_not_count_skips_as_proof(plan, tmp_path, monkeypatch, body, expected):
    test = tmp_path / "test_probe.py"
    test.write_text(body)
    plan["cases"][0]["command"] = [sys.executable, "-m", "pytest", "-q", str(test)]
    plan["cases"][0]["resources"]["packages"] = ["pytest"]
    monkeypatch.setattr(planner, "verify_checkout", lambda *args: None)
    result = planner.run_cpu(tmp_path, plan, tmp_path / ".scratch/logs")
    assert result["cases"][0]["status"] == expected
    if expected == "pass":
        transport.verify_files(tmp_path, result)


def test_generated_plan_reads_immutable_policy_and_artifacts(tmp_path, monkeypatch):
    import subprocess

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path, text=True).strip()

    git("init", "-q")
    git("config", "user.name", "Validation Test")
    git("config", "user.email", "test@example.invalid")
    policy_inputs = (planner.CATALOG, planner.POLICY, ".agents/harness/risk_rules.yaml")
    monkeypatch.setattr(planner, "POLICY_INPUTS", policy_inputs)
    catalog = {
        "schema_version": 1,
        "cases": [case("geometry", ["geometry.logical_samples", "repository.consistency"])],
    }
    catalog["cases"][0]["artifacts"] = ["tests/geometry.py"]
    files = {
        planner.CATALOG: yaml.safe_dump(catalog),
        # This tests immutable Git inputs. Production policy expansion has
        # separate owner tests and must not outgrow this one-case fixture.
        planner.POLICY: yaml.safe_dump(
            {
                "schema_version": 1,
                "static_paths": [],
                "owners": [
                    {
                        "paths": ["src/flow_factory/utils/dist.py"],
                        "requires": ["geometry.logical_samples"],
                        "reason": "Fixture geometry owner.",
                    }
                ],
                "interactions": [],
            }
        ),
        ".agents/harness/risk_rules.yaml": (ROOT / ".agents/harness/risk_rules.yaml").read_text(),
        ".agents/harness/test_routes.yaml": (ROOT / ".agents/harness/test_routes.yaml").read_text(),
        "tests/geometry.py": "def test_geometry(): assert 4 // 2 == 2\n",
        "src/flow_factory/utils/dist.py": "def rank(world, tp): return world\n",
        ".gitignore": ".scratch/\n",
    }
    for path, content in files.items():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    git("add", ".")
    git("commit", "-qm", "baseline")
    base = git("rev-parse", "HEAD")
    (tmp_path / "src/flow_factory/utils/dist.py").write_text(
        "def rank(world, tp): return world // tp\n"
    )
    files["tests/geometry.py"] = "def test_geometry(): assert 8 // 4 == 2\n"
    (tmp_path / "tests/geometry.py").write_text(files["tests/geometry.py"])
    git("add", ".")
    git("commit", "-qm", "fix logical geometry")
    head = git("rev-parse", "HEAD")
    generated = planner.generate(tmp_path, base, head, [], [])
    assert generated["tested_commit"] == head
    assert generated["selection"]["uncovered_obligations"] == []
    assert generated["selection"]["changed_symbols"]["src/flow_factory/utils/dist.py"] == ["rank"]
    assert (
        generated["artifacts"]["tests/geometry.py"]
        == hashlib.sha256(files["tests/geometry.py"].encode()).hexdigest()
    )
    planner.verify_checkout(tmp_path, generated)
    replay = planner.replay(tmp_path, base, head)
    assert replay["execution_eligible"] is False
    assert "geometry.logical_samples" in replay["requirements"]["obligations"]
    assert "review.owner:tests/geometry.py" not in replay["requirements"]["obligations"]
    assert "geometry" in replay["requirements"]["mandatory"]
    assert set(replay["requirements"]["obligations"]) == {
        item["id"] for item in generated["obligations"] if item["required"]
    }
    (tmp_path / "tests/geometry.py").write_text("def test_geometry(): pass\n")
    with pytest.raises(ValueError, match="clean checkout"):
        planner.verify_checkout(tmp_path, generated)
    assert planner.generate(tmp_path, base, head, [], []) == generated
    # A tracked symlink is a clean Git blob, but cannot authorize reading an
    # executable artifact from outside the checkout on a runner.
    target = tmp_path / "tests/geometry.py"
    target.unlink()
    target.symlink_to(ROOT / "tests/scripts/test_validation_planner.py")
    git("add", ".")
    git("commit", "-qm", "external artifact link")
    with pytest.raises(ValueError, match="not a regular Git file"):
        planner.generate(tmp_path, base, "HEAD", [], [])


def test_import_rejects_symlinked_scratch_before_creating_directories(plan, tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    result = passed(plan, root)
    bundle = transport.make_bundle(root, plan, result)
    import shutil

    shutil.rmtree(root / ".scratch")
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / ".scratch").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        transport.import_bundle(root, plan, bundle)
    assert list(outside.iterdir()) == []


def test_fetch_accepts_only_matching_manual_evidence_workflow(plan, monkeypatch):
    import io
    import zipfile

    expected = {"schema_version": 1, "example": "bounded payload"}
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        archive.writestr("bundle.json", json.dumps(expected))
    name = transport.artifact_name(plan)
    artifacts = {
        "artifacts": [
            {
                "id": 1,
                "name": name,
                "expired": False,
                "created_at": "2026-01-02",
                "workflow_run": {"id": 11, "head_sha": "c" * 40},
                "size_in_bytes": 100,
            },
            {
                "id": 2,
                "name": name,
                "expired": False,
                "created_at": "2026-01-01",
                "workflow_run": {"id": 12, "head_sha": plan["tested_commit"]},
                "size_in_bytes": 100,
            },
        ]
    }
    calls = []

    def output(argv):
        endpoint = argv[-1]
        calls.append(endpoint)
        if "artifacts?" in endpoint:
            return json.dumps(artifacts).encode()
        if endpoint.endswith("/runs/12"):
            return json.dumps(
                {
                    "event": "workflow_dispatch",
                    "path": ".github/workflows/quality.yml",
                    "conclusion": "success",
                }
            ).encode()
        if endpoint.endswith("/artifacts/2/zip"):
            return data.getvalue()
        raise AssertionError(endpoint)

    monkeypatch.setattr(transport.subprocess, "check_output", output)
    assert transport.fetch_bundle("example/repository", plan) == expected
    assert not any(endpoint.endswith("/runs/11") for endpoint in calls)
    artifacts["artifacts"][1]["expired"] = True
    assert transport.fetch_bundle("example/repository", plan) is None


def test_only_project_version_has_a_proven_metadata_fast_path():
    before = '[project]\nname = "flow-factory"\nversion = "0.1.0"\ndependencies = ["torch>=2.10"]\n'
    after = before.replace('"0.1.0"', '"0.3.0"')
    assert planner.package_version_only(before, after)
    assert not planner.package_version_only(before, after.replace("torch>=2.10", "torch>=2.11"))
    assert not planner.package_version_only(before, after.replace("version", "unknown"))
    assert not planner.package_version_only(
        before, after + '\n[build-system]\nrequires=["arbitrary"]\n'
    )
    assert not planner.package_version_only(None, after)


def test_selected_case_requires_its_dependency_identities(plan, tmp_path):
    plan["cases"][0]["resources"]["packages"] = ["required-library"]
    result = passed(plan, tmp_path)
    with pytest.raises(ValueError, match="missing dependency identities"):
        planner.report(plan, [result], plan["tested_commit"])
    result["environment"]["packages"]["Required_Library"] = "1.2.3"
    assert planner.report(plan, [result], plan["tested_commit"])[0]["status"] == "pass"


def test_p06_owner_requires_its_semantic_evidence():
    path = "src/flow_factory/trainers/common/model_preparation.py"
    policy = yaml.safe_load((ROOT / planner.POLICY).read_text())
    scope = {"paths": [path], "facets": [], "scope_complete": True}
    required = planner.infer_requirements(scope, policy, {})["obligations"]
    assert set(
        [
            "preparation.contracts",
            "preparation.variant_tp",
            "preparation.ddp",
            "preparation.fsdp2",
            "preparation.zero2",
            "parallel.tp_parity",
            "parallel.sp_parity",
        ]
    ) <= set(required)
    assert not any(name.startswith("unmapped:") for name in required)


def test_p07_owner_requires_its_semantic_evidence():
    path = "src/flow_factory/utils/tensor_state.py"
    policy = yaml.safe_load((ROOT / planner.POLICY).read_text())
    scope = {"paths": [path], "facets": [], "scope_complete": True}
    required = planner.infer_requirements(scope, policy, {})["obligations"]
    assert set(
        ["parameter_state.contracts", "parameter_state.fsdp2", "parameter_state.fsdp2-tp"]
    ) <= set(required)
    assert not any(name.startswith("unmapped:") for name in required)


def test_p08_owner_requires_its_semantic_evidence():
    path = "src/flow_factory/rewards/reward_processor.py"
    policy = yaml.safe_load((ROOT / planner.POLICY).read_text())
    scope = {"paths": [path], "facets": [], "scope_complete": True}
    required = planner.infer_requirements(scope, policy, {})["obligations"]
    assert set(["reward.lifecycle"]) <= set(required)
    assert not any(name.startswith("unmapped:") for name in required)


@pytest.mark.parametrize(
    "path",
    [
        "src/flow_factory/ema/ema.py",
        "src/flow_factory/models/variants.py",
        "src/flow_factory/models/abc.py",
        "src/flow_factory/trainers/common/model_preparation.py",
    ],
)
def test_snapshot_writers_require_forward_and_pending_graph_evidence(path):
    policy = yaml.safe_load((ROOT / planner.POLICY).read_text())
    scope = {"paths": [path], "facets": [], "scope_complete": True}
    required = planner.infer_requirements(scope, policy, {})["obligations"]
    assert {"checkpoint.snapshot_forward.tp1", "checkpoint.snapshot_forward.tp2"} <= set(required)


def test_original_dtype_changes_require_real_mixed_precision_roles():
    path = "src/flow_factory/models/abc.py"
    policy = yaml.safe_load((ROOT / planner.POLICY).read_text())
    scope = {"paths": [path], "facets": [], "scope_complete": True}
    required = planner.infer_requirements(
        scope, policy, {path: ["BaseAdapter._trained_component_uniform_dtype"]}
    )["obligations"]
    assert {"precision.original_contracts", "precision.variant_original"} <= set(required)


@pytest.mark.parametrize(
    "path,expected_native",
    [
        (
            "test_context_parallel_online",
            {"wan-cp-grpo-tp1", "wan-cp-grpo-tp2", "wan-cp-dmd2-tp1", "wan-cp-dmd2-tp2"},
        ),
        (
            "test_context_parallel_evaluation",
            {
                "wan-cp-evaluation-tp1",
                "wan-cp-evaluation-tp2",
                "wan-evaluation-cp1-sft",
                "wan-evaluation-cp1-grpo",
            },
        ),
        (
            "test_context_parallel_subgroups",
            {"wan-cp-subgroup-grpo-tp1-dp4", "wan-tp-cp-subgroup-grpo-dp2"},
        ),
        (
            "context_parallel_online_runner",
            {
                "wan-cp-grpo-tp1",
                "wan-cp-grpo-tp2",
                "wan-cp-dmd2-tp1",
                "wan-cp-dmd2-tp2",
                "wan-cp-subgroup-grpo-tp1-dp4",
                "wan-tp-cp-subgroup-grpo-dp2",
            },
        ),
        (
            "online_optimizer_observations",
            {
                "wan-cp-grpo-tp1",
                "wan-cp-grpo-tp2",
                "wan-cp-dmd2-tp1",
                "wan-cp-dmd2-tp2",
                "wan-cp-subgroup-grpo-tp1-dp4",
                "wan-tp-cp-subgroup-grpo-dp2",
                "wan-overlap-native-a",
                "wan-overlap-native-b",
                "wan-overlap-native-c",
                "flux-overlap-native-d",
                "flux-overlap-native-e",
            },
        ),
        (
            "context_parallel_evaluation_runner",
            {
                "wan-cp-evaluation-tp1",
                "wan-cp-evaluation-tp2",
                "wan-evaluation-cp1-sft",
                "wan-evaluation-cp1-grpo",
            },
        ),
        (
            "wan_evaluation_fixture",
            {
                "wan-cp-evaluation-tp1",
                "wan-cp-evaluation-tp2",
                "wan-evaluation-cp1-sft",
                "wan-evaluation-cp1-grpo",
            },
        ),
        ("dmd2_objective_reference", {"wan-cp-dmd2-tp1", "wan-cp-dmd2-tp2"}),
        ("online_observations", {"wan-cp-dmd2-tp1", "wan-cp-dmd2-tp2"}),
    ],
)
def test_native_fixture_selection_follows_extracted_owners(path, expected_native):
    path = f"tests/distributed/{path}.py"
    catalog = yaml.safe_load((ROOT / planner.CATALOG).read_text())
    policy = yaml.safe_load((ROOT / planner.POLICY).read_text())
    scope = {
        "paths": [path],
        "facets": [],
        "docs_only_paths": [],
        "logging_only_paths": [],
        "scope_complete": True,
    }
    requirements = planner.requirements_for(scope, {}, catalog, policy)
    cases, missing = planner.select_cases(catalog, requirements, [])
    assert not missing
    selected = {case["id"]: case for case in cases if case["selected"]}
    native = {name for name, case in selected.items() if case["resources"]["gpus"]}
    assert native == expected_native
    expected_cpu = (
        "online-numerical-reference-contracts"
        if path.endswith(("dmd2_objective_reference.py", "online_observations.py"))
        else "native-fixture-ownership-contracts"
    )
    assert expected_cpu in selected


def test_native_fixture_catalog_binds_transitive_local_imports_and_explicit_recipes():
    import ast

    catalog = yaml.safe_load((ROOT / planner.CATALOG).read_text())
    selected = {
        case["id"]: case
        for case in catalog["cases"]
        if any(
            path.endswith(
                ("context_parallel_online_runner.py", "context_parallel_evaluation_runner.py")
            )
            for path in case["artifacts"]
        )
    }
    assert len(selected) == 11
    directory = ROOT / "tests/distributed"
    for name, case in selected.items():
        artifacts = set(case["artifacts"])
        for path in artifacts:
            if not path.startswith("tests/distributed/") or not path.endswith(".py"):
                continue
            for node in ast.parse((ROOT / path).read_text()).body:
                modules = (
                    [alias.name for alias in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
                )
                for module in modules:
                    target = directory / (module.replace(".", "/") + ".py")
                    if target.is_file():
                        assert str(target.relative_to(ROOT)) in artifacts, (name, path, module)
        assert "tests/distributed/conftest.py" in artifacts
        if name.startswith("wan-cp-dmd2"):
            # The recipe imports the DMD2 observer only on the concrete DMD2 path.
            assert {
                "tests/distributed/online_observations.py",
                "tests/distributed/dmd2_objective_reference.py",
            } <= artifacts
        if name.startswith("wan-"):
            assert "tests/distributed/wan_sft_fixture.py" in artifacts
            assert "tests/distributed/wan_online_fixture.py" in artifacts
        if "evaluation" in name:
            assert "tests/distributed/wan_evaluation_fixture.py" in artifacts
            assert "tests/distributed/test_context_parallel_online.py" not in artifacts
            assert "tests/distributed/online_optimizer_observations.py" not in artifacts


def _real_source_requirements(path, before, after):
    assert before != after
    return planner.infer_requirements(
        {"paths": [path], "facets": [], "scope_complete": True},
        yaml.safe_load((ROOT / planner.POLICY).read_text()),
        {path: planner.changed_symbols(before, after)},
    )


def _insert_owner_probe(source, owner, method):
    import ast

    tree = ast.parse(source)
    if owner:
        tree = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == owner
        )
    node = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == method
    )
    lines = source.splitlines(keepends=True)
    indent = " " * node.body[0].col_offset
    lines.insert(node.body[0].end_lineno, indent + "_validation_owner_probe = 1\n")
    return "".join(lines)


@pytest.mark.parametrize("change", ["layout", "version", "default", "decorator"])
def test_actual_model_capability_metadata_keeps_conservative_native_routes(change):
    path = "src/flow_factory/models/context_parallel.py"
    before = (ROOT / path).read_text()
    replacements = {
        "layout": (
            'Literal["wan_dense_video", "flux_joint_image_text"]',
            'Literal["wan_dense_video"]',
        ),
        "version": ("version: int = 1", "version: int = 2"),
        "default": (
            'transport: Literal["ulysses"] = "ulysses"',
            'transport: Literal["ulysses"] = "other"',
        ),
        "decorator": ("@dataclass(frozen=True)", "@dataclass(frozen=False)"),
    }
    old, new = replacements[change]
    required = _real_source_requirements(path, before, before.replace(old, new, 1))["obligations"]
    assert {
        "preparation.cp_probe",
        "parallel.cp_sft",
        "preparation.tp_cp_probe",
        "parallel.tp_cp_sft",
    } <= set(required)
    assert {"overlap.native.flux.guard_ordered", "overlap.native.flux.dppo_ready"} <= set(required)


@pytest.mark.parametrize(
    "path,owner,method,expected",
    [
        (
            "src/flow_factory/models/context_parallel.py",
            None,
            "validate_prepared_context_parallel",
            {
                "preparation.cp_probe",
                "parallel.cp_sft",
                "preparation.tp_cp_probe",
                "parallel.tp_cp_sft",
            },
        ),
        (
            "src/flow_factory/models/context_parallel.py",
            None,
            "compose_wan_sequence_boundaries",
            {
                "preparation.cp_probe",
                "parallel.cp_sft",
                "preparation.tp_cp_probe",
                "parallel.tp_cp_sft",
            },
        ),
        (
            "src/flow_factory/trainers/common/model_preparation.py",
            "ModelPreparation",
            "prepare",
            {
                "checkpoint.cp_state.tp1",
                "checkpoint.cp_state.tp2",
                "algorithm.cp.dmd2.tp1",
                "algorithm.cp.dmd2.tp2",
            },
        ),
        (
            "src/flow_factory/trainers/common/context_parallel.py",
            None,
            "full_output_context_parallel_loss",
            {
                "checkpoint.cp_state.tp1",
                "checkpoint.cp_state.tp2",
                "algorithm.cp.dmd2.tp1",
                "algorithm.cp.dmd2.tp2",
                "parallel.cp_sft",
                "parallel.tp_cp_sft",
            },
        ),
        (
            "src/flow_factory/trainers/common/runtime_identity.py",
            None,
            "_trainer_identity_payload",
            {"checkpoint.cp_state.tp1", "checkpoint.cp_state.tp2"},
        ),
        (
            "src/flow_factory/trainers/distillation/dmd2.py",
            "DMD2Trainer",
            "_fake_replay_loss",
            {"algorithm.cp.dmd2.tp1", "algorithm.cp.dmd2.tp2"},
        ),
    ],
)
def test_actual_runtime_preparation_normalization_and_state_routes_remain(
    path, owner, method, expected
):
    before = (ROOT / path).read_text()
    required = _real_source_requirements(path, before, _insert_owner_probe(before, owner, method))
    assert expected <= set(required["obligations"])


@pytest.mark.parametrize(
    "method", [None, "sample", "_guard_ratio", "_scalar_component_statistic", "optimize"]
)
def test_actual_guard_owner_selects_coupled_objective_and_flux_intersection(method):
    path = "src/flow_factory/trainers/rl/grpo.py"
    current = (ROOT / path).read_text()
    if method is None:
        declaration = "    context_parallel_training_capability: ClassVar[ContextParallelTrainingCapability] = (\n        ContextParallelTrainingCapability()\n    )\n\n"
        prefix, guard = current.split("class GRPOGuardTrainer", 1)
        before, after = (
            prefix + "class GRPOGuardTrainer" + guard.replace(declaration, "", 1),
            current,
        )
    else:
        before, after = current, _insert_owner_probe(current, "GRPOGuardTrainer", method)
    required = _real_source_requirements(path, before, after)
    assert {
        "algorithm.coupled_objectives",
        "optimization.full_output_backward",
        "overlap.native.flux.guard_ordered",
    } <= set(required["obligations"])
    cases, missing = planner.select_cases(
        yaml.safe_load((ROOT / planner.CATALOG).read_text()), required, []
    )
    assert not missing
    assert {case["id"] for case in cases if case["selected"] and case["resources"]["gpus"]} == {
        "flux-overlap-native-d"
    }


@pytest.mark.parametrize(
    "method", ["__init__", "enable_kl_loss", "sample", "sample_batch", "optimize"]
)
def test_actual_grpo_base_owner_retains_inherited_guard_and_dppo_consumers(method):
    path = "src/flow_factory/trainers/rl/grpo.py"
    before = (ROOT / path).read_text()
    required = _real_source_requirements(
        path, before, _insert_owner_probe(before, "GRPOTrainer", method)
    )["obligations"]
    assert {
        "algorithm.cp.grpo.tp1",
        "algorithm.cp.grpo.tp2",
        "overlap.native.wan.ordered",
        "overlap.native.wan.tp_ready",
        "overlap.native.wan.subgroup_ready",
        "overlap.native.flux.guard_ordered",
        "overlap.native.flux.dppo_ready",
    } <= set(required)
    assert not any(name.startswith("algorithm.cp.dmd2.") for name in required)


@pytest.mark.parametrize(
    "addition,unknown",
    [
        ("\ndef unreviewed_helper():\n    return 1\n", "unreviewed_helper"),
        ("\nUNREVIEWED_SETTING = 1\n", "<module>"),
    ],
)
@pytest.mark.parametrize("mixed", [False, True])
def test_actual_unknown_grpo_changes_stay_visible_or_conservative(addition, unknown, mixed):
    path = "src/flow_factory/trainers/rl/grpo.py"
    before = (ROOT / path).read_text()
    after = _insert_owner_probe(before, "GRPOGuardTrainer", "_guard_ratio") if mixed else before
    required = _real_source_requirements(path, before, after + addition)
    if unknown == "<module>":
        assert {
            "algorithm.cp.grpo.tp1",
            "algorithm.cp.grpo.tp2",
            "algorithm.cp.dmd2.tp1",
            "algorithm.cp.dmd2.tp2",
            "overlap.native.flux.guard_ordered",
            "overlap.native.flux.dppo_ready",
        } <= set(required["obligations"])
    else:
        obligation = f"review.symbol:{path}:{unknown}"
        assert obligation in required["obligations"]
        _, missing = planner.select_cases(
            yaml.safe_load((ROOT / planner.CATALOG).read_text()), required, []
        )
        assert obligation in missing


@pytest.mark.parametrize(
    "case_id",
    [
        "collective-settling",
        "tp-reward-agreement",
        "model-reward-agreement-tp1-cp2",
        "model-reward-agreement-tp2-cp2",
        "wan-fsdp2-tp",
        "preparation-fsdp2-tp",
        "preparation-cp-probe",
        "preparation-ddp",
        "preparation-fsdp2",
        "preparation-zero2",
        "wan-sft-cp",
        "preparation-tp-cp-probe",
        "wan-sft-tp-cp",
        "wan-cp-state-tp1",
        "wan-cp-state-tp2",
    ],
)
def test_remaining_distributed_consumers_bind_top_level_local_imports(case_id):
    import ast

    catalog = yaml.safe_load((ROOT / planner.CATALOG).read_text())
    case = next(case for case in catalog["cases"] if case["id"] == case_id)
    assert "tests/distributed/conftest.py" in case["artifacts"]
    modules = [
        argument.split("::")[0]
        for argument in case["command"]
        if argument.split("::")[0].endswith(".py")
    ]
    assert modules and all(
        any(
            isinstance(node, ast.ImportFrom) and node.module == "conftest"
            for node in ast.parse((ROOT / path).read_text()).body
        )
        for path in modules
    )
    directory = ROOT / "tests/distributed"
    artifacts = set(case["artifacts"])
    for path in artifacts:
        if not path.endswith(".py"):
            continue
        for node in ast.parse((ROOT / path).read_text()).body:
            imported = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
            )
            for module in imported:
                target = directory / (module.replace(".", "/") + ".py")
                if target.is_file():
                    assert str(target.relative_to(ROOT)) in artifacts, (case_id, path, module)


@pytest.mark.parametrize("case_id", ["flux-overlap-native-d", "flux-overlap-native-e"])
def test_flux_native_catalog_binds_actual_transitive_fixture_imports(case_id):
    import ast

    catalog = yaml.safe_load((ROOT / planner.CATALOG).read_text())
    case = next(case for case in catalog["cases"] if case["id"] == case_id)
    artifacts = set(case["artifacts"])
    assert {
        "tests/distributed/context_parallel_overlap_runner.py",
        "tests/distributed/conftest.py",
        "tests/distributed/flux_cp_model_control.py",
        "tests/distributed/flux_sft_fixture.py",
        "tests/distributed/wan_eval_reward.py",
    } <= artifacts
    assert "tests/distributed/flux_online_reward.py" not in artifacts
    assert "tests/distributed/test_flux_packed_dmd2.py" not in artifacts
    directory = ROOT / "tests/distributed"
    for path in artifacts:
        for node in ast.parse((ROOT / path).read_text()).body:
            modules = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
            )
            for module in modules:
                target = directory / (module.replace(".", "/") + ".py")
                if target.is_file():
                    assert str(target.relative_to(ROOT)) in artifacts, (case_id, path, module)


@pytest.mark.parametrize(
    "path,expected",
    [
        ("flux_cp_model_control.py", {"flux-overlap-native-d", "flux-overlap-native-e"}),
        ("flux_sft_fixture.py", {"flux-overlap-native-d", "flux-overlap-native-e"}),
        ("test_flux_packed_dmd2.py", set()),
        ("test_coupled_objective_reference.py", set()),
    ],
)
def test_flux_leaf_artifact_changes_select_genuine_consumers(path, expected):
    path = "tests/distributed/" + path
    scope = {
        "paths": [path],
        "facets": [],
        "scope_complete": True,
        "docs_only_paths": [],
        "logging_only_paths": [],
    }
    catalog = yaml.safe_load((ROOT / planner.CATALOG).read_text())
    required = planner.requirements_for(
        scope, {}, catalog, yaml.safe_load((ROOT / planner.POLICY).read_text())
    )
    selected, missing = planner.select_cases(catalog, required, [])
    assert not missing
    assert {
        case["id"] for case in selected if case["selected"] and case["resources"]["gpus"]
    } == expected
