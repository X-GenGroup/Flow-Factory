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
def test_cp_admission_keeps_unknown_functions_visible_and_module_contracts_required(
    addition, unknown
):
    path = "src/flow_factory/trainers/common/context_parallel.py"
    policy = yaml.safe_load((ROOT / planner.POLICY).read_text())
    before = "def validate_context_parallel_config(config):\n    return False\n"
    after = before.replace("False", "True") + addition
    symbols = planner.changed_symbols(before, after)
    assert "validate_context_parallel_config" in symbols and unknown in symbols
    required = planner.infer_requirements(
        {"paths": [path], "facets": [], "scope_complete": True}, policy, {path: symbols}
    )
    catalog = yaml.safe_load((ROOT / "config/validation/catalog.yaml").read_text())
    _, missing = planner.select_cases(catalog, required, [])
    if unknown == "<module>":
        assert not missing
        assert {
            "parallel.cp_contracts",
            "overlap.cp_admission",
            "optimization.full_output_backward",
            "overlap.native.wan.ordered",
            "overlap.native.wan.tp_ready",
            "overlap.native.wan.subgroup_ready",
        } <= set(required["obligations"])
    else:
        assert f"review.symbol:{path}:{unknown}" in required["obligations"]
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
    assert native == (
        {"flux-overlap-native-e"}
        if trainer == "DPPOTrainer"
        else (
            {"wan-nft-overlap-native", "flux-nft-overlap-native"}
            if trainer == "DiffusionNFTTrainer"
            else {"wan-tdm-r1-cp-lifecycle"} if trainer == "TDMR1Trainer" else set()
        )
    )


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


def _large_bundle(plan, root, content=None):
    result = passed(plan, root)
    log = root / ".scratch/probe.log"
    log.write_text(content if content is not None else "native assertion passed\n" * 4000)
    result["cases"][0]["evidence"] = {
        ".scratch/probe.log": hashlib.sha256(log.read_bytes()).hexdigest()
    }
    return result, transport.make_bundle(root, plan, result)


def _compressed_envelope(plan, raw):
    import base64
    import gzip

    return {
        "schema_version": 2,
        "base_commit": plan["base_commit"],
        "tested_commit": plan["tested_commit"],
        "encoding": "gzip+base64",
        "decoded_bytes": len(raw),
        "decoded_sha256": hashlib.sha256(raw).hexdigest(),
        "payload": base64.b64encode(gzip.compress(raw, mtime=0)).decode("ascii"),
    }


def test_compressed_bundle_preserves_complete_results_files_and_dispatch_identity(plan, tmp_path):
    result, bundle = _large_bundle(plan, tmp_path)
    assert bundle["schema_version"] == 2
    assert bundle["base_commit"] == plan["base_commit"]
    assert bundle["tested_commit"] == plan["tested_commit"]
    assert bundle["decoded_bytes"] > transport.MAX_BUNDLE_BYTES
    assert transport.make_bundle(tmp_path, plan, result) == bundle
    destination = tmp_path / "bundle.json"
    planner.write_json(destination, bundle)
    assert destination.stat().st_size <= transport.MAX_BUNDLE_BYTES
    imported = transport.import_bundle(tmp_path, plan, json.loads(destination.read_text()))
    transport.verify_files(tmp_path, imported)
    original = copy.deepcopy(result)
    original["cases"][0]["evidence"] = imported["cases"][0]["evidence"]
    assert imported == original
    log = next(iter(imported["cases"][0]["evidence"]))
    assert (tmp_path / log).read_bytes() == (tmp_path / ".scratch/probe.log").read_bytes()
    assert planner.source_plan(tmp_path, plan, bundle) is plan
    assert planner.report(plan, [imported], plan["tested_commit"])[0]["status"] == "pass"


def test_small_bundle_retains_ordinary_schema1_document(plan, tmp_path):
    result = passed(plan, tmp_path)
    expected = {
        "schema_version": 1,
        "base_commit": plan["base_commit"],
        "tested_commit": plan["tested_commit"],
        "results": result,
        "files": {".scratch/probe.log": "one assertion passed\n"},
    }
    assert transport.make_bundle(tmp_path, plan, result) == expected


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("schema_version", True, "source identity"),
        ("schema_version", 3, "source identity"),
        ("base_commit", "c" * 40, "source identity"),
        ("tested_commit", "c" * 40, "source identity"),
        ("encoding", "zlib+base64", "envelope"),
        ("decoded_bytes", True, "invalid length"),
        ("decoded_bytes", 0, "invalid length"),
        ("decoded_bytes", transport.MAX_DECODED_BUNDLE_BYTES + 1, "exceeds"),
        ("decoded_bytes", 7, "length mismatch"),
        ("decoded_sha256", "d" * 64, "hash mismatch"),
        ("payload", "not base64!", "base64"),
        ("payload", "é", "base64"),
        ("payload", 42, "base64"),
        ("extra", "undeclared", "envelope"),
    ],
)
def test_compressed_envelope_rejects_invalid_fields_before_writes(
    plan, tmp_path, field, value, error
):
    _, bundle = _large_bundle(plan, tmp_path)
    bundle[field] = value
    with pytest.raises(ValueError, match=error):
        transport.import_bundle(tmp_path, plan, bundle)
    assert not (tmp_path / ".scratch/validation-evidence").exists()


@pytest.mark.parametrize(
    "mutation", ["truncated", "trailing", "second-member", "bad-crc", "not-gzip", "base64-padding"]
)
def test_compressed_bundle_rejects_malformed_single_member_stream(plan, tmp_path, mutation):
    import base64
    import gzip

    _, bundle = _large_bundle(plan, tmp_path)
    raw = base64.b64decode(bundle["payload"])
    if mutation == "truncated":
        raw = raw[:-1]
    elif mutation == "trailing":
        raw += b"trailing"
    elif mutation == "second-member":
        raw += gzip.compress(b"second member", mtime=0)
    elif mutation == "bad-crc":
        raw = raw[:-8] + bytes([raw[-8] ^ 1]) + raw[-7:]
    elif mutation == "not-gzip":
        raw = b"not gzip"
    bundle["payload"] = base64.b64encode(raw).decode()
    if mutation == "base64-padding":
        bundle["payload"] += "="
    with pytest.raises(ValueError, match="gzip|base64"):
        transport.import_bundle(tmp_path, plan, bundle)
    assert not (tmp_path / ".scratch/validation-evidence").exists()


def test_compressed_bundle_bounds_actual_expansion_despite_forged_length(plan, tmp_path):
    raw = b"x" * (transport.MAX_DECODED_BUNDLE_BYTES + 1)
    bundle = _compressed_envelope(plan, raw)
    bundle["decoded_bytes"] = transport.MAX_DECODED_BUNDLE_BYTES
    assert len(json.dumps(bundle)) < transport.MAX_BUNDLE_BYTES
    with pytest.raises(ValueError, match="length mismatch"):
        transport.import_bundle(tmp_path, plan, bundle)
    assert not (tmp_path / ".scratch/validation-evidence").exists()


@pytest.mark.parametrize(
    "mutation,error",
    [
        ("source", "inner source"),
        ("nested", "inner source"),
        ("extra", "inner source"),
        ("whitespace", "canonical"),
        ("duplicate", "canonical"),
        ("utf8", "canonical"),
        ("json", "canonical"),
        ("array", "canonical"),
        ("nan", "canonical"),
        ("log", "content mismatch"),
        ("result", "commit"),
    ],
)
def test_compressed_inner_keeps_schema_source_and_log_validation(plan, tmp_path, mutation, error):
    result = passed(plan, tmp_path)
    inner = transport.make_bundle(tmp_path, plan, result)
    if mutation == "source":
        inner["tested_commit"] = "c" * 40
    elif mutation == "nested":
        inner["schema_version"] = 2
    elif mutation == "extra":
        inner["extra"] = "undeclared"
    elif mutation == "log":
        inner["files"][".scratch/probe.log"] = "forged pass"
    elif mutation == "result":
        inner["results"]["tested_commit"] = "c" * 40
    raw = json.dumps(inner, sort_keys=True, separators=(",", ":")).encode()
    if mutation == "whitespace":
        raw = b" " + raw
    elif mutation == "duplicate":
        raw = b'{"schema_version":1,' + raw[1:]
    elif mutation == "utf8":
        raw = b"\xff"
    elif mutation == "json":
        raw = b"{"
    elif mutation == "array":
        raw = b"[]"
    elif mutation == "nan":
        raw = b'{"schema_version":NaN}'
    bundle = _compressed_envelope(plan, raw)
    with pytest.raises(ValueError, match=error):
        transport.import_bundle(tmp_path, plan, bundle)
    assert not (tmp_path / ".scratch/validation-evidence").exists()


def test_bundle_rejects_oversize_decoded_and_incompressible_dispatch(plan, tmp_path):
    import random

    with pytest.raises(ValueError, match="exceeds 2 MiB"):
        _large_bundle(plan, tmp_path, "x" * transport.MAX_DECODED_BUNDLE_BYTES)
    generator = random.Random(19)
    content = "".join(chr(generator.randrange(32, 127)) for _ in range(100000))
    with pytest.raises(ValueError, match="Compressed evidence bundle exceeds 60 KB"):
        _large_bundle(plan, tmp_path, content)


def test_bundle_compresses_when_cli_pretty_document_crosses_dispatch_limit(
    plan, tmp_path, monkeypatch
):
    result = passed(plan, tmp_path)
    ordinary = transport.make_bundle(tmp_path, plan, result)
    compact_size = len(json.dumps(ordinary).encode())
    pretty_size = len(json.dumps(ordinary, indent=2, sort_keys=True).encode()) + 1
    assert compact_size < pretty_size
    monkeypatch.setattr(transport, "MAX_BUNDLE_BYTES", pretty_size - 1)
    compressed = transport.make_bundle(tmp_path, plan, result)
    assert compressed["schema_version"] == 2
    path = tmp_path / "bundle.json"
    planner.write_json(path, compressed)
    assert path.stat().st_size <= transport.MAX_BUNDLE_BYTES


def test_compressed_log_corruption_never_creates_partial_import_files(plan, tmp_path):
    result = passed(plan, tmp_path)
    bundle = transport.make_bundle(tmp_path, plan, result)
    second = ".scratch/second.log"
    bundle["results"]["cases"][0]["evidence"][second] = hashlib.sha256(b"expected").hexdigest()
    bundle["files"][second] = "corrupted"
    raw = json.dumps(bundle, sort_keys=True, separators=(",", ":")).encode()
    with pytest.raises(ValueError, match="content mismatch"):
        transport.import_bundle(tmp_path, plan, _compressed_envelope(plan, raw))
    assert not (tmp_path / ".scratch/validation-evidence").exists()


def test_compressed_bundle_retains_large_selected_case_set_and_artifact_identities(plan, tmp_path):
    plan["cases"] = [case(f"contract-{index}", ["contract"]) for index in range(120)]
    result = passed(plan, tmp_path)
    row = result["cases"][0]
    result["cases"] = [{**copy.deepcopy(row), "id": selected["id"]} for selected in plan["cases"]]
    bundle = transport.make_bundle(tmp_path, plan, result)
    assert bundle["schema_version"] == 2
    imported = transport.import_bundle(tmp_path, plan, bundle)
    assert [row["id"] for row in imported["cases"]] == [
        selected["id"] for selected in plan["cases"]
    ]
    assert imported["plan_sha256"] == result["plan_sha256"]
    assert imported["artifacts"] == result["artifacts"]
    assert planner.report(plan, [imported], plan["tested_commit"])[0]["status"] == "pass"


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


@pytest.mark.parametrize("compressed", [False, True])
def test_fetch_accepts_only_matching_manual_evidence_workflow(
    plan, monkeypatch, tmp_path, compressed
):
    import io
    import zipfile

    expected = (
        _large_bundle(plan, tmp_path)[1]
        if compressed
        else {"schema_version": 1, "example": "bounded payload"}
    )
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
                "scheduler-replay-flux-native",
                "art-av-stream-native",
                "art-flux-ddp-native",
                "art-wan-tp-cp-native",
                "flux-complete-tp-sft",
                "sd35-adaln-tp2-supervised-update",
                "flux2-guard-ordered-full",
                "flux2-klein-dppo-ordered-lora",
                "flux2-tp-sp-cp-reference",
                "ltx2-joint-av-tp-sp-cp-reference",
                "ltx20-t2av-grpo-ordered-lora",
                "ltx23-i2av-grpo-ordered-full",
                "flux-dual-sp-cp-reference",
                "flux-kontext-sp-guard-ordered-full",
                "flux-kontext-guard-ordered-full",
                "flux-kontext-tp-cp-reference",
                "flux-tp-cp-guard-ordered-full",
                "flux-tp-cp-dppo-ready-lora",
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
                "flux-offline-dpo-cp",
                "wan-offline-dpo-cp",
                "wan-nft-overlap-native",
                "flux-nft-overlap-native",
                "qwen-overlap-native",
                "qwen-awm-overlap-native",
                "qwen-crd-overlap-native",
                "qwen-tp-sp-cp-guard-ordered-full",
                "flux-online-dpo-overlap-native",
                "wan-dgpo-overlap-native",
                "wan-cp-opd-tp2-lora",
                "wan-cp-tdm-tp2-full",
                "wan-tdm-r1-cp-lifecycle",
                "wan-i2v-tp-sp-cp-reference",
                "wan-i2v-cp-tdm-tp2-full",
                "qwen-edit-guard-tp-sp-cp-ordered-full",
                "qwen-edit-tp-sp-cp-reference",
                "wan-remaining-tp-sp-cp-reference",
                "wan-i2v_dual-grpo-ordered-full",
                "wan-i2v_expanded-grpo-ordered-full",
                "wan-flf2v-grpo-ordered-lora",
                "qwen21-tp-sp-cp-reference",
                "qwen21-guard-tp-sp-cp-ordered-lora",
                "sd35-dppo-ordered-full",
                "sd35-tp-sp-cp-reference",
                "z_image-guard-ordered-lora",
                "z_image-tp-sp-cp-reference",
                "flux3-distilled-dppo-ordered-lora",
                "flux3-guard-ordered-full",
                "flux3-tp-sp-cp-reference",
                "bagel-guard-tp-sp-cp-ordered-lora",
                "bagel-tp-sp-cp-reference",
                "minimax-h3-fl2va-grpo-ordered-lora",
                "minimax-h3-packed-tp-sp-cp-reference",
                "minimax-h3-ref2va-grpo-ordered-full",
                "sensenova-tp-sp-cp-reference",
                "sensenova-guard-tp-sp-cp-ordered-lora",
                "minimax_h3-refiner-sp-cp-reference",
                "minimax_h3-refiner-sp-cuda",
                "z_image-refiner-sp-cp-reference",
                "z_image-refiner-sp-cuda",
                "flux2-continuous-sp-cp-reference",
                "flux3-continuous-sp-cp-reference",
                "bagel-sp-final-norm-reference",
                "qwen21-sp-output-reference",
                "sensenova-sp-pointwise-head-reference",
                "qwen-edit-text-sp-cp-reference",
                "sd35-text-sp-cp-reference",
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
    path = "src/flow_factory/models/parallelism/context.py"
    before = (ROOT / path).read_text()
    replacements = {
        "layout": (
            "layout: ContextParallelLayout",
            "layout: str",
        ),
        "version": ("version: int = 1", "version: int = 2"),
        "default": (
            'transport: Literal["ulysses", "ring"] = "ulysses"',
            'transport: Literal["ulysses", "ring"] = "other"',
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
            "src/flow_factory/models/parallelism/context.py",
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
            "src/flow_factory/models/wan/_context_parallel.py",
            None,
            "compose_wan_sequence_boundaries",
            {
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


@pytest.mark.parametrize("change", ["boundary", "layout", "unknown"])
def test_wan_owned_cp_boundary_routes_follow_actual_model_reach(change):
    path = "src/flow_factory/models/wan/_context_parallel.py"
    before = (ROOT / path).read_text()
    if change == "boundary":
        after = _insert_owner_probe(before, None, "compose_wan_sequence_boundaries")
    elif change == "layout":
        after = before.replace('DENSE_VIDEO = "wan_dense_video"', 'DENSE_VIDEO = "changed"')
    else:
        after = before + "\ndef unreviewed_boundary():\n    return None\n"
    required = _real_source_requirements(path, before, after)["obligations"]
    if change == "unknown":
        assert f"review.symbol:{path}:unreviewed_boundary" in required
        return
    assert {"parallel.cp_contracts", "preparation.tp_cp_probe", "parallel.tp_cp_sft"} <= set(
        required
    )
    if change == "layout":
        assert {"preparation.cp_probe", "parallel.cp_sft"} <= set(required)
    else:
        assert "preparation.cp_probe" not in required
        assert "parallel.cp_sft" not in required
    assert not any("flux" in name or "qwen" in name for name in required)


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


@pytest.mark.parametrize("method", ["enable_kl_loss", "sample", "sample_batch", "optimize"])
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


@pytest.mark.parametrize(
    "case_id",
    [
        "flux-overlap-native-d",
        "flux-overlap-native-e",
        "flux-tp-cp-guard-ordered-full",
        "flux-tp-cp-dppo-ready-lora",
    ],
)
def test_flux_native_catalog_binds_actual_transitive_fixture_imports(case_id):
    import ast

    catalog = yaml.safe_load((ROOT / planner.CATALOG).read_text())
    case = next(case for case in catalog["cases"] if case["id"] == case_id)
    artifacts = set(case["artifacts"])
    assert {
        "tests/distributed/context_parallel_overlap_runner.py",
        "tests/distributed/conftest.py",
        "tests/distributed/flux_cp_model_control.py",
        "tests/distributed/context_parallel_model_control.py",
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


@pytest.mark.parametrize("case_id", ["flux-tp-cp-guard-ordered-full", "flux-tp-cp-dppo-ready-lora"])
def test_flux_tp_cp_catalog_preserves_cp_identity_and_binds_runtime_topology(
    monkeypatch, tmp_path, case_id
):
    import runpy
    from types import ModuleType

    catalog = yaml.safe_load((ROOT / planner.CATALOG).read_text())
    case = next(row for row in catalog["cases"] if row["id"] == case_id)
    path, function = next(argument for argument in case["command"] if ".py::" in argument).split(
        "::", 1
    )
    captured = []
    runner = ModuleType("context_parallel_overlap_runner")
    runner.qualify_overlap = lambda directory, options: captured.append(options)
    monkeypatch.setitem(sys.modules, runner.__name__, runner)
    runpy.run_path(str(ROOT / path))[function](tmp_path)
    assert len(captured) == 1
    options = captured[0]
    assert options["tp_size"] == options["cp_size"] == options["dp_size"] == 2
    assert case["execution_capability"] == f"flux-{options['algorithm']}-cp-v1"
    validator.validate_topology(case)
    old_id = (
        "flux-overlap-native-d" if options["training_mode"] == "full" else "flux-overlap-native-e"
    )
    old = next(row for row in catalog["cases"] if row["id"] == old_id)
    assert old["execution_capability"] == case["execution_capability"]
    assert old["topology"]["tp"] == 1 and old["topology"]["ranks_per_node"] == 4
    for current, other in ((old, case), (case, old)):
        validator.validate_topology(current)
        world = current["topology"]["ranks_per_node"]
        receipt = {
            "runtime": {
                "world_size": world,
                "ranks_completed": list(range(world)),
                "device": "cuda",
                "distributed_type": "FSDP",
                "fsdp_version": 2,
            }
        }
        validator.validate_runtime(current, receipt)
        with pytest.raises(ValueError, match="observed world_size disagrees"):
            validator.validate_runtime(other, receipt)
    bad = copy.deepcopy(case)
    bad["topology"].update(tp=4, ranks_per_node=16)
    bad["resources"]["gpus"] = 16
    with pytest.raises(ValueError, match="Flux coupled CP evidence requires TP=1 or 2"):
        validator.validate_topology(bad)


@pytest.mark.parametrize(
    "path,expected",
    [
        (
            "flux_cp_model_control.py",
            {
                "flux-complete-tp-sft",
                "flux2-guard-ordered-full",
                "flux2-klein-dppo-ordered-lora",
                "flux-kontext-sp-guard-ordered-full",
                "flux-kontext-guard-ordered-full",
                "flux-tp-cp-guard-ordered-full",
                "flux-tp-cp-dppo-ready-lora",
                "flux-overlap-native-d",
                "flux-overlap-native-e",
                "flux-nft-overlap-native",
                "flux-online-dpo-overlap-native",
                "sd35-dppo-ordered-full",
                "z_image-guard-ordered-lora",
                "flux3-distilled-dppo-ordered-lora",
                "flux3-guard-ordered-full",
                "minimax_h3-refiner-sp-cuda",
                "z_image-refiner-sp-cuda",
            },
        ),
        (
            "flux_sft_fixture.py",
            {
                "scheduler-replay-flux-native",
                "art-flux-ddp-native",
                "native-attention-runtime",
                "flux-complete-tp-sft",
                "flux2-guard-ordered-full",
                "flux2-klein-dppo-ordered-lora",
                "flux-dual-sp-cp-reference",
                "flux-kontext-sp-guard-ordered-full",
                "flux-kontext-guard-ordered-full",
                "flux-kontext-tp-cp-reference",
                "flux-tp-cp-guard-ordered-full",
                "flux-tp-cp-dppo-ready-lora",
                "flux-overlap-native-d",
                "flux-overlap-native-e",
                "flux-offline-dpo-cp",
                "flux-nft-overlap-native",
                "flux-online-dpo-overlap-native",
                "sd35-dppo-ordered-full",
                "z_image-guard-ordered-lora",
                "flux3-distilled-dppo-ordered-lora",
                "flux3-guard-ordered-full",
                "minimax_h3-refiner-sp-cuda",
                "z_image-refiner-sp-cuda",
            },
        ),
        ("test_flux_packed_dmd2.py", set()),
        ("test_coupled_objective_reference.py", set()),
        (
            "qwen_cp_fixture.py",
            {
                "qwen-overlap-native",
                "qwen-awm-overlap-native",
                "qwen-crd-overlap-native",
                "qwen-tp-sp-cp-reference",
                "qwen-tp-sp-cp-guard-ordered-full",
                "qwen-edit-guard-tp-sp-cp-ordered-full",
                "qwen-edit-tp-sp-cp-reference",
                "qwen21-tp-sp-cp-reference",
                "qwen21-guard-tp-sp-cp-ordered-lora",
                "bagel-guard-tp-sp-cp-ordered-lora",
                "sensenova-tp-sp-cp-reference",
                "sensenova-guard-tp-sp-cp-ordered-lora",
                "qwen21-sp-output-reference",
                "sensenova-sp-pointwise-head-reference",
                "qwen-edit-text-sp-cp-reference",
            },
        ),
        (
            "qwen_coupled_fixture.py",
            {
                "qwen-overlap-native",
                "qwen-awm-overlap-native",
                "qwen-crd-overlap-native",
                "qwen-tp-sp-cp-guard-ordered-full",
                "qwen-edit-guard-tp-sp-cp-ordered-full",
                "qwen21-tp-sp-cp-reference",
                "qwen21-guard-tp-sp-cp-ordered-lora",
                "bagel-guard-tp-sp-cp-ordered-lora",
                "sensenova-tp-sp-cp-reference",
                "sensenova-guard-tp-sp-cp-ordered-lora",
                "qwen21-sp-output-reference",
                "sensenova-sp-pointwise-head-reference",
            },
        ),
        (
            "context_parallel_model_control.py",
            {
                "flux-complete-tp-sft",
                "flux2-guard-ordered-full",
                "flux2-klein-dppo-ordered-lora",
                "flux-kontext-sp-guard-ordered-full",
                "flux-kontext-guard-ordered-full",
                "flux-tp-cp-guard-ordered-full",
                "flux-tp-cp-dppo-ready-lora",
                "flux-overlap-native-d",
                "flux-overlap-native-e",
                "flux-nft-overlap-native",
                "flux-online-dpo-overlap-native",
                "qwen-overlap-native",
                "qwen-awm-overlap-native",
                "qwen-crd-overlap-native",
                "qwen-tp-sp-cp-guard-ordered-full",
                "qwen-edit-guard-tp-sp-cp-ordered-full",
                "qwen21-tp-sp-cp-reference",
                "qwen21-guard-tp-sp-cp-ordered-lora",
                "sd35-dppo-ordered-full",
                "z_image-guard-ordered-lora",
                "flux3-distilled-dppo-ordered-lora",
                "flux3-guard-ordered-full",
                "bagel-guard-tp-sp-cp-ordered-lora",
                "sensenova-tp-sp-cp-reference",
                "sensenova-guard-tp-sp-cp-ordered-lora",
                "minimax_h3-refiner-sp-cuda",
                "z_image-refiner-sp-cuda",
                "qwen21-sp-output-reference",
                "sensenova-sp-pointwise-head-reference",
            },
        ),
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


@pytest.fixture
def reuse_repository(tmp_path):
    """Use real Git objects so ancestry, modes and unlisted inputs are exercised."""
    import subprocess

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path, text=True).strip()

    git("init", "-q")
    git("config", "user.name", "Evidence Test")
    git("config", "user.email", "test@example.invalid")
    for path in (
        "roadmap/handoff.md",
        "docs/development/validation/automatic-validation.md",
        "scripts/validation_plan.py",
        "scripts/validation_evidence.py",
        "tests/scripts/test_validation_planner.py",
        "src/model.py",
        "tests/native.py",
        "requirements/train.txt",
        "config/validation/catalog.yaml",
        ".github/workflows/quality.yml",
    ):
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("original\n")
    git("add", ".")
    git("commit", "-qm", "original")
    return tmp_path, git, git("rev-parse", "HEAD")


@pytest.mark.parametrize("path", ["roadmap/handoff.md", *sorted(planner.REUSE_TOOL_PATHS)])
def test_reuse_accepts_only_proven_documentation_and_cpu_tool_edits(reuse_repository, path):
    root, git, source = reuse_repository
    (root / path).write_text("updated\n")
    git("commit", "-qam", "update")
    assert planner.reuse_changes(root, source, git("rev-parse", "HEAD")) == [path]


@pytest.mark.parametrize(
    "path",
    [
        "src/model.py",
        "tests/native.py",
        "requirements/train.txt",
        "config/validation/catalog.yaml",
        ".github/workflows/quality.yml",
    ],
)
def test_reuse_rejects_changed_execution_policy_and_dependency_inputs(reuse_repository, path):
    root, git, source = reuse_repository
    (root / path).write_text("updated\n")
    git("commit", "-qam", "update")
    with pytest.raises(ValueError, match="unchanged execution inputs"):
        planner.reuse_changes(root, source, git("rev-parse", "HEAD"))


@pytest.mark.parametrize("mutation", ["add", "delete", "mode", "symlink"])
def test_reuse_rejects_file_identity_changes(reuse_repository, mutation):
    root, git, source = reuse_repository
    target = root / "roadmap/handoff.md"
    if mutation == "add":
        (root / "roadmap/another.md").write_text("new\n")
    elif mutation == "mode":
        target.chmod(0o755)
    else:
        target.unlink()
        if mutation == "symlink":
            target.symlink_to("../src/model.py")
    git("add", ".")
    git("commit", "-qm", "change identity")
    with pytest.raises(ValueError, match="file-mode changes"):
        planner.reuse_changes(root, source, git("rev-parse", "HEAD"))


def test_reuse_rejects_nonancestor_and_changed_base(reuse_repository, plan):
    import subprocess

    root, git, source = reuse_repository
    (root / "roadmap/handoff.md").write_text("updated\n")
    git("commit", "-qam", "update")
    with pytest.raises(subprocess.CalledProcessError):
        planner.reuse_changes(root, git("rev-parse", "HEAD"), source)
    with pytest.raises(ValueError, match="comparison base"):
        planner.source_plan(root, plan, {"base_commit": "c" * 40})


def test_ancestor_report_preserves_provenance_and_requires_current_cpu_proof(
    plan, tmp_path, monkeypatch
):
    previous = copy.deepcopy(plan)
    original_results = passed(previous, tmp_path)
    current = copy.deepcopy(plan)
    current["tested_commit"] = "c" * 40
    for name in sorted(planner.FRESH_REUSE_CASES):
        current["cases"].append(case(name, ["contract"]))
    fresh = planner.empty_results(current)
    fresh["cases"] = [
        {**copy.deepcopy(original_results["cases"][0]), "id": name}
        for name in sorted(planner.FRESH_REUSE_CASES)
    ]
    monkeypatch.setattr(planner, "reuse_changes", lambda *args: ["roadmap/handoff.md"])
    monkeypatch.setattr(planner, "verify_generated_plan", lambda *args: None)
    original_bytes = json.dumps(original_results, sort_keys=True)
    summary, combined = planner.report(
        current,
        [fresh],
        current["tested_commit"],
        inherited=[(previous, original_results)],
        root=tmp_path,
    )
    assert summary["status"] == "pass"
    assert summary["evidence_reuse"][0]["tested_commit"] == previous["tested_commit"]
    assert summary["evidence_reuse"][0]["cases"] == ["contract-test"]
    assert combined["cases"][-1]["source_commit"] == previous["tested_commit"]
    assert json.dumps(original_results, sort_keys=True) == original_bytes
    with pytest.raises(ValueError, match="fresh harness"):
        planner.report(
            current,
            [],
            current["tested_commit"],
            inherited=[(previous, original_results)],
            root=tmp_path,
        )
    failed = copy.deepcopy(fresh)
    failed["cases"].append({"id": "contract-test", "status": "fail", "reason": "real failure"})
    with pytest.raises(ValueError, match="Conflicting ancestor"):
        planner.report(
            current,
            [failed],
            current["tested_commit"],
            inherited=[(previous, original_results)],
            root=tmp_path,
        )
    current["cases"][0]["command"] = ["python3", "changed.py"]
    with pytest.raises(ValueError, match="case changed"):
        planner.inherited_results(tmp_path, current, previous, original_results)


def test_reuse_rejects_declared_tool_artifact_changes_and_tampered_logs(
    plan, tmp_path, monkeypatch
):
    previous = copy.deepcopy(plan)
    result = passed(previous, tmp_path)
    current = copy.deepcopy(plan)
    current["tested_commit"] = "c" * 40
    monkeypatch.setattr(planner, "verify_generated_plan", lambda *args: None)
    monkeypatch.setattr(planner, "reuse_changes", lambda *args: ["scripts/validation_plan.py"])
    for item in (previous, current):
        item["cases"][0]["artifacts"] = ["scripts/validation_plan.py"]
    result["plan_sha256"] = validator.plan_sha256(previous)
    with pytest.raises(ValueError, match="test/config changed"):
        planner.inherited_results(tmp_path, current, previous, result)
    (tmp_path / ".scratch/probe.log").write_text("tampered\n")
    with pytest.raises(ValueError, match="content mismatch"):
        planner.inherited_results(tmp_path, current, previous, result)


def test_fetch_prefers_exact_and_stops_at_changed_execution_inputs(plan, monkeypatch):
    calls = []
    monkeypatch.setattr(
        transport, "fetch_bundle", lambda repo, p: calls.append(p) or {"exact": True}
    )
    assert planner.fetch_evidence(ROOT, "owner/repo", plan) == {"exact": True}
    assert len(calls) == 1
    monkeypatch.setattr(transport, "fetch_bundle", lambda *args: None)
    monkeypatch.setattr(planner.agent_diff, "git", lambda *args: b"head\nprevious\nolder\n")
    visited = []

    def reject(*args):
        visited.append(args[-1]["tested_commit"])
        raise ValueError("changed executable input")

    monkeypatch.setattr(planner, "source_plan", reject)
    assert planner.fetch_evidence(ROOT, "owner/repo", plan) is None
    assert visited == ["previous"]


@pytest.mark.parametrize(
    "path,expected",
    [
        (
            "src/flow_factory/trainers/offline/offline_dpo.py",
            {"flux-offline-dpo-cp", "wan-offline-dpo-cp"},
        ),
        (
            "tests/distributed/offline_dpo_observations.py",
            {"flux-offline-dpo-cp", "wan-offline-dpo-cp", "flux-online-dpo-overlap-native"},
        ),
    ],
)
def test_offline_preference_leaf_selects_its_cross_model_witnesses(path, expected):
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


def test_sparse_checkpoint_helper_invalidates_cpu_and_native_evidence():
    path = "src/flow_factory/trainers/common/fsdp_optimizer_checkpoint.py"
    catalog = yaml.safe_load((ROOT / planner.CATALOG).read_text())
    scope = {"paths": [path], "facets": [], "scope_complete": True}
    requirements = planner.requirements_for(
        scope, {}, catalog, yaml.safe_load((ROOT / planner.POLICY).read_text())
    )
    selected, missing = planner.select_cases(catalog, requirements, [])
    assert not missing
    expected = {"sparse-adamw-checkpoint-cpu", "sparse-adamw-checkpoint-fsdp2"}
    assert {case["id"] for case in selected if case["selected"]} == expected | {
        "repository-consistency"
    }
    for case in catalog["cases"]:
        if case["id"] in expected:
            assert path in case["artifacts"]


def test_accelerator_sparse_load_keeps_generic_loader_regressions():
    path = "src/flow_factory/trainers/loader.py"
    catalog = yaml.safe_load((ROOT / planner.CATALOG).read_text())
    scope = {"paths": [path], "facets": [], "scope_complete": True}
    requirements = planner.requirements_for(
        scope,
        {path: ["Accelerator", "Accelerator.load_state"]},
        catalog,
        yaml.safe_load((ROOT / planner.POLICY).read_text()),
    )
    selected, missing = planner.select_cases(catalog, requirements, [])
    assert not missing
    assert {
        "sparse-adamw-checkpoint-cpu",
        "sparse-adamw-checkpoint-fsdp2",
        "roles-ddp",
        "roles-fsdp2",
        "roles-zero2",
    } <= {case["id"] for case in selected if case["selected"]}


@pytest.mark.parametrize(
    "path,expected",
    [
        ("wan_nft_fixture.py", {"wan-nft-overlap-native"}),
        ("flux_nft_fixture.py", {"flux-nft-overlap-native"}),
        ("nft_objective_reference.py", {"wan-nft-overlap-native", "flux-nft-overlap-native"}),
        (
            "context_parallel_overlap_runner.py",
            {
                "scheduler-replay-flux-native",
                "flux-complete-tp-sft",
                "sd35-adaln-tp2-supervised-update",
                "flux2-guard-ordered-full",
                "flux2-klein-dppo-ordered-lora",
                "flux2-tp-sp-cp-reference",
                "ltx2-joint-av-tp-sp-cp-reference",
                "ltx20-t2av-grpo-ordered-lora",
                "ltx23-i2av-grpo-ordered-full",
                "flux-dual-sp-cp-reference",
                "flux-kontext-sp-guard-ordered-full",
                "flux-kontext-guard-ordered-full",
                "flux-kontext-tp-cp-reference",
                "flux-tp-cp-guard-ordered-full",
                "flux-tp-cp-dppo-ready-lora",
                "wan-overlap-native-a",
                "wan-overlap-native-b",
                "wan-overlap-native-c",
                "flux-overlap-native-d",
                "flux-overlap-native-e",
                "wan-nft-overlap-native",
                "flux-nft-overlap-native",
                "qwen-overlap-native",
                "qwen-awm-overlap-native",
                "qwen-crd-overlap-native",
                "qwen-tp-sp-cp-guard-ordered-full",
                "flux-online-dpo-overlap-native",
                "wan-dgpo-overlap-native",
                "qwen-edit-guard-tp-sp-cp-ordered-full",
                "qwen-edit-tp-sp-cp-reference",
                "wan-remaining-tp-sp-cp-reference",
                "wan-i2v_dual-grpo-ordered-full",
                "wan-i2v_expanded-grpo-ordered-full",
                "wan-flf2v-grpo-ordered-lora",
                "qwen21-tp-sp-cp-reference",
                "qwen21-guard-tp-sp-cp-ordered-lora",
                "sd35-dppo-ordered-full",
                "sd35-tp-sp-cp-reference",
                "z_image-guard-ordered-lora",
                "z_image-tp-sp-cp-reference",
                "flux3-distilled-dppo-ordered-lora",
                "flux3-guard-ordered-full",
                "flux3-tp-sp-cp-reference",
                "bagel-guard-tp-sp-cp-ordered-lora",
                "bagel-tp-sp-cp-reference",
                "minimax-h3-fl2va-grpo-ordered-lora",
                "minimax-h3-packed-tp-sp-cp-reference",
                "minimax-h3-ref2va-grpo-ordered-full",
                "sensenova-tp-sp-cp-reference",
                "sensenova-guard-tp-sp-cp-ordered-lora",
                "minimax_h3-refiner-sp-cp-reference",
                "minimax_h3-refiner-sp-cuda",
                "z_image-refiner-sp-cp-reference",
                "z_image-refiner-sp-cuda",
                "flux2-continuous-sp-cp-reference",
                "flux3-continuous-sp-cp-reference",
                "bagel-sp-final-norm-reference",
                "qwen21-sp-output-reference",
                "sensenova-sp-pointwise-head-reference",
                "qwen-edit-text-sp-cp-reference",
                "sd35-text-sp-cp-reference",
            },
        ),
    ],
)
def test_nft_fixture_routing_selects_actual_consumers(path, expected):
    scope = {
        "paths": ["tests/distributed/" + path],
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


@pytest.mark.parametrize(
    "case_id", ["nft-objective-contracts", "wan-nft-overlap-native", "flux-nft-overlap-native"]
)
def test_nft_catalog_binds_transitive_fixture_imports(case_id):
    import ast

    catalog = yaml.safe_load((ROOT / planner.CATALOG).read_text())
    case = next(row for row in catalog["cases"] if row["id"] == case_id)
    artifacts = set(case["artifacts"])
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
    "path,symbol",
    [
        ("src/flow_factory/trainers/abc.py", "BaseTrainer._backward_full_output_loss"),
        (
            "src/flow_factory/trainers/common/context_parallel.py",
            "full_output_context_parallel_loss",
        ),
        ("src/flow_factory/trainers/decoupled/runtime.py", "iter_decoupled_replay_batches"),
        ("src/flow_factory/trainers/common/group_noise.py", "shared_group_noise"),
    ],
)
def test_shared_output_and_noising_owners_preserve_nft_intersections(path, symbol):
    policy = yaml.safe_load((ROOT / planner.POLICY).read_text())
    scope = {"paths": [path], "facets": ["objective"], "scope_complete": True}
    required = planner.infer_requirements(scope, policy, {path: [symbol]})
    assert {
        "algorithm.nft_objectives",
        "algorithm.nft.wan_overlap",
        "algorithm.nft.flux_overlap",
    } <= set(required["obligations"])


@pytest.mark.parametrize("case_id", ["qwen-cp-contracts", "qwen-overlap-native"])
def test_qwen_catalog_binds_transitive_fixture_imports(case_id):
    import ast

    catalog = yaml.safe_load((ROOT / planner.CATALOG).read_text())
    case = next(case for case in catalog["cases"] if case["id"] == case_id)
    artifacts = set(case["artifacts"])
    assert {
        "tests/distributed/context_parallel_model_control.py",
        "tests/distributed/qwen_cp_fixture.py",
        "tests/distributed/qwen_coupled_fixture.py",
        "tests/distributed/online_optimizer_observations.py",
    } <= artifacts
    directory = ROOT / "tests/distributed"
    for path in artifacts:
        for node in ast.parse((ROOT / path).read_text()).body:
            if isinstance(node, ast.ImportFrom) and node.module:
                target = directory / (node.module.replace(".", "/") + ".py")
                if target.is_file():
                    assert str(target.relative_to(ROOT)) in artifacts, (path, node.module)


def test_qwen_adapter_owner_selects_masked_native_training_and_objective_contracts():
    path = "src/flow_factory/models/qwen_image/qwen_image.py"
    before = (ROOT / path).read_text()
    after = _insert_owner_probe(before, "QwenImageAdapter", "forward")
    required = _real_source_requirements(path, before, after)["obligations"]
    assert {
        "parallel.qwen_cp_contracts",
        "overlap.native.qwen.guard_ordered",
        "algorithm.coupled_objectives",
        "optimization.full_output_backward",
    } <= set(required)


def test_qwen_catalog_capability_matches_native_recipe_options(monkeypatch, tmp_path):
    import runpy
    from types import ModuleType

    catalog = yaml.safe_load((ROOT / planner.CATALOG).read_text())
    case = next(row for row in catalog["cases"] if row["id"] == "qwen-overlap-native")
    selector = next(argument for argument in case["command"] if ".py::" in argument)
    path, function = selector.split("::", 1)
    captured = []
    runner = ModuleType("context_parallel_overlap_runner")
    runner.qualify_overlap = lambda directory, options: captured.append(options)
    monkeypatch.setitem(sys.modules, runner.__name__, runner)

    # Execute the actual leaf's recipe construction without starting CUDA workers.
    # The runtime capability is composed from these options, not the catalog alias.
    runpy.run_path(str(ROOT / path))[function](tmp_path)
    assert len(captured) == 1
    options = captured[0]
    expected = f"{options['model_family']}-{options['algorithm']}-cp-v1"
    assert case["execution_capability"] == expected
    validator.validate_topology(case)


@pytest.mark.parametrize("case_id", ["awm-objective-contracts", "qwen-awm-overlap-native"])
def test_awm_catalog_binds_transitive_fixture_imports(case_id):
    import ast

    catalog = yaml.safe_load((ROOT / planner.CATALOG).read_text())
    case = next(row for row in catalog["cases"] if row["id"] == case_id)
    artifacts = set(case["artifacts"])
    for path in artifacts:
        for node in ast.walk(ast.parse((ROOT / path).read_text())):
            modules = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
            )
            for module in modules:
                for target in (
                    ROOT / "tests/distributed" / (module.replace(".", "/") + ".py"),
                    ROOT / (module.replace(".", "/") + ".py"),
                ):
                    if target.is_file() and target.is_relative_to(ROOT / "tests"):
                        assert str(target.relative_to(ROOT)) in artifacts, (case_id, path, module)


def test_awm_catalog_capability_matches_real_recipe_and_rejects_invalid_composition(
    monkeypatch, tmp_path
):
    import runpy
    from types import ModuleType

    catalog = yaml.safe_load((ROOT / planner.CATALOG).read_text())
    case = next(row for row in catalog["cases"] if row["id"] == "qwen-awm-overlap-native")
    path, function = next(argument for argument in case["command"] if ".py::" in argument).split(
        "::", 1
    )
    captured = []
    runner = ModuleType("context_parallel_overlap_runner")
    runner.qualify_overlap = lambda directory, options: captured.append(options)
    monkeypatch.setitem(sys.modules, runner.__name__, runner)
    runpy.run_path(str(ROOT / path))[function](tmp_path)
    assert len(captured) == 1
    options = captured[0]
    assert case["execution_capability"] == f"{options['model_family']}-{options['algorithm']}-cp-v1"
    validator.validate_topology(case)
    for key, value, message in (
        ("execution_capability", "qwen-awm-cp-v1", "declared supported"),
        ("backend", "ddp", "FSDP2"),
    ):
        bad = copy.deepcopy(case)
        bad[key] = value
        with pytest.raises(ValueError, match=message):
            validator.validate_topology(bad)
    bad = copy.deepcopy(case)
    bad["topology"].update(tp=2, ranks_per_node=8)
    bad["resources"]["gpus"] = 8
    with pytest.raises(ValueError, match="Qwen CP evidence requires TP=1"):
        validator.validate_topology(bad)


@pytest.mark.parametrize(
    "path,symbol",
    [
        ("src/flow_factory/trainers/rl/awm.py", "AWMTrainer.optimize"),
        ("src/flow_factory/trainers/abc.py", "BaseTrainer._backward_full_output_loss"),
        (
            "src/flow_factory/trainers/common/context_parallel.py",
            "full_output_context_parallel_loss",
        ),
        ("src/flow_factory/trainers/decoupled/runtime.py", "iter_decoupled_replay_batches"),
        ("src/flow_factory/trainers/common/group_noise.py", "shared_group_noise"),
        ("src/flow_factory/models/qwen_image/qwen_image.py", "QwenImageAdapter.forward"),
    ],
)
def test_awm_owner_changes_preserve_independent_objective_and_native_witness(path, symbol):
    policy = yaml.safe_load((ROOT / planner.POLICY).read_text())
    scope = {"paths": [path], "facets": ["objective"], "scope_complete": True}
    required = planner.infer_requirements(scope, policy, {path: [symbol]})
    assert {"algorithm.awm_objectives", "algorithm.awm.qwen_overlap"} <= set(
        required["obligations"]
    )


@pytest.mark.parametrize(
    "path", ["qwen_awm_fixture.py", "awm_observations.py", "awm_objective_reference.py"]
)
def test_awm_leaf_artifact_changes_select_only_its_native_consumer(path):
    scope = {
        "paths": ["tests/distributed/" + path],
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
    assert {row["id"] for row in selected if row["selected"] and row["resources"]["gpus"]} == {
        "qwen-awm-overlap-native"
    }


def test_online_dpo_leaf_selects_pair_proofs_and_one_native_intersection():
    path = "src/flow_factory/trainers/rl/dpo.py"
    catalog = yaml.safe_load((ROOT / planner.CATALOG).read_text())
    scope = {"paths": [path], "facets": [], "scope_complete": True}
    required = planner.requirements_for(
        scope, {}, catalog, yaml.safe_load((ROOT / planner.POLICY).read_text())
    )
    selected, missing = planner.select_cases(catalog, required, [])
    assert not missing
    names = {row["id"] for row in selected if row["selected"]}
    assert {
        "online-dpo-cp-contracts",
        "online-dpo-logical-pairs",
        "flux-online-dpo-overlap-native",
    } <= names
    assert {row["id"] for row in selected if row["selected"] and row["resources"]["gpus"]} == {
        "flux-online-dpo-overlap-native"
    }


@pytest.mark.parametrize(
    "case_id",
    ["online-dpo-cp-contracts", "online-dpo-logical-pairs", "flux-online-dpo-overlap-native"],
)
def test_online_dpo_catalog_binds_transitive_observers(case_id):
    import ast

    catalog = yaml.safe_load((ROOT / planner.CATALOG).read_text())
    case = next(row for row in catalog["cases"] if row["id"] == case_id)
    artifacts = set(case["artifacts"])
    for path in artifacts:
        for node in ast.walk(ast.parse((ROOT / path).read_text())):
            modules = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
            )
            for module in modules:
                target = ROOT / "tests/distributed" / (module.replace(".", "/") + ".py")
                if target.is_file():
                    assert str(target.relative_to(ROOT)) in artifacts, (case_id, path, module)


def test_online_dpo_catalog_capability_matches_actual_native_recipe(monkeypatch, tmp_path):
    from types import ModuleType

    catalog = yaml.safe_load((ROOT / planner.CATALOG).read_text())
    case = next(row for row in catalog["cases"] if row["id"] == "flux-online-dpo-overlap-native")
    selector = next(argument for argument in case["command"] if ".py::" in argument)
    path, function = selector.split("::", 1)
    captured = []
    runner = ModuleType("context_parallel_overlap_runner")
    runner.qualify_overlap = lambda directory, options: captured.append(options)
    monkeypatch.setitem(sys.modules, runner.__name__, runner)
    # Importing the CPU fixture definitions would load the training stack. Execute
    # the actual native function AST in isolation; its recipe body is unchanged.
    import ast

    module = ast.parse((ROOT / path).read_text())
    node = next(
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == function
    )
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(ROOT / path), "exec"), namespace)
    namespace[function](tmp_path)
    assert len(captured) == 1
    options = captured[0]
    assert case["execution_capability"] == f"{options['model_family']}-{options['algorithm']}-cp-v1"
    assert (
        case["topology"]["ranks_per_node"]
        == options["dp_size"] * options["cp_size"] * options["tp_size"]
    )
    validator.validate_topology(case)
    incompatible = copy.deepcopy(case)
    incompatible["topology"].update(tp=2, ranks_per_node=8)
    with pytest.raises(ValueError, match="Flux CP evidence requires TP=1"):
        validator.validate_topology(incompatible)


@pytest.mark.parametrize(
    "case_id",
    ["dgpo-objective-contracts", "dgpo-logical-group-contracts", "wan-dgpo-overlap-native"],
)
def test_dgpo_catalog_binds_transitive_fixture_imports(case_id):
    import ast

    catalog = yaml.safe_load((ROOT / planner.CATALOG).read_text())
    case = next(row for row in catalog["cases"] if row["id"] == case_id)
    artifacts = set(case["artifacts"])
    for path in artifacts:
        for node in ast.walk(ast.parse((ROOT / path).read_text())):
            modules = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
            )
            for module in modules:
                target = (
                    ROOT / (module.replace(".", "/") + ".py")
                    if module.startswith("tests.")
                    else ROOT / "tests/distributed" / (module.replace(".", "/") + ".py")
                )
                if target.is_file():
                    assert str(target.relative_to(ROOT)) in artifacts, (case_id, path, module)


def test_dgpo_leaf_selects_one_native_and_logical_scope_witness():
    path = "src/flow_factory/trainers/rl/dgpo.py"
    before = (ROOT / path).read_text()
    after = _insert_owner_probe(before, "DGPOTrainer", "_compute_group_dgpo_loss")
    required = _real_source_requirements(path, before, after)
    assert {
        "algorithm.dgpo_objectives",
        "algorithm.dgpo.logical_groups",
        "algorithm.dgpo.wan_overlap",
    } <= set(required["obligations"])
    catalog = yaml.safe_load((ROOT / planner.CATALOG).read_text())
    selected, missing = planner.select_cases(catalog, required, [])
    assert not missing
    assert {case["id"] for case in selected if case["selected"] and case["resources"]["gpus"]} == {
        "wan-dgpo-overlap-native"
    }


def test_dgpo_native_recipe_matches_catalog_identity(monkeypatch, tmp_path):
    import runpy
    from types import ModuleType

    catalog = yaml.safe_load((ROOT / planner.CATALOG).read_text())
    case = next(row for row in catalog["cases"] if row["id"] == "wan-dgpo-overlap-native")
    selector = next(argument for argument in case["command"] if ".py::" in argument)
    path, function = selector.split("::", 1)
    captured = []
    runner = ModuleType("context_parallel_overlap_runner")
    runner.qualify_overlap = lambda directory, options: captured.append(options)
    monkeypatch.setitem(sys.modules, runner.__name__, runner)
    runpy.run_path(str(ROOT / path))[function](tmp_path)
    (options,) = captured
    assert case["execution_capability"] == f"{options['model_family']}-{options['algorithm']}-cp-v1"
    assert options["sampler_type"] == "global_batch"
    assert options["dp_size"] == options["cp_size"] == options["tp_size"] == 2
    validator.validate_topology(case)


@pytest.mark.parametrize(
    "path,symbol",
    [
        ("src/flow_factory/trainers/abc.py", "BaseTrainer._backward_full_output_loss"),
        ("src/flow_factory/utils/group_coordinator.py", "GroupCoordinator.reduce_sum"),
    ],
)
def test_shared_group_and_backward_owners_retain_dgpo_witness(path, symbol):
    policy = yaml.safe_load((ROOT / planner.POLICY).read_text())
    scope = {"paths": [path], "facets": ["objective"], "scope_complete": True}
    required = planner.infer_requirements(scope, policy, {path: [symbol]})
    assert {"algorithm.dgpo_objectives", "algorithm.dgpo.wan_overlap"} <= set(
        required["obligations"]
    )


@pytest.mark.parametrize(
    "case_id",
    ["crd-objective-contracts", "crd-logical-gather-contracts", "qwen-crd-overlap-native"],
)
def test_crd_catalog_binds_transitive_fixture_imports(case_id):
    import ast

    catalog = yaml.safe_load((ROOT / planner.CATALOG).read_text())
    case = next(row for row in catalog["cases"] if row["id"] == case_id)
    artifacts = set(case["artifacts"])
    for path in artifacts:
        for node in ast.walk(ast.parse((ROOT / path).read_text())):
            modules = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
            )
            for module in modules:
                for target in (
                    ROOT / "tests/distributed" / (module.replace(".", "/") + ".py"),
                    ROOT / (module.replace(".", "/") + ".py"),
                ):
                    if target.is_file() and target.is_relative_to(ROOT / "tests"):
                        assert str(target.relative_to(ROOT)) in artifacts, (case_id, path, module)


def test_crd_leaf_selects_objective_logical_gather_and_one_native_witness():
    path = "src/flow_factory/trainers/rl/crd.py"
    catalog = yaml.safe_load((ROOT / planner.CATALOG).read_text())
    scope = {"paths": [path], "facets": [], "scope_complete": True}
    required = planner.requirements_for(
        scope, {}, catalog, yaml.safe_load((ROOT / planner.POLICY).read_text())
    )
    assert {
        "algorithm.crd_objectives",
        "algorithm.crd.logical_gather",
        "algorithm.crd.qwen_overlap",
    } <= set(required["obligations"])
    selected, missing = planner.select_cases(catalog, required, [])
    assert not missing
    assert {row["id"] for row in selected if row["selected"] and row["resources"]["gpus"]} == {
        "qwen-crd-overlap-native"
    }


def test_aligned_gather_owner_retains_its_crd_consumer_and_backend_contracts():
    path = "src/flow_factory/utils/dist.py"
    required = planner.infer_requirements(
        {"paths": [path], "facets": ["distributed"], "scope_complete": True},
        yaml.safe_load((ROOT / planner.POLICY).read_text()),
        {path: ["gather_aligned_floating_tensors"]},
    )
    assert {
        "algorithm.crd_objectives",
        "algorithm.crd.logical_gather",
        "algorithm.crd.qwen_overlap",
    } <= set(required["obligations"])


def test_crd_native_recipe_matches_catalog_and_rejects_unqualified_topology(monkeypatch, tmp_path):
    import runpy
    from types import ModuleType

    catalog = yaml.safe_load((ROOT / planner.CATALOG).read_text())
    case = next(row for row in catalog["cases"] if row["id"] == "qwen-crd-overlap-native")
    path, function = next(argument for argument in case["command"] if ".py::" in argument).split(
        "::", 1
    )
    captured = []
    runner = ModuleType("context_parallel_overlap_runner")
    runner.qualify_overlap = lambda directory, options: captured.append(options)
    monkeypatch.setitem(sys.modules, runner.__name__, runner)
    runpy.run_path(str(ROOT / path))[function](tmp_path)
    (options,) = captured
    assert options["sampler_type"] == "global_batch" and options["scheduling_mode"] == "ordered"
    assert case["execution_capability"] == f"{options['model_family']}-{options['algorithm']}-cp-v1"
    validator.validate_topology(case)
    for key, value, message in (
        ("execution_capability", "qwen-crd-cp-v1", "declared supported"),
        ("backend", "ddp", "FSDP2"),
    ):
        bad = copy.deepcopy(case)
        bad[key] = value
        with pytest.raises(ValueError, match=message):
            validator.validate_topology(bad)
    bad = copy.deepcopy(case)
    bad["topology"].update(tp=2, ranks_per_node=8)
    bad["resources"]["gpus"] = 8
    with pytest.raises(ValueError, match="Qwen CP evidence requires TP=1"):
        validator.validate_topology(bad)


@pytest.mark.parametrize(
    "case_id", ["opd-objective-contracts", "wan-cp-opd-tp2-lora", "prepared-lora-strided-placement"]
)
def test_opd_catalog_binds_transitive_fixture_imports(case_id):
    import ast

    catalog = yaml.safe_load((ROOT / planner.CATALOG).read_text())
    case = next(row for row in catalog["cases"] if row["id"] == case_id)
    artifacts = set(case["artifacts"])
    for path in artifacts:
        for node in ast.walk(ast.parse((ROOT / path).read_text())):
            modules = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
            )
            for module in modules:
                for directory in ("", "tests/distributed", "tests/trainers", "tests/models"):
                    target = ROOT / directory / (module.replace(".", "/") + ".py")
                    if target.is_file():
                        assert str(target.relative_to(ROOT)) in artifacts, (case_id, path, module)


def test_opd_leaf_selects_one_reward_free_native_witness():
    path = "src/flow_factory/trainers/distillation/opd/trainer.py"
    before = (ROOT / path).read_text()
    after = _insert_owner_probe(before, "DiffusionOPDTrainer", "_distill")
    required = _real_source_requirements(path, before, after)
    assert {"algorithm.opd.contracts", "algorithm.opd.wan_lifecycle"} <= set(
        required["obligations"]
    )
    catalog = yaml.safe_load((ROOT / planner.CATALOG).read_text())
    selected, missing = planner.select_cases(catalog, required, [])
    assert not missing
    assert {case["id"] for case in selected if case["selected"] and case["resources"]["gpus"]} == {
        "wan-cp-opd-tp2-lora"
    }
    case = next(row for row in selected if row["id"] == "wan-cp-opd-tp2-lora")
    assert case["oracle"]["reward"] == "none"
    assert case["topology"]["dp_shard"] == case["topology"]["tp"] == case["topology"]["cp"] == 2
    validator.validate_topology(case)


def test_shared_full_output_owner_retains_opd_lifecycle():
    policy = yaml.safe_load((ROOT / planner.POLICY).read_text())
    path = "src/flow_factory/trainers/common/context_parallel.py"
    scope = {"paths": [path], "facets": ["objective"], "scope_complete": True}
    required = planner.infer_requirements(
        scope, policy, {path: ["full_output_context_parallel_loss"]}
    )
    assert {"algorithm.opd.contracts", "algorithm.opd.wan_lifecycle"} <= set(
        required["obligations"]
    )


@pytest.mark.parametrize(
    "path,symbol",
    [
        ("src/flow_factory/models/abc.py", "BaseAdapter.load_lora_parameter_values"),
        ("src/flow_factory/models/lora_parameter_loading.py", "load_lora_parameter_values"),
    ],
)
def test_prepared_lora_import_selects_placement_and_native_teacher_proofs(path, symbol):
    policy = yaml.safe_load((ROOT / planner.POLICY).read_text())
    scope = {"paths": [path], "facets": ["model_adapter"], "scope_complete": True}
    required = planner.infer_requirements(scope, policy, {path: [symbol]})
    assert {
        "algorithm.opd.contracts",
        "algorithm.opd.wan_lifecycle",
        "loading.prepared_lora_placements",
    } <= set(required["obligations"])
    if path.endswith("/abc.py"):
        assert {"checkpoint.snapshot_forward.tp1", "checkpoint.snapshot_forward.tp2"} <= set(
            required["obligations"]
        )
    catalog = yaml.safe_load((ROOT / planner.CATALOG).read_text())
    _, missing = planner.select_cases(catalog, required, [])
    assert not missing


@pytest.mark.parametrize("case_id", ["tdm-objective-contracts", "wan-cp-tdm-tp2-full"])
def test_tdm_catalog_binds_transitive_fixture_imports(case_id):
    test_opd_catalog_binds_transitive_fixture_imports(case_id)


def test_tdm_leaf_selects_one_reward_free_native_witness():
    path = "src/flow_factory/trainers/distillation/tdm.py"
    before = (ROOT / path).read_text()
    after = _insert_owner_probe(before, "TDMTrainer", "_fake_boundary_loss")
    required = _real_source_requirements(path, before, after)
    assert {"algorithm.tdm.contracts", "algorithm.tdm.wan_lifecycle"} <= set(
        required["obligations"]
    )
    catalog = yaml.safe_load((ROOT / planner.CATALOG).read_text())
    selected, missing = planner.select_cases(catalog, required, [])
    assert not missing
    assert {case["id"] for case in selected if case["selected"] and case["resources"]["gpus"]} == {
        "wan-cp-tdm-tp2-full",
        "wan-tdm-r1-cp-lifecycle",
    }
    validator.validate_topology(next(row for row in selected if row["id"] == "wan-cp-tdm-tp2-full"))


def test_shared_role_backward_retains_tdm_and_r1_lifecycles():
    policy = yaml.safe_load((ROOT / planner.POLICY).read_text())
    path = "src/flow_factory/trainers/distillation/distillation_runtime.py"
    required = planner.infer_requirements(
        {"paths": [path], "facets": ["objective"], "scope_complete": True},
        policy,
        {path: ["run_role_phase"]},
    )
    assert {
        "algorithm.tdm.contracts",
        "algorithm.tdm.wan_lifecycle",
        "algorithm.tdm_r1.contracts",
        "algorithm.tdm_r1.wan_lifecycle",
    } <= set(required["obligations"])


@pytest.mark.parametrize(
    "case_id", ["tdm-r1-objective-reference", "tdm-r1-logical-identity", "wan-tdm-r1-cp-lifecycle"]
)
def test_tdm_r1_catalog_binds_transitive_fixture_imports(case_id):
    test_opd_catalog_binds_transitive_fixture_imports(case_id)


def test_tdm_r1_leaf_selects_one_native_overlap_witness():
    path = "src/flow_factory/trainers/distillation/tdm_r1.py"
    before = (ROOT / path).read_text()
    after = _insert_owner_probe(before, "TDMR1Trainer", "_surrogate_boundary_loss")
    required = _real_source_requirements(path, before, after)
    assert {
        "algorithm.tdm_r1.contracts",
        "algorithm.tdm_r1.logical_identity",
        "algorithm.tdm_r1.wan_lifecycle",
    } <= set(required["obligations"])
    catalog = yaml.safe_load((ROOT / planner.CATALOG).read_text())
    selected, missing = planner.select_cases(catalog, required, [])
    assert not missing
    assert {case["id"] for case in selected if case["selected"] and case["resources"]["gpus"]} == {
        "wan-tdm-r1-cp-lifecycle"
    }
    case = next(row for row in selected if row["id"] == "wan-tdm-r1-cp-lifecycle")
    assert case["oracle"]["reward"] == "varying"
    validator.validate_topology(case)


def test_shared_tdm_owner_retains_derived_r1_proof():
    path = "src/flow_factory/trainers/distillation/tdm.py"
    policy = yaml.safe_load((ROOT / planner.POLICY).read_text())
    required = planner.infer_requirements(
        {"paths": [path], "facets": ["objective"], "scope_complete": True},
        policy,
        {path: ["TDMTrainer._generator_score_terms"]},
    )
    assert {
        "algorithm.tdm.contracts",
        "algorithm.tdm.wan_lifecycle",
        "algorithm.tdm_r1.contracts",
        "algorithm.tdm_r1.wan_lifecycle",
    } <= set(required["obligations"])


@pytest.mark.parametrize(
    "case_id", ["wan-i2v-cp-contracts", "wan-i2v-tp-sp-cp-reference", "wan-i2v-cp-tdm-tp2-full"]
)
def test_wan_i2v_catalog_binds_transitive_fixture_imports(case_id):
    test_opd_catalog_binds_transitive_fixture_imports(case_id)


def test_wan_i2v_adapter_selects_native_image_kv_reference_and_lifecycle():
    path = "src/flow_factory/models/wan/wan2_i2v.py"
    catalog = yaml.safe_load((ROOT / planner.CATALOG).read_text())
    required = planner.requirements_for(
        {"paths": [path], "facets": [], "scope_complete": True},
        {},
        catalog,
        yaml.safe_load((ROOT / planner.POLICY).read_text()),
    )
    selected, missing = planner.select_cases(catalog, required, [])
    assert not missing
    assert {row["id"] for row in selected if row["selected"] and row["resources"]["gpus"]} == {
        "wan-i2v-tp-sp-cp-reference",
        "wan-i2v-cp-tdm-tp2-full",
        "wan-remaining-tp-sp-cp-reference",
        "wan-i2v_dual-grpo-ordered-full",
        "wan-i2v_expanded-grpo-ordered-full",
        "wan-flf2v-grpo-ordered-lora",
    }


@pytest.mark.parametrize(
    "case_id", ["qwen-cp-contracts", "qwen-tp-sp-cp-reference", "qwen-tp-sp-cp-guard-ordered-full"]
)
def test_qwen_tp_sp_catalog_binds_transitive_fixture_imports(case_id):
    test_opd_catalog_binds_transitive_fixture_imports(case_id)


def test_qwen_composition_owner_selects_reference_and_lifecycle():
    path = "src/flow_factory/models/qwen_image/_context_parallel.py"
    catalog = yaml.safe_load((ROOT / planner.CATALOG).read_text())
    required = planner.requirements_for(
        {"paths": [path], "facets": [], "scope_complete": True},
        {},
        catalog,
        yaml.safe_load((ROOT / planner.POLICY).read_text()),
    )
    selected, missing = planner.select_cases(catalog, required, [])
    assert not missing
    assert {row["id"] for row in selected if row["selected"] and row["resources"]["gpus"]} == {
        "qwen-tp-sp-cp-reference",
        "qwen-tp-sp-cp-guard-ordered-full",
        "qwen-edit-tp-sp-cp-reference",
        "qwen-edit-guard-tp-sp-cp-ordered-full",
    }


def test_qwen_tp_sp_catalog_matches_lifecycle_recipe(monkeypatch, tmp_path):
    import runpy
    from types import ModuleType

    catalog = yaml.safe_load((ROOT / planner.CATALOG).read_text())
    case = next(row for row in catalog["cases"] if row["id"] == "qwen-tp-sp-cp-guard-ordered-full")
    path, function = case["command"][-1].split("::")
    captured = []
    runner = ModuleType("context_parallel_overlap_runner")
    runner.qualify_overlap = lambda directory, options: captured.append(options)
    monkeypatch.setitem(sys.modules, runner.__name__, runner)
    runpy.run_path(str(ROOT / path))[function](tmp_path)
    (options,) = captured
    assert options["tp_size"] == options["cp_size"] == options["dp_size"] == 2
    assert options["sequence_parallel"] is True and options["training_mode"] == "full"
    assert case["execution_capability"] == f"{options['model_family']}-{options['algorithm']}-cp-v1"
    validator.validate_topology(case)


@pytest.mark.parametrize(
    "case_id",
    [
        "flux-kontext-cp-contracts",
        "flux-kontext-tp-cp-reference",
        "flux-kontext-guard-ordered-full",
    ],
)
def test_kontext_catalog_binds_transitive_fixture_imports(case_id):
    test_opd_catalog_binds_transitive_fixture_imports(case_id)


def test_kontext_adapter_selects_reference_crop_and_lifecycle():
    path = "src/flow_factory/models/flux/flux1_kontext.py"
    catalog = yaml.safe_load((ROOT / planner.CATALOG).read_text())
    required = planner.requirements_for(
        {"paths": [path], "facets": [], "scope_complete": True},
        {},
        catalog,
        yaml.safe_load((ROOT / planner.POLICY).read_text()),
    )
    selected, missing = planner.select_cases(catalog, required, [])
    assert not missing
    assert {row["id"] for row in selected if row["selected"] and row["resources"]["gpus"]} == {
        "flux-complete-tp-sft",
        "flux-kontext-tp-cp-reference",
        "flux-kontext-guard-ordered-full",
    }


@pytest.mark.parametrize(
    "symbol",
    [
        "Flux1Adapter.tensor_parallel_plan",
        "Flux1Adapter.validate_context_parallel_module",
        "Flux1Adapter._validate_context_parallel_config",
        "Flux1Adapter._validate_context_parallel_inputs",
    ],
)
def test_shared_flux_declarations_retain_kontext_proof(symbol):
    path = "src/flow_factory/models/flux/flux1.py"
    required = planner.infer_requirements(
        {"paths": [path], "facets": [], "scope_complete": True},
        yaml.safe_load((ROOT / planner.POLICY).read_text()),
        {path: [symbol]},
    )
    assert {
        "model.flux_kontext.cp_contracts",
        "model.flux_kontext.cp_reference",
        "model.flux_kontext.overlap_lifecycle",
    } <= set(required["obligations"])


@pytest.mark.parametrize(
    "case_id",
    [
        "qwen-edit-cp-contracts",
        "qwen-edit-tp-sp-cp-reference",
        "qwen-edit-guard-tp-sp-cp-ordered-full",
    ],
)
def test_qwen_edit_catalog_binds_transitive_fixture_imports(case_id):
    test_opd_catalog_binds_transitive_fixture_imports(case_id)


def test_qwen_edit_adapter_selects_ordered_reference_and_lifecycle_witnesses():
    path = "src/flow_factory/models/qwen_image/qwen_image_edit_plus.py"
    catalog = yaml.safe_load((ROOT / planner.CATALOG).read_text())
    required = planner.requirements_for(
        {"paths": [path], "facets": [], "scope_complete": True},
        {},
        catalog,
        yaml.safe_load((ROOT / planner.POLICY).read_text()),
    )
    selected, missing = planner.select_cases(catalog, required, [])
    assert not missing
    assert {row["id"] for row in selected if row["selected"] and row["resources"]["gpus"]} == {
        "qwen-edit-tp-sp-cp-reference",
        "qwen-edit-guard-tp-sp-cp-ordered-full",
    }


@pytest.mark.parametrize(
    "path,symbol",
    [
        (
            "src/flow_factory/models/qwen_image/_context_parallel.py",
            "compose_qwen_sequence_boundaries",
        ),
        (
            "src/flow_factory/models/qwen_image/qwen_image.py",
            "QwenImageAdapter.tensor_parallel_plan",
        ),
        (
            "src/flow_factory/models/qwen_image/qwen_image.py",
            "QwenImageAdapter.sequence_parallel_plan",
        ),
    ],
)
def test_shared_qwen_composition_and_parallel_plans_retain_edit_witnesses(path, symbol):
    required = planner.infer_requirements(
        {"paths": [path], "facets": [], "scope_complete": True},
        yaml.safe_load((ROOT / planner.POLICY).read_text()),
        {path: [symbol]},
    )
    assert {
        "model.qwen_edit.cp_contracts",
        "model.qwen_edit.cp_reference",
        "model.qwen_edit.guard_lifecycle",
    } <= set(required["obligations"])


def test_qwen_edit_lifecycle_binds_dynamic_fixture_and_preserves_catalog_capability(
    monkeypatch, tmp_path
):
    from types import ModuleType

    catalog = yaml.safe_load((ROOT / planner.CATALOG).read_text())
    case = next(
        row for row in catalog["cases"] if row["id"] == "qwen-edit-guard-tp-sp-cp-ordered-full"
    )
    captured = []
    runner = ModuleType("context_parallel_overlap_runner")
    runner.qualify_overlap = lambda directory, options: captured.append(options)
    monkeypatch.setitem(sys.modules, runner.__name__, runner)
    path, function = next(argument for argument in case["command"] if ".py::" in argument).split(
        "::", 1
    )
    # Execute the actual lifecycle entry point without importing GPU-only probe dependencies.
    import ast

    node = next(
        node
        for node in ast.parse((ROOT / path).read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == function
    )
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), namespace)
    namespace[function](tmp_path)
    (options,) = captured
    assert f"tests/distributed/{options['case_module']}.py" in case["artifacts"]
    assert options["model_family"] == "qwen-image-edit-plus"
    assert options["algorithm"] == "grpo-guard" and options["tp_size"] == 2
    assert options["scheduling_mode"] == "ordered"
    assert case["execution_capability"] == "qwen-image-edit-plus-grpo-guard-cp-v1"
    validator.validate_topology(case)


@pytest.mark.parametrize("symbol", ["parse_qwen_condition_sizes", "_geometry_sequence"])
def test_qwen_geometry_parser_retains_cached_edit_evaluation_witness(symbol):
    path = "src/flow_factory/models/qwen_image/_output.py"
    catalog = yaml.safe_load((ROOT / planner.CATALOG).read_text())
    required = planner.requirements_for(
        {"paths": [path], "facets": [], "scope_complete": True},
        {path: [symbol]},
        catalog,
        yaml.safe_load((ROOT / planner.POLICY).read_text()),
    )
    assert {"model.qwen_edit.cp_contracts", "model.qwen_edit.guard_lifecycle"} <= set(
        required["obligations"]
    )
    selected, missing = planner.select_cases(catalog, required, [])
    assert not missing
    assert {row["id"] for row in selected if row["selected"] and row["resources"]["gpus"]} == {
        "qwen-edit-guard-tp-sp-cp-ordered-full",
    }


@pytest.mark.parametrize(
    "case_id",
    ["qwen21-cp-contracts", "qwen21-tp-sp-cp-reference", "qwen21-guard-tp-sp-cp-ordered-lora"],
)
def test_qwen21_catalog_binds_transitive_fixture_imports(case_id):
    test_opd_catalog_binds_transitive_fixture_imports(case_id)


@pytest.mark.parametrize(
    "path",
    [
        "src/flow_factory/models/qwen_image_21/qwen_image_21.py",
        "src/flow_factory/models/qwen_image_21/_parallel.py",
    ],
)
def test_qwen21_model_owner_selects_joint_and_cached_witnesses(path):
    catalog = yaml.safe_load((ROOT / planner.CATALOG).read_text())
    required = planner.requirements_for(
        {"paths": [path], "facets": [], "scope_complete": True},
        {},
        catalog,
        yaml.safe_load((ROOT / planner.POLICY).read_text()),
    )
    selected, missing = planner.select_cases(catalog, required, [])
    assert not missing
    expected_gpu = {
        "qwen21-tp-sp-cp-reference",
        "qwen21-guard-tp-sp-cp-ordered-lora",
    }
    if path == "src/flow_factory/models/qwen_image_21/_parallel.py":
        expected_gpu.add("qwen21-sp-output-reference")
    assert {
        row["id"] for row in selected if row["selected"] and row["resources"]["gpus"]
    } == expected_gpu


def test_qwen21_lifecycle_runs_both_native_cache_modes_with_real_fixture(monkeypatch, tmp_path):
    import ast
    from types import ModuleType

    catalog = yaml.safe_load((ROOT / planner.CATALOG).read_text())
    case = next(
        row for row in catalog["cases"] if row["id"] == "qwen21-guard-tp-sp-cp-ordered-lora"
    )
    captured = []
    runner = ModuleType("context_parallel_overlap_runner")
    runner.qualify_overlap = lambda directory, options: captured.append(options)
    monkeypatch.setitem(sys.modules, runner.__name__, runner)
    path, function = next(argument for argument in case["command"] if ".py::" in argument).split(
        "::", 1
    )
    node = next(
        node
        for node in ast.parse((ROOT / path).read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == function
    )
    decorator = ast.unparse(node.decorator_list[0])
    assert decorator == "pytest.mark.parametrize('use_kv_cache', [False, True])"
    node.decorator_list = []
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), namespace)
    for cached in (False, True):
        namespace[function](tmp_path, cached)
    assert [row["use_kv_cache"] for row in captured] == [False, True]
    for options in captured:
        assert options["case_module"] == "qwen21_coupled_fixture"
        assert options["sequence_parallel"] is True
        assert options["training_mode"] == "lora"
        assert (
            case["execution_capability"]
            == f"{options['model_family']}-{options['algorithm']}-cp-v1"
        )
    validator.validate_topology(case)


@pytest.mark.parametrize(
    "path",
    [
        "src/flow_factory/models/context_parallel.py",
        "src/flow_factory/trainers/common/model_preparation.py",
        "tests/distributed/test_context_parallel_runtime.py",
    ],
)
def test_cp_runtime_changes_select_transport_and_physical_owner_evidence(path):
    policy = yaml.safe_load((ROOT / planner.POLICY).read_text())
    catalog = yaml.safe_load((ROOT / "config/validation/catalog.yaml").read_text())
    scope = {"paths": [path], "facets": ["distributed"], "scope_complete": True}
    required = planner.infer_requirements(scope, policy, {})
    cases, missing = planner.select_cases(catalog, required, [])
    assert not missing
    selected = {case["id"]: case for case in cases if case["selected"]}
    assert {
        "cp-runtime-contracts",
        "cp-transport-mesh-tp1",
        "cp-transport-mesh-tp2",
        "native-ring-attention",
    } <= selected.keys()
    primitive = selected["native-ring-attention"]
    assert primitive["execution_scope"] == "attention_probe"
    assert primitive["resources"]["gpus"] == 2
    assert primitive["topology"]["cp"] == 2
    for tp in (1, 2):
        case = selected[f"cp-transport-mesh-tp{tp}"]
        assert case["topology"]["tp"] == tp
        assert case["topology"]["ranks_per_node"] == 4 * tp
        assert case["execution_scope"] == "topology_only"
    for name in (
        "cp-runtime-contracts",
        "cp-transport-mesh-tp1",
        "cp-transport-mesh-tp2",
        "native-ring-attention",
    ):
        assert "tests/distributed/test_context_parallel_runtime.py" in selected[name]["artifacts"]
        assert "src/flow_factory/models/context_parallel.py" in selected[name]["artifacts"]
        assert (
            "src/flow_factory/trainers/common/model_preparation.py" in selected[name]["artifacts"]
        )


@pytest.mark.parametrize(
    "case_id",
    [
        "flux-sequence-contracts",
        "flux-dual-sp-cp-reference",
        "flux-kontext-sp-guard-ordered-full",
        "flux-complete-tp-sft",
    ],
)
def test_flux_sequence_catalog_binds_transitive_fixture_imports(case_id):
    test_opd_catalog_binds_transitive_fixture_imports(case_id)


def test_flux_sequence_owner_selects_native_reference_and_conditioned_lifecycle():
    path = "src/flow_factory/models/flux/_sequence_parallel.py"
    catalog = yaml.safe_load((ROOT / planner.CATALOG).read_text())
    required = planner.requirements_for(
        {"paths": [path], "facets": [], "scope_complete": True},
        {},
        catalog,
        yaml.safe_load((ROOT / planner.POLICY).read_text()),
    )
    selected, missing = planner.select_cases(catalog, required, [])
    assert not missing
    assert {case["id"] for case in selected if case["selected"] and case["resources"]["gpus"]} == {
        "flux-complete-tp-sft",
        "flux-dual-sp-cp-reference",
        "flux-kontext-sp-guard-ordered-full",
    }


def test_flux_sequence_reference_requires_the_declared_composed_geometry():
    import copy

    catalog = yaml.safe_load((ROOT / planner.CATALOG).read_text())
    case = next(case for case in catalog["cases"] if case["id"] == "flux-dual-sp-cp-reference")
    validator.validate_topology(case)
    broken = copy.deepcopy(case)
    broken["topology"].update(tp=4, dp_shard=1)
    with pytest.raises(ValueError, match="DP2 TP2 CP2"):
        validator.validate_topology(broken)


@pytest.mark.parametrize(
    "case_id",
    [
        "packed-tp-layout-contracts",
        "packed-tp-native-contracts",
        "packed-tp-composed-state",
        "packed-tp-cuda-state",
    ],
)
def test_packed_tp_catalog_binds_transitive_fixture_imports(case_id):
    test_opd_catalog_binds_transitive_fixture_imports(case_id)


@pytest.mark.parametrize(
    "path,symbol",
    [
        ("src/flow_factory/models/packed_tensor_parallel.py", "PackedColwiseParallel"),
        ("src/flow_factory/models/tensor_parallel.py", "derive_lora_plan"),
    ],
)
def test_packed_owner_requires_native_math_and_canonical_fresh_state(path, symbol):
    catalog = yaml.safe_load((ROOT / planner.CATALOG).read_text())
    required = planner.requirements_for(
        {"paths": [path], "facets": [], "scope_complete": True},
        {path: [symbol]},
        catalog,
        yaml.safe_load((ROOT / planner.POLICY).read_text()),
    )
    selected, missing = planner.select_cases(catalog, required, [])
    assert not missing
    assert {
        "packed-tp-layout-contracts",
        "packed-tp-native-contracts",
        "packed-tp-composed-state",
        "packed-tp-cuda-state",
    } <= {row["id"] for row in selected if row["selected"]}


def test_flux_complete_tp_sft_has_one_two_rank_native_scenario():
    catalog = yaml.safe_load((ROOT / planner.CATALOG).read_text())
    case = next(row for row in catalog["cases"] if row["id"] == "flux-complete-tp-sft")
    validator.validate_topology(case)
    assert case["resources"]["gpus"] == case["topology"]["tp"] == 2
    assert (
        case["topology"]["dp_shard"]
        == case["topology"]["dp_replicate"]
        == case["topology"]["cp"]
        == 1
    )
    assert case["command"][-1].endswith("::test_cuda_flux_tp2_single_stream_sft")
    assert {
        "tests/distributed/flux_kontext_cp_fixture.py",
        "tests/distributed/flux_sft_fixture.py",
    } <= set(case["artifacts"])


@pytest.mark.parametrize(
    "case_id",
    [
        "wan-remaining-cp-contracts",
        "wan-remaining-tp-sp-cp-reference",
        "wan-i2v_dual-grpo-ordered-full",
        "wan-i2v_expanded-grpo-ordered-full",
        "wan-flf2v-grpo-ordered-lora",
    ],
)
def test_wan_remaining_catalog_binds_transitive_fixture_imports(case_id):
    test_opd_catalog_binds_transitive_fixture_imports(case_id)


@pytest.mark.parametrize(
    "path",
    [
        "src/flow_factory/models/wan/wan2_t2v.py",
        "src/flow_factory/models/wan/wan2_i2v.py",
        "src/flow_factory/models/wan/_cp_contract.py",
        "src/flow_factory/models/wan/_context_parallel.py",
        "tests/distributed/wan_remaining_cp_fixture.py",
        "tests/distributed/wan_i2v_expanded_coupled_fixture.py",
    ],
)
def test_wan_remaining_owners_keep_all_native_variant_obligations(path):
    required = planner.infer_requirements(
        {"paths": [path], "facets": [], "scope_complete": True},
        yaml.safe_load((ROOT / planner.POLICY).read_text()),
        {path: ["compose_wan_sequence_boundaries"]},
    )
    assert {
        "model.wan_remaining.contracts",
        "model.wan_remaining.reference",
        "model.wan_remaining.i2v_dual_lifecycle",
        "model.wan_remaining.i2v_expanded_lifecycle",
        "model.wan_remaining.flf2v_lifecycle",
    } <= set(required["obligations"])


@pytest.mark.parametrize(
    "path",
    [
        "src/flow_factory/models/wan/_conditioning.py",
        "src/flow_factory/models/wan/_output.py",
    ],
)
def test_wan_condition_consumers_remain_in_cp_selection(path):
    required = planner.infer_requirements(
        {"paths": [path], "facets": [], "scope_complete": True},
        yaml.safe_load((ROOT / planner.POLICY).read_text()),
        {},
    )
    assert {
        "model.wan_remaining.contracts",
        "model.wan_remaining.i2v_dual_lifecycle",
        "model.wan_remaining.i2v_expanded_lifecycle",
        "model.wan_remaining.flf2v_lifecycle",
    } <= set(required["obligations"])


@pytest.mark.parametrize(
    "path",
    [
        "tests/distributed/test_context_parallel_training.py",
        "tests/distributed/test_context_parallel_state.py",
        "tests/distributed/online_optimizer_observations.py",
        "tests/distributed/context_parallel_model_control.py",
        "tests/distributed/test_dense_optimizer_oracle.py",
    ],
)
def test_dense_optimizer_owner_requires_all_element_and_mixed_mesh_contracts(path):
    catalog = yaml.safe_load((ROOT / planner.CATALOG).read_text())
    required = planner.requirements_for(
        {
            "paths": [path],
            "facets": [],
            "scope_complete": True,
            "docs_only_paths": [],
            "logging_only_paths": [],
        },
        {},
        catalog,
        yaml.safe_load((ROOT / planner.POLICY).read_text()),
    )
    selected, missing = planner.select_cases(catalog, required, [])
    assert not missing
    assert "dense-optimizer-oracle-contracts" in {row["id"] for row in selected if row["selected"]}


def test_dense_optimizer_catalog_binds_all_local_oracle_dependencies():
    test_opd_catalog_binds_transitive_fixture_imports("dense-optimizer-oracle-contracts")


@pytest.mark.parametrize(
    "case_id",
    [
        "ltx2-cp-contracts",
        "ltx2-joint-av-tp-sp-cp-reference",
        "ltx20-t2av-grpo-ordered-lora",
        "ltx23-i2av-grpo-ordered-full",
    ],
)
def test_ltx2_catalog_binds_transitive_native_av_fixtures(case_id):
    test_opd_catalog_binds_transitive_fixture_imports(case_id)


@pytest.mark.parametrize(
    "path",
    [
        "src/flow_factory/models/ltx2/ltx2_t2av.py",
        "src/flow_factory/models/ltx2/ltx2_i2av.py",
        "src/flow_factory/models/ltx2/_context_parallel.py",
        "src/flow_factory/models/ltx2/_conditioning.py",
        "tests/distributed/ltx2_coupled_observations.py",
        "tests/distributed/ltx23_i2av_coupled_fixture.py",
    ],
)
def test_ltx2_owners_retain_both_joint_av_layout_lifecycles(path):
    required = planner.infer_requirements(
        {"paths": [path], "facets": [], "scope_complete": True},
        yaml.safe_load((ROOT / planner.POLICY).read_text()),
        {},
    )
    assert {
        "model.ltx2.cp_contracts",
        "model.ltx2.cp_reference",
        "model.ltx2.cp_ltx20_lifecycle",
        "model.ltx2.cp_ltx23_lifecycle",
    } <= set(required["obligations"])


def test_ltx2_cpu_case_does_not_claim_the_separate_four_rank_control():
    catalog = yaml.safe_load((ROOT / planner.CATALOG).read_text())
    case = next(case for case in catalog["cases"] if case["id"] == "ltx2-cp-contracts")
    assert case["command"][-2:] == ["-k", "not rectangular"]
    assert (
        case["topology"]["ranks_per_node"] == case["topology"]["cp"] == case["topology"]["tp"] == 1
    )
    assert case["resources"]["gpus"] == 0


FLUX2_CASES = {
    "flux2-parallel-contracts",
    "flux2-tp-sp-contracts",
    "flux2-tp-sp-cp-reference",
    "flux2-guard-ordered-full",
    "flux2-klein-dppo-ordered-lora",
}


@pytest.mark.parametrize("case_id", sorted(FLUX2_CASES))
def test_flux2_catalog_binds_transitive_native_fixtures(case_id):
    test_opd_catalog_binds_transitive_fixture_imports(case_id)


@pytest.mark.parametrize(
    "path",
    [
        "src/flow_factory/models/flux/_flux2_parallel.py",
        "src/flow_factory/models/flux/flux2.py",
        "src/flow_factory/models/flux/flux2_klein.py",
        "src/flow_factory/models/packed_tensor_parallel.py",
    ],
)
def test_flux2_packed_and_model_owners_retain_native_consumers(path):
    catalog = yaml.safe_load((ROOT / planner.CATALOG).read_text())
    required = planner.requirements_for(
        {"paths": [path], "facets": [], "scope_complete": True},
        {},
        catalog,
        yaml.safe_load((ROOT / planner.POLICY).read_text()),
    )
    selected, missing = planner.select_cases(catalog, required, [])
    assert not missing
    assert FLUX2_CASES <= {case["id"] for case in selected if case["selected"]}


@pytest.mark.parametrize(
    "case_id",
    ["flux2-tp-sp-cp-reference", "flux2-guard-ordered-full", "flux2-klein-dppo-ordered-lora"],
)
def test_flux2_evidence_requires_its_exact_geometry(case_id):
    catalog = yaml.safe_load((ROOT / planner.CATALOG).read_text())
    case = next(case for case in catalog["cases"] if case["id"] == case_id)
    validator.validate_topology(case)
    broken = copy.deepcopy(case)
    broken["topology"].update(tp=4, dp_shard=1)
    with pytest.raises(ValueError, match="DP2 TP2 CP2"):
        validator.validate_topology(broken)


@pytest.mark.parametrize(
    "case_id",
    [
        "sd35-cp-contracts",
        "sd35-tp-sp-cp-reference",
        "sd35-dppo-ordered-full",
        "sd35-adaln-tp2-supervised-update",
    ],
)
def test_sd35_catalog_binds_transitive_fixture_imports(case_id):
    test_opd_catalog_binds_transitive_fixture_imports(case_id)


def test_sd35_adaln_catalog_keeps_the_incremental_tp_only_geometry():
    catalog = yaml.safe_load((ROOT / planner.CATALOG).read_text())
    case = next(row for row in catalog["cases"] if row["id"] == "sd35-adaln-tp2-supervised-update")
    validator.validate_topology(case)
    assert case["backend"] == "tp" and case["qa"] == "Q2"
    assert case["resources"]["gpus"] == case["topology"]["tp"] == 2
    assert (
        case["topology"]["dp_shard"]
        == case["topology"]["dp_replicate"]
        == case["topology"]["cp"]
        == 1
    )
    assert case["command"][-1].endswith("::test_cuda_sd35_adaln_tp2_supervised_update")
    broken = copy.deepcopy(case)
    broken["backend"] = "fsdp2"
    with pytest.raises(ValueError, match="dp_shard >= 2"):
        validator.validate_topology(broken)


@pytest.mark.parametrize(
    "path",
    [
        "src/flow_factory/models/stable_diffusion/sd3_5.py",
        "src/flow_factory/models/stable_diffusion/_context_parallel.py",
        "tests/distributed/sd35_cp_fixture.py",
        "tests/distributed/sd35_coupled_fixture.py",
        "tests/distributed/test_sd35_context_parallel.py",
        "tests/distributed/test_sd35_context_parallel_contracts.py",
    ],
)
def test_sd35_owners_retain_model_and_lifecycle_witnesses(path):
    catalog = yaml.safe_load((ROOT / planner.CATALOG).read_text())
    required = planner.requirements_for(
        {
            "paths": [path],
            "facets": [],
            "scope_complete": True,
            "docs_only_paths": [],
            "logging_only_paths": [],
        },
        {},
        catalog,
        yaml.safe_load((ROOT / planner.POLICY).read_text()),
    )
    selected, missing = planner.select_cases(catalog, required, [])
    assert not missing
    assert {"sd35-tp-sp-cp-reference", "sd35-dppo-ordered-full"} <= {
        row["id"] for row in selected if row["selected"]
    }


@pytest.mark.parametrize(
    "case_id", ["z_image-cp-contracts", "z_image-tp-sp-cp-reference", "z_image-guard-ordered-lora"]
)
def test_z_image_catalog_binds_transitive_fixture_imports(case_id):
    test_opd_catalog_binds_transitive_fixture_imports(case_id)


@pytest.mark.parametrize(
    "path",
    [
        "src/flow_factory/models/z_image/z_image.py",
        "src/flow_factory/models/z_image/_context_parallel.py",
        "tests/distributed/z_image_cp_fixture.py",
        "tests/distributed/z_image_coupled_fixture.py",
        "tests/distributed/test_z_image_context_parallel.py",
        "tests/distributed/test_z_image_context_parallel_contracts.py",
    ],
)
def test_z_image_owners_retain_model_and_lifecycle_witnesses(path):
    catalog = yaml.safe_load((ROOT / planner.CATALOG).read_text())
    required = planner.requirements_for(
        {
            "paths": [path],
            "facets": [],
            "scope_complete": True,
            "docs_only_paths": [],
            "logging_only_paths": [],
        },
        {},
        catalog,
        yaml.safe_load((ROOT / planner.POLICY).read_text()),
    )
    selected, missing = planner.select_cases(catalog, required, [])
    assert not missing
    assert {"z_image-tp-sp-cp-reference", "z_image-guard-ordered-lora"} <= {
        row["id"] for row in selected if row["selected"]
    }


FLUX3_CASES = {
    "flux3-parallel-contracts",
    "flux3-tp-sp-contracts",
    "flux3-tp-sp-cp-reference",
    "flux3-guard-ordered-full",
    "flux3-distilled-dppo-ordered-lora",
}


@pytest.mark.parametrize("case_id", sorted(FLUX3_CASES))
def test_flux3_catalog_binds_transitive_native_fixtures(case_id):
    test_opd_catalog_binds_transitive_fixture_imports(case_id)


@pytest.mark.parametrize(
    "path",
    [
        "src/flow_factory/models/flux/_flux3_parallel.py",
        "src/flow_factory/models/flux/flux3.py",
        "src/flow_factory/models/packed_tensor_parallel.py",
    ],
)
def test_flux3_packed_and_model_owners_retain_native_consumers(path):
    catalog = yaml.safe_load((ROOT / planner.CATALOG).read_text())
    required = planner.requirements_for(
        {"paths": [path], "facets": [], "scope_complete": True},
        {},
        catalog,
        yaml.safe_load((ROOT / planner.POLICY).read_text()),
    )
    selected, missing = planner.select_cases(catalog, required, [])
    assert not missing
    assert FLUX3_CASES <= {case["id"] for case in selected if case["selected"]}


@pytest.mark.parametrize(
    "case_id",
    ["flux3-tp-sp-cp-reference", "flux3-guard-ordered-full", "flux3-distilled-dppo-ordered-lora"],
)
def test_flux3_evidence_requires_its_exact_geometry(case_id):
    catalog = yaml.safe_load((ROOT / planner.CATALOG).read_text())
    case = next(case for case in catalog["cases"] if case["id"] == case_id)
    validator.validate_topology(case)
    broken = copy.deepcopy(case)
    broken["topology"].update(tp=4, dp_shard=1)
    with pytest.raises(ValueError, match="DP2 TP2 CP2"):
        validator.validate_topology(broken)


def test_flux3_schedule_contract_binds_its_gloo_harness_dependency():
    catalog = yaml.safe_load((ROOT / planner.CATALOG).read_text())
    case = next(row for row in catalog["cases"] if row["id"] == "flux3-parallel-contracts")
    path = "tests/distributed/conftest.py"
    assert path in case["artifacts"]
    required = planner.requirements_for(
        {
            "paths": [path],
            "facets": [],
            "scope_complete": True,
            "docs_only_paths": [],
            "logging_only_paths": [],
        },
        {},
        catalog,
        yaml.safe_load((ROOT / planner.POLICY).read_text()),
    )
    selected, missing = planner.select_cases(catalog, required, [])
    assert not missing
    assert case["id"] in {row["id"] for row in selected if row["selected"]}


@pytest.mark.parametrize(
    "case_id",
    [
        "bagel-cp-contracts",
        "bagel-cp-native-cpu",
        "bagel-tp-sp-cp-reference",
        "bagel-guard-tp-sp-cp-ordered-lora",
    ],
)
def test_bagel_catalog_binds_transitive_fixture_imports(case_id):
    test_opd_catalog_binds_transitive_fixture_imports(case_id)


@pytest.mark.parametrize(
    "path",
    [
        "src/flow_factory/models/bagel/_parallel.py",
        "tests/distributed/native_adam_window_oracle.py",
        "tests/distributed/bagel_arithmetic_operators.py",
        "tests/distributed/bagel_arithmetic_reference.py",
        "tests/distributed/bagel_own_adam_reference.py",
        "tests/distributed/test_bagel_arithmetic_reference.py",
    ],
)
def test_bagel_parallel_owner_selects_packed_math_reference_and_lifecycle(path):
    catalog = yaml.safe_load((ROOT / planner.CATALOG).read_text())
    required = planner.requirements_for(
        {
            "paths": [path],
            "facets": [],
            "scope_complete": True,
            "docs_only_paths": [],
            "logging_only_paths": [],
        },
        {},
        catalog,
        yaml.safe_load((ROOT / planner.POLICY).read_text()),
    )
    selected, missing = planner.select_cases(catalog, required, [])
    assert not missing
    assert {
        "bagel-cp-contracts",
        "bagel-cp-native-cpu",
        "bagel-tp-sp-cp-reference",
        "bagel-guard-tp-sp-cp-ordered-lora",
    } <= {row["id"] for row in selected if row["selected"]}
    expected_gpu = {"bagel-tp-sp-cp-reference", "bagel-guard-tp-sp-cp-ordered-lora"}
    # The output reference imports the helpers/oracles, not the separate arithmetic test.
    if path != "tests/distributed/test_bagel_arithmetic_reference.py":
        expected_gpu.add("bagel-sp-final-norm-reference")
    if path == "tests/distributed/native_adam_window_oracle.py":
        expected_gpu |= {
            "sensenova-tp-sp-cp-reference",
            "sensenova-guard-tp-sp-cp-ordered-lora",
            "sensenova-sp-pointwise-head-reference",
        }
    assert {
        row["id"] for row in selected if row["selected"] and row["resources"]["gpus"]
    } == expected_gpu


@pytest.mark.parametrize(
    "case_id",
    [
        "minimax-h3-cp-contracts",
        "minimax-h3-pretrained-tp4-loading-capacity",
        "minimax-h3-packed-tp-sp-cp-reference",
        "minimax-h3-fl2va-grpo-ordered-lora",
        "minimax-h3-ref2va-grpo-ordered-full",
    ],
)
def test_h3_catalog_binds_transitive_native_av_fixtures(case_id):
    test_opd_catalog_binds_transitive_fixture_imports(case_id)


@pytest.mark.parametrize(
    "path",
    [
        "src/flow_factory/models/minimax_h3/adapters.py",
        "src/flow_factory/models/minimax_h3/workflow.py",
        "src/flow_factory/models/minimax_h3/_context_parallel.py",
        "src/flow_factory/models/minimax_h3/_chunking.py",
        "tests/distributed/h3_coupled_observations.py",
        "tests/distributed/h3_modulation_reference.py",
        "tests/distributed/h3_ref2va_coupled_fixture.py",
    ],
)
def test_h3_owners_retain_both_joint_av_layout_lifecycles(path):
    required = planner.infer_requirements(
        {"paths": [path], "facets": [], "scope_complete": True},
        yaml.safe_load((ROOT / planner.POLICY).read_text()),
        {},
    )
    assert {
        "model.minimax_h3.cp_contracts",
        "model.minimax_h3.cp_reference",
        "model.minimax_h3.cp_fl2va_lifecycle",
        "model.minimax_h3.cp_ref2va_lifecycle",
    } <= set(required["obligations"])


def test_h3_cpu_case_does_not_claim_the_separate_four_rank_control():
    catalog = yaml.safe_load((ROOT / planner.CATALOG).read_text())
    case = next(case for case in catalog["cases"] if case["id"] == "minimax-h3-cp-contracts")
    assert case["command"][-2:] == ["-k", "not all_parameter_gradients"]
    assert (
        case["topology"]["ranks_per_node"] == case["topology"]["cp"] == case["topology"]["tp"] == 1
    )
    assert case["resources"]["gpus"] == 0
    assert "tests/models/minimax_h3/test_chunking.py" in case["command"]
    assert "tests/models/minimax_h3/test_chunking.py" in case["artifacts"]
    assert "tests/distributed/test_h3_context_parallel.py" in case["artifacts"]


def test_h3_released_capacity_qualifies_tp4_loading_without_cp_equivalence():
    catalog = yaml.safe_load((ROOT / planner.CATALOG).read_text())
    case = next(
        case
        for case in catalog["cases"]
        if case["id"] == "minimax-h3-pretrained-tp4-loading-capacity"
    )
    assert case["qa"] == "Q3" and case["backend"] == "fsdp2"
    assert case["fixture_kind"] == "pretrained_weights"
    assert case["topology"] == {
        "nodes": 1,
        "ranks_per_node": 8,
        "dp_replicate": 1,
        "dp_shard": 2,
        "tp": 4,
        "cp": 1,
        "ep": 1,
    }
    assert case["resources"]["gpus"] == 8
    assert case["oracle"]["kind"] == "assertions"
    assert "execution_capability" not in case
    assert case["command"][-1].endswith("::test_released_h3_ref2va_tp4_lora_preparation_capacity")
    validator.validate_topology(case)
    required = planner.requirements_for(
        {
            "paths": ["tests/distributed/test_h3_pretrained_capacity.py"],
            "facets": [],
            "scope_complete": True,
            "docs_only_paths": [],
            "logging_only_paths": [],
        },
        {},
        catalog,
        yaml.safe_load((ROOT / planner.POLICY).read_text()),
    )
    selected, missing = planner.select_cases(catalog, required, [])
    assert not missing
    assert case["id"] in {row["id"] for row in selected if row["selected"]}


@pytest.mark.parametrize(
    "path,symbol,required",
    [
        (
            "src/flow_factory/models/minimax_h3/adapters.py",
            "_MiniMaxH3WorkflowAdapter.tensor_parallel_plan",
            True,
        ),
        (
            "src/flow_factory/models/minimax_h3/adapters.py",
            "_MiniMaxH3WorkflowAdapter.sequence_parallel_plan",
            True,
        ),
        (
            "src/flow_factory/models/minimax_h3/adapters.py",
            "_MiniMaxH3WorkflowAdapter.preprocess_func",
            False,
        ),
        ("src/flow_factory/models/minimax_h3/workflow.py", "build_h3_component_runtime", True),
        ("src/flow_factory/models/minimax_h3/workflow.py", "load_h3_workflow_pipeline", True),
        ("src/flow_factory/models/minimax_h3/workflow.py", "preprocess_h3_workflow", False),
    ],
)
def test_h3_released_capacity_tracks_only_placement_and_loading_symbols(path, symbol, required):
    requirements = planner.infer_requirements(
        {"paths": [path], "facets": [], "scope_complete": True},
        yaml.safe_load((ROOT / planner.POLICY).read_text()),
        {path: [symbol]},
    )
    assert ("model.minimax_h3.pretrained_capacity" in requirements["obligations"]) is required


@pytest.mark.parametrize(
    "case_id",
    [
        "sensenova-cp-contracts",
        "sensenova-cp-native-cpu",
        "sensenova-tp-sp-cp-reference",
        "sensenova-guard-tp-sp-cp-ordered-lora",
    ],
)
def test_sensenova_catalog_binds_transitive_fixture_imports(case_id):
    test_opd_catalog_binds_transitive_fixture_imports(case_id)


@pytest.mark.parametrize(
    "path",
    [
        "src/flow_factory/models/sensenova/_context_parallel.py",
        "tests/distributed/native_adam_window_oracle.py",
    ],
)
def test_sensenova_parallel_owner_selects_native_derivatives_and_both_head_lifecycles(path):
    catalog = yaml.safe_load((ROOT / planner.CATALOG).read_text())
    required = planner.requirements_for(
        {
            "paths": [path],
            "facets": [],
            "scope_complete": True,
            "docs_only_paths": [],
            "logging_only_paths": [],
        },
        {},
        catalog,
        yaml.safe_load((ROOT / planner.POLICY).read_text()),
    )
    selected, missing = planner.select_cases(catalog, required, [])
    assert not missing
    assert {
        "sensenova-cp-contracts",
        "sensenova-cp-native-cpu",
        "sensenova-tp-sp-cp-reference",
        "sensenova-guard-tp-sp-cp-ordered-lora",
    } <= {row["id"] for row in selected if row["selected"]}
    assert {"sensenova-tp-sp-cp-reference", "sensenova-guard-tp-sp-cp-ordered-lora"} <= {
        row["id"] for row in selected if row["selected"] and row["resources"]["gpus"]
    }


def test_sensenova_contract_and_derivative_oracles_remain_distinct():
    catalog = yaml.safe_load((ROOT / planner.CATALOG).read_text())
    cases = {row["id"]: row for row in catalog["cases"]}
    assert cases["sensenova-cp-contracts"]["oracle"]["kind"] == "assertions"
    assert cases["sensenova-cp-native-cpu"]["oracle"]["kind"] == "numerical"


@pytest.mark.parametrize(
    "path,symbol,expected",
    [
        (
            "src/flow_factory/models/tensor_parallel.py",
            "derive_lora_plan",
            {"sp-replicated-lora-contracts", "flux-complete-tp-sft"},
        ),
        (
            "src/flow_factory/trainers/common/runtime_identity.py",
            "_parameter_schema",
            {"tp-prepared-state-identity"},
        ),
        (
            "tests/distributed/test_sp_lora_parameters.py",
            "<module>",
            {"sp-replicated-lora-contracts"},
        ),
        (
            "tests/distributed/test_tp_parameter_identity.py",
            "<module>",
            {"tp-prepared-state-identity"},
        ),
    ],
)
def test_sp_foundation_contracts_are_required_without_manual_case_flags(path, symbol, expected):
    catalog = yaml.safe_load((ROOT / planner.CATALOG).read_text())
    policy = yaml.safe_load((ROOT / planner.POLICY).read_text())
    required = planner.requirements_for(
        {
            "paths": [path],
            "facets": [],
            "scope_complete": True,
            "docs_only_paths": [],
            "logging_only_paths": [],
        },
        {path: [symbol]},
        catalog,
        policy,
    )
    selected, missing = planner.select_cases(catalog, required, [])
    assert not missing
    assert expected <= {row["id"] for row in selected if row["selected"]}


def test_sp_foundation_extra_witnesses_follow_only_their_changed_symbols():
    policy = yaml.safe_load((ROOT / planner.POLICY).read_text())
    for path, symbol, absent in [
        (
            "src/flow_factory/models/tensor_parallel.py",
            "peel_peft_wrapper",
            "model.flux.complete_tp_sft",
        ),
        (
            "src/flow_factory/trainers/common/runtime_identity.py",
            "_evaluation_loader_schema",
            "parallel.tp_prepared_state_layout",
        ),
    ]:
        required = planner.infer_requirements(
            {"paths": [path], "facets": [], "scope_complete": True}, policy, {path: [symbol]}
        )
        assert absent not in required["obligations"]


def test_flux_sequence_owner_requires_padded_cp_contracts():
    policy = yaml.safe_load((ROOT / planner.POLICY).read_text())
    path = "src/flow_factory/models/flux/_sequence_parallel.py"
    scope = {"paths": [path], "facets": ["distributed"], "scope_complete": True}
    obligations = planner.infer_requirements(scope, policy, {})["obligations"]
    assert "model.flux.complete_sp_cp_contracts" in obligations
    assert "model.flux.sequence_cp_reference" in obligations


@pytest.mark.parametrize("family", ["z_image", "minimax_h3"])
def test_refiner_sequence_owner_selects_math_and_prepared_composition(family):
    policy = yaml.safe_load((ROOT / planner.POLICY).read_text())
    path = f"src/flow_factory/models/{family}/_sequence_parallel.py"
    scope = {"paths": [path], "facets": ["distributed"], "scope_complete": True}
    obligations = planner.infer_requirements(scope, policy, {})["obligations"]
    assert f"model.{family}.refiner_sp_contracts" in obligations
    assert f"model.{family}.refiner_sp_cp_reference" in obligations
    assert f"review.owner:{path}" not in obligations


@pytest.mark.parametrize("family", ["flux2", "flux3"])
def test_continuous_flux_owner_selects_each_execution_boundary(family):
    policy = yaml.safe_load((ROOT / planner.POLICY).read_text())
    path = f"src/flow_factory/models/flux/_{family}_parallel.py"
    scope = {"paths": [path], "facets": ["distributed"], "scope_complete": True}
    obligations = planner.infer_requirements(scope, policy, {})["obligations"]
    assert {
        f"model.{family}.continuous_sp_{suffix}"
        for suffix in ("cp_contracts", "cuda", "cp_reference")
    } <= obligations.keys()


@pytest.mark.parametrize(
    "module,key",
    [
        ("qwen_image_21/_parallel.py", "qwen21"),
        ("bagel/_parallel.py", "bagel"),
        ("sensenova/_sequence_parallel.py", "sensenova"),
    ],
)
def test_output_gather_owner_keeps_prepared_update_evidence(module, key):
    policy = yaml.safe_load((ROOT / planner.POLICY).read_text())
    path = f"src/flow_factory/models/{module}"
    scope = {"paths": [path], "facets": ["distributed"], "scope_complete": True}
    obligations = planner.infer_requirements(scope, policy, {})["obligations"]
    assert f"model.{key}.sp_output_reference" in obligations


@pytest.mark.parametrize("family", ["stable_diffusion", "qwen_image"])
def test_text_sequence_owner_keeps_distinct_cpu_cuda_and_prepared_obligations(family):
    policy = yaml.safe_load((ROOT / planner.POLICY).read_text())
    path = f"src/flow_factory/models/{family}/_sequence_parallel.py"
    scope = {"paths": [path], "facets": ["distributed"], "scope_complete": True}
    obligations = planner.infer_requirements(scope, policy, {})["obligations"]
    expected = {
        f"model.text_streams.{name}"
        for name in ("sp_contracts", "sp_cp_contracts", "sp_cuda", "sd35_prepared", "qwen_prepared")
    }
    assert expected <= obligations.keys()
    assert f"review.owner:{path}" not in obligations


@pytest.mark.parametrize(
    "path",
    [
        "examples/sft/lora/wan2_t2v/wan21_tp_cp.yaml",
        "examples/grpo/lora/sd3_5/default.yaml",
        "multinode_examples/train.yaml",
    ],
)
@pytest.mark.parametrize("change", ["comment", "scalar", "mode"])
def test_yaml_documentation_proof_preserves_real_configuration_owner_routing(path, change):
    before = "# See guidance/acceleration.md\nparallelism:\n  tp_size: 2\n"
    after = before.replace("guidance/acceleration.md", "guidance/configuration/acceleration.md")
    if change == "scalar":
        after = after.replace("tp_size: 2", "tp_size: 4")
    snapshot = {
        "paths": [path, ".agents/knowledge/README.md"],
        "pairs": {path: (before, after)},
        "diffs": {},
        "mode_changes": [path] if change == "mode" else [],
        "complete": True,
        "uninspected": [],
    }
    rules = yaml.safe_load((ROOT / ".agents/harness/risk_rules.yaml").read_text())
    scope, symbols = planner.classify_snapshot(snapshot, rules)
    catalog = yaml.safe_load((ROOT / planner.CATALOG).read_text())
    policy = yaml.safe_load((ROOT / planner.POLICY).read_text())
    required = planner.requirements_for(scope, symbols, catalog, policy)
    selected, missing = planner.select_cases(catalog, required, [])
    selected_ids = {row["id"] for row in selected if row["selected"]}
    gpu_ids = {row["id"] for row in selected if row["selected"] and row["resources"]["gpus"]}
    assert scope["scope_complete"]
    if change == "comment":
        assert scope["docs_only_paths"] == [path]
        assert selected_ids == {"harness-contracts", "repository-consistency"}
        assert not missing and not gpu_ids
    else:
        assert scope["docs_only_paths"] == []
        if path.endswith("wan21_tp_cp.yaml"):
            assert "preparation-tp-cp-probe" in gpu_ids
            assert "wan-sft-tp-cp" in gpu_ids
            assert not missing
        else:
            assert f"review.owner:{path}" in missing


@pytest.mark.parametrize(
    "path",
    [
        "MANIFEST.in",
        "docker/docker-cuda/Dockerfile",
        "scripts/validate_release_artifacts.py",
        "scripts/check_installed_release.py",
        "scripts/check_release_identity.py",
        ".github/workflows/release.yml",
        ".github/workflows/container.yml",
    ],
)
def test_packaging_owners_select_artifact_controls_without_gpu_sweep(path):
    policy = yaml.safe_load((ROOT / planner.POLICY).read_text())
    catalog = yaml.safe_load((ROOT / "config/validation/catalog.yaml").read_text())
    scope = {"paths": [path], "facets": [], "scope_complete": True}
    required = planner.infer_requirements(scope, policy, {})
    cases, missing = planner.select_cases(catalog, required, [])
    assert not missing
    selected = [case for case in cases if case["selected"]]
    assert "release-artifact-contracts" in {case["id"] for case in selected}
    assert all(case["resources"]["gpus"] == 0 for case in selected)


@pytest.mark.parametrize("newline", [b"\r", b"\r\n", b"\n"])
@pytest.mark.parametrize("repeat", [1, 4096])
def test_evidence_transport_preserves_log_bytes(plan, tmp_path, newline, repeat):
    results = passed(plan, tmp_path)
    path = tmp_path / ".scratch/probe.log"
    content = (b"one assertion passed" + newline) * repeat
    path.write_bytes(content)
    results["cases"][0]["evidence"][".scratch/probe.log"] = hashlib.sha256(content).hexdigest()
    bundle = transport.make_bundle(tmp_path, plan, results)
    assert bundle["schema_version"] == (1 if repeat == 1 else 2)
    restored = transport.import_bundle(tmp_path, plan, bundle)
    relative = next(iter(restored["cases"][0]["evidence"]))
    assert (tmp_path / relative).read_bytes() == content


@pytest.mark.parametrize(
    "path,expected",
    [
        (
            "src/flow_factory/utils/torch_runtime.py",
            {"torch-runtime-policy", "native-attention-runtime"},
        ),
        ("scripts/benchmark_torch_training.py", {"torch-training-benchmark-cli"}),
        (
            "src/flow_factory/loading/backend.py",
            {"fsdp-checkpoint-replay", "minimax-h3-ref2va-grpo-ordered-full"},
        ),
    ],
)
def test_torch_compatibility_owners_retain_executable_witnesses(path, expected):
    policy = yaml.safe_load((ROOT / planner.POLICY).read_text())
    catalog = yaml.safe_load((ROOT / planner.CATALOG).read_text())
    required = planner.infer_requirements(
        {"paths": [path], "facets": [], "scope_complete": True}, policy, {}
    )
    cases, missing = planner.select_cases(catalog, required, [])
    assert not missing
    assert expected <= {row["id"] for row in cases if row["selected"]}


def test_registry_declarations_do_not_admit_resolver_body_changes():
    path = "src/flow_factory/trainers/registry.py"
    old = '_TRAINER_REGISTRY = {"grpo": "pkg.GRPO"}\n\ndef resolve(name):\n    return name\n'
    declaration = old.replace('"grpo": "pkg.GRPO"', '"grpo": "pkg.GRPO", "art": "pkg.ART"')
    required = _real_source_requirements(path, old, declaration)["obligations"]
    assert "execution.contracts" in required
    assert not any(key.startswith("review.owner:") for key in required)
    body = old.replace("return name", "return name.lower()")
    required = _real_source_requirements(path, old, body)["obligations"]
    assert any(key.startswith("review.owner:") for key in required)


@pytest.mark.parametrize(
    "symbol", ["GRPOTrainer.__init__", "GRPOTrainer.training_args", "GRPOTrainer.optimize"]
)
def test_replay_initialization_focus_keeps_objective_changes_broad(symbol):
    path = "src/flow_factory/trainers/rl/grpo.py"
    required = planner.infer_requirements(
        {
            "paths": [path],
            "facets": [],
            "scope_complete": True,
            "semantic_symbols": {path: [symbol]},
        },
        yaml.safe_load((ROOT / planner.POLICY).read_text()),
        {path: [symbol]},
    )["obligations"]
    if symbol != "GRPOTrainer.optimize":
        assert {
            "scheduler.replay.contracts",
            "scheduler.replay.coupled",
            "execution.contracts",
        } <= set(required)
        assert "overlap.native.wan.subgroup_ready" not in required
    else:
        assert "overlap.native.wan.subgroup_ready" in required


@pytest.mark.parametrize(
    "symbol,covered",
    [
        ("TrainingArguments.validate_configuration", True),
        ("TrainingArguments.get_num_train_timesteps", True),
        ("TrainingArguments.__post_init__", False),
    ],
)
def test_training_configuration_hook_does_not_admit_unrelated_base_changes(symbol, covered):
    path = "src/flow_factory/hparams/training_args/_base.py"
    required = planner.infer_requirements(
        {
            "paths": [path],
            "facets": [],
            "scope_complete": True,
            "semantic_symbols": {path: [symbol]},
        },
        yaml.safe_load((ROOT / planner.POLICY).read_text()),
        {path: [symbol]},
    )["obligations"]
    if covered:
        assert {"execution.contracts", "scheduler.replay.contracts"} <= set(required)
        assert not any(key.startswith("review.") for key in required)
    else:
        assert any(key.startswith("review.") for key in required)


def test_replay_policy_retains_all_factory_and_endpoint_sampling_controls():
    catalog = yaml.safe_load((ROOT / planner.CATALOG).read_text())
    cases = {row["id"]: row for row in catalog["cases"]}
    assert (
        "tests/scheduler/test_replay_selection.py"
        in cases["scheduler-replay-contracts"]["artifacts"]
    )
    native = cases["scheduler-replay-flux-native"]
    assert native["resources"]["gpus"] == 2
    assert native["topology"]["dp_shard"] == 2 and native["topology"]["cp"] == 1
    assert {
        "tests/distributed/flux_replay_selection_fixture.py",
        "tests/distributed/context_parallel_overlap_runner.py",
    } <= set(native["artifacts"])


def test_differentiable_reward_admission_selects_focused_contracts():
    """Additive derivative declarations keep execution and gradient proof obligations."""
    policy = yaml.safe_load((ROOT / "config/validation/selection.yaml").read_text())
    semantic = {
        "src/flow_factory/trainers/abc.py": ["BaseTrainer._init_reward_model"],
        "src/flow_factory/hparams/args.py": [
            "Arguments._validate_training_feedback_contract",
            "Arguments._validate_every_source_has_a_reward",
        ],
        "src/flow_factory/hparams/training_args/_base.py": [
            "TrainingArguments.reward_gradient_fields"
        ],
    }
    scope = {
        "paths": list(semantic),
        "semantic_symbols": semantic,
        "scope_complete": True,
        "facets": [],
        "docs_only_paths": [],
        "uninspected": [],
    }
    required = planner.infer_requirements(scope, policy, semantic)
    text = str(required)
    assert "feedback.differentiable.contracts" in text
    assert "execution.contracts" in text
    assert "review.owner:" not in text
    assert "overlap.loop" not in text
