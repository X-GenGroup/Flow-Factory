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
    assert {"full-output-backward-contracts", "roles-ddp", "roles-zero2", "roles-fsdp2"} <= selected


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
    assert all(case["resources"]["gpus"] == 0 for case in cases if case["selected"])


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
