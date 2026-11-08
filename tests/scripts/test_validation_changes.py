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

"""Structural routing proves bindings and retains unknown or behavioral edits."""

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import validation_changes as changes
import validation_plan as planner


@pytest.fixture
def bindings():
    sources = {
        "src/flow_factory/models/tensor_parallel.py": "from .parallelism.tensor import Style, Other\n",
        "src/flow_factory/models/parallelism/tensor.py": "class Style:\n    pass\nclass Other:\n    pass\n",
    }
    return changes.ImportBindings(sources.get)


def test_same_binding_imports_preserve_nested_owners(bindings):
    before = "from ..tensor_parallel import Style\nclass Adapter:\n    def prepare(self):\n        from ..tensor_parallel import Other\n        return Style, Other\n"
    after = before.replace("..tensor_parallel", "..parallelism.tensor")
    assert (
        changes.semantic_changes("src/flow_factory/models/flux/flux1.py", before, after, bindings)
        == []
    )


@pytest.mark.parametrize("replacement", ["Other as Style", "Style as Changed", "Missing as Style"])
def test_different_or_unknown_binding_is_not_a_move(bindings, replacement):
    before = "from ..tensor_parallel import Style\n"
    after = f"from ..parallelism.tensor import {replacement}\n"
    assert changes.semantic_changes(
        "src/flow_factory/models/flux/flux1.py", before, after, bindings
    )


def test_import_cannot_cross_executable_statement(bindings):
    before = "from ..tensor_parallel import Style\nobserve()\n"
    after = "observe()\nfrom ..parallelism.tensor import Style\n"
    assert changes.semantic_changes(
        "src/flow_factory/models/flux/flux1.py", before, after, bindings
    )


def test_changed_body_keeps_its_actual_owner(bindings):
    before = "from ..tensor_parallel import Style\ndef forward(x):\n    return x\n"
    after = "from ..parallelism.tensor import Style\ndef forward(x):\n    return x * 2\n"
    assert changes.semantic_changes(
        "src/flow_factory/models/flux/flux1.py", before, after, bindings
    ) == ["forward"]


def test_shadowed_cyclic_wildcard_bindings_fail_closed():
    for source in [
        "from .parallelism.tensor import Style\nStyle = object()\n",
        "from .tensor_parallel import Style\n",
        "from .parallelism.tensor import *\n",
    ]:
        resolver = changes.ImportBindings(lambda path: source)
        assert resolver.resolve("flow_factory.models.tensor_parallel", "Style") is None


def test_export_bridge_requires_unchanged_definitions(bindings):
    path = "src/flow_factory/models/tensor_parallel.py"
    before = "class Style:\n    pass\nclass Other:\n    pass\n"
    after = "from .parallelism.tensor import Style, Other\n"
    assert changes.export_bridge(path, before, after, bindings)
    assert not changes.export_bridge(
        path, before.replace("pass", "def forward(self): return 2", 1), after, bindings
    )
    assert not changes.export_bridge(path, before, after + "register()\n", bindings)
    assert not changes.export_bridge(path, before, after + "flag = True\n", bindings)


def test_class_methods_and_fields_have_separate_owners(bindings):
    before = "class A(Base):\n    capability = 1\n    def forward(self, x): return x\n"
    after = before.replace("return x", "return x * 2")
    assert changes.semantic_changes("src/a.py", before, after, bindings) == ["A.forward"]
    assert changes.semantic_changes("src/a.py", before, before.replace("= 1", "= 2"), bindings) == [
        "A.capability"
    ]
    assert changes.semantic_changes(
        "src/a.py", before, before.replace("Base", "Other"), bindings
    ) == ["A"]


def test_direct_test_edits_do_not_select_unrelated_gpu_nodes():
    case = {"command": ["python3", "-m", "pytest", "tests/test_model.py::test_cuda[full]"]}
    assert not changes.test_case_affected(case, "tests/test_model.py", ["test_dependency_gate"])
    assert changes.test_case_affected(case, "tests/test_model.py", ["test_cuda"])
    for symbols in (["_fixture"], ["<module>"], ["test_dependency_gate", "_fixture"]):
        assert changes.test_case_affected(case, "tests/test_model.py", symbols)


def test_whole_test_command_retains_changed_leaf():
    assert changes.test_case_affected(
        {"command": ["pytest", "tests/test_model.py"]}, "tests/test_model.py", ["test_cpu"]
    )


def test_focused_route_requires_complete_symbol_coverage():
    policy = {
        "static_paths": [],
        "owners": [{"paths": ["src/model.py"], "requires": ["full"], "reason": "fallback"}],
        "focused_owners": [
            {
                "paths": ["src/model.py"],
                "symbols": ["compose"],
                "requires": ["focused"],
                "reason": "boundary",
            }
        ],
    }
    scope = {
        "paths": ["src/model.py"],
        "facets": [],
        "scope_complete": True,
        "semantic_symbols": {"src/model.py": ["compose"]},
    }
    assert "focused" in planner.infer_requirements(scope, policy, {})["obligations"]
    for unknown in ("forward", "<module>", "A.capability"):
        updated = copy.deepcopy(scope)
        updated["semantic_symbols"]["src/model.py"].append(unknown)
        obligations = planner.infer_requirements(updated, policy, {})["obligations"]
        assert "full" in obligations and "focused" not in obligations


def test_binding_routing_does_not_lower_raw_risk():
    policy = {"static_paths": [], "owners": []}
    scope = {
        "profile": "R4",
        "paths": ["src/model.py"],
        "facets": ["distributed"],
        "scope_complete": True,
        "import_only_paths": ["src/model.py"],
    }
    result = planner.infer_requirements(scope, policy, {})
    assert "parallel.import_contracts" in result["obligations"]
    assert scope["profile"] == "R4"


def test_shared_worker_change_follows_both_call_graphs():
    before = "def _cpu_worker(): pass\ndef _gpu_worker(): pass\ndef test_cpu(): launch(_cpu_worker)\ndef test_cuda(): launch(_gpu_worker)\n"
    after = before.replace("def _cpu_worker(): pass", "def _cpu_worker(): return 1")
    graph = changes.test_dependencies(before, after)
    cpu = {"command": ["pytest", "tests/test_model.py::test_cpu"]}
    gpu = {"command": ["pytest", "tests/test_model.py::test_cuda"]}
    assert changes.test_case_affected(cpu, "tests/test_model.py", ["_cpu_worker"], graph)
    assert not changes.test_case_affected(gpu, "tests/test_model.py", ["_cpu_worker"], graph)
    assert changes.test_case_affected(gpu, "tests/test_model.py", ["<module>"], graph)


def test_dynamic_helpers_keep_file_wide_routing():
    source = "def test_cuda(): globals()[name]()\n"
    assert changes.test_dependencies(source, source) is None
    assert changes.test_case_affected(
        {"command": ["pytest", "tests/model.py::test_cuda"]}, "tests/model.py", ["_worker"], None
    )


def test_aggregate_harness_command_retains_all_its_declared_test_inputs():
    assert changes.test_case_affected(
        {"command": ["python3", "scripts/quality.py", "cpu"]},
        "tests/scripts/test_validation_planner.py",
        ["test_plan"],
    )


def test_whole_module_relocation_includes_global_constants():
    sources = {
        "src/flow_factory/models/context_parallel.py": "from .parallelism.context import degree\n",
        "src/flow_factory/models/parallelism/context.py": "FACTOR = 2\ndef degree(x): return x * FACTOR\n",
    }
    bindings = changes.ImportBindings(sources.get)
    original = {
        "src/flow_factory/models/context_parallel.py": "FACTOR = 1\ndef degree(x): return x * FACTOR\n"
    }
    path = "src/flow_factory/models/parallelism/context.py"
    assert changes.whole_module_move(path, sources[path], original, bindings) is None
    assert changes.whole_module_move(
        path, sources[path].replace("= 2", "= 1"), original, bindings
    ) == next(iter(original))


def test_focused_routes_require_source_derived_symbols():
    policy = {
        "static_paths": [],
        "owners": [{"paths": ["src/model.py"], "requires": ["full"], "reason": "fallback"}],
        "focused_owners": [
            {
                "paths": ["src/model.py"],
                "symbols": ["compose"],
                "requires": ["focused"],
                "reason": "boundary",
            }
        ],
    }
    scope = {"paths": ["src/model.py"], "facets": [], "scope_complete": True}
    required = planner.infer_requirements(scope, policy, {"src/model.py": ["compose"]})
    assert "full" in required["obligations"] and "focused" not in required["obligations"]


def test_decorated_helper_keeps_conservative_dependency_scope():
    source = "@pytest.fixture(autouse=True)\ndef setup(): pass\ndef test_cuda(): pass\n"
    assert changes.test_dependencies(source, source) is None


def test_stored_helper_cannot_be_pruned_by_direct_call_graph():
    source = "def _worker(): pass\ncallback = _worker\ndef test_cuda(): launch(callback)\n"
    assert changes.test_dependencies(source, source) is None


def test_import_time_helper_registration_keeps_file_wide_coverage():
    source = "def _worker(): pass\nregister(_worker)\ndef test_cuda(): registry.run()\n"
    assert changes.test_dependencies(source, source) is None


def test_export_bridge_preserves_constants_and_module_execution(bindings):
    path = "src/flow_factory/models/tensor_parallel.py"
    before = "class Style:\n    pass\nclass Other:\n    pass\n"
    after = "from .parallelism.tensor import Style, Other\n"
    for suffix in ("FLAG = True\n", "register()\n", "if enabled: register()\n"):
        assert not changes.export_bridge(path, before + suffix, after, bindings)
    sources = {
        path: after + "from .parallelism.tensor import FLAG\n",
        "src/flow_factory/models/parallelism/tensor.py": before + "FLAG = True\n",
    }
    resolver = changes.ImportBindings(sources.get)
    assert changes.export_bridge(path, before + "FLAG = True\n", sources[path], resolver)
    assert not changes.export_bridge(path, before + "FLAG = False\n", sources[path], resolver)
