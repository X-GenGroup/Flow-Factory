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

"""Negative fixtures for import-free repository contracts and their input selection."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from check_repository import check, hook_paths, select_checks
from repo_checks.core import Repository
from repo_checks.dependencies import check_dependencies
from repo_checks.documents import check_documents, check_roadmap
from repo_checks.registries import REGISTRIES, check_examples, check_registries


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.PIPE)


def write(root, name, text):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    git(root, "add", "--", name)
    return path


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-q")
    return tmp_path


@pytest.fixture
def registries(repo):
    write(
        repo,
        "src/flow_factory/example.py",
        "raise RuntimeError('must never be imported')\nclass Thing: pass\n",
    )
    for name, spec in REGISTRIES.items():
        target = (
            "Thing" if name in ("training_args", "optimizer") else "'flow_factory.example.Thing'"
        )
        extra = ", 'none': None" if name == "logger" else ""
        write(
            repo,
            "src/flow_factory/" + spec.path,
            "from flow_factory.example import Thing\n"
            + f"{spec.variable} = {{'demo': {target}{extra}}}\n",
        )
    return repo


def inspect_registries(root):
    view = Repository(root)
    tables = check_registries(view)
    check_examples(view, tables)
    return view


def test_real_repository_full_inventory_is_clean_and_import_free():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json, sys; sys.path.insert(0, 'scripts'); "
            "from check_repository import check; from pathlib import Path; "
            "result = check(Path.cwd()); "
            "assert 'torch' not in sys.modules and 'flow_factory' not in sys.modules; "
            "print(json.dumps(result))",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(completed.stdout)
    assert result["findings"] == []


def test_registry_aliases_and_explicit_reexports_are_static(registries):
    write(
        registries,
        "src/flow_factory/export/__init__.py",
        "from ..example import Thing as Public\nAlias = Public\n",
    )
    write(
        registries,
        "src/flow_factory/models/registry.py",
        "_MODEL_ADAPTER_REGISTRY = {'demo': 'flow_factory.export.Alias', 'alias': 'flow_factory.export.Public'}\n",
    )
    view = inspect_registries(registries)
    assert view.findings == []
    assert view.unverified == set()


@pytest.mark.parametrize("suffix", ["Missing", "Thing"])
def test_missing_registry_symbol_or_module_fails(registries, suffix):
    write(
        registries,
        "src/flow_factory/models/registry.py",
        f"_MODEL_ADAPTER_REGISTRY = {{'demo': 'flow_factory.example.{suffix}'}}\n",
    )
    if suffix == "Thing":
        git(registries, "rm", "-f", "src/flow_factory/example.py")
    view = inspect_registries(registries)
    assert any(f.rule == "registry-target" and "missing" in f.message for f in view.findings)


def test_reexport_cycle_is_a_failure(registries):
    write(registries, "src/flow_factory/example.py", "from .export import Thing\n")
    write(registries, "src/flow_factory/export.py", "from .example import Thing\n")
    assert any("cyclic" in f.message for f in inspect_registries(registries).findings)


@pytest.mark.parametrize("shadow", ["Thing = None", "Thing = factory()", "del Thing"])
def test_shadowed_symbol_is_not_reported_as_a_valid_class(registries, shadow):
    write(registries, "src/flow_factory/example.py", "class Thing: pass\n" + shadow + "\n")
    assert any(f.rule == "registry-target" for f in inspect_registries(registries).findings)


@pytest.mark.parametrize("second", ["demo", "DEMO"])
def test_duplicate_registry_keys_detected_before_normalization(registries, second):
    write(
        registries,
        "src/flow_factory/rewards/registry.py",
        f"_REWARD_MODEL_REGISTRY = {{'demo': 'flow_factory.example.Thing', '{second}': 'flow_factory.example.Thing'}}\n_REWARD_MODEL_REGISTRY = {{k.lower(): v for k, v in _REWARD_MODEL_REGISTRY.items()}}\n",
    )
    assert any(f.rule == "registry-duplicate" for f in inspect_registries(registries).findings)


@pytest.mark.parametrize("declaration", ["dict(demo='flow_factory.example.Thing')", "{**OTHER}"])
def test_nonliteral_registry_declarations_fail_closed(registries, declaration):
    write(
        registries,
        "src/flow_factory/models/registry.py",
        f"_MODEL_ADAPTER_REGISTRY = {declaration}\n",
    )
    assert any(
        f.rule in ("registry-declaration", "registry-key")
        for f in inspect_registries(registries).findings
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "_MODEL_ADAPTER_REGISTRY.update({'later': 'pkg.Thing'})",
        "_MODEL_ADAPTER_REGISTRY['later'] = 'pkg.Thing'",
        "if available:\n    _MODEL_ADAPTER_REGISTRY['later'] = 'pkg.Thing'",
        "_MODEL_ADAPTER_REGISTRY |= {'later': 'pkg.Thing'}",
    ],
)
def test_unsupported_registry_mutations_fail_closed(registries, mutation):
    path = registries / "src/flow_factory/models/registry.py"
    path.write_text(path.read_text() + mutation + "\n")
    assert any(f.rule == "registry-declaration" for f in inspect_registries(registries).findings)


def test_trainer_argument_keys_must_match(registries):
    write(
        registries,
        "src/flow_factory/hparams/training_args/_registry.py",
        "_TRAINING_ARGS_REGISTRY = {}\n",
    )
    assert any(f.rule == "registry-parity" for f in inspect_registries(registries).findings)


def test_example_shapes_case_normalization_and_external_paths(registries):
    write(
        registries,
        "examples/train.yaml",
        "model: {model_type: DEMO}\ntrain: {trainer_type: demo}\nlog: {logging_backend: null}\nrewards: {reward_model: DeMo}\neval_rewards: [{reward_model: other_package.Custom}]\nacceleration: {shared: {name: demo}, rollout: [{name: DEMO}]}\noptimizers: [{optimizer: DEMO}]\n",
    )
    write(registries, "examples/prompt_enhance/custom.yaml", "enhancer: DEMO\n")
    view = inspect_registries(registries)
    assert view.findings == []
    assert len(view.unverified) == 1
    assert "other_package.Custom" in next(iter(view.unverified))


def test_external_plugin_is_reported_without_failing_static_gate(registries):
    write(registries, "examples/external.yaml", "model: {model_type: plugin.models.Custom}\n")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/check_repository.py"),
            "--root",
            str(registries),
            "--hook",
        ],
        env=dict(os.environ, FF_QUALITY_CHANGED_PATHS='["examples/external.yaml"]'),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "UNVERIFIED: external Python path plugin.models.Custom" in result.stdout


@pytest.mark.parametrize(
    "config",
    [
        "model: {model_type: de_mo}",
        "model: {model_type: ' demo'}",
        "model: {model_type: flow_factory.example.Missing}",
        "rewards: [{reward_model: invalid}]",
        "rewards: null",
        "acceleration: {shared: [{params: {}}]}",
        "acceleration: {shared: demo}",
        "optimizers: [{optimizer: plugin.Custom}]",
        "config_file: config/removed.yaml",
    ],
)
def test_invalid_recipe_reference_is_not_normalized_away(registries, config):
    write(registries, "examples/train.yaml", config + "\n")
    assert any(f.rule == "example-reference" for f in inspect_registries(registries).findings)


def test_registry_edit_rechecks_unchanged_examples(registries):
    write(registries, "examples/train.yaml", "model: {model_type: demo}\n")
    assert inspect_registries(registries).findings == []
    write(registries, "src/flow_factory/models/registry.py", "_MODEL_ADAPTER_REGISTRY = {}\n")
    assert any(f.path == "examples/train.yaml" for f in inspect_registries(registries).findings)


def test_scheduler_names_remain_case_sensitive(registries):
    write(
        registries,
        "src/flow_factory/scheduler/registry.py",
        "_SCHEDULER_REGISTRY = {'Demo': 'flow_factory.example.Thing', 'demo': 'flow_factory.example.Thing'}\n",
    )
    view = Repository(registries)
    tables = check_registries(view)
    assert view.findings == []
    assert set(tables["scheduler"]) == {"Demo", "demo"}


def test_markdown_parser_handles_code_references_images_html_and_duplicate_headings(repo):
    write(
        repo,
        "guidance/target file.md",
        '# Hello `World` & Café!\n# Hello World & Café!\n<a id="explicit"></a>\n',
    )
    write(repo, "image.svg", "<svg/>\n")
    write(
        repo,
        "README.md",
        """# Links
[ref][target]
[target]: <guidance/target file.md#hello-world--café>
[duplicate](guidance/target%20file.md#hello-world--café-1)
<a href="guidance/target%20file.md#explicit">custom anchor</a>
![image](/image.svg)
`[not a link](missing.md)`
```md
[not a link](missing.md)
```
<!-- [not a link](missing.md) -->
[remote](https://example.invalid/not-checked)
""",
    )
    view = Repository(repo)
    check_documents(view)
    assert view.findings == []


@pytest.mark.parametrize("change", ["delete", "heading"])
def test_unchanged_incoming_links_fail_after_target_change(repo, change):
    target = write(repo, "guidance/target.md", "# Valid\n")
    write(repo, "README.md", "[link](guidance/target.md#valid)\n")
    if change == "delete":
        git(repo, "rm", "-f", "guidance/target.md")
    else:
        target.write_text("# Renamed\n")
    view = Repository(repo)
    check_documents(view)
    assert len(view.findings) == 1 and view.findings[0].path == "README.md"


def test_untracked_file_cannot_satisfy_link_and_escape_is_rejected(repo):
    (repo / "untracked.md").write_text("# Untracked\n")
    write(repo, "README.md", "[untracked](untracked.md)\n[escape](../outside.md)\n")
    view = Repository(repo)
    check_documents(view)
    assert len(view.findings) == 2


def item(name, dependencies="none", status="planned"):
    return f'<a id="{name.lower()}"></a>\n## {name}: work\n\n- **Status:** {status}\n- **Depends on:** {dependencies}\n'


@pytest.mark.parametrize(
    "content,rule",
    [
        (item("P01") + item("P01"), "roadmap-id"),
        (item("P01", "[P02](#p02)"), "roadmap-dependency"),
        (item("P01", "[P02](#p02)") + item("P02", "[P01](#p01)"), "roadmap-cycle"),
        (item("P01", "[P01](#p01)"), "roadmap-cycle"),
        (item("P01", "P02") + item("P02"), "roadmap-dependency"),
        (item("P01", status="complete"), "roadmap-status"),
        (item("P01", "[P02](#p01)") + item("P02"), "roadmap-dependency"),
        (item("P01").replace('<a id="p01"></a>', ""), "roadmap-id"),
    ],
)
def test_invalid_roadmap_fails(repo, content, rule):
    write(repo, "roadmap/milestones/test.md", content)
    view = Repository(repo)
    check_roadmap(view, check_documents(view))
    assert any(f.rule == rule for f in view.findings)


def test_roadmap_children_and_cross_file_dependencies(repo):
    write(repo, "roadmap/milestones/a.md", item("P01") + item("P01a"))
    write(repo, "roadmap/milestones/b.md", item("P02", "[P01a](a.md#p01a) for final acceptance"))
    view = Repository(repo)
    check_roadmap(view, check_documents(view))
    assert view.findings == []


@pytest.mark.parametrize(
    "source",
    [
        "from flow_factory.trainers.common import group_noise\n",
        "from ..trainers.common.group_noise import stable_text_id\n",
        "from flow_factory import trainers\n",
        "import flow_factory.trainers as trainers\n",
        "import importlib as loader\nloader.import_module('flow_factory.trainers')\n",
        "from importlib import import_module as load\nload('flow_factory.trainers')\n",
        "__import__('flow_factory.trainers')\n",
    ],
)
def test_scheduler_cannot_reintroduce_trainer_dependency(repo, source):
    write(repo, "src/flow_factory/scheduler/sample.py", source)
    view = Repository(repo)
    check_dependencies(view)
    assert any(f.rule == "dependency-scheduler" for f in view.findings)


def test_contracts_are_independent_but_no_speculative_runtime_dag(repo):
    write(repo, "src/flow_factory/contracts/sample.py", "from ..models import BaseAdapter\n")
    write(repo, "src/flow_factory/scheduler/sample.py", "from ..utils.base import stable_text_id\n")
    write(
        repo, "src/flow_factory/trainers/example.py", "from flow_factory.scheduler import sample\n"
    )
    view = Repository(repo)
    check_dependencies(view)
    assert [f.rule for f in view.findings] == ["dependency-contracts"]


def test_tracked_inputs_are_bounded_and_untracked_artifacts_are_not_read(repo):
    write(repo, "README.md", "# Tracked\n")
    artifact = repo / ".checkpoints/large.bin"
    artifact.parent.mkdir()
    with artifact.open("wb") as stream:
        stream.truncate(1024 * 1024 * 1024)
    view = Repository(repo)
    check_documents(view)
    assert view.bytes_read == len("# Tracked\n")
    assert view.findings == []
    write(repo, "guidance/huge.md", "a" * (1024 * 1024 + 1))
    with pytest.raises(ValueError, match="budget"):
        Repository(repo).read("guidance/huge.md")


def test_source_symlink_is_not_followed(repo):
    external = repo.parent / "external.md"
    external.write_text("must not read\n")
    (repo / "README.md").symlink_to(external)
    git(repo, "add", "README.md")
    with pytest.raises(ValueError, match="symlink"):
        Repository(repo).read("README.md")


def test_selection_includes_deletions_and_changed_validator_inputs():
    assert select_checks(["README.md"]) == {"documents"}
    assert "registries" in select_checks(["src/flow_factory/models/deleted.py"])
    assert "registries" in select_checks(["config/deleted.yaml"])
    assert "harness" in select_checks([".agents/harness/risk_rules.yaml"])
    assert "docstrings" in select_checks(["scripts/audit_docstrings.py"])
    assert len(select_checks(["scripts/repo_checks/documents.py"])) == 5
    assert select_checks([]) == set()


def test_hook_comparison_paths_override_empty_index_diff(repo, monkeypatch):
    monkeypatch.setenv("FF_QUALITY_CHANGED_PATHS", '["src/flow_factory/models/deleted.py"]')
    assert hook_paths(repo) == ["src/flow_factory/models/deleted.py"]
    monkeypatch.setenv("FF_QUALITY_CHANGED_PATHS", '"not a list"')
    with pytest.raises(ValueError, match="JSON list"):
        hook_paths(repo)
