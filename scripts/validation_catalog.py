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

"""Inspect and explicitly run catalogued proof cases without importing the training stack."""

import argparse
import ast
import copy
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Callable

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "config/validation/catalog.yaml"
_CASE_FIELDS = ("id", "covers", "fixture", "command", "qa", "backend", "topology", "oracle")
_SPEC = importlib.util.spec_from_file_location(
    "catalog_plan_contract", ROOT / "scripts/validate_validation_plan.py"
)
_PLAN = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PLAN)


def _path(value: str, root: Path) -> Path:
    """Require an existing repository-relative regular file without escaping symlinks."""
    _PLAN.text(value, "artifact")
    relative = PurePosixPath(value)
    _PLAN.require(
        not relative.is_absolute() and ".." not in relative.parts, f"invalid artifact {value!r}"
    )
    path = root / value
    _PLAN.require(
        path.is_file() and path.resolve().is_relative_to(root.resolve()),
        f"missing or external artifact {value!r}",
    )
    return path


def validate_catalog(
    catalog: dict, *, root: Path = ROOT, source: Callable[[str], bytes | str | None] | None = None
) -> None:
    """Check executable metadata and local targets; never import model or GPU packages."""
    _PLAN.mapping(catalog, "catalog")
    _PLAN.require(
        type(catalog.get("schema_version")) is int and catalog["schema_version"] == 1,
        "unsupported catalog schema_version",
    )
    cases = _PLAN.indexed(catalog.get("cases"), "catalog.cases")
    texts = {}

    def read(path):
        if path not in texts:
            relative = PurePosixPath(path)
            _PLAN.require(
                not relative.is_absolute() and ".." not in relative.parts,
                f"invalid artifact {path!r}",
            )
            data = source(path) if source is not None else _path(path, root).read_bytes()
            _PLAN.require(isinstance(data, (str, bytes)), f"missing or external artifact {path!r}")
            _PLAN.require(len(data) <= 1024 * 1024, f"artifact exceeds source budget: {path!r}")
            texts[path] = data.decode("utf-8") if isinstance(data, bytes) else data
        return texts[path]

    facets = set(yaml.safe_load(read(".agents/harness/test_routes.yaml"))["facets"])
    for name, case in cases.items():
        _PLAN.require(
            re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name) is not None, f"invalid case id {name!r}"
        )
        for field in ("covers", "facets", "owners", "artifacts", "command"):
            _PLAN.strings(case.get(field), f"{name}.{field}")
        _PLAN.require(set(case["facets"]) <= facets, f"{name}: unknown facets")
        _PLAN.require(
            all(re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", item) for item in case["covers"]),
            f"{name}: invalid coverage ID",
        )
        _PLAN.text(case.get("fixture"), f"{name}.fixture")
        _PLAN.require(
            case.get("fixture_kind") in {"synthetic", "model_structure", "pretrained_weights"},
            f"{name}: unknown fixture_kind",
        )
        _PLAN.require(
            case["command"][0] in {"python3", "python"},
            f"{name}: command must bind the selected Python interpreter",
        )
        _PLAN.validate_topology(case)
        oracle = case.get("oracle")
        _PLAN.mapping(oracle, f"{name}.oracle")
        _PLAN.require(oracle.get("kind") in _PLAN.ORACLES, f"{name}: unknown oracle kind")
        _PLAN.text(oracle.get("description"), f"{name}.oracle.description")
        _PLAN.require(
            oracle.get("reward") in {"none", "constant", "varying", "real"},
            f"{name}: unknown reward mode",
        )
        _PLAN.require(
            not (
                oracle["reward"] == "constant" and oracle["kind"] in {"numerical", "overlap_timing"}
            ),
            f"{name}: constant reward cannot prove learning or timing",
        )
        _PLAN.require(
            type(case.get("cost_seconds")) is int and case["cost_seconds"] > 0,
            f"{name}: cost_seconds must be a positive estimate",
        )
        for item in case["artifacts"]:
            read(item)
        for arg in case["command"][1:]:
            target, *selectors = arg.split("::")
            if target.endswith(".py"):
                _PLAN.require(
                    target in case["artifacts"],
                    f"{name}: executable {target} missing from artifacts",
                )
                content = read(target)
                if selectors:
                    tree = ast.parse(content, filename=target)
                    for selector in selectors:
                        symbol = selector.split("[", 1)[0]
                        found = next(
                            (
                                node
                                for node in tree.body
                                if isinstance(
                                    node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                                )
                                and node.name == symbol
                            ),
                            None,
                        )
                        _PLAN.require(
                            found is not None,
                            f"{name}: missing test selector {selector!r} in {target}",
                        )
                        tree = found
        resources = case.get("resources")
        _PLAN.mapping(resources, f"{name}.resources")
        _PLAN.require(
            set(resources) == {"gpus", "packages", "environment", "model_paths"},
            f"{name}: unexpected resource keys",
        )
        gpus = resources["gpus"]
        _PLAN.require(type(gpus) is int and gpus >= 0, f"{name}: gpus must be nonnegative")
        local_ranks = case["topology"]["ranks_per_node"]
        _PLAN.require(
            gpus >= local_ranks if case["qa"] != "Q0" else gpus == 0,
            f"{name}: GPU resources do not match QA/local placement",
        )
        for field in ("packages", "model_paths"):
            value = resources[field]
            _PLAN.require(
                isinstance(value, list) and len(value) == len(set(value)),
                f"{name}: invalid {field}",
            )
            for item in value:
                _PLAN.text(item, f"{name}.{field}")
        _PLAN.mapping(resources["environment"], f"{name}.environment")
        for key, value in resources["environment"].items():
            _PLAN.require(
                isinstance(key, str) and re.fullmatch(r"[A-Z][A-Z0-9_]*", key) is not None,
                f"{name}: invalid environment key",
            )
            _PLAN.text(value, f"{name}.environment.{key}")
        _PLAN.require(
            case["fixture_kind"] != "pretrained_weights"
            or (case["qa"] == "Q3" and resources["model_paths"]),
            f"{name}: pretrained weights require Q3 and an explicit model path",
        )
        _PLAN.require(
            case["qa"] != "Q3" or case["fixture_kind"] == "pretrained_weights",
            f"{name}: Q3 must exercise pretrained weights",
        )


def load_catalog(path: Path = DEFAULT_CATALOG, *, root: Path = ROOT) -> dict:
    """Read and validate the catalog using only the quality environment dependencies."""
    catalog = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    validate_catalog(catalog, root=root)
    return catalog


def selected_case_fields(case: dict) -> dict:
    """Return the existing plan schema fields; the planner adds selection and reasons."""
    return {key: copy.deepcopy(case[key]) for key in _CASE_FIELDS}


def prepare_command(case: dict, python: str) -> list[str]:
    """Bind the catalog's Python command to an explicit interpreter, without a shell."""
    _PLAN.require(case["command"][0] in {"python", "python3"}, "unsupported command interpreter")
    return [python, *case["command"][1:]]


def preflight(case: dict, python: str, environment: dict[str, str]) -> list[str]:
    """Check declared prerequisites in the interpreter that will execute the case."""
    missing = []
    for key in case["resources"]["model_paths"]:
        value = environment.get(key)
        if not value or not Path(value).is_dir():
            missing.append(f"set {key} to an existing local pretrained checkpoint directory")
    if case["topology"]["nodes"] != 1:
        missing.append("multi-node cases require an explicit distributed launcher")
    probe = """import importlib.metadata as m, json, sys
missing=[]
for name in json.loads(sys.argv[1]):
    try: m.version(name)
    except m.PackageNotFoundError: missing.append(name)
gpus=0
if int(sys.argv[2]):
    import torch
    gpus=torch.cuda.device_count()
print(json.dumps(dict(missing=missing,gpus=gpus)))
"""
    result = subprocess.run(
        [
            python,
            "-c",
            probe,
            json.dumps(case["resources"]["packages"]),
            str(case["resources"]["gpus"]),
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode:
        return [*missing, "dependency/GPU probe failed: " + result.stderr[-2000:]]
    state = json.loads(result.stdout.strip().splitlines()[-1])
    missing.extend("missing package " + name for name in state["missing"])
    if state["gpus"] < case["resources"]["gpus"]:
        missing.append(f"needs {case['resources']['gpus']} visible GPUs, found {state['gpus']}")
    return missing


def main() -> int:
    """List or inspect cases; execution requires the explicit --run switch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--case")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    try:
        catalog = load_catalog(args.catalog)
        cases = {case["id"]: case for case in catalog["cases"]}
        if not args.case:
            _PLAN.require(not args.run, "--run requires --case")
            print(json.dumps({"schema_version": 1, "cases": list(cases)}, indent=2))
            return 0
        _PLAN.require(args.case in cases, f"unknown case {args.case!r}")
        case = cases[args.case]
        if not args.run:
            print(json.dumps(case, indent=2))
            return 0
        environment = dict(os.environ)
        environment.update(case["resources"]["environment"])
        environment["PYTHONPATH"] = (
            str(ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", "")
        )
        missing = preflight(case, args.python, environment)
        if missing:
            print(
                json.dumps({"case": args.case, "status": "blocked", "reasons": missing}, indent=2)
            )
            return 2
        result = subprocess.run(
            prepare_command(case, args.python), cwd=ROOT, env=environment, timeout=args.timeout
        )
        return result.returncode
    except (ValueError, OSError, subprocess.TimeoutExpired) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
