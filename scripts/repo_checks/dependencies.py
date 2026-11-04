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

"""Established direct import boundaries, not a speculative whole-framework layering."""

import ast

from .core import Repository, dotted, import_name


def imports(path: str, tree: ast.Module):
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                aliases[alias.asname or alias.name] = alias.name
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            parent = import_name(path, node)
            yield node.lineno, parent
            for alias in node.names:
                target = f"{parent}.{alias.name}"
                aliases[alias.asname or alias.name] = target
                yield node.lineno, target
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = dotted(node.func) or ""
            first, dot, rest = name.partition(".")
            name = aliases.get(first, first) + (dot + rest if dot else "")
            if name in ("importlib.import_module", "__import__") and node.args:
                argument = node.args[0]
                if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                    yield node.lineno, argument.value


def check_dependencies(repo: Repository) -> None:
    for path in sorted(repo.paths):
        if not path.endswith(".py") or not repo.exists(path):
            continue
        contracts = path.startswith("src/flow_factory/contracts/")
        scheduler = path.startswith("src/flow_factory/scheduler/")
        if not (contracts or scheduler):
            continue
        reported = set()
        for line, name in imports(path, repo.tree(path)):
            if line in reported:
                continue
            if scheduler and (
                name == "flow_factory.trainers" or name.startswith("flow_factory.trainers.")
            ):
                repo.error(
                    path,
                    line,
                    "dependency-scheduler",
                    f"scheduler must not import trainers: {name}",
                )
                reported.add(line)
            if (
                contracts
                and name.startswith("flow_factory.")
                and not (
                    name == "flow_factory.contracts" or name.startswith("flow_factory.contracts.")
                )
            ):
                repo.error(
                    path,
                    line,
                    "dependency-contracts",
                    f"contracts must not import runtime owners: {name}",
                )
                reported.add(line)
