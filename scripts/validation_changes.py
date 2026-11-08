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

"""Bounded structural evidence for change-aware test selection, never a risk waiver."""

from __future__ import annotations

import ast
import copy
import importlib.util
from typing import Callable

PARALLEL_MODULES = (
    "flow_factory.models.tensor_parallel",
    "flow_factory.models.context_parallel",
    "flow_factory.models.packed_tensor_parallel",
    "flow_factory.models.parallelism",
)


def module_name(path: str) -> str:
    return path.removeprefix("src/").removesuffix(".py").replace("/", ".")


def absolute_import(path: str, node: ast.ImportFrom) -> str:
    if not node.level:
        return node.module or ""
    package = module_name(path).rsplit(".", 1)[0]
    return importlib.util.resolve_name("." * node.level + (node.module or ""), package)


def in_parallel_package(module: str) -> bool:
    return any(module == root or module.startswith(root + ".") for root in PARALLEL_MODULES)


class ImportBindings:
    """Resolve explicit in-tree compatibility imports without importing model code.

    This proves the binding selected by an import, not equivalence of the bound
    implementation. Changed implementations retain their own coverage routes.
    Unknown, cyclic, wildcard and dynamically shadowed bindings fail closed.
    """

    def __init__(self, source: Callable[[str], str | bytes | None]):
        self.source = source
        self.cache: dict[str, ast.Module | None] = {}

    def tree(self, module: str) -> ast.Module | None:
        if module not in self.cache:
            raw = self.source("src/" + module.replace(".", "/") + ".py")
            try:
                self.cache[module] = ast.parse(raw) if raw is not None else None
            except (SyntaxError, ValueError, RecursionError):
                self.cache[module] = None
        return self.cache[module]

    def resolve(self, module: str, name: str, seen: tuple = ()) -> tuple[str, str] | None:
        key = (module, name)
        if key in seen or len(seen) >= 12 or not in_parallel_package(module):
            return None
        tree = self.tree(module)
        if tree is None:
            return None
        bindings = []
        local = False
        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                if any(a.name == "*" for a in node.names):
                    return None
                for alias in node.names:
                    if (alias.asname or alias.name) == name:
                        path = "src/" + module.replace(".", "/") + ".py"
                        target = absolute_import(path, node)
                        bindings.append((target, alias.name))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name == "__getattr__":
                    return None
                if node.name == name:
                    bindings.append(key)
                    local = True
            elif isinstance(node, (ast.Assign, ast.AnnAssign)) and any(
                isinstance(n, ast.Name) and n.id == name
                for n in (node.targets if isinstance(node, ast.Assign) else [node.target])
            ):
                bindings.append(key)
                local = True
            elif any(
                isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store) and n.id == name
                for n in ast.walk(node)
            ):
                return None
        if len(bindings) != 1:
            return None
        target = bindings[0]
        return key if local else self.resolve(*target, (*seen, key))

    def normalize(self, path: str, source: str) -> ast.Module:
        resolver = self

        class Normalize(ast.NodeTransformer):
            def visit_ImportFrom(self, node):
                module = absolute_import(path, node)
                if not in_parallel_package(module):
                    node.module = module
                    node.level = 0
                    return node
                output = []
                for alias in node.names:
                    target = resolver.resolve(module, alias.name)
                    if target is None:
                        return node
                    item = ast.ImportFrom(
                        module=target[0],
                        names=[ast.alias(name=target[1], asname=alias.asname or alias.name)],
                        level=0,
                    )
                    item._parallel_binding = True
                    output.append(item)
                return output

            def generic_visit(self, node):
                node = super().generic_visit(node)
                # Import reorderings are normalized only within an existing
                # consecutive import block, never across executable statements.
                for field, value in ast.iter_fields(node):
                    if not isinstance(value, list):
                        continue
                    result = []
                    block = []

                    def flush():
                        if not block:
                            return
                        known = [n for n in block if getattr(n, "_parallel_binding", False)]
                        other = [n for n in block if not getattr(n, "_parallel_binding", False)]
                        result.extend(other + sorted(known, key=ast.dump))
                        block.clear()

                    for child in value:
                        if isinstance(child, (ast.Import, ast.ImportFrom)):
                            block.append(child)
                        else:
                            flush()
                            result.append(child)
                    flush()
                    setattr(node, field, result)
                return node

        return Normalize().visit(ast.parse(source))


def definitions(tree: ast.Module | None) -> dict[str, str]:
    """Partition classes into headers, fields and methods; preserve whole functions."""
    found: dict[str, str] = {}
    if tree is None:
        return found

    def add(name, node):
        found[name] = found.get(name, "") + ast.dump(node, include_attributes=False)

    def visit(body, prefix=""):
        remaining = []
        for node in body:
            if isinstance(node, ast.ClassDef):
                header = copy.deepcopy(node)
                header.body = []
                add(prefix + node.name, header)
                if not prefix:
                    add("<definition_order>", ast.Constant(value=node.name))
                visit(node.body, prefix + node.name + ".")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                add(prefix + node.name, node)
                if not prefix:
                    add("<definition_order>", ast.Constant(value=node.name))
            elif prefix and isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if len(targets) == 1 and isinstance(targets[0], ast.Name):
                    add(prefix + targets[0].id, node)
                else:
                    remaining.append(node)
            elif isinstance(node, ast.ImportFrom) and in_parallel_package(node.module or ""):
                add(prefix + "<parallel_imports>", node)
                marker = ast.Expr(value=ast.Constant(value="parallel-import-block"))
                if not remaining or ast.dump(remaining[-1]) != ast.dump(marker):
                    remaining.append(marker)
            elif not (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                remaining.append(node)
        if remaining:
            add(prefix + "<module>", ast.Module(body=remaining, type_ignores=[]))

    visit(tree.body)
    return found


def semantic_changes(
    path: str, before: str | None, after: str | None, bindings: ImportBindings
) -> list[str]:
    try:
        old = definitions(bindings.normalize(path, before) if before is not None else None)
        new = definitions(bindings.normalize(path, after) if after is not None else None)
        return sorted(k for k in old.keys() | new.keys() if old.get(k) != new.get(k))
    except (SyntaxError, ValueError, RecursionError, ImportError):
        return ["<unparsed>"]


def export_bridge(
    path: str, before: str | None, after: str | None, bindings: ImportBindings
) -> bool:
    """Recognize a compatibility module whose old definitions remain AST-identical.

    New destinations are still changed source owners. This recognizes only the
    bridge; it does not exempt destination implementations or their dependencies.
    """
    if before is None or after is None:
        return False
    try:
        old, new = bindings.normalize(path, before), ast.parse(after)
        if any(not isinstance(n, (ast.ImportFrom, ast.Expr, ast.Assign)) for n in new.body):
            return False
        for n in new.body:
            if isinstance(n, ast.Expr) and not (
                isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)
            ):
                return False
            if isinstance(n, ast.Assign) and not (
                len(n.targets) == 1
                and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id == "__all__"
                and isinstance(ast.literal_eval(n.value), (tuple, list))
            ):
                return False
        # Unchanged functions alone do not prove a bridge: module constants or
        # registration side effects could have disappeared beside them.
        for node in old.body:
            if isinstance(
                node,
                (ast.Import, ast.ImportFrom, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
            ):
                continue
            if (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                continue
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                return False
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if len(targets) != 1 or not isinstance(targets[0], ast.Name):
                return False
            name = targets[0].id
            if name == "__all__":
                target = new
            else:
                resolved = bindings.resolve(module_name(path), name)
                if resolved is None or resolved == (module_name(path), name):
                    return False
                target_path = "src/" + resolved[0].replace(".", "/") + ".py"
                target = bindings.normalize(target_path, bindings.source(target_path))
            assignments = [
                n
                for n in target.body
                if isinstance(n, type(node)) and ast.dump(n) == ast.dump(node)
            ]
            if len(assignments) != 1:
                return False
        nodes = [
            n
            for n in old.body
            if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        if not nodes:
            return False
        for node in nodes:
            resolved = bindings.resolve(module_name(path), node.name)
            if resolved is None or resolved == (module_name(path), node.name):
                return False
            target_path = "src/" + resolved[0].replace(".", "/") + ".py"
            target = bindings.normalize(target_path, bindings.source(target_path))
            matches = [
                n for n in target.body if isinstance(n, type(node)) and n.name == resolved[1]
            ]
            if len(matches) != 1 or ast.dump(node) != ast.dump(matches[0]):
                return False
        return True
    except (SyntaxError, ValueError, TypeError, RecursionError, ImportError):
        return False


def test_case_affected(
    case: dict, path: str, symbols: list[str], dependencies: dict | None = None
) -> bool:
    """Narrow direct test edits; shared fixtures, module state and helpers stay broad."""
    if not symbols or any(name.startswith("<") or "." in name for name in symbols):
        return True
    if any(not name.startswith("test_") for name in symbols) and dependencies is None:
        return True
    arguments = [arg for arg in case["command"] if arg == path or arg.startswith(path + "::")]
    if not arguments:
        return not any(arg.startswith("tests/") for arg in case["command"]) or any(
            not name.startswith("test_") for name in symbols
        )
    for arg in arguments:
        if arg == path:
            return True
        node = arg.split("::", 1)[1].split("[", 1)[0]
        if any(
            name == node
            or name.startswith(node + ".")
            or name in (dependencies or {}).get(node, [])
            for name in symbols
        ):
            return True
    return False


def test_dependencies(before: str | None, after: str | None) -> dict | None:
    """Conservatively trace in-file helper references, including passed worker functions.

    Module state edits are handled separately. Missing/dynamic source falls back
    to file-wide selection; old and new graphs are unioned for removed helpers.
    """
    graph = {}
    try:
        for source in (before, after):
            if source is None:
                continue
            tree = ast.parse(source)
            if any(
                isinstance(n, ast.Name)
                and n.id in {"globals", "locals", "eval", "exec", "__import__"}
                for n in ast.walk(tree)
            ):
                return None
            if any(
                isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                and not n.name.startswith("test_")
                and n.decorator_list
                for n in tree.body
            ):
                return None
            functions = {
                n.name: n
                for n in tree.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            if len(functions) > 1000:
                return None
            # An aliased/stored helper or a method-mediated callback needs more
            # than the bounded top-level call graph, so keep file-wide coverage.
            for node in tree.body:
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
                    isinstance(item, ast.Name)
                    and isinstance(item.ctx, ast.Load)
                    and item.id in functions
                    for item in ast.walk(node)
                ):
                    return None
            for name, node in functions.items():
                names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
                names.update(n.arg for n in ast.walk(node) if isinstance(n, ast.arg))
                names.update(
                    n.value
                    for n in ast.walk(node)
                    if isinstance(n, ast.Constant) and isinstance(n.value, str)
                )
                graph.setdefault(name, set()).update(names & functions.keys())
        for name in graph:
            pending = list(graph[name])
            seen = set()
            while pending:
                dependency = pending.pop()
                if dependency not in seen:
                    seen.add(dependency)
                    pending.extend(graph.get(dependency, []))
            graph[name] = seen
        return {k: sorted(v) for k, v in graph.items()}
    except (SyntaxError, ValueError, RecursionError):
        return None


def whole_module_move(
    path: str, after: str, originals: dict[str, str], bindings: ImportBindings
) -> str | None:
    """Recognize a complete module relocation, including constants and imports."""

    class DropDocs(ast.NodeTransformer):
        def visit_Expr(self, node):
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                return None
            return self.generic_visit(node)

    try:
        current = ast.dump(DropDocs().visit(bindings.normalize(path, after)))
        for previous, source in originals.items():
            if current == ast.dump(DropDocs().visit(bindings.normalize(previous, source))):
                return previous
    except (SyntaxError, ValueError, ImportError, RecursionError):
        return None
    return None
