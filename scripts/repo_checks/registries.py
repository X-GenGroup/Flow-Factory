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

"""Read authoritative registry declarations and explicit example references via AST/YAML."""

import ast
from dataclasses import dataclass

import yaml

from .core import Repository, dotted, import_name, module_name


@dataclass(frozen=True)
class Registry:
    path: str
    variable: str
    case_sensitive: bool = False
    python_path: bool = True


REGISTRIES = {
    "model": Registry("models/registry.py", "_MODEL_ADAPTER_REGISTRY"),
    "trainer": Registry("trainers/registry.py", "_TRAINER_REGISTRY"),
    "reward": Registry("rewards/registry.py", "_REWARD_MODEL_REGISTRY"),
    "accelerator": Registry("acceleration/registry.py", "_ACCELERATOR_REGISTRY"),
    "scheduler": Registry("scheduler/registry.py", "_SCHEDULER_REGISTRY", True, False),
    "logger": Registry("logger/registry.py", "_LOGGER_REGISTRY"),
    "enhancer": Registry("prompt_enhance/registry.py", "_PROMPT_ENHANCER_REGISTRY"),
    "training_args": Registry("hparams/training_args/_registry.py", "_TRAINING_ARGS_REGISTRY"),
    "optimizer": Registry(
        "hparams/optimizer_args/_registry.py", "_OPTIMIZER_ARGS_REGISTRY", False, False
    ),
}


class Symbols:
    def __init__(self, repo: Repository):
        self.repo = repo
        self.bindings: dict[str, dict[str, str | None]] = {}

    def module(self, name: str) -> str:
        base = "src/" + name.replace(".", "/")
        for path in (base + ".py", base + "/__init__.py"):
            if self.repo.exists(path):
                return path
        raise ValueError(f"missing built-in module {name!r}")

    def names(self, path: str) -> dict[str, str | None]:
        if path not in self.bindings:
            names: dict[str, str | None] = {}
            for node in self.repo.tree(path).body:
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    names[node.name] = None
                elif isinstance(node, ast.ImportFrom):
                    parent = import_name(path, node)
                    for alias in node.names:
                        names[alias.asname or alias.name] = f"{parent}.{alias.name}"
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        names[alias.asname or alias.name.split(".")[0]] = (
                            alias.name if alias.asname else alias.name.split(".")[0]
                        )
                elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    target = dotted(node.value) if node.value else None
                    for name in targets:
                        if isinstance(name, ast.Name) and node.value is not None:
                            names[name.id] = f"@{target}" if target else "!non-static"
                elif isinstance(node, ast.Delete):
                    for name in node.targets:
                        if isinstance(name, ast.Name):
                            names.pop(name.id, None)
            self.bindings[path] = names
        return self.bindings[path]

    def local(self, path: str, name: str, seen: frozenset[str]) -> None:
        marker = f"{path}:{name}"
        if marker in seen:
            raise ValueError(f"cyclic alias/re-export at {marker}")
        seen = seen | {marker}
        first, _, rest = name.partition(".")
        names = self.names(path)
        if first not in names:
            raise ValueError(f"missing or non-static symbol {name!r} in {path}")
        target = names[first]
        if target == "!non-static":
            raise ValueError(f"non-static symbol {name!r} in {path}")
        if target is None:
            if rest:
                raise ValueError(f"nested symbol {name!r} needs an explicit module re-export")
            return
        target += "." + rest if rest else ""
        if target.startswith("@"):
            self.local(path, target[1:], seen)
        else:
            self.resolve(target, seen)

    def resolve(self, target: str, seen: frozenset[str] = frozenset()) -> None:
        parts = target.split(".")
        if len(parts) < 2 or not all(part.isidentifier() for part in parts):
            raise ValueError(f"invalid Python class path {target!r}")
        if parts[0] != "flow_factory":
            self.repo.unverified.add(
                f"external Python path {target}: requires runtime/plugin evidence"
            )
            return
        path = self.module(".".join(parts[:-1]))
        self.local(path, parts[-1], seen)


def read_registry(repo: Repository, spec: Registry, symbols: Symbols) -> dict[str, str | None]:
    path = "src/flow_factory/" + spec.path
    values: dict[str, str | None] = {}
    declarations: list[ast.AST] = []
    normalization = ast.parse(
        f"{{k.lower(): v for k, v in {spec.variable}.items()}}", mode="eval"
    ).body
    normalized = False
    for node in repo.tree(path).body:
        if not isinstance(
            node,
            (
                ast.Assign,
                ast.AnnAssign,
                ast.Expr,
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
                ast.Import,
                ast.ImportFrom,
            ),
        ) and any(
            isinstance(child, ast.Name) and child.id == spec.variable for child in ast.walk(node)
        ):
            raise ValueError(
                f"{path}:{node.lineno}: conditional/dynamic registry mutation needs an explicit static declaration"
            )
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id == spec.variable for t in targets):
                if node.value and ast.dump(node.value) == ast.dump(normalization):
                    if not declarations or normalized or spec.case_sensitive:
                        raise ValueError(
                            f"{path}:{node.lineno}: unsupported registry normalization order"
                        )
                    normalized = True
                else:
                    declarations.append(node.value)
            elif any(
                isinstance(t, ast.Subscript) and dotted(t.value) == spec.variable for t in targets
            ):
                raise ValueError(
                    f"{path}:{node.lineno}: static registry cannot use item assignments"
                )
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            callee = dotted(node.value.func) or ""
            if callee.startswith(spec.variable + "."):
                raise ValueError(f"{path}:{node.lineno}: static registry cannot use {callee}")
    if len(declarations) != 1 or not isinstance(declarations[0], ast.Dict):
        raise ValueError(f"{path}: {spec.variable} requires one literal dictionary declaration")
    for key, value in zip(declarations[0].keys, declarations[0].values):
        line = getattr(key, "lineno", 1)
        if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
            repo.error(
                path,
                line,
                "registry-key",
                "registry keys must be literal strings, without unpacking",
            )
            continue
        name = key.value if spec.case_sensitive else key.value.lower()
        if name in values:
            repo.error(path, line, "registry-duplicate", f"duplicate lookup key {name!r}")
        if not normalized and key.value != name:
            repo.error(
                path,
                line,
                "registry-case",
                f"runtime lowercases lookups but key {key.value!r} is not lowercase",
            )
        try:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                target = value.value
                symbols.resolve(target)
            elif (
                isinstance(value, ast.Constant)
                and value.value is None
                and name == "none"
                and spec.variable == "_LOGGER_REGISTRY"
            ):
                target = None
            elif dotted(value):
                target = f"{module_name(path)}.{dotted(value)}"
                symbols.local(path, dotted(value), frozenset())
            else:
                raise ValueError(
                    "registry targets must be class paths or explicit imported symbols"
                )
        except ValueError as error:
            repo.error(path, line, "registry-target", str(error))
            target = None
        values[name] = target
    return values


def check_registries(repo: Repository, specs: dict[str, Registry] = REGISTRIES) -> dict[str, dict]:
    symbols = Symbols(repo)
    tables = {}
    for name, spec in specs.items():
        try:
            tables[name] = read_registry(repo, spec, symbols)
        except (ValueError, SyntaxError) as error:
            repo.error("src/flow_factory/" + spec.path, 1, "registry-declaration", str(error))
            tables[name] = {}
    if "trainer" in tables and "training_args" in tables:
        difference = tables["trainer"].keys() ^ tables["training_args"].keys()
        if difference:
            repo.error(
                "src/flow_factory/trainers/registry.py",
                1,
                "registry-parity",
                f"trainer/training-argument keys differ: {sorted(difference)}",
            )
    return tables


def check_examples(repo: Repository, tables: dict[str, dict]) -> None:
    symbols = Symbols(repo)
    for path in sorted(repo.paths):
        if (
            not path.startswith("examples/")
            or not path.endswith((".yaml", ".yml"))
            or not repo.exists(path)
        ):
            continue
        try:
            config = yaml.safe_load(repo.read(path))
            if not isinstance(config, dict):
                raise ValueError("example must be a YAML mapping")

            def reference(kind: str, value: object, field: str) -> None:
                if kind == "logger" and value is None:
                    return
                if not isinstance(value, str) or not value:
                    raise ValueError(f"{field}: expected a nonempty registry key or Python path")
                if value.lower() in tables[kind]:
                    return
                if not REGISTRIES[kind].python_path or "." not in value:
                    raise ValueError(f"{field}: unknown {kind} key {value!r}")
                symbols.resolve(value)

            def mapping(value: object, field: str) -> dict:
                if not isinstance(value, dict):
                    raise ValueError(f"{field}: expected a mapping")
                return value

            def entries(value: object, field: str) -> list:
                if isinstance(value, dict):
                    return [value]
                if isinstance(value, list):
                    return value
                raise ValueError(f"{field}: expected a mapping or list")

            if path.startswith("examples/prompt_enhance/"):
                reference("enhancer", config.get("enhancer"), "enhancer")
                continue
            for section, field, kind in (
                ("model", "model_type", "model"),
                ("train", "trainer_type", "trainer"),
                ("log", "logging_backend", "logger"),
            ):
                if section in config:
                    group = mapping(config[section], section)
                    if field in group:
                        reference(kind, group[field], f"{section}.{field}")
            for section in ("rewards", "eval_rewards"):
                if section in config:
                    for item in entries(config[section], section):
                        reference(
                            "reward",
                            mapping(item, section).get("reward_model"),
                            section + ".reward_model",
                        )
            if config.get("acceleration") is not None:
                group = mapping(config["acceleration"], "acceleration")
                for slot in ("shared", "rollout"):
                    if group.get(slot) is not None:
                        for item in entries(group[slot], "acceleration." + slot):
                            reference(
                                "accelerator",
                                mapping(item, slot).get("name"),
                                "acceleration." + slot,
                            )
            if "optimizers" in config:
                if not isinstance(config["optimizers"], list):
                    raise ValueError("optimizers: expected a list")
                for item in config["optimizers"]:
                    group = mapping(item, "optimizers")
                    if "optimizer" in group:
                        reference("optimizer", group["optimizer"], "optimizers.optimizer")
            target = config.get("config_file")
            if target is not None and (not isinstance(target, str) or not repo.exists(target)):
                raise ValueError(f"config_file: missing tracked launcher config {target!r}")
        except (ValueError, SyntaxError, yaml.YAMLError) as error:
            repo.error(path, 1, "example-reference", str(error))
