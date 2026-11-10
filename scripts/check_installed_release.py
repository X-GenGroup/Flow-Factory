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

"""Exercise installed entry points and lazy registries outside a source checkout."""

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-version", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    import flow_factory
    from flow_factory.models.registry import _MODEL_ADAPTER_REGISTRY, get_model_adapter_class
    from flow_factory.trainers.registry import _TRAINER_REGISTRY, get_trainer_class

    origin = Path(flow_factory.__file__).resolve()
    if not origin.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError(
            f"Imported checkout or external environment instead of installed artifact: {origin}"
        )
    distribution = importlib.metadata.distribution("flow-factory")
    if distribution.version != args.expected_version:
        raise RuntimeError(f"Installed version {distribution.version} != {args.expected_version}")
    direct = json.loads(distribution.read_text("direct_url.json") or "{}")
    if direct.get("dir_info", {}).get("editable"):
        raise RuntimeError("Release verification requires a non-editable install")
    optional_unavailable = {}
    resolved_models = []
    for name in _MODEL_ADAPTER_REGISTRY:
        try:
            get_model_adapter_class(name)
        except ImportError as error:
            from flow_factory.utils.imports import is_flash_attn_available

            if (
                name != "bagel"
                or is_flash_attn_available("2.5.8")
                or "The Bagel model adapter requires flash-attn" not in str(error)
            ):
                raise
            optional_unavailable[name] = str(error)
        else:
            resolved_models.append(name)
    for name in _TRAINER_REGISTRY:
        get_trainer_class(name)
    for name in (
        "ff-train",
        "ff-prompt-enhance",
        "flow-factory-train",
        "flow-factory-prompt-enhance",
    ):
        subprocess.run(
            [str(Path(sys.executable).parent / name), "--help"],
            check=True,
            timeout=90,
            stdout=subprocess.DEVNULL,
        )
    report = {
        "status": "pass",
        "python": platform.python_version(),
        "origin": str(origin),
        "version": distribution.version,
        "models": sorted(_MODEL_ADAPTER_REGISTRY),
        "resolved_models": sorted(resolved_models),
        "optional_unavailable": optional_unavailable,
        "trainers": sorted(_TRAINER_REGISTRY),
        "direct_url": direct,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "accelerate", "diffusers", "transformers", "peft", "torchcodec")
        },
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
