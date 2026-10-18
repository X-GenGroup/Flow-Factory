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

"""Launcher and example-config tests for ff-prompt-enhance."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import pytest

from flow_factory import cli
from flow_factory.prompt_enhance import PromptEnhanceArguments, get_prompt_enhancer_class

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("entry_point", "module"),
    [
        (cli.prompt_enhance_cli, "flow_factory.prompt_enhance"),
        (cli.train_cli, "flow_factory.train"),
    ],
)
def test_single_process_launch_runs_the_selected_module(
    entry_point,
    module: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("enhancer: flux3\ninput_dir: in\noutput_dir: out\n", encoding="utf-8")
    commands: List[List[str]] = []
    monkeypatch.setattr(cli.subprocess, "run", lambda cmd, check: commands.append(cmd))
    monkeypatch.setattr(cli, "get_gpu_count", lambda: 0)
    monkeypatch.setattr(cli, "resolve_multinode_env", lambda: {})
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.setattr(sys, "argv", ["ff", str(config)])

    entry_point()

    assert commands == [[sys.executable, "-m", module, str(config)]]


@pytest.mark.parametrize(
    "example",
    sorted((ROOT / "examples" / "prompt_enhance").glob("*.yaml")),
    ids=lambda path: path.stem,
)
def test_prompt_enhance_examples_parse_and_resolve(example: Path) -> None:
    args = PromptEnhanceArguments.load_from_yaml(str(example))
    get_prompt_enhancer_class(args.enhancer)
    assert args.num_variants >= 1
    assert args.output_dir != args.input_dir
