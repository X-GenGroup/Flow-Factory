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

import re
from pathlib import Path
from typing import Dict, Tuple

from flow_factory.trainers.registry import get_trainer_class, list_registered_trainers

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / "guidance/workflow"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _overlap_table() -> Dict[str, Tuple[str, ...]]:
    """Map each trainer type in the overlap guide's support table to its sampler layouts."""
    text = _text(WORKFLOW / "reward_optimization_overlap.md")
    section = text[text.index("## Supported Trainers") : text.index("## Configuration")]
    table: Dict[str, Tuple[str, ...]] = {}
    for line in section.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 2 or not cells[0].startswith("`"):
            continue
        layouts = tuple(sorted(re.findall(r"`([a-z_]+)`", cells[1])))
        for trainer_type in re.findall(r"`([a-z0-9-]+)`", cells[0]):
            table[trainer_type] = layouts
    return table


def test_workflow_guide_links_execution_mode_guides() -> None:
    main = _text(ROOT / "guidance/workflow.md")
    assert "## Execution Modes" in main
    for guide in ("async_reward.md", "reward_optimization_overlap.md"):
        assert (WORKFLOW / guide).is_file()
        assert f"](workflow/{guide})" in main


def test_rewards_guide_routes_async_execution_to_workflow_guides() -> None:
    rewards = _text(ROOT / "guidance/rewards.md")
    assert "](workflow/async_reward.md)" in rewards
    assert "](workflow/reward_optimization_overlap.md)" in rewards
    assert "reward_optimization_overlap_mode" not in rewards


def test_overlap_guide_lists_every_validated_requirement() -> None:
    overlap = " ".join(_text(WORKFLOW / "reward_optimization_overlap.md").split())
    for requirement in (
        "`async_reward: true`",
        "`device: cpu`",
        "`train.global_std: false`",
        "`train.num_inner_epochs: 1`",
        "`train.shuffle_samples: false`",
    ):
        assert requirement in overlap


def test_overlap_guide_support_table_matches_trainer_contracts() -> None:
    table = _overlap_table()
    built_in = {
        name
        for name, class_path in list_registered_trainers().items()
        if class_path.startswith("flow_factory.trainers.")
    }
    assert set(table) <= built_in
    for trainer_type in sorted(built_in):
        contract = get_trainer_class(trainer_type).reward_optimization_overlap_contract
        if contract.supported:
            assert table.get(trainer_type) == tuple(
                sorted(contract.sampler_group_placements)
            ), trainer_type
            assert set(contract.scheduling_modes) == {"ordered", "ready"}, trainer_type
        else:
            assert trainer_type not in table, trainer_type
