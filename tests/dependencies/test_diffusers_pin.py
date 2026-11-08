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

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_DIFFUSERS_REQUIREMENT = "diffusers>=0.42.1"
EXPECTED_WAN_PROMPT_DEPENDENCY = "ftfy"


def test_project_metadata_requires_released_diffusers_with_flux3_support() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    diffusers_requirements = re.findall(r'"(diffusers[^"]*)"', pyproject)

    assert diffusers_requirements == [EXPECTED_DIFFUSERS_REQUIREMENT], (
        "pyproject.toml must contain exactly one released diffusers requirement with "
        f"FLUX.3 support; expected={EXPECTED_DIFFUSERS_REQUIREMENT!r}, "
        f"observed={diffusers_requirements!r}"
    )


def test_project_metadata_requires_wan_prompt_normalization_dependency() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    ftfy_requirements = re.findall(r'"(ftfy[^"]*)"', pyproject)

    assert ftfy_requirements == [EXPECTED_WAN_PROMPT_DEPENDENCY], (
        "pyproject.toml must contain exactly one core ftfy requirement for Wan prompt "
        f"normalization; expected={EXPECTED_WAN_PROMPT_DEPENDENCY!r}, "
        f"observed={ftfy_requirements!r}"
    )


def test_container_diffusers_pin_satisfies_the_project_floor() -> None:
    root = Path(__file__).resolve().parents[2]
    constraints = (root / "docker/docker-cuda/constraints.txt").read_text()
    pinned = next(line for line in constraints.splitlines() if line.startswith("diffusers=="))
    installed_version = pinned.split("==", 1)[1]
    assert installed_version in Requirement(EXPECTED_DIFFUSERS_REQUIREMENT).specifier
