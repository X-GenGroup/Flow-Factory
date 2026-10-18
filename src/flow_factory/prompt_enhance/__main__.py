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

"""Entry point of ``python -m flow_factory.prompt_enhance <config.yaml>``."""

import argparse
from typing import Optional, Sequence

from .args import PromptEnhanceArguments
from .runner import run_prompt_enhancement


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Parse the YAML config path and run prompt enhancement."""
    parser = argparse.ArgumentParser(description="Flow-Factory Prompt Enhancement")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    parsed = parser.parse_args(argv)
    run_prompt_enhancement(PromptEnhanceArguments.load_from_yaml(parsed.config))


if __name__ == "__main__":
    main()
