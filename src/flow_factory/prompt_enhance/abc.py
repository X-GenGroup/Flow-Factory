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

"""Interface implemented by every prompt enhancer."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, List

import torch
from PIL import Image

from .args import PromptEnhanceArguments


class BasePromptEnhancer(ABC):
    """Rewrite prompts with one enhancer model.

    The runner owns dataset I/O, sharding, and seeding; an enhancer owns only model
    loading and generation. ``enhance`` runs inside an isolated RNG scope seeded with
    ``seeds[0]``, so global ``torch.manual_seed`` calls stay local to it. Enhancers that
    generate one prompt at a time should seed each prompt with its own entry of ``seeds``,
    which does not depend on batching or sharding.
    """

    uses_images: ClassVar[bool] = False
    supported_generation_keys: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, args: PromptEnhanceArguments, device: torch.device) -> None:
        unsupported = sorted(set(args.generation) - self.supported_generation_keys)
        if unsupported:
            raise ValueError(
                f"{type(self).__name__} does not accept generation keys {unsupported}; "
                f"supported keys are {sorted(self.supported_generation_keys)}"
            )
        self.args = args
        self.device = device

    @abstractmethod
    def load(self) -> None:
        """Load the enhancer model onto ``self.device``."""

    @abstractmethod
    def enhance(
        self,
        prompts: List[str],
        images: List[List[Image.Image]],
        seeds: List[int],
    ) -> List[str]:
        """Return one enhanced prompt per input prompt, in input order.

        Args:
            prompts: Source prompts of one batch.
            images: Reference images of each prompt; empty lists for prompt-only rows.
                Always empty when ``uses_images`` is ``False``.
            seeds: Generation seed of each prompt's ``(row, variant)``.
        """

    def describe(self) -> Dict[str, Any]:
        """Return enhancer facts recorded in the output manifest."""
        return {}
