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

# src/flow_factory/models/bagel/__init__.py
"""Bagel model adapter.

Integrates ByteDance's Bagel multimodal model into Flow-Factory.
Supports Text-to-Image and Image(s)-to-Image generation tasks.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "BagelAdapter",
    "BagelSample",
    "BagelI2ISample",
    "BagelPseudoPipeline",
]


def __getattr__(name: str) -> Any:
    """Load optional-kernel Bagel classes only when callers request them."""
    if name in {"BagelAdapter", "BagelSample", "BagelI2ISample"}:
        return getattr(import_module(".bagel", __name__), name)
    if name == "BagelPseudoPipeline":
        return getattr(import_module(".pipeline", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
