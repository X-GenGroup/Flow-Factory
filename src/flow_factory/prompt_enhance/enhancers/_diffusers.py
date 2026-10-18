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

"""Load only the text-side components of a Diffusers pipeline."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import torch


def load_text_only_pipeline(
    pipeline_class: Any,
    model_name_or_path: Optional[str],
    *,
    keep: Sequence[str],
    dtype: torch.dtype,
) -> Any:
    """Instantiate ``pipeline_class`` with every component outside ``keep`` set to ``None``.

    Passing ``None`` for a component makes Diffusers skip loading it, so the
    transformer, VAEs, and other generation-only weights never reach memory.

    Args:
        pipeline_class: Diffusers pipeline class that owns the enhancement method.
        model_name_or_path: Pipeline repo or local directory.
        keep: Component names the enhancer needs.
        dtype: Load dtype of the kept model components.

    Returns:
        Pipeline whose skipped components are ``None``.

    Raises:
        ValueError: If ``model_name_or_path`` is unset.
    """
    if not model_name_or_path:
        raise ValueError(
            f"{pipeline_class.__name__} prompt enhancement requires model_name_or_path"
        )
    config = pipeline_class.load_config(model_name_or_path)
    skipped = {
        name: None
        for name, value in config.items()
        if not name.startswith("_")
        and isinstance(value, (list, tuple))
        and len(value) == 2
        and name not in keep
    }
    return pipeline_class.from_pretrained(model_name_or_path, dtype=dtype, **skipped)
