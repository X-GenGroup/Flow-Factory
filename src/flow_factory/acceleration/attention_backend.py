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

# src/flow_factory/acceleration/attention_backend.py
"""Attention-backend accelerator: the single code path for diffusers backend selection.

The backend is requested as an ``attention_backend`` entry in the acceleration ``shared``
list, so it flows through the same plugin mechanism as other transformer-level acceleration.
It is applied after ``accelerator.prepare`` / ``post_init``; list it before
``torch_compile`` so the compiled graph captures the backend.

The backend name is taken from the required ``backend`` param and forwarded to
diffusers' ``set_attention_backend`` verbatim, including approximate backends like
``sage``.

Marked ``stage='both'`` / ``safety='lossless'``: a backend is applied to the
transformer shared by rollout ``inference()`` and training ``forward()``, so the
two stay consistent (the property the validator's lossless category guarantees)
even for approximate kernels and coupled algorithms.
"""

from typing import TYPE_CHECKING

from ..utils.logger_utils import setup_logger
from .abc import BaseAccelerator

if TYPE_CHECKING:
    from ..models.abc import BaseAdapter

logger = setup_logger(__name__)


class AttentionBackendAccelerator(BaseAccelerator):
    """Set the diffusers attention backend on every transformer component.

    See https://huggingface.co/docs/diffusers/main/en/optimization/attention_backends
    for the full list of supported backends.

    Args:
        backend: Backend name forwarded to ``transformer.set_attention_backend``
            (e.g. ``native`` / ``flash`` / ``_flash_3`` / ``_flash_3_hub`` /
            ``sage`` / ``xformers``). Required; given in the entry's ``params``.
    """

    safety = "lossless"
    stage = "both"

    def setup(self, adapter: "BaseAdapter") -> None:
        backend = self.params.get("backend")
        if not backend:
            raise ValueError(
                "AttentionBackendAccelerator requires a `backend` param, e.g. "
                "`{ name: attention_backend, params: { backend: _flash_3_hub } }`."
            )

        applied = False
        for name in adapter.transformer_names:
            transformer = adapter.get_component(name)
            if hasattr(transformer, "set_attention_backend"):
                transformer.set_attention_backend(backend)
                applied = True
                if adapter.accelerator.is_main_process:
                    logger.info(
                        "AttentionBackendAccelerator: set backend '%s' for '%s'.", backend, name
                    )
        if not applied:
            raise ValueError(
                f"AttentionBackendAccelerator: backend '{backend}' requested but none of the "
                f"adapter's transformer components {adapter.transformer_names} support "
                "`set_attention_backend`. Models with a custom attention implementation (e.g. "
                "Bagel, which forces flash_attention_2 at load) must not use this accelerator; "
                "remove the `attention_backend` entry from the acceleration config."
            )
