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

"""Local-only tensor snapshots for native objective observers."""

import dataclasses

import torch


def cpu(value):
    """Detach only local tensor values; never materialize a DTensor collectively."""
    if isinstance(value, torch.Tensor):
        if hasattr(value, "to_local"):
            value = value.to_local()
        return value.detach().cpu().clone()
    if dataclasses.is_dataclass(value):
        return {field.name: cpu(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {key: cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)
