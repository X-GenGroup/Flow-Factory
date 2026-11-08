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

"""Keep the user-facing TP/SP capability table tied to registry contracts."""

import importlib.util

import pytest

from flow_factory.models.registry import get_model_adapter_class, list_registered_models
from flow_factory.models.tensor_parallel import resolve_parallel_plan, sequence_parallel_enabled

# model -> (TP plan, SP plan, SP enabled by default). Every registry entry must
# be classified; adding an adapter requires a deliberate support-table update.
CAPABILITIES = {
    "sd3-5": (True, True, True),
    "flux1": (True, True, False),
    "flux1-kontext": (True, True, False),
    "flux2": (True, True, False),
    "flux2-klein": (True, True, False),
    "flux3": (True, False, False),
    "qwen-image": (True, True, False),
    "qwen-image-edit-plus": (True, True, False),
    "qwen-image-2.1": (True, True, False),
    "z-image": (True, True, False),
    "wan2_i2v": (True, True, False),
    "wan2_t2v": (True, True, True),
    "bagel": (True, False, False),
    "sensenova": (True, True, False),
    "ltx2_t2av": (True, True, False),
    "ltx2_i2av": (True, True, False),
    "minimax-h3-t2va": (True, True, False),
    "minimax-h3-fl2va": (True, True, False),
    "minimax-h3-ref2va": (True, True, False),
}


def test_every_builtin_model_has_an_explicit_parallelism_classification() -> None:
    assert CAPABILITIES.keys() == list_registered_models().keys()


@pytest.mark.parametrize("name", sorted(CAPABILITIES))
def test_default_and_explicit_sp_match_the_published_capabilities(name: str) -> None:
    if name == "bagel" and importlib.util.find_spec("flash_attn") is None:
        pytest.skip("Bagel capability import requires optional flash-attn")
    adapter = get_model_adapter_class(name)
    has_tp, has_sp, default = CAPABILITIES[name]
    assert bool(adapter.tensor_parallel_plan()) == has_tp
    assert bool(adapter.sequence_parallel_plan()) == has_sp
    assert sequence_parallel_enabled(adapter, None) is default
    assert (
        resolve_parallel_plan(adapter, sequence_parallel=False).keys()
        == adapter.tensor_parallel_plan().keys()
    )
    automatic = resolve_parallel_plan(adapter, sequence_parallel=None)
    explicit = resolve_parallel_plan(adapter, sequence_parallel=default)
    assert automatic.keys() == explicit.keys()
    if has_sp:
        assert resolve_parallel_plan(adapter, sequence_parallel=True).keys() == (
            adapter.tensor_parallel_plan().keys() | adapter.sequence_parallel_plan().keys()
        )
    else:
        with pytest.raises(ValueError, match="declares no sequence-parallel plan"):
            resolve_parallel_plan(adapter, sequence_parallel=True)
