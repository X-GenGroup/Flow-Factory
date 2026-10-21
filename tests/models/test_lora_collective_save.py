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

"""A collectively gathered LoRA must survive wrapper segments in the model's module paths.

FSDP2 with activation checkpointing puts ``_checkpoint_wrapped_module`` into every block path,
while ``torch.distributed.checkpoint`` gathers wrapper-free FQNs. PEFT selects adapter entries
from the module paths, so the two used to match nothing and every FSDP2 LoRA checkpoint was an
empty ``adapter_model.safetensors``.
"""

from types import SimpleNamespace

import pytest
import torch
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

from flow_factory.models.abc import (
    LORA_ADAPTER_WEIGHTS_NAME,
    BaseAdapter,
    _unwrapped_parameter_name,
)

LORA_CONFIG = dict(r=2, lora_alpha=4, target_modules=["to_q"])


class Block(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.to_q = torch.nn.Linear(4, 4)
        self.ff = torch.nn.Linear(4, 4)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.ff(self.to_q(inputs))


class Net(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList(Block() for _ in range(2))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            inputs = block(inputs)
        return inputs


class CollectiveAdapter(BaseAdapter):
    """Drive ``BaseAdapter._save_lora`` down its collective path without a pipeline."""

    decode_latents = forward = inference = load_pipeline = None

    def __init__(self, gathered: dict) -> None:
        self.accelerator = SimpleNamespace(
            is_main_process=True,
            wait_for_everyone=lambda: None,
            unwrap_model=lambda module: module,
        )
        self._gathered = gathered

    def _requires_collective_state_dict(self) -> bool:
        return True

    def get_state_dict(self, model, unwrap=True, state_dict_keys=None, ignore_frozen_params=False):
        return dict(self._gathered)


def _checkpoint_wrapped_lora_model(adapter_name: str = "default") -> torch.nn.Module:
    torch.manual_seed(0)
    base = Net()
    base.requires_grad_(False)
    model = get_peft_model(base, LoraConfig(**LORA_CONFIG), adapter_name=adapter_name)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_" in name:
                parameter.copy_(torch.randn_like(parameter))
    blocks = model.base_model.model.blocks
    for index, block in enumerate(blocks):
        blocks[index] = checkpoint_wrapper(block)
    return model


def _gathered_lora(model: torch.nn.Module) -> dict:
    """Mirror ``get_model_state_dict``: wrapper-free FQNs of the trainable LoRA tensors."""
    return {
        _unwrapped_parameter_name(name): parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def test_collective_lora_save_writes_wrapper_free_keys_that_round_trip(tmp_path):
    model = _checkpoint_wrapped_lora_model()
    gathered = _gathered_lora(model)
    assert any("_checkpoint_wrapped_module" in name for name, _ in model.named_parameters())
    assert len(gathered) == 4 and not any("_checkpoint_wrapped_module" in k for k in gathered)

    CollectiveAdapter(gathered)._save_lora(model, str(tmp_path))

    saved = load_file(str(tmp_path / LORA_ADAPTER_WEIGHTS_NAME))
    assert sorted(saved) == [
        f"base_model.model.blocks.{block}.to_q.lora_{matrix}.weight"
        for block in (0, 1)
        for matrix in ("A", "B")
    ]
    torch.manual_seed(1)
    fresh = get_peft_model(Net(), LoraConfig(**LORA_CONFIG))
    result = set_peft_model_state_dict(fresh, saved)
    assert not result.unexpected_keys
    restored = {name: parameter for name, parameter in fresh.named_parameters() if "lora_" in name}
    assert restored.keys() == gathered.keys()
    for name, value in gathered.items():
        assert torch.equal(restored[name], value), name


def test_collective_lora_save_of_a_variant_adapter_lands_flat_and_round_trips(tmp_path):
    adapter_name = "role_a"
    model = _checkpoint_wrapped_lora_model(adapter_name)
    gathered = _gathered_lora(model)
    adapter = CollectiveAdapter(gathered)
    adapter.component_variant_registry = SimpleNamespace(
        active_variant="variant",
        get_spec=lambda variant: SimpleNamespace(adapter_name=adapter_name),
    )

    adapter._save_lora(model, str(tmp_path))

    assert not (tmp_path / adapter_name).exists()
    saved = load_file(str(tmp_path / LORA_ADAPTER_WEIGHTS_NAME))
    assert sorted(saved) == [
        f"base_model.model.blocks.{block}.to_q.lora_{matrix}.weight"
        for block in (0, 1)
        for matrix in ("A", "B")
    ]
    torch.manual_seed(1)
    fresh = get_peft_model(Net(), LoraConfig(**LORA_CONFIG), adapter_name=adapter_name)
    result = set_peft_model_state_dict(fresh, saved, adapter_name=adapter_name)
    assert not result.unexpected_keys
    restored = {name: parameter for name, parameter in fresh.named_parameters() if "lora_" in name}
    assert restored.keys() == gathered.keys()
    for name, value in gathered.items():
        assert torch.equal(restored[name], value), name


def test_collective_lora_save_rejects_keys_outside_the_model(tmp_path):
    model = _checkpoint_wrapped_lora_model()
    gathered = _gathered_lora(model)
    gathered["base_model.model.missing.lora_A.default.weight"] = torch.zeros(2, 4)

    with pytest.raises(KeyError, match="unknown keys"):
        CollectiveAdapter(gathered)._save_lora(model, str(tmp_path))


def test_collective_lora_save_rejects_an_empty_adapter(tmp_path):
    model = _checkpoint_wrapped_lora_model()

    with pytest.raises(RuntimeError, match="to hold tensors"):
        CollectiveAdapter({})._save_lora(model, str(tmp_path))
