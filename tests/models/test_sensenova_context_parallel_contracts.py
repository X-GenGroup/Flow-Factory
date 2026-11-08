# Copyright 2026 Jayce-Ping
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""SenseNova's native prefix/layout and fail-closed CP contracts."""

import json
from dataclasses import asdict

import pytest
import torch
from test_sensenova import _tiny_config

from flow_factory.models.sensenova.modeling.neo_unify.modeling_neo_chat import NEOChatModel
from flow_factory.models.sensenova.pipeline import SenseNovaDenoiser, SenseNovaTextPrefix
from flow_factory.models.sensenova.sensenova import SenseNovaAdapter, SenseNovaContextParallelLayout


def test_sensenova_cp_capability_has_model_owned_stable_identity():
    capability = SenseNovaAdapter.context_parallel_capability
    assert capability.layout is SenseNovaContextParallelLayout.PREFIX_SPATIAL_FLOW
    assert json.loads(json.dumps(asdict(capability))) == dict(
        layout="sensenova_prefix_spatial_flow",
        transport="ulysses",
        output_gradient="local_slice",
        version=1,
        tensor_parallel=True,
        required_precision_policy=None,
    )
    assert SenseNovaAdapter.context_parallel_diffusers_revision is None


@pytest.mark.parametrize("pixel", [False, True])
def test_sensenova_cp_validates_both_native_flow_heads(pixel):
    module = SenseNovaDenoiser(NEOChatModel(_tiny_config(pixel)))
    SenseNovaAdapter.validate_context_parallel_module(module, 2, 2)
    with pytest.raises(ValueError, match="Q/KV heads"):
        SenseNovaAdapter.validate_context_parallel_module(module, 4, 2)
    module.model.language_model.model.config.attention_dropout = 0.1
    with pytest.raises(ValueError, match="dropout"):
        SenseNovaAdapter.validate_context_parallel_module(module, 2, 2)


def test_sensenova_cp_rejects_foreign_module_without_mutation():
    module = torch.nn.Linear(2, 2)
    with pytest.raises(ValueError, match="native SenseNovaDenoiser"):
        SenseNovaAdapter.validate_context_parallel_module(module, 2, 2)
    assert not hasattr(module, "_parallel_config")


def test_sensenova_prefix_invocation_is_typed_and_separate_from_generation():
    module = SenseNovaDenoiser(NEOChatModel(_tiny_config(True)))
    module.enable_gradient_checkpointing()
    assert all(block.gradient_checkpointing for block in module.model.language_model.model.layers)
    assert not module.model.language_model.model.gradient_checkpointing
    request = SenseNovaTextPrefix(
        torch.tensor([[2, 3, 4]]),
        torch.stack(
            [torch.arange(3), torch.zeros(3, dtype=torch.long), torch.zeros(3, dtype=torch.long)]
        ),
        {"full_attention": None},
    )
    with pytest.raises(ValueError, match="cannot share"):
        module(prefix=request, latents=torch.zeros(1, 3, 32, 32))
    with pytest.raises(TypeError, match="Unsupported SenseNova prefix"):
        module(prefix="invalid")
    with pytest.raises(ValueError, match="missing required"):
        module(latents=torch.zeros(1, 3, 32, 32))
    module.disable_gradient_checkpointing()
    assert not any(getattr(child, "gradient_checkpointing", False) for child in module.modules())


@pytest.mark.parametrize("pixel", [False, True], ids=["u10", "u15"])
def test_sensenova_native_model_checkpoint_roundtrip_preserves_both_heads(tmp_path, pixel):
    torch.manual_seed(721)
    original = SenseNovaDenoiser(NEOChatModel(_tiny_config(pixel)))
    root = tmp_path / "model"
    original.model.save_pretrained(root)
    restored = SenseNovaDenoiser(NEOChatModel.from_pretrained(root))
    assert restored.model.config.vision_config.llm_hidden_size == (32,)
    assert restored.model.config.vision_config.downsample_ratio == (0.5,)
    assert restored.model.config.use_pixel_head is pixel
    for name, value in original.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)
    for model in (original, restored):
        SenseNovaAdapter.validate_context_parallel_module(model, 2, 2)
    # A real prefix and generation re-enters the restored modules, including the
    # native spatial head; state equality alone cannot detect config nesting.
    request = SenseNovaTextPrefix(
        torch.tensor([[2, 3, 4]]),
        torch.stack(
            [torch.arange(3), torch.zeros(3, dtype=torch.long), torch.zeros(3, dtype=torch.long)]
        ),
        {"full_attention": None},
    )
    outputs = []
    latent = torch.randn(1, 3, 64, 64)
    for model in (original, restored):
        prefix = model(prefix=request)
        indexes = model.model._build_t2i_image_indexes(2, 2, 3, device=torch.device("cpu"))
        outputs.append(
            model(
                latents=latent,
                timestep=torch.tensor(0.4),
                past_key_values=prefix.past_key_values,
                indexes_image=indexes,
                attention_mask={"full_attention": None},
                image_size=(64, 64),
                noise_scale=1.0,
            )
        )
    torch.testing.assert_close(*outputs, rtol=0, atol=0)


@pytest.mark.parametrize("value", [[], [1, 2], [[0.5]]])
@pytest.mark.parametrize("name", ["llm_hidden_size", "downsample_ratio"])
def test_sensenova_vision_config_rejects_ambiguous_singletons(name, value):
    from flow_factory.models.sensenova.modeling.neo_unify.configuration_neo_vit import (
        NEOVisionConfig,
    )

    with pytest.raises(ValueError, match="requires one scalar"):
        NEOVisionConfig(**{name: value})
