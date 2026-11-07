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

"""Native callback ordering with local TP stand-ins, without numerical CP claims."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from diffusers.hooks.context_parallel import EquipartitionSharder, apply_context_parallel

from diffusers import QwenImageTransformer2DModel
from flow_factory.models.qwen_image._context_parallel import compose_qwen_sequence_boundaries


class _Block(torch.nn.Module):
    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        encoder_hidden_states_mask,
        temb,
        image_rotary_emb=None,
        joint_attention_kwargs=None,
        modulate_index=None,
    ):
        self.observed = (
            hidden_states.clone(),
            encoder_hidden_states.clone(),
            encoder_hidden_states_mask.clone(),
            None if modulate_index is None else modulate_index.clone(),
        )
        return encoder_hidden_states, hidden_states


def _native_hooks():
    model = torch.nn.Module()
    model.img_in = torch.nn.Identity()
    model.transformer_blocks = torch.nn.ModuleList([_Block(), _Block()])
    model.pos_embed = torch.nn.Identity()
    model.proj_out = torch.nn.Identity()
    mesh = SimpleNamespace(size=lambda: 2, get_group=lambda: "cp")
    config = SimpleNamespace(
        ulysses_anything=False, ring_anything=False, _mesh=mesh, _flattened_mesh=mesh
    )
    apply_context_parallel(model, config, QwenImageTransformer2DModel._cp_plan)
    return model


@pytest.mark.parametrize("cp_rank", [0, 1])
@pytest.mark.parametrize("tp_rank", [0, 1])
@pytest.mark.parametrize("positional", [False, True])
@pytest.mark.parametrize("with_modulation", [False, True])
def test_native_cp_precedes_image_sp_and_keeps_text_replicated(
    monkeypatch, cp_rank, tp_rank, positional, with_modulation
):
    model = _native_hooks()
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group: cp_rank)
    events = []

    def scatter_image(module, args, output):
        events.append(("image_sp", output.shape[1]))
        return output.chunk(2, dim=1)[tp_rank]

    def scatter_index(module, args, kwargs):
        value = kwargs.get("modulate_index") if kwargs else args[6]
        if value is None:
            return None
        events.append(("index_sp", value.shape[1]))
        narrowed = value.chunk(2, dim=1)[tp_rank]
        if kwargs:
            return args, {**kwargs, "modulate_index": narrowed}
        return (*args[:6], narrowed), kwargs

    def gather_sp(module, args, output):
        events.append(("gather_sp", output.shape[1]))
        return torch.cat([output, output], dim=1)

    def gather_cp(tensor, dim, mesh):
        events.append(("gather_cp", tensor.shape[dim]))
        return torch.cat([tensor, tensor], dim=dim)

    model.img_in.register_forward_hook(scatter_image)
    for block in model.transformer_blocks:
        block.register_forward_pre_hook(scatter_index, with_kwargs=True)
    model.proj_out.register_forward_hook(gather_sp)
    monkeypatch.setattr(EquipartitionSharder, "unshard", gather_cp)
    rope_hook = model.pos_embed._diffusers_hook.get_hook("cp_input---pos_embed")
    boundaries = compose_qwen_sequence_boundaries(model)
    assert len(boundaries) == len(model.transformer_blocks) + 3
    assert model.pos_embed._diffusers_hook.get_hook("cp_input---pos_embed") is rope_hook

    image = torch.arange(16).reshape(1, 16, 1).repeat(2, 1, 1).float()
    text = torch.arange(6).reshape(1, 6, 1).repeat(2, 1, 1).float()
    mask = torch.tensor([[1, 0, 1, 1, 0, 1], [1, 1, 0, 1, 0, 0]])
    index = torch.arange(16).reshape(1, 16).repeat(2, 1) if with_modulation else None
    hidden = model.img_in(image)
    expected_text = text[:, 3 * cp_rank : 3 * (cp_rank + 1)].clone()
    rotary = model.pos_embed((image[0], text[0]))
    for block in model.transformer_blocks:
        values = dict(
            hidden_states=hidden,
            encoder_hidden_states=text,
            encoder_hidden_states_mask=mask,
            temb=torch.ones(2, 1),
            image_rotary_emb=rotary,
            joint_attention_kwargs=None,
            modulate_index=index,
        )
        text, hidden = block(*values.values()) if positional else block(**values)
        received_image, received_text, received_mask, received_index = block.observed
        start = 8 * cp_rank + 4 * tp_rank
        torch.testing.assert_close(received_image, image[:, start : start + 4])
        torch.testing.assert_close(received_text, expected_text)
        assert received_text.shape == (2, 3, 1)
        torch.testing.assert_close(received_mask, mask[:, 3 * cp_rank : 3 * (cp_rank + 1)])
        if with_modulation:
            torch.testing.assert_close(received_index, index[:, start : start + 4])
        else:
            assert received_index is None
    assert model.proj_out(hidden).shape == image.shape
    assert events[0] == ("image_sp", 8)
    assert [event for event in events if event[0] == "index_sp"] == (
        [("index_sp", 8)] * 2 if with_modulation else []
    )
    assert events[-2:] == [("gather_sp", 4), ("gather_cp", 8)]


@pytest.mark.parametrize("corruption", ["input", "gather", "last_mask"])
def test_unqualified_native_metadata_fails_before_any_boundary_mutation(corruption):
    model = _native_hooks()
    if corruption == "input":
        hook = model.transformer_blocks[0]._diffusers_hook.get_hook(
            "cp_input---transformer_blocks.0"
        )
        hook.metadata = {
            **hook.metadata,
            "hidden_states": replace(hook.metadata["hidden_states"], split_dim=2),
        }
    elif corruption == "gather":
        hook = model.proj_out._diffusers_hook.get_hook("cp_output---proj_out")
        hook.metadata = [replace(hook.metadata[0], gather_dim=2)]
    else:
        hook = model.transformer_blocks[-1]._diffusers_hook.get_hook(
            "cp_input---transformer_blocks.*"
        )
        hook.metadata = {"modulate_index": hook.metadata["modulate_index"]}
    modules = [model.img_in, *model.transformer_blocks, model.proj_out, model.pos_embed]
    before = [(dict(m._forward_pre_hooks), dict(m._forward_hooks), m.forward) for m in modules]
    with pytest.raises(ValueError, match="Qwen TP/SP\\+CP requires"):
        compose_qwen_sequence_boundaries(model)
    assert [
        (dict(m._forward_pre_hooks), dict(m._forward_hooks), m.forward) for m in modules
    ] == before
