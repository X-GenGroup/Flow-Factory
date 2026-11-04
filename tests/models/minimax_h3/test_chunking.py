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

from copy import deepcopy
from weakref import ref

import pytest
import torch
from diffusers.models.transformers.transformer_minimax_h3 import (
    MiniMaxH3AttnProcessor,
    _apply_rotary_emb,
)
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from flow_factory.models.minimax_h3._chunking import (
    _apply_h3_rotary_chunk,
    _apply_h3_rotary_chunks,
    _ChunkedFeedForward,
    _ChunkedH3AttnProcessor,
    _ChunkedLoraLinear,
    _ChunkedRMSNorm,
    install_h3_attention_norm_chunking,
    install_h3_feed_forward_chunking,
    install_h3_in_forward_block_checkpointing,
    install_h3_lora_projection_chunking,
    install_h3_rotary_chunking,
)


class SwiGLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(5, 28, bias=False)
        self.activation = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
        return hidden_states * self.activation(gate)


class FeedForwardFake(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.ModuleList(
            [
                SwiGLU(),
                nn.Dropout(0.0),
                nn.Linear(14, 5, bias=False),
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class BlockFake(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = nn.Module()
        self.attn.heads = 1
        self.attn.fused_projections = False
        self.attn.to_q = nn.Linear(5, 5, bias=False)
        self.attn.to_k = nn.Linear(5, 5, bias=False)
        self.attn.to_v = nn.Linear(5, 5, bias=False)
        self.attn.norm_q = nn.RMSNorm(5)
        self.attn.norm_k = nn.RMSNorm(5)
        self.attn.to_out = nn.ModuleList([nn.Linear(5, 5, bias=False), nn.Dropout(0.0)])
        self.attn.processor = MiniMaxH3AttnProcessor()
        self.ff = FeedForwardFake()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.ff(hidden_states)


class TransformerFake(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_refiner = nn.Module()
        self.token_refiner.refiner_blocks = nn.ModuleList([BlockFake(), BlockFake()])
        self.transformer_blocks = nn.ModuleList([BlockFake(), BlockFake(), BlockFake()])


def _feed_forwards(transformer: TransformerFake) -> list[nn.Module]:
    return [
        *[block.ff for block in transformer.token_refiner.refiner_blocks],
        *[block.ff for block in transformer.transformer_blocks],
    ]


def _lora_transformer(*, lora_dropout: float = 0.0) -> nn.Module:
    model = get_peft_model(
        TransformerFake(),
        LoraConfig(
            r=2,
            lora_alpha=4,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            init_lora_weights="gaussian",
            lora_dropout=lora_dropout,
        ),
    )
    for module in model.modules():
        if hasattr(module, "lora_B") and "default" in module.lora_B:
            nn.init.normal_(module.lora_B["default"].weight)
    return model


def test_chunking_preserves_parameter_tree_and_is_idempotent() -> None:
    transformer = TransformerFake()
    state = deepcopy(transformer.state_dict())
    keys_before = tuple(state)
    parameter_ids_before = {
        name: id(parameter) for name, parameter in transformer.named_parameters()
    }

    assert install_h3_feed_forward_chunking(transformer, max_tokens=4) == 5
    assert install_h3_feed_forward_chunking(transformer, max_tokens=4) == 5

    assert all(isinstance(module, _ChunkedFeedForward) for module in _feed_forwards(transformer))
    assert tuple(transformer.state_dict()) == keys_before
    assert {
        name: id(parameter) for name, parameter in transformer.named_parameters()
    } == parameter_ids_before
    transformer.load_state_dict(state, strict=True)
    assert "transformer_blocks.0.ff.net.0.proj.weight" in transformer.state_dict()
    assert not any("inner" in name for name, _ in transformer.named_modules())


def test_in_forward_checkpointing_preserves_state_and_backward() -> None:
    torch.manual_seed(13)
    direct = TransformerFake().double()
    checkpointed = deepcopy(direct)
    state_keys = tuple(checkpointed.state_dict())
    parameter_ids = {name: id(parameter) for name, parameter in checkpointed.named_parameters()}

    assert install_h3_in_forward_block_checkpointing(checkpointed) == 5
    assert install_h3_in_forward_block_checkpointing(checkpointed) == 5
    assert tuple(checkpointed.state_dict()) == state_keys
    assert {
        name: id(parameter) for name, parameter in checkpointed.named_parameters()
    } == parameter_ids

    direct_input = torch.randn(2, 7, 5, dtype=torch.float64, requires_grad=True)
    checkpointed_input = direct_input.detach().clone().requires_grad_(True)
    direct_block = direct.transformer_blocks[0]
    checkpointed_block = checkpointed.transformer_blocks[0]
    direct_output = direct_block(direct_input)
    checkpointed_output = checkpointed_block(checkpointed_input)
    output_gradient = torch.randn_like(direct_output)
    direct_output.backward(output_gradient)
    checkpointed_output.backward(output_gradient)

    torch.testing.assert_close(checkpointed_output, direct_output)
    torch.testing.assert_close(checkpointed_input.grad, direct_input.grad)
    for direct_parameter, checkpointed_parameter in zip(
        direct_block.parameters(), checkpointed_block.parameters()
    ):
        torch.testing.assert_close(checkpointed_parameter.grad, direct_parameter.grad)


def test_in_forward_checkpoint_starts_after_module_pre_forward_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transformer = TransformerFake().double()
    install_h3_in_forward_block_checkpointing(transformer)
    block = transformer.transformer_blocks[0]
    checkpoint_dtypes = []
    save_on_cpu_calls = []

    class RecordingSaveOnCPU:
        def __init__(self, *, pin_memory: bool, device_type: str) -> None:
            save_on_cpu_calls.append((pin_memory, device_type))

        def __enter__(self) -> None:
            return None

        def __exit__(self, *exc_info) -> None:
            return None

    monkeypatch.setattr(
        "flow_factory.models.minimax_h3._chunking.save_on_cpu",
        RecordingSaveOnCPU,
    )
    monkeypatch.setattr(
        "flow_factory.models.minimax_h3._chunking.checkpoint",
        lambda function, owner, hidden_states, **kwargs: checkpoint_dtypes.append(
            hidden_states.dtype
        )
        or function(owner, hidden_states),
    )
    handle = block.register_forward_pre_hook(lambda _module, inputs: (inputs[0].to(torch.float64),))

    block(torch.randn(2, 7, 5, dtype=torch.float32, requires_grad=True))
    handle.remove()

    assert checkpoint_dtypes == [torch.float64]
    assert save_on_cpu_calls == [(False, "cpu")]


def test_chunking_handles_remainder_and_preserves_forward_backward() -> None:
    torch.manual_seed(17)
    direct = TransformerFake().double()
    chunked = deepcopy(direct)
    install_h3_feed_forward_chunking(chunked, max_tokens=4)
    direct_ff = direct.transformer_blocks[0].ff
    chunked_ff = chunked.transformer_blocks[0].ff
    chunk_sizes: list[int] = []
    handle = chunked_ff.net[0].register_forward_pre_hook(
        lambda _module, inputs: chunk_sizes.append(inputs[0].shape[1])
    )
    direct_input = torch.randn(2, 9, 5, dtype=torch.float64, requires_grad=True)
    chunked_input = direct_input.detach().clone().requires_grad_(True)
    output_gradient = torch.randn(2, 9, 5, dtype=torch.float64)

    direct_output = direct_ff(direct_input)
    chunked_output = chunked_ff(chunked_input)
    direct_output.backward(output_gradient)
    chunked_output.backward(output_gradient)
    handle.remove()

    assert chunk_sizes[:3] == [4, 4, 1]
    assert sorted(chunk_sizes[3:]) == [1, 4, 4]
    torch.testing.assert_close(chunked_output, direct_output)
    torch.testing.assert_close(chunked_input.grad, direct_input.grad)
    for direct_parameter, chunked_parameter in zip(direct_ff.parameters(), chunked_ff.parameters()):
        torch.testing.assert_close(chunked_parameter.grad, direct_parameter.grad)


def test_short_feed_forward_uses_one_execution() -> None:
    transformer = TransformerFake()
    install_h3_feed_forward_chunking(transformer, max_tokens=4)
    feed_forward = transformer.transformer_blocks[0].ff
    chunk_sizes: list[int] = []
    handle = feed_forward.net[0].register_forward_pre_hook(
        lambda _module, inputs: chunk_sizes.append(inputs[0].shape[1])
    )

    feed_forward(torch.randn(2, 4, 5))
    handle.remove()

    assert chunk_sizes == [4]


def test_chunking_preserves_nested_checkpoint_backward() -> None:
    torch.manual_seed(19)
    direct = TransformerFake().double()
    chunked = deepcopy(direct)
    install_h3_feed_forward_chunking(chunked, max_tokens=4)
    direct_ff = direct.transformer_blocks[0].ff
    chunked_ff = chunked.transformer_blocks[0].ff
    direct_input = torch.randn(2, 9, 5, dtype=torch.float64, requires_grad=True)
    chunked_input = direct_input.detach().clone().requires_grad_(True)
    output_gradient = torch.randn(2, 9, 5, dtype=torch.float64)

    direct_output = direct_ff(direct_input)
    chunked_output = checkpoint(chunked_ff, chunked_input, use_reentrant=False)
    direct_output.backward(output_gradient)
    chunked_output.backward(output_gradient)

    torch.testing.assert_close(chunked_output, direct_output)
    torch.testing.assert_close(chunked_input.grad, direct_input.grad)
    for direct_parameter, chunked_parameter in zip(direct_ff.parameters(), chunked_ff.parameters()):
        torch.testing.assert_close(chunked_parameter.grad, direct_parameter.grad)


def test_chunking_checkpoints_only_long_grad_enabled_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transformer = TransformerFake()
    install_h3_feed_forward_chunking(transformer, max_tokens=4)
    feed_forward = transformer.transformer_blocks[0].ff
    calls: list[tuple[int, bool, bool]] = []

    def recording_checkpoint(function, chunk, *, use_reentrant, preserve_rng_state):
        calls.append((chunk.shape[1], use_reentrant, preserve_rng_state))
        return function(chunk)

    monkeypatch.setattr(
        "flow_factory.models.minimax_h3._chunking.checkpoint",
        recording_checkpoint,
    )

    feed_forward(torch.randn(2, 9, 5, requires_grad=True))
    with torch.no_grad():
        feed_forward(torch.randn(2, 9, 5))
    feed_forward(torch.randn(2, 4, 5, requires_grad=True))

    assert calls == [(4, False, True), (4, False, True), (1, False, True)]


def test_chunking_supports_frozen_input_with_trainable_parameters() -> None:
    direct = TransformerFake().double()
    chunked = deepcopy(direct)
    install_h3_feed_forward_chunking(chunked, max_tokens=4)
    direct_ff = direct.transformer_blocks[0].ff
    chunked_ff = chunked.transformer_blocks[0].ff
    for parameter in direct_ff.parameters():
        parameter.requires_grad_(False)
    for parameter in chunked_ff.parameters():
        parameter.requires_grad_(False)
    direct_ff.net[2].weight.requires_grad_(True)
    chunked_ff.net[2].weight.requires_grad_(True)
    hidden_states = torch.randn(2, 9, 5, dtype=torch.float64)

    direct_output = direct_ff(hidden_states)
    chunked_output = chunked_ff(hidden_states)
    direct_output.sum().backward()
    chunked_output.sum().backward()

    torch.testing.assert_close(chunked_output, direct_output)
    torch.testing.assert_close(chunked_ff.net[2].weight.grad, direct_ff.net[2].weight.grad)


def test_nested_chunk_checkpoint_preserves_cpu_autocast_dtype() -> None:
    transformer = TransformerFake()
    install_h3_feed_forward_chunking(transformer, max_tokens=4)
    feed_forward = transformer.transformer_blocks[0].ff
    hidden_states = torch.randn(2, 9, 5, requires_grad=True)
    projected_dtypes: list[torch.dtype] = []
    handle = feed_forward.net[2].register_forward_pre_hook(
        lambda _module, inputs: projected_dtypes.append(inputs[0].dtype)
    )

    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = checkpoint(feed_forward, hidden_states, use_reentrant=False)
    output.float().sum().backward()
    handle.remove()

    assert output.dtype == torch.bfloat16
    assert len(projected_dtypes) >= 6
    assert set(projected_dtypes) == {torch.bfloat16}


@pytest.mark.parametrize("max_tokens", [0, -1, True, 1.5])
def test_chunking_rejects_invalid_max_tokens(max_tokens: object) -> None:
    with pytest.raises(ValueError, match="max_tokens expected a positive int"):
        install_h3_feed_forward_chunking(TransformerFake(), max_tokens=max_tokens)


def test_chunking_rejects_conflicting_reinstallation() -> None:
    transformer = TransformerFake()
    install_h3_feed_forward_chunking(transformer, max_tokens=4)

    with pytest.raises(ValueError, match="already uses max_tokens=4.*conflicting 8"):
        install_h3_feed_forward_chunking(transformer, max_tokens=8)


def test_attention_norm_chunking_preserves_parameter_tree_and_is_idempotent() -> None:
    transformer = TransformerFake()
    state = deepcopy(transformer.state_dict())
    keys_before = tuple(state)
    parameter_ids_before = {
        name: id(parameter) for name, parameter in transformer.named_parameters()
    }

    assert install_h3_attention_norm_chunking(transformer, max_tokens=4) == 10
    assert install_h3_attention_norm_chunking(transformer, max_tokens=4) == 10

    norms = [
        norm
        for block in (
            *transformer.token_refiner.refiner_blocks,
            *transformer.transformer_blocks,
        )
        for norm in (block.attn.norm_q, block.attn.norm_k)
    ]
    assert all(isinstance(norm, _ChunkedRMSNorm) for norm in norms)
    assert tuple(transformer.state_dict()) == keys_before
    assert {
        name: id(parameter) for name, parameter in transformer.named_parameters()
    } == parameter_ids_before
    transformer.load_state_dict(state, strict=True)
    assert "transformer_blocks.0.attn.norm_q.weight" in transformer.state_dict()
    assert not any("inner" in name for name, _ in transformer.named_modules())


def test_attention_norm_chunking_preserves_remainder_forward_backward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(23)
    direct = TransformerFake().double()
    chunked = deepcopy(direct)
    install_h3_attention_norm_chunking(chunked, max_tokens=4)
    direct_norm = direct.transformer_blocks[0].attn.norm_q
    chunked_norm = chunked.transformer_blocks[0].attn.norm_q
    direct_input = torch.randn(2, 9, 3, 5, dtype=torch.float64, requires_grad=True)
    chunked_input = direct_input.detach().clone().requires_grad_(True)
    output_gradient = torch.randn(2, 9, 3, 5, dtype=torch.float64)

    direct_output = direct_norm(direct_input)
    direct_output.backward(output_gradient)
    chunk_sizes: list[int] = []
    direct_rms_norm = F.rms_norm

    def recording_rms_norm(hidden_states, normalized_shape, weight=None, eps=None):
        chunk_sizes.append(hidden_states.shape[1])
        return direct_rms_norm(hidden_states, normalized_shape, weight, eps)

    monkeypatch.setattr(F, "rms_norm", recording_rms_norm)
    chunked_output = chunked_norm(chunked_input)
    chunked_output.backward(output_gradient)

    assert chunk_sizes == [4, 4, 1]
    torch.testing.assert_close(chunked_output, direct_output)
    torch.testing.assert_close(chunked_input.grad, direct_input.grad)
    torch.testing.assert_close(chunked_norm.weight.grad, direct_norm.weight.grad)


def test_attention_norm_chunking_preserves_bfloat16_dtype() -> None:
    transformer = TransformerFake().bfloat16()
    direct = transformer.transformer_blocks[0].attn.norm_q(torch.randn(1, 7, 3, 5).bfloat16())
    install_h3_attention_norm_chunking(transformer, max_tokens=4)

    chunked = transformer.transformer_blocks[0].attn.norm_q(torch.randn(1, 7, 3, 5).bfloat16())

    assert chunked.dtype == direct.dtype == torch.bfloat16


def test_attention_norm_chunking_rejects_conflicting_reinstallation() -> None:
    transformer = TransformerFake()
    install_h3_attention_norm_chunking(transformer, max_tokens=4)

    with pytest.raises(ValueError, match="already uses max_tokens=4.*conflicting 8"):
        install_h3_attention_norm_chunking(transformer, max_tokens=8)


def test_chunked_token_local_operations_avoid_full_output_concatenation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transformer = TransformerFake()
    install_h3_feed_forward_chunking(transformer, max_tokens=4)
    install_h3_attention_norm_chunking(transformer, max_tokens=4)

    monkeypatch.setattr(
        torch,
        "cat",
        lambda *args, **kwargs: pytest.fail("chunk aggregation must not call torch.cat"),
    )

    hidden_states = torch.randn(2, 9, 5, requires_grad=True)
    feed_forward = transformer.transformer_blocks[0].ff(hidden_states)
    normalized = transformer.transformer_blocks[0].attn.norm_q(hidden_states)
    (feed_forward + normalized).sum().backward()


def test_lora_projection_chunking_preserves_peft_tree_and_is_idempotent() -> None:
    model = _lora_transformer()
    transformer = model.get_base_model()
    state = deepcopy(model.state_dict())
    keys_before = tuple(state)
    parameter_ids_before = {name: id(parameter) for name, parameter in model.named_parameters()}

    assert install_h3_lora_projection_chunking(transformer, max_tokens=4) == 20
    assert install_h3_lora_projection_chunking(transformer, max_tokens=4) == 20

    projections = [
        projection
        for block in (
            *transformer.token_refiner.refiner_blocks,
            *transformer.transformer_blocks,
        )
        for projection in (
            block.attn.to_q,
            block.attn.to_k,
            block.attn.to_v,
            block.attn.to_out[0],
        )
    ]
    assert all(isinstance(projection, _ChunkedLoraLinear) for projection in projections)
    assert tuple(model.state_dict()) == keys_before
    assert {
        name: id(parameter) for name, parameter in model.named_parameters()
    } == parameter_ids_before
    model.load_state_dict(state, strict=True)


def test_lora_projection_chunking_can_leave_the_output_projection_whole() -> None:
    """Under sequence parallelism ``to_out.0`` reduce-scatters along the tokens.

    A chunk would be scattered on its own and the pieces would interleave, so the
    output projection runs whole while Q/K/V keep their bounded chunks.
    """
    transformer = _lora_transformer().get_base_model()

    assert (
        install_h3_lora_projection_chunking(
            transformer, max_tokens=4, chunk_output_projection=False
        )
        == 15
    )

    for block in (*transformer.token_refiner.refiner_blocks, *transformer.transformer_blocks):
        assert all(
            isinstance(projection, _ChunkedLoraLinear)
            for projection in (block.attn.to_q, block.attn.to_k, block.attn.to_v)
        )
        assert not isinstance(block.attn.to_out[0], _ChunkedLoraLinear)


def test_lora_projection_chunking_preserves_remainder_forward_backward() -> None:
    torch.manual_seed(29)
    direct_model = _lora_transformer().double()
    chunked_model = deepcopy(direct_model)
    chunked_transformer = chunked_model.get_base_model()
    install_h3_lora_projection_chunking(chunked_transformer, max_tokens=4)
    direct = direct_model.get_base_model().transformer_blocks[0].attn.to_k
    chunked = chunked_transformer.transformer_blocks[0].attn.to_k
    chunk_sizes: list[int] = []
    handle = chunked.base_layer.register_forward_pre_hook(
        lambda _module, inputs: chunk_sizes.append(inputs[0].shape[1])
    )
    direct_input = torch.randn(2, 9, 5, dtype=torch.float64, requires_grad=True)
    chunked_input = direct_input.detach().clone().requires_grad_(True)
    output_gradient = torch.randn(2, 9, 5, dtype=torch.float64)

    direct_output = direct(direct_input)
    chunked_output = chunked(chunked_input)
    direct_output.backward(output_gradient)
    chunked_output.backward(output_gradient)
    handle.remove()

    assert chunk_sizes == [4, 4, 1]
    torch.testing.assert_close(chunked_output, direct_output)
    torch.testing.assert_close(chunked_input.grad, direct_input.grad)
    for direct_parameter, chunked_parameter in zip(direct.parameters(), chunked.parameters()):
        torch.testing.assert_close(chunked_parameter.grad, direct_parameter.grad)


def test_lora_projection_chunking_preserves_mixed_batch_adapter_names() -> None:
    direct_model = _lora_transformer()
    chunked_model = deepcopy(direct_model)
    install_h3_lora_projection_chunking(chunked_model.get_base_model(), max_tokens=4)
    direct = direct_model.get_base_model().transformer_blocks[0].attn.to_v
    chunked = chunked_model.get_base_model().transformer_blocks[0].attn.to_v
    hidden_states = torch.randn(2, 9, 5)
    adapter_names = ["default", "__base__"]

    direct_output = direct(hidden_states, adapter_names=adapter_names)
    chunked_output = chunked(hidden_states, adapter_names=adapter_names)

    torch.testing.assert_close(chunked_output, direct_output)


def test_lora_projection_chunking_rejects_conflicting_reinstallation() -> None:
    transformer = _lora_transformer().get_base_model()
    install_h3_lora_projection_chunking(transformer, max_tokens=4)

    with pytest.raises(ValueError, match="already uses max_tokens=4.*conflicting 8"):
        install_h3_lora_projection_chunking(transformer, max_tokens=8)


def test_lora_projection_chunking_rejects_nonzero_dropout() -> None:
    transformer = _lora_transformer(lora_dropout=0.1).get_base_model()

    with pytest.raises(TypeError, match="requires zero dropout"):
        install_h3_lora_projection_chunking(transformer, max_tokens=4)


def test_lora_projection_chunking_rejects_existing_execution_hooks() -> None:
    transformer = _lora_transformer().get_base_model()
    transformer.transformer_blocks[0].attn.to_q.register_forward_pre_hook(
        lambda _module, _inputs: None
    )

    with pytest.raises(TypeError, match="must be configured before execution hooks"):
        install_h3_lora_projection_chunking(transformer, max_tokens=4)


def test_lora_projection_chunking_preserves_non_reentrant_checkpoint_backward() -> None:
    direct_model = _lora_transformer().double()
    chunked_model = deepcopy(direct_model)
    install_h3_lora_projection_chunking(chunked_model.get_base_model(), max_tokens=4)
    direct = direct_model.get_base_model().transformer_blocks[0].attn.to_q
    chunked = chunked_model.get_base_model().transformer_blocks[0].attn.to_q
    direct_input = torch.randn(2, 9, 5, dtype=torch.float64, requires_grad=True)
    chunked_input = direct_input.detach().clone().requires_grad_(True)

    direct_output = direct(direct_input)
    chunked_output = checkpoint(chunked, chunked_input, use_reentrant=False)
    direct_output.sum().backward()
    chunked_output.sum().backward()

    torch.testing.assert_close(chunked_output, direct_output)
    torch.testing.assert_close(chunked_input.grad, direct_input.grad)
    for direct_parameter, chunked_parameter in zip(direct.parameters(), chunked.parameters()):
        torch.testing.assert_close(chunked_parameter.grad, direct_parameter.grad)


def test_rotary_chunking_preserves_remainder_forward_backward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(31)
    direct_input = torch.randn(2, 9, 3, 8, dtype=torch.float64, requires_grad=True)
    chunked_input = direct_input.detach().clone().requires_grad_(True)
    direct_cos = torch.randn(9, 6, dtype=torch.float64, requires_grad=True)
    chunked_cos = direct_cos.detach().clone().requires_grad_(True)
    direct_sin = torch.randn(9, 6, dtype=torch.float64, requires_grad=True)
    chunked_sin = direct_sin.detach().clone().requires_grad_(True)
    output_gradient = torch.randn(2, 9, 3, 8, dtype=torch.float64)
    chunk_sizes: list[int] = []

    def recording_rotary_chunk(hidden_states, chunk_cos, chunk_sin):
        chunk_sizes.append(hidden_states.shape[1])
        return _apply_h3_rotary_chunk(hidden_states, chunk_cos, chunk_sin)

    monkeypatch.setattr(
        "flow_factory.models.minimax_h3._chunking._apply_h3_rotary_chunk",
        recording_rotary_chunk,
    )

    direct_output = _apply_rotary_emb(direct_input, direct_cos, direct_sin)
    chunked_output = _apply_h3_rotary_chunks(
        chunked_input,
        chunked_cos,
        chunked_sin,
        max_tokens=4,
    )
    direct_output.backward(output_gradient)
    chunked_output.backward(output_gradient)

    assert chunk_sizes == [4, 4, 1]
    torch.testing.assert_close(chunked_output, direct_output)
    torch.testing.assert_close(chunked_input.grad, direct_input.grad)
    torch.testing.assert_close(chunked_cos.grad, direct_cos.grad)
    torch.testing.assert_close(chunked_sin.grad, direct_sin.grad)


def test_rotary_chunking_preserves_processor_and_parameter_tree() -> None:
    transformer = TransformerFake()
    attention_backend = object()
    parallel_config = object()
    first_processor = transformer.token_refiner.refiner_blocks[0].attn.processor
    first_processor._attention_backend = attention_backend
    first_processor._parallel_config = parallel_config
    state = deepcopy(transformer.state_dict())
    keys_before = tuple(state)
    parameter_ids_before = {
        name: id(parameter) for name, parameter in transformer.named_parameters()
    }
    processor_ids_before = [
        id(block.attn.processor)
        for block in (
            *transformer.token_refiner.refiner_blocks,
            *transformer.transformer_blocks,
        )
    ]

    assert install_h3_rotary_chunking(transformer, max_tokens=4) == 5
    assert install_h3_rotary_chunking(transformer, max_tokens=4) == 5

    processors = [
        block.attn.processor
        for block in (
            *transformer.token_refiner.refiner_blocks,
            *transformer.transformer_blocks,
        )
    ]
    assert all(isinstance(processor, _ChunkedH3AttnProcessor) for processor in processors)
    assert [id(processor) for processor in processors] == processor_ids_before
    assert processors[0]._attention_backend is attention_backend
    assert processors[0]._parallel_config is parallel_config
    assert tuple(transformer.state_dict()) == keys_before
    assert {
        name: id(parameter) for name, parameter in transformer.named_parameters()
    } == parameter_ids_before
    transformer.load_state_dict(state, strict=True)


def test_rotary_chunking_processor_preserves_attention_forward_backward() -> None:
    torch.manual_seed(37)
    direct_transformer = TransformerFake().double()
    chunked_transformer = deepcopy(direct_transformer)
    install_h3_rotary_chunking(chunked_transformer, max_tokens=4)
    direct = direct_transformer.transformer_blocks[0].attn
    chunked = chunked_transformer.transformer_blocks[0].attn
    direct_input = torch.randn(2, 9, 5, dtype=torch.float64, requires_grad=True)
    chunked_input = direct_input.detach().clone().requires_grad_(True)
    cos = torch.randn(9, 4, dtype=torch.float64)
    sin = torch.randn(9, 4, dtype=torch.float64)
    output_gradient = torch.randn(2, 9, 5, dtype=torch.float64)

    direct_output = direct.processor(direct, direct_input, (cos, sin))
    chunked_output = checkpoint(
        lambda value: chunked.processor(chunked, value, (cos, sin)),
        chunked_input,
        use_reentrant=False,
    )
    direct_output.backward(output_gradient)
    chunked_output.backward(output_gradient)

    torch.testing.assert_close(chunked_output, direct_output)
    torch.testing.assert_close(chunked_input.grad, direct_input.grad)
    for direct_parameter, chunked_parameter in zip(direct.parameters(), chunked.parameters()):
        torch.testing.assert_close(chunked_parameter.grad, direct_parameter.grad)


def test_rotary_chunking_rejects_conflicting_reinstallation() -> None:
    transformer = TransformerFake()
    install_h3_rotary_chunking(transformer, max_tokens=4)

    with pytest.raises(ValueError, match="already uses max_tokens=4.*conflicting 8"):
        install_h3_rotary_chunking(transformer, max_tokens=8)


def test_chunked_attention_releases_qkv_before_output_projection() -> None:
    transformer = TransformerFake()
    install_h3_rotary_chunking(transformer, max_tokens=4)
    attention = transformer.transformer_blocks[0].attn
    qkv_refs = []

    def recording_dispatch(query, key, value, **kwargs):
        del kwargs
        qkv_refs.extend((ref(query), ref(key), ref(value)))
        return torch.zeros_like(query)

    attention.processor.flow_factory_dispatch_attention_fn = recording_dispatch

    def require_released_qkv(_module, _inputs):
        assert len(qkv_refs) == 3
        assert all(tensor_ref() is None for tensor_ref in qkv_refs)

    handle = attention.to_out[0].register_forward_pre_hook(require_released_qkv)
    attention.processor(attention, torch.randn(2, 9, 5))
    handle.remove()
