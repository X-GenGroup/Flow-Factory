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

"""Plan construction for tensor parallelism, checked without a process group.

The properties that need real collectives live in
``tests/distributed/test_tensor_parallel_sharding.py``. What is checked here is
the plan the applier is handed, because every hazard the module guards against
is first visible in that mapping: a pattern that matches nothing, a parameter
left unnamed, or a LoRA wrapper targeted instead of its base layer.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

from flow_factory.models.tensor_parallel import (
    ReplicateParallel,
    ShardedInputReplicatedWeight,
    build_total_plan,
    localize_attention_head_counts,
)


class _Attention(nn.Module):
    def __init__(self, dim: int = 16, heads: int = 4) -> None:
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)
        self.norm_q = nn.RMSNorm(dim // heads)


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = _Attention()
        self.norm = nn.LayerNorm(16)


class _Model(nn.Module):
    def __init__(self, blocks: int = 2) -> None:
        super().__init__()
        self.transformer_blocks = nn.ModuleList(_Block() for _ in range(blocks))
        self.proj_out = nn.Linear(16, 16)


SHARD_PLAN = {
    "transformer_blocks.*.attn.to_q": ColwiseParallel(),
    "transformer_blocks.*.attn.to_out": RowwiseParallel(),
    "transformer_blocks.*.attn.norm_q": ShardedInputReplicatedWeight(),
}


def test_the_completed_plan_names_every_parameter_owning_module() -> None:
    """An unnamed parameter is what makes gradient clipping fail later.

    Without FSDP it stays a plain tensor and `aten._foreach_norm` rejects the
    mix; with FSDP it becomes a dp-only DTensor and `aten.stack` rejects the
    differing mesh. Neither is silent, but both surface far from the cause.
    """
    model = _Model()
    plan = build_total_plan(model, SHARD_PLAN)

    owners = {
        name
        for name, module in model.named_modules()
        if name and any(True for _ in module.named_parameters(recurse=False))
    }

    assert owners - set(plan) == set()


def test_modules_outside_the_shard_plan_are_completed_as_replicated() -> None:
    """The boundary projections stay replicated, but still enter the plan."""
    plan = build_total_plan(_Model(), SHARD_PLAN)

    assert isinstance(plan["proj_out"], ReplicateParallel)
    assert isinstance(plan["transformer_blocks.0.norm"], ReplicateParallel)


def test_the_shard_plan_survives_completion_unchanged() -> None:
    """Completion must not override a style the architecture asked for."""
    plan = build_total_plan(_Model(), SHARD_PLAN)

    assert isinstance(plan["transformer_blocks.0.attn.to_q"], ColwiseParallel)
    assert isinstance(plan["transformer_blocks.1.attn.to_out"], RowwiseParallel)
    # Per-head qk-norm must keep the sharded-input style: a plainly replicated
    # one leaves each rank with a partial-sum weight gradient.
    assert isinstance(plan["transformer_blocks.0.attn.norm_q"], ShardedInputReplicatedWeight)


def test_a_wildcard_expands_over_every_repeated_block() -> None:
    """One `*` stands for exactly one path segment."""
    plan = build_total_plan(_Model(blocks=3), SHARD_PLAN)

    assert {
        "transformer_blocks.0.attn.to_q",
        "transformer_blocks.1.attn.to_q",
        "transformer_blocks.2.attn.to_q",
    } <= set(plan)


def test_a_pattern_matching_nothing_is_rejected() -> None:
    """A renamed layer must fail loudly, not silently leave the plan short."""
    with pytest.raises(ValueError, match="matched no module"):
        build_total_plan(_Model(), {"transformer_blocks.*.attn.to_qkv": ColwiseParallel()})


def test_the_plan_reaches_through_lora_wrappers_to_the_base_layer() -> None:
    """`ColwiseParallel` takes only Linear/Embedding, never PEFT's wrapper.

    The adapter matrices get styles of their own, chosen so the LoRA branch
    lands in the same placement as the base branch it is added to.
    """
    peft = pytest.importorskip("peft")
    model = peft.get_peft_model(
        _Model(),
        peft.LoraConfig(r=4, target_modules=["to_q", "to_out"]),
    )
    inner = model.base_model.model
    plan = build_total_plan(inner, SHARD_PLAN)

    # The wrapper itself owns no parameters and must not be targeted.
    assert "transformer_blocks.0.attn.to_q" not in plan
    assert isinstance(plan["transformer_blocks.0.attn.to_q.base_layer"], ColwiseParallel)

    # colwise: lora_A replicated (full input, small rank), lora_B sharded like the base.
    assert isinstance(plan["transformer_blocks.0.attn.to_q.lora_A.default"], ReplicateParallel)
    assert isinstance(plan["transformer_blocks.0.attn.to_q.lora_B.default"], ColwiseParallel)

    # rowwise: the activation arrives sharded, so lora_A must be sharded too.
    assert isinstance(plan["transformer_blocks.0.attn.to_out.base_layer"], RowwiseParallel)
    assert isinstance(plan["transformer_blocks.0.attn.to_out.lora_A.default"], RowwiseParallel)
    assert isinstance(plan["transformer_blocks.0.attn.to_out.lora_B.default"], ReplicateParallel)


def test_head_counts_become_this_rank_s_share() -> None:
    """A processor reshaping with `attn.heads` must read the local count."""
    model = _Model()
    assert model.transformer_blocks[0].attn.heads == 4

    adjusted = localize_attention_head_counts(model, tp_size=2)

    assert adjusted == 2
    assert all(block.attn.heads == 2 for block in model.transformer_blocks)


def test_an_indivisible_head_count_is_rejected() -> None:
    """Heads must divide the degree, or a rank would hold a partial head."""
    model = _Model()
    model.transformer_blocks[0].attn.heads = 3

    with pytest.raises(ValueError, match="divisible by the tensor-parallel degree"):
        localize_attention_head_counts(model, tp_size=2)


def test_the_qwen_image_21_plan_covers_the_published_projections() -> None:
    """The adapter plan must match what upstream diffusers declares for the class.

    Mirroring it keeps one source of truth for which projections shard; the two
    qk-norm entries are Flow-Factory's addition, which upstream omits because
    inference never takes their gradient.
    """
    from flow_factory.models.qwen_image_21.qwen_image_21 import QwenImage21Adapter

    plan = QwenImage21Adapter.tensor_parallel_plan()

    colwise = {k for k, v in plan.items() if isinstance(v, ColwiseParallel)}
    rowwise = {k for k, v in plan.items() if isinstance(v, RowwiseParallel)}
    per_head = {k for k, v in plan.items() if isinstance(v, ShardedInputReplicatedWeight)}

    assert colwise == {
        "transformer_blocks.*.attn.to_q",
        "transformer_blocks.*.attn.to_k",
        "transformer_blocks.*.attn.to_v",
        "transformer_blocks.*.img_mlp.proj",
        "transformer_blocks.*.img_mlp.gate_layer",
    }
    assert rowwise == {
        "transformer_blocks.*.attn.to_out.0",
        "transformer_blocks.*.img_mlp.out",
    }
    assert per_head == {
        "transformer_blocks.*.attn.norm_q",
        "transformer_blocks.*.attn.norm_k",
    }


def test_the_base_adapter_declares_no_plan_by_default() -> None:
    """An adapter without a plan must not silently train unsharded.

    The loader turns the empty default into a rejection when `tp_size > 1`.
    """
    from flow_factory.models.abc import BaseAdapter

    assert BaseAdapter.tensor_parallel_plan() == {}
