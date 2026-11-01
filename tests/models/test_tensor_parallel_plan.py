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


def test_an_unsharded_attention_keeps_its_head_count() -> None:
    """Localization is driven by the shards, so it is a no-op without them.

    The head count is rewritten only where a dim-0 shard actually exists, which
    is what lets a feed-forward-only plan leave replicated attention alone. The
    cases that do shard need real process groups and live in
    ``tests/distributed/test_tensor_parallel_sharding.py``.
    """
    model = _Model()

    adjusted = localize_attention_head_counts(model, tp_size=2)

    assert adjusted == 0
    assert all(block.attn.heads == 4 for block in model.transformer_blocks)


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


def test_the_sd3_5_plan_shards_both_mmdit_streams_and_the_dual_attention() -> None:
    """Both halves of a joint attention must shard on the same head split.

    ``JointTransformerBlock`` holds one ``Attention`` for both streams and
    concatenates their per-head projections along the sequence axis before a
    single softmax, so naming only ``to_q``/``to_k``/``to_v`` would leave the
    text stream with every head and make the concatenation impossible. ``attn2``
    is the second, image-only attention SD3.5 adds on its dual-attention layers.
    """
    from flow_factory.models.stable_diffusion.sd3_5 import SD3_5Adapter

    plan = SD3_5Adapter.tensor_parallel_plan()

    colwise = {k for k, v in plan.items() if isinstance(v, ColwiseParallel)}
    rowwise = {k for k, v in plan.items() if isinstance(v, RowwiseParallel)}
    per_head = {k for k, v in plan.items() if isinstance(v, ShardedInputReplicatedWeight)}

    assert colwise == {
        "transformer_blocks.*.attn.to_q",
        "transformer_blocks.*.attn.to_k",
        "transformer_blocks.*.attn.to_v",
        "transformer_blocks.*.attn.add_q_proj",
        "transformer_blocks.*.attn.add_k_proj",
        "transformer_blocks.*.attn.add_v_proj",
        "transformer_blocks.*.attn2.to_q",
        "transformer_blocks.*.attn2.to_k",
        "transformer_blocks.*.attn2.to_v",
        "transformer_blocks.*.ff.net.0.proj",
        "transformer_blocks.*.ff_context.net.0.proj",
    }
    assert rowwise == {
        "transformer_blocks.*.attn.to_out.0",
        "transformer_blocks.*.attn.to_add_out",
        "transformer_blocks.*.attn2.to_out.0",
        "transformer_blocks.*.ff.net.2",
        "transformer_blocks.*.ff_context.net.2",
    }
    assert per_head == {
        "transformer_blocks.*.attn.norm_q",
        "transformer_blocks.*.attn.norm_k",
        "transformer_blocks.*.attn.norm_added_q",
        "transformer_blocks.*.attn.norm_added_k",
        "transformer_blocks.*.attn2.norm_q",
        "transformer_blocks.*.attn2.norm_k",
    }


def test_the_sd3_5_qk_norms_declare_the_head_axis_its_processor_uses() -> None:
    """SD3's processor transposes before normalizing, so heads sit on axis 1.

    The style's default axis is 2, which is where the Qwen processors leave the
    head dimension. Declaring the wrong axis describes a global shape the tensor
    does not have.
    """
    from flow_factory.models.stable_diffusion.sd3_5 import SD3_5Adapter

    per_head = [
        style
        for style in SD3_5Adapter.tensor_parallel_plan().values()
        if isinstance(style, ShardedInputReplicatedWeight)
    ]

    assert per_head and all(style.head_axis == 1 for style in per_head)


def test_the_qwen_image_plan_covers_the_published_projections() -> None:
    """Mirror upstream's ``_tp_plan`` and add only the qk-norms it omits.

    Upstream leaves ``norm_q``/``norm_k``/``norm_added_q``/``norm_added_k`` out
    because inference never takes their gradient; training does, and a
    replicated style would leave each rank with a partial sum.
    """
    from diffusers.models.transformers.transformer_qwenimage import (
        QwenImageTransformer2DModel,
    )
    from flow_factory.models.qwen_image.qwen_image import QwenImageAdapter

    plan = QwenImageAdapter.tensor_parallel_plan()

    upstream = QwenImageTransformer2DModel._tp_plan
    assert {k for k, v in plan.items() if isinstance(v, ColwiseParallel)} == {
        k for k, v in upstream.items() if v == "colwise"
    }
    assert {k for k, v in plan.items() if isinstance(v, RowwiseParallel)} == {
        k for k, v in upstream.items() if v == "rowwise"
    }
    assert {k for k, v in plan.items() if isinstance(v, ShardedInputReplicatedWeight)} == {
        "transformer_blocks.*.attn.norm_q",
        "transformer_blocks.*.attn.norm_k",
        "transformer_blocks.*.attn.norm_added_q",
        "transformer_blocks.*.attn.norm_added_k",
    }
    # The Qwen processors reshape without a transpose, so heads land on axis 2,
    # which is the style's default.
    assert all(
        style.head_axis == 2
        for style in plan.values()
        if isinstance(style, ShardedInputReplicatedWeight)
    )


def test_qwen_image_edit_plus_reuses_the_qwen_image_plan() -> None:
    """One transformer class must not acquire two plans that can drift apart.

    Edit-Plus is a different pipeline over the same
    ``QwenImageTransformer2DModel``; it derives from ``BaseAdapter``
    independently, so the reuse is an explicit delegation rather than
    inheritance and is worth pinning.
    """
    from flow_factory.models.qwen_image.qwen_image import QwenImageAdapter
    from flow_factory.models.qwen_image.qwen_image_edit_plus import QwenImageEditPlusAdapter

    plan = QwenImageEditPlusAdapter.tensor_parallel_plan()
    reference = QwenImageAdapter.tensor_parallel_plan()

    assert plan.keys() == reference.keys()
    assert all(type(plan[key]) is type(reference[key]) for key in reference)


@pytest.mark.parametrize("architecture", ["sd3", "qwen-image"])
def test_the_new_plans_resolve_against_the_real_module_trees(architecture: str) -> None:
    """Every pattern must name a module the diffusers class actually has.

    Built on the meta device at two layers, so this stays a CPU test. A renamed
    projection -- ``net.2`` becoming ``net.2.proj``, say -- makes
    ``build_total_plan`` raise rather than silently shard less than intended,
    and this is where that is caught without a GPU.
    """
    if architecture == "sd3":
        from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel
        from flow_factory.models.stable_diffusion.sd3_5 import SD3_5Adapter

        with torch.device("meta"):
            model = SD3Transformer2DModel(
                num_layers=2,
                attention_head_dim=64,
                num_attention_heads=24,
                caption_projection_dim=1536,
                # Slot 0 dual-attention, slot 1 the context_pre_only tail, so
                # both optional block shapes are present.
                dual_attention_layers=(0,),
                qk_norm="rms_norm",
            )
        shard_plan = SD3_5Adapter.tensor_parallel_plan()
    else:
        from diffusers.models.transformers.transformer_qwenimage import (
            QwenImageTransformer2DModel,
        )
        from flow_factory.models.qwen_image.qwen_image import QwenImageAdapter

        with torch.device("meta"):
            model = QwenImageTransformer2DModel(num_layers=2)
        shard_plan = QwenImageAdapter.tensor_parallel_plan()

    plan = build_total_plan(model, shard_plan)

    owners = {
        name
        for name, module in model.named_modules()
        if name and any(True for _ in module.named_parameters(recurse=False))
    }
    assert owners - set(plan) == set()
    # Block 0 is the richest shape in both architectures (SD3.5 puts dual
    # attention there and the context_pre_only tail last), so every sharding
    # pattern must land on it. `_expand_globs` already rejects a pattern that
    # matches nothing anywhere; this additionally rejects one that only matches
    # the blocks it was not written for.
    wanted = {
        pattern.replace("*", "0")
        for pattern, style in shard_plan.items()
        if isinstance(style, (ColwiseParallel, RowwiseParallel))
    }
    assert wanted <= set(plan)


def test_the_base_adapter_declares_no_plan_by_default() -> None:
    """An adapter without a plan must not silently train unsharded.

    The loader turns the empty default into a rejection when `tp_size > 1`.
    """
    from flow_factory.models.abc import BaseAdapter

    assert BaseAdapter.tensor_parallel_plan() == {}


def _styles(plan: dict) -> tuple[set, set, set]:
    colwise = {k for k, v in plan.items() if isinstance(v, ColwiseParallel)}
    rowwise = {k for k, v in plan.items() if isinstance(v, RowwiseParallel)}
    per_head = {k for k, v in plan.items() if isinstance(v, ShardedInputReplicatedWeight)}
    assert set(plan) == colwise | rowwise | per_head
    return colwise, rowwise, per_head


def test_the_flux1_plan_mirrors_upstream_for_the_dual_stream_blocks() -> None:
    """The dual-stream entries must match what diffusers declares for the class.

    FLUX.1 fuses nothing in these blocks -- its feed-forward is
    `gelu-approximate`, not GEGLU -- so every projection upstream marks
    colwise/rowwise is reachable with the plain styles.
    """
    from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
    from flow_factory.models.flux.flux1 import Flux1Adapter

    colwise, rowwise, per_head = _styles(Flux1Adapter.tensor_parallel_plan())
    upstream = FluxTransformer2DModel._tp_plan

    assert colwise == {
        key
        for key, value in upstream.items()
        if value == "colwise" and key.startswith("transformer_blocks.")
    }
    assert rowwise == {
        key
        for key, value in upstream.items()
        if value == "rowwise" and key.startswith("transformer_blocks.")
    }
    # Upstream omits the qk-norms because inference never takes their gradient.
    assert per_head == {
        "transformer_blocks.*.attn.norm_q",
        "transformer_blocks.*.attn.norm_k",
        "transformer_blocks.*.attn.norm_added_q",
        "transformer_blocks.*.attn.norm_added_k",
    }


def test_the_flux1_plan_leaves_the_fused_single_stream_blocks_alone() -> None:
    """`proj_out` consumes `cat([attn_output, mlp_hidden_states])`.

    Its input dimension is packed from two independently column-sharded
    streams, and no placement describes that layout while keeping
    `full_tensor()` equal to the original weight, so sharding it would corrupt
    every saved checkpoint. Replicating it then forces the projections that feed
    it to stay replicated too, which is the whole single-stream stack.
    """
    from flow_factory.models.flux.flux1 import Flux1Adapter

    plan = Flux1Adapter.tensor_parallel_plan()

    assert not [key for key in plan if key.startswith("single_transformer_blocks")]


def test_the_flux1_kontext_plan_is_the_flux1_plan() -> None:
    """Kontext loads the same `FluxTransformer2DModel`, so it inherits the plan."""
    from flow_factory.models.flux.flux1 import Flux1Adapter
    from flow_factory.models.flux.flux1_kontext import Flux1KontextAdapter

    expected = Flux1Adapter.tensor_parallel_plan()
    actual = Flux1KontextAdapter.tensor_parallel_plan()

    assert set(actual) == set(expected)
    assert all(type(actual[key]) is type(expected[key]) for key in expected)


def test_the_flux2_plan_shards_only_the_unfused_double_stream_attention() -> None:
    """Both of FLUX.2's feed-forward paths are fused, in both block stacks.

    `ff.linear_in` packs SwiGLU's `[gate, up]`; the single-stream block packs
    `[q, k, v, gate, up]` into `to_qkv_mlp_proj` and `[attn, mlp]` into
    `to_out`. A replicated `linear_in` leaves no shard for `linear_out` to
    consume, so the whole feed-forward stays replicated and only the
    double-stream attention is left.
    """
    from flow_factory.models.flux.flux2 import Flux2Adapter

    colwise, rowwise, per_head = _styles(Flux2Adapter.tensor_parallel_plan())

    assert colwise == {
        "transformer_blocks.*.attn.to_q",
        "transformer_blocks.*.attn.to_k",
        "transformer_blocks.*.attn.to_v",
        "transformer_blocks.*.attn.add_q_proj",
        "transformer_blocks.*.attn.add_k_proj",
        "transformer_blocks.*.attn.add_v_proj",
    }
    assert rowwise == {
        "transformer_blocks.*.attn.to_out.0",
        "transformer_blocks.*.attn.to_add_out",
    }
    assert per_head == {
        "transformer_blocks.*.attn.norm_q",
        "transformer_blocks.*.attn.norm_k",
        "transformer_blocks.*.attn.norm_added_q",
        "transformer_blocks.*.attn.norm_added_k",
    }
    assert not [key for key in colwise | rowwise if "linear_in" in key or "linear_out" in key]
    assert not [key for key in colwise | rowwise if key.startswith("single_transformer_blocks")]


def test_the_flux2_klein_plan_is_the_flux2_plan() -> None:
    """Klein loads the same `Flux2Transformer2DModel`, so it inherits the plan."""
    from flow_factory.models.flux.flux2 import Flux2Adapter
    from flow_factory.models.flux.flux2_klein import Flux2KleinAdapter

    expected = Flux2Adapter.tensor_parallel_plan()
    actual = Flux2KleinAdapter.tensor_parallel_plan()

    assert set(actual) == set(expected)
    assert all(type(actual[key]) is type(expected[key]) for key in expected)


def test_the_flux3_plan_names_every_block_stack() -> None:
    """FLUX.3's blocks live in four places, two of them under a `ModuleDict`.

    One `*` stands for exactly one path segment, so `mode_blocks.*` cannot reach
    `mode_blocks.image.0`; each token mode has to be named.
    """
    from flow_factory.models.flux.flux3 import Flux3Adapter

    colwise, rowwise, per_head = _styles(Flux3Adapter.tensor_parallel_plan())
    stacks = (
        "mode_blocks.image",
        "mode_blocks.image_cond",
        "txt_mode_blocks",
        "single_transformer_blocks",
    )

    assert colwise == {
        f"{stack}.*.attn.{projection}"
        for stack in stacks
        for projection in ("to_q", "to_k", "to_v")
    }
    assert rowwise == {f"{stack}.*.attn.to_out.0" for stack in stacks}
    assert per_head == {
        f"{stack}.*.attn.{norm}" for stack in stacks for norm in ("norm_q", "norm_k")
    }
    # The SwiGLU `ff.linear_in` packs `[gate, up]` into one weight, which no
    # placement shards with a correct global view; `linear_out` then has no
    # sharded input to consume.
    assert not [key for key in colwise | rowwise if ".ff." in key]


def test_the_wan_plans_shard_the_feed_forward_and_nothing_else() -> None:
    """Wan's qk-norm is across-head, so no attention projection may be named.

    ``WanAttention`` builds ``RMSNorm(dim_head * heads)`` and the processor
    applies it before the head split, so a column-parallel q/k would normalize
    over a slice of the inner dimension. That is a wrong forward, not the
    partial-sum gradient ``ShardedInputReplicatedWeight`` repairs, so the plan
    has to stop at the feed-forward.
    """
    from flow_factory.models.wan.wan2_i2v import Wan2_I2V_Adapter
    from flow_factory.models.wan.wan2_t2v import Wan2_T2V_Adapter

    expected_colwise = {"blocks.*.ffn.net.0.proj"}
    expected_rowwise = {"blocks.*.ffn.net.2"}

    for adapter in (Wan2_T2V_Adapter, Wan2_I2V_Adapter):
        plan = adapter.tensor_parallel_plan()
        assert {k for k, v in plan.items() if isinstance(v, ColwiseParallel)} == expected_colwise
        assert {k for k, v in plan.items() if isinstance(v, RowwiseParallel)} == expected_rowwise
        assert not any(isinstance(v, ShardedInputReplicatedWeight) for v in plan.values())
        assert not any(
            key.rsplit(".", 1)[-1] in {"to_q", "to_k", "to_v", "add_k_proj", "add_v_proj"}
            for key in plan
        )


def test_the_two_wan_plans_are_the_same_mapping() -> None:
    """Both adapters shard the same ``WanTransformer3DModel``.

    The adapter hierarchy is flat, so the plan is repeated rather than
    inherited; this is what keeps the two copies from drifting apart.
    """
    from flow_factory.models.wan.wan2_i2v import Wan2_I2V_Adapter
    from flow_factory.models.wan.wan2_t2v import Wan2_T2V_Adapter

    t2v = Wan2_T2V_Adapter.tensor_parallel_plan()
    i2v = Wan2_I2V_Adapter.tensor_parallel_plan()

    assert set(t2v) == set(i2v)
    assert all(type(t2v[key]) is type(i2v[key]) for key in t2v)


def test_the_ltx2_plans_shard_both_feed_forwards_and_no_attention() -> None:
    """LTX-2 is blocked twice: across-head qk-norm and full-width RoPE.

    ``LTX2Attention`` rejects any ``qk_norm`` but ``rms_norm_across_heads``, and
    the processor applies RoPE to the flat projection before the head split, so
    the sharded width would also mismatch the ``rope`` module's output.
    """
    from flow_factory.models.ltx2.ltx2_i2av import LTX2_I2AV_Adapter
    from flow_factory.models.ltx2.ltx2_t2av import LTX2_T2AV_Adapter

    expected_colwise = {
        "transformer_blocks.*.ff.net.0.proj",
        "transformer_blocks.*.audio_ff.net.0.proj",
    }
    expected_rowwise = {
        "transformer_blocks.*.ff.net.2",
        "transformer_blocks.*.audio_ff.net.2",
    }

    for adapter in (LTX2_T2AV_Adapter, LTX2_I2AV_Adapter):
        plan = adapter.tensor_parallel_plan()
        assert {k for k, v in plan.items() if isinstance(v, ColwiseParallel)} == expected_colwise
        assert {k for k, v in plan.items() if isinstance(v, RowwiseParallel)} == expected_rowwise
        assert not any("attn" in key for key in plan)


def test_the_two_ltx2_plans_are_the_same_mapping() -> None:
    """Both adapters shard the same ``LTX2VideoTransformer3DModel``."""
    from flow_factory.models.ltx2.ltx2_i2av import LTX2_I2AV_Adapter
    from flow_factory.models.ltx2.ltx2_t2av import LTX2_T2AV_Adapter

    t2av = LTX2_T2AV_Adapter.tensor_parallel_plan()
    i2av = LTX2_I2AV_Adapter.tensor_parallel_plan()

    assert set(t2av) == set(i2av)
    assert all(type(t2av[key]) is type(i2av[key]) for key in t2av)


def test_every_video_plan_pattern_resolves_against_its_transformer() -> None:
    """A plan key that names no module is the failure mode worth catching on CPU.

    ``build_total_plan`` raises on an unmatched pattern, so expanding each plan
    against a two-layer instance of the real diffusers class is what proves the
    paths are the ones these architectures actually have -- including that Wan's
    stack is ``blocks`` and LTX-2's is ``transformer_blocks``.
    """
    pytest.importorskip("diffusers")
    from diffusers.models.transformers.transformer_ltx2 import LTX2VideoTransformer3DModel
    from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
    from flow_factory.models.ltx2.ltx2_t2av import LTX2_T2AV_Adapter
    from flow_factory.models.wan.wan2_t2v import Wan2_T2V_Adapter

    wan = WanTransformer3DModel(
        num_layers=2, num_attention_heads=4, attention_head_dim=8, ffn_dim=64
    )
    wan_plan = build_total_plan(wan, Wan2_T2V_Adapter.tensor_parallel_plan())
    assert isinstance(wan_plan["blocks.0.ffn.net.0.proj"], ColwiseParallel)
    assert isinstance(wan_plan["blocks.1.ffn.net.2"], RowwiseParallel)
    # Replicated completion still has to reach the attention it declines to shard.
    assert isinstance(wan_plan["blocks.0.attn1.to_q"], ReplicateParallel)
    assert isinstance(wan_plan["blocks.0.attn1.norm_q"], ReplicateParallel)

    ltx2 = LTX2VideoTransformer3DModel(
        num_layers=2,
        num_attention_heads=2,
        attention_head_dim=8,
        audio_num_attention_heads=2,
        audio_attention_head_dim=4,
    )
    ltx2_plan = build_total_plan(ltx2, LTX2_T2AV_Adapter.tensor_parallel_plan())
    assert isinstance(ltx2_plan["transformer_blocks.0.ff.net.0.proj"], ColwiseParallel)
    assert isinstance(ltx2_plan["transformer_blocks.1.audio_ff.net.2"], RowwiseParallel)
    assert isinstance(ltx2_plan["transformer_blocks.0.audio_to_video_attn.to_v"], ReplicateParallel)


def test_both_video_families_qk_norm_over_the_whole_inner_dimension() -> None:
    """The measurement the plans rest on, taken from the instantiated modules.

    If an upstream release ever narrows these norms to ``[attention_head_dim]``,
    the attention projections become shardable with
    ``ShardedInputReplicatedWeight`` and this assertion is what says so.
    """
    pytest.importorskip("diffusers")
    from diffusers.models.transformers.transformer_ltx2 import LTX2VideoTransformer3DModel
    from diffusers.models.transformers.transformer_wan import WanTransformer3DModel

    wan = WanTransformer3DModel(
        num_layers=1, num_attention_heads=4, attention_head_dim=8, ffn_dim=64
    )
    attention = wan.blocks[0].attn1
    assert attention.heads == 4
    assert tuple(attention.norm_q.normalized_shape) == (4 * 8,)
    assert tuple(attention.norm_k.normalized_shape) == (4 * 8,)

    ltx2 = LTX2VideoTransformer3DModel(
        num_layers=1,
        num_attention_heads=2,
        attention_head_dim=8,
        audio_num_attention_heads=2,
        audio_attention_head_dim=4,
    )
    block = ltx2.transformer_blocks[0]
    assert tuple(block.attn1.norm_q.normalized_shape) == (2 * 8,)
    assert tuple(block.audio_attn1.norm_q.normalized_shape) == (2 * 4,)


def test_the_z_image_plan_covers_every_block_stack() -> None:
    """All three stacks are the same block, so naming only `layers` would be arbitrary.

    `noise_refiner` and `context_refiner` are 2 blocks each against 30 in
    `layers`, but they are `ZImageTransformerBlock` too, and the plan's `*`
    stands for one path segment -- so each stack needs its own entries.
    """
    from flow_factory.models.z_image.z_image import ZImageAdapter

    plan = ZImageAdapter.tensor_parallel_plan()
    stacks = {"layers", "noise_refiner", "context_refiner"}

    for stack in stacks:
        assert {k for k in plan if k.startswith(f"{stack}.")} == {
            f"{stack}.*.attn.to_q".replace("attn", "attention"),
            f"{stack}.*.attention.to_k",
            f"{stack}.*.attention.to_v",
            f"{stack}.*.attention.to_out.0",
            f"{stack}.*.attention.norm_q",
            f"{stack}.*.attention.norm_k",
            f"{stack}.*.feed_forward.w1",
            f"{stack}.*.feed_forward.w2",
            f"{stack}.*.feed_forward.w3",
        }
    assert {k.split(".")[0] for k in plan} == stacks


def test_the_z_image_plan_separates_the_unpacked_swiglu() -> None:
    """Z-Image keeps SwiGLU's up and gate as two Linears, so both are columnwise.

    This is the hazard MiniMax-H3 hits and Z-Image does not: a single packed
    `[hidden, gate]` weight has no placement that both splits each half and
    reconstructs to the original, so H3 leaves its fused projection replicated.
    """
    from flow_factory.models.z_image.z_image import ZImageAdapter

    plan = ZImageAdapter.tensor_parallel_plan()

    assert isinstance(plan["layers.*.feed_forward.w1"], ColwiseParallel)
    assert isinstance(plan["layers.*.feed_forward.w3"], ColwiseParallel)
    assert isinstance(plan["layers.*.feed_forward.w2"], RowwiseParallel)
    assert isinstance(plan["layers.*.attention.norm_q"], ShardedInputReplicatedWeight)


def test_the_z_image_plan_names_no_optional_or_modulation_module() -> None:
    """A pattern matching nothing is rejected, so optional stacks must stay out.

    `siglip_refiner` exists only when `siglip_feat_dim` is set, which neither
    published checkpoint does; naming it would make `build_total_plan` raise on
    every real Z-Image. `adaLN_modulation` is present but emits full-width
    modulation vectors applied to replicated activations.
    """
    from flow_factory.models.z_image.z_image import ZImageAdapter

    plan = ZImageAdapter.tensor_parallel_plan()

    assert not [k for k in plan if "siglip" in k]
    assert not [k for k in plan if "adaLN" in k]


def test_the_sensenova_plan_shards_both_decoder_branches() -> None:
    """The understanding branch fills the cache the generation branch attends over.

    Only `*_mot_gen` is trained, but sharding it alone leaves the prefix KV cache
    holding all 8 kv heads while the denoising step holds `8 / tp_size`, and the
    sequence-axis concat fails on the head axis.
    """
    from flow_factory.models.sensenova.sensenova import SenseNovaAdapter

    plan = SenseNovaAdapter.tensor_parallel_plan()
    prefix = "model.language_model.model.layers.*"

    for branch in ("", "_mot_gen"):
        assert isinstance(plan[f"{prefix}.self_attn.q_proj{branch}"], ColwiseParallel)
        assert isinstance(plan[f"{prefix}.self_attn.k_proj{branch}"], ColwiseParallel)
        assert isinstance(plan[f"{prefix}.self_attn.v_proj{branch}"], ColwiseParallel)
        assert isinstance(plan[f"{prefix}.self_attn.o_proj{branch}"], RowwiseParallel)
        assert isinstance(plan[f"{prefix}.mlp{branch}.gate_proj"], ColwiseParallel)
        assert isinstance(plan[f"{prefix}.mlp{branch}.up_proj"], ColwiseParallel)
        assert isinstance(plan[f"{prefix}.mlp{branch}.down_proj"], RowwiseParallel)

    # Four per-head norms per branch: the temporal and spatial half of q and k.
    assert {k for k, v in plan.items() if isinstance(v, ShardedInputReplicatedWeight)} == {
        f"{prefix}.self_attn.{projection}_norm{suffix}{branch}"
        for projection in ("q", "k")
        for suffix in ("", "_hw")
        for branch in ("", "_mot_gen")
    }


def test_the_sensenova_plan_leaves_the_vocabulary_boundary_replicated() -> None:
    """Column parallelism on the embedding or the head emits an unreduced axis."""
    from flow_factory.models.sensenova.sensenova import SenseNovaAdapter

    plan = SenseNovaAdapter.tensor_parallel_plan()

    assert not [k for k in plan if "embed_tokens" in k or "lm_head" in k]
    assert not [k for k in plan if "vision_model" in k or "fm_modules" in k]


def test_the_bagel_plan_shards_the_feed_forward_and_not_the_attention() -> None:
    """Bagel's MoT attention cannot be column-sharded, and leaving it out is the plan.

    `PackedAttentionMoT` scatters both branches' projections into one
    preallocated `new_zeros((L, num_heads * head_dim))` buffer and writes the
    attention output back through `torch.zeros_like`, which conflates
    `num_heads * head_dim` with `hidden_size`. Those are equal only while
    attention is unsharded. `num_heads` is also captured from the config at
    `__init__` and is not named `heads`, so `localize_attention_head_counts`
    cannot reach it. The feed-forward branches have neither problem: `Qwen2MLP`
    is an unpacked SwiGLU and both branches emit full-width activations.
    """
    pytest.importorskip(
        "flash_attn", reason="Bagel's vendored attention imports it at module scope"
    )
    from flow_factory.models.bagel.bagel import BagelAdapter

    plan = BagelAdapter.tensor_parallel_plan()

    assert set(plan) == {
        f"model.layers.*.mlp{branch}.{leaf}"
        for branch in ("", "_moe_gen")
        for leaf in ("gate_proj", "up_proj", "down_proj")
    }
    for branch in ("", "_moe_gen"):
        assert isinstance(plan[f"model.layers.*.mlp{branch}.gate_proj"], ColwiseParallel)
        assert isinstance(plan[f"model.layers.*.mlp{branch}.up_proj"], ColwiseParallel)
        assert isinstance(plan[f"model.layers.*.mlp{branch}.down_proj"], RowwiseParallel)


@pytest.mark.parametrize("architecture", ["z-image", "sensenova", "bagel"])
def test_the_unified_and_z_image_plans_resolve_against_their_real_trees(
    architecture: str,
) -> None:
    """Every pattern must name a module the real class has, at the real widths.

    Built on the meta device at two layers, so this stays a CPU test. Two of the
    three are not diffusers transformers -- SenseNova's component is a
    ``SenseNovaDenoiser`` wrapping a custom Transformers composite model, Bagel's
    is a vendored ``Qwen2ForCausalLM`` -- so this is also where the ``model.``
    prefixes those wrappers introduce are checked.
    """
    if architecture == "z-image":
        from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel
        from flow_factory.models.z_image.z_image import ZImageAdapter

        with torch.device("meta"):
            # Tongyi-MAI/Z-Image transformer/config.json, with the depths cut.
            model = ZImageTransformer2DModel(
                dim=3840,
                n_layers=2,
                n_refiner_layers=1,
                n_heads=30,
                n_kv_heads=30,
                cap_feat_dim=2560,
                axes_dims=[32, 48, 48],
                axes_lens=[1536, 512, 512],
            )
        shard_plan = ZImageAdapter.tensor_parallel_plan()
        # 30 heads divide by 2 and by neither 4 nor 8.
        assert model.config.n_heads % 2 == 0
        assert model.config.n_heads % 4 != 0
    elif architecture == "sensenova":
        from flow_factory.models.sensenova.modeling.neo_unify.configuration_neo_chat import (
            NEOChatConfig,
        )
        from flow_factory.models.sensenova.modeling.neo_unify.modeling_neo_chat import NEOChatModel
        from flow_factory.models.sensenova.pipeline import SenseNovaDenoiser
        from flow_factory.models.sensenova.sensenova import SenseNovaAdapter

        with torch.device("meta"):
            # sensenova/SenseNova-U1.5-8B-MoT-SFT config.json, with the depth cut.
            model = SenseNovaDenoiser(
                NEOChatModel(
                    NEOChatConfig(
                        vision_config={
                            "hidden_size": 1024,
                            "llm_hidden_size": 4096,
                            "downsample_ratio": 0.5,
                            "patch_size": 16,
                            "num_channels": 3,
                        },
                        llm_config={
                            "architectures": ["Qwen3ForCausalLM"],
                            "hidden_size": 4096,
                            "intermediate_size": 12288,
                            "num_hidden_layers": 2,
                            "num_attention_heads": 32,
                            "num_key_value_heads": 8,
                            "head_dim": 128,
                            "vocab_size": 151936,
                        },
                        downsample_ratio=0.5,
                        patch_size=16,
                        fm_head_dim=1536,
                        fm_head_layers=2,
                        fm_head_mlp_ratio=1,
                        use_pixel_head=True,
                        concat_time_token_num=0,
                        add_noise_scale_embedding=True,
                        noise_scale_max_value=16.0,
                        noise_scale=1.0,
                        noise_scale_mode="resolution",
                        noise_scale_base_image_seq_len=64,
                        t_eps=0.05,
                        template="neo1_0",
                        time_schedule="standard",
                        time_shift_type="exponential",
                        base_shift=0.5,
                        max_shift=1.15,
                        base_image_seq_len=64,
                        max_image_seq_len=4096,
                    )
                )
            )
        shard_plan = SenseNovaAdapter.tensor_parallel_plan()
        attention = model.model.language_model.model.layers[0].self_attn
        # Per-head qk-norm over half a head, not across heads.
        assert attention.q_norm.weight.shape == (attention.head_dim // 2,)
    else:
        pytest.importorskip(
            "flash_attn", reason="Bagel's vendored attention imports it at module scope"
        )
        from flow_factory.models.bagel.bagel import BagelAdapter
        from flow_factory.models.bagel.modeling.bagel.qwen2_navit import (
            Qwen2Config,
            Qwen2ForCausalLM,
        )

        with torch.device("meta"):
            # ByteDance-Seed/BAGEL-7B-MoT llm_config.json, with the depth cut.
            model = Qwen2ForCausalLM(
                Qwen2Config(
                    hidden_size=3584,
                    intermediate_size=18944,
                    num_hidden_layers=2,
                    num_attention_heads=28,
                    num_key_value_heads=4,
                    vocab_size=152064,
                    qk_norm=True,
                    layer_module="Qwen2MoTDecoderLayer",
                    pad_token_id=151643,
                )
            )
        shard_plan = BagelAdapter.tensor_parallel_plan()
        # The conflation the plan refuses to shard around: the attention output
        # width and the hidden width are the same number only while unsharded.
        attention = model.model.layers[0].self_attn
        assert attention.num_heads * attention.head_dim == attention.hidden_size
        assert not hasattr(attention, "heads")

    plan = build_total_plan(model, shard_plan)

    owners = {
        name
        for name, module in model.named_modules()
        if name and any(True for _ in module.named_parameters(recurse=False))
    }
    assert owners - set(plan) == set()
    wanted = {
        pattern.replace("*", "0")
        for pattern, style in shard_plan.items()
        if isinstance(style, (ColwiseParallel, RowwiseParallel))
    }
    assert wanted <= set(plan)
