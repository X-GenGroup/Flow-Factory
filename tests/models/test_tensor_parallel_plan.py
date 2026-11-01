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
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    PrepareModuleOutput,
    RowwiseParallel,
)

from flow_factory.models.tensor_parallel import (
    ComposedParallelStyle,
    ReplicateParallel,
    SequenceBroadcastOutput,
    SequenceBroadcastParameter,
    ShardedAcrossHeadRMSNorm,
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


def test_the_sd3_5_qk_norms_declare_the_shard_axis_its_processor_uses() -> None:
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

    assert per_head and all(style.shard_axis == 1 for style in per_head)


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
        style.shard_axis == 2
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


def test_the_flux2_family_declines_tensor_parallelism() -> None:
    """An empty plan is the verdict, not an omission.

    FLUX.2 fuses almost everything: a single-stream block holds only
    ``to_qkv_mlp_proj`` (packing ``[q, k, v, gate, up]``) and ``to_out``
    (packing ``[attn, mlp]``), and the double blocks' ``ff.linear_in`` packs
    SwiGLU's ``[gate, up]``. A packed weight has no placement that both computes
    correctly and keeps ``full_tensor()`` equal to the original, so a plan that
    preserves checkpoints reaches 7.5% of FLUX.2-dev -- at which coverage
    tensor parallelism costs 92% more parameter memory per rank and half the
    throughput at ``tp=2``.

    An empty plan is what makes the backend validator reject ``tp_size > 1``
    with a reason, so this asserts the emptiness rather than a key set.
    """
    from flow_factory.models.flux.flux2 import Flux2Adapter
    from flow_factory.models.flux.flux2_klein import Flux2KleinAdapter

    assert Flux2Adapter.tensor_parallel_plan() == {}
    # Klein states its own, for different numbers: 3.88B/9.08B over shallower
    # single stacks, which needs no sharding at all on eight ranks.
    assert Flux2KleinAdapter.tensor_parallel_plan() == {}


@pytest.mark.parametrize("variant", ["dev", "klein"])
def test_the_flux2_family_names_the_single_stream_output_projection(variant: str) -> None:
    """The bare ``attn.to_out`` is unreachable by suffix and needs its path.

    The double blocks wrap it in a ``ModuleList`` (``attn.to_out.0``) while the
    single blocks hold it directly, so the ``.0`` suffix misses the single
    stack and the bare suffix resolves to the ``ModuleList``, which PEFT
    rejects. Left unnamed this drops 22.5% of FLUX.2-dev's parameters from the
    adapter without any error.
    """
    from flow_factory.models.flux.flux2 import Flux2Adapter
    from flow_factory.models.flux.flux2_klein import Flux2KleinAdapter

    adapter = Flux2Adapter if variant == "dev" else Flux2KleinAdapter
    targets = list(adapter.default_target_modules.fget(adapter))

    assert len(targets) == len(set(targets)), "duplicate target"
    assert "attn.to_out" not in targets, "the bare suffix hits the double blocks' ModuleList"
    paths = [t for t in targets if t.startswith("single_transformer_blocks")]
    # Must reach past the deepest published stack (48 for FLUX.2-dev).
    assert len(paths) > 48
    assert all(t.endswith(".attn.to_out") for t in paths)


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


def test_the_wan_plans_give_every_across_head_norm_its_own_style() -> None:
    """Wan's qk-norm spans all heads, so it cannot take the per-head style.

    ``WanAttention`` builds ``RMSNorm(dim_head * heads)`` and the processor
    applies it before the head split, so column-parallel q/k split the
    *normalized* axis. Replicating the weight there is a wrong forward, not the
    partial-sum gradient ``ShardedInputReplicatedWeight`` repairs, and using
    that style would be the plausible mistake -- hence the explicit assertion
    that it appears nowhere in these plans.
    """
    from flow_factory.models.wan.wan2_i2v import Wan2_I2V_Adapter
    from flow_factory.models.wan.wan2_t2v import Wan2_T2V_Adapter

    for adapter in (Wan2_T2V_Adapter, Wan2_I2V_Adapter):
        plan = adapter.tensor_parallel_plan()
        norms = {key for key in plan if key.rsplit(".", 1)[-1].startswith("norm")}
        assert norms, adapter.__name__
        assert all(isinstance(plan[key], ShardedAcrossHeadRMSNorm) for key in norms)
        assert not any(isinstance(style, ShardedInputReplicatedWeight) for style in plan.values())
        # Every query projection sharded means every norm it feeds must be too.
        for stack in ("attn1", "attn2"):
            assert isinstance(plan[f"blocks.*.{stack}.to_q"], ColwiseParallel)
            assert isinstance(plan[f"blocks.*.{stack}.to_out.0"], RowwiseParallel)


def test_the_image_variant_shards_its_extra_key_value_stream() -> None:
    """I2V adds a second key/value stream into the *same* cross-attention.

    Leaving ``add_k_proj``/``add_v_proj`` at full width while ``attn2.to_q`` is
    head-sharded is caught only at runtime, by a batched matmul complaining
    "expected [20, 128] but got [20, 256]". The two plans are therefore *not*
    the same mapping, which is the thing worth pinning: T2V must not name these
    paths, because its transformer does not have them and the glob expansion
    would reject the plan.
    """
    from flow_factory.models.wan.wan2_i2v import Wan2_I2V_Adapter
    from flow_factory.models.wan.wan2_t2v import Wan2_T2V_Adapter

    t2v = Wan2_T2V_Adapter.tensor_parallel_plan()
    i2v = Wan2_I2V_Adapter.tensor_parallel_plan()

    added = {
        "blocks.*.attn2.add_k_proj",
        "blocks.*.attn2.add_v_proj",
        "blocks.*.attn2.norm_added_k",
    }
    assert set(i2v) - set(t2v) == added
    assert not set(t2v) - set(i2v)
    assert all(type(t2v[key]) is type(i2v[key]) for key in t2v)
    assert isinstance(i2v["blocks.*.attn2.norm_added_k"], ShardedAcrossHeadRMSNorm)


_LTX2_ATTENTION_STACKS = (
    "attn1",
    "audio_attn1",
    "attn2",
    "audio_attn2",
    "audio_to_video_attn",
    "video_to_audio_attn",
)


def test_the_ltx2_plans_shard_all_six_attention_stacks_and_both_feed_forwards() -> None:
    """Both of LTX-2's blockers have a placement, so the whole block shards.

    ``LTX2Attention`` rejects any ``qk_norm`` but ``rms_norm_across_heads``,
    which :class:`ShardedAcrossHeadRMSNorm` answers; and the processor applies
    RoPE to the flat ``[B, S, inner_dim]`` projection, which a
    ``PrepareModuleOutput`` on the four rotary modules answers by narrowing
    their ``(cos, sin)`` to the same channel range ``ColwiseParallel`` gave
    ``to_q``/``to_k``.
    """
    from flow_factory.models.ltx2.ltx2_i2av import LTX2_I2AV_Adapter
    from flow_factory.models.ltx2.ltx2_t2av import LTX2_T2AV_Adapter

    expected_colwise = {
        f"transformer_blocks.*.{stack}.{leaf}"
        for stack in _LTX2_ATTENTION_STACKS
        for leaf in ("to_q", "to_k", "to_v")
    } | {
        "transformer_blocks.*.ff.net.0.proj",
        "transformer_blocks.*.audio_ff.net.0.proj",
    }
    expected_rowwise = {
        f"transformer_blocks.*.{stack}.to_out.0" for stack in _LTX2_ATTENTION_STACKS
    } | {
        "transformer_blocks.*.ff.net.2",
        "transformer_blocks.*.audio_ff.net.2",
    }
    expected_across_head = {
        f"transformer_blocks.*.{stack}.{norm}"
        for stack in _LTX2_ATTENTION_STACKS
        for norm in ("norm_q", "norm_k")
    }

    for adapter in (LTX2_T2AV_Adapter, LTX2_I2AV_Adapter):
        plan = adapter.tensor_parallel_plan()
        assert {k for k, v in plan.items() if isinstance(v, ColwiseParallel)} == expected_colwise
        assert {k for k, v in plan.items() if isinstance(v, RowwiseParallel)} == expected_rowwise
        assert {
            k for k, v in plan.items() if isinstance(v, ShardedAcrossHeadRMSNorm)
        } == expected_across_head


def test_the_ltx2_plan_narrows_every_rotary_table_to_the_column_shard() -> None:
    """The entry that unlocked attention, and the axis it has to use.

    ``apply_interleaved_rotary_emb`` multiplies a full-width ``cos``/``sin``
    against the flat projection, so a column-sharded query meets a table twice
    its width -- ``tensor a (2048) must match tensor b (4096)``. The rotary
    modules return a ``(cos, sin)`` pair and own no parameters, so declaring
    both outputs ``Replicate`` and redistributing them to ``Shard(-1)`` is a
    local narrow with no collective, and lands each rank on exactly the channels
    ``ColwiseParallel`` gave it.

    All four modules are named: ``rope`` feeds video self-attention,
    ``audio_rope`` audio self-attention, and the two ``cross_attn`` ones feed the
    audio-to-video and video-to-audio pair from both sides.
    """
    from flow_factory.models.ltx2.ltx2_t2av import LTX2_T2AV_Adapter

    plan = LTX2_T2AV_Adapter.tensor_parallel_plan()

    rotary = {k for k, v in plan.items() if isinstance(v, PrepareModuleOutput)}
    assert rotary == {"rope", "audio_rope", "cross_attn_rope", "cross_attn_audio_rope"}
    for path in rotary:
        style = plan[path]
        # One layout per returned tensor: `(cos, sin)`.
        assert len(style.output_layouts) == 2
        assert all(layout.is_replicate() for layout in style.output_layouts)
        assert all(layout == Shard(-1) for layout in style.desired_output_layouts)
        assert style.use_local_output is True


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
    assert isinstance(wan_plan["blocks.0.attn1.to_q"], ColwiseParallel)
    assert isinstance(wan_plan["blocks.0.attn1.norm_q"], ShardedAcrossHeadRMSNorm)
    # Completion still has to reach what the plan declines to name -- here the
    # bare modulation table the block owns directly.
    assert isinstance(wan_plan[""], ReplicateParallel)

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
    assert isinstance(ltx2_plan["transformer_blocks.0.audio_to_video_attn.to_v"], ColwiseParallel)
    assert isinstance(
        ltx2_plan["transformer_blocks.0.video_to_audio_attn.norm_q"], ShardedAcrossHeadRMSNorm
    )
    # The rotary modules own no parameters, so completion leaves the narrowing
    # entry alone rather than composing a parameter style beside it.
    assert isinstance(ltx2_plan["cross_attn_audio_rope"], PrepareModuleOutput)


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


def _minimax_h3_transformer():
    """Two blocks of the real class on meta, at MiniMax-H3's published widths."""
    pytest.importorskip("diffusers")
    from diffusers.models.transformers.transformer_minimax_h3 import (
        MiniMaxH3Transformer3DModel,
    )

    with torch.device("meta"):
        # MiniMaxAI/MiniMax-H3 transformer/config.json, with the depth cut.
        return MiniMaxH3Transformer3DModel(
            num_attention_heads=56,
            attention_head_dim=128,
            hidden_size=5376,
            num_layers=2,
            num_refiner_layers=1,
            ffn_dim=14336,
            text_dim=5120,
            time_embed_dim=2688,
        )


def test_the_h3_sequence_parallel_plan_cuts_the_per_token_modulation_index() -> None:
    """The entry no other architecture needs, and the one Wan's plan has no model for.

    H3 addresses its AdaLN table *per row of the packed sequence*:
    ``adaln_indices`` is ``(seq_len,)`` and every block applies it as
    ``scale.index_select(0, adaln_indices)``. A block handed a token slice and
    the whole index vector therefore broadcasts a full-length modulation against
    a sliced hidden state, so every block -- not only the first -- has to re-cut
    it, and ``norm_out`` has to do the same for ``timestep_indices``. Measured on
    two ranks, omitting it fails loudly ("the size of tensor a (32) must match
    the size of tensor b (64)"), which is the one failure mode here that is not
    silent.
    """
    from flow_factory.models.minimax_h3.adapters import MiniMaxH3T2VAAdapter

    plan = MiniMaxH3T2VAAdapter.sequence_parallel_plan()

    # `adaln_indices` is the third positional argument of the block's forward,
    # `timestep_indices` the third of `norm_out`'s.
    for path, arity in (("transformer_blocks.*", 4), ("transformer_blocks.0", 4), ("norm_out", 3)):
        entry = plan[path]
        assert isinstance(entry, PrepareModuleInput), path
        assert len(entry.input_layouts) == arity, path
        assert len(entry.desired_input_layouts) == arity, path
        assert isinstance(entry.input_layouts[2], Replicate), path
        assert entry.desired_input_layouts[2] == Shard(0), path
        # The conditioning embedding is replicated and must be left alone, or
        # `PrepareModuleInput` would redistribute a tensor with no token axis.
        assert entry.input_layouts[1] is None, path
        assert entry.use_local_output is True, path

    # Only the first block scatters; the rest receive an already-sharded stream.
    assert isinstance(plan["transformer_blocks.0"].input_layouts[0], Replicate)
    assert plan["transformer_blocks.0"].desired_input_layouts[0] == Shard(1)
    assert plan["transformer_blocks.*"].input_layouts[0] == Shard(1)
    assert plan["transformer_blocks.*"].desired_input_layouts[0] == Shard(1)


def test_the_h3_sequence_parallel_plan_gathers_only_at_attention() -> None:
    """H3's feed-forward shards along tokens, which is why its plan is not Wan's.

    Wan gathers at its ``ffn`` because the tensor-parallel plan column/row-shards
    it. H3's is a packed SwiGLU with no correct column placement, so it stays
    replicated -- and a replicated position-wise layer is already right on a
    token slice. Leaving it sharded is the whole point: ``ff`` is 35.8% of a
    block and its ``2 * ffn_dim`` intermediate is the largest activation here, so
    it runs on ``1 / tp_size`` of the tokens with no collective at all.
    """
    from flow_factory.models.minimax_h3.adapters import MiniMaxH3T2VAAdapter

    plan = MiniMaxH3T2VAAdapter.sequence_parallel_plan()

    gathers = {
        path
        for path, style in plan.items()
        if isinstance(style, PrepareModuleInput)
        and any(isinstance(layout, Replicate) for layout in style.desired_input_layouts)
    }
    assert gathers == {"transformer_blocks.*.attn"}
    assert plan["transformer_blocks.*.attn"].input_layouts == (Shard(1), None, None)
    # Its out-projection reduce-scatters instead of all-reducing, which is the
    # same two collectives an all-reduce already was.
    out_projection = plan["transformer_blocks.*.attn.to_out.0"]
    assert isinstance(out_projection, RowwiseParallel)
    assert out_projection.output_layouts == (Shard(1),)

    # Every weight applied once per token inside the sharded region, including
    # both feed-forward projections and all three affine norms.
    broadcast = {
        path for path, style in plan.items() if isinstance(style, SequenceBroadcastParameter)
    }
    assert broadcast == {
        "transformer_blocks.*.norm1",
        "transformer_blocks.*.norm2",
        "transformer_blocks.*.adaln_proj.linear",
        "transformer_blocks.*.ff.net.0.proj",
        "transformer_blocks.*.ff.net.2",
        "norm_out.norm",
        "norm_out.linear",
    }
    # `temb` is broadcast over every token and was never sharded, so nothing in
    # the plan's placements records that its gradient is a partial sum. Omitting
    # this entry left the time embedder at 40% of its gradient on two ranks,
    # with a bit-identical loss.
    assert isinstance(plan["time_embedder"], SequenceBroadcastOutput)


def test_the_h3_sequence_parallel_plan_restores_the_sequence_at_the_output_heads() -> None:
    """Both heads gather, and neither can be left sharded.

    The root selects each modality's rows with ``index_select(1, video_indices)``
    over *global* positions, so the sequence has to be whole by then. Gathering
    at the heads rather than after the last block keeps ``norm_out``'s modulation
    sharded and moves far less: they narrow 5376 to 96 and to 32.
    """
    from flow_factory.models.minimax_h3.adapters import MiniMaxH3T2VAAdapter

    plan = MiniMaxH3T2VAAdapter.sequence_parallel_plan()

    for head in ("proj_out", "audio_proj_out"):
        entry = plan[head]
        assert isinstance(entry, ComposedParallelStyle), head
        # Both treatments are needed: `PrepareModuleOutput` distributes nothing,
        # and `apply_tensor_parallel` rejects a plan that leaves a parameter
        # undistributed.
        assert isinstance(entry.styles[0], SequenceBroadcastParameter), head
        assert isinstance(entry.styles[1], PrepareModuleOutput), head
        assert entry.styles[1].output_layouts == (Shard(1),)
        assert all(isinstance(l, Replicate) for l in entry.styles[1].desired_output_layouts)
        assert entry.styles[1].use_local_output is True


def test_the_h3_overlay_resolves_against_the_real_tree_and_stays_total() -> None:
    """The merged plan is what the trainer applies, so it is what has to be total.

    ``build_total_plan`` raises on a pattern that names no module, and
    ``apply_tensor_parallel`` raises afterwards on any parameter left
    undistributed -- so expanding the overlay layered over the tensor-parallel
    plan, against the real class, is what proves the paths exist and that
    completion still reaches everything the overlay does not name (the input
    projections and the token refiner, which sit outside the sharded region).
    """
    from flow_factory.models.minimax_h3.adapters import MiniMaxH3T2VAAdapter

    model = _minimax_h3_transformer()
    adapter = MiniMaxH3T2VAAdapter
    merged = {**adapter.tensor_parallel_plan(), **adapter.sequence_parallel_plan()}
    plan = build_total_plan(model, merged)

    owners = {
        name
        for name, module in model.named_modules()
        if name and any(True for _ in module.named_parameters(recurse=False))
    }
    assert owners - set(plan) == set()
    # The wildcard reached both blocks, and the exact path for the first one won.
    assert isinstance(plan["transformer_blocks.1.adaln_proj.linear"], SequenceBroadcastParameter)
    assert isinstance(plan["transformer_blocks.0"], PrepareModuleInput)
    assert isinstance(plan["transformer_blocks.0"].input_layouts[0], Replicate)
    assert plan["transformer_blocks.1"].input_layouts[0] == Shard(1)
    # The head split survives the overlay: only `to_out.0` is restated.
    assert isinstance(plan["transformer_blocks.0.attn.to_q"], ColwiseParallel)
    assert isinstance(plan["transformer_blocks.0.attn.norm_q"], ShardedInputReplicatedWeight)
    assert plan["transformer_blocks.0.attn.to_out.0"].output_layouts == (Shard(1),)
    # Outside the sharded region, so completion is the right answer for both.
    assert isinstance(plan["context_embedder"], ReplicateParallel)
    assert isinstance(plan["token_refiner.refiner_blocks.0.ff.net.2"], ReplicateParallel)


def test_the_three_h3_adapters_declare_one_sequence_parallel_plan() -> None:
    """All three shard the same transformer, under two different component names.

    Ref2VA's is ``transformer_ref`` and the other two are ``transformer``, but
    the class behind them is ``MiniMaxH3Transformer3DModel`` either way, so the
    plan lives once on the shared mixin rather than three times.
    """
    from flow_factory.models.minimax_h3.adapters import (
        MiniMaxH3FL2VAAdapter,
        MiniMaxH3Ref2VAAdapter,
        MiniMaxH3T2VAAdapter,
    )

    adapters = (MiniMaxH3T2VAAdapter, MiniMaxH3FL2VAAdapter, MiniMaxH3Ref2VAAdapter)
    # A classmethod binds to its class, so the three attributes are distinct
    # objects wrapping one function; `__func__` is what identity has to be read
    # from.
    assert len({adapter.sequence_parallel_plan.__func__ for adapter in adapters}) == 1
    plans = [adapter.sequence_parallel_plan() for adapter in adapters]
    assert all(set(plan) == set(plans[0]) for plan in plans)
    assert all(type(plan[key]) is type(plans[0][key]) for plan in plans for key in plan)


def _image_stream_overlays() -> dict:
    """The three image adapters whose overlay shards an image token axis."""
    from flow_factory.models.qwen_image.qwen_image import QwenImageAdapter
    from flow_factory.models.qwen_image_21.qwen_image_21 import QwenImage21Adapter
    from flow_factory.models.stable_diffusion.sd3_5 import SD3_5Adapter

    return {
        "sd3-5": SD3_5Adapter,
        "qwen-image": QwenImageAdapter,
        "qwen-image-2.1": QwenImage21Adapter,
    }


def test_the_sd3_5_overlay_scatters_at_the_patch_embedding_not_at_a_block() -> None:
    """The scatter has to be an *output* transform, and that is not a stylistic choice.

    diffusers calls ``JointTransformerBlock`` with keyword arguments normally and
    positionally through ``_gradient_checkpointing_func``, and
    ``PrepareModuleInput`` serves one convention or the other. A forward hook on
    ``pos_embed`` -- the module whose output *is* the image stream -- is immune to
    that, which is why this overlay verifies identically with and without
    activation checkpointing while Qwen-Image-2.1's cannot.
    """
    from flow_factory.models.stable_diffusion.sd3_5 import SD3_5Adapter

    plan = SD3_5Adapter.sequence_parallel_plan()

    scatter = plan["pos_embed"]
    assert isinstance(scatter, PrepareModuleOutput)
    assert isinstance(scatter.output_layouts[0], Replicate)
    assert scatter.desired_output_layouts[0] == Shard(1)
    assert scatter.use_local_output is True
    # No block-input transform at all: nothing in the overlay names a block
    # itself, only its children.
    assert not [
        path
        for path, style in plan.items()
        if isinstance(style, PrepareModuleInput) and path.count(".") == 1
    ]


def test_the_sd3_5_overlay_shards_the_image_stream_and_not_the_text_stream() -> None:
    """SD3.5's text is 589 tokens against 4,096 image tokens at 1024x1024.

    Sharding it would buy a few percent of the activation memory, add a
    transition at ``to_add_out`` and ``ff_context``, require a
    :class:`SequenceBroadcastOutput` on every ``norm1_context.linear``, and 589
    -- 77 CLIP positions plus 512 T5 ones -- is odd, so it would not divide even
    at ``tp=2``. So no text-stream module may appear in the overlay -- and the joint attention still sees both whole,
    because ``attn`` gathers its image input before the concatenation.
    """
    from flow_factory.models.stable_diffusion.sd3_5 import SD3_5Adapter

    plan = SD3_5Adapter.sequence_parallel_plan()

    assert not [
        path
        for path in plan
        if any(token in path for token in ("to_add_out", "ff_context", "norm1_context"))
    ]
    # The image stream's three out-projections reduce-scatter; the text
    # stream's keep the plain rowwise entry of the tensor-parallel plan.
    assert {path for path, style in plan.items() if isinstance(style, RowwiseParallel)} == {
        "transformer_blocks.*.attn.to_out.0",
        "transformer_blocks.*.attn2.to_out.0",
        "transformer_blocks.*.ff.net.2",
    }
    # Both attentions and the image feed-forward gather on entry.
    assert {path for path, style in plan.items() if isinstance(style, PrepareModuleInput)} == {
        "transformer_blocks.*.attn",
        "transformer_blocks.*.attn2",
        "transformer_blocks.*.ff",
    }


def test_the_sd3_5_overlay_sums_the_modulation_at_the_linear_not_the_norm() -> None:
    """``shift_msa``/``scale_msa`` never leave ``norm1``, so its tuple is not enough.

    ``AdaLayerNormZero`` consumes two of its six chunks inside its own forward
    and returns the other four. Reducing the returned tuple would leave those
    two with this rank's share of the tokens -- bit-identical loss, 1/tp
    gradient. Reducing at ``linear``'s output covers all six (nine on a
    dual-attention block) and simultaneously repairs the gradient flowing on to
    ``time_text_embed``, which is why that module needs no entry of its own.
    """
    from flow_factory.models.stable_diffusion.sd3_5 import SD3_5Adapter

    plan = SD3_5Adapter.sequence_parallel_plan()

    assert {path for path, style in plan.items() if isinstance(style, SequenceBroadcastOutput)} == {
        "transformer_blocks.*.norm1.linear",
        "norm_out.linear",
    }
    assert "transformer_blocks.*.norm1" not in plan
    assert "time_text_embed" not in plan


def test_the_qwen_image_overlay_scatters_at_img_in() -> None:
    """``img_in``'s output is the image stream, so the scatter is convention-free.

    Its own weight gradient is unaffected: it runs on the whole replicated
    sequence before the scatter and the scatter's backward all-gathers, so every
    rank computes the entire gradient and the completion's ``ReplicateParallel``
    is right for it.
    """
    from flow_factory.models.qwen_image.qwen_image import QwenImageAdapter

    plan = QwenImageAdapter.sequence_parallel_plan()

    scatter = plan["img_in"]
    assert isinstance(scatter, PrepareModuleOutput)
    assert isinstance(scatter.output_layouts[0], Replicate)
    assert scatter.desired_output_layouts[0] == Shard(1)
    assert not [
        path
        for path, style in plan.items()
        if isinstance(style, PrepareModuleInput) and path.count(".") == 1
    ]


def test_the_qwen_image_overlay_shards_the_image_stream_and_not_the_text_stream() -> None:
    """512 text tokens against 4,096 image tokens, and Edit-Plus lengthens only the image.

    So ``txt_mod``, ``txt_mlp`` and ``to_add_out`` must stay out of the overlay,
    and the entries that are in it must be the image stream's: one gather per
    attention and per image feed-forward, one reduce-scatter per image
    out-projection.
    """
    from flow_factory.models.qwen_image.qwen_image import QwenImageAdapter

    plan = QwenImageAdapter.sequence_parallel_plan()

    assert not [
        path
        for path in plan
        if any(token in path for token in ("txt_mod", "txt_mlp", "to_add_out"))
    ]
    assert {path for path, style in plan.items() if isinstance(style, RowwiseParallel)} == {
        "transformer_blocks.*.attn.to_out.0",
        "transformer_blocks.*.img_mlp.net.2",
    }
    assert {path for path, style in plan.items() if isinstance(style, PrepareModuleInput)} == {
        "transformer_blocks.*.attn",
        "transformer_blocks.*.img_mlp",
    }
    # `img_mod` emits six full-width vectors per block, broadcast over every
    # image token; `txt_mod` does the same over a replicated stream and needs
    # nothing.
    assert {path for path, style in plan.items() if isinstance(style, SequenceBroadcastOutput)} == {
        "transformer_blocks.*.img_mod",
        "norm_out.linear",
    }


def test_qwen_image_edit_plus_reuses_the_qwen_image_sequence_parallel_plan() -> None:
    """One transformer class must not acquire two overlays that can drift apart.

    The shard plan is already delegated for this reason; the overlay has to be
    too, because it names paths in the same ``QwenImageTransformer2DModel`` tree
    and makes the same dual-stream choice.
    """
    from flow_factory.models.qwen_image.qwen_image import QwenImageAdapter
    from flow_factory.models.qwen_image.qwen_image_edit_plus import QwenImageEditPlusAdapter

    plan = QwenImageEditPlusAdapter.sequence_parallel_plan()
    reference = QwenImageAdapter.sequence_parallel_plan()

    assert plan.keys() == reference.keys()
    assert all(type(plan[key]) is type(reference[key]) for key in reference)


def test_the_qwen_image_21_overlay_reslices_the_modulation_mask_in_every_block() -> None:
    """2.1 materializes its modulation per token, so the mask has to shard with it.

    ``causal_condition`` makes ``_select_modulation_rows`` build a
    ``[batch, tokens, dim]`` tensor through
    ``torch.where(target_token_mask, real_row, t0_row)`` rather than broadcasting
    a ``[batch, 1, dim]`` vector, four times per block. The model hands every
    block the same whole-length mask, so every block -- not only the first --
    has to re-cut it; measured on two ranks, omitting it is loud ("The size of
    tensor a (2112) must match the size of tensor b (1056)").

    The entries are keyed on the block's *keyword* arguments, which is the
    un-checkpointed calling convention: ``train.enable_gradient_checkpointing``
    defaults to false, a name survives a reordering of the signature that a slot
    index would not, and checkpointing already discards the tensors this overlay
    shards.
    """
    from flow_factory.models.qwen_image_21.qwen_image_21 import QwenImage21Adapter

    plan = QwenImage21Adapter.sequence_parallel_plan()

    for path in ("transformer_blocks.*", "transformer_blocks.0"):
        entry = plan[path]
        assert isinstance(entry, PrepareModuleInput), path
        # Keyword-addressed, so nothing is declared about positional slots.
        assert entry.input_layouts is None, path
        assert isinstance(entry.input_kwarg_layouts["target_token_mask"], Replicate), path
        assert entry.desired_input_kwarg_layouts["target_token_mask"] == Shard(0), path
        assert entry.use_local_output is True, path

    # Only the first block scatters the hidden states; the rest receive an
    # already-sharded stream. `_expand_globs` lets the exact path override the
    # wildcard regardless of order.
    first = plan["transformer_blocks.0"]
    assert isinstance(first.input_kwarg_layouts["hidden_states"], Replicate)
    assert first.desired_input_kwarg_layouts["hidden_states"] == Shard(1)
    assert "hidden_states" not in plan["transformer_blocks.*"].input_kwarg_layouts

    # `norm_out` repeats the modulation once more and is called with three
    # positional arguments, so its mask slice is declared positionally.
    norm_out = plan["norm_out"]
    assert isinstance(norm_out, PrepareModuleInput)
    assert len(norm_out.input_layouts) == 3
    assert norm_out.input_layouts[:2] == (None, None)
    assert isinstance(norm_out.input_layouts[2], Replicate)
    assert norm_out.desired_input_layouts[2] == Shard(0)


def test_the_qwen_image_21_overlay_sums_both_consumers_of_the_timestep_embedding() -> None:
    """``temb`` feeds exactly two modules, and each must reduce its own path.

    Reducing at ``time_text_embed`` instead would scale the already-reduced
    ``modulation`` path by ``tp_size``; reducing at neither leaves
    ``modulation.1.weight`` and ``norm_out.linear.weight`` at this rank's share
    of the tokens with a bit-identical loss.
    """
    from flow_factory.models.qwen_image_21.qwen_image_21 import QwenImage21Adapter

    plan = QwenImage21Adapter.sequence_parallel_plan()

    assert {path for path, style in plan.items() if isinstance(style, SequenceBroadcastOutput)} == {
        "modulation",
        "norm_out.linear",
    }
    assert "time_text_embed" not in plan
    # Single-stream: the joint sequence carries text and image together, so
    # there is no second stream to leave replicated and no `txt_*` entry.
    assert not [path for path in plan if "txt" in path]


@pytest.mark.parametrize("architecture", ["sd3-5", "qwen-image", "qwen-image-2.1"])
def test_every_image_overlay_gathers_the_sequence_back_at_the_output_projection(
    architecture: str,
) -> None:
    """``proj_out`` does two jobs, and both are easy to miss.

    It gathers -- the caller's unpatchify needs every token -- and it holds a
    weight applied once per token on a slice, so each rank accumulates only its
    share of the gradient. Gathering *after* the projection rather than before is
    what makes the transition cheap: it narrows the hidden width to
    ``patch**2 * out_channels``, 24x on SD3.5, 48x on Qwen-Image and 64x on 2.1.
    Dropping the :class:`SequenceBroadcastParameter` half leaves a bit-identical
    loss and a gradient at exactly 1/tp.
    """
    adapter = _image_stream_overlays()[architecture]

    entry = adapter.sequence_parallel_plan()["proj_out"]

    assert isinstance(entry, ComposedParallelStyle)
    kinds = {type(style) for style in entry.styles}
    assert SequenceBroadcastParameter in kinds
    gather = next(style for style in entry.styles if isinstance(style, PrepareModuleOutput))
    assert gather.output_layouts[0] == Shard(1)
    assert isinstance(gather.desired_output_layouts[0], Replicate)
    assert gather.use_local_output is True


@pytest.mark.parametrize("architecture", ["sd3-5", "qwen-image", "qwen-image-2.1"])
def test_every_image_overlay_refines_the_shard_plan_rather_than_extending_it(
    architecture: str,
) -> None:
    """A rowwise entry in the overlay must restate one the shard plan already had.

    The trainer merges the two with override semantics, so an out-projection
    that reduce-scatters instead of all-reducing *is* the same module. A rowwise
    key the shard plan does not have would mean the overlay had started sharding
    a weight on its own, which is not what sequence parallelism is.
    """
    adapter = _image_stream_overlays()[architecture]

    shard_plan = adapter.tensor_parallel_plan()
    overlay = adapter.sequence_parallel_plan()

    reduce_scattered = {
        path for path, style in overlay.items() if isinstance(style, RowwiseParallel)
    }
    assert reduce_scattered
    assert reduce_scattered <= set(shard_plan)
    for path in reduce_scattered:
        assert isinstance(shard_plan[path], RowwiseParallel)
        # The shard plan's entry all-reduces (replicated output); the overlay's
        # reduce-scatters onto the token axis.
        assert shard_plan[path].output_layouts[0].is_replicate()
        assert overlay[path].output_layouts[0] == Shard(1)


@pytest.mark.parametrize("architecture", ["sd3-5", "qwen-image", "qwen-image-2.1"])
def test_every_image_overlay_resolves_against_its_real_tree_and_stays_total(
    architecture: str,
) -> None:
    """The merged plan must name every parameter-owning module of the real class.

    Built on the meta device at two layers, so this stays a CPU test. Two
    properties are at stake: a renamed module makes ``_expand_globs`` raise
    rather than silently shard less, and an activation-only entry on a module
    that owns parameters has to be completed with a parameter style or
    ``apply_tensor_parallel`` rejects the plan.
    """
    pytest.importorskip("diffusers")
    adapter = _image_stream_overlays()[architecture]

    if architecture == "sd3-5":
        from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel

        with torch.device("meta"):
            model = SD3Transformer2DModel(
                num_layers=2,
                attention_head_dim=64,
                num_attention_heads=24,
                caption_projection_dim=1536,
                dual_attention_layers=(0,),
                qk_norm="rms_norm",
            )
    elif architecture == "qwen-image":
        from diffusers.models.transformers.transformer_qwenimage import (
            QwenImageTransformer2DModel,
        )

        with torch.device("meta"):
            model = QwenImageTransformer2DModel(num_layers=2)
    else:
        from diffusers.models.transformers.transformer_qwenimage21 import (
            QwenImage21Transformer2DModel,
        )

        with torch.device("meta"):
            # Qwen/Qwen-Image-2.1 transformer/config.json, with the depth cut.
            model = QwenImage21Transformer2DModel(
                num_layers=2,
                attention_head_dim=128,
                num_attention_heads=32,
                context_in_dim=4096,
                mlp_ratio=3,
            )

    merged = {**adapter.tensor_parallel_plan(), **adapter.sequence_parallel_plan()}
    plan = build_total_plan(model, merged)

    owners = {
        name
        for name, module in model.named_modules()
        if name and any(True for _ in module.named_parameters(recurse=False))
    }
    assert owners - set(plan) == set()
    # Every activation-only entry that landed on a parameter owner was paired
    # with a parameter style rather than replacing one.
    for path, style in plan.items():
        if path in owners and isinstance(style, (PrepareModuleInput, PrepareModuleOutput)):
            pytest.fail(f"{path} owns parameters and was left with an activation-only style")
    # The modulation reducer sits on a `Linear`, so completion has to compose it.
    for path, style in merged.items():
        if isinstance(style, SequenceBroadcastOutput) and path.replace("*", "0") in owners:
            assert isinstance(plan[path.replace("*", "0")], ComposedParallelStyle)


# --------------------------------------------------------------------------- #
# LTX-2 and Z-Image sequence-parallel overlays
# --------------------------------------------------------------------------- #


def test_the_ltx2_overlay_shards_the_audio_axis_too_and_says_why() -> None:
    """Sharding only the video axis would over-reduce the audio tables by ``tp``.

    Each ``LTX2VideoTransformerBlock`` owns four bare modulation tables, two per
    stream, and :class:`SequenceBroadcastParameter` is per *module*. With the
    audio axis replicated every rank has already computed the audio tables'
    gradients whole, so declaring them ``Partial`` multiplies them -- measured at
    exactly 2.000 on two ranks, with a bit-identical loss. Sharding both axes
    makes the one entry on ``transformer_blocks.*`` correct for all four.

    The marker of that decision in the plan is that both streams' entry and exit
    projections carry a boundary, and both streams' out-projections
    reduce-scatter.
    """
    from flow_factory.models.ltx2.ltx2_t2av import LTX2_T2AV_Adapter

    overlay = LTX2_T2AV_Adapter.sequence_parallel_plan()

    for path in ("proj_in", "audio_proj_in"):
        assert isinstance(overlay[path], PrepareModuleOutput)
        assert overlay[path].output_layouts[0].is_replicate()
        assert overlay[path].desired_output_layouts[0] == Shard(1)
    for path in ("proj_out", "audio_proj_out"):
        assert isinstance(overlay[path], ComposedParallelStyle)
        kinds = {type(style) for style in overlay[path].styles}
        assert SequenceBroadcastParameter in kinds
        gather = next(s for s in overlay[path].styles if isinstance(s, PrepareModuleOutput))
        assert gather.output_layouts[0] == Shard(1)
        assert gather.desired_output_layouts[0].is_replicate()

    reduce_scattered = {p for p, s in overlay.items() if isinstance(s, RowwiseParallel)}
    assert reduce_scattered == {
        f"transformer_blocks.*.{stack}.to_out.0" for stack in _LTX2_ATTENTION_STACKS
    } | {"transformer_blocks.*.ff.net.2", "transformer_blocks.*.audio_ff.net.2"}
    assert all(overlay[path].output_layouts[0] == Shard(1) for path in reduce_scattered)
    assert isinstance(overlay["transformer_blocks.*"], SequenceBroadcastParameter)
    assert isinstance(overlay[""], SequenceBroadcastParameter)


def test_the_ltx2_overlay_matches_each_stacks_own_call_convention() -> None:
    """One layout per argument the module is *actually* called with.

    LTX-2's block calls its six attention stacks three different ways, and a
    layout tuple that does not match raises "module inputs and input_layouts
    should have same length". ``attn1``/``audio_attn1`` take only keywords;
    ``attn2``/``audio_attn2``/``ff``/``audio_ff`` one positional argument; and
    the two cross-modal stacks one positional argument *and* a token-sharded
    ``encoder_hidden_states``, because audio-to-video takes video queries with
    audio keys and video-to-audio the reverse.
    """
    from flow_factory.models.ltx2.ltx2_t2av import LTX2_T2AV_Adapter

    overlay = LTX2_T2AV_Adapter.sequence_parallel_plan()

    for stack in ("attn1", "audio_attn1"):
        entry = overlay[f"transformer_blocks.*.{stack}"]
        assert entry.input_layouts is None
        assert entry.input_kwarg_layouts == {"hidden_states": Shard(1)}
        assert entry.desired_input_kwarg_layouts["hidden_states"].is_replicate()

    for name in ("attn2", "audio_attn2", "ff", "audio_ff"):
        entry = overlay[f"transformer_blocks.*.{name}"]
        assert entry.input_layouts == (Shard(1),)
        assert entry.desired_input_layouts[0].is_replicate()
        assert not entry.input_kwarg_layouts

    for stack in ("audio_to_video_attn", "video_to_audio_attn"):
        entry = overlay[f"transformer_blocks.*.{stack}"]
        assert entry.input_layouts == (Shard(1),)
        assert entry.input_kwarg_layouts == {"encoder_hidden_states": Shard(1)}
        assert entry.desired_input_kwarg_layouts["encoder_hidden_states"].is_replicate()

    assert all(
        style.use_local_output
        for path, style in overlay.items()
        if isinstance(style, PrepareModuleInput)
    )


def test_the_ltx2_overlay_scatters_at_the_projections_not_at_the_first_block() -> None:
    """Wan scatters at ``blocks.0``; LTX-2 cannot, and the reason is the plan's.

    The root calls each block with keyword arguments on the eager path and with
    26 positional ones under ``_gradient_checkpointing_func``, so a
    ``PrepareModuleInput`` written for either convention raises under the other.
    ``proj_in``/``proj_out`` take one positional argument either way, and
    gathering after ``proj_out`` is also 32x cheaper than gathering after the
    last block -- the projection narrows ``inner_dim`` to ``out_channels``.
    """
    from flow_factory.models.ltx2.ltx2_t2av import LTX2_T2AV_Adapter

    overlay = LTX2_T2AV_Adapter.sequence_parallel_plan()

    assert not any(
        path == "transformer_blocks.0" or path.startswith("transformer_blocks.0.")
        for path in overlay
    )
    boundaries = {
        path
        for path, style in overlay.items()
        if isinstance(style, PrepareModuleOutput)
        or (
            isinstance(style, ComposedParallelStyle)
            and any(isinstance(s, PrepareModuleOutput) for s in style.styles)
        )
    }
    assert boundaries == {"proj_in", "audio_proj_in", "proj_out", "audio_proj_out"}


def test_the_ltx2_overlay_sums_every_conditioning_head() -> None:
    """Six modules emit values broadcast over tokens, and all six need reducing.

    Omitting them leaves a bit-identical loss and the whole conditioning stack
    at this rank's share of its gradient -- measured at 0.496-0.522 of the true
    value on two ranks. ``time_embed`` and ``audio_time_embed`` return
    ``(temb, embedded_timestep)`` and *both* are consumed -- ``temb`` by every
    block, ``embedded_timestep`` by the root's output modulation -- so both
    indices are named; the four cross-modal heads only use their first output.
    """
    from flow_factory.models.ltx2.ltx2_t2av import LTX2_T2AV_Adapter

    overlay = LTX2_T2AV_Adapter.sequence_parallel_plan()

    reducers = {p: s for p, s in overlay.items() if isinstance(s, SequenceBroadcastOutput)}
    assert set(reducers) == {
        "time_embed",
        "audio_time_embed",
        "av_cross_attn_video_scale_shift",
        "av_cross_attn_audio_scale_shift",
        "av_cross_attn_video_a2v_gate",
        "av_cross_attn_audio_v2a_gate",
    }
    assert reducers["time_embed"].output_indices == (0, 1)
    assert reducers["audio_time_embed"].output_indices == (0, 1)
    assert all(
        reducers[path].output_indices == (0,)
        for path in reducers
        if path.startswith("av_cross_attn_")
    )


def test_the_two_ltx2_overlays_are_the_same_mapping() -> None:
    """Both adapters shard the same ``LTX2VideoTransformer3DModel``."""
    from flow_factory.models.ltx2.ltx2_i2av import LTX2_I2AV_Adapter
    from flow_factory.models.ltx2.ltx2_t2av import LTX2_T2AV_Adapter

    t2av = LTX2_T2AV_Adapter.sequence_parallel_plan()
    i2av = LTX2_I2AV_Adapter.sequence_parallel_plan()

    assert set(t2av) == set(i2av)
    assert all(type(t2av[key]) is type(i2av[key]) for key in t2av)


def test_the_z_image_overlay_shards_only_the_unified_stack() -> None:
    """``noise_refiner`` and ``context_refiner`` run before the sequence exists.

    They operate on two separate ragged sequences that
    ``_build_unified_sequence`` has not concatenated yet, so sharding either
    would need its own scatter/gather pair for 4 of 34 blocks. Leaving them
    token-replicated is also what makes the root's
    ``x_pad_token``/``cap_pad_token`` need no entry: they are consumed upstream
    of the scatter, where every rank's gradient for them is already whole.
    """
    from flow_factory.models.z_image.z_image import ZImageAdapter

    overlay = ZImageAdapter.sequence_parallel_plan()

    assert all(path.startswith(("layers.", "all_final_layer.")) for path in overlay), sorted(
        overlay
    )
    assert "" not in overlay
    scatter = overlay["layers.0"]
    # Seven positional arguments on both the eager and the checkpointed path;
    # only the hidden states are token-sharded, and the full-length rotary table
    # and attention mask stay untouched because attention gathers.
    assert len(scatter.input_layouts) == 7
    assert scatter.input_layouts[0].is_replicate()
    assert scatter.desired_input_layouts[0] == Shard(1)
    assert set(scatter.input_layouts[1:]) == {None}
    assert set(scatter.desired_input_layouts[1:]) == {None}


def test_the_z_image_overlay_reduces_the_modulation_at_the_block_not_the_embedder() -> None:
    """The placement Wan's plan shape suggests is wrong here, and silently so.

    ``adaLN_modulation.0`` emits four ``[B, 1, dim]`` vectors broadcast over
    every token, so the gradient at its output -- and therefore its own weight
    gradient -- is a partial sum. Reducing it at the ``nn.Sequential``'s output
    fixes the weight and everything upstream in one hook. Reducing at
    ``t_embedder`` instead would also sum the contribution of
    ``noise_refiner``'s two modulated blocks, which run outside the sharded
    region and whose share is already whole on every rank, multiplying it by
    ``tp_size``.
    """
    from flow_factory.models.z_image.z_image import ZImageAdapter

    overlay = ZImageAdapter.sequence_parallel_plan()

    reducers = {p for p, s in overlay.items() if isinstance(s, SequenceBroadcastOutput)}
    assert reducers == {"layers.*.adaLN_modulation", "all_final_layer.*.adaLN_modulation"}
    assert "t_embedder" not in overlay
    assert not any(path.startswith("noise_refiner") for path in overlay)


def test_the_z_image_overlay_sums_all_four_affine_block_norms() -> None:
    """Z-Image's norms are affine, unlike Wan's -- so none of them can be skipped.

    ``RMSNorm(dim, eps=norm_eps)`` defaults ``elementwise_affine=True``, so each
    of the four owns a ``[dim]`` weight applied per token and accumulates only
    this rank's share of its gradient. The per-head ``norm_q``/``norm_k`` keep
    their tensor-parallel style instead: the gather at ``attention`` means their
    input is token-whole, exactly as it is without this overlay.
    """
    from flow_factory.models.z_image.z_image import ZImageAdapter

    overlay = ZImageAdapter.sequence_parallel_plan()

    broadcast = {p for p, s in overlay.items() if isinstance(s, SequenceBroadcastParameter)}
    assert broadcast == {
        "layers.*.attention_norm1",
        "layers.*.attention_norm2",
        "layers.*.ffn_norm1",
        "layers.*.ffn_norm2",
        "all_final_layer.*.linear",
    }
    assert not any("norm_q" in path or "norm_k" in path for path in overlay)


def test_the_z_image_overlay_gathers_after_the_final_projection() -> None:
    """60x cheaper than gathering after the last block, and it has to be there.

    ``all_final_layer.linear`` narrows 3840 -> ``patch**2 * f_patch *
    out_channels`` = 64, and ``unpatchify`` immediately afterwards slices each
    sample at its *global* image length, so the gather cannot be deferred past
    it either.
    """
    from flow_factory.models.z_image.z_image import ZImageAdapter

    gather = ZImageAdapter.sequence_parallel_plan()["all_final_layer.*"]

    assert isinstance(gather, PrepareModuleOutput)
    assert gather.output_layouts[0] == Shard(1)
    assert gather.desired_output_layouts[0].is_replicate()
    assert gather.use_local_output is True


@pytest.mark.parametrize("architecture", ["ltx2", "z-image"])
def test_the_new_overlays_refine_the_shard_plan_rather_than_extending_it(
    architecture: str,
) -> None:
    """Every rowwise entry in an overlay restates one the shard plan already had.

    The trainer merges the two with override semantics, so an out-projection
    that reduce-scatters instead of all-reducing *is* the same module. A rowwise
    key the shard plan does not have would mean the overlay had started sharding
    a weight on its own, which is not what sequence parallelism is.
    """
    if architecture == "ltx2":
        from flow_factory.models.ltx2.ltx2_t2av import LTX2_T2AV_Adapter as adapter
    else:
        from flow_factory.models.z_image.z_image import ZImageAdapter as adapter

    shard_plan = adapter.tensor_parallel_plan()
    overlay = adapter.sequence_parallel_plan()

    reduce_scattered = {p for p, s in overlay.items() if isinstance(s, RowwiseParallel)}
    assert reduce_scattered
    assert reduce_scattered <= set(shard_plan)
    for path in reduce_scattered:
        assert shard_plan[path].output_layouts[0].is_replicate()
        assert overlay[path].output_layouts[0] == Shard(1)


@pytest.mark.parametrize("architecture", ["ltx2", "z-image"])
def test_the_new_overlays_resolve_against_their_real_trees_and_stay_total(
    architecture: str,
) -> None:
    """The merged plan must name every parameter-owning module of the real class.

    Two properties are at stake: a renamed module makes ``_expand_globs`` raise
    rather than silently shard less, and an activation-only entry on a module
    that owns parameters has to be completed with a parameter style or
    ``apply_tensor_parallel`` rejects the plan. The root is the case that makes
    the difference here -- LTX-2 keeps two bare modulation tables there and
    names it explicitly, Z-Image keeps two pad tokens there and deliberately
    does not.
    """
    pytest.importorskip("diffusers")
    if architecture == "ltx2":
        from diffusers.models.transformers.transformer_ltx2 import LTX2VideoTransformer3DModel
        from flow_factory.models.ltx2.ltx2_t2av import LTX2_T2AV_Adapter as adapter

        with torch.device("meta"):
            model = LTX2VideoTransformer3DModel(
                num_layers=2,
                num_attention_heads=2,
                attention_head_dim=8,
                audio_num_attention_heads=2,
                audio_attention_head_dim=4,
            )
    else:
        from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel
        from flow_factory.models.z_image.z_image import ZImageAdapter as adapter

        with torch.device("meta"):
            # Tongyi-MAI/Z-Image transformer/config.json with the depths cut.
            model = ZImageTransformer2DModel(
                all_patch_size=(2,),
                all_f_patch_size=(1,),
                in_channels=16,
                dim=3840,
                n_layers=2,
                n_refiner_layers=1,
                n_heads=30,
                n_kv_heads=30,
                norm_eps=1e-5,
                qk_norm=True,
                cap_feat_dim=2560,
                siglip_feat_dim=None,
                rope_theta=256.0,
                axes_dims=[32, 48, 48],
                axes_lens=[1536, 512, 512],
            )

    merged = {**adapter.tensor_parallel_plan(), **adapter.sequence_parallel_plan()}
    plan = build_total_plan(model, merged)

    owners = {
        name
        for name, module in model.named_modules()
        if any(True for _ in module.named_parameters(recurse=False))
    }
    assert owners - set(plan) == set()
    for path, style in plan.items():
        if path in owners and isinstance(style, (PrepareModuleInput, PrepareModuleOutput)):
            pytest.fail(f"{path} owns parameters and was left with an activation-only style")
    # The root owns bare modulation tables on LTX-2 and pad tokens on Z-Image,
    # so both depend on completion reaching `""` -- but only LTX-2's are applied
    # per token, which is why only LTX-2 names it.
    assert "" in owners
    assert isinstance(
        plan[""],
        SequenceBroadcastParameter if architecture == "ltx2" else ReplicateParallel,
    )


def test_the_sensenova_overlay_needs_no_conditioning_reduction() -> None:
    """The entry every other architecture here gets wrong, SenseNova does not need.

    ``fm_modules.timestep_embedder`` and ``noise_scale_embedder`` are evaluated
    *per token* -- the pipeline expands the timestep to ``image_token_num`` and
    adds the result into the embeddings before the stack -- and the live tree has
    no AdaLN at all, since ``SimpleMLPAdaLN`` is reachable only through the deep
    ``fm_head`` that ``use_pixel_head=True`` does not build. Nor does any layer
    or root own a bare ``nn.Parameter``. So the overlay is boundaries, gathers,
    reduce-scatters and norm weights, and nothing else.
    """
    from flow_factory.models.sensenova.sensenova import SenseNovaAdapter

    overlay = SenseNovaAdapter.sequence_parallel_plan()

    assert not any(isinstance(style, SequenceBroadcastOutput) for style in overlay.values())
    assert "" not in overlay
    assert all(path.startswith("model.language_model.model.") for path in overlay), sorted(overlay)
    assert not any("fm_modules" in path or "embedder" in path for path in overlay)


def test_the_sensenova_overlay_reaches_self_attention_by_keyword() -> None:
    """``forward_und``/``forward_gen`` pass ``hidden_states=`` with no positionals.

    A positional layout tuple raises "module inputs and input_layouts should
    have same length" there, because ``inputs`` is empty. The decoder layer
    itself is the opposite case -- exactly one positional argument on both the
    eager path and under ``GradientCheckpointingLayer``, which binds the keywords
    into a ``partial`` and checkpoints only ``*args``.
    """
    from flow_factory.models.sensenova.sensenova import SenseNovaAdapter

    overlay = SenseNovaAdapter.sequence_parallel_plan()
    prefix = "model.language_model.model.layers"

    attention = overlay[f"{prefix}.*.self_attn"]
    assert attention.input_layouts is None
    assert attention.input_kwarg_layouts == {"hidden_states": Shard(1)}
    assert attention.desired_input_kwarg_layouts["hidden_states"].is_replicate()

    scatter = overlay[f"{prefix}.0"]
    assert scatter.input_layouts == (Replicate(),)
    assert scatter.desired_input_layouts == (Shard(1),)
    assert not scatter.input_kwarg_layouts

    for branch in ("", "_mot_gen"):
        feed_forward = overlay[f"{prefix}.*.mlp{branch}"]
        assert feed_forward.input_layouts == (Shard(1),)
        assert feed_forward.desired_input_layouts[0].is_replicate()


def test_the_sensenova_overlay_covers_both_branches_everywhere() -> None:
    """Both decoder branches run on the same sharded sequence, so both need every entry.

    The prefix forward uses the understanding branch and the denoising step the
    generation branch, but ``layers.0``'s scatter applies to both, so a plan that
    reduce-scattered only ``_mot_gen`` would leave the prefix's residual stream
    at a length its out-projections no longer produce.
    """
    from flow_factory.models.sensenova.sensenova import SenseNovaAdapter

    shard_plan = SenseNovaAdapter.tensor_parallel_plan()
    overlay = SenseNovaAdapter.sequence_parallel_plan()
    prefix = "model.language_model.model.layers.*"

    reduce_scattered = {p for p, s in overlay.items() if isinstance(s, RowwiseParallel)}
    assert reduce_scattered == {
        f"{prefix}.self_attn.o_proj{branch}" for branch in ("", "_mot_gen")
    } | {f"{prefix}.mlp{branch}.down_proj" for branch in ("", "_mot_gen")}
    assert reduce_scattered <= set(shard_plan)
    for path in reduce_scattered:
        assert shard_plan[path].output_layouts[0].is_replicate()
        assert overlay[path].output_layouts[0] == Shard(1)

    broadcast = {p for p, s in overlay.items() if isinstance(s, SequenceBroadcastParameter)}
    assert broadcast == {
        f"{prefix}.{norm}{branch}"
        for norm in ("input_layernorm", "post_attention_layernorm")
        for branch in ("", "_mot_gen")
    }
    # The per-head qk-norms keep their tensor-parallel style: the gather at
    # `self_attn` leaves their input head-sharded and token-whole.
    assert not any("q_norm" in path or "k_norm" in path for path in overlay)


def test_the_sensenova_overlay_gathers_at_the_generation_norm_at_full_width() -> None:
    """``norm_mot_gen`` is the exit, and SenseNova has nothing narrow to hide behind.

    With the pure-generation token mix the adapter always passes,
    ``Qwen3Model.forward`` takes the ``norm_mot_gen`` branch wholesale and
    returns its output as ``last_hidden_state``, so that is the last operation on
    the sequence. The gather is therefore at the full 4096 hidden width, unlike
    Wan's 5120 -> 64 and Z-Image's 3840 -> 64 output projections. The
    understanding branch's ``norm`` is deliberately *not* gathered: the prefix
    forward runs under ``no_grad`` and every caller discards its hidden states,
    keeping only the KV cache, which ``self_attn`` fills after the gather.
    """
    from flow_factory.models.sensenova.sensenova import SenseNovaAdapter

    overlay = SenseNovaAdapter.sequence_parallel_plan()

    entry = overlay["model.language_model.model.norm_mot_gen"]
    assert isinstance(entry, ComposedParallelStyle)
    assert any(isinstance(style, SequenceBroadcastParameter) for style in entry.styles)
    gather = next(style for style in entry.styles if isinstance(style, PrepareModuleOutput))
    assert gather.output_layouts[0] == Shard(1)
    assert gather.desired_output_layouts[0].is_replicate()
    assert "model.language_model.model.norm" not in overlay


@pytest.mark.parametrize("architecture", ["sd3-5", "qwen-image", "qwen-image-2.1"])
def test_completion_does_not_demote_the_output_projection_to_replicated(
    architecture: str,
) -> None:
    """``build_total_plan`` must leave the one entry whose absence is silent alone.

    Completion exists to pair a parameter style with an activation-only entry,
    and ``proj_out`` already carries both. If it ever started overwriting a
    composed entry instead of skipping it, the gather would survive and the
    :class:`SequenceBroadcastParameter` would not -- a forward that is correct
    on every rank, a bit-identical loss, and a weight gradient at this rank's
    share of the tokens. On a packed sequence that share need not even be 1/tp:
    a rank holding no loss-bearing token would get exactly zero.
    """
    pytest.importorskip("diffusers")
    adapter = _image_stream_overlays()[architecture]

    class _Stub(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj_out = nn.Linear(8, 4)

    overlay = adapter.sequence_parallel_plan()
    completed = build_total_plan(_Stub(), {"proj_out": overlay["proj_out"]})

    entry = completed["proj_out"]
    assert isinstance(entry, ComposedParallelStyle)
    assert any(isinstance(style, SequenceBroadcastParameter) for style in entry.styles)
    assert not any(isinstance(style, ReplicateParallel) for style in entry.styles)


def test_the_sensenova_overlay_resolves_against_the_real_tree_and_stays_total() -> None:
    """The merged plan must name every parameter-owning module of the real wrapper.

    Built on the meta device at two layers, so this stays a CPU test. SenseNova
    is the one architecture here whose component is not a diffusers transformer,
    so this is also where the ``model.language_model.model.`` prefix is checked
    against the live tree rather than against the config.
    """
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

    merged = {
        **SenseNovaAdapter.tensor_parallel_plan(),
        **SenseNovaAdapter.sequence_parallel_plan(),
    }
    plan = build_total_plan(model, merged)

    owners = {
        name
        for name, module in model.named_modules()
        if any(True for _ in module.named_parameters(recurse=False))
    }
    assert owners - set(plan) == set()
    for path, style in plan.items():
        if path in owners and isinstance(style, (PrepareModuleInput, PrepareModuleOutput)):
            pytest.fail(f"{path} owns parameters and was left with an activation-only style")
    # `norm_mot_gen` owns a weight *and* sits on the gather boundary, so the
    # overlay has to compose the two rather than pick one.
    assert isinstance(plan["model.language_model.model.norm_mot_gen"], ComposedParallelStyle)
    # No bare parameters anywhere, which is why the root takes the plain
    # replicated completion and not a broadcast style.
    assert "" not in owners
