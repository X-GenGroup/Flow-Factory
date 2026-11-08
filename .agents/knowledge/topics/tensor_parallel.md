# Tensor Parallelism

Flow-Factory shards a trainable denoiser across a `tp` mesh dimension and composes that with FSDP2
over `dp_shard`. The mechanism lives in `src/flow_factory/models/tensor_parallel.py`; each adapter
declares *what* to shard through `BaseAdapter.tensor_parallel_plan()`.

The shared `contracts/topology.py::ParallelTopology` owns dimension products and logical sample
scopes. See [parallel topology](../../../guidance/parallelism.md) for canonical rank order,
runtime placement validation and checkpoint-schema compatibility. The separate
[Wan CP SFT capabilities](../../../guidance/context_parallel.md) distinguish CP-only from
qualified TP/SP+CP composition; the latter preserves independent TP and CP peers.

Enable it with a top-level block; absent, nothing changes:

```yaml
parallelism:
  tp_size: 8        # ranks per tensor-parallel group
  dp_shard_size: 1  # optional; derived from the launched world otherwise
```

## Why tensor parallelism touches data parallelism

FSDP all-gathers a parameter before the layer that needs it, so after the gather every rank holds
the whole weight and the rank axis *is* the data axis. Tensor parallelism never reconstitutes the
parameter: each rank computes with its own slice and the ranks exchange **activations**. That only
produces the right answer if every rank in the group is working on the *same* sample.

So a tensor-parallel group is **one data-parallel replica**, and the launched world is
`replicas * tp_size`. Tensor parallelism does not add capacity for more samples; it spends ranks
that would otherwise have been replicas. Every piece of sample geometry has to count replicas:

| Concern | Entry point |
|---|---|
| Runtime geometry (`world_size`, `rank`) | `utils/dist.data_parallel_geometry(accelerator)` |
| Config-time geometry, before any `Accelerator` | `Arguments.data_parallel_world_size()` |
| Sample-level reductions | `utils/dist.reduce_across_replicas`, `all_reduce_across_replicas_` |
| Group-scoped gathers | `utils/group_coordinator.GroupCoordinator.collective_group` |

A reduction over the launched world instead of the replicas counts every sample `tp_size` times.
Sometimes that cancels (a mean over duplicated values is unchanged) and sometimes it does not (a
count, a group-sum into a sigmoid). Do not reason case by case about which: scope the collective.

`data_parallel_geometry` reads `parallelism_config.data_parallel_size` and then indexes
`mesh["dp"]`. It must **not** test `"dp" in mesh.mesh_dim_names`: Accelerate builds the joint `dp`
dimension with `DeviceMesh._flatten`, which records it in the flatten mapping and leaves it out of
`mesh_dim_names` — the root names stay `('dp_shard', 'tp')`, so that test is false for every real
mesh and silently falls back to the launched world. `tests/distributed/test_data_parallel_geometry.py`
runs on real process groups for exactly this reason; a stub mesh cannot catch it.

It reads the size only when the plan reserves ranks for model parallelism
(`non_data_parallel_size > 1`); otherwise the launched world is the replica set, as with no plan at
all. The opposite order was a real regression: an accelerate config carrying a
`parallelism_config` section with no sizes, or `--use_parallelism_config`, yields an all-ones plan,
and under FSDP2 Accelerate never validates it — `_validate_accelerator` returns early because
`multi_device` excludes FSDP — so `data_parallel_size` stays 1 and every rank drew the same samples.
Measured with a real two-rank FSDP2 launch. A two-rank measurement had earlier cleared it; under DDP
Accelerate does rewrite the plan to `dp_replicate_size = num_processes`, which is why.

## The plan

`tensor_parallel_plan()` is a classmethod returning `{module path: ParallelStyle}`, where `*` stands
for one path segment. The backend validator reads it before any adapter is constructed, so a run
requesting `tp_size > 1` against an empty plan is rejected rather than silently training unsharded.

| Style | Use |
|---|---|
| `ColwiseParallel()` | q/k/v, FFN up-projections. Shards weight dim 0; output `Shard(-1)`. |
| `RowwiseParallel()` | attention out-projection, FFN down-projection. Shards weight dim 1; all-reduces. |
| `ShardedInputReplicatedWeight()` | per-head qk-norm: replicated weight, head-sharded input. |
| `ReplicateParallel()` | not written in a plan — the applier completes with it. |

Four properties were established on real GPUs and are what the module exists to preserve.

**1. The plan must be total.** Any parameter the plan does not name stays a plain tensor (pure TP) or
becomes a dp-only DTensor (TP+FSDP), and gradient clipping then raises — "mixed torch.Tensor and
DTensor" without FSDP, "All operands in aten.stack must have the same mesh" with it. Gradients are
numerically correct either way; this is a dispatch constraint, which is why `build_total_plan`
completes every plan with `ReplicateParallel` over every parameter-owning module.

**2. A replicated weight whose input is a shard needs its own style.** Per-head qk-norm is the case:
the weight is `[attention_head_dim]` and shared by every head, but column-parallel q/k leave a rank
holding only its own heads, so the weight gradient is a partial sum. Treating it as plainly
replicated is silently wrong and **invisible in the total gradient norm** — those gradients are
~1e-5 against a norm of ~1. The style lives in the autograd graph rather than in a parameter hook,
because `fully_shard` replaces parameter objects and a `register_post_accumulate_grad_hook` is then
lost without a word (it passed at dp=1 and failed at dp=2).

**3. torch's own styles unfreeze parameters.** `ColwiseParallel._partition_linear_fn` rebuilds each
parameter as `nn.Parameter(distribute_tensor(...))` without forwarding `requires_grad`, which
defaults to True. On a LoRA run that silently turns every sharded `base_layer` back into a trainable
weight — 28 gradients at tp=1, 42 at tp=2. `apply_tensor_parallel` snapshots trainability before
`parallelize_module` and restores it after. **A parity test must compare the set of parameters that
have gradients, not only their values.**

**4. Packed activations need their own layout.** A fused `Linear` cannot directly use a local
`Shard(0)` output as `[q_rank, k_rank, v_rank]`: contiguous weight rows may contain whole blocks.
Native `_StridedShard` alone asserts during `full_tensor()` on the supported Torch2.10 tuple.
`PackedColwiseParallel` and `PackedRowwiseParallel` retain ordinary canonical 2-D weight shards
and exchange only activations using native differentiable all-to-all. The row exchange is the
inverse of the column exchange. The LoRA owner preserves these styles on column B and row A.
Model-only state, optimizer state and names remain native; no persistent permutation or save-time
mapping exists. This adds communication and requires an explicit family plan. Existing unconverted
packed modules stay replicated until their model geometry and gradients are qualified. See
[packed projection contracts](../../../guidance/tensor_parallel.md#canonical-packed-projections).

**5. One adapter serves a family whose members differ.** The plan is a `classmethod` read before any
weight is loaded, so it cannot branch on the loaded config — and a pattern matching no module is
rejected, which is what keeps a renamed layer loud. Those two collide where a family is not uniform:
Wan's I2V adapter covers Wan2.1-I2V, whose attention carries a second key/value stream for the CLIP
image conditioning (`added_kv_proj_dim=5120`), and Wan2.2-I2V, which has none (`None` — checked
against the published `Wan2.2-I2V-A14B` and `Wan2.2-TI2V-5B` configs). Naming those three modules
unconditionally meant `tp_size > 1` on the shipped `examples/grpo/*/wan22/i2v.yaml` died in glob
expansion *after both 14B transformers were loaded*.

`ConditionalEntry(style, group)` marks an entry whose modules only some checkpoints have. Absence is
allowed; the `group` carries the condition, and within one group either every pattern matches or
none does — so a **typo** in one of three is still rejected, on the checkpoint that has them. Reach
for it only for a genuine architecture split, never to quiet a pattern you have not verified.

**LoRA targets constrain the plan, not just the other way round.** `derive_lora_plan` can derive an
adapter pairing for `ColwiseParallel` and `RowwiseParallel` and for nothing else. An activation
transform relocated to `base_layer` hooks one arm of `base_layer(x) + lora_B(lora_A(x))`; a
`lora_B -> ColwiseParallel` against a full-width base arm halves the adapter arm; and a
`lora_A -> ReplicateParallel` inside the sequence-parallel region accumulates only this rank's token
share — the silent `1/tp_size` gradient. All three are now refused at plan construction, naming the
module, rather than surfacing mid-forward or not at all.

## Where the plan is applied

`BaseTrainer._apply_tensor_parallel`, between `_apply_backend_checkpointing_constraints()` and
`self.optimizer = self._init_optimizer()`. That position is the only correct one:

- weights are loaded, weight-only resume is applied, LoRA is injected, dtypes are aligned;
- the optimizer does not exist yet, so Accelerate's `data_ptr` remapping during `prepare` still
  works — building it first yields "a parameter in the optimizer couldn't be switched to its sharded
  version";
- `prepare` has not run, so `fully_shard` sees DTensor parameters and composes the two meshes;
- EMA, reference and algorithm snapshots are allocated later, so they clone already-sharded
  parameters.

Two things the applier must do and a reader would not expect:

- **peel the PEFT wrapper.** After LoRA injection `get_component` returns a `PeftModel` whose
  children sit under `base_model.model.`, so every plan pattern matches nothing and the run is
  silently unsharded. `peel_peft_wrapper` resolves to the base model, which still carries the
  injected `lora.Linear` modules the LoRA plan targets.
- **rebind the variant registry.** Sharding replaces every planned parameter object, leaving the
  registry's ownership records pointing at tensors the modules no longer hold.

LoRA is sharded by `derive_lora_plan`, which targets `.base_layer` (torch's styles accept only
`nn.Linear`/`nn.Embedding`) and exploits `lora_B` being linear so a `Partial` placement passes
through it: colwise base → `lora_A` replicated + `lora_B` colwise; rowwise base → `lora_A` rowwise +
`lora_B` replicated, so base and adapter outputs merge before a single all-reduce.

## What it costs and what it buys

**On a fixed world, tensor parallelism never reduces resident parameter memory below
FSDP-only sharding.** Per-rank parameter bytes are `(covered + uncovered * tp_size) / world_size`,
because FSDP splits everything across the ranks it is given while tensor parallelism splits only
what its plan names -- and it takes those ranks away from FSDP. That expression equals the
FSDP-only `total / world_size` when the plan is total and is strictly larger otherwise. Measured on
MiniMax-H3, whose plan covers 7.707B of 33.123B parameters (23.3%):

| dp_shard | tp | per-rank transformer params | vs tp=1 |
|---|---|---|---|
| 8 | 1 | 7.71 GB | baseline |
| 4 | 2 | 13.63 GB | 1.77x |
| 2 | 4 | 25.46 GB | 3.30x |
| 1 | 8 | 49.14 GB | 6.37x, and rejected |

End-to-end on 8x80GB, H3 Flow-GRPO with a PickScore reward at 768x1344 and 124 frames: `dp=8, tp=1`
completes with a 72.7 GB peak, `tp=2` and `tp=4` both OOM, and `tp=8` is rejected before it starts.
**Tensor parallelism is the wrong tool for H3**; the earlier probe numbers showing it saving memory
were taken at `dp=1`, where there is no FSDP to compare against.

**The steady state is not what stops H3, though: applying the plan has a startup bound of its own,
and that is what fails.** `distribute_tensor` moves each parameter to the GPU before it shards it,
and the plan is applied before `prepare`, so the target arrives with FSDP not yet run -- and under
FSDP2 its master weights are fp32. Each rank must hold `P_master * (c / t + (1 - c))` before
`fully_shard` can divide it, where FSDP alone never holds more than its own `1 / W` share, because
`fully_shard` moves only that share. For H3 -- 33.12B fp32 parameters at 23.3% coverage -- that is
**117 GB at `tp=2` and 109 GB at `tp=4`**, past an 80 GB card before anything else is counted, and
its 62 GiB bf16 text encoder is still resident from preprocessing at that point. Every H3
tensor-parallel attempt on record failed there, inside `apply_tensor_parallel` and before the first
step: `tp=2` and `tp=4`, with and without the overlay, 128x224 up to 768x1344, and all six
algorithms of the campaign matrix. It is why the peak never moved with the resolution or with the
overlay: no activation existed yet. An earlier version of this section read that same 80,995 MiB as
the steady-state parameter penalty, and inferred from a `tp=1` resolution sweep that activations are
a minor term for H3. Both readings were wrong. The sweep compared the text encoder's 65.2 GB
preprocessing plateau, which is the run's peak at 128x224, with the 72.7 GB training peak at
768x1344; training itself went from 34.7 to 72.7 GB. **FSDP2 without tensor parallelism trains
every H3 algorithm at the campaign geometry, and it is the configuration to use.** Lifting the bound
means applying the plan before the weights are materialized -- meta-device initialization followed
by sharded loading, the order torchtitan uses -- which is a loading change rather than a plan
change. The bound is per model and worth computing before enabling tensor parallelism on a large,
low-coverage one: at `tp=2` it is 6.0 GB for SD3.5-Medium (2.24B, 65.1%) against H3's 117 GB.

Sequence parallelism changes neither number. With the overlay enabled H3 fails at the same place, at
the same 80,995 MiB, at `tp=2`, at `tp=4` and at a sixteen-times-smaller resolution. Coverage still
decides whether either technique is worth running once a model can start; it is not a detail.

So the cases where it does pay are specific:

* **a plan that covers most of the model.** Z-Image covers 97.8%, SenseNova 92.4%, Qwen-Image 66.5%,
  SD3.5 65.1%. For those the resident penalty is small or nil and the activation saving is real.
* **the all-gather peak.** FSDP materializes a whole layer's parameters during its forward; tensor
  parallelism never does. For a single very wide layer that transient can be the binding constraint
  even when resident storage is not.
* **activation memory at long sequences**, but only where activations actually dominate. Measured on
  Wan at 75,600 video tokens, `tp=2` cut the peak just 8.6% (53.2 -> 48.6 GB) even with the whole
  block sharded, and the saving was exactly the halved feed-forward intermediate. The remaining
  ~47 GB is per-token full-width fp32 modulation arithmetic, which no plan of this shape can
  shard -- sequence parallelism is what would. At a short sequence the same plan saves 37%, so for
  this architecture the benefit *shrinks* as the sequence grows.

**Under FSDP2, `tp_size <= world_size / 2`.** FSDP needs a dimension of its own, so an
all-tensor-parallel world is rejected; `_build_parallelism_config` says so directly rather than
letting Accelerate's message blame expert parallelism. It also derives `dp_shard_size` from the
launched world, since Accelerate does not: `tp_size: 4` alone on eight ranks would otherwise be read
as a plan for four.

## Per-model plan coverage

Every registered model declares a plan, and two of them declare it empty. Coverage is the fraction
of the trainable denoiser the plan shards; the rest is replicated and is what the table above
charges `tp_size` for.

| Model | Coverage | Max tp | Note |
|---|---|---|---|
| z-image | 97.8% | **2** | 30 heads, so only tp=2 divides |
| sensenova | 92.4% | 8 | 8 kv heads, read from `llm_config` and enforced before the weights load |
| bagel | 87.4% of a layer | unbounded | feed-forward only; attention needs the vendored packed buffer split |
| qwen-image, qwen-image-edit-plus | 66.5% | 8 | |
| sd3-5 | 65.1% | 8 | |
| flux1, flux1-kontext | 99.608% | 8 | dual/single attention, MLP and AdaLN weights; canonical 1:4 packed output. Opt-in image SP gathers before the first single block; text and single-stream token activations remain replicated |
| flux3 | 29.3% | 8 | |
| flux2, flux2-klein | **declined** | — | empty plan; see below |
| minimax-h3 (3 variants) | 23.3% | 8 | packed SwiGLU and full-width modulation stay replicated |
| wan2_t2v, wan2_i2v | **98.4%** | 8 | whole block; across-head qk-norm needs its own style |
| ltx2_t2av, ltx2_i2av | **98.16%** | 8 | whole block; across-head qk-norm plus sharded rotary tables |

Wan and LTX-2 build qk-norm over `num_heads * head_dim` and apply it before splitting heads, so
column-parallel q/k/v splits the *normalized* axis and makes the forward wrong, not just the
gradient -- it raises on the normalized shape rather than failing quietly.
`ShardedAcrossHeadRMSNorm` completes that case by sharding the weight alongside the projection and
all-reducing the mean square, a `[batch, seq, 1]` collective. With it Wan goes from 40.3% of a
block to 100% (98.4% of the model), and its measured peak at two ranks drops 7.07 -> 4.45 GB, or
3.15 GB at four. `apply_tensor_parallel` then pairs each attention's `norm_q` and `norm_k`: a
forward hook on `to_k` records the key, `norm_q` packs both statistics flat into one all-reduce,
and `norm_k` normalizes its own input with the stored statistic when that input is the recorded
key -- matched by storage and version, because a module with a full backward hook rewraps its
output in an alias after its forward hooks run. Anything else falls back to the unpaired norm, and
the attention's `always_call` hook empties the pairing on every exit, a raised forward and an
early-stopped checkpoint recompute included. One collective per attention instead of two, in each
direction (`tests/distributed/test_across_head_norm_pairing.py`).

Low-precision q/k norms share one FP32 local activation view between their statistic and direct
paths. Both derivative branches accumulate there before one FP16/BF16 cast; separate casts can
lose nearly cancelling gradients while forward tensors match. Paired keys retain the same view
until consumed, and all cleanup paths clear it. This retains an FP32 activation per local q/key
slice for backward; the peak-memory numbers above precede this correction.


LTX-2 needed one thing more. Its processor applies the rotary embedding to flat
`[B, S, inner_dim]` activations, and the `cos`/`sin` it consumes are full-width *activations* --
sharding those is an input transform, not a weight placement, so it is not expressible as a
`ParallelStyle` on the attention module. Expressed instead as an output transform on each of the
four `rope` modules, which narrows the tables where the projections were split. That takes the plan
from 42.67% of the model to **98.16%**, or 2.72x -> 1.055x per-rank parameter bytes at tp=4.

Where to narrow depends on the table layout, and the first version got the shipped one wrong. Both
published checkpoints, `Lightricks/LTX-2` and LTX-2.3, set `rope_type: split`, not the class default
`interleaved`. Split tables are `[B, heads, S, head_dim / 2]`, and `apply_split_rotary_emb` reshapes
the query to the head count `cos.shape[1]` reports, so a last-axis slice kept every shape consistent
and rotated the wrong channel pairs -- **silently** wrong, off by 0.16 in the float64 parity test.
`RotaryTableSlice` reads `rope_type` off each rotary module when the plan is applied and narrows
split tables on the head axis and interleaved ones on the channel axis; an unknown layout raises.
`tests/distributed/test_ltx2_rotary_parity.py` holds interleaved, split, and split with sequence
parallelism to 7e-15 against the unsharded model.

### The FLUX.2 family declines it

`flux2` is FLUX.2-dev (32.22B, 8 double + 48 single blocks, Mistral3 text encoder, distilled
guidance) and `flux2-klein` is Klein-4B/9B (3.88B over 5+20, 9.08B over 8+24, Qwen3 text encoder,
real CFG). They are sibling adapters, not a subclass pair, sharing only
`Flux2Transformer2DModel` — so the module paths match while the geometry does not.

Both return `{}`, which makes the backend validator reject `tp_size > 1` with a reason. The
single-stream block holds exactly two weights, both packed (`to_qkv_mlp_proj` on its output axis,
`to_out` on its input axis), and the double blocks' `ff.linear_in` packs SwiGLU — so a plan that
keeps checkpoints valid reaches 7.5% of dev and 9.7%/11.8% of the two Kleins. At those coverages
`tp=2` costs dev **92% more parameter memory per rank** (4.03B → 7.75B) and **0.514x throughput**,
since only 5.4% of the forward FLOPs are split while the rest is recomputed on every rank of the
group. Klein needs nothing: FSDP alone leaves 0.48B and 1.14B parameters per rank on eight.

Sharding the *input* feature dimension instead — `RowwiseParallel(input_layouts=Replicate())`, which
has a correct global view because the packing is on the other axis — would reach 97.5% and was
verified exact in fp64. It still does not earn its place: it splits parameters but not activations,
so on a fixed world it only stops being harmful, at 22x the per-step communication. `derive_lora_plan`
forwards the base style's `input_layouts` to `lora_A` so that option stays open; the
`RowwiseParallel` default would otherwise declare a replicated activation to be a shard.

## Sequence parallelism

`parallelism.sequence_parallel` extends a tensor-parallel run to shard the token axis
wherever the plan left activations replicated -- the norms, the residual stream, the modulation
arithmetic. It reuses the tensor-parallel group rather than adding a mesh dimension, so it requires
`tp_size > 1`. Its activation traffic matches tensor parallelism alone in **both directions** -- an
all-reduce is a reduce-scatter followed by an all-gather, which is exactly the pair it substitutes
-- but only because every attention and feed-forward is entered through a region entry. torch's
`ColwiseParallel` lifts its own input, so its backward all-reduces the input gradient once per
projection: q, k and v cost three full-size all-reduces per stream under tensor parallelism and
three full-*sequence* all-reduces under sequence parallelism, with no reduce-scatter at all.
`SharedInputColwiseParallel` leaves each projection's input gradient a partial sum, and the
`ColumnParallelRegion` (one all-reduce) or `SequenceParallelRegion` (all-gather forward,
reduce-scatter backward) on the enclosing module reduces the sum once. On a joint-attention plus
SwiGLU block at tp=2 that took the backward from 8 all-reduces to 3 under tensor parallelism and
to 2 reduce-scatters plus 1 all-reduce under sequence parallelism
(`tests/distributed/test_shared_input_collectives.py`). What sequence parallelism adds on top are
small all-reduces of replicated parameters (norm weights, modulation tables) and broadcast
conditioning inside the sharded region. MiniMax-H3 keeps the per-projection lifts: it cannot run
tensor parallelism on an 80 GB card anyway.

Measured on 2x L20Y at Wan2.1-14B's width (two blocks, 32,760 tokens, bf16, full finetune), one
forward and backward: 462.2 -> 453.4 ms at tp=2 (-1.9%) and 446.9 -> 431.9 ms with sequence
parallelism (-3.4%), peak memory unchanged; at 32k tokens attention compute dominates the step. The
end-to-end runs moved more, rollout included, which has no backward for the entry to save:
SD3.5-M GRPO's cycle went 43.0 -> 34.8 s, Wan2.1-14B's 88.2 -> 76.6 s and Qwen-Image-2.1's
441.4 -> 360.8 s at tp=2, with step-0 rewards unchanged to the fourth decimal. That is consistent
with the projections no longer going through DTensor dispatch -- a shared-input projection is a
plain `F.linear` on its local rows -- which costs most where steps are short, though no ablation
separates the two causes. The q/k norm pairing takes the statistic all-reduce from 40.7 to 17.3 us
per attention per direction on the same pair of cards, a saving lost in the noise of a step.

A projection that leaves its input gradient a partial sum is silently wrong outside a region --
the loss is bit-identical and everything upstream trains on this rank's share -- so
`apply_tensor_parallel` refuses one without an enclosing entry, and checks on each projection's
first gradient-recording call that its input really came through one (an entry that does not name
the argument a processor feeds a projection from is as silent as no entry).

It is what the memory measurements kept pointing at. Wan at 75,600 video tokens in bf16:

| | peak | transient (activations + gradients) |
|---|---|---|
| tp=1 | 53.2 GB | 51.5 GB |
| tp=2 | 48.6 GB | 47.5 GB |
| **tp=2 + SP** | **35.8 GB** | **34.7 GB** |

Tensor parallelism removed 3.9 GB; sequence parallelism removed 12.8 GB more. The gap is the
full-width fp32 modulation chain, which no weight-sharding plan can reach.

At the production settings below tensor parallelism alone costs about as much throughput as it
saves memory, while the pair saves three to four times more for the same communication volume. The
default is still **per adapter**, because an overlay is a claim about every input the architecture
accepts: a prompt-dependent prefix in the sharded sequence (Qwen-Image-2.1's vision-language
output) or a per-token tensor the overlay was never checked against turns a working
tensor-parallel run into one that fails after the load. The flag is tri-state: unset means the
adapter's `sequence_parallel_by_default` -- on for SD3.5 and Wan T2V, verified end to end on real
weights, off elsewhere and announced in the log -- `true` requires an overlay and errors on an
architecture without one, `false` means off. `models/tensor_parallel.sequence_parallel_enabled` is
the one implementation of that rule, and the runtime identity records its result rather than the
configured flag.

Adapters declare `sequence_parallel_plan()`, merged over `tensor_parallel_plan()` with override
semantics.

### Measured: where it starts paying

Wan, two blocks, bf16, one sample per rank, both halves of the trade on the same runs
(`.scratch/tp_probe/sweep_balance_point.py`):

| tokens | tp=1 peak | tp=2 peak | tp=2+SP peak | tp=1 step | tp=2 step | tp=2+SP step | throughput vs tp=1 |
|---|---|---|---|---|---|---|---|
| 3,024 | 6.04 GB | 5.07 | **4.05** | 68 ms | 57 | 56 | 0.60 / **0.61** |
| 14,196 | 21.16 | 19.04 | **14.18** | 592 | 372 | 359 | 0.80 / **0.83** |
| 30,324 | 43.00 | 39.20 | **28.92** | 2104 | 1220 | 1184 | 0.86 / **0.89** |
| 61,236 | **OOM** | **OOM** | **57.22** | — | — | 4038 | — |

Three things to read off it:

* **Sequence parallelism saves a constant ~33% while tensor parallelism's share shrinks** from 16%
  to 9% as the sequence grows. The gap is the full-width fp32 modulation chain, which scales with
  tokens and which weight sharding cannot reach.
* **It moves the feasible boundary, not just a percentage.** At 61,236 tokens tensor parallelism
  OOMs on 80 GB either way and sequence parallelism runs. That is the result worth having.
* **Tensor parallelism's throughput penalty shrinks with sequence length** -- 0.60 at 3k tokens,
  0.89 at 30k -- because the overhead is fixed (collective latency, poor arithmetic intensity on
  halved matmuls) and longer sequences amortize it. Sequence parallelism is consistently 1-3%
  *faster* than tensor parallelism alone at the same communication volume, because it moves less
  memory. So it is never the thing to leave off.

At a fixed *global* batch rather than a fixed per-rank one the throughput column would be worse,
since a tensor-parallel group of `t` ranks consumes one sample between them. The numbers above
answer "how fast is one rank's work when split `t` ways", which is the question when memory is what
forced the split.

**Gradient checkpointing does not subsume it.** That was the obvious objection -- production turns
checkpointing on, and on H3 it had already reduced activations to 10% of the peak. Re-measured with
it enabled, on 80 GB:

| tokens | tp=1 | tp=2 | tp=2+SP |
|---|---|---|---|
| 14,196 | 12.90 GB | 11.19 | **8.75** |
| 61,236 | 47.96 | 44.13 | **32.78** |
| 97,104 | **OOM** | 69.32 | **51.11** |
| 121,296 | OOM | **OOM** | **63.49** |
| 148,176 | OOM | OOM | OOM |

Sequence parallelism still saves 32%, essentially unchanged from the 33% without checkpointing.
Checkpointing stores a block's *input* and recomputes inside it, and the recompute still
materializes the full-width fp32 modulation tensors -- the peak is set by that transient, and
sharding the token axis shards the transient too. The two techniques compose rather than overlap.

The feasible sequence boundary moves accordingly: between 61k and 97k tokens at `tp=1`, between 97k
and 121k at `tp=2`, and between 121k and 148k with sequence parallelism -- roughly **double**. At
121,296 tokens it is the only configuration that runs at all.

**Which settles what the whole investment is for.** At the production settings above -- gradient
checkpointing on, a long sequence -- tensor parallelism alone costs 8% throughput for 8% memory,
which is not a trade worth making. Adding sequence parallelism costs 7% throughput for 26% memory
and roughly double the feasible sequence. So on these architectures tensor parallelism is not the
feature; it is the group that sequence parallelism needs, and the two are worth enabling together or
not at all. `.scratch/tp_probe/report_balance_point.py` collates the runs if more are added.

### The mechanism

Activations stay **plain local tensors**; only the transitions touch DTensor. That is deliberate:
`ReplicateParallel` computes on local views, so a DTensor leaking into a block's own arithmetic
meets a plain modulation vector and `aten.mul` refuses. Five kinds of entry:

| Entry | Purpose |
|---|---|
| `SequenceParallelRegion(gather=…, replicated=…)` | gather before attention and the feed-forward, so rotary embeddings, masks and cross-attention text need no slicing; its backward reduce-scatters the summed gradient of every column-parallel consumer |
| `RowwiseParallel(output_layouts=Shard(1))` | reduce-scatter instead of all-reduce on the way out |
| `SequenceBroadcastParameter()` | every parameter inside the sharded region |
| `SequenceBroadcastOutput(output_indices=…)` | a conditioning value broadcast over tokens |
| `PrepareModuleInput` / `PrepareModuleOutput` at the stack boundary | scatter in, gather out |

Region entries name their inputs and find them by keyword or position from the module's
`forward` signature, so one entry serves both of diffusers' calling conventions.
`PrepareModuleInput` still needs **one layout per positional argument the module is actually called
with** (`None` for the ones that are not sharded); a mismatch raises "module inputs and
input_layouts should have same length". Gather after the *output projection* rather than after the
last block where that projection narrows the hidden dimension — Wan's is 5120 → 64, so the gathered
tensor is eighty times smaller and the final modulation stays sharded.

Use `ComposedParallelStyle` where a module needs both a parameter treatment and an activation
transform: `PrepareModuleInput`/`Output` distribute nothing, and a boundary module that owns
parameters would otherwise fail the plan's totality check. `build_total_plan` composes them
automatically when completing.

### One loud limitation: the token count must divide `tp_size`

`PrepareModuleInput` lifts a plain local tensor with `DTensor.from_local`, which has no global
length to read and **infers `local * tp_size`**. A 65-token sequence scattered 33/32 makes the two
ranks reconstruct 66 and 64, and the gather then issues an all-gather with mismatched sizes.

**It deadlocks rather than raising.** The two ranks post all-gathers of *different sizes*, which
NCCL does not reject: measured on LTX-2 at 125 tokens, both GPUs at 100%, not one line of output,
four minutes until killed. A silent deadlock is worse to debug than a silent wrong answer, and the
condition is not always the user's to choose -- Qwen-Image-2.1's sharded axis includes the
vision-language encoder's output, whose length comes from the prompt. SenseNova avoids the same
trap structurally: its overlay shards only the generation branch of its mixture-of-transformers
layers, so the prompt prefix, which runs the understanding branch, is never split, and its
`_GenerationBranchLayout` checks the image grid's divisibility itself on generation calls.

`apply_tensor_parallel` therefore guards it: for every scatter the plan declares, a **prepended**
hook checks the token count against `tp_size` and raises with the arithmetic. It costs no
collective, because the scatter is the one place that still sees the whole sequence -- and it has to
be prepended, or it measures the slice the scatter's own hook has already substituted, and 1,089
tokens pass as 545.

A scatter takes three forms and all three are guarded: a positional argument, a keyword argument,
and a module's *output*. The output form is the better one and the one SD3.5 and Qwen-Image use --
diffusers calls a transformer block by keyword in the eager forward and positionally through
`_gradient_checkpointing_func`, so a block-input transform can satisfy only one convention, while a
forward hook is convention-free.

What detection must *not* do is as important, each item below being a false positive found by
running it:

* **Read the plan before it is applied.** `RowwiseParallel._apply` assigns itself
  `desired_input_layouts = (Shard(-1),)`, because a row-parallel matmul needs its input sharded on
  the contraction axis. Scan an applied plan and every `to_out` looks like a scatter -- SD3.5 then
  rejects its own 77-token CLIP stream.
* **Take the axis off the placement**, not from a hard-coded `shape[1]`. Qwen-Image-2.1 scatters a
  1-D token mask with `Shard(0)`.
* **Never treat a negative `Shard` dim as the token axis.** LTX-2 narrows its rotary tables with
  `Shard(-1)`.
* **Compare `desired_input_layouts` against `input_layouts`.** A style whose two sides agree is a
  *pass-through*: it declares the layout the argument already arrives in so the rest of a positional
  signature can be described. H3 writes `transformer_blocks.*` that way, with only
  `transformer_blocks.0` doing the real `Replicate() -> Shard(1)`. Counting the pass-through puts the
  hook on blocks 1..N, where the hidden state has already been sliced, so it measures `S / tp` and
  demands `(S / tp) % tp == 0` -- effectively `tp**2 | S`. Measured at `S=6, tp=2`: `6 % 2 == 0`, and
  the run died with "must be divisible by tp_size=2; received 3". Three of every four otherwise-legal
  lengths at tp=4. This one fails *closed*, which makes it the easiest to miss and the most annoying
  to hit.
* **Address an output by position.** A module returning a hidden state beside a mask would
  otherwise have the mask measured on an axis the plan never sharded.

What an unguarded scatter would have cost is architecture-dependent, which is the reason to guard
all three forms rather than only the loudest: LTX-2 deadlocks, while SD3.5 and Qwen-Image happen to
have something position-indexed downstream of the gather -- an unpatchify, a rotary multiply -- that
trips on the inflated length first. Measured at 1,089 image tokens on two ranks, those two report
`shape '[1, 33, 33, 2, 2, 16]' is invalid for input of size 69760` and `the size of tensor a (1090)
must match the size of tensor b (1089)`, with the two ranks disagreeing. Loud, but naming neither
parallelism nor the sequence.

**The limit is the training-side norm, and this framework keeps it.** Every stack that trains
requires divisibility: Megatron-LM asserts it in `_split_along_first_dim` and `divide()`,
TransformerEngine raises in `reduce_scatter_along_first_dim` while its gather infers
`local * world` exactly as `from_local` does, and diffusers' default `EquipartitionSharder` asserts
it. Every stack that accepts uneven lengths is inference-only: diffusers' `ulysses_anything` and
`ring_anything` raise `NotImplementedError` in backward, vLLM, vllm-omni and SGLang pad and trim,
and xDiT replicates Qwen's prompt when it does not divide and shards only the image. yunchang, under
xDiT, has the same `local * world` inference with no guard at all.

Four uneven-length designs were checked against this codebase, and none survives. Padding at the
scatter is refuted by LTX-2, which carries two independent token lengths on one axis, and by H3,
whose pad rows become non-zero at block 0 and ride the residual stream. Padding the VL prefix
upstream pads only one of the two places that own the prefix and mask, which breaks rollout/replay
parity. Carrying the true length through `from_local(shape=)` -- once recorded here as the real
fix -- lets the length side channel collide silently, so ranks diverge without an error (proven on
real process groups). Replicating the prefix as xDiT does is ~1,050 lines across shared
infrastructure, and it saves nothing on the pass that fails: for Qwen-Image-2.1 that is the no-grad
KV-cache prefill, the prefix plus a 2x2 dummy target, refused at the campaign geometry at 61 tokens.
The cached decode, where the overlay would save memory, is sliced to target tokens that already
divide.

**So the configuration for a prompt-dependent prefix is `parallelism: {tp_size: N,
sequence_parallel: false}`**, verified on Qwen-Image-2.1 at the campaign geometry (below), and the
guard's message names it. If the overlay is ever wanted there, a per-pass bypass for the no-grad
prefill is far smaller than any general mechanism. Survey and refutations:
`.scratch/gpu_matrix/uneven_sharding_analysis.md`.

### Two silent failures

**Both leave the loss bit-identical.** The forward is genuinely correct on every rank; only a
gradient is wrong, so a test comparing losses -- or even the gradient norm -- passes.

1. **A parameter in the sharded region** is applied per token, so each rank accumulates a partial
   sum of its gradient. Measured at exactly 1/2 on two ranks for Wan's `scale_shift_table`.
   `SequenceBroadcastParameter` distributes it `Replicate` and hands `forward` a local view whose
   backward declares the gradient `Partial`.

   **Do not look for a clean 1/tp.** The magnitude is uninformative, measured four different ways:
   H3's packed sequence put `audio_proj_out` at *exactly zero* on one rank, every audio row having
   landed on the other; LTX-2's audio tables came out at *exactly 2.0* when the style was applied
   where it was not needed, over-reducing; SD3.5's shared `time_text_embed` landed at 0.5117 rather
   than 0.5000, because the intact text stream contributed the excess; and Qwen-Image-2.1's
   `modulation.1` at 0.752, because `causal_condition` splits it into two rows that degrade by
   different factors. Only the per-parameter comparison against the `tp=1` run is evidence.
2. **A conditioning activation broadcast over tokens** carries the same debt one level upstream,
   and no placement records it because the value itself was never sharded. Wan's time embedder came
   back at **27%** of its true gradient, H3's at 40% -- and in H3's case the *global gradient norm
   moved only 1.5%*, so any check with a tolerance on the norm passes.
   `SequenceBroadcastOutput` is Megatron's `f` operator: identity forward, summed gradient
   backward.

Lifting the activations to DTensors cannot reach either: a bare parameter has no module boundary to
hook, and a `Linear` handed a DTensor sharded on an interior axis dies in `F.linear`, which flattens
batch into sequence. Per-head qk-norm takes the same partial-gradient view for its head shard
(`ShardedInputReplicatedWeight`). Lifting its input to a head-sharded DTensor gave a bit-identical
gradient but spent up to 4x the time per forward in DTensor dispatch (measured on CPU), and broke
H3's chunked qk-norm, whose `new_empty` output buffer came out replicated.

`derive_lora_plan` carries the base style's `output_layouts` through `lora_A`, or the adapter branch
is added to a reduce-scattered base at the wrong length ("size of tensor a (160) must match tensor b
(320)"), and gives `lora_B` the broadcast treatment it then needs.

### Verifying a plan

As for tensor parallelism — compare every parameter's gradient and the gradient count, never the
loss alone. Then additionally check that the **output shape is restored**: a boundary that scatters
without gathering produces a short sequence that the loss reduction hides.

## Checkpointing

Measured: **torch handles the composed mesh correctly as shipped**, and the framework needs no new
mechanism. Both paths were verified bit for bit against a never-sharded reference at
`(dp, tp) = (1,2), (2,1), (2,2)`, full and LoRA — see
`tests/distributed/test_tensor_parallel_checkpointing.py`.

- **model-only** (`BaseAdapter.get_state_dict` → `get_model_state_dict(full_state_dict=True,
  broadcast_from_rank0=True)`) reproduces every weight exactly. Only rank 0 materializes the dict,
  which is also the only rank that writes it.
- **exact state** (`accelerator.save_state`/`load_state`, a sharded DCP checkpoint) restores the
  model and the optimizer moments, which inherit the parameter's placements.

The layout to know: a column-parallel weight is sharded on dim 0 by tensor parallelism, so data
parallelism cannot also take a contiguous `Shard(0)` of it. FSDP2 expresses the interleave as
`_StridedShard(0, split_factor=tp_size)`, and a full gather reassembles across both axes. Getting
that wrong does not raise — it writes a checkpoint that loads cleanly and holds a permutation.

**Resume geometry is part of runtime identity.** `dp=8, tp=1` and `dp=4, tp=2` launch the same
number of ranks while sharding parameters and optimizer state differently, so
`runtime_identity._parallelism_schema` records the mesh dimensions; without it the identity gate
compares the two as equal and a resume loads incompatible shards. It returns `None` when no
`parallelism` block is configured, which keeps every existing checkpoint's digest unchanged.

FSDP2 exact state includes EMA, fixed references and algorithm snapshots through rank-local
JSON+safetensors payloads. The v2 storage envelope hashes every physical rank's payload and
validates its global shape, mesh and placements before backend restoration. CPU-offloaded DTensor
copies retain CPU storage with their existing CUDA mesh; do not gather them or reconstruct their
mesh from checkpoint input. FSDP1 auxiliary state and per-node checkpoint directories remain
unsupported. See [parameter state and exact resume](../../../guidance/parameter_state.md) and
`tests/distributed/test_parameter_state_resume.py` for the compatibility and qualification scope.

## Multi-role algorithms

A multi-role algorithm materializes one trainable variant per role, and the two storage modes
behave differently under sharding:

- **LoRA roles are extra PEFT adapters on one module.** There is a single object to shard and
  `derive_lora_plan` reaches every adapter's matrices, so this was correct from the start. Sharding
  the module twice would halve `heads` twice, which is why the apply loop de-duplicates by object
  identity rather than iterating routes blindly.
- **Full roles are `copy.deepcopy(base_module)` under their own bundle route.** Sharding only
  `adapter.get_component(name)` left the replicas holding plain tensors. Measured at tp=2 with
  DMD2's topology: `generator` on `('dp_shard','tp')`, the deep-copied `fake` on `('dp_shard',)`,
  and the first `clip_grad_norm_` raising `DTensor does not support cross-mesh operation` — after
  the weights were loaded. Neither guard caught it: `apply_tensor_parallel`'s totality assertion
  inspects only the root it was handed, and `rebind_parameters` compares *global* shapes, which
  sharding leaves unchanged.

`_apply_tensor_parallel` therefore iterates `registry.resolve_route(variant, component)` for every
declared variant. `resolve_route` is the authority on where a variant's copy lives; matching route
names by prefix would be guessing. The registry is frozen by `materialize()` before this point, so
the route set cannot grow afterwards — which is what makes one pass sufficient.

**Snapshots compose correctly, and that is measured rather than assumed.** TDM-R1 keeps a slow EMA
of its surrogate, taken after `prepare` and after `rebind_parameters`, so the snapshot clones a
DTensor. At tp=2 and at dp=2 × tp=2 (`tests/distributed/test_multirole_variant_snapshots.py`): the
snapshot carries its parameter's mesh *and* placements; the EMA matches an unsharded reference
exactly after three updates; `use_snapshot` installs and restores bit for bit; and role-scoped
`clip_grad_norm_` — which `role_optimization` hands only the active role's parameters, and which
Accelerate groups by `grad.device_mesh` — reports the true global norm to 1e-6.

The mesh assertion worth keeping is across **every** role's route, not the active one. A role whose
module was left unsharded makes `ModelBundle`'s single root mix meshes, but the active role on its
own stays homogeneous either way, so a check scoped to it passes while the run still dies at the
first step.

Note that `parameter_records` holds the identities captured at materialization. A snapshot taken
before `rebind_parameters` would clone the stale pre-shard tensors and report plain `Tensor` on both
sides — consistent, and wrong. The ordering that makes it right is fixed:
`_apply_tensor_parallel` → `prepare` → `rebind_parameters` → `_initialize_snapshots`.

## Backend support

FSDP2 only. `validate_tensor_parallel_backend_plan` rejects everything else, and the reason is not
an omission: DDP and DeepSpeed ZeRO-2 do not shard parameters, so there is no memory problem for
tensor parallelism to solve there. Muon + tensor parallelism is also rejected — its Newton-Schulz
iteration computes `grad @ grad.T`, and DTensor's global `.shape`/`.ndim` let the guard pass while a
spec that degenerated to a local tensor would orthogonalize a *shard*, which is wrong and silent.

The same gate runs **before the weights load**, which is the only place a rejection is cheap, so
anything checkable from the adapter class belongs in it. It reads `tensor_parallel_plan()` (an empty
one means nothing would be sharded while the samplers already divided the batch by `tp_size`) and,
whenever the overlay applies -- asked for by name, or the adapter's default --
`sequence_parallel_plan()` as well, so neither an explicit request nor a default-on adapter can
discover a missing overlay after the load.

## Sample identity is the precondition

Tensor-parallel peers must consume the same samples **and draw the same noise**. Identity-seeded
noise (#267) already keys rollout `x_T`, per-step SDE noise, and training timesteps to the sample
rather than the process RNG, and the sampling plan is rank-independent — so peers that receive the
same sample automatically draw the same noise.

The process RNG covers whatever is left. `utils/dist.seed_data_parallel_replica` seeds it from the
data-parallel coordinate, which equals the process index without a plan -- exactly
`set_seed(seed, device_specific=True)` -- and is shared by the peers of a group under one. Seeded
per rank, the leftovers were silent correctness bugs: DMD2's forward-process noise, DGPO's
independent noise and H3's condition noise each made the peers feed the plan different inputs, and
the TP output matched neither rank's unsharded result (`tests/distributed/test_replica_seeding.py`).
The shared stream holds only while every peer makes the same calls in the same order, so a draw
that runs on some ranks of a group and not others breaks it again. Prefer a generator from
`trainers/common/group_noise.py` (`independent_batch_generator` for a keyed draw,
`sequenced_draw_generator` when repeated draws must each be fresh), which also makes the draw
independent of the layout.

### Rewards: split between the peers, one source per value

The peers hold the same samples, so scoring them all on every peer is `tp_size` times the reward
work -- and agreement would then rest on every scorer returning the same value twice, which a
sampled VLM judgment or a remote call falling back to a default does not. `RewardProcessor` splits
the scoring instead: a pointwise reward scores the rank-local samples with
`index % tp_size == tp_rank`, a groupwise one whole groups by order of first appearance, and the
other peers hold `0` where the reward applies and `NaN` where it does not.
`utils/dist.combine_tensor_parallel_shares` sums them in one float64 all-reduce, exact because each
element has a single nonzero term. All three exits combine -- `compute_rewards`, a streamed tile
and `finalize` -- and eval goes through `finalize`.

Three things this has to keep true. Applicability is marked on **every** peer, scored there or
not, because aggregation reads `applicable_rewards` on each. A peer owning no sample still
assembles its placeholder rows, or it would skip the combine every other peer joins. And the sync
scoring runs inside `fork_rng` under tensor parallelism: peers scoring different samples would
otherwise advance the shared RNG stream by different amounts if a reward drew from it. Async
rewards run on worker threads outside that guard and must not draw from the process RNG at all.
The VAE decode that feeds the rewards stays duplicated: sampling runs in lockstep across the peers,
so a decode split inside the loop is still on every peer's critical path.

## Which algorithms are verified on GPU

Every algorithm below ran to completion on 8x L20Y with
`config/accelerate_configs/fsdp2.yaml` and `parallelism: {tp_size: 2}`, which leaves
`dp_shard_size: 4`. All on SD3.5-Medium so the algorithm is the only variable; sequence
parallelism was left to its default, overlaying 10 entries on every run.

| algorithm | finetune | trainable roles | epochs | peak / rank | wall |
|---|---|---|---|---|---|
| sft | LoRA | 1 | 3 | 17,163 MiB | 70 s |
| offline-dpo | LoRA | 1 + reference | 3 | 17,245 MiB | 76 s |
| grpo | LoRA | 1 | 3 | 16,791 MiB | 133 s |
| grpo + EMA | LoRA | 1 + EMA | 4 | 16,723 MiB | 156 s |
| tdm | **full** | generator, fake | 3 | 21,253 MiB | 193 s |
| dmd2 | **full** | generator, fake | 3 | 19,963 MiB | 124 s |
| tdm-r1 | **full** | generator, fake, surrogate | 10 | ~32,700 MiB | 37 min |
| grpo + eval + EMA | LoRA | 1 + EMA | 5 | 47,131 MiB | 530 s |

Exit status is the weakest possible claim, so each was additionally checked for having
trained at all:

- **Gradients are non-zero and per-role distinct.** `tdm` reported
  `generator=1.02 / fake=0.0057`, `dmd2` `generator=0.044 / fake=0.209`, `tdm-r1`
  `generator=0.46 / fake=0.011 / surrogate=1.10`. A role whose module was left unsharded
  does not produce a plausible norm -- it raises at the first clip.
- **Every full-parameter run shows one `Tensor-parallel plan applied to <role>/transformer`
  line per role**, 577 entries each. Before the route fix only the canonical route appeared.
  LoRA runs show one line for `base/transformer` with 959 entries, the extra ones being the
  `lora_A`/`lora_B` derivations.
- **The offline algorithms really step.** `sft` and `offline-dpo` each performed 3 optimizer
  updates over 3 epochs. The `batches_per_epoch: 0` in their logs is a configuration echo of
  the unset field, not the batch count -- worth knowing before reading it as a dead run.
- **GRPO's `train/loss=0` is group centering, not a dead path**: `ratio_min = ratio_max = 1`
  on the first inner step of an epoch, while `clipped_loss_std=0.41` over
  [-0.66, 0.79] and `grad_norm=0.0015`. Advantages are group-centered, so the mean is zero
  by construction.
- **EMA moves.** `ema_update_interval` is counted in **optimizer steps**, not epochs, despite
  what the shipped GRPO recipe's comment says: `EMAModuleWrapper.step` gates on
  `(optimization_step + 1) % update_step_interval`. At the recipe's value of 4 against a
  4-epoch run the EMA would update once at most and the run would pass having exercised
  nothing. With it at 1 and `save_freq: 1` -- `save_ema` defaults to True, so each checkpoint
  installs the EMA weights through `copy_ema_to` and restores them -- the saved EMA weights
  move monotonically across checkpoints: max |delta| `9.0e-5 -> 1.8e-4 -> 2.7e-4 -> 3.6e-4`
  with 380 of 382 tensors changing, which is `ema = 0.99*ema + 0.01*param` accumulating. Only
  half the tensors move by the first checkpoint because PEFT zero-initializes `lora_B`.
  `ema_device: cuda` is what keeps both operands DTensors on one mesh; `cpu` takes the
  cross-device branch and mixes a plain tensor with a DTensor, which is the documented gap.
- **Evaluation tracks the EMA weights it runs under.** The eval reward was bit-identical at
  epochs 0 and 2 (`reward_pick_score_mean=0.8307`) and moved at epoch 4 (`0.8325`), which is what
  `ema_update_interval: 4` over `gradient_step_per_epoch: 2` predicts: one EMA step by epoch 2,
  taking 1% of a barely-moved LoRA, and two by epoch 4. Constant metrics would otherwise be
  indistinguishable from evaluation ignoring the weights. The spread within one evaluation --
  std 0.0538 over [0.7045, 0.9162] across 16 prompts -- rules out a degenerate eval set.
- **ODE replay stays bit-exact.** `tdm` and `dmd2` kept their shipped `replay_rtol: 0` and
  `replay_atol: 0` rather than loosening them to pass, and neither tripped. Replay re-runs the
  same forward on the same rank with the same weights, so the reduction order is unchanged.

What these runs do **not** show is a memory win: SD3.5's plan covers 65.1%, so at tp=2 the
per-rank parameter cost is 1.349x the FSDP-only share. Tensor parallelism costs resident
memory at this coverage. The savings live at `c ~ 1` -- see the balance point above.

The offline fixtures hold two rows each, so they exercise the pipeline and the replica scoping
rather than convergence.

### Evaluation is data-parallel scoped, and Accelerate does it

The eval `DataLoader` is built with **no sampler** (`data_utils/loader.py:550`) and then handed
to `accelerator.prepare`, while the training loader is never prepared and carries an explicit
dp-scoped sampler. That asymmetry looks like a bug -- shard the eval set over all 8 launched
ranks and a tensor-parallel pair evaluates two different prompts while holding two halves of
one model, with every collective still matching in shape, so nothing raises.

It is not a bug. Accelerate 1.15's `prepare_data_loader` reads the device mesh and collapses
the tensor-parallel dimension itself:

```python
process_index = process_index // (submesh_tp_size * submesh_cp_size)
num_processes = submesh_fsdp_size * submesh_dp_size
```

with its own comment stating the intent -- "generating same batch across TP ranks and different
batch across FSDP and DP ranks". `Accelerator.prepare_data_loader` passes
`self.torch_device_mesh` through `_prepare_device_mesh`, so the eval loader gets it.

Measured on 8 gloo ranks with a `(dp_shard=4, tp=2)` mesh and a 32-item dataset
(`.scratch/tp_probe/eval_dataloader_scope.py`): tensor-parallel peers `(0,1) (2,3) (4,5) (6,7)`
receive **identical** item lists, the four data-parallel replicas receive **distinct** ones, and
their union covers the set **exactly once**. Note that `PartialState` does not pick up the real
rank under a CPU gloo launch -- left to default every rank reported index 0 and took the same
shard, which is the probe being wrong, not the framework.

### Per-epoch sample retention, which tensor parallelism amplifies

An end-to-end `examples/grpo/lora/sd3_5/default.yaml` at `tp_size: 2` -- evaluation and EMA both
on, the shipped batch geometry untouched -- **OOMs in the third evaluation's VAE decode**. GPU
memory climbs in a regular staircase, one step per epoch, with flat plateaus between:
19.3 -> 38.7 -> 57.4 -> 76.2 GB, about +18.7 GB per epoch. `offload_samples_to_cpu: true` and
`PYTORCH_ALLOC_CONF=expandable_segments:True` both fail to stop it (the second attempt still
reported 68.8 GiB genuinely allocated against 6.1 GiB reserved-but-unallocated).

Three runs locate it on the per-sample axis:

| run | replicas | GAS | samples/rank/epoch | growth/epoch | per sample | result |
|---|---|---|---|---|---|---|
| tp=2, 48 unique | 4 | 18 | 288 | +18.7 GB | 65 MB | OOM, 3rd eval, peak 81.1 GB |
| tp=2, 16 unique | 4 | 6 | 96 | +6.5 GB | 68 MB | completed, peak 47.1 GB |
| **tp=1, 48 unique** | 8 | 9 | 144 | +2.2 GB | **15 MB** | completed, peak 25.1 GB |

So the retention is pre-existing and proportional to the samples a rank acquires per epoch --
but a retained sample costs **~4.4x more under tensor parallelism**, and tensor parallelism also
doubles the count by halving the replicas at a fixed global batch. 2.2 GB/epoch becomes
18.7 GB/epoch, which is the difference between finishing and not.

#### Root cause, and the invariant that was missing

Not a sample leak -- the per-sample framing above is how it first looked, and the samples are
innocent, which is why `offload_samples_to_cpu` never helped.

**A style that computes one placement and returns another ends in a collective, and torch issues
it with `async_op=True` (`RowwiseParallel._prepare_output_fn`). `DTensor.to_local()` then hands
back `local_tensor.view_as(local_tensor)` without unwrapping the `AsyncCollectiveTensor`, so what
leaves the module is a promise whose buffer is owned by the functional-collectives work registry,
not by the tensor's refcount.**

Torch's default is right for its own usage: the next op in the same forward consumes the result,
which triggers the wait and clears the registry. It stops being right the moment a forward can end
*without* that consumption -- and tensor parallelism inside activation checkpointing is exactly
there. `torch.utils.checkpoint`'s non-reentrant recompute runs with `early_stop=True` and aborts
through `_StopRecomputationError` as soon as backward's demands are met, so anything issued and not
yet consumed stays in flight for the process's lifetime. A raised forward and a discarded branch
open the same hole.

The chain, each link measured:

1. **Allocation site.** An allocation-history snapshot (`stacks="python"`) put 8.42 GiB in 552 live
   blocks of 15.6 MiB on `_functional_collectives.all_reduce` <- `placement_types._reduce_value` <-
   `_redistribute.Redistribute.forward`.
2. **The shape names the module.** `(16, 333, 1536)` bf16: 8 samples x CFG, `77 CLIP + 256 T5`,
   `caption_projection_dim` -- SD3.5's context stream, whose `attn.to_add_out` the plan covers
   rowwise.
3. **The count names the loop.** 288 live tensors added per epoch = 24 blocks x 12 micro-batches,
   of which 276 carry a `to_add_out` (diffusers makes the last block `context_pre_only`).
   276 x 15.6 MiB = 4.3 GiB, the whole of the measured growth.
4. **The holder is not Python.** Every retained tensor reported **no Python referrer at all** once
   the probe stopped counting its own `gc.get_objects()` list -- a 751,248-element list "holding"
   every tensor was the probe measuring itself. A live CUDA tensor with no Python referrer is held
   from C++, and torch names the holder itself: its exit warning counts "unwaited collective calls"
   in exact multiples of the layer count -- 24, 48, 96, 144, 432, 1008, 1296, 3024.
5. **Three hypotheses eliminated.** `_CheckpointFrame` retention: zero frames, holders and handles
   alive at the boundary, so the recompute cache is not it. Sequence parallelism: ablating the
   overlay left the leak *and the live tensor counts* identical (7106 -> 14235 -> 14523), so it is
   pure tensor parallelism. And redistribute alone is not enough -- a bare `nn.Linear` under torch's
   own `RowwiseParallel`, under FSDP2, under checkpointing, with a no-grad rollout phase, holds flat
   (`.scratch/tp_probe/checkpointed_collective_leak.py`), because there the residual add consumes
   the result and discharges it.
6. **The A/B, and then the fix.** With `fsdp_activation_checkpointing: false` the growth is exactly
   zero. With it left **on** and the collectives discharged, the numbers land on the same place:

   | | ep0 | ep1 | ep2 | live CUDA tensors |
   |---|---|---|---|---|
   | before | 6.40 | 12.15 | 16.64 GiB | 7106 -> 14235 -> 14523 |
   | checkpointing off | 6.40 | 7.64 | 7.64 GiB | 13947 -> 13947 |
   | **discharged, checkpointing on** | 6.40 | 7.64 | **7.64 GiB** | 13947 -> **13947** |

**The invariant, which is the actual deliverable:** *no activation leaves the plan carrying an
undischarged collective.* Everything downstream of our region -- activation checkpointing, CPU
offload, sample storage, reward workers -- is written against ordinary tensor semantics, where the
refcount owns the memory. An in-flight collective result satisfies none of that, so the boundary of
our region is where it has to be settled. `_settle_collective_outputs` enforces it in the one place
the applier already owns, with the boundaries derived from the plan the way the divisibility
guard's are, so a new adapter inherits it.

Two details worth keeping:

* **It is a side effect, not a transform.** `trigger_wait`'s return value is dropped and the hook
  returns `None`. Both `wait` and `trigger_wait` hand back the *unwrapped* `elem`, a different
  object without the autograd metadata the module's output carries -- returning it made backward
  fail with "element 0 of tensors does not require grad and does not have a grad_fn". We discharge
  the obligation and change nothing that flows.
* **It costs nothing.** A settled output is not an `AsyncCollectiveTensor` and the hook does
  nothing; and `async_op` buys overlap only when there is independent work between issue and
  consumption, which at a module boundary there is not.

This is the third member of a family worth recognizing on sight -- `release Qwen-Image 2.1 no-grad
prefix caches` (activation checkpointing saved a `layer_cache` whose K/V kept their `grad_fn`,
forming a cycle only a backward could break) and `bound eval-loop memory under torch.compile` (a
grad-forcing forward built a graph across a no-grad denoising loop). All three are the same shape:
**a never-completed path leaves state alive that the normal path would have reclaimed.** The
retained object differs each time -- an autograd cycle, a graph, a work-registry entry -- so the
tell is the *shape of the control flow*, not the object.

What was missing was not the insight but the test. `tests/distributed/test_collective_settling.py`
now asserts the obligation directly, and the acceptance criterion for anything in this family is a
population that stays flat across iterations rather than a memory total that looks reasonable once.

## Historical two-arm GPU experiment, and why incrementality is the harder half

This section records the TP branch experiment; current PRs use [selected evidence](gpu_validation.md).
The retained `config/gpu_validation/framework_upgrade.yaml` asks for 45 jobs on 32 ranks across 4 machines: 12
algorithm-profile pairs x 3 backends, plus 9 overlap supplementals. On 8 ranks, 41 of those 45 are
reachable -- bagel's 3 need `ByteDance-Seed/BAGEL-7B-MoT`, which is not on this host, and the
overlap supplementals need the remote reward services the manifest points at.
`.scratch/gpu_matrix/build_matrix.py` generates the runnable set from that manifest: the profiles,
algorithms, recipes and backend config files are read from it rather than invented, and the geometry
is the manifest's own on the first attempt.

**Arm A carries no `parallelism` block, and it is the half that matters most.** The work this
feature needed reaches `utils/dist.py`, `group_coordinator.py`, `reward_processor.py`, `args.py` and
`advantage_processor.py`, every one of which executes on runs that never see a tensor-parallel mesh.
A capability PR that only tests its capability is testing the half that is easy to notice. The DPO
`reduce_sum` degradation found on this branch -- a metric silently reduced to one rank's share on
*every* backend -- is what arm A exists to catch, and it would not have appeared in any
tensor-parallel run.

**The first pass reported 40 of 44, and 16 of those were not what they claimed.** The generator
ignored each profile's `model_type`, and the committed manifest uses SD3.5 recipes as templates, so
the MiniMax-H3 `nft`, `offline-dpo`, `online-dpo` and `sft` jobs trained SD3.5 on every backend --
the H3 `sft` row matched SD3.5's to the MiB, which is how review caught it. Those rows are
withdrawn. **Of the 28 jobs that ran what they say, 24 pass.** Peak MiB per rank; a dagger marks a
job reduced below the manifest geometry to fit 80 GB, and 124 frames is kept throughout because the
H3 adapter's VAE accepts only `17n+5` within 120..360, so only the spatial size can move.

| profile | algorithm | ddp | zero2 | fsdp2 | fsdp2 + tp=2 |
|---|---|---|---|---|---|
| qwen-image-2.1 | grpo | 48,467 | 34,107 | 25,901 | guard, below |
| sd3-5 | nft | 26,743 | 20,709 | 15,083 | 16,087 |
| sd3-5 | offline-dpo | 33,873 | 24,309 | 14,867 | 15,877 |
| sd3-5 | online-dpo | 37,875 | 28,199 | 14,913 | 15,915 |
| sd3-5 | sft | 19,239 | 14,175 | 14,839 | 15,833 |
| minimax-h3 | grpo | 79,375† | 79,159† | **65,109** | out of memory |
| minimax-h3 | tdm | out of memory | 78,177† | pass | out of memory |

The withdrawn H3 jobs were rerun on `MiniMaxAI/MiniMax-H3` itself, FSDP2 without a `parallelism`
block, on this branch and on main (`online-dpo` and `nft` after setting the outer batch to the
`B=1` the H3 workflows require):

| algorithm | main | this branch |
|---|---|---|
| sft | 65,445 | 65,101 |
| offline-dpo | 65,101 | 65,101 |
| online-dpo | 65,527 | 65,527 |
| nft | 65,527 | 65,597 |
| grpo | 65,249 | 65,249 |
| tdm | 65,807 | 65,807 |

All pass on both, and `sft` logs the same loss as main at every step. Every H3 job with
`tp_size: 2` fails between 80.6 and 81.0 GB, inside `apply_tensor_parallel`: the startup bound
described in the H3 section, reached before any training step.

### The four that do not pass, and why none of them is this branch

Each was driven until the cause was unambiguous rather than left at the first failure.

**`qwen-image-2.1 grpo` + tp=2 -- the divisibility guard, working.** It refuses a
`target_token_mask` of length 605, 631, 687: Qwen-Image-2.1's sharded sequence includes the VL
encoder's output, whose length comes from the prompt, so a divisible one is not the user's to
choose. This is the limitation recorded above, and the guard is what turns it from a silent NCCL
deadlock into a message naming the axis. **The same job with `sequence_parallel: false` passes at
26,527 MiB**, which separates "tensor parallelism works on this model" from "sequence parallelism
cannot shard this sequence".

**Three MiniMax-H3 jobs -- capacity, of two different kinds.** `tdm` on ddp ends with ~77 GiB
already allocated and a few hundred MiB free: ddp replicates parameters, gradients and optimizer
state, and `grpo` on **fsdp2 passes at 65,109 MiB on the manifest's full 576x1024x124**. The two
tp=2 jobs never reach a step. `B__grpo__fsdp2__tp2` peaked at 80,893 MiB both before and after its
resolution was cut because it fails inside `apply_tensor_parallel`, where each rank must hold
`P_master * (c / t + (1 - c))` -- 117 GB for H3 at tp=2 -- before `fully_shard` divides anything.
An earlier version of this paragraph read that peak as the steady-state parameter penalty; the H3
section above explains why it is the startup bound instead.

Four earlier failures in this matrix were **mine, not the framework's**, and each first looked like
a product bug: `num_inference_steps` overridden without clamping the recipe's
`scheduler.sde_steps`, which the validator refused -- correctly; `num_sde_steps` left above the
clamped list; a video fallback that shrank frames to 13, which the H3 adapter rejected outright; and
a fallback resolution of 144, which its VAE rejected for not being a multiple of 32. `ImageBind` was
also simply not installed, which cost a round to notice. The generator now gets all four right.

## Verifying a new plan

Three things, in order of how easily each hides a bug:

1. **The gradient count**, which must be identical. A change there is torch's styles unfreezing
   `base_layer`, and it is the only symptom that bug has.
2. **Every parameter's gradient**, not the total norm. A wrong per-head qk-norm gradient is ~1e-5
   against a norm of ~1 and a wrong replicated convolution gradient moved SD3.5's norm by 0.18%,
   both of which a tolerance on the norm waves through.
3. The loss and the norm, as a cheap summary.

Compare at a **fixed replica count** — `dp=4, tp=1` against `dp=2, tp=2` changes the global batch
and measures nothing — and remember FSDP reduces gradients over `dp`, so proving two replicas saw
different samples needs the *pre-reduction local* loss.

**Do not expect bit-identical gradients in fp32.** Row parallelism splits a dot product into
`tp_size` partial sums and all-reduces them, which changes the accumulation order and perturbs the
last bits. The observed floor is ~1e-6 element-wise relative in fp32 across every architecture
measured (MiniMax-H3 1.9e-6, SD3.5 1.7e-6, Qwen-Image 1.4e-6, Z-Image 1.2e-6) and ~1e-14 in fp64,
which is what identifies it as accumulation order rather than a sharding error. H3's earlier
"bit-identical in fp32" result was measured on 6-decimal gradient sums and does not hold
element-wise. The loss often *is* bit-identical even when gradients are not, because the forward's
row-parallel reduction is a single all-reduce of already-summed terms.

**A plan under suspicion should be checked in fp64 — but read the floor off the same plan, not off
1e-14.** `ShardedAcrossHeadRMSNorm` once upcast its `[batch, seq, 1]` statistic to fp32 whatever the
input, which put an fp64 floor of ~6e-7 under every plan containing it (measured on one LTX-2
model: 6.3e-7 with the attention entries that use it, 5.5e-15 with only the feed-forward ones). It
now accumulates in the input's own dtype from fp32 up -- 1.6e-13 element-wise in fp64 on two ranks
-- and in bf16 scales and weights in fp32 and rounds once, as `F.rms_norm` does; casting the scale
to bf16 first had moved 35% of outputs off torch's. Compare a candidate plan against the *same
architecture's* tensor-parallel run rather than against an absolute tolerance -- "sequence
parallelism added nothing above the reduction-order floor" is both the stronger claim and the
easier one to make.

Known open item: Qwen-Image-2.1 shows a 2.7e-4 forward difference and 3.4% gradient-norm difference
in a real `ff-train` run that the isolated probe cannot reproduce, against a 0.19% reduction-order
noise floor. Precision, world size, data assignment, activation checkpointing, the trainable
parameter set, gradient placements and the model config are all ruled out; the unexplained
difference is the real input layout (prefix KV cache, block-causal segmented attention, image tokens
scattered into the text sequence). H3 has none of those features and is exact, so the mechanism is
not implicated.

## Cross-refs

- [`constraints.md` #18b](../constraints.md#18b-tensor-parallelism-scopes-samples-not-ranks)
- [`constraints.md` #19](../constraints.md#19-fsdp-cpu-efficient-loading)
- [`constraints.md` #26](../constraints.md#26-fail-fast-error-handling)
- [`topics/samplers.md`](samplers.md) -- the replica count a sampler partitions over
- [`topics/component_variants.md`](component_variants.md) -- the registry rebound after sharding
- [`topics/minimax_h3.md`](minimax_h3.md) -- the architecture whose plan is verified exact
