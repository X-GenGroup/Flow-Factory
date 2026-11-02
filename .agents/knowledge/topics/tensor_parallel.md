# Tensor Parallelism

Flow-Factory shards a trainable denoiser across a `tp` mesh dimension and composes that with FSDP2
over `dp_shard`. The mechanism lives in `src/flow_factory/models/tensor_parallel.py`; each adapter
declares *what* to shard through `BaseAdapter.tensor_parallel_plan()`.

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

**4. A packed projection has no correct placement.** A `Linear` fusing several logical outputs (a
SwiGLU's `[hidden, gate]`, a fused `qkv`) cannot be column-sharded with `Shard(0)`: that hands rank
0 the whole first block instead of a slice of each. diffusers solves the *computation* by slicing
per block and declaring the result `Shard(0)`, but the DTensor's global view is then a permutation
of the original weight — which is why its `save_pretrained` refuses to run under tensor parallelism.
Flow-Factory saves checkpoints, so that trade is unavailable. Measured on real process groups:
`Shard(0)` reconstructs a permutation and `_StridedShard(0, split_factor=blocks)` asserts. **Leave
packed modules replicated and record the resulting replicated fraction in the plan's docstring**, so
a reader can judge whether the plan is worth using.

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
completes with a 72.7 GB peak, `tp=2` and `tp=4` both OOM above 80 GB, and `tp=8` is rejected before
it starts. Raising the resolution from 128x224 to 768x1344 -- 36x the pixels -- moved the `tp=1`
peak only from 65.2 to 72.7 GB, which says activations are a minor term here and no activation
saving could pay for the parameter penalty. **Tensor parallelism is the wrong tool for H3**; the
earlier probe numbers showing it saving memory were taken at `dp=1`, where there is no FSDP to
compare against.

**Sequence parallelism does not rescue it either, and the measurement is unusually clean.** With
the overlay enabled the same run OOMs at `tp=4` and at `tp=2`, and so does a sixteen-times-smaller
resolution -- at the *same* 80,995 MiB peak in all three cases, identical to `tp=2` without the
overlay. A peak that does not move with the sequence length is not an activation peak. H3's ceiling
is the per-rank parameter penalty and nothing else, which is what 23.3% coverage buys. Coverage is
the number that decides whether either technique is worth running; it is not a detail.

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
| sensenova | 92.4% | 8 | 8 kv heads; the ceiling is not enforced at plan time (no `heads` attribute) |
| bagel | 87.4% of a layer | unbounded | feed-forward only; attention needs the vendored packed buffer split |
| qwen-image, qwen-image-edit-plus | 66.5% | 8 | |
| sd3-5 | 65.1% | 8 | |
| flux1, flux1-kontext | 36.2% | 8 | no sequence-parallel plan yet; 38 single-stream blocks shard nothing, `proj_out` fusing attention with MLP |
| flux3 | 29.3% | 8 | |
| flux2, flux2-klein | **declined** | — | empty plan; see below |
| minimax-h3 (3 variants) | 23.3% | 8 | packed SwiGLU and full-width modulation stay replicated |
| wan2_t2v, wan2_i2v | **98.4%** | 8 | whole block; across-head qk-norm needs its own style |
| ltx2_t2av, ltx2_i2av | **98.16%** | 8 | whole block; across-head qk-norm plus sharded rotary tables |

Wan and LTX-2 build qk-norm over `num_heads * head_dim` and apply it before splitting heads, so
column-parallel q/k/v splits the *normalized* axis and makes the forward wrong, not just the
gradient -- it raises on the normalized shape rather than failing quietly.
`ShardedAcrossHeadRMSNorm` completes that case by sharding the weight alongside the projection and
all-reducing the mean square, which costs one `[batch, seq, 1]` collective per norm. With it Wan
goes from 40.3% of a block to 100% (98.4% of the model), and its measured peak at two ranks drops
7.07 -> 4.45 GB, or 3.15 GB at four.

LTX-2 needed one thing more. Its processor applies the rotary embedding to flat
`[B, S, inner_dim]` activations, and the `cos`/`sin` it consumes are full-width *activations* --
sharding those is an input transform, not a weight placement, so it is not expressible as a
`ParallelStyle` on the attention module. Expressed instead as a `PrepareModuleOutput` on each of the
four `rope` modules, which slices the tables on the channel axis where the projections were split.
That takes the plan from 42.67% of the model to **98.16%**, or 2.72x -> 1.055x per-rank parameter
bytes at tp=4. It assumes `rope_type="interleaved"`, which is the class default: a non-interleaved
table would be **silently** wrong, because `apply_split_rotary_emb` reads its head layout from the
table it is handed.

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
`tp_size > 1`, and it costs **no extra communication**: an all-reduce is a reduce-scatter followed
by an all-gather, which is exactly the pair it substitutes.

It is what the memory measurements kept pointing at. Wan at 75,600 video tokens in bf16:

| | peak | transient (activations + gradients) |
|---|---|---|
| tp=1 | 53.2 GB | 51.5 GB |
| tp=2 | 48.6 GB | 47.5 GB |
| **tp=2 + SP** | **35.8 GB** | **34.7 GB** |

Tensor parallelism removed 3.9 GB; sequence parallelism removed 12.8 GB more. The gap is the
full-width fp32 modulation chain, which no weight-sharding plan can reach.

**It is on by default**, because at the production settings below tensor parallelism alone costs
about as much throughput as it saves memory while the pair saves three to four times more for the
same communication volume. The flag is tri-state: unset means "use it wherever the adapter has a
plan", `true` means "require one" and errors on an architecture that has none, `false` means off.
An architecture without a derivation yet falls back to tensor parallelism alone and warns, since
that is where most of the saving would have come from.

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
| `PrepareModuleInput(Shard(1) → Replicate, use_local_output=True)` | gather before attention and the feed-forward, so rotary embeddings, masks and cross-attention text need no slicing |
| `RowwiseParallel(output_layouts=Shard(1))` | reduce-scatter instead of all-reduce on the way out |
| `SequenceBroadcastParameter()` | every parameter inside the sharded region |
| `SequenceBroadcastOutput(output_indices=…)` | a conditioning value broadcast over tokens |
| `PrepareModuleInput` / `PrepareModuleOutput` at the stack boundary | scatter in, gather out |

`PrepareModuleInput` needs **one layout per positional argument the module is actually called
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
condition is not always the user's to choose -- SenseNova's sharded axis includes the text prefix,
whose length comes from the prompt, and Qwen-Image-2.1's includes the vision-language encoder's
output.

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

The real fix for the underlying limitation is different and still outstanding: `from_local` accepts
the true global `shape`, so a scatter/gather pair of this framework's own could carry the length the
scatter saw into every gather of the region, needing no collective
(`.scratch/tp_probe/probe_uneven_sequence.py`, `cpu_uneven_token_axis.py`).
`PrepareModuleInput` does not expose `shape`, which is why the shipped plans inherit the limit.

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

`ShardedInputReplicatedWeight` cannot substitute for either. It declares a module's *input* sharded,
which needs a module boundary a bare parameter does not have — and a `Linear` handed a DTensor
sharded on an interior axis dies in `F.linear`, which flattens batch into sequence.

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

Exact-state resume combined with EMA, reference parameters or algorithm snapshots is rejected under
FSDP regardless of tensor parallelism (`BaseTrainer._validate_distributed_runtime_children`);
model-only checkpoints remain supported. That rejection is also what keeps a DTensor out of the
exact-state serializer: `_multirole_snapshot_state_dict` hands `TrainerRuntimeState` the snapshot
tensors by reference, and `_encode_node` refuses any non-plain `torch.Tensor`. Both the save path
(`_save_exact_training_state`) and the resume path call the validator first, and tensor parallelism
requires FSDP2, so the combination is unreachable rather than merely untested.

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
when `parallelism.sequence_parallel` was spelled `true` rather than left to default on,
`sequence_parallel_plan()` as well. The default *is* to fall back with a warning where the adapter
is silent -- most adapters that can shard weights cannot yet shard the token axis -- but an explicit
request is a different statement, and `_apply_tensor_parallel` is too late to discover it.

## Sample identity is the precondition

Tensor-parallel peers must consume the same samples **and draw the same noise**. Identity-seeded
noise (#267) already keys rollout `x_T`, per-step SDE noise, and training timesteps to the sample
rather than the process RNG, and the sampling plan is rank-independent — so peers that receive the
same sample automatically draw the same noise. `set_seed(seed, device_specific=True)` makes the
process RNG rank-specific, so **any unseeded `torch.rand`/`randn` on a training path is a
tensor-parallel correctness bug**, not merely a reproducibility wart. Pass a generator from
`trainers/common/group_noise.py` (`independent_batch_generator` for a keyed draw,
`sequenced_draw_generator` when repeated draws must each be fresh).

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

#### Root cause: activation checkpointing pins the redistribute's collective output

Not a sample leak at all -- the per-sample framing above is how it first looked, and the samples
are innocent. `offload_samples_to_cpu` not helping was the clue.

**FSDP2 activation checkpointing saves each block's inputs. Under tensor parallelism one of those
inputs is the output of a DTensor `redistribute`, and that output is an async-collective buffer
that torch's work registry pins until `wait_tensor()` is called on that exact tensor.** The
checkpoint holds the saved tensor, the registry holds the buffer, and neither is released when the
autograd graph is torn down.

The chain, each link measured:

1. **The allocation site.** An allocation-history snapshot (`stacks="python"`) at epoch 3 puts
   8.42 GiB in 552 live blocks of 15.6 MiB on
   `_functional_collectives.all_reduce` <- `placement_types._reduce_value` <-
   `_redistribute.Redistribute.forward` -- the `Partial -> Replicate` reduction a
   `RowwiseParallel` performs. A companion group of 24 blocks comes from the
   `reduce_scatter_tensor` half, which is sequence parallelism's.
2. **The shape names the module.** `(16, 333, 1536)` bf16: 16 = 8 samples x CFG,
   `333 = 77 CLIP + 256 T5`, `1536 = caption_projection_dim`. That is SD3.5's *context* stream,
   whose output projection `attn.to_add_out` the plan covers with `RowwiseParallel`.
3. **The count names the loop.** 276 retained per epoch = **23 x 12**. diffusers sets
   `context_pre_only = (i == num_layers - 1)`, so of 24 blocks exactly 23 own a `to_add_out`;
   12 is the micro-batches per epoch. 276 x 15.6 MiB = 4.3 GiB, which is the whole of the
   measured ~4.5 GiB/epoch growth at that geometry.
4. **Torch says so itself.** Its exit warning counts "unwaited collective calls" in exact
   multiples of 24 across these runs -- 24, 48, 96, 144, 432, 1008, 1296, 3024.
5. **Redistribute alone is not the retainer.** A bare `nn.Linear` under torch's own
   `RowwiseParallel(use_local_output=True)`, called 60 times under `no_grad` with every reference
   dropped, holds flat at +32 MiB -- with *and* without an explicit `wait_tensor`
   (`.scratch/tp_probe/rowwise_collective_retention.py`). Something in the training stack has to
   be holding the output for the registry entry to matter.
6. **A/B on the checkpointing switch, which closes it.** Same config, same geometry, five epochs:

   | `fsdp_activation_checkpointing` | ep0 | ep1 | ep2 | ep3 | ep4 | live tensors |
   |---|---|---|---|---|---|---|
   | `true` (the shipped default) | 6.40 | 12.15 | 16.64 | 21.13 | **25.62 GiB** | 14235 -> 15099 |
   | **`false`** | 6.40 | 7.64 | **7.64** | — | — | 13947 -> **13947** |

   With it off, allocated memory is flat to the digit and the live tensor count does not move.

This is the same family as the `release Qwen-Image 2.1 no-grad prefix caches` fix, which is where
to look for prior art: there, activation checkpointing saved a `layer_cache` whose K/V kept their
`grad_fn`, forming an autograd cycle only a backward could break. Here the retained object carries
no `grad_fn` at all -- `requires_grad=False`, `leaf=True` -- so it is the collective's work-registry
entry rather than a graph cycle, but the trigger is the same saved-inputs mechanism.

**Why tensor parallelism and not FSDP alone:** without a tensor-parallel plan there is no
`redistribute`, so a block's saved input is an ordinary tensor that dies with the graph. The
registry only has something to pin once a collective produced it.

**What to do today:** `fsdp_activation_checkpointing: false`. At this geometry that is strictly
better anyway -- its flat 7.64 GiB is below what the checkpointed run already holds by epoch 1 --
though a long-sequence model will not have that luxury. Scaling
`unique_sample_num_per_epoch` down with the replica count also buys epochs, but only slows the
slope.

**The fix to validate next:** force the redistribute output to be waited before anything can save
it, so the registry entry clears -- `funcol.wait_tensor` on the output inside the styles we own,
and an output transform for the ones torch's own `RowwiseParallel` produces. The A/B above is the
test it has to pass: live tensor count flat across epochs with checkpointing *on*.

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
1e-14.** `ShardedAcrossHeadRMSNorm` upcasts only its `[batch, seq, 1]` statistic to fp32, which is
the memory trade its docstring describes, so any plan containing it has an fp64 floor of ~6e-7
rather than ~1e-14. Measured on one LTX-2 model: 6.3e-7 with the attention entries that use it,
5.5e-15 with only the feed-forward ones. In fp32, where every real run lives, that is far below the
reduction-order noise. Compare a candidate plan against the *same architecture's* tensor-parallel
run rather than against an absolute tolerance -- "sequence parallelism added nothing above the
reduction-order floor" is both the stronger claim and the easier one to make.

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
