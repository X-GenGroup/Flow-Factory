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
| flux1, flux1-kontext | 36.2% | 8 | 38 single-stream blocks shard nothing: `proj_out` fuses attention with MLP |
| flux3 | 29.3% | 8 | |
| flux2, flux2-klein | **declined** | — | empty plan; see below |
| minimax-h3 (3 variants) | 23.3% | 8 | packed SwiGLU and full-width modulation stay replicated |
| wan2_t2v, wan2_i2v | **98.4%** | 8 | whole block; across-head qk-norm needs its own style |
| ltx2_t2av, ltx2_i2av | 43.5% of a block | 8 | feed-forward only: full-width RoPE on flat activations |

Wan and LTX-2 build qk-norm over `num_heads * head_dim` and apply it before splitting heads, so
column-parallel q/k/v splits the *normalized* axis and makes the forward wrong, not just the
gradient -- it raises on the normalized shape rather than failing quietly.
`ShardedAcrossHeadRMSNorm` completes that case by sharding the weight alongside the projection and
all-reducing the mean square, which costs one `[batch, seq, 1]` collective per norm. With it Wan
goes from 40.3% of a block to 100% (98.4% of the model), and its measured peak at two ranks drops
7.07 -> 4.45 GB, or 3.15 GB at four.

LTX-2 is still feed-forward only, for a second reason the style does not reach: its processor
applies the rotary embedding to flat `[B, S, inner_dim]` activations, and the `cos`/`sin` it
consumes are full-width *activations*. Sharding those is an input transform, not a weight
placement, so it is not expressible as a `ParallelStyle` at all.

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
model-only checkpoints remain supported.

## Backend support

FSDP2 only. `validate_tensor_parallel_backend_plan` rejects everything else, and the reason is not
an omission: DDP and DeepSpeed ZeRO-2 do not shard parameters, so there is no memory problem for
tensor parallelism to solve there. Muon + tensor parallelism is also rejected — its Newton-Schulz
iteration computes `grad @ grad.T`, and DTensor's global `.shape`/`.ndim` let the guard pass while a
spec that degenerated to a local tensor would orthogonalize a *shard*, which is wrong and silent.

## Sample identity is the precondition

Tensor-parallel peers must consume the same samples **and draw the same noise**. Identity-seeded
noise (#267) already keys rollout `x_T`, per-step SDE noise, and training timesteps to the sample
rather than the process RNG, and the sampling plan is rank-independent — so peers that receive the
same sample automatically draw the same noise. `set_seed(seed, device_specific=True)` makes the
process RNG rank-specific, so **any unseeded `torch.rand`/`randn` on a training path is a
tensor-parallel correctness bug**, not merely a reproducibility wart. Pass a generator from
`trainers/common/group_noise.py` (`independent_batch_generator` for a keyed draw,
`sequenced_draw_generator` when repeated draws must each be fresh).

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
element-wise; **a plan under suspicion should be checked in fp64**, where every architecture
measured so far does reach parity. The loss often *is* bit-identical even when gradients are not,
because the forward's row-parallel reduction is a single all-reduce of already-summed terms.

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
