# Sampler System

**Read when**: Editing `data_utils/sampler*`, hparams sampler/batch fields.

---

## Overview

Generation acquisition uses **K-Repeat Sampling** to create `K` copies of each unique prompt for
group-wise advantage estimation. Four framework samplers differ in how repeated samples are
distributed across ranks. Dataset acquisition is a separate path and uses PyTorch's official
`DistributedSampler` without K-repeat geometry.

| Property | DistributedKRepeat | GroupContiguous | GroupDistributed | GroupTiled |
|----------|--------------------|-----------------|------------------|------------|
| **Group placement** | Arbitrary across ranks/batches | Complete group on one rank | Complete groups in one global microbatch | Complete groups in the smallest global-microbatch window |
| **Window** | Unbounded until acquisition end | Rank-local K rows | 1 global batch | `K / gcd(W*B, K)` global batches |
| **Constraints** | Base epoch tiling | Base tiling + `M % W == 0` | `K <= W*B` and `(W*B) % K == 0` | Every positive `K`, `W`, and `B`; M closes complete windows |
| **Primary use** | Legacy full-acquisition feedback | Lowest-communication rank-local feedback | DGPO/TDM-R1 whole-group microbatches | Streamed group feedback without padding M to W |

### Offline Dataset Sampler

SFT and offline DPO always use `torch.utils.data.DistributedSampler`, call
`set_epoch(data_epoch)`, and exhaust the finite loader once per data epoch. Every source weight is
`1`; `gradient_accumulation_steps` is explicit; and each rank's batch count must be divisible by
it. The loader is not passed to `Accelerator.prepare()`. With the official sampler's default
`drop_last=False`, a non-divisible global tail is repeated deterministically to equalize rank
lengths; the complete resulting loader traversal, rather than global sample uniqueness, defines
the epoch.

---

## How Each Sampler Works

### DistributedKRepeatSampler

1. Select `M` unique indices from the dataset (deterministic via seed + epoch).
2. Repeat each index `K` times → `M * K` total samples.
3. **Shuffle all `M * K` samples globally** (breaking group locality).
4. Partition into iterations of size `W * B` (world_size * batch_size).
5. Each rank takes its slice: `[offset + rank * B : offset + rank * B + B]`.

**Result**: A single group's K copies are **scattered** across multiple ranks and multiple batches. Group-wise operations (advantage normalization, groupwise rewards) require **all-gather** or similar cross-rank communication.

### GroupContiguousSampler

1. Select `M` unique indices from the dataset (same deterministic logic).
2. Shuffle group order (not individual samples within groups).
3. **Partition groups across ranks**: rank `r` gets groups `[r * (M/W) : (r+1) * (M/W)]`.
4. Expand each group's index by repeating it `K` times, **keeping groups contiguous**.
5. Each rank yields batches of size `B` from its local contiguous block.

**Result**: All K copies of any given group reside on a **single rank**. Group-wise reward computation and advantage estimation can be performed locally without cross-rank communication.

### GroupDistributedSampler

1. Select `M` unique indices from the dataset (same deterministic logic).
2. Shuffle group order (all ranks see the same permutation).
3. Pack `W * B / K` complete groups into each global microbatch.
4. Deal rank-major slices of size B. When `K % W == 0`, retain the legacy layout where every
   rank receives `K / W` copies of every group.

**Result**: Every global microbatch contains only complete K-groups. In the packed layout a rank
may see only a subset of those groups, so DGPO/TDM-R1 gather the small integer
`(source_id, unique_id)` rows to build a shared dense group-id space, then use
`scatter_add + accelerator.reduce` for the group loss. The equal-share layout retains the old
rank-identical group sequence and derives dense ids locally.

### GroupTiledSampler

Let `Q = W * B` and `d = gcd(Q, K)`. One global tile contains:

```text
global_batches_per_tile = K / d
groups_per_tile         = Q / d
```

The sampler flattens `groups_per_tile` complete K-groups, then deals each consecutive global
batch in rank-major slices. A group may cross microbatch boundaries, but never a tile boundary.
When `K` divides `Q`, this degenerates to a one-global-batch layout. When `K > Q`, one group spans
multiple global batches.

This layout is valid for objectives whose per-sample optimizer loss only needs advantages after a
complete group window. Objectives such as DGPO and TDM-R1 that need a complete group inside every
optimizer microbatch deliberately reject it through their sampler selection contract.

---

## Geometric Constraints

Define the following variables:

| Symbol | Meaning | Config field |
|--------|---------|-------------|
| `M` | Unique samples per epoch | `training_args.unique_sample_num_per_epoch` |
| `K` | Group size (repeats per sample) | `training_args.group_size` |
| `W` | World size (number of GPUs/ranks) | `accelerator.num_processes` |
| `B` | Per-device batch size | `training_args.per_device_batch_size` |
| `G` | Gradient steps per epoch | `training_args.gradient_step_per_epoch` |

### Base Constraint (All Samplers)

The constraint depends on whether `gradient_accumulation_steps` is set manually or derived automatically.

**Auto mode** (`gradient_accumulation_steps: "auto"`):
```
M * K  ≡  0  (mod W * B * G)
```
**Why**: The total sample count `M * K` must be evenly divisible into `G` gradient steps, each consisting of `(M * K) / G` samples distributed across `W` ranks with batch size `B`. The auto-adjustment step size is:

```python
step = (W * B * G) // gcd(K, W * B)
M_adjusted = ceil(M / step) * step
```

**Manual mode** (`gradient_accumulation_steps` set to an integer):
```
M * K  ≡  0  (mod W * B)
```
`G` is excluded because `gradient_step_per_epoch` plays no role when GAS is explicitly provided. The step size is:

```python
step = (W * B) // gcd(K, W * B)
M_adjusted = ceil(M / step) * step
```

Both use **GCD-based** rounding — finding the smallest multiple that satisfies divisibility.

### Additional Constraint (GroupContiguousSampler Only)

```
M  ≡  0  (mod W)
```

**Why**: Groups are partitioned across ranks by assigning `M / W` complete groups to each rank. If `M` is not divisible by `W`, some ranks would get fewer groups, causing uneven workload and potential deadlocks.

Combined with the base constraint, the effective step for GroupContiguousSampler is:

**Auto mode**:
```python
base_step = (W * B * G) // gcd(K, W * B)
step = lcm(base_step, W)
M_adjusted = ceil(M / step) * step
```

**Manual mode**:
```python
base_step = (W * B) // gcd(K, W * B)
step = lcm(base_step, W)
M_adjusted = ceil(M / step) * step
```

Both use **LCM-based** rounding — strictly more constrained than the base case.

### Additional Constraints (GroupDistributedSampler Only)

```
K  <=  W * B
(W * B)  ≡  0  (mod K)
```

**Why**: A global microbatch of `W * B` samples must tile into one or more complete groups of size
K. A group may be smaller than W; it then occupies a subset of ranks in that microbatch.

The alignment function `_align_for_group_distributed` validates K without changing it, then aligns
M with the shared GCD-based rule.

### GroupTiledSampler Window Constraint

`GroupTiledSampler` imposes no divisibility relation between K and the global batch Q. It aligns M
to a multiple of `Q / gcd(Q, K)`, which is exactly the number of unique groups needed to close one
tile. The shared `_base_unique_sample_step()` already expresses this rule (and includes the
gradient-step multiplier in auto-GAS mode).

### Alignment Location

Sampler alignment is implemented in `Arguments._align_batch_geometry()` in `hparams/args.py`.
This method runs after `_resolve_sampler_type()` determines which sampler to use and selects the
appropriate rounding strategy.

### Derived Values

After `M` is adjusted, `_align_batch_geometry()` computes:

```python
num_batches_per_epoch = (M * K) // (W * B)
```

Then, in **auto mode** only:
```python
gradient_accumulation_steps = max(1, num_batches_per_epoch // G)
```

Then `Arguments.__post_init__` applies the per-timestep multiplier, also in **auto mode** only:

```python
gradient_accumulation_steps *= num_train_timesteps  # all trainers (via get_num_train_timesteps())
```

#### Manual ``gradient_accumulation_steps``

When the user explicitly sets ``gradient_accumulation_steps`` to an integer
(not ``"auto"``), the automatic derivation is bypassed:

- ``_align_batch_geometry()`` still adjusts ``M`` but only enforces sampler
  constraints (``M*K ≡ 0 (mod W*B)``), excluding ``G`` from the divisor.
- The ``× num_train_timesteps`` multiplier is skipped.
- The user-provided value is passed directly to ``Accelerator``.
- ``gradient_step_per_epoch`` is ignored for accumulation computation.

---

## Sampler Selection Logic

### User-Facing Parameter: `data_args.sampler_type`

The `sampler_type` field in `DataArguments` allows users to explicitly choose a strategy:

| Value | Behavior |
|-------|----------|
| `"auto"` (default) | Resolve from the algorithm's `SamplerSelectionContract` and current geometry |
| `"distributed_k_repeat"` | Arbitrary placement; full-acquisition group feedback only |
| `"group_contiguous"` | All K members on one rank |
| `"group_distributed"` | Complete groups in every global microbatch |
| `"group_tiled"` | Complete groups in a GCD-derived global-batch window |

### Resolution Logic: `Arguments._resolve_sampler_type()`

Sampler properties live in dependency-neutral `SamplerLayoutContract` values; algorithm
requirements live in each training argument class's `SamplerSelectionContract`. Resolution never
branches on a trainer name:

1. An explicit sampler is preserved when its group placement and geometry satisfy the algorithm
   contract; otherwise parsing fails instead of silently rewriting user intent.
2. `auto` tries the algorithm's ordered placement preferences and picks the first geometrically
   valid sampler.
3. Flexible trainers retain the legacy preference for rank-local groups, with the old
   `distributed_k_repeat` no-padding fallback when rewards are synchronous.
4. DGPO declares global-batch placement only. TDM-R1 declares rank-local/global-batch placement
   and requires complete groups per optimizer microbatch. Online DPO dynamically declares
   rank-local placement while reward/optimization overlap is enabled.
5. Async pointwise rewards are valid for cross-rank layouts. Async groupwise reward model classes
   are rejected after model resolution unless groups are rank-local.

Reward/optimization overlap adds a separate `RewardOptimizationOverlapContract`: it intersects
the trainer's supported scheduling modes with sampler placements. The sampler decides where a
group closes; the objective decides whether that closure is sufficient for early optimization.

### Sampler Factory (`data_utils/sampler_loader.py`)

```python
SAMPLER_REGISTRY = {
    "distributed_k_repeat": DistributedKRepeatSampler,
    "group_contiguous": GroupContiguousSampler,
    "group_distributed": GroupDistributedSampler,
    "group_tiled": GroupTiledSampler,
}
sampler_cls = SAMPLER_REGISTRY[config.data_args.sampler_type]
```

---

## Initialisation Sequence

The `Arguments.__post_init__` pipeline for sampler and batch geometry:

```
Arguments.__post_init__()
  ├─ _resolve_scheduler_sde_defaults()   # Fill sde_steps / num_sde_steps
  ├─ _resolve_sampler_type()             # Choose sampler → write data_args.sampler_type
  ├─ _align_batch_geometry()             # Align M + compute num_batches; derive GAS (auto mode only)
  └─ grad_accum *= num_train_timesteps   # Auto mode only: per-timestep multiplier (all algorithms)
```

`TrainingArguments.__post_init__` sets a placeholder value for `num_batches_per_epoch` and,
in auto mode, a placeholder for `gradient_accumulation_steps`. Both are overwritten by
`_align_batch_geometry()`. When `gradient_accumulation_steps` is manually set to an integer,
`_manual_gradient_accumulation_steps` is set to `True` and the value is preserved unchanged
throughout the rest of the initialisation sequence.

---

## When to Use Which Sampler

### Use GroupContiguousSampler (preferred, auto-selected when constraints are met) when:
- The geometric constraints `M % W == 0` and `(M/W)*K % B == 0` are satisfiable
- You want to minimise cross-rank communication
- An async groupwise reward must see all K members on one rank
- Online DPO overlap must transform a complete local group into a preference pair

### Use GroupDistributedSampler when:
- The objective itself requires complete groups in each global optimizer microbatch
- `K <= W*B` and `K` divides `W*B`
- DGPO or cross-rank TDM-R1 needs shared group logits/noise with bounded communication

### Use GroupTiledSampler when:
- Reward/advantage feedback can close over several optimizer microbatches
- Padding M to a multiple of W would add substantial rollout work
- K does not divide `W*B`, or K is larger than one global batch
- Rewards are pointwise when their async computation crosses ranks

### Use DistributedKRepeatSampler (fallback) when:
- The `group_contiguous` geometric constraints cannot be satisfied with the given M/W/K/B
- You want maximum flexibility in parameter choices (fewer constraints on `M`)
- GPU memory is limited and you cannot afford the M-padding required by `group_contiguous`

---

## Gather Logic Compatibility

`AdvantageProcessor` derives locality from the sampler contract rather than checking a sampler
name. Canonical group identity is the exact int64 pair `(source_id, unique_id)`; never pack it into
a float reward payload because prompt hashes may exceed 2^53.

### AdvantageProcessor Communication Optimization

| Operation | rank-local | ordinary cross-rank | streamed cross-rank |
|-----------|------------|---------------------|---------------------|
| Group identities | local NumPy grouping | separate int64 gather | one int64 gather per acquisition, cached by work unit |
| Rewards | local | float32 gather | no per-sample gather on optimizer work units |
| Advantage stats | local | computed from gathered rows | one packed float64 SUM of `(invalid, count, sum, sum_sq)` per work unit |
| Final metrics | distributed scalar stats as needed | local over gathered acquisition | one acquisition-level reward gather after optimization |

The streaming reduction supports both built-in feedback orders when `global_std=false`:

- `sum`: aggregate weighted rewards, then normalize each complete group.
- `gdpo`: normalize every reward within each complete group, then aggregate weights.

Custom reducers and acquisition-wide statistics deliberately remain full-acquisition barriers.

The `AdvantageProcessor` is instantiated in `BaseTrainer._init_reward_model()` with `sampler_type=self.config.data_args.sampler_type`. Reward-based trainers (GRPO, GRPOGuard, NFT, AWM, DPO, DGPO, CRD) delegate advantage computation to `self.advantage_processor.compute_advantages()` via their own `compute_advantages()` method, invoked from `prepare_feedback()` after each `sample()` epoch (see `guidance/workflow.md` for `sample` → `prepare_feedback` → `optimize`). The distillation trainer `diffusion-opd` is the exception: its `prepare_feedback()` is a no-op and it does not use `AdvantageProcessor`. DPO forms chosen/rejected pairs at the start of `optimize()`, not in `prepare_feedback()`. DGPO handles group loss in its own `_compute_group_dgpo_loss()` via `scatter_add + reduce`.

Reward-model request batches are independent of optimizer work-unit boundaries. A pointwise request
may contain rows from adjacent work units; its future gates every work unit containing one of
those rows. Groupwise async requests remain keyed by the source-aware group identity.

For multi-source reward-overlap runs, `WeightedSourceBatchScheduler` shuffles source blocks rather
than individual source batches. One block is the layout's smallest group-complete window:
`K / gcd(B, K)` batches for rank-local placement, one batch for global-batch placement, and
`K / gcd(W*B, K)` batches for global-tile placement. Per-source alignment makes every source quota
an exact number of these blocks. The legacy non-overlap schedule keeps one-batch blocks.

---

## Validation Errors

GroupContiguousSampler raises explicit errors if constraints are violated:

1. **`M % W != 0`**: `"unique_sample_num ({M}) must be divisible by num_replicas ({W})"`
2. **`(M/W * K) % B != 0`**: `"groups_per_rank * group_size ({...}) must be divisible by batch_size ({B})"`

These are caught at sampler construction time. The auto-adjustment in `_align_batch_geometry()` should prevent (1) from ever triggering in normal usage, but manual config overrides can still violate it.

---

## Impact on Other Components

- **Constraint #9 in [`../constraints.md`](../constraints.md)**: No train dataloader is prepared via `accelerator.prepare()`; its selected grouped or official sampler owns distribution.
- **RewardProcessor**: groups by `(source_id, unique_id)` so equal prompt hashes from independent
  sources never share reward context.
- **AdvantageProcessor**: source-aware grouping and weights use the same canonical identity;
  streamed cross-rank work reuses acquisition metadata.
- **Bagel/packed adapters**: `AcquisitionManifest` records original rollout batch boundaries.
  Scheduling may reorder complete work units but cannot split or repack a bsz>1 forward.

---

## YAML Configuration Example

```yaml
data:
  dataset_dir: data/my_dataset
  sampler_type: auto  # also: distributed_k_repeat / group_contiguous / group_distributed / group_tiled
```

## Cross-refs

- UP: [`constraints.md` #9](../constraints.md#9-accelerator-prepare-scope), [`constraints.md` #9a](../constraints.md#9a-sampler-geometric-constraints), [`constraints.md` #9c](../constraints.md#9c-rewardoptimization-overlap), [Architecture Execution Pipelines](../architecture.md#execution-pipelines), [Architecture Advantage Computation](../architecture.md#advantage-computation)
