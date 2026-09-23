# Sampler System

**Read when**: Editing `data_utils/sampler*`, hparams sampler/batch fields.

---

## Overview

Generation acquisition uses **K-Repeat Sampling** to create `K` copies of each unique prompt for
group-wise advantage estimation. Three framework samplers differ in how repeated samples are
distributed across ranks. Dataset acquisition is a separate path and uses PyTorch's official
`DistributedSampler` without K-repeat geometry.

| Property | DistributedKRepeatSampler | GroupContiguousSampler | GroupDistributedSampler |
|----------|--------------------------|----------------------|------------------------|
| **Distribution** | Shuffled globally; same group's K copies spread across different ranks | Contiguous; all K copies of a group stay on the **same rank** | Complete K-groups packed into each global microbatch; equal-share layout retained when `K % W == 0` |
| **Cross-rank communication** | Required for group-wise reward aggregation | Not required — each rank holds complete groups | Packed layout: integer UID gather plus `scatter_add + reduce`; equal-share layout: reduce only |
| **Geometric constraints** | 1 constraint (base) | 2 constraints (base + divisibility) | `K <= W*B` and `(W*B) % K == 0` |
| **Auto-adjustment** | GCD-based rounding | LCM-based rounding (stricter) | K is preserved; M uses base GCD alignment |
| **Use case** | Fallback when geometric constraints for group_contiguous are unsatisfied | Default when constraints are met (minimal communication) | DGPO and cross-rank TDM-R1 group loss |

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
may see only a subset of those groups, so DGPO/TDM-R1 gather the small integer UID vector to build
a shared dense group-id space, then use `scatter_add + accelerator.reduce` for the group loss.
The equal-share layout retains the old rank-identical UID sequence and derives dense ids locally.

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

### Alignment Location

Both alignment strategies are implemented in `Arguments._align_batch_geometry()` in `hparams/args.py`. This method runs after `_resolve_sampler_type()` determines which sampler to use, and selects the appropriate rounding strategy accordingly.

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

The `sampler_type` field in `DataArguments` (`hparams/data_args.py`) allows users to explicitly choose a sampler strategy:

| Value | Behavior |
|-------|----------|
| `"auto"` (default) | Prefer `group_contiguous` (minimal communication); fall back to `distributed_k_repeat` when geometric constraints cannot be satisfied. DGPO overrides to `group_distributed`. |
| `"distributed_k_repeat"` | Force use of `DistributedKRepeatSampler` (fewer geometric constraints, extra all-gather communication) |
| `"group_contiguous"` | Force use of `GroupContiguousSampler` (all K copies on same rank, stricter constraints) |
| `"group_distributed"` | Force complete-group global microbatch packing (DGPO/TDM-R1) |

### Resolution Logic: `Arguments._resolve_sampler_type()`

The `_resolve_sampler_type()` method in `hparams/args.py` resolves the final sampler type and writes it back to `data_args.sampler_type`:

```python
# 1. Detect async rewards (any reward config with async_reward=True)
self._has_async_rewards = any(
    getattr(cfg, 'async_reward', False)
    for cfg in all_reward_configs
)
user_choice = self.data_args.sampler_type
trainer_type = str(training_args.trainer_type).lower()

# 2. TDM-R1 resolves separately: ordinary async rewards require group_contiguous,
#    while overlap may use group_distributed and validates pointwise-only rewards.
if trainer_type == "tdm-r1":
    return self._resolve_tdm_r1_sampler_type(user_choice)

# 3. Async override: a user-requested distributed_k_repeat OR group_distributed
#    is forced to group_contiguous when async rewards are on (DGPO is exempt).
if (user_choice in {"distributed_k_repeat", "group_distributed"}
        and self._has_async_rewards and trainer_type != "dgpo"):
    self.data_args.sampler_type = "group_contiguous"

# 4. "auto" (non-DGPO): default to group_contiguous; only pick distributed_k_repeat
#    when groups-per-rank FAILS but local batch tiling holds. Otherwise stay on
#    group_contiguous and let _align_batch_geometry() pad M to satisfy constraints.
if user_choice == "auto" and trainer_type != "dgpo":
    groups_per_rank_ok = (m % world_size == 0)
    local_batch_tiling_ok = (m // world_size * K % B == 0)
    if not groups_per_rank_ok and local_batch_tiling_ok:
        self.data_args.sampler_type = "distributed_k_repeat"
    else:
        self.data_args.sampler_type = "group_contiguous"

# 5. DGPO always forces group_distributed.
if trainer_type == "dgpo" and self.data_args.sampler_type != "group_distributed":
    self.data_args.sampler_type = "group_distributed"
```

**Key behaviors**:
- DGPO trainer forces `group_distributed` regardless of user setting (via `_resolve_sampler_type`)
- TDM-R1 resolves between the two group-preserving samplers. Async reward without overlap requires
  `group_contiguous`; overlap may retain `group_distributed`, whose later validation accepts only
  pointwise rewards.
- `"auto"` defaults to `group_contiguous`; it picks `distributed_k_repeat` **only** when groups-per-rank fails (`M % W != 0`) **but** local batch tiling holds (`(M/W)*K % B == 0`). When both fail it stays on `group_contiguous` and `_align_batch_geometry()` pads `M`.
- Async rewards force `group_contiguous` when the user requested `distributed_k_repeat` **or** `group_distributed` (DGPO exempt), emitting a warning
- User can manually select `group_contiguous` without async rewards (e.g., to reduce cross-rank communication)

### Sampler Factory (`data_utils/sampler_loader.py`)

```python
SAMPLER_REGISTRY = {
    "distributed_k_repeat": DistributedKRepeatSampler,
    "group_contiguous": GroupContiguousSampler,
    "group_distributed": GroupDistributedSampler,
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
- Any reward model uses `async_reward=True` (automatically forced)
- You want to minimise cross-rank communication
- DPO trainer — all K copies needed on same rank for reliable pair formation

### Use DistributedKRepeatSampler (fallback) when:
- The `group_contiguous` geometric constraints cannot be satisfied with the given M/W/K/B
- You want maximum flexibility in parameter choices (fewer constraints on `M`)
- GPU memory is limited and you cannot afford the M-padding required by `group_contiguous`

---

## Gather Logic Compatibility

Both samplers are **fully compatible** with existing gather/reduce/advantage logic. The `AdvantageProcessor` (`advantage/advantage_processor.py`) automatically selects the communication strategy based on `data_args.sampler_type`:

### AdvantageProcessor Communication Optimization

| Operation | `distributed_k_repeat` | `group_contiguous` | `group_distributed` |
|-----------|----------------------|-------------------|---------------------|
| Gather rewards | Single `accelerator.gather()` (packed tensor) | **Skipped** — local data used directly | Single `accelerator.gather()` (same as `distributed_k_repeat`) |
| Gather unique_ids | Packed into same gather call | **Skipped** — local `np.unique()` | Packed into same gather call |
| Group construction | `np.unique()` over W×B items | `np.unique()` over B items only | `np.unique()` over W×B items |
| Scatter advantages | `reshape(W, B)[rank]` | **Direct return** — already local | `reshape(W, B)[rank]` |

> `group_on_same_rank` is `True` **only** for `group_contiguous` (`advantage/advantage_processor.py`); `group_distributed` takes the **same gather path** as `distributed_k_repeat`. DGPO builds a global dense UID space from each global microbatch, then uses `scatter_add` + `accelerator.reduce(SUM)` + `sigmoid` in the **DGPO loss** (`trainers/rl/dgpo.py`), not in `AdvantageProcessor`.

The `AdvantageProcessor` is instantiated in `BaseTrainer._init_reward_model()` with `sampler_type=self.config.data_args.sampler_type`. Reward-based trainers (GRPO, GRPOGuard, NFT, AWM, DPO, DGPO, CRD) delegate advantage computation to `self.advantage_processor.compute_advantages()` via their own `compute_advantages()` method, invoked from `prepare_feedback()` after each `sample()` epoch (see `guidance/workflow.md` for `sample` → `prepare_feedback` → `optimize`). The distillation trainer `diffusion-opd` is the exception: its `prepare_feedback()` is a no-op and it does not use `AdvantageProcessor`. DPO forms chosen/rejected pairs at the start of `optimize()`, not in `prepare_feedback()`. DGPO handles group loss in its own `_compute_group_dgpo_loss()` via `scatter_add + reduce`.

When `GroupContiguousSampler` is used:
1. **Groupwise Reward Computation** (`reward_processor.py`): `gather_samples()` → `group_samples()` → stride → compute → `all_reduce` → scatter — works correctly (gather collects redundant data but logic is sound)
2. **Advantage Computation** (`advantage/advantage_processor.py`): All K copies on same rank → no gather/scatter needed → `group_on_same_rank=True` path
3. **Async Reward (RewardBuffer)**: Still requires `GroupContiguousSampler` — enforced by override logic

---

## Validation Errors

GroupContiguousSampler raises explicit errors if constraints are violated:

1. **`M % W != 0`**: `"unique_sample_num ({M}) must be divisible by num_replicas ({W})"`
2. **`(M/W * K) % B != 0`**: `"groups_per_rank * group_size ({...}) must be divisible by batch_size ({B})"`

These are caught at sampler construction time. The auto-adjustment in `_align_batch_geometry()` should prevent (1) from ever triggering in normal usage, but manual config overrides can still violate it.

---

## Impact on Other Components

- **Constraint #9 in [`../constraints.md`](../constraints.md)**: No train dataloader is prepared via `accelerator.prepare()`; its selected grouped or official sampler owns distribution.
- **RewardProcessor**: When GroupContiguousSampler is active, groupwise rewards can be computed locally per rank. When DistributedKRepeatSampler is active, the RewardProcessor must gather group members across ranks.
- **AdvantageProcessor**: Automatically skips `accelerator.gather()` calls when `sampler_type == "group_contiguous"` (all group members already local); uses `all_reduce(count, sum, sum_sq)` for global_std (3 scalars). When `sampler_type == "distributed_k_repeat"`, packs all rewards + unique_ids into a single tensor for one `accelerator.gather()` call.

---

## YAML Configuration Example

```yaml
data:
  dataset_dir: data/my_dataset
  sampler_type: auto  # or "distributed_k_repeat" / "group_contiguous" / "group_distributed"
```

## Cross-refs

- UP: [`constraints.md` #9](../constraints.md#9-accelerator-prepare-scope), [`constraints.md` #9a](../constraints.md#9a-sampler-geometric-constraints), [Architecture Execution Pipelines](../architecture.md#execution-pipelines), [Architecture Advantage Computation](../architecture.md#advantage-computation)
