# Sampler System

**Read when**: Editing `data_utils/sampler*`, hparams sampler/batch fields.

---

## Overview

Flow-Factory uses **K-Repeat Sampling** (Stage 2 of the pipeline) to generate `K` copies of each unique prompt for group-wise advantage estimation. Two sampler implementations exist, differing in **how repeated samples are distributed across ranks**.

| Property | DistributedKRepeatSampler | GroupContiguousSampler |
|----------|--------------------------|----------------------|
| **Distribution** | Shuffled globally; same group's K copies spread across different ranks | Contiguous; all K copies of a group stay on the **same rank** |
| **Cross-rank communication** | Required for group-wise reward aggregation | Not required — each rank holds complete groups |
| **Geometric constraints** | 1 constraint (base) | 2 constraints (base + divisibility) |
| **Auto-adjustment** | GCD-based rounding | LCM-based rounding (stricter) |
| **Use case** | Fallback when geometric constraints for group_contiguous are unsatisfied | Default when constraints are met (minimal communication) |

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

### Base Constraint (Both Samplers)

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
gradient_accumulation_steps *= num_train_timesteps  # GRPO, NFT, AWM, DPO
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
| `"auto"` (default) | Prefer `group_contiguous` (minimal communication); fall back to `distributed_k_repeat` when geometric constraints (`M % W == 0`, `(M/W)*K % B == 0`) cannot be satisfied |
| `"distributed_k_repeat"` | Force use of `DistributedKRepeatSampler` (fewer geometric constraints, extra all-gather communication) |
| `"group_contiguous"` | Force use of `GroupContiguousSampler` (all K copies on same rank, stricter constraints) |

### Resolution Logic: `Arguments._resolve_sampler_type()`

The `_resolve_sampler_type()` method in `hparams/args.py` resolves the final sampler type and writes it back to `data_args.sampler_type`:

```python
# 1. Detect async rewards
self._has_async_rewards = any(
    getattr(cfg, 'async_reward', False)
    for cfg in all_reward_configs
)

# 2. Resolve sampler type
user_choice = self.data_args.sampler_type
if user_choice != "auto":
    # Only override if distributed_k_repeat + async rewards (hard conflict)
    if user_choice == "distributed_k_repeat" and self._has_async_rewards:
        self.data_args.sampler_type = "group_contiguous"
else:
    # Prefer group_contiguous; fall back if geometric constraints fail
    if m % world_size == 0 and (m // world_size * K) % B == 0:
        self.data_args.sampler_type = "group_contiguous"
    else:
        self.data_args.sampler_type = "distributed_k_repeat"
```

**Key behaviors**:
- `"auto"` defaults to `group_contiguous` when constraints are met — minimises communication
- Falls back to `distributed_k_repeat` only when `M % W != 0` or `(M/W)*K % B != 0`
- Async rewards **always force** `group_contiguous`, even if user explicitly requests `distributed_k_repeat` (with a warning)
- User can manually select `group_contiguous` without async rewards (e.g., to reduce cross-rank communication)

### Sampler Factory (`data_utils/sampler_loader.py`)

```python
sampler_cls = (
    GroupContiguousSampler
    if config.data_args.sampler_type == "group_contiguous"
    else DistributedKRepeatSampler
)
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

| Operation | `distributed_k_repeat` | `group_contiguous` |
|-----------|----------------------|-------------------|
| Gather rewards | Single `accelerator.gather()` (packed tensor) | **Skipped** — local data used directly |
| Gather unique_ids | Packed into same gather call | **Skipped** — local `np.unique()` |
| Group construction | `np.unique()` over W×B items | `np.unique()` over B items only |
| Scatter advantages | `reshape(W, B)[rank]` | **Direct return** — already local |

The `AdvantageProcessor` is instantiated in `BaseTrainer._init_reward_model()` with `sampler_type=self.config.data_args.sampler_type`. All trainers (GRPO, GRPOGuard, NFT, AWM, DPO) delegate advantage computation to `self.advantage_processor.compute_advantages()` via their own `compute_advantages()` method, invoked from `prepare_feedback()` after each `sample()` epoch (see `guidance/workflow.md` for `sample` → `prepare_feedback` → `optimize`). DPO forms chosen/rejected pairs at the start of `optimize()`, not in `prepare_feedback()`.

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

- **Constraint #9 in [`../constraints.md`](../constraints.md)**: The dataloader is NOT prepared via `accelerator.prepare()` — both samplers handle distribution themselves.
- **RewardProcessor**: When GroupContiguousSampler is active, groupwise rewards can be computed locally per rank. When DistributedKRepeatSampler is active, the RewardProcessor must gather group members across ranks.
- **AdvantageProcessor**: Automatically skips `accelerator.gather()` calls when `sampler_type == "group_contiguous"` (all group members already local); uses `all_reduce(count, sum, sum_sq)` for global_std (3 scalars). When `sampler_type == "distributed_k_repeat"`, packs all rewards + unique_ids into a single tensor for one `accelerator.gather()` call.

---

## YAML Configuration Example

```yaml
data:
  dataset_dir: data/my_dataset
  sampler_type: auto  # or "distributed_k_repeat" / "group_contiguous"
```

## Cross-refs

- `constraints.md` #9, #9a (accelerator prepare scope, sampler geometric constraints)
- `architecture.md` "Six-Stage Training Pipeline" (Stage 2: K-Repeat Sampling)
- `architecture.md` "Advantage Computation" (communication path depends on sampler type)
