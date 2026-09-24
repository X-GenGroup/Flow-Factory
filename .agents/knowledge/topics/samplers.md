# Sampler System

**Read when**: Editing `data_utils/sampler*`, grouped identities, hparams sampler/batch fields,
or reward/optimization overlap topology.

---

## Ownership

Generation acquisition has four separate concerns:

1. `contracts/sampler.py` declares layout geometry and algorithm selection capabilities.
2. `data_utils/sampling_plan.py` selects `U` dataset rows, assigns exact group/member/sample IDs,
   and places the resulting `U * K` members on ranks.
3. `data_utils/sampler.py` is only the infinite epoch/batch-sampler facade over that immutable
   plan. Legacy sampler class names remain public aliases.
4. `utils/group_coordinator.py` scopes group-relative reward payloads, identity gathers, and
   statistic reductions to one rank, one subgroup, or the full world without changing global DDP
   optimization.

Dataset acquisition is separate: SFT and offline DPO use PyTorch's official
`DistributedSampler`, never the grouped generation planner.

The sampling plan, not prompt content, owns generated-sample identity. Every assignment carries:

- `group_id`: shared by exactly `K` comparison members;
- `group_member_id`: `0 .. K-1` inside that group;
- `sample_id`: unique inside the source-local acquisition stream;
- the independent dataset index used to fetch model input.

`SamplingDatasetView` transports these values through DataLoader workers as reserved columns.
`BaseTrainer.sample_batch()` removes them before adapter inference and attaches them one-to-one to
the returned `BaseSample`s. `BaseSample.unique_id` is the compatibility accessor for the planned
`sampling_group_id`; samples created outside a planned training loader retain the legacy content
fingerprint fallback. Canonical cross-source group identity remains exact int64
`(source_id, unique_id)`.

---

## Layouts

Let:

| Symbol | Meaning |
|---|---|
| `U` | unique groups per acquisition/epoch (`unique_sample_num_per_epoch`) |
| `K` | members per comparison group (`group_size`) |
| `W` | data-parallel world size |
| `B` | rank-local rollout/replay microbatch size |
| `Q` | global microbatch size, `W * B` |
| `R` | ranks per subgroup for `subgroup_tile` |

All layouts require `U * K` to divide into complete synchronized rank-local batches. The planner
expresses the stronger layout-specific requirement as a minimum `U` multiple and hparams aligns
to it before model allocation.

| Semantic name | Placement | Smallest group-complete window | Minimum `U` multiple |
|---|---|---|---|
| `global_random` | shuffle all `U*K` members globally | none before acquisition end | `Q / gcd(Q,K)` |
| `contiguous_shard` | flatten group-major, then give each rank one contiguous shard | none in general | `Q / gcd(Q,K)` |
| `rank_local` | every group belongs to one rank | `K / gcd(B,K)` batches | `W * B / gcd(B,K)` |
| `global_batch` | every global microbatch contains complete groups | 1 batch | `Q / K`; requires `K <= Q` and `Q % K == 0` |
| `global_tile` | groups close in an all-rank GCD tile | `K / gcd(Q,K)` batches | `Q / gcd(Q,K)` |
| `subgroup_tile` | groups close independently inside contiguous `R`-rank subgroups | `K / gcd(R*B,K)` batches | `(W/R) * (R*B / gcd(R*B,K))` |

`subgroup_tile` additionally requires `R > 0` and `W % R == 0`. It degenerates to
`rank_local` at `R=1` and to `global_tile` at `R=W`. Compared with `global_tile`, a smaller `R`
may lengthen the group-complete window, but group-relative reward/advantage communication moves
only within `R` ranks instead of all `W` ranks.

### Legacy aliases

The following config names and imports remain valid:

| Legacy name | Semantic layout |
|---|---|
| `distributed_k_repeat` | `global_random` |
| `group_contiguous` | `rank_local` |
| `group_distributed` | `global_batch` |
| `group_tiled` | `global_tile` |

`auto` continues to write legacy names today so existing printed configs, checkpoints, and tests do
not drift. Explicit semantic names are accepted by the same registry and resolve to the same
immutable layout contracts.

---

## Placement Details

### `global_random`

Select `U` distinct dataset rows with `seed + epoch`, expand each to `K` planned members, randomly
permute all members, then deal rank-major global microbatches. This has the fewest placement
constraints and the least locality. Group-relative feedback needs full-acquisition collection.

### `contiguous_shard`

Select and shuffle groups, flatten them group-major, split the flat acquisition into `W`
contiguous equal shards, then batch each shard. It preserves more sequential locality than random
placement but can split a group at a rank boundary, so it does not promise an incremental
group-complete window. `rank_local` is its stricter no-split specialization.

### `rank_local`

Partition shuffled groups across ranks, then flatten each rank's complete groups. A group may span
several local microbatches when `K > B`, but it never communicates for group normalization. Async
groupwise reward models and online-DPO overlap require this property.

### `global_batch`

Pack complete groups in every synchronized global microbatch. If `K % W == 0`, preserve the
historic striped/equal-share policy (`K/W` members of every group on every rank); otherwise deal
rank-major slices from complete packed groups. DGPO and cross-rank TDM-R1 require this stronger
microbatch contract.

### `global_tile`

With `d = gcd(Q,K)`, flatten `Q/d` complete groups and deal `K/d` consecutive global batches.
Groups may cross a microbatch but never the tile. When `K | Q`, this degenerates to
`global_batch` geometry, though it remains a distinct declared placement capability.

### `subgroup_tile`

Split ranks into deterministic contiguous groups `[0,R)`, `[R,2R)`, etc. In every synchronized
window, each subgroup receives `R*B/gcd(R*B,K)` distinct complete groups and closes them in
`K/gcd(R*B,K)` batches. Group IDs are globally distinct even though comparison collectives are
subgroup-scoped.

The optimizer, readiness agreement, timing reductions, and final DDP gradient synchronization
remain global. Synchronous groupwise reward payloads, group-relative identity gathers, and packed
`(count,sum,sum_sq)` reductions are subgroup-scoped. Async groupwise reward models still require
`rank_local`; pointwise async rewards work with subgroup tiles.

---

## Alignment

`Arguments._align_batch_geometry()` resolves the sampler first, asks its layout contract for the
minimum `U` multiple, and combines it with the optimizer-epoch step via `lcm`. There are no
sampler-name branches in alignment.

With automatic gradient accumulation, the existing base step also includes
`gradient_step_per_epoch`; with an explicit integer GAS it only enforces rank/batch divisibility.
After alignment:

```python
num_batches_per_epoch = U * K // (W * B)
```

Multi-source allocation gives every source an integer multiple of the same layout step. During
reward overlap, `WeightedSourceBatchScheduler` switches sources only after the selected layout's
group-complete window, including subgroup windows.

Distillation algorithms that deliberately reject automatic geometry changes still validate their
declared sampler contract rather than silently replacing it.

---

## Algorithm Capability Matrix

Algorithms declare requirements with `SamplerSelectionContract`; they do not select concrete
sampler classes in trainer logic.

| Family | Accepted topology |
|---|---|
| GRPO / DPPO / GRPO-Guard / NFT / AWM / CRD | flexible; all six layouts for ordinary acquisition, bounded layouts for overlap |
| Online DPO with overlap | `rank_local` (one complete local reward group becomes one pair) |
| DGPO | `global_batch` only |
| TDM-R1 | `rank_local` or `global_batch`, with complete groups in every optimizer microbatch |
| DMD2 / TDM and other generation-without-feedback paths | flexible layout, subject to their manual geometry contract |
| SFT / offline DPO | grouped sampler is bypassed; official finite `DistributedSampler` |

An explicit incompatible layout fails during config parsing. `auto` walks the algorithm's ordered
placement preferences; it never rewrites an explicit choice. `subgroup_tile` is explicit and
requires `data.sampler_subgroup_size`; auto does not guess a hardware subgroup boundary.

Reward/optimization overlap separately intersects:

- the algorithm's `RewardOptimizationOverlapContract`;
- the sampler's bounded group-complete window;
- the reducer's streaming capability (`global_std=false` for built-ins);
- adapter replay-batch preservation requirements.

`global_random` and `contiguous_shard` remain valid full-acquisition layouts but are intentionally
rejected for streamed overlap.

---

## Communication and Replay

`AdvantageProcessor` consumes a `GroupCoordinator` rather than branching on sampler names.

| Work | `rank_local` | global layouts | `subgroup_tile` |
|---|---|---|---|
| Group identity | local | int64 gather over `W` | int64 gather over `R` |
| Synchronous groupwise reward payloads | local | gather/reduce over `W` | gather/reduce over `R` |
| Pointwise reward rows for advantage | local | gather over `W` | gather over `R` |
| Streamed group stats | local | one packed SUM over `W` | one packed SUM over `R` |
| Global logging stats / DDP | global scalar reductions | global | global |

During overlap, the trainer gathers exact identities once per acquisition, caches dense mappings
per work unit, and does not gather per-sample rewards for each tile. Built-in `sum` and `gdpo`
reducers use one packed `(invalid,count,sum,sum_sq)` reduction per runnable tile when
`global_std=false`.

Hot-path sample transport is tensor-first. Optional scalar metadata, string byte lengths, planned
identity, and uniform int64 fields such as `prompt_ids` share one int64 GPU gather. All annotated
string fields then share at most one padded UTF-8 uint8 gather; an all-empty/`None` payload skips
that second collective. Same-dtype dense tensors are packed where the transfer/copy tradeoff is
favorable. Python object gathering remains only for genuinely non-tensorizable custom fields and
distributed error details after a tensor failure flag has already fired. Online DPO transports
only its required `advantage` extra rather than the complete reward dictionaries and applicability
sets. The fixed tile header also carries UID-preparation failure state, avoiding a separate success-
path error reduction.
For non-overlapped online DPO with cross-rank groups, trajectory gathering and pair sharding use
the same `GroupCoordinator`: global layouts span `W`, while `subgroup_tile` confines both to `R`.
The planned `rank_local` geometry and explicit cross-rank round-robin sharding both produce equal
optimizer work on every participating rank, so DPO does not run a second pair-count collective or
pickle-broadcast a padding pair. Only the final pair statistics and DDP optimization remain global.

Reward request batch size is independent of sampler tiles. One pointwise request may span several
optimizer work units; each future gates every work unit containing one of its rows. Groupwise
requests remain keyed by canonical `(source_id, unique_id)`.

Bagel and other pack-composition-dependent adapters remain orthogonal to placement. An
`AcquisitionManifest` records original bsz>1 rollout microbatches; reward tiles may reorder whole
work units but may not split or repack those microbatches. Planned identity is attached after the
adapter returns and therefore does not alter packed forward inputs.

---

## Configuration

```yaml
data:
  sampler_type: subgroup_tile
  sampler_subgroup_size: 8  # required here; must divide WORLD_SIZE
```

For existing configs:

```yaml
data:
  sampler_type: auto
  sampler_subgroup_size: null
```

## Cross-references

- [`constraints.md` #9](../constraints.md#9-accelerator-prepare-scope)
- [`constraints.md` #9a](../constraints.md#9a-sampler-geometric-constraints)
- [`constraints.md` #9c](../constraints.md#9c-rewardoptimization-overlap)
- [Architecture: execution pipelines](../architecture.md#execution-pipelines)
- [Architecture: advantage computation](../architecture.md#advantage-computation)
