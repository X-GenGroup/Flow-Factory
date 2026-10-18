# Reward/Optimization Overlap

Reward/optimization overlap keeps every training reward asynchronous from rollout through
optimization. Rewards are submitted during rollout, as with [async rewards](async_reward.md), and
optimization then consumes complete reward *tiles* as soon as they are ready while later tiles are
still being scored. The cycle behaves like rollout followed directly by optimization, with feedback
running in the background. With `train.reward_optimization_overlap: false` (the default), the
[training workflow](../workflow.md) runs its stages in order as written.

## Table of Contents

- [How It Works](#how-it-works)
- [Requirements](#requirements)
- [Supported Trainers](#supported-trainers)
- [Configuration](#configuration)
- [Tile Scheduling](#tile-scheduling)
- [Tiles and Reward Requests](#tiles-and-reward-requests)
- [Metrics](#metrics)

## How It Works

The whole rollout still finishes before the first optimizer update, so one acquisition never mixes
samples from two policy versions. Async reward work submitted during rollout continues while
complete tiles enter optimization:

```text
rollout batch 0 ──► reward futures ───────────────────────────────┐
rollout batch 1 ──► reward futures ───────────────┐               │
...                                               ▼               ▼
rollout complete ──► globally ready tile 0 ──► optimize ──► next ready tile
```

One overlapped GRPO epoch, in the same form as the synchronous epoch in the
[training workflow](../workflow.md#putting-it-all-together):

```text
Epoch N
├── DataLoader (a bounded group-complete layout, e.g. rank_local)
├── Sampling: adapter.inference() per batch; add_samples() submits async reward requests
├── Seal: submit the last partial requests and plan tiles (complete groups and GAS windows)
├── Tile loop (every rank selects the same tile)
│   ├── Poll: which tiles have every reward on every rank?
│   ├── Resolve the selected tile's rewards and compute its group-relative advantages
│   └── Optimize the tile with the trainer's ordinary loss
└── Finish: wait for every request, then compute acquisition-wide advantage metrics
```

Evaluation does not stream: it finalizes its reward buffers before computing metrics.

## Requirements

The trainer validates these conditions when the config loads, before any model is allocated:

- the trainer declares overlap support for the selected scheduling mode and sampler layout (see
  [Supported Trainers](#supported-trainers));
- there is at least one training reward, and every training reward sets `async_reward: true`;
- every training reward sets `device: cpu`, so reward CUDA work cannot contend with optimization;
- the advantage reducer closes within complete groups, which for the built-in `sum` and `gdpo`
  aggregations means `train.global_std: false` (every trainer defaults it to `true`, so set it
  explicitly);
- `train.num_inner_epochs: 1` and `train.shuffle_samples: false`;
- each rank's acquisition forms complete groups, complete microbatches, and complete gradient
  accumulation windows.

An async groupwise reward additionally needs every group on one rank, so it works only with the
`rank_local` layout; cross-rank layouts accept pointwise rewards.

## Supported Trainers

| Trainer (`trainer_type`) | Sampler layouts (`data.sampler_type`) |
|---|---|
| `grpo`, `grpo-guard`, `dppo`, `nft`, `awm`, `crd` | `rank_local`, `global_batch`, `global_tile`, `subgroup_tile` |
| `dgpo` | `global_batch` |
| `dpo` (online DPO) | `rank_local` |
| `tdm-r1` | `rank_local`, `global_batch` |

`global_random` and `contiguous_shard` place group members anywhere, so they are rejected. SFT,
offline DPO, DiffusionOPD, DMD2, and reward-free TDM have no runtime reward to overlap and keep
their ordinary execution path. All supported trainers accept both scheduling modes.

## Configuration

```yaml
train:
  reward_optimization_overlap: true
  reward_optimization_overlap_mode: ready  # ready | ordered
  reward_optimization_overlap_poll_interval: 0.05
  advantage_aggregation: sum                 # sum or gdpo; both stream with global_std: false
  global_std: false
  num_inner_epochs: 1
  shuffle_samples: false

data:
  sampler_type: rank_local

rewards:
  - name: remote_quality
    reward_model: flow_factory.rewards.my_reward_remote.RemotePointwiseRewardModel
    device: cpu
    async_reward: true
    num_workers: 8
    batch_size: 8
    server_url: http://reward-router:8000
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `reward_optimization_overlap` | `bool` | `false` | Optimize complete async reward tiles while later rewards run |
| `reward_optimization_overlap_mode` | `ordered` or `ready` | `ready` | Preserve rollout order, or take the first globally ready tile |
| `reward_optimization_overlap_poll_interval` | `float` | `0.05` | Initial seconds between readiness polls |

## Tile Scheduling

`ordered` waits for the earliest outstanding tile and preserves rollout update order. `ready`
selects the lowest tile id among all globally ready tiles, bypassing a straggler but making update
order depend on reward latency. Both modes select the same tile on every rank: a tile is ready only
when every rank holds all of its rewards.

Each unsuccessful poll is a global synchronization point. The poll interval is the initial delay;
consecutive unsuccessful polls back off together, up to a bounded delay, and reset as soon as a
tile is selected.

The trainer may also do reward-independent optimizer preparation while rewards run, such as CRD
pass 1 or TDM-R1's fake-role updates.

## Tiles and Reward Requests

A tile is the smallest rank-local span of samples that closes both reward groups and gradient
accumulation. The trainer's tile geometry decides how reward groups become optimizer examples:

- Rank-local objectives close both K-groups and gradient accumulation within a tile; online DPO
  counts one preference pair per K-group.
- DGPO and `global_batch` TDM-R1 instead close groups in each global microbatch.
- TDM-R1 streams one rollout batch per tile while its surrogate gradients close across the full
  acquisition.

Pointwise reward requests are sized by each reward's own `batch_size`, independent of tiles. A
request may contain rows from adjacent tiles, and its result gates every tile that contains one of
those rows. Only the final partial request is submitted when sampling is sealed. This keeps remote
servers efficiently batched without changing group or gradient-accumulation boundaries.

For an adapter whose batched forward depends on pack composition, such as Bagel at
`per_device_batch_size > 1`, an `AcquisitionManifest` records every original rollout microbatch.
Ready scheduling may reorder whole tiles, but validation rejects any tile that would split or
repack one of those recorded batches.

## Metrics

Overlap emits the four `timing/` metrics described in the
[training workflow](../workflow.md#timing-metrics), so a run without overlap is a same-shape baseline.
It adds:

| Metric | Meaning |
|---|---|
| `timing/reward_overlap/stream_seconds` | Time from the completed rollout until the streamed reward/optimization cycle finishes. |
| `timing/reward_overlap/first_tile_seconds` | Time from the completed rollout until the first globally ready tile can optimize. |
| `timing/reward_overlap/wait_seconds` | Sleep time between polls with no globally selectable tile. |
| `timing/reward_overlap/coordination_seconds` | Local polling plus distributed readiness-coordination time. |
| `timing/reward_overlap/preparation_seconds` | Reward-independent optimizer preparation performed while async rewards run. |
| `timing/reward_overlap/optimization_started_while_rewards_pending_seconds` | Duration of optimizer calls launched while later reward tiles were still pending. A reward may finish during the call, so this measures scheduler exposure rather than exact hidden wall time. |
| `timing/reward_overlap/optimization_started_after_rewards_ready_seconds` | Duration of optimizer calls launched after all reward tiles were ready. |
| `timing/reward_overlap/optimization_started_while_rewards_pending_ratio` | Fraction of optimizer time whose calls began while later rewards were pending. |

Structural counters are training state rather than durations, so they stay under
`train/reward_overlap/`: `tile_count`, `samples_per_tile`, `poll_count`,
`readiness_collective_elements`, `out_of_order_tiles`, and `reused_group_metadata_batches`.
