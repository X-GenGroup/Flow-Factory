# Async Rewards

Async rewards overlap Stage 4 (reward computation) with Stage 3 (rollout). Each reward entry with
`async_reward: true` gets its own worker-thread pool, and samples are scored while the adapter keeps
generating. Advantages and optimization still wait for every reward, so the stage order in the
[training workflow](../workflow.md) is otherwise unchanged. To also overlap rewards with optimization,
see [Reward/optimization overlap](reward_optimization_overlap.md).

## Table of Contents

- [When to Use](#when-to-use)
- [Configuration](#configuration)
- [How It Works](#how-it-works)
- [Worker Lifecycle](#worker-lifecycle)
- [Notes](#notes)

## When to Use

Async rewards help when reward computation waits on I/O, such as an HTTP reward server or a served
VLM judge. A local GPU reward model gains nothing and competes with rollout for the device.

| Scenario | Setting |
|---|---|
| Remote reward server or HTTP API | `async_reward: true`, `num_workers: 4` or more, `device: cpu` |
| Served VLM judge (OpenAI-compatible API) | `async_reward: true`, `num_workers: 4` or more |
| Local GPU reward model | `async_reward: false` (default) |

## Configuration

Async and synchronous rewards can be mixed in one run:

```yaml
rewards:
  # Async: HTTP reward with concurrent requests
  - name: "remote_aesthetic"
    reward_model: "flow_factory.rewards.my_reward_remote.RemotePointwiseRewardModel"
    server_url: "http://localhost:8000"
    device: cpu           # keep response tensors off the training GPU
    batch_size: 16
    async_reward: true
    num_workers: 4        # up to four requests executing
    max_pending_requests: 8  # optional: bound queued plus executing requests

  # Sync: local GPU reward (default)
  - name: "pick_score"
    reward_model: "PickScore"
    batch_size: 16
```

[`examples/template/sd3_5/async_reward.yaml`](../../examples/template/sd3_5/async_reward.yaml) is a
complete GRPO config that mixes a synchronous PickScore reward with an async VLM judge.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `async_reward` | `bool` | `false` | Compute this reward in worker threads during rollout |
| `num_workers` | `int` | `1` | Concurrent tasks for this reward; set it above 1 for I/O-bound rewards |
| `max_pending_requests` | positive `int` or `null` | `null` | Optional limit on unfinished requests per reward. `null` preserves unrestricted submission within one finite acquisition; a positive value enables backpressure. |
| `batch_size` | `int` | `16` | Samples per pointwise request, which is also the async dispatch threshold |
| `device` | `cpu` or `cuda` | `cuda` | Where the reward model and its outputs live; use `cpu` for HTTP clients |

## How It Works

`RewardBuffer` splits the configured rewards into an async lane and a synchronous lane:

```text
rollout (training process)
  batch 0 ─► add_samples() ─► batch 1 ─► add_samples() ─► ... ─► last batch ─► add_samples()
                │                            │
                └─ submit ready tasks to each async reward's request lane

after rollout: finalize()
  1. compute synchronous rewards on the training process
  2. submit the remaining partial pointwise batches, then wait for every async task
  3. merge both lanes and attach the rewards to the samples
```

An async **pointwise** reward submits a task each time its pending samples reach its own
`batch_size`, so request size does not depend on the rollout batch size; the remaining partial batch
is submitted when the rollout ends. An async **groupwise** reward submits a task when all K samples
of a group have arrived on the rank, which requires a sampler that keeps every group on one rank
(`data.sampler_type: rank_local`, formerly `group_contiguous`); the trainer rejects other layouts
when it starts.

When configured, each lane admits at most `max_pending_requests` unfinished requests, including running work and
queued work. At capacity, submission waits for any request in that lane to finish. Workers complete
requests without needing the training thread to collect their results, so a producer can make
progress before feedback or optimization begins. Successful, failed and cancelled requests all
release capacity; failures are raised only when feedback collects results, preserving the existing
distributed error-agreement boundary. A slow lane can delay acquisition, but cannot occupy another
lane's workers.

This bound controls request backlog, not the full rollout's sample storage: completed reward rows
and sample references remain available for tile reconstruction and final acquisition statistics.
Existing configs retain unrestricted submission within each finite rollout. A positive limit can
reduce request backlog but can also move reward waiting into rollout and reduce overlap with later
optimization. Choose it for the workload's service capacity; it is not a throughput optimization.
`num_workers` and an explicit `max_pending_requests` must be positive integers.
The default is omitted from the exported configuration to preserve existing exact-checkpoint
identities. A positive limit is exported and identity-locked because it can affect ready-tile order.

With `num_workers > 1`, requests to an I/O-bound reward run in parallel, because Python releases the
GIL during network I/O:

```text
num_workers=1 (serial):     [request 500 ms] [request 500 ms] [request 500 ms]  → ~1500 ms
num_workers=4 (concurrent): [request 500 ms]                                    → ~500 ms
                            [request 500 ms]
                            [request 500 ms]
```

Evaluation keeps one buffer per evaluation dataset and finalizes it the same way, so async
evaluation rewards overlap evaluation sampling.

## Worker Lifecycle

`RewardBuffer.clear()` cancels queued work, drains any running work and clears cycle references
while retaining the same executor pools. Streaming seals further submission; an aborted cycle must
be cleared before reuse. `BaseTrainer.start()` owns terminal cleanup for training and evaluation
buffers, draining workers on success and ordinary failure. A closed buffer cannot be reopened;
construct a new trainer/buffer for a new run. Custom `cleanup` overrides must accept the keyword
`wait` and close every resource they own.

Interrupt cleanup remains nonblocking for the CLI's existing process-exit path. Python cannot stop
an already running request: library callers that catch an interrupt can release their external
requests and call `trainer.cleanup(wait=True)` to join the same workers. Configure finite timeouts
in remote reward clients. Queued tasks are cancelled in both shutdown modes; teardown failures do
not replace the original training failure.

## Notes

- **Group identity** is `(source_id, unique_id)`, so equal prompt hashes from different dataset
  sources never share a groupwise request or a normalization context.
- **Independent lanes**: each async reward has its own thread pool sized by its `num_workers`, so a
  slow reward cannot occupy another reward's threads.
- **`num_workers`** affects only async rewards; synchronous rewards always run on the training
  process.
- **Errors** raised in a worker thread are re-raised on the training process when the result is
  collected.
- **Remote output device**: set `device: cpu` for HTTP rewards so response tensors are not
  allocated on the training GPU from worker threads. Reward/optimization overlap requires it.
- **CUDA rewards**: an async reward on `cuda` runs on its own CUDA stream, ordered after the rollout
  batch that produced its inputs. It still shares the GPU with rollout.
- **Protocol compatibility**: `RemotePointwiseRewardModel` expects `GET /health` and `POST /compute`
  returning `{"rewards": [...]}`. OpenAI-compatible or RewardService `/score` endpoints need a thin
  custom reward or gateway; async scheduling does not translate wire protocols.
