# Workflow Guidance

## Table of Contents

- [Overview](#overview)
- [Stage 1: Data Preprocessing](#stage-1-data-preprocessing)
- [Offline Dataset Training](#offline-dataset-training)
- [Stage 2: K-Repeat Sampling](#stage-2-k-repeat-sampling)
- [Stage 3: Trajectory Generation](#stage-3-trajectory-generation)
- [Stage 4: Reward Computation](#stage-4-reward-computation)
- [Stage 5: Advantage Computation](#stage-5-advantage-computation)
- [Stage 6: Policy Optimization](#stage-6-policy-optimization)
- [Putting It All Together](#putting-it-all-together)

## Overview

Flow-Factory has one training kernel with orthogonal execution contracts. An algorithm declares
where examples come from and whether they need runtime feedback:

| Contract axis | Value | Runtime behavior |
|---|---|---|
| Acquisition | `generation` | Run adapter inference to create a rollout collection. |
| Acquisition | `dataset` | Fetch and optimize every batch from a finite dataloader. |
| Feedback | `runtime_reward` | Compute rewards and advantages before optimization. |
| Feedback | `none` | Pass acquired examples directly to the objective. |

This produces three currently useful compositions:

```text
online RL                 generation + runtime_reward
generation distillation   generation + none
SFT / offline DPO         dataset    + none
```

Acquisition is not a model concern, and feedback is not inferred from a batch shape. The trainer,
its algorithm-specific arguments, and the shared driver must declare the same immutable contract.
Pipeline I/O is a separate adapter contract that validates input modalities, ordered references,
output modality/rates, geometry ownership, and batch capability. The data schema therefore does
not need model-specific tensor names, while the objective does not need file-format branches.

The familiar online RL path still executes six stages:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        Flow-Factory Online Rollout Cycle                        │
│                                                                                 │
│  ┌─────────────┐    ┌──────────┐    ┌──────────────┐    ┌───────────────┐       │
│  │    Data     │    │ K-Repeat │    │  Trajectory  │    │    Reward     │       │
│  │Preprocessing│───►│ Sampling │───►│  Generation  │───►│ Computation   │       │
│  │  (offline)  │    │          │    │  (Adapter)   │    │               │       │
│  └─────────────┘    └─────▲────┘    └──────────────┘    └───────┬───────┘       │
│                           │                                     │               │
│                           │    ┌──────────────┐    ┌────────────▼─-─┐           │
│                           │    │   Policy     │    │   Advantage    │           │
│                           └────│ Optimization │◄───│  Computation   │           │
│                                └──────────────┘    └────────────────┘           │
└─────────────────────────────────────────────────────────────────────────────────┘
```

The high-level loop lives once in `BaseTrainer.start()`. Its acquisition driver selects either a
generated collection or one finite dataset traversal:

```python
# src/flow_factory/trainers/abc.py — BaseTrainer.start()
def start(self):
    while self.should_continue_training():
        driver.prepare_cycle(self, progress, seed=train.seed)
        driver.run_cycle(self, progress)
        if contract.acquisition is AcquisitionMode.GENERATION:
            self.adapter.ema_step(step=progress.rollout_iteration)
        # Runs once per completed rollout iteration or fully exhausted data epoch.
        self._after_acquisition_cycle()
        # Advance only after the selected acquisition completes successfully.
        progress = progress.advance_acquisition(contract.acquisition, completed=True)
```

Generation acquisition retains the familiar sequence:

```python
samples = self.sample()            # Stages 2 + 3
self.prepare_feedback(samples)     # Stages 4 + 5 (rewards + advantages)
self.optimize(samples)             # Stage 6 (DPO: pair formation + loss here)
```

Dataset acquisition never calls `sample()` or `adapter.inference()` for training. It calls
`optimize_batch(batch)` for every loader batch and advances the data epoch only after exhaustion.
This is the concrete form of treating the RL sampling stage as a no-op/fetch operation for an
offline algorithm, without teaching online trainers about offline batch formats.


## Stage 1: Data Preprocessing

**Goal**: Encode raw text prompts (and optional condition images / videos / audio files) into
model-ready tensor representations before repeated training work, eliminating redundant condition
encoding and enabling component offloading.

### Input / Output

| | Description |
|---|---|
| **Input** | Raw dataset containing prompts and optional condition media paths. Offline objectives use the `input` portion of strict V2 records. |
| **Output** | Cached HuggingFace Dataset with input tensors (`prompt_embeds`, `prompt_ids`, condition latents, etc.). Output supervision is excluded. |

### How It Works

Each model adapter exposes a `preprocess_func` that encodes raw inputs into tensors. The `GeneralDataset` class orchestrates this via HuggingFace's `.map()` with automatic caching:

```python
# src/flow_factory/data_utils/dataset.py — GeneralDataset._preprocess_batch()
def _preprocess_batch(self, batch, image_dir, video_dir, audio_dir):
    # 1. Prepare text prompts
    prompt = batch["prompt"]
    # 2. Load images from disk (if applicable)
    # 3. Load videos from disk (if applicable)
    # 4. Load audio files from disk (if applicable, via utils.audio.load_audio)
    # 5. Call model-specific preprocess function
    preprocess_res = self._preprocess_func(**filtered_args)
    # 6. Move tensors to CPU for caching
    # 7. Return batch dict with encoded tensors + metadata
```

The preprocess function is model-specific. For example, Flux.2 encodes prompts via its text encoder and images via its VAE:

```python
# src/flow_factory/models/flux/flux2.py — Flux2Adapter.preprocess_func()
def preprocess_func(self, prompt, images, ...):
    batch = self.encode_prompt(prompt=prompt, ...)       # → prompt_embeds, prompt_ids
    if has_images:
        batch.update(self.encode_image(images=images, ...))  # → image_latents, image_ids
    return batch
```

> **Audio is symmetric**: `audio_dir` is the third optional input handled by `_preprocess_batch`, parallel to `image_dir` / `video_dir`. Audio-aware adapters (e.g. the LTX-2 audio-video adapter) override `encode_audio` to consume the loaded `audios` batch; text/image/video-only adapters inherit the no-op `BaseAdapter.encode_audio` and ignore the column entirely.

### Key Points

- **Distributed preprocessing**: When running on multiple GPUs, each rank processes a shard of the dataset independently. The orchestrator (`loader._create_or_load_dataset`) routes each rank's `Dataset.map` output directly to its final per-rank Arrow file via `cache_file_name=`, so a shard is written to disk exactly once. After all ranks finish, the consolidator (local-main for `preprocess_parallelism="local"`, global rank 0 for `"global"`) writes only `state.json` and `dataset_info.json` referencing the existing per-rank files and atomically renames `.tmp` → final cache directory — no row data is re-serialized.
- **Cache layout**: The merged cache directory looks like `{cache_dir}/{fingerprint}/_parts/rank_{i:05d}_of_{N:05d}/cache-{fingerprint}_shard{i}of{N-1}.arrow`, plus the top-level `state.json` and `dataset_info.json`. While preprocessing is in flight, the same content lives under `{cache_dir}/{fingerprint}.tmp/`, with a `_build_meta.json` sentinel that records `num_shards` so a subsequent run with the same `num_shards` can resume from any per-rank Arrow files that were already written before a crash, while a different `num_shards` triggers a clean wipe.
- **No HF default-cache copy**: Because each `map()` call sets `cache_file_name`, HuggingFace does **not** also write a duplicate `cache-*.arrow` under `~/.cache/huggingface/datasets/...`.
- **Intelligent caching**: A hash fingerprint of `(dataset, split, max_dataset_size, preprocess_func source, preprocess_kwargs, extra_hash_strs)` (the last includes `model_type` and `model_name_or_path`) determines the cache path. Subsequent runs that match the fingerprint take the fast path without any `Dataset.map` invocation.
- **Component offloading**: Text and condition encoders can be offloaded after cache creation. An
  offline adapter's declared runtime condition-preparer and output-codec components are reloaded
  through the component lifecycle for on-the-fly condition realization and target encoding.
- **No target cache**: SFT targets and offline-DPO chosen/rejected candidates are decoded per
  dataset access and encoded per training microbatch. Target payloads, latent states, and metadata
  never enter the Arrow condition cache.

### Configuration

```yaml
data:
  datasets:
    - name: example
      dataset_dir: "path/to/dataset"
      train: {weight: 1}
  enable_preprocess: true          # Enable offline preprocessing
  force_reprocess: false           # Force re-encoding even if cache exists; essential if code is modified without changing config
  preprocessing_batch_size: 16     # Batch size for encoding
  cache_dir: "~/.cache/flow_factory/datasets"
  preprocess_parallelism: "local"  # "local" = per-node parallelism (no shared FS required); "global" = cross-node (shared FS required)
```

## Offline Dataset Training

SFT and offline DPO use strict V2 JSONL manifests described in the
[dataset guide](datasets.md#offline-v2-records). Their finite loader is constructed with PyTorch's
official `DistributedSampler`, even for `num_replicas=1`, and is not prepared by Accelerator:

```text
sampler.set_epoch(data_epoch)
for batch in dataloader:
    condition = cached prompt/input tensors
    output = freshly decoded target or chosen/rejected media
    trainer.optimize_batch(batch)  # prepares condition once inside the microbatch
data_epoch += 1  # only after clean exhaustion
```

Inside `optimize_batch`, the trainer calls
`adapter.prepare_condition_state(condition)` exactly once, binds every target
candidate through that prepared object, and reuses its model-forward view for all
policy/reference passes. The driver must not prepare it a second time.

One complete dataloader traversal is one offline epoch. Source weights must be `1`,
`data.sampler_type` remains `auto`, and `gradient_accumulation_steps` is an explicit integer. The
rank-local batch count must be divisible by gradient accumulation; the framework does not add
batches to close a partial accumulation window or flush one at epoch end. The official
`DistributedSampler` retains its standard `drop_last=False` behavior and may repeat global tail
indices to equalize rank lengths. Those indices are part of the finite loader traversal.

Progress uses three independent counters:

| Counter | Advances when | Used by |
|---|---|---|
| `rollout_iteration` | One generation acquisition completes. | Online rollout cadence and its compatibility `epoch`. |
| `data_epoch` | One finite offline dataloader traversal completes. | Offline epoch, sampler shuffle epoch, save/eval boundaries. |
| `optimizer_step` | One optimizer update completes. | Training metrics and optimizer-step state. |

Offline EMA updates on optimizer-step cadence because one data epoch may contain many optimizer
updates. Online algorithms retain their rollout-cycle EMA cadence. Similarly,
`num_train_timesteps` means independently sampled flow-matching terms averaged inside one offline
microbatch; it neither advances `optimizer_step` nor multiplies gradient accumulation.

An exact-state checkpoint (`log.save_model_only: false`) locks the realized training loader plus
evaluation cadence, sampling arguments, ordered eval loaders, per-dataset overrides, and eval
reward configuration. Online resume replays evaluation after its pre-rollout checkpoint, and
evaluation adapters or rewards may consume global RNG before the next training acquisition. MPS
cannot currently save exact state because Accelerate does not serialize MPS RNG; use
`log.save_model_only: true` on Apple Silicon.

The adapter prepares an input-owned condition state exactly once per offline batch. An identity
preparer returns cached fields unchanged. A conditioned model may instead realize geometry-bound
VAE latents, masks, or stochastic prefixes and split them into model-forward and output-codec
contexts. SFT reuses that state for its target; offline DPO reuses the same object for chosen and
rejected encoding and for both policy/reference arms. Candidate-specific output context is bound
only after this input state exists, so neither candidate can accidentally own or redraw an input
condition.

The target codec is adapter-owned but role-neutral at its numerical core. Condition and output
encoding reuse the same pixel preprocessing, VAE transform, normalization, and packing helpers.
Their semantic policies remain explicit: official condition paths commonly use posterior
`argmax`, while stochastic target training may use posterior `sample` with an optional generator.
Sharing a transform must never silently erase that role boundary. Target media remains on demand;
the prepared-condition boundary does not introduce a target-pixel or target-latent cache.

For offline DPO, chosen and rejected arms share one prepared input state, the primary timestep,
component-time mapping, and diffusion noise. Both policy arms run before one frozen-reference
scope covers both reference forwards. SFT has no reference branch. Multi-component adapters may
specialize only the offline flow-matching objective reduction (for example, a sum of per-modality
means) without changing the trajectory-wide reducer used by online policy likelihoods.


## Stage 2: K-Repeat Sampling

Stages 2–6 in this guide describe generation acquisition. Dataset acquisition replaces Stages 2–5
with the finite-loader path above and enters its objective through `optimize_batch()`.

**Goal**: Construct batches where each unique prompt appears exactly $K$ times (`group_size`), enabling group-relative advantage computation.

### Input / Output

| | Description |
|---|---|
| **Input** | Preprocessed dataset of $N$ samples |
| **Output** | Batches of encoded prompts, where each prompt is repeated $K$ times across the distributed cluster |

### How It Works

`DistributedKRepeatSampler` is the legacy arbitrary-placement layout:

```python
# src/flow_factory/data_utils/sampler.py — DistributedKRepeatSampler.__iter__()
def __iter__(self):
    while True:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        # 1. Randomly select M unique prompts
        indices = torch.randperm(len(self.dataset), generator=g)[:self.m].tolist()
        # 2. Repeat each prompt K times → M*K total samples
        repeated = [idx for idx in indices for _ in range(self.k)]
        # 3. Shuffle, so a group's K copies spread across ranks instead of
        #    landing contiguously on one
        order = torch.randperm(len(repeated), generator=g).tolist()
        shuffled = [repeated[i] for i in order]
        # 4. Each iteration hands every rank one contiguous slice of the shuffle
        for i in range(self.num_batches_per_epoch):
            start = i * self.sample_num_per_iteration + self.rank * self.batch_size
            yield shuffled[start : start + self.batch_size]
```

### Key Points

- **Deterministic seeding**: All ranks share the same `seed + epoch` generator, ensuring identical permutation and K-repeat ordering — no explicit cross-rank communication needed.
- **Automatic alignment**: Argument resolution adjusts `unique_sample_num` upward so the selected
  sampler closes its global batches and group windows.
- **Group identification**: The canonical comparison key is the exact int64 pair
  `(source_id, unique_id)`. The source namespace prevents equal prompt/condition hashes from
  independent datasets from sharing reward context.
- **Group-preserving layouts**: `group_contiguous` puts every K-group on one rank;
  `group_distributed` closes groups in every global microbatch; and `group_tiled` closes them in
  the smallest `K / gcd(world_size * per_device_batch_size, K)`-microbatch window. The selected
  algorithm declares which placements it can consume.
- **Multi-source overlap**: source mixing shuffles whole group-complete windows, not individual
  batches, so one comparison group never changes source midway through acquisition.

### Configuration

```yaml
train:
  per_device_batch_size: 2       # Batch size per GPU
  group_size: 4                  # K — repetitions per prompt
  unique_sample_num_per_epoch: 64  # M — unique prompts per epoch
```

> **Effective samples per epoch** = $M \times K$. For example, with `M=64, K=4`, each epoch generates 256 samples across the cluster.


## Stage 3: Trajectory Generation

**Goal**: Run the denoising model to generate images/videos from noise, collecting the **necessary** denoising trajectory (latents and log-probabilities at each timestep).

### Input / Output

| | Description |
|---|---|
| **Input** | Batched **raw input** (`prompt`, `images`) or **encoded tensors** (`prompt_embeds`, `image_latents`) from the dataloader. |
| **Output** | `List[BaseSample]` — each sample contains: generated image/video, denoising trajectory (`all_latents`), log-probabilities (`log_probs`), timestep schedule, and prompt info |

### How It Works

The trainer's `sample()` method switches the adapter to rollout mode and runs inference:

```python
# src/flow_factory/trainers/rl/grpo.py — GRPOTrainer.sample()
def sample(self) -> List[BaseSample]:
    trajectory_indices = compute_trajectory_indices(
        train_timestep_indices=self.adapter.get_train_step_indices(),
        num_inference_steps=self.training_args.num_inference_steps,
    )
    # generate_samples() (BaseTrainer) switches the adapter to rollout mode,
    # loops the dataloader, runs adapter.inference() under no_grad + autocast,
    # buffers rewards (reward_buffer), and returns the collected samples.
    return self.generate_samples(
        reward_buffer=self.reward_buffer,
        compute_log_prob=True,
        trajectory_indices=trajectory_indices,
    )
```

Inside `adapter.inference()`, the model runs a multi-step denoising loop (SDE or ODE), collecting latents and computing log-probabilities at each step. The result is packaged into `BaseSample` dataclass instances:

```python
# Example: src/flow_factory/models/flux/flux1.py — Inference result
BaseSample(
    timesteps=timesteps,                # (T+1,) schedule
    all_latents=stacked_latents,        # (num_stored, seq_len, C) — selectively stored
    log_probs=stacked_log_probs,        # (num_stored,) — per-step log π(a|s)
    latent_index_map=latent_index_map,  # (T+1,) maps step → storage index
    log_prob_index_map=log_prob_index_map,
    image=decoded_image,                # (C, H, W) tensor
    prompt=prompt_text,
    prompt_embeds=prompt_embeds,
    ...
)
```

### Algorithm-Specific Differences

| Algorithm | `compute_log_prob` | `trajectory_indices` | Notes |
|-----------|-------------------|---------------------|-------|
| **GRPO** | `True` | Only train timesteps | Needs log-prob for policy ratio; selective storage saves memory. |
| **DiffusionNFT** | `False` | `[-1]` (final only) | Only needs final clean latent $x_1$; log-prob not required |
| **AWM** | `False` | `[-1]` (final only) | Same as NFT; log-prob computed later during optimization |
| **DGPO** | `False` | `[-1]` (final only) | Same trajectory policy as NFT/AWM; optimization uses fresh `TimeSampler` timesteps |
| **CRD** | `False` | `[-1]` (final only) | Same trajectory policy as NFT/AWM; reward distillation against CFG-guided teacher reference |

### Key Points

- **Selective trajectory recording**: `trajectory_indices` controls which denoising steps are stored. For GRPO, only steps corresponding to `train_timesteps` are kept to reduce memory.
- **SDE dynamics for exploration**: GRPO injects noise during sampling via SDE formulation, enabling the log-probability computation required for policy gradients. NFT, AWM, DGPO, and CRD use decoupled sampling (typically ODE) with `compute_log_prob=False`.
- **Off-policy sampling**: NFT optionally uses EMA parameters for sampling (`off_policy: true`), while the current policy is optimized — stabilizing training.


## Stage 4: Reward Computation

**Goal**: Score each generated sample using one or more reward models.

### Input / Output

| | Description |
|---|---|
| **Input** | `List[BaseSample]` with generated images/videos and prompts |
| **Output** | `Dict[str, Tensor]` — reward name → per-sample scores (aligned with local samples) |

### How It Works

The `RewardProcessor` handles batched, distributed reward computation:

```python
# src/flow_factory/rewards/reward_processor.py — RewardProcessor.compute_rewards()
def compute_rewards(self, samples, store_to_samples=True, epoch=0, split='all'):
    results = {}
    # Pointwise rewards: local computation per rank
    if self._pointwise_models:
        results.update(self._compute_pointwise_rewards(samples, epoch))
    # Groupwise rewards: gather → compute → scatter
    if self._groupwise_models:
        results.update(self._compute_groupwise_rewards(samples, epoch))
    # Store rewards in each sample's extra_kwargs
    if store_to_samples:
        for i, sample in enumerate(samples):
            sample.extra_kwargs['rewards'] = {k: v[i] for k, v in results.items()}
    return results
```

### Key Points

- **Pointwise vs Groupwise**: Pointwise models (e.g., PickScore, CLIP) compute rewards independently per sample — no cross-rank communication needed. Groupwise models (e.g., ranking-based) require gathering all group members first.
- **Automatic deduplication**: If multiple reward entries share the same model config, they reuse a single model instance.
- **Flexible inputs**: Reward models declare `required_fields` (e.g., `("prompt", "image")`) and optionally receive raw tensors (`use_tensor_inputs=True`) or PIL images.
- **Remote reward servers**: For reward models with incompatible dependencies, Flow-Factory supports HTTP-based reward computation in isolated environments.
- **Incremental readiness**: Supported trainers can seal an async-only `RewardBuffer` after rollout and consume complete reward tiles while slower tiles are still being scored.

### Configuration

```yaml
rewards:
  - name: "aesthetic"
    reward_model: "PickScore"
    weight: 1.0
    batch_size: 16
  - name: "text_align"
    reward_model: "CLIP"
    weight: 0.5
    batch_size: 32
```

> See [Reward Guidance](rewards.md) for detailed reward model configuration.


## Stage 5: Advantage Computation

**Goal**: Convert raw rewards into normalized, group-relative advantages that serve as the optimization signal.

### Input / Output

| | Description |
|---|---|
| **Input** | Per-sample rewards (`Dict[str, Tensor]`) and samples with canonical `(source_id, unique_id)` identity |
| **Output** | Per-sample advantage scalar stored in `sample.extra_kwargs['advantage']` |

### How It Works

```python
# src/flow_factory/trainers/abc.py — BaseTrainer.compute_advantages()
def compute_advantages(self, samples, rewards, store_to_samples=True, aggregation_func=None):
    # Thin wrapper: resolve the aggregation strategy, then delegate to
    # AdvantageProcessor (advantage/advantage_processor.py). The processor is
    # communication-aware and auto-selects the gather-vs-local path; it performs
    # the gather -> weighted-aggregate -> group-by-(source_id, unique_id) -> normalize ->
    # scatter sequence summarized below.
    aggregation_func = aggregation_func or self.training_args.advantage_aggregation
    return self.advantage_processor.compute_advantages(
        samples=samples,
        rewards=rewards,
        store_to_samples=store_to_samples,
        aggregation_func=aggregation_func,
    )
```

### Aggregation Strategies

| Strategy | Formula | Use Case |
|----------|---------|----------|
| `sum` | $A = \text{normalize}(\sum_i w_i \cdot r_i)$ | Default GRPO: advantage of weighted reward sum |
| `gdpo` | $A = \sum_i w_i \cdot A_i$, with optional acquisition BN | Per-reward group normalization first, then combine |

### Key Points

- **Cross-rank synchronization**: Ordinary cross-rank feedback gathers rewards and exact int64
  identities separately. Streamed cross-rank work reuses one acquisition-level identity mapping
  and reduces packed group statistics, avoiding a per-work-unit reward/identity gather.
- **Group-relative normalization**: Within each group (same prompt), rewards are zero-centered and variance-normalized. This makes the advantage signal invariant to absolute reward scale.
- **Optional batch normalization** (GDPO): GDPO always normalizes each reward independently within its group. `global_std: true` additionally normalizes the combined advantages across the acquisition; `false` leaves them group-local.

### Configuration

```yaml
train:
  advantage_aggregation: 'sum'    # Options: 'sum', 'gdpo'
  global_std: false               # Keep normalization group-local; required for reward/optimization overlap
  adv_clip_range: [-5.0, 5.0]    # Clip advantages to prevent outliers
```


## Stage 6: Policy Optimization

**Goal**: Update the denoising model through the selected online or offline objective.

### Input / Output

| | Description |
|---|---|
| **Input** | Generated `List[BaseSample]` for generation acquisition, or one typed offline batch for dataset acquisition. |
| **Output** | Updated model parameters; logged loss metrics |

### How It Works (GRPO)

Stages 4–5 run in `prepare_feedback()` (reward buffer finalize, then `AdvantageProcessor`). Stage 6 is `optimize()` only:

```python
# Stages 4-5 - src/flow_factory/trainers/abc.py - BaseTrainer.prepare_feedback()
# (concrete; a distillation trainer such as diffusion-opd overrides it to a no-op)
def prepare_feedback(self, samples):
    rewards = self.reward_buffer.finalize(store_to_samples=True, split='all')
    self.compute_advantages(samples, rewards, store_to_samples=True)
    # ... log advantage metrics ...

# Stage 6 — GRPOTrainer.optimize()
def optimize(self, samples):
    for inner_epoch in range(num_inner_epochs):
        # Shuffle and re-batch
        shuffled = permute(samples)
        batches = [BaseSample.stack(chunk) for chunk in chunks(shuffled)]

        self.adapter.train()
        for batch in batches:
            # Iterate through train timesteps
            for timestep_index in scheduler.train_timesteps:
                with accelerator.accumulate(*trainable_components):
                    # 1. Get old log-prob from trajectory
                    old_log_prob = batch['log_probs'][log_prob_idx]
                    # 2. Forward pass → new log-prob
                    output = self.adapter.forward(latents=x_t, t=t, ...)
                    # 3. PPO-style clipped loss
                    ratio = exp(output.log_prob - old_log_prob)
                    unclipped = -adv * ratio
                    clipped   = -adv * clamp(ratio, 1-ε, 1+ε)
                    loss = mean(max(unclipped, clipped))
                    # 4. Optional KL regularization
                    if enable_kl_loss:
                        loss += kl_beta * KL(current || reference)
                    # 5. Backward + optimizer step
                    accelerator.backward(loss)
                    optimizer.step()
```

### Reward/Optimization Overlap

Every generation trainer with runtime group-relative feedback can pipeline Stages 4–6 after
rollout: GRPO, GRPO-Guard, DPPO, DiffusionNFT, AWM, CRD, DGPO, online DPO, and TDM-R1. The
rollout still finishes before the first optimizer update, so one acquisition never mixes generation
from different policy versions. Async reward work already submitted during rollout continues while
complete tiles enter optimization:

```text
rollout batch 0 ──► reward futures ───────────────────────────────┐
rollout batch 1 ──► reward futures ───────────────┐               │
...                                               ▼               ▼
rollout complete ──► globally ready tile 0 ──► optimize ──► next ready tile
```

Enable it with:

```yaml
train:
  reward_optimization_overlap: true
  reward_optimization_overlap_mode: ready  # ready | ordered
  reward_optimization_overlap_poll_interval: 0.05
  advantage_aggregation: sum
  global_std: false
  num_inner_epochs: 1
  shuffle_samples: false

data:
  sampler_type: group_contiguous

rewards:
  - name: remote_quality
    reward_model: flow_factory.rewards.my_reward_remote.RemotePointwiseRewardModel
    device: cpu
    async_reward: true
    num_workers: 8
    batch_size: 8
    server_url: http://reward-router:8000
```

`ordered` waits for the earliest outstanding tile and preserves rollout update order. `ready`
selects the lowest tile id among all globally ready tiles, bypassing a straggler but making update
order dependent on reward latency. Both modes require identical tile selection on every rank.

Generated acquisition cycles publish wall-clock metrics under a separate `timing/` namespace so
profiling data does not share the `train/` namespace with loss, reward, and tile-geometry metrics.
Durations are reduced as rank-wise maxima and therefore describe the distributed critical path:

| Metric | Meaning |
|---|---|
| `timing/rollout_seconds` | Complete rollout duration. |
| `timing/feedback_seconds` | Active reward-resolution and advantage-computation time on the training process. |
| `timing/optimization_seconds` | Total optimizer work in the acquisition cycle. |
| `timing/cycle_seconds` | End-to-end rollout, feedback, and optimization duration. |
| `timing/reward_overlap/stream_seconds` | Time from the completed rollout until the streamed reward/optimization cycle finishes. |
| `timing/reward_overlap/first_tile_seconds` | Time from the completed rollout until the first globally ready tile can optimize. |
| `timing/reward_overlap/wait_seconds` | Sleep time between polls with no globally selectable tile. |
| `timing/reward_overlap/coordination_seconds` | Local polling plus distributed readiness-coordination time. |
| `timing/reward_overlap/preparation_seconds` | Reward-independent optimizer preparation performed while async rewards run (for example CRD pass 1 or TDM-R1 fake TTUR). |
| `timing/reward_overlap/optimization_started_while_rewards_pending_seconds` | Duration of optimizer calls launched while later reward tiles were still pending. A reward may finish during the call, so this is scheduler exposure rather than exact hidden wall time. |
| `timing/reward_overlap/optimization_started_after_rewards_ready_seconds` | Duration of optimizer calls launched after all reward tiles were ready. |
| `timing/reward_overlap/optimization_started_while_rewards_pending_ratio` | Fraction of optimizer time whose calls began while later rewards were pending. |

Overlap-disabled runs emit the four top-level metrics too, which makes a same-shape baseline
directly comparable without changing logger grouping. Structural counters such as tile count remain
under `train/reward_overlap/` because they are training state rather than durations.

The tile planner describes how reward groups become optimizer examples. Rank-local objectives close
both K-groups and gradient accumulation; online DPO counts one preference pair per K-group. DGPO
and `group_distributed` TDM-R1 instead close groups in each global microbatch. TDM-R1 streams one
rollout batch per tile while its surrogate gradients close across the full acquisition. Exact
overlap is rejected when any training reward is synchronous, when either built-in aggregation
would need acquisition-wide standardization (`global_std: true`), when a reward client is not
CPU-side, or when the trainer has not declared the capability. With `global_std: false`, both
weighted-sum and GDPO advantages close within complete groups and can optimize ready tiles. SFT,
offline DPO, DiffusionOPD, DMD2, and reward-free TDM keep their existing execution path.

Pointwise reward request batching is independent of optimizer work-unit geometry. Each reward
model fills requests according to its own `batch_size`; a request may contain stable acquisition
rows from adjacent work units, and its future gates each unit containing one of those rows. This
keeps remote servers efficiently batched without changing group or gradient-accumulation
boundaries. The configured poll interval is the initial readiness delay; consecutive unsuccessful
global polls back off together (up to a bounded delay) and reset after a work unit advances.

For a pack-composition-dependent adapter such as Bagel at `per_device_batch_size > 1`, an
`AcquisitionManifest` records every original rollout microbatch. Ready scheduling may reorder
whole work units, but validation rejects any unit that would split or repack one of those recorded
batches.

> **`shuffle_samples` and on-policy ratio**: the optimize loop reorders `samples` each inner epoch (`train.shuffle_samples: true`, the default). For adapters whose batched `forward()` is *pack-composition-dependent* (e.g. Bagel NaViT packing), this makes a training micro-batch pack a different sample set than its rollout pack, so the on-policy `ratio != 1`. Set `train.shuffle_samples: false` for such adapters (with matched sampling/training `per_device_batch_size`) so each micro-batch reproduces its rollout pack. See the train-inference consistency topic doc.

### Algorithm-Specific Optimization

| Algorithm | Optimization Strategy |
|-----------|-----------------------|
| **GRPO** | Iterates over stored trajectory timesteps; computes ratio from old/new log-probs; PPO clipping |
| **GRPO-Guard** | Same as GRPO but with timestep-dependent loss reweighting to mitigate ratio bias |
| **DiffusionNFT** | Samples fresh timesteps; interpolates $x_t = (1-t)x_1 + t\epsilon$; contrastive objective with normalized rewards |
| **AWM** | Samples fresh timesteps; weights velocity matching loss by advantage; PPO clipping + EMA-KL regularization |
| **DGPO** | Samples fresh timesteps via `TimeSampler`; applies group-level preference objective with optional PPO clipping and EMA-reference KL |
| **CRD** | Samples fresh timesteps; reward distillation against CFG-guided teacher with adaptive KL; old/sampling model snapshots and centered advantages |
| **DPO** | Online preference loss on reward-ranked pairs formed at the start of `optimize` after advantages |
| **SFT** | On-the-fly target encoding followed by independently noised flow-matching loss in `optimize_batch` |
| **Offline DPO** | On-the-fly chosen/rejected encoding, shared timestep/noise, and policy-vs-reference DPO loss in `optimize_batch` |

### Optimizer Configuration

One optimizer root is built for the whole run, with one parameter group per trainable
variant. Declare them in the top-level `optimizers:` section, one entry per variant,
resolved by name; a single-policy algorithm has exactly one, which every shipped
example writes as `name: default`:

```yaml
optimizers:
  - name: generator
    optimizer: muon
    learning_rate: 2.0e-5
  - name: fake
    optimizer: adamw
    learning_rate: 1.0e-5
    update_frequency: 5
```

Omitting the section entirely still works: the flat `train.learning_rate`,
`adam_betas`, `adam_weight_decay` and `adam_epsilon` fields are translated once, in
`Arguments.__post_init__`, into the same single default entry. That shorthand is kept
for backward compatibility, but new configs should be explicit.

`max_grad_norm` belongs to the optimizer entry, not to `train:`. It is a property of
one optimization problem: roles already take different learning rates, and nothing
requires their clip budgets to match. A single-policy run is the N=1 case where the
two spellings coincide. When `optimizers:` is declared the entry owns the value and
it is mirrored onto `training_args.max_grad_norm`, so the shared gradient step reads
one resolved number and the two can never disagree.

Backends honor it differently, which is worth knowing when a clip appears to have no
effect. DDP applies it per call. FSDP2 applies it through DTensor-aware clipping.
DeepSpeed clips inside its own engine and ignores the value handed to
`accelerator.clip_grad_norm_`, which only reports the resulting norm, so the
threshold is published to the plugin through `ACCELERATE_GRADIENT_CLIPPING` before
the Accelerator is built.

### Component Loading and Frozen Precision

`model.component_load_dtypes` controls the dtype passed to the native component
loader. It overlays model-specific adapter defaults and accepts the same selector
forms as the frozen policy:

```yaml
model:
  component_load_dtypes:
    transformers: bf16  # Every declared transformer component.
    vae: fp32           # One concrete component.
```

`transformer` is a concrete component name; `transformers` is a group selector.
Set `component_load_dtypes: {default: null}` to disable an adapter's load-dtype
defaults and delegate every component to its native loader. Diffusers load-time
FP32 protections are applied before the post-load policy.

At runtime, `ModelLoadCoordinator` compiles logical component names into
exactly-once physical-root requests. This matters for aliases such as Bagel's
`transformer -> bagel.language_model`: the `bagel` root is target-owned, so the
prepared `language_model` route must stay in place while frozen siblings such as
the ViT and positional embeddings may be moved independently. Backend strategies
then apply these role contracts:

- TARGET-owned logical routes enter the prepared DDP/DeepSpeed/FSDP bundle;
  auxiliary movement of a containing physical root excludes those routes.
- AUXILIARY and REWARD resources remain full per-rank replicas. FSDP auxiliary
  roots receive a cached sampled-fingerprint check; reward loaders receive an
  isolated replicated-load scope.
- HOST roots such as tokenizers, processors, and schedulers are never sharded.
- FSDP2 CPU-efficient loading is enabled only for adapters that explicitly
  declare compatible selective component loading.

`model.frozen_parameters_dtype` accepts either one dtype for every frozen
component or a selector mapping:

```yaml
model:
  frozen_parameters_dtype:
    default: null       # Do not mutate dtype after loading.
    transformers: bf16  # Component group override.
    vae: fp32           # Concrete component override.
```

Concrete component names take priority over the `transformers` and
`text_encoders` groups, which take priority over `default`. A null value at any
level leaves that component untouched after loading. The scalar form
(`frozen_parameters_dtype: bf16`) remains shorthand for applying one dtype to
every frozen component.

FSDP2 still gives every trainable component a uniform FP32 original/master dtype;
its configured mixed-precision policy determines compute dtype. Component
selectors control non-trainable components under FSDP2 and frozen parameters
inside target components on backends that preserve original parameters.

### Variant Memory Placement

Component variants are optimizer-owned live parameters, not disposable model
copies. Once `accelerator.prepare()` has run, do **not** call `.to("cpu")`,
`off_load_components()`, or a custom offload/onload routine on an individual
trainable variant:

- DDP reducer hooks are bound to the prepared parameter devices.
- FSDP1 `FlatParameter` and FSDP2 `DTensor` placement belong to the sharded root.
- DeepSpeed ZeRO owns parameter partitions and optimizer state placement.

This also applies while a role is inactive in a multi-role phase. “Inactive” only
means that its optimizer groups and gradients are hidden for that phase; its
parameters still belong to the prepared root. Flow-Factory intentionally exposes no
`offload_variant()` / `onload_variant()` API because moving those parameters would
invalidate reducer hooks, optimizer identities, or shards.

The supported memory controls are:

| State | Supported placement |
|---|---|
| Text/image/audio encoders and VAE outside the prepared root | `on_load_components()` / `off_load_components()` lifecycle |
| Legacy named parameter snapshots | `add_named_parameters(..., device="cpu")`; copied into live parameters only inside their use context |
| Sampling EMA | `train.ema_device: cpu` or `cuda` |
| Variant-local EMA snapshots | Kept on the owning variant's device; no manual move API |
| Rollout samples | `train.offload_samples_to_cpu: true` |
| Prepared full/LoRA variants and optimizer state | Backend-managed DDP/FSDP/ZeRO placement only |

For whole-root CPU offload, configure the FSDP or DeepSpeed backend rather than
moving one role manually. CPU snapshots reduce persistent VRAM but add a synchronous
copy whenever installed; use them for infrequent teacher/reference passes, not as a
per-layer streaming mechanism.

`optimizer` selects both the
implementation and the argument schema (`hparams/optimizer_args/`), so AdamW and Muon
hyperparameters never share a class:

| Optimizer | Own fields |
|---|---|
| `adamw` | `betas`, `eps` |
| `muon` | `momentum`, `nesterov`, `ns_coefficients`, `ns_steps`, `adjust_lr_fn`, `fallback_betas`, `fallback_eps` |

`torch.optim.Muon` orthogonalizes matrices and rejects any parameter that is not 2D,
so a Muon variant is driven by two algorithms at once: Muon for its matrices and
AdamW for its biases, normalization scales and embeddings, which the `fallback_`
fields configure. `optimizer/loader.py` wraps that pair in a `CompositeOptimizer` so
the framework still prepares exactly one root. An all-AdamW run gets a plain
`torch.optim.AdamW`, unchanged. Muon requires a PyTorch build that exposes
`torch.optim.Muon` (included in standard releases from PyTorch 2.9; runtime capability detection
is authoritative). Muon combined with
DeepSpeed is refused at startup as unverified, and FSDP1 flattens matrices into
incompatible parameters; use DDP or FSDP2.

### Key Points

- **Generation inner epochs**: Generated samples can be reused for multiple optimization passes (`num_inner_epochs`), amortizing rollout cost. Offline `max_epochs` instead counts full loader traversals.
- **Gradient accumulation**: The `accelerator.accumulate()` context handles gradient accumulation across timesteps and micro-batches, with optimizer steps only at sync boundaries.
- **KL regularization**: Optional penalty keeping the policy close to a reference model (or EMA model for AWM), preventing reward hacking.
- **Per-timestep iteration**: GRPO iterates over each stored trajectory timestep, computing loss at each. NFT, AWM, DGPO, and CRD sample fresh timesteps independently of the sampling trajectory.

## Putting It All Together

A complete epoch with GRPO on a 8×GPU cluster:

```
Epoch N
├── DataLoader (DistributedKRepeatSampler)
│   └── Select 64 unique prompts × 4 repeats = 256 samples
│       → 32 samples per GPU (256 / 8)
│       → 16 batches per GPU (32 / batch_size=2)
│
├── Sampling (torch.no_grad)
│   └── For each batch: adapter.inference(compute_log_prob=True)
│       → 32 BaseSample per GPU, each with trajectory + log-probs
│
├── prepare_feedback(samples)
│   ├── Reward computation: RewardProcessor / buffer finalize → Dict[str, Tensor(32,)] per GPU
│   └── Advantage computation: gather → group by (source_id, unique_id) → normalize → scatter
│
└── optimize(samples) — Stage 6 only (num_inner_epochs × batches × timesteps)
    ├── Shuffle 32 samples → re-batch
    ├── For each batch, for each timestep:
    │   ├── Forward pass → new log-prob
    │   ├── PPO-clipped loss with advantage
    │   ├── + Optional KL penalty
    │   └── Backward + gradient accumulation
    └── Optimizer step at sync boundaries
```

*Online DPO*: form chosen/rejected pairs at the **start** of `optimize()` (after advantages exist), then run the preference loss; there is no pair formation in `prepare_feedback()`.

A complete offline epoch is a different acquisition shape:

```text
Data epoch N
├── DistributedSampler.set_epoch(N)
├── Exhaust every rank-local dataloader batch
│   ├── Reuse cached prompt/input condition
│   ├── Decode target or chosen/rejected media from source
│   ├── Encode output state on the fly
│   ├── Compute SFT or offline-DPO loss
│   └── Optimizer step at explicit GAS sync boundaries
└── Advance data_epoch only after clean exhaustion
```

There is no rollout, training reward, advantage computation, or online pair formation in this
path. Save/evaluation boundaries use the completed data epoch; training metrics use the independent
optimizer-step counter.

## Structured multimodal trajectories

`StructuredTrajectory` is authoritative for multimodal adapters; legacy
`trajectory is None` tensors remain only for backward-compatible single-component
models. New multimodal adapters emit structured trajectories only.

For T transitions, component state maps have length `T + 1`;
log-probability and callback maps have length T. An index of `-1` means that coordinate was not collected,
and an omitted log-probability or callback collection is represented by `None`.

The component order is adapter-owned, for example `("video", "audio")` in MiniMax
H3. Conditioning is packed and replayed by the adapter; trainers such as GRPO,
GRPO-Guard, DPPO, DiffusionNFT, AWM, DPO, DGPO, CRD, and DiffusionOPD consume the
same state interface. H3 accepts neutral guidance `1.0`; framework-interface compatibility
does not itself establish real-weight numerical parity.
