# Workflow Guidance

## Table of Contents

- [Overview](#overview)
- [Stage 1: Data Preprocessing](#stage-1-data-preprocessing)
- [Stage 2: K-Repeat Sampling](#stage-2-k-repeat-sampling)
- [Stage 3: Trajectory Generation](#stage-3-trajectory-generation)
- [Stage 4: Reward Computation](#stage-4-reward-computation)
- [Stage 5: Advantage Computation](#stage-5-advantage-computation)
- [Stage 6: Policy Optimization](#stage-6-policy-optimization)
- [Putting It All Together](#putting-it-all-together)

## Overview

Flow-Factory follows an **online RL** training paradigm for diffusion/flow-matching models. Each epoch executes a six-stage pipeline:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          Flow-Factory Training Epoch                            │
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

The high-level training loop (shared by all algorithms) is defined in each trainer's `start()` method:

```python
# src/flow_factory/trainers/grpo.py — GRPOTrainer.start()
def start(self):
    while True:
        # Checkpoint & Evaluation (omitted for brevity)
        samples = self.sample()       # Stage 2 + 3
        self.optimize(samples)        # Stage 4 + 5 + 6
        self.epoch += 1
```

> **Note**: Stage 1 (preprocessing) runs *once* before training begins and is cached to disk. Stages 2–6 repeat every epoch.


## Stage 1: Data Preprocessing

**Goal**: Encode raw text prompts (and optional images/videos) into model-ready tensor representations *before* training begins, eliminating redundant computation during the RL loop and enabling components offloading such as **text-encoder** and **image-encoder**.

### Input / Output

| | Description |
|---|---|
| **Input** | Raw dataset: `train.jsonl` or `train.txt` containing prompts, optional image/video paths |
| **Output** | Cached HuggingFace Dataset on disk with pre-encoded tensors (`prompt_embeds`, `prompt_ids`, `pooled_prompt_embeds`, `image_latents`, etc.) |

### How It Works

Each model adapter exposes a `preprocess_func` that encodes raw inputs into tensors. The `GeneralDataset` class orchestrates this via HuggingFace's `.map()` with automatic caching:

```python
# src/flow_factory/data_utils/dataset.py — GeneralDataset._preprocess_batch()
def _preprocess_batch(self, batch, image_dir, video_dir):
    # 1. Prepare text prompts
    prompt = batch["prompt"]
    # 2. Load images from disk (if applicable)
    # 3. Load videos from disk (if applicable)
    # 4. Call model-specific preprocess function
    preprocess_res = self._preprocess_func(**filtered_args)
    # 5. Move tensors to CPU for caching
    # 6. Return batch dict with encoded tensors + metadata
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

### Key Points

- **Distributed preprocessing**: When running on multiple GPUs, each rank processes a shard of the dataset independently, then merges results. Controlled by `enable_preprocess` in data config.
- **Intelligent caching**: A hash fingerprint of `(dataset, model_type, model_path, preprocess_kwargs)` determines the cache path. Subsequent runs skip preprocessing if the cache is valid.
- **Component offloading**: Text encoders and VAEs are loaded for preprocessing, then offloaded before the training loop to free VRAM for the denoising model.

### Configuration

```yaml
data:
  dataset: "path/to/dataset"
  enable_preprocess: true          # Enable offline preprocessing
  force_reprocess: false           # Force re-encoding even if cache exists; essential if code is modified without changing config
  preprocessing_batch_size: 16     # Batch size for encoding
  cache_dir: "~/.cache/flow_factory/datasets"
```


## Stage 2: K-Repeat Sampling

**Goal**: Construct batches where each unique prompt appears exactly $K$ times (`group_size`), enabling group-relative advantage computation.

### Input / Output

| | Description |
|---|---|
| **Input** | Preprocessed dataset of $N$ samples |
| **Output** | Batches of encoded prompts, where each prompt is repeated $K$ times across the distributed cluster |

### How It Works

The `DistributedKRepeatSampler` handles this:

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
        # 3. Distribute evenly across all GPUs
        per_rank = chunk(repeated, self.num_replicas)[self.rank]
        # 4. Yield batches of size `per_device_batch_size`
        yield from chunk(per_rank, self.batch_size)
```

### Key Points

- **Deterministic seeding**: All ranks share the same `seed + epoch` generator, ensuring identical permutation and K-repeat ordering — no explicit cross-rank communication needed.
- **Automatic alignment**: The sampler adjusts `unique_sample_num` upward to ensure `M * K` is evenly divisible by `batch_size * num_replicas`.
- **Group identification**: Each sample carries a `unique_id` (hash of prompt + conditions). During advantage computation, samples are grouped by this ID across all ranks.

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
| **Input** | Batched **raw input** (`prompt`, `images`) or **encoded tensors** (`prompt_embedds`, `image_latents`) from the dataloader. |
| **Output** | `List[BaseSample]` — each sample contains: generated image/video, denoising trajectory (`all_latents`), log-probabilities (`log_probs`), timestep schedule, and prompt info |

### How It Works

The trainer's `sample()` method switches the adapter to rollout mode and runs inference:

```python
# src/flow_factory/trainers/grpo.py — GRPOTrainer.sample()
def sample(self):
    self.adapter.rollout()  # Disable grad, set eval mode
    samples = []
    trajectory_indices = compute_trajectory_indices(
        train_timestep_indices=self.adapter.scheduler.train_timesteps,
        num_inference_steps=self.training_args.num_inference_steps,
    )
    with torch.no_grad(), self.autocast():
        for batch in dataloader:
            sample_batch = self.adapter.inference(
                compute_log_prob=True,
                trajectory_indices=trajectory_indices,
                **batch,
            )
            samples.extend(sample_batch)
    return samples
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

### Key Points

- **Selective trajectory recording**: `trajectory_indices` controls which denoising steps are stored. For GRPO, only steps corresponding to `train_timesteps` are kept to reduce memory.
- **SDE dynamics for exploration**: GRPO injects noise during sampling via SDE formulation, enabling the log-probability computation required for policy gradients. NFT and AWM use standard ODE solvers.
- **Off-policy sampling**: NFT optionally use EMA parameters for sampling (`off_policy: true`), while the current policy is optimized — stabilizing training.


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
| **Input** | Per-sample rewards (`Dict[str, Tensor]`) and sample list with `unique_id` |
| **Output** | Per-sample advantage scalar stored in `sample.extra_kwargs['advantage']` |

### How It Works

```python
# src/flow_factory/trainers/grpo.py — GRPOTrainer.compute_advantage_weighted_sum()
def compute_advantage_weighted_sum(self, samples, rewards, store_to_samples=True):
    # 1. Gather rewards from all ranks
    gathered_rewards = {k: accelerator.gather(v).cpu().numpy() for k, v in rewards.items()}
    # 2. Weighted sum of multiple rewards
    aggregated = sum(arr * weight for arr, weight in ...)
    # 3. Group by unique_id
    unique_ids = [s.unique_id for s in samples]
    gathered_ids = accelerator.gather(unique_ids)
    _, group_indices = np.unique(gathered_ids, return_inverse=True)
    # 4. Normalize within each group: (r - mean) / std
    for group_id in np.unique(group_indices):
        mask = (group_indices == group_id)
        advantages[mask] = (aggregated[mask] - mean) / std
    # 5. Scatter back to local rank
    advantages = advantages.reshape(num_processes, -1)[process_index]
```

### Aggregation Strategies

| Strategy | Formula | Use Case |
|----------|---------|----------|
| `sum` | $A = \text{normalize}(\sum_i w_i \cdot r_i)$ | Default GRPO: advantage of weighted reward sum |
| `gdpo` | $A = \text{BN}(\sum_i w_i \cdot A_i)$ | Per-reward normalization first, then combine |

### Key Points

- **Cross-rank synchronization**: Advantages are computed globally — rewards from all ranks are gathered, normalized, then scattered back. This ensures consistent group-level statistics.
- **Group-relative normalization**: Within each group (same prompt), rewards are zero-centered and variance-normalized. This makes the advantage signal invariant to absolute reward scale.
- **Batch normalization** (GDPO): For multi-reward scenarios, GDPO normalizes each reward independently before combining, preventing one reward from dominating.

### Configuration

```yaml
train:
  advantage_aggregation: 'sum'    # Options: 'sum', 'gdpo'
  global_std: false               # Use global std instead of per-group std
  adv_clip_range: [-5.0, 5.0]    # Clip advantages to prevent outliers
```


## Stage 6: Policy Optimization

**Goal**: Update the denoising model's parameters using the computed advantages and PPO-style clipped policy gradient.

### Input / Output

| | Description |
|---|---|
| **Input** | `List[BaseSample]` with advantages, trajectories, and log-probs stored |
| **Output** | Updated model parameters; logged loss metrics |

### How It Works (GRPO)

```python
# src/flow_factory/trainers/grpo.py — GRPOTrainer.optimize()
def optimize(self, samples):
    rewards = self.reward_processor.compute_rewards(samples)   # Stage 4
    advantages = self.compute_advantages(samples, rewards)      # Stage 5

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

### Algorithm-Specific Optimization

| Algorithm | Optimization Strategy |
|-----------|-----------------------|
| **GRPO** | Iterates over stored trajectory timesteps; computes ratio from old/new log-probs; PPO clipping |
| **GRPO-Guard** | Same as GRPO but with timestep-dependent loss reweighting to mitigate ratio bias |
| **DiffusionNFT** | Samples fresh timesteps; interpolates $x_t = (1-t)x_1 + t\epsilon$; contrastive objective with normalized rewards |
| **AWM** | Samples fresh timesteps; weights velocity matching loss by advantage; PPO clipping + EMA-KL regularization |

### Key Points

- **Inner epochs**: Samples can be reused for multiple optimization passes (`num_inner_epochs`), amortizing the cost of sampling.
- **Gradient accumulation**: The `accelerator.accumulate()` context handles gradient accumulation across timesteps and micro-batches, with optimizer steps only at sync boundaries.
- **KL regularization**: Optional penalty keeping the policy close to a reference model (or EMA model for AWM), preventing reward hacking.
- **Per-timestep iteration**: GRPO iterates over each stored trajectory timestep, computing loss at each. NFT and AWM sample fresh timesteps independently of the sampling trajectory.


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
├── Reward Computation
│   └── RewardProcessor scores all 32 samples per GPU
│       → Dict[str, Tensor(32,)]
│
├── Advantage Computation
│   ├── Gather rewards across 8 GPUs → 256 total scores
│   ├── Group by unique_id → 64 groups of 4
│   ├── Normalize within each group
│   └── Scatter back → 32 advantages per GPU
│
└── Optimization (num_inner_epochs × batches × timesteps)
    ├── Shuffle 32 samples → re-batch
    ├── For each batch, for each timestep:
    │   ├── Forward pass → new log-prob
    │   ├── PPO-clipped loss with advantage
    │   ├── + Optional KL penalty
    │   └── Backward + gradient accumulation
    └── Optimizer step at sync boundaries
```