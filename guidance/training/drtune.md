# DRTune

DRTune uses a differentiable reward on the decoded Euler endpoint or an early
clean estimate. It adapts the sparse stop-gradient estimator to native SD3.5
text-to-image training.
[Deep Reward Supervisions for Tuning Text-to-Image Diffusion Models](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/10922.pdf)

## Recipe

Start with the [SD3.5 PickScore example](../../examples/drtune/lora/sd3_5/pickscore.yaml).
Set the prompt dataset path and run `ff-train examples/drtune/lora/sd3_5/pickscore.yaml`.

```yaml
train:
  trainer_type: drtune
  num_inference_steps: 40
  drtune_backprop_steps: 2
  drtune_stop_step_range: [32, 36]
scheduler:
  dynamics_type: ODE
```

`num_inference_steps` defines the full native schedule T. When
`drtune_stop_step_range` is omitted or null, all T predictions run and the reward
scores the native final sample. With `[32, 36]`, DRTune draws an integer N
uniformly from 32 through 36 each optimizer microbatch. These are prediction
counts. For N=32, the last visited prediction has zero-based index 31 and
projects the live latent to a clean estimate using its current sigma and
velocity, through the same adapter method as ReFL.

DRTune then selects K = `drtune_backprop_steps` predictions within the N visited
positions. It uses stride `floor(N / K)` and a uniform offset from zero through
`N - 1 - (K - 1) * stride`. The stop-first draw guarantees that every selected
prediction is visited. It also covers divisible schedules and K=N, which
require special care when reading the interval printed in Algorithm 1.
Choose `1 <= K <= minimum <= maximum <= T` for a stop interval. A fixed interval
such as `[32, 32]` always visits 32 predictions.

All ranks share the two draws for each microbatch, and a training-state resume
restores the same stop and mask sequence. The denoiser sees detached inputs.
Selected predictions keep parameter gradients, while the Euler state and
ReFL-style projection keep their direct state gradient even when the last
prediction is unselected. `scheduler.replay_steps` controls trajectory storage.
The training logger reports `train/drtune/visited_steps` and
`train/drtune/backprop_steps` averaged across the accumulated microbatches.

For a two-GPU data-only FSDP2 run, apply these settings to the recipe. Keep
`enable_gradient_checkpointing: true` in the training section to reduce the
activations retained by selected calls.

```yaml
config_file: config/accelerate_configs/fsdp2.yaml
num_processes: 2
parallelism:
  dp_shard_size: 2
```

## Support and qualification

The supported composition uses SD3.5, Euler ODE, local, data-only DDP or FSDP2
policy execution, and replicated auxiliaries. See the
[DRTune roadmap](../../roadmap/milestones/v0.5x-v0.6.md#p22) for the selected
gradient, continuation and GPU qualification cases.
