# DRTune

DRTune uses the decoded sampling endpoint to supervise selected denoiser calls
across the Euler ODE chain. This implementation adapts the sparse stop-gradient
estimator to native SD3.5 text-to-image training.
[Deep Reward Supervisions for Tuning Text-to-Image Diffusion Models](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/10922.pdf)

## Recipe

Start with the [SD3.5 PickScore example](../../examples/drtune/lora/sd3_5/pickscore.yaml).
Set the prompt dataset path and run `ff-train examples/drtune/lora/sd3_5/pickscore.yaml`.

```yaml
train:
  trainer_type: drtune
  num_inference_steps: 40
  drtune_backprop_steps: 2
scheduler:
  dynamics_type: ODE
```

`drtune_backprop_steps` selects K equally spaced predictions from the T Euler
steps, with a seeded offset shared by every rank for each optimizer microbatch.
The trainer acquires the same initial noise, conditioning and native schedule
as DRaFT-K. It runs all T predictions and detaches each denoiser input. Selected
predictions retain parameter gradients, and all Euler updates carry the live
solver state forward. The synchronous differentiable reward reaches early
selected predictions through that state even when later predictions use no-grad.

Choose K between 1 and T, inclusive. K=T retains parameter gradients in every
prediction while keeping the stop-gradient rule. The chosen positions are
independent of `scheduler.replay_steps`, which controls saved transitions.
Activation checkpointing can further reduce the memory held by selected calls.

## Support and qualification

The initial implementation uses SD3.5, Euler ODE, local or data-only DDP policy
execution, and replicated auxiliaries. CPU evidence covers independent Euler
derivatives, native endpoint parity, exact resumed updates and two-rank Gloo
DDP. The two-GPU CUDA DDP witness and early-stop projection are tracked in the
[DRTune roadmap](../../roadmap/milestones/v0.5x-v0.6.md#p22).
