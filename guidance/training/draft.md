# DRaFT-K

DRaFT-K maximizes a differentiable reward on the image produced by the sampling
chain. This implementation adapts [Directly Fine-Tuning Diffusion Models on
Differentiable Rewards](https://arxiv.org/html/2309.17400#S4) to Euler flow
matching on SD3.5.

## Recipe

Start with the [SD3.5 PickScore example](../../examples/draft/lora/sd3_5/pickscore.yaml).
Prepare a prompt dataset with a `prompt` column, set its path in the recipe, and
run `ff-train examples/draft/lora/sd3_5/pickscore.yaml`.

```yaml
train:
  trainer_type: draft
  num_inference_steps: 40
  draft_backprop_steps: 2
scheduler:
  dynamics_type: ODE
```

The trainer acquires the initial latent, conditioning, and native schedule for
each sample. During optimization, the current policy runs the first `T - K`
steps under no-grad and the last `K` contiguous steps with gradients. The frozen
VAE and synchronous [differentiable reward](rewards.md) preserve gradients to the
policy. The prompt source selects its usual reward weights.

Set `draft_backprop_steps: 1` for the last transition alone or set it to
`num_inference_steps` for the full chain. Activation memory grows with the length
of the differentiated suffix. The scheduler replay retention setting has no
effect on the unroll, which starts from the initial latent and uses each live
Euler state.

## Support and qualification

The initial implementation admits native SD3.5 image generation, Euler ODE,
and a data-only layout with replicated auxiliaries. Local CPU tests check the
native endpoint, the first update, and exact next-update continuation. An
independent FP32 Euler reference checks gradient truncation for short, partial,
and full suffixes. CUDA DDP and FSDP2 policy qualification is tracked in
[P21](../../roadmap/milestones/v0.5x-v0.6.md#p21).
