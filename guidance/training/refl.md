# ReFL with differentiable rewards

ReFL trains a selected denoising prediction using gradients through a frozen decoder
and reward network. Set `train.trainer_type: refl` and choose a reward with an image
input-gradient declaration. SD3.5 and FLUX.1 provide the initial qualified image layouts.

## Gradient program

The framework generates a detached rollout and retains the states selected by
`scheduler.replay_steps`. Each optimization microbatch selects one of these transitions
using its sampling identity, seed, acquisition iteration and inner epoch. The policy
predicts velocity at that state. The adapter maps velocity to a clean latent using its
own time and velocity-direction convention, then decodes floating RGB pixels. Frozen
reward weights transmit image gradients back through the decoder to the policy.

The decoder and reward remain in eval mode with frozen weights. Gradient accumulation,
optimizer updates, checkpointing and source routing use the shared trainer owners.
Each microbatch builds and releases its own graph. The initial implementation performs
a complete no-grad rollout before replaying the selected prediction.

## Objective and recipe

The [SD3.5/PickScore example](../../examples/refl/lora/sd3_5/pickscore.yaml) adapts
ReFL's late-step clean-prediction objective to flow matching and PickScore. Prepare a
JSONL dataset with a `prompt` column in the configured dataset directory, then run
`ff-train examples/refl/lora/sd3_5/pickscore.yaml`.

`refl_reward_scale` multiplies the summed source-weighted reward losses. With
`refl_reward_target: null`, the loss maximizes each reward directly. A finite target
uses `relu(target - reward)`. The target is expressed in that scorer's score units;
PickScore retains the framework's /26 scaling. Each reward loss is averaged over the
policy microbatch with zero weight for inapplicable sources, then reward terms are summed.

The original ImageReward implementation samples positions 30–39 of a 40-step rollout
and uses a hinge target of 2 in ImageReward units. The supplied PickScore recipe uses
the same candidate positions and an explicitly chosen target of 1 in normalized
PickScore units. It is a flow-matching/PickScore adaptation. [Author implementation](https://github.com/zai-org/ImageReward/blob/main/ImageReward/ReFL.py)

## Configuration ownership

- `scheduler.replay_steps` determines which actual rollout transitions are retained.
- ReFL samples one retained position for each optimization microbatch. Consequently the
  shared accumulation count is one backward term per microbatch, even with many candidates.
- ODE dynamics and the Euler flow scheduler define the initial supported solver path.
- Ordinary `num_inner_epochs`, update count and accumulation remain configurable.
- `rewards` defines differentiable training objectives. Unsupported rewards fail before
  loading. Use `eval_rewards` for OCR, API and other observation-only rewards.
- Scoring is synchronous on the optimization thread. Set `async_reward: false`.

## Parallelism and continuation

The policy uses its existing DDP/FSDP2 preparation and declared TP/SP/CP layout. Reward
and decoder weights are replicated auxiliary components. CP uses the shared full-output
loss normalization. Component sharding follows in roadmap P20.

Continuation uses drained acquisition checkpoints. Reward identity, objective parameters,
retained positions and precision participate in the existing runtime execution identity.
The selected position follows sample identity, so parallel peers and resumed runs use the
same selection without an arrival-dependent random draw.

Selected tests cover an independent CPU derivative program, native SD3.5 DDP continuation,
and FLUX.1 FSDP2 with TP2/SP/CP2/DP2. Independent BF16 training runs exercise the
complete lifecycle. A separate FP32 control fixes replay operands and parameter
states to compare elementwise gradients and updates against the unpartitioned reference. Pretrained reward
input derivatives have a separate PickScore CUDA test. These tests establish the stated
functional paths; quality reproduction belongs to an explicitly recorded training experiment.
