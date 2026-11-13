# ReFL reward-gradient training

ReFL optimizes a generated clean-state prediction through a frozen differentiable
reward. The framework first acquires an ODE trajectory without gradients, selects
one retained transition per optimization microbatch, and replays its policy
prediction with gradients enabled. The model adapter projects that prediction to
clean data and decodes it through its native image, video or audiovisual decoder.

Decoder and reward parameters stay frozen in eval mode. Their input derivatives
connect the reward objective to the policy. Gradient accumulation, optimizer
updates, source routing and checkpointing use the shared trainer owners.

## Model coverage

All 19 registered adapters expose differentiable decoding. Existing model-specific
input conditions, supported batch sizes and parallel layouts continue to apply.

| Model adapters | Decoded media | Model-owned operations |
|---|---|---|
| SD3.5, Z-Image | Image | VAE scaling, shift and image postprocessing |
| FLUX.1, FLUX.1-Kontext | Image | Target-token unpacking, VAE scaling and reference-conditioned geometry |
| FLUX.2, FLUX.2-Klein | Image | Position-ID unpacking, latent batch normalization and unpatchification |
| FLUX.3 | Image | Target-grid unpacking and retained reference crop-back |
| Qwen-Image, Qwen-Image-Edit-Plus | Image | Target-only unpacking and channel-wise VAE normalization |
| Qwen-Image 2.1 | RGBA image | Packed 64-channel latent normalization and native RGBA decoding |
| Bagel | Image | Packed-row reconstruction and per-sample native VAE decoding |
| SenseNova-U1 | Image | Normalized pixel-state conversion |
| Wan T2V, Wan I2V | Video | Video VAE normalization and native frame ordering |
| LTX2 T2AV, LTX2 I2AV | Video and audio | Separate component clocks, video VAE, audio VAE and vocoder |
| MiniMax H3 T2VA, FL2VA, Ref2VA | Video and stereo audio | Data-directed velocity, stored component sigmas and target-only native decode blocks |

Image tensors use BCHW, video tensors use BFCHW, and audio waveforms use BCS.
Native decoder precision follows the generation autocast policy. Model-specific
packing, conditions and output crop geometry stay in the adapter. Inactive
conditioning tokens, such as LTX2 I2AV's first frame, remain fixed during clean-state
projection and receive no reward gradient through that prediction.

## Objective and recipe

The [SD3.5/PickScore example](../../examples/refl/lora/sd3_5/pickscore.yaml) adapts
ReFL's late-step clean-prediction objective to flow matching and PickScore. Prepare a
JSONL dataset with a `prompt` column, then run
`ff-train examples/refl/lora/sd3_5/pickscore.yaml`.

For another model, use its ordinary model/data/conditioning configuration and select
`train.trainer_type: refl`. Configure ODE generation and the retained replay positions
through the shared scheduler settings.

```yaml
train:
  trainer_type: refl
  num_inference_steps: 40
  refl_reward_scale: 1.0
  refl_reward_target: null
scheduler:
  dynamics_type: ODE
  replay_steps: [30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
rewards:
  - name: alignment
    reward_model: pickscore
    async_reward: false
```

`refl_reward_scale` multiplies the summed source-weighted reward losses. A null
`refl_reward_target` maximizes each reward directly. A finite target uses
`relu(target - reward)` in the scorer's units. Each reward loss is averaged over
the policy microbatch with zero weight for inapplicable sources, then terms are summed.

The original ImageReward implementation samples positions 30–39 of a 40-step rollout
and uses a hinge target of 2 in ImageReward units. The supplied PickScore recipe uses
the same candidate positions and a target of 1 in the framework's normalized PickScore
units. [Author implementation](https://github.com/zai-org/ImageReward/blob/main/ImageReward/ReFL.py)

## Reward modalities

PickScore accepts images and videos. Its differentiable video path averages the
normalized score over every frame. RGB channels of RGBA images follow CLIP's ordinary
RGB conversion. Resize, crop and normalization stay in Torch, retaining pixel gradients.

Each scorer declares complete differentiable input sets. The dispatcher selects a
compatible set from the adapter's outputs and passes only those fields to that scorer.
For example, an audiovisual model can use a video scorer and a separate audio scorer,
with independent source weights. A joint scorer declares both `video` and `audio` in
one input set. Add an audio-capable or joint scorer for an explicit audio objective.
The built-in differentiated scorer in this milestone is PickScore.

Use `rewards` for differentiable training objectives and `eval_rewards` for OCR, API
or other observation-only scores. Unsupported model/reward pairs fail before loading
weights. See [reward capabilities](rewards.md#input-gradient-capabilities) for custom scorers.

DDP groups the guidance branches of one prediction under one prepared forward.
Multimodal differentiated feedback enables unused-parameter detection, allowing
unscored output heads to keep `grad=None` and skip optimizer updates. LTX2 and H3
have two-card video-only reward tests covering this path. A two-card Wan I2V test checks
shared timestep selection and updates through both sharded denoisers.

## Configuration and continuation

- `scheduler.replay_steps` selects retained rollout transitions. ReFL draws one candidate
  per global optimization microbatch. Epoch, inner epoch and microbatch position seed
  the draw, so all shards choose the same conditional denoiser and drained resumes agree.
- ODE generation supports the Euler flow solver and MiniMax H3's native flow solver.
- `num_inner_epochs`, update count and gradient accumulation remain configurable. The
  accumulation count uses one backward term per microbatch, independently of candidate count.
- Decoder components must be frozen policy auxiliaries. Their names are declared by each adapter.
- Scoring runs synchronously on the optimization thread with `async_reward: false`.
- Policy training uses existing DDP/FSDP2 and declared TP/SP/CP layouts. Reward and decoder
  components are replicated auxiliaries. P20 introduces more flexible component preparation.

Continuation uses drained acquisition checkpoints. Reward revisions, preprocessing,
input capabilities, objective parameters, replay positions and precision participate
in runtime identity. The SD3.5 DDP test checks exact next-update continuation.

## Validation

The selected model campaign runs two complete acquisition/update cycles for every
registered adapter with small native model structures. It checks native-decoder
value parity, nonzero media and latent gradients, policy parameter updates, and frozen
auxiliary weights. LTX2 I2AV additionally exercises FSDP2 + TP2/SP + CP2 + DP2 on eight
GPUs, including both video/audio gradients and fixed conditioning tokens.

Independent CPU tests check objective derivatives, stored per-component sigmas and
H3's velocity direction. The separate FLUX.1 FP32 control compares gradients and
updates across parallel layouts. Pretrained PickScore evidence covers image and video
input derivatives and complete SD3.5/Wan reward-to-policy training. These native small-model
checks establish functional support. Pretrained policy quality is measured by a training experiment.

Image-conditioned preprocessing uses cache format version 3 to retain Hugging Face
Image metadata on a single rank. Earlier preprocessing caches are rebuilt automatically.
