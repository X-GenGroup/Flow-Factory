# Sampling, replay and re-noising

A rollout produces states `x_0 ... x_T` through `T` transitions. Exploration,
retention and training-time re-noising are separate choices: using the same
timestep does not imply using the same latent or the same objective.

| Choice | Configuration | What it controls |
|---|---|---|
| Rollout exploration | `scheduler.dynamics_type`, `noise_level`, `sde_steps`, `num_sde_steps` | Where the sampler injects SDE noise |
| Actual transition replay | `scheduler.replay_steps` | Which recorded transitions ART/GRPO/Guard/DPPO optimize |
| Endpoint re-noising | `train.time_sampling_strategy`, `timestep_range`, `time_shift`, `num_train_timesteps` | How NFT/AWM/DGPO/CRD draw training coordinates and construct new noisy states |

## Select actual rollout transitions

`scheduler.replay_steps` uses **zero-based transition indices**, not timestep values.
For example, index `2` selects `x_2 -> x_3`, even when the shifted scheduler timestep
at that position is `750`.

| Value | Replay selection | Stored states |
|---|---|---|
| `auto` (default) | The realized SDE steps, or every step for ODE | Both endpoints of each selected transition |
| `all` | All `T` transitions | All `T+1` states |
| `[0, 3, 7]` | Those transitions, in the given replay order | The union of their endpoints |
| `null` or `[]` | No transitions | None, unless the objective separately needs a terminal state |

Selection does not change the noise schedule, consume a new RNG draw, or force an
optimizer update count. Automatic gradient accumulation uses the selected number
of loss terms; explicitly configured accumulation keeps its existing precedence.

```yaml
# ART: deterministic generation, sparse replay, plus the required final latent.
train:
  trainer_type: art
  num_inference_steps: 20
scheduler:
  dynamics_type: ODE
  replay_steps: [0, 5, 10, 15]
```

```yaml
# GRPO/Guard/DPPO: explore at three steps, train at two.
scheduler:
  dynamics_type: Flow-SDE
  sde_steps: [0, 1, 2]
  num_sde_steps: null
  replay_steps: [0, 2]
```

A likelihood objective needs a stochastic transition for each selected step.
With random SDE-step selection, keep `replay_steps: auto`; a fixed subset is valid
when every member is guaranteed to be noisy. The framework rejects incompatible
selections instead of silently intersecting them and changing accumulation.
ART also needs a nonempty transition selection. These are payload requirements,
not restrictions on the number of optimizer updates or inner epochs.

## Re-noise a terminal state

NFT, AWM and DGPO save the final latent and construct a new state
`x_t = (1 - sigma_t) * x_final + sigma_t * noise`. Their default `discrete`
strategy draws times from the rollout scheduler grid within `timestep_range`.
It does **not** restore the corresponding rollout intermediate state.

The shared time sampler also supports `discrete_with_init`, `discrete_wo_init`,
`uniform` and `logit_normal`. `num_train_timesteps` counts newly drawn loss terms;
it is independent of the number of retained rollout transitions. These existing
`train.*` fields, defaults and RNG streams remain unchanged. `replay_steps` does
not select the training times or add intermediate storage for endpoint objectives.

For example, NFT can train at ten grid-sampled coordinates while retaining only
one final latent. Setting `replay_steps: null` does not discard that required
terminal state. Evaluation already requests no trajectory recording explicitly.

Distillation objectives retain their existing requirements: OPD selects a band of
real transitions through `train.timestep_range`; DMD2/TDM retain the boundaries
their later queries need. This release does not reinterpret those controls as
endpoint re-noising or silently replace their objective-specific selection.
