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

### GRPO, Guard and DPPO: start with `auto`

**`num_sde_steps: 2` with `replay_steps: auto` preserves the existing automatic
selection behavior.** Omitting `replay_steps` has the same effect because `auto`
is its default. Existing configurations do not need a new replay setting.

These three settings act in sequence:

| Setting | Meaning | Default / YAML `null` |
|---|---|---|
| `sde_steps` | Candidate transition indices eligible for noise | Omitted or `null`: `0 .. T-2`, excluding the final transition |
| `num_sde_steps` | Number of distinct candidates that receive noise | Omitted: **1**. `null`: **every candidate**. An integer must be between 1 and the candidate count |
| `replay_steps` | Transitions used by the training objective | Omitted or `auto`: the **actual noisy selection**. `null` means no replay and is invalid for these likelihood objectives |

The scheduler draws without replacement when the count is smaller than the
candidate pool. The draw is reproducible under its current seed. `auto` follows
that same selection and order; it does not draw another subset. Unselected
transitions still execute during generation, with zero SDE noise.

**Randomly select two steps from the default pool:**

```yaml
train:
  trainer_type: grpo  # The scheduler settings also apply to grpo-guard and dppo.
  num_inference_steps: 20
scheduler:
  dynamics_type: Flow-SDE
  noise_level: 0.7
  sde_steps: null
  num_sde_steps: 2
  replay_steps: auto  # Optional: this is already the default.
```

Here the candidate pool is `[0, ..., 18]`. If the realized selection is `[7, 2]`,
those two transitions receive noise and are replayed for training. All 20
generation transitions still run; the other 18 are deterministic. The two
replayed transitions contribute two loss terms per sample, not necessarily two
optimizer updates.

**Randomly select two steps from four explicit candidates:**

```yaml
train:
  trainer_type: grpo  # Also applies to grpo-guard and dppo.
  num_inference_steps: 20
scheduler:
  dynamics_type: Flow-SDE
  noise_level: 0.7
  sde_steps: [0, 1, 2, 3]  # The candidate pool, not four guaranteed noisy steps.
  num_sde_steps: 2         # Draw two distinct positions from that pool.
  replay_steps: auto      # Train on exactly the two positions drawn.
```

For example, if the draw is `[2, 0]`, only transitions 2 and 0 receive noise and
are replayed for training. Candidates 1 and 3, and every transition outside the
pool, still execute with zero SDE noise. The actual pair depends on the scheduler
seed; `[2, 0]` is illustrative. Setting `num_sde_steps: 4` or `null` instead would
make all four candidates noisy and train on all four with `auto`.

**Always add noise and train at two fixed steps:**

```yaml
scheduler:
  dynamics_type: Flow-SDE
  sde_steps: [0, 2]
  num_sde_steps: null  # Use both candidates; 2 is equivalent for this pool.
  replay_steps: auto
```

**Add noise at three fixed steps, but train at only two of them:**

```yaml
scheduler:
  dynamics_type: Flow-SDE
  sde_steps: [0, 1, 2]
  num_sde_steps: null  # All three are noisy on every rollout.
  replay_steps: [0, 2]
```

Step 1 still injects noise and affects the generated sample, but contributes no
replay loss. This is an optional way to separate exploration from training;
ordinary random two-step training only needs the first example.

For a pool of `[0, 1, 2]`, the combinations are:

| `num_sde_steps` | `replay_steps` | Noisy transitions | Training transitions |
|---|---|---|---|
| omitted (`1`) | omitted (`auto`) | One randomly selected candidate | That same one |
| `2` | `auto` | Two randomly selected candidates | Those same two |
| `null` or `3` | `auto` | `[0, 1, 2]` | `[0, 1, 2]` |
| `null` or `3` | `[0, 2]` | `[0, 1, 2]` | `[0, 2]` |
| `2` | `[0, 2]` | Two randomly selected candidates | **Rejected:** either requested position could be deterministic |

GRPO, Guard and DPPO compute transition likelihoods, so every replay position
must have stochastic dynamics and positive noise. With a fixed replay list,
set `num_sde_steps: null` (or the full pool size) and choose positions inside
`sde_steps`. For example, a random selection of `[0, 1]` cannot supply the noisy
transition required by a fixed replay list of `[0, 2]`. The framework rejects
that configuration instead of silently dropping step 2 and changing loss counts.

These requirements are checked while resolving the configuration, including when
`train.gradient_accumulation_steps` is set explicitly, before the training loader
creates an Accelerator or loads model weights. Counting replay loss terms does not
perform admission or sample a random set of positions;
the optimizer loop reads the scheduler's actual selection for that rollout.

`replay_steps: all` means **all `T` generation transitions**, not all noisy
transitions. It therefore usually includes deterministic steps and is unsuitable
for these objectives; use `auto` to train on all *realized noisy* transitions.
Also, `num_sde_steps: null` means all candidates, whereas omitting it keeps the
default count of one. ODE generation and an empty replay selection cannot supply
the stochastic transition likelihoods these trainers require.

### ART: select ODE transitions

For ART, `auto` selects every ODE transition. An explicit list can reduce stored
states and replay work while the objective still retains its required final latent:

```yaml
train:
  trainer_type: art
  num_inference_steps: 20
scheduler:
  dynamics_type: ODE
  replay_steps: [0, 5, 10, 15]
```

ART needs a nonempty transition selection. ODE sets the SDE fields to an empty
pool, zero noisy steps and zero noise; `num_sde_steps` does not limit ART replay.
These are payload requirements, not restrictions on the number of optimizer
updates or inner epochs.

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
