# Algorithm Guidance

## Table of Contents

- [Overview](#overview)
- [GRPO](#grpo)
   - [Background](#background)
   - [Dynamics Type](#dynamics-type)
   - [Efficiency Strategies](#efficiency-strategies)
     - [Mixing SDE and ODE](#mixing-sde-and-ode)
     - [Decoupled Training and Inference Resolution](#decoupled-training-and-inference-resolution)
   - [Regularization](#regularization)
     - [KL-loss](#kl-loss)
     - [GRPO-Guard](#grpo-guard)

- [DiffusionNFT](#diffusionnft)

- [AWM: Advantage Weighted Matching](#awm-advantage-weighted-matching)

- [References](#references)

## Overview

Flow-Factory provides unified implementations of state-of-the-art RL algorithms for flow-matching models. All algorithms share the same model adapter and reward interfaces, enabling direct comparison under controlled conditions.

At a high level, the supported algorithms fall into two paradigms:

- **Coupled paradigm (GRPO and variants)**: Training timesteps are coupled with the SDE-based sampling dynamics, requiring tractable log-probability computation for policy gradient optimization.
- **Decoupled paradigm (DiffusionNFT, AWM)**: Training timesteps are decoupled from the actual sampling dynamics, making them inherently solver-agnostic — any ODE solver can be used for trajectory generation without modifying the training procedure.

## GRPO

### Background

GRPO has achieved significant success in Flow Matching models. In contrast to the standard deterministic ODE-style update rule:

$$
x_{t+\mathrm{d}t} = x_{t} + v_{\theta}(x_t, t) \mathrm{d}t
$$

References [[1]](#ref1) and [[2]](#ref2) incorporate noise to facilitate RL exploration, proposing the following SDE-based update rule:

$$
x_{t+\mathrm{d}t} = x_{t} + [v_{\theta}(x_t, t) + \frac{\sigma_{t}^{2}}{2t}(x_t + (1-t)v_{\theta}(x_t, t))]\mathrm{d}t + \sigma_{t} \sqrt{\mathrm{d}t} \epsilon
$$

where $\epsilon \sim \mathcal{N}(0, I)$ and $\sigma_t$ denotes the noise schedule. This SDE formulation enables the log-probability computation required for policy gradient optimization.

The formulation of $\sigma_t$ differs between methods: it is defined as $\eta\sqrt{\frac{t}{1-t}}$ in Flow-GRPO [[1]](#ref1) and as $\eta$ in DanceGRPO [[2]](#ref2), where $\eta \in [0,1]$ is a hyperparameter controlling the noise level. See the [Dynamics Type](#dynamics-type) section for a complete summary.

This algorithm is implemented as `grpo`. To use this algorithm, set config with:

```yaml
train:
    trainer_type: grpo
```

### Dynamics Type

Flow-Factory implements multiple SDE dynamics through a unified `SDESchedulerMixin` interface. Users can switch between formulations via a single configuration parameter, facilitating systematic comparison of their effects on training stability and sample quality.

| Dynamics   | Noise Schedule $\sigma_t$              | Reference                    |
|------------|----------------------------------------|------------------------------|
| `Flow-SDE` | $\eta\sqrt{t/(1-t)}$                 | Flow-GRPO [[1]](#ref1)       |
| `Dance-SDE`| $\eta$ (constant)                     | DanceGRPO [[2]](#ref2)       |
| `CPS`      | $\sigma_{t-1}\sin(\eta\pi/2)$        | FlowCPS [[8]](#ref8)         |
| `ODE`      | $0$ (deterministic)                   | For NFT [[7]](#ref7) / AWM [[9]](#ref9) |

To switch between these formulations, set:

```yaml
train:
    dynamics_type: 'Flow-SDE' # Options are ['Flow-SDE', 'Dance-SDE', 'CPS', 'ODE'].
```

> **Note**: `ODE` dynamics produce deterministic trajectories and cannot provide log-probability estimates. Therefore, `ODE` can only be used with decoupled algorithms such as `NFT` and `AWM`. See the [DiffusionNFT](#diffusionnft) and [AWM](#awm-advantage-weighted-matching) sections.


### Efficiency Strategies


#### Mixing SDE and ODE

Training with the original Flow-GRPO and DanceGRPO methods is computationally expensive, as they require computing log probabilities and optimizing across all denoising steps.

Subsequent works, such as MixGRPO [[3]](#ref3) and TempFlow-GRPO [[4]](#ref4), investigated the effects of mixing ODE and SDE denoising rules. They found that applying SDE updates for only $1\sim 2$ steps—and optimizing only those corresponding steps—is sufficient. This approach significantly reduces the cost of the optimization stage and results in faster performance improvements.

To control this behavior, you can configure `train_steps` and `num_train_steps` as follows:

```yaml
train:
    # Candidate steps for SDE noise (early steps typically provide more sample diversity)
    train_steps: [1, 2, 3] 
    
    # Randomly select `1` step from the specified `train_steps` list (e.g., step 2) 
    # to use SDE denoising. All other steps will use the standard ODE solver.
    num_train_steps: 1
```

#### Decoupled Training and Inference Resolution

Flow-GRPO demonstrates that *lower-quality images, generated via fewer denoising steps, are often sufficient for reward computation and GRPO optimization*. PaCo-RL[[6]](#ref6) validates this insight from the perspective of **resolution**.

Research indicates that training on moderately low-resolution images yields sufficient reward signals to guide optimization effectively. Furthermore, *performance gains achieved at lower resolutions successfully transfer to high-resolution outputs*. Given that the computational complexity of modern Diffusion Transformers grows quadratically with image resolution, this decoupling significantly reduces training costs.

You can configure a smaller resolution for the sampling and optimization loop while maintaining the target resolution for inference and evaluation:

```yaml
train:
    resolution: 256  # Reduced resolution (int or [height, width]) for faster RL loops
eval:
    resolution: 1024 # Full resolution for validation and inference
```

### Regularization

#### KL-Loss

To tame the policy model's behavior and maintain proximity to the original reference model, two types of KL loss are available:

```yaml
train:
    kl_type: 'v-based' # Options: 'x-based', 'v-based'
    kl_beta: 0.04 # KL divergence beta
    ref_param_device: 'same_as_model' # Options: cpu, same_as_model
```

Here, `x-based` calculates the KL loss in the **latent space**,
while v-based calculates it in the **predicted velocity space** (or **noise space**).
The `kl_beta` parameter controls the coefficient of the KL divergence term.

**Memory Considerations**: Since calculating KL loss requires maintaining a copy of the original model, *VRAM usage scales with the number of trainable parameters*. 
- **LoRA Training**: The overhead is minimal and efficient.
- **Full-Parameter Fine-Tuning**: The overhead is significant. You may want to set `ref_param_device` to `cpu` to save memory.
- **No KL-Loss**: Setting `kl_beta` to `0` automatically disables this term and eliminates extra memory usage.


#### GRPO-Guard

The SDE formulation used in Flow-GRPO[[1]](#ref1) and DanceGRPO[[2]](#ref2) inherently results in a *negatively biased ratio distribution* during GRPO optimization. GRPO-Guard [[5]](#ref5) analyzes this phenomenon and proposes a normalization technique to mitigate reward hacking.

This normalization aligns with the time-step-dependent (and noise-level-dependent) loss re-weighting strategy introduced in TempFlow-GRPO[[4]](#ref4). By rebalancing the gradient contributions across different time steps, this strategy stabilizes training and effectively reduces reward hacking.

To enable this reweighting strategy, switch the `trainer_type` to `grpo-guard`:
```yaml
train:
    trainer_type: 'grpo-guard'
    dynamics_type: 'Flow-SDE'
```
> ‼️ **Note**: Currently, `grpo-guard` reweighting is only compatible with `Flow-GRPO` dynamics. Therefore, dynamics_type must be explicitly set to `Flow-SDE`.

## DiffusionNFT

This algorithm is introduced in [[7]](#ref7). Unlike GRPO, which couples sampling dynamics with training timesteps, **DiffusionNFT** decouples them entirely by optimizing a contrastive objective directly on the forward flow-matching process.

Concretely, DiffusionNFT contrasts implicit positive and negative policies ($v_\theta^+$ and $v_\theta^-$), weighted by a normalized reward $r \in [0, 1]$, to identify a policy improvement direction *without* requiring tractable likelihood estimation or SDE-based sampling. This makes the algorithm inherently solver-agnostic.

To use this algorithm, set:

```yaml
train:
    trainer_type: 'nft'
```

Since DiffusionNFT decouples training from sampling dynamics, you can freely choose the sampling solver. Using the `ODE` solver during sampling typically yields higher image quality:

```yaml
train:
  num_train_timesteps: 2 # Number of timesteps to train on. Set `null` to all timesteps.
  time_sampling_strategy: discrete_with_init # Options: uniform, logit_normal, discrete, discrete_with_init, discrete_wo_init
  time_shift: 3.0
  timestep_fraction: 0.3 # Train using only the first 30% of timesteps.

scheduler:
    dynamics_type: 'ODE' # Other options are also available.
```

> **Note**: Since Reinforcement Learning typically requires exploration, it is often beneficial to experiment with SDE-based `dynamics_type` settings as well. Using `CPS`[[8]](#ref8) for NFT sampling is also a good choice.

### Old Policy via EMA

The original DiffusionNFT implementation maintains two separate EMA copies of the model: one for general EMA smoothing and one as the "old policy" used for off-policy sampling. Flow-Factory simplifies this design by retaining only a single EMA copy that serves as the old policy. This reduces memory overhead while preserving the core stabilization mechanism.

When `off_policy` is enabled, the EMA model is used to generate trajectories during sampling, while the current policy is optimized against these trajectories. This off-policy setup stabilizes training by preventing the sampling distribution from shifting too rapidly.

```yaml
train:
  off_policy: true  # Use EMA parameters for off-policy sampling
  ema_decay_schedule: "piecewise_linear"  # Options: constant, power, linear, piecewise_linear, cosine, warmup_cosine
  ema_decay: 0.5        # EMA decay rate (0 to disable)
  ema_update_interval: 1  # EMA update interval (in epochs)
  ema_device: "cuda"      # Device to store EMA model (options: cpu, cuda)
```

> **Tip**: The `piecewise_linear` schedule is recommended for DiffusionNFT. It starts with a lower decay rate to allow faster initial policy divergence and gradually increases the decay to stabilize later training. You can fine-tune this behavior with `flat_steps` and `ramp_rate`.

## AWM: Advantage Weighted Matching

This algorithm is introduced in [[9]](#ref9). **Advantage Weighted Matching** further aligns RL optimization with the flow-matching pretraining objective by weighting the standard velocity matching loss with per-sample advantages. This formulation incorporates reward-based guidance directly into the velocity matching loss, effectively aligning the optimization target with the original flow-matching objective.

Like DiffusionNFT, AWM decouples training from sampling dynamics and is therefore solver-agnostic. To use this algorithm, set:

```yaml
train:
    trainer_type: 'awm'
```

The relevant sampling and timestep configuration parameters are the same as those described in the [DiffusionNFT](#diffusionnft) section.

### Training Stability

AWM typically converges faster than other algorithms due to its direct advantage weighting on the velocity matching loss. However, this rapid update dynamic also makes it more prone to training instability — the policy can diverge quickly if left unconstrained, leading to reward hacking or training collapse.

To stabilize AWM training, it is strongly recommended to combine **EMA-based KL regularization** with **PPO-style clipping**:

```yaml
train:
  trainer_type: 'awm'
  # EMA KL regularization: penalizes deviation from the EMA-smoothed policy
  ema_kl_beta: 0.1        # Coefficient of KL loss between current policy and EMA policy
  ema_decay: 0.9           # EMA decay rate
  ema_decay_schedule: 'power'  # Options: constant, power, linear, piecewise_linear, cosine, warmup_cosine
  ema_update_interval: 1   # EMA update interval (in epochs)
  ema_device: "cuda"
  # PPO-style clipping: prevents excessively large policy updates
  clip_range: 1.0e-5       # Clipping range for the policy ratio
  adv_clip_range: 5.0      # Advantage clipping range
```

> ‼️ **Important**: Disabling both `ema_kl_beta` and `clip_range` simultaneously is **not recommended** for AWM, as the unconstrained advantage weighting can easily lead to training collapse. In practice, `ema_kl_beta` serves as a soft constraint that keeps the current policy close to a moving average, while `clip_range` provides a hard constraint on per-step policy updates.

### AWM Weighting

AWM computes a per-sample matching loss $\ell = \|v_\theta(x_t, t) - ({\epsilon} - {x}_0)\|^2$ and then applies a weighting function $w(\ell, t)$ before multiplying by the advantage. Different weighting strategies control how the raw matching loss magnitude and timestep position influence the gradient signal:

```yaml
train:
  awm_weighting: 'ghuber'  # Options: Uniform, t, t**2, huber, ghuber
  ghuber_power: 0.25        # Power parameter for generalized Huber weighting (only used with 'ghuber')
```

| Weighting  | Formula $w(\ell, t)$                                                  | Description                                                                                           |
|------------|-----------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|
| `Uniform`  | $\ell$                                                                | No reweighting. All timesteps contribute equally.                                                     |
| `t`        | $t \cdot \ell$                                                        | Linear timestep weighting. Upweights noisier (larger $t$) timesteps.                                  |
| `t**2`     | $t^2 \cdot \ell$                                                      | Quadratic timestep weighting. More aggressively upweights noisier timesteps.                          |
| `huber`    | $t \cdot (\sqrt{\ell + \varepsilon} - \varepsilon)$                   | Huber-style loss that suppresses large matching errors, weighted by $t$.                              |
| `ghuber`   | $\frac{t}{p} \cdot ((\ell + \varepsilon)^{p} - \varepsilon^{p})$     | Generalized Huber loss with power $p$ (`ghuber_power`). Provides tunable robustness against outliers. |

Here $\varepsilon$ is a small constant for numerical stability and $p$ denotes `ghuber_power` (default `0.25`).

> **Tip**: `ghuber` with a small power (e.g., `0.25`) provides a good balance between robustness and gradient signal strength. `Uniform` is the simplest baseline and works well when reward signals are clean and low-variance.


## References

* <a name="ref1"></a>[1] [**Flow-GRPO:** Training Flow Matching Models via Online RL](https://arxiv.org/abs/2505.05470)
* <a name="ref2"></a>[2] [**DanceGRPO:** Unleashing GRPO on Visual Generation](https://arxiv.org/abs/2505.07818)
* <a name="ref3"></a>[3] [**MixGRPO:** Unlocking Flow-based GRPO Efficiency with Mixed ODE-SDE](https://arxiv.org/abs/2507.21802)
* <a name="ref4"></a>[4] [**TempFlow-GRPO:** When Timing Matters for GRPO in Flow Models](https://arxiv.org/abs/2508.04324)
* <a name="ref5"></a>[5] [**GRPO-Guard:** Mitigating Implicit Over-Optimization in Flow Matching via Regulated Clipping](https://arxiv.org/abs/2510.22319)
* <a name="ref6"></a>[6] [**PaCo-RL**: Advancing Reinforcement Learning for Consistent Image Generation with Pairwise Reward Modeling](https://arxiv.org/abs/2512.04784)
* <a name="ref7"></a>[7] [**DiffusionNFT**: Online Diffusion Reinforcement with Forward Process](https://arxiv.org/abs/2509.16117)
* <a name="ref8"></a>[8] [**<u>C</u>oefficients-<u>P</u>reserving <u>S</u>ampling** for Reinforcement Learning with Flow Matching](https://arxiv.org/abs/2509.05952)
* <a name="ref9"></a>[9] [**<u>A</u>dvantage <u>W</u>eighted <u>M</u>atching**: Aligning RL with Pretraining in Diffusion Models](https://arxiv.org/abs/2509.25050)