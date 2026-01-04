# Algorithm Guidance

## Table of Contents

1. [GRPO](#grpo)
   - 1.1 [Background](#background)
   - 1.2 [Efficiency Strategies](#efficiency-strategies)
     - 1.2.1 [Mixing SDE and ODE](#mixing-sde-and-ode)
     - 1.2.2 [Decoupled Training and Inference Resolution](#decoupled-training-and-inference-resolution)
   - 1.3 [Regularization](#regularization)

2. [DiffusionNFT](#diffusionnft)

3. [References](#references)

## GRPO

### Background

GRPO has achieved significant success in Flow Matching models. In contrast to the standard deterministic ODE-style update rule:$$x_{t+\mathrm{d}t} = x_{t} + v_{\theta}(x_t, t) \mathrm{d}t$$

References [[1]](#ref1) and [[2]](#ref2) incorporate noise to facilitate RL exploration, proposing the following SDE-based update rule:

$$
x_{t+\mathrm{d}t} = x_{t} + [v_{\theta}(x_t, t) + \frac{\sigma_{t}^{2}}{2t}(x_t + (1-t)v_{\theta}(x_t, t))]\mathrm{d}t + \sigma_{t} \sqrt{\mathrm{d}t} \epsilon
$$
where $\epsilon \sim \mathcal{N}(0, I)$ and $\sigma_t$ denotes the noise schedule. The formulation of $\sigma_t$ differs between methods: it is defined as $\eta\sqrt{\frac{t}{1-t}}$ in Flow-GRPO [[1]](#ref1) and as $\eta$ in DanceGRPO [[2]](#ref2), where $\eta \in [0,1]$ is a hyperparameter controlling the noise level.

This algorithm is implemented as `grpo`. To use this algorithm, set config with:

```yaml
train:
    trainer_type: grpo
```

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

The SDE formulation used in Flow-GRPO[[1]](#ref1) and DanceGRPO[[2]](#ref2) inherently results in a *negatively biased ratio distribution* during GRPO optimization. GRPO-Guard [[5]](#ref5) analyzes this phenomenon and proposes a normalization technique to mitigate reward hacking.

This normalization aligns with the time-step-dependent (and noise-level-dependent) loss re-weighting strategy introduced in TempFlow-GRPO[[4]](#ref4). By rebalancing the gradient contributions across different time steps, this strategy stabilizes training and effectively reduces reward hacking.

To enable this reweighting strategy, switch the `trainer_type` to `grpo-guard`:
```yaml
train:
    trainer_type: 'grpo-guard'
    dynamics_type: 'Flow-SDE'
```
> ‼️ Note: Currently, `grpo-guard` reweighting is only compatible with `Flow-GRPO` dynamics. Therefore, dynamics_type must be explicitly set to `Flow-SDE`.

## DiffusionNFT

This algorithm is introduced in [[7]](#ref7), to use this algorithm, set:
```yaml
train:
    trainer_type: 'nft'
```


### References

* <a name="ref1"></a>[1] [**Flow-GRPO:** Training Flow Matching Models via Online RL](https://arxiv.org/abs/2505.05470)
* <a name="ref2"></a>[2] [**DanceGRPO:** Unleashing GRPO on Visual Generation](https://arxiv.org/abs/2505.07818)
* <a name="ref3"></a>[3] [**MixGRPO:** Unlocking Flow-based GRPO Efficiency with Mixed ODE-SDE](https://arxiv.org/abs/2507.21802)
* <a name="ref4"></a>[4] [**TempFlow-GRPO:** When Timing Matters for GRPO in Flow Models](https://arxiv.org/abs/2508.04324)
* <a name="ref5"></a>[5] [**GRPO-Guard:** Mitigating Implicit Over-Optimization in Flow Matching via Regulated Clipping](https://arxiv.org/abs/2510.22319)
* <a name="ref6"></a>[6] [**PaCo-RL**: Advancing Reinforcement Learning for Consistent Image Generation with Pairwise Reward Modeling](https://arxiv.org/abs/2512.04784)
* <a name="ref7"></a>[7] [**DiffusionNFT**: Online Diffusion Reinforcement with Forward Process](https://arxiv.org/abs/2509.16117)