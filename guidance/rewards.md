# Reward Model Guidance

Flow-Factory provides a flexible reward model system that supports both built-in and custom reward models for reinforcement learning.

## Table of Contents

- [Built-in Reward Models](#built-in-reward-models)
- [Using Built-in Reward Models](#using-built-in-reward-models)
- [Creating Custom Reward Models](#creating-custom-reward-models)
- [Multi-Reward Training](#multi-reward-training)
- [Decoupling Training and Evaluation Reward Models](#decoupling-training-and-evaluation-reward-models)

## Built-in Reward Models

The following reward models are pre-registered and ready to use:

| Name | Description | Reference |
|------|-------------|-----------|
| `PickScore` | CLIP-based aesthetic scoring model | [PickScore](https://huggingface.co/yuvalkirstain/PickScore_v1) |

## Using Built-in Reward Models

Simply specify the reward model in your config file:

```yaml
rewards:
  - name: "aesthetic"
    reward_model: "PickScore"
    dtype: "bfloat16"
    device: "cuda"
    batch_size: 16
```

For single reward, you can also use the shorthand format:

```yaml
rewards:
  name: "aesthetic"
  reward_model: "PickScore"
  batch_size: 16
```

## Creating Custom Reward Models

To implement a custom reward model, refer to `src/flow_factory/rewards/my_reward.py`.

**1. Create your reward model class:**

```python
# src/flow_factory/rewards/my_reward.py
from flow_factory.rewards import BaseRewardModel, RewardModelOutput
from flow_factory.hparams import RewardArguments
from accelerate import Accelerator
import torch

class CustomRewardModel(BaseRewardModel):
    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        super().__init__(config, accelerator)
        # Available attributes:
        # - self.config: RewardArguments instance
        # - self.device: accelerator.device if config.device='cuda', else config.device
        # - self.dtype: same as config.dtype
        # - self.accelerator: accelerator instance
        
    @torch.no_grad()
    def __call__(
        self,
        prompt: list[str],
        image: Optional[list[Image.Image]] = None,
        video: Optional[list[list[Image.Image]]] = None,
        **kwargs,
    ) -> RewardModelOutput:
        rewards = torch.randn(len(prompt), device=self.device)
        return RewardModelOutput(rewards=rewards)
```

**2. Register and use in config:**

```yaml
rewards:
  - name: "custom"
    reward_model: "flow_factory.rewards.CustomRewardModel"  # Full Python path
    batch_size: 16
```

Alternatively, register with decorator:

```python
from flow_factory.rewards import register_reward_model

@register_reward_model('MyReward')
class CustomRewardModel(BaseRewardModel):
    ...
```

Then use:

```yaml
rewards:
  - name: "custom"
    reward_model: "MyReward"  # Registered name
```

## Multi-Reward Training

Train with multiple reward signals by adding entries to `rewards`:

```yaml
rewards:
  - name: "aesthetic"
    reward_model: "PickScore"
    weight: 1.0
    batch_size: 16
    
  - name: "text_align"
    reward_model: "CLIPScore"
    weight: 0.5
    batch_size: 32
```

Rewards are aggregated as weighted sum: $\text{total} = (\text{weight}_{i} \times \text{reward}_{i})$

**Automatic deduplication:** Identical configurations share the same model instance to save GPU memory.

```yaml
rewards:
  - name: "aesthetic_1"
    reward_model: "PickScore"
    batch_size: 16
    
  - name: "aesthetic_2"
    reward_model: "PickScore"  # Same config → reuses model above
    batch_size: 16
```

## Decoupling Training and Evaluation Reward Models

Use different reward models for training and evaluation:

```yaml
# Training rewards
rewards:
  - name: "fast_score"
    reward_model: "PickScore"
    batch_size: 32

# Evaluation rewards (optional)
eval_rewards:
  - name: "hps"
    reward_model: "my_rewards.HPSv2RewardModel"
    batch_size: 8
```

If `eval_rewards` is not specified, training rewards are reused for evaluation.

**Use cases:**
- Train with fast model, evaluate with slower but more accurate model
- Cross-model evaluation to detect overfitting