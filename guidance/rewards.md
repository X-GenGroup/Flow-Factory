# Reward Model Guidance

Flow-Factory provides a flexible reward model system that supports both built-in and custom reward models for reinforcement learning.

## Table of Contents

- [Built-in Reward Models](#built-in-reward-models)
- [Using Built-in Reward Models](#using-built-in-reward-models)
- [Creating Custom Reward Models](#creating-custom-reward-models)
- [Decoupling Training and Evaluation Reward Models](#decoupling-training-and-evaluation-reward-models)

## Built-in Reward Models

The following reward models are pre-registered and ready to use:

| Name | Description | Reference |
|------|-------------|-----------|
| `PickScore` | CLIP-based aesthetic scoring model | [PickScore](https://huggingface.co/yuvalkirstain/PickScore_v1) |

## Using Built-in Reward Models

Simply specify the reward model name in your config file:
```yaml
reward:
  reward_model: "PickScore"
  dtype: "bfloat16"
  device: "cuda"
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
    def __init__(self, reward_args: RewardArguments, accelerator: Accelerator):
        super().__init__(reward_args, accelerator)
        # Initialize your model here
        # Available attributes:
        # self.reward_args = reward_args
        # - self.device: device to load model on. If `reward_args.device='cuda'`, it is `accelerator.device`
        # - self.dtype: data type for model - same as `reward_args.dtype`
        # - self.accelerator: accelerator instance
        
    @torch.no_grad()
    def __call__(
        self,
        prompt: list[str],
        image: Optional[list[Image.Image]] = None,
        video: Optional[list[list[Image.Image]]] = None,
        **kwargs,
    ) -> RewardModelOutput:
        # Compute rewards
        rewards = torch.randn(len(prompt), device=self.device)
        
        return RewardModelOutput(
            rewards=rewards,
            extra_info={},  # Optional metadata
        )
```

**2. Register and use in config:**
```yaml
reward:
  reward_model: "flow_factory.rewards.CustomRewardModel"  # Full Python path
  dtype: "bfloat16"
  device: "cuda"
  batch_size: 16
```

Alternatively, register your model using the decorator:
```python
from flow_factory.rewards import register_reward_model

@register_reward_model('MyReward')
class CustomRewardModel(BaseRewardModel):
    ...
```

Then use the registered name:
```yaml
reward:
  reward_model: "MyReward"  # Use registered name
```

## Decoupling Training and Evaluation Reward Models

You can use different reward models for training and evaluation by specifying `eval_reward`:
```yaml
# Training reward model
reward:
  reward_model: "PickScore"
  dtype: "bfloat16"
  batch_size: 16

# Evaluation reward model (optional)
eval_reward:
  reward_model: "my_rewards.hps_v2.HPSv2RewardModel"
  dtype: "float32"
  batch_size: 8
```

If `eval_reward` is not specified, the training reward model will be reused for evaluation (no duplicate loading).

**Use cases:**
- Train with fast reward model, evaluate with more accurate but slower model
- Compare multiple reward signals during evaluation
- Cross-model evaluation to avoid potential overfitting