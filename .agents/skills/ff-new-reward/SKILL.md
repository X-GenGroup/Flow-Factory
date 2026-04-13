---
name: ff-new-reward
description: "Complete workflow for adding a new reward model. Covers pointwise vs groupwise design, __call__ contract, registration, YAML config, multi-reward setup, and verification. Trigger: 'add reward', 'new reward model', 'custom reward', 'scoring function'."
---

# New Reward Model Integration

> **Authoritative reference**: `guidance/rewards.md` — read it first.
> **Template**: `src/flow_factory/rewards/my_reward.py`

## Prerequisites

Determine your reward type:
- **Pointwise**: Each sample scored independently (e.g., aesthetic score, CLIP similarity)
- **Groupwise**: Scores depend on comparison within a group (e.g., ranking, preference)

## Phase 1: Design

1. **Choose base class**: `PointwiseRewardModel` or `GroupwiseRewardModel`
2. **Identify required inputs**: What fields from `Sample` does your reward need?
   - Common: `prompt`, `image`, `video`, `condition_images`, `condition_videos`
   - Set `required_fields` tuple accordingly
3. **Input format**: PIL Images (default) or Tensors?
   - Set `use_tensor_inputs = True` if your model needs raw tensors

## Phase 2: Implementation

### Create the reward model file

```python
# src/flow_factory/rewards/<my_reward>.py
from .abc import PointwiseRewardModel, RewardModelOutput
from ..hparams import RewardArguments
from accelerate import Accelerator
from typing import Optional, List
from PIL import Image
import torch

class MyRewardModel(PointwiseRewardModel):
    required_fields = ("prompt", "image")
    use_tensor_inputs = False

    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        super().__init__(config, accelerator)
        # Load your model, processor, etc.
        # Use self.device and self.dtype from base class

    @torch.no_grad()
    def __call__(
        self,
        prompt: List[str],
        image: Optional[List[Image.Image]] = None,
        video: Optional[List[List[Image.Image]]] = None,
        condition_images=None,
        condition_videos=None,
        **kwargs,
    ) -> RewardModelOutput:
        # Compute rewards — shape must be (batch_size,) for Pointwise
        # or (group_size,) for Groupwise
        rewards = torch.zeros(len(prompt), device=self.device)
        return RewardModelOutput(rewards=rewards)
```

### Key constraints for `__call__`:
- **Pointwise**: Input length = `config.batch_size`. Return rewards shape `(batch_size,)`
- **Groupwise**: Input length = `group_size`. You handle batching yourself. Return rewards shape `(group_size,)`
- Always use `@torch.no_grad()` decorator
- Return `RewardModelOutput` (not raw tensors)

## Phase 3: Register

Add to `_REWARD_MODEL_REGISTRY` in `src/flow_factory/rewards/registry.py`:
```python
'my_reward': 'flow_factory.rewards.<my_reward>.MyRewardModel',
```

## Phase 4: Configuration

Use in YAML config:
```yaml
rewards:
  - name: "my_reward"
    reward_model: "my_reward"        # Must match registry key
    model_path: "org/model-name"     # HuggingFace model path (if applicable)
    dtype: "bfloat16"
    device: "cuda"
    batch_size: 16
```

Multi-reward setup:
```yaml
rewards:
  - name: "aesthetic"
    reward_model: "PickScore"
    weight: 0.7
  - name: "custom"
    reward_model: "my_reward"
    weight: 0.3
```

## Phase 5: Verification

- [ ] `__init__` loads model without errors
- [ ] `__call__` returns correct reward shape
- [ ] Rewards are numerically reasonable (not all zeros, no NaN/Inf)
- [ ] Works with `RewardProcessor` dispatch (Pointwise/Groupwise routing)
- [ ] Works in multi-reward setup with weight aggregation
- [ ] Device placement correct (respects `config.device`)
- [ ] Registry entry resolves: `get_reward_model_class('my_reward')`

## Common Pitfalls

1. **Wrong return shape** — Pointwise must return `(batch_size,)`, Groupwise `(group_size,)`
2. **Forgetting `@torch.no_grad()`** — causes reward computation to build unnecessary graph, OOM
3. **Hardcoding device** — use `self.device` from base class, not `torch.device('cuda')`
4. **Not setting `required_fields`** — `RewardProcessor` won't pass the right data to your model
5. **Mixing paradigms** — don't inherit `PointwiseRewardModel` if your reward needs group context
