# Reward Model Guidance

Flow-Factory provides a flexible reward model system that supports both built-in and custom reward models for reinforcement learning.

## Table of Contents

- [Reward Model Types](#reward-model-types)
- [Built-in Reward Models](#built-in-reward-models)
- [Using Built-in Reward Models](#using-built-in-reward-models)
- [Creating Custom Reward Models](#creating-custom-reward-models)
  - [Pointwise Reward Model](#pointwise-reward-model)
  - [Groupwise Reward Model](#groupwise-reward-model)
  - [Class Attributes](#class-attributes)
- [Remote Reward Server](#remote-reward-server)
- [Multi-Reward Training](#multi-reward-training)
- [Decoupling Training and Evaluation Reward Models](#decoupling-training-and-evaluation-reward-models)

## Reward Model Types

Flow-Factory supports two paradigms for computing rewards:

| Type | Description |
|------|-------------|
| **Pointwise** | Computes independent scores for each sample |
| **Groupwise** | Computes rewards that requires all samples of a group|

**Pointwise** models evaluate each sample independently, returning absolute scores (e.g., PickScore, CLIP similarity).

**Groupwise** models evaluate all samples in a group together, enabling rewards that depend on how a sample compares to others in the same group.

## Built-in Reward Models

| Name | Type | Description | Reference |
|------|------|-------------|-----------|
| `PickScore` | Pointwise | CLIP-based aesthetic scoring | [PickScore](https://huggingface.co/yuvalkirstain/PickScore_v1) |
| `CLIP` | Pointwise | Image-text cosine similarity | [CLIP](https://huggingface.co/openai/clip-vit-large-patch14) |
| `PickScore_Rank` | Groupwise | Ranking-based reward using PickScore | [PickScore](https://huggingface.co/yuvalkirstain/PickScore_v1) |

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

### Pointwise Reward Model

Pointwise models receive batches of size `batch_size` and compute independent scores.
```python
# src/flow_factory/rewards/my_reward.py
from flow_factory.rewards import PointwiseRewardModel, RewardModelOutput
from flow_factory.hparams import RewardArguments
from accelerate import Accelerator
from typing import Optional, List
from PIL import Image
import torch

class MyPointwiseReward(PointwiseRewardModel):
    """Custom pointwise reward model."""
    
    required_fields = ("prompt", "image")  # Declare required inputs
    
    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        super().__init__(config, accelerator)
        # Available: self.config, self.device, self.dtype, self.accelerator
        # Initialize your model here
        
    @torch.no_grad()
    def __call__(
        self,
        prompt: List[str],
        image: Optional[List[Image.Image]] = None,
        video: Optional[List[List[Image.Image]]] = None,
        condition_images: Optional[List[List[Image.Image]]] = None,
        condition_videos: Optional[List[List[List[Image.Image]]]] = None,
    ) -> RewardModelOutput:
        # Input length equals self.config.batch_size
        rewards = torch.zeros(len(prompt), device=self.device)
        return RewardModelOutput(rewards=rewards)
```

### Groupwise Reward Model

Groupwise models receive the entire group at once and handle batching internally.
```python
# src/flow_factory/rewards/my_reward.py
from flow_factory.rewards import GroupwiseRewardModel, RewardModelOutput
from flow_factory.hparams import RewardArguments
from accelerate import Accelerator
from typing import Optional, List
from PIL import Image
import torch

class MyGroupwiseReward(GroupwiseRewardModel):
    """Custom groupwise reward model with ranking."""
    
    required_fields = ("prompt", "image")
    
    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        super().__init__(config, accelerator)
        # Initialize your scoring model here
        
    @torch.no_grad()
    def __call__(
        self,
        prompt: List[str],
        image: Optional[List[Image.Image]] = None,
        video: Optional[List[List[Image.Image]]] = None,
        condition_images: Optional[List[List[Image.Image]]] = None,
        condition_videos: Optional[List[List[List[Image.Image]]]] = None,
    ) -> RewardModelOutput:
        # Input length equals group_size (NOT batch_size)
        # Handle batching internally using self.config.batch_size
        group_size = len(prompt)
        
        # Example: compute scores in batches, then rank
        all_scores = []
        for i in range(0, group_size, self.config.batch_size):
            batch_scores = self._score_batch(
                prompt[i:i + self.config.batch_size],
                image[i:i + self.config.batch_size],
            )
            all_scores.append(batch_scores)
        
        raw_scores = torch.cat(all_scores, dim=0)
        
        # Convert to rank-based rewards: [0, 1, ..., n-1] / n
        ranks = raw_scores.argsort().argsort()
        rewards = ranks.float() / group_size
        
        return RewardModelOutput(rewards=rewards)
```

### Class Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `required_fields` | `Tuple[str, ...]` | `("prompt", "image")` | Fields required from `Sample` for reward computation |
| `use_tensor_inputs` | `bool` | `False` | Input format for media fields |

`use_tensor_inputs` controls the format of media inputs (`image`, `video`, `condition_images`, `condition_videos`):

| Value | Format |
|-------|--------|
| `False` (default) | PIL Images |
| `True` | PyTorch Tensors (range `[0, 1]`) |

**Tensor shapes when `use_tensor_inputs=True`:**

| Field | Shape |
|-------|-------|
| `image` | `List[Tensor(C, H, W)]` |
| `video` | `List[Tensor(T, C, H, W)]` |
| `condition_images` | `List[Tensor(N, C, H, W)]` or `List[List[Tensor(C, H, W)]]`* |
| `condition_videos` | `List[Tensor(N, T, C, H, W)]` or `List[List[Tensor(T, C, H, W)]]`* |

*Stacked tensor if all conditions have same size; nested list otherwise.

**Example with tensor inputs:**
```python
class TensorBasedReward(PointwiseRewardModel):
    """Reward model that operates directly on tensors."""
    
    required_fields = ("prompt", "image") # Do not add unnecessary field since it may require more process communications.
    use_tensor_inputs = True  # Receive tensors instead of PIL
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: List[str],
        image: Optional[List[torch.Tensor]] = None,  # List of (C, H, W) tensors, range in [0, 1]
        video: Optional[List[torch.Tensor] = None, # List of (T, C, H, W) tensors, range in [0, 1]
        condition_images: Optional[List[Union[torch.Tensor, List[torch.Tensor]]]] = None, # A batch of condition image list
        condition_videos: Optional[List[Union[torch.Tensor, List[torch.Tensor]]]] = None, # A batch of condition video list
    ) -> RewardModelOutput:
        # Stack and process directly on GPU
        rewards = torch.zeros_like(prompt, dtype=torch.float32)
        return RewardModelOutput(rewards=rewards)
```

### Model Type Comparison

| Aspect | Pointwise | Groupwise |
|--------|-----------|-----------|
| Input size | `batch_size` samples | `group_size` samples |
| Batching | Handled by trainer | Handled internally |
| Reward semantics | Absolute scores | Relative/ranking-based |

### Register and Use
```yaml
rewards:
  - name: "custom"
    reward_model: "flow_factory.rewards.MyPointwiseReward"  # Full Python path
    batch_size: 16
```

## Remote Reward Server

For reward models with incompatible dependencies (different Python versions, CUDA requirements, or conflicting packages), Flow-Factory supports running reward computation in an **isolated environment** via HTTP.

### Architecture

```
Training Process (Flow-Factory)          Reward Server (Isolated Env)
┌────────────────────────────┐          ┌────────────────────────────┐
│ RemotePointwiseRewardModel │◄──HTTP──►│ YourRewardServer           │
│ (auto serialization)       │          │ (implement compute_reward) │
└────────────────────────────┘          └────────────────────────────┘
```

### Server Setup

Check `reward_server/example_server.py` and implement `compute_reward()`:

```python
# reward_server/example_server.py
from typing import List, Optional
from PIL import Image

class MyRewardServer(RewardServer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = load_your_model()  # Initialize your model

    def compute_reward(
        self,
        prompts: List[str],
        images: Optional[List[Image.Image]] = None,
        videos: Optional[List[List[Image.Image]]] = None,
        **kwargs,
    ) -> List[float]:
        # Your reward logic here
        return [self.model.score(p, i) for p, i in zip(prompts, images)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reward Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    server = MyRewardServer(host=args.host, port=args.port)
    server.run()
```

Start the server before training:
```bash
# In your reward model's environment
conda activate reward_env
python my_reward_server.py --port 8000
```

### Training Config

```yaml
rewards:
  - name: "remote_reward"
    reward_model: "flow_factory.rewards.remote.RemotePointwiseRewardModel"
    server_url: "http://localhost:8000" # Use different ports if your have multiple reward servers
    batch_size: 16
    timeout: 60.0        # optional, default 60s
    retry_attempts: 3    # optional, default 3
```

For groupwise rewards, use `RemoteGroupwiseRewardModel`:
```yaml
rewards:
  - name: "remote_ranking"
    reward_model: "flow_factory.rewards.remote.RemoteGroupwiseRewardModel"
    server_url: "http://localhost:8000"
    timeout: 60.0        # optional, default 60s
    retry_attempts: 3    # optional, default 3
```

### Server Dependencies

Install via optional dependency:
```bash
pip install "flow-factory[reward-server]"
```

Or install manually (for isolated environments without Flow-Factory):
```bash
pip install fastapi uvicorn pillow
```

### When to Use

Use Remote Reward Server **only when reward model dependencies conflict with Flow-Factory**, such as:

| Scenario | Example |
|----------|---------|
| PyTorch version conflict | Reward model requires PyTorch 1.x |
| Package conflict | Reward model needs an older `transformers` version |
| Python version mismatch | Reward model only supports Python 3.8 |

**When NOT to use** (prefer direct implementation in `flow_factory.rewards.my_reward`):

| Scenario | Recommended Approach |
|----------|---------------------|
| VLM-based reward | Deploy with vLLM/SGLang, call via OpenAI SDK in your customized reward model |
| Closed-source API | Use `requests`, OpenAI SDK or official SDK directly in `__call__()` |
| Compatible dependencies | Implement as standard `PointwiseRewardModel` |

## Multi-Reward Training

Train with multiple reward signals by adding entries to `rewards`:
```yaml
rewards:
  - name: "aesthetic"
    reward_model: "PickScore"
    weight: 1.0
    batch_size: 16
    
  - name: "text_align"
    reward_model: "CLIP"
    weight: 0.5
    batch_size: 32
```

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