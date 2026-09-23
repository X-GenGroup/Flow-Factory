# Reward Model Guidance

Flow-Factory provides a flexible reward model system that supports both built-in and custom reward models for reinforcement learning.

## Table of Contents

- [Reward Model Types](#reward-model-types)
- [Built-in Reward Models](#built-in-reward-models)
- [VLM-as-Judge](#vlm-as-judge)
  - [Example: Rational Rewards](#example-rational-rewards)
- [Using Built-in Reward Models](#using-built-in-reward-models)
- [Creating Custom Reward Models](#creating-custom-reward-models)
  - [Pointwise Reward Model](#pointwise-reward-model)
  - [Groupwise Reward Model](#groupwise-reward-model)
  - [Class Attributes](#class-attributes)
- [Multi-Reward Training](#multi-reward-training)
- [Decoupling Training and Evaluation Reward Models](#decoupling-training-and-evaluation-reward-models)
- [Async Reward Computation](#async-reward-computation)
- [Remote Reward Server](#remote-reward-server)

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
| `ocr` | Pointwise | Text-rendering accuracy via PP-OCRv5 (rewards correctly rendered text) | [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) |
| `clap` | Pointwise | Audio-text cosine similarity via LAION CLAP (`transformers.ClapModel`); for audio / audio-video models | [CLAP](https://github.com/LAION-AI/CLAP) |
| `imagebind` | Pointwise | Audio-video / text-audio / text-video alignment via Meta ImageBind (CC-BY-NC-SA, NonCommercial) | [ImageBind](https://github.com/facebookresearch/ImageBind) |
| `GenEval` | Pointwise | Compositional T2I evaluation (object count, color, position) via Mask2Former + CLIP | [GenEval](https://github.com/djghosh13/geneval) |
| `geneval2_soft_tifa` | Pointwise | GenEval2 Soft-TIFA: per-atom VQA soft-match via local Qwen3-VL, AM/GM aggregation; `vqa_list` from dataset `metadata` or a `data_path` JSONL. Needs `pip install -e ".[geneval2]"` for exact GM parity | [GenEval2](https://github.com/facebookresearch/GenEval2) |
| `hpsv2` | Pointwise | Human Preference Score v2 (OpenCLIP ViT-H-14 + HPS checkpoint). Install with `uv pip install hpsv2 --no-deps` | [HPSv2](https://github.com/tgxs002/HPSv2) |
| `vllm_evaluate` | Pointwise | VLM with a binary Yes/No question; reward from logprobs via OpenAI-compatible API | [VLM-as-Judge](#vlm-as-judge) |
| `rational_rewards_t2i` | Pointwise | T2I rubric judge (remote VLM); see [VLM-as-Judge](#vlm-as-judge) and [Example: Rational Rewards](#example-rational-rewards) | [Rational Rewards](https://github.com/TIGER-AI-Lab/RationalRewards) |
| `rational_rewards_edit` | Pointwise | Image-edit rubric (source + edited). Same setup family as T2I variant | [Rational Rewards](https://github.com/TIGER-AI-Lab/RationalRewards) |
| `qwen_image_bench` | Pointwise | Qwen-Image-Bench "Q-Judger" (remote VLM); hierarchical 5-dim / 56-facet scoring, faithful per-prompt `dims_en`; see [VLM-as-Judge](#vlm-as-judge) and [Example: Qwen-Image-Bench](#example-qwen-image-bench) | [Qwen-Image-Bench](https://github.com/QwenLM/Qwen-Image-Bench) |

## VLM-as-Judge

**VLM-as-Judge** means using a **vision–language model** to score (or score-and-parse) generated images, usually by calling a **remotely served** model over an **OpenAI-compatible** HTTP API (`/v1/chat/completions` or similar). Training stays in Flow-Factory; the heavy judge typically runs in a separate process, commonly on [vLLM](https://github.com/vllm-project/vllm) ``vllm serve`` or a compatible stack. These call paths are usually **I/O bound**—see [Async Reward Computation](#async-reward-computation) to overlap HTTP latency with sampling.

**Built-in implementations:**

- **`vllm_evaluate`** (registry key: ``vllm_evaluate``) asks a short **Yes/No** question, reads **logprobs** from the completion, and returns a scalar reward. It fits when you want a light judge prompt and no rubric parsing in Python.
- **`rational_rewards_t2i`** and **`rational_rewards_edit`** (keys: ``rational_rewards_t2i``, ``rational_rewards_edit``) send a **long structured rubric** in the user message, parse the assistant reply into per-aspect scores, then aggregate. They follow the same HTTP/OpenAI client pattern; deployment steps for serving the weights are illustrated below under **Example: Rational Rewards**.
- **`qwen_image_bench`** (key: ``qwen_image_bench``) runs the [Qwen-Image-Bench](https://github.com/QwenLM/Qwen-Image-Bench) "Q-Judger". It scores facets on a 3-level hierarchy (5 L1 dimensions / 23 L2 / 56 L3), each rated 0/1/2/N/A and aggregated L3→L2→L1→total (0-100), normalized to [0, 1]. Which L1 dimensions are scored is resolved **per prompt** from a ``dims_en`` checklist (faithful mode) when present, else a configurable fixed ``dimensions`` list. Deployment under **Example: Qwen-Image-Bench**.

### Example: Rational Rewards

``rational_rewards_t2i`` and ``rational_rewards_edit`` call a **remote** vision-language model through an **OpenAI-compatible** HTTP API. The usual deployment is [vLLM](https://github.com/vllm-project/vllm) ``vllm serve``.

1. **Install** the judge stack in an environment that has vLLM (see vLLM docs for CUDA / driver requirements). Training only needs ``pip install openai`` in the Flow-Factory environment.
2. **Start the server** (example wrapper; reward model weights are [TIGER-Lab/RationalRewards-8B-T2I](https://huggingface.co/TIGER-Lab/RationalRewards-8B-T2I) for T2I and [TIGER-Lab/RationalRewards-8B-Edit](https://huggingface.co/TIGER-Lab/RationalRewards-8B-Edit) for image edit):

   ```bash
   # T2I rubric judge (default MODEL_PATH in the script is this repo id)
   export CUDA_VISIBLE_DEVICES=0,1
   export MODEL_PATH="TIGER-Lab/RationalRewards-8B-T2I"
   ./scripts/start_vllm_rational_reward.sh --max-model-len 8192
   # With two GPUs in CUDA_VISIBLE_DEVICES, the script sets --data-parallel-size to 2 unless you override DATA_PARALLEL_SIZE.

   # Image-edit rubric judge (separate process or machine)
   # export CUDA_VISIBLE_DEVICES=2,3
   # export MODEL_PATH="TIGER-Lab/RationalRewards-8B-Edit"
   # ./scripts/start_vllm_rational_reward.sh --max-model-len 8192
   ```

   Override ``PORT``, ``SERVED_MODEL_NAME``, ``TENSOR_PARALLEL_SIZE``, ``DATA_PARALLEL_SIZE``, or ``VLLM_BIN`` via environment variables documented in ``scripts/start_vllm_rational_reward.sh``.

3. **Point training YAML** at the API: set ``api_base_url`` to ``http://<host>:<port>/v1`` (trailing ``/v1`` is required for ``AsyncOpenAI``) and set ``vlm_model`` to the same string as vLLM’s ``--served-model-name``. The start script defaults that to ``RationalRewards-8B-T2I`` when ``MODEL_PATH`` is the T2I checkpoint, and ``RationalRewards-8B-Edit`` when ``MODEL_PATH`` is the edit checkpoint (override with ``SERVED_MODEL_NAME`` if you choose a different id).

**Example NFT LoRA configs** (placeholders ``127.0.0.1:8000`` — change to your judge host):

| Config | Reward | Task |
|--------|--------|------|
| ``examples/nft/lora/qwen_image/rational_rewards_t2i.yaml`` | ``rational_rewards_t2i`` | Qwen-Image T2I |
| ``examples/nft/lora/flux1/rational_rewards_t2i.yaml`` | ``rational_rewards_t2i`` | FLUX.1-dev T2I |
| ``examples/nft/lora/qwen_image_edit_plus/rational_rewards_edit.yaml`` | ``rational_rewards_edit`` | Qwen-Image-Edit-Plus |
| ``examples/nft/lora/flux1_kontext/rational_rewards_edit.yaml`` | ``rational_rewards_edit`` | FLUX.1-Kontext |

Rubric format and project background: [TIGER-AI-Lab/RationalRewards](https://github.com/TIGER-AI-Lab/RationalRewards). Tuning how parsed aspect scores map to the final scalar: adjust ``aggregate_aspect_scores`` in ``src/flow_factory/rewards/rational_rewards_t2i.py`` (shared with edit via ``supported_aspects``) or post-process in the edit module after parsing.

### Example: Qwen-Image-Bench

``qwen_image_bench`` calls the **remote** Qwen-Image-Bench judge ([Qwen/Qwen-Image-Bench](https://huggingface.co/Qwen/Qwen-Image-Bench), a fine-tuned ~27B Qwen3-VL) over an **OpenAI-compatible** API, the same deployment pattern as Rational Rewards.

1. **Install** vLLM in the judge environment; training only needs ``pip install openai``.
2. **Start the server** (wrapper script defaults ``MODEL_PATH=Qwen/Qwen-Image-Bench`` and ``SERVED_MODEL_NAME=Qwen-Image-Bench``). The judge emits a ``<think>…</think>`` section before the JSON scores, so keep ``--max-model-len`` well above (image tokens + ``max_tokens``):

   ```bash
   export CUDA_VISIBLE_DEVICES=0,1,2,3
   ./scripts/start_vllm_qwen_image_bench.sh --max-model-len 32768
   ```

   Override ``PORT``, ``SERVED_MODEL_NAME``, ``TENSOR_PARALLEL_SIZE``, ``GPU_MEMORY_UTILIZATION``, or ``VLLM_BIN`` via environment variables documented in ``scripts/start_vllm_qwen_image_bench.sh``.

3. **Point training YAML** at the API: set ``api_base_url`` to ``http://<host>:<port>/v1`` and ``vlm_model`` to the same string as ``--served-model-name`` (default ``Qwen-Image-Bench``).

**Per-prompt checklists (``dims_en``).** The judge scores only the L1 dimensions a prompt declares. Build the dataset with ``python dataset/qwen_image_bench/prepare.py`` (downloads [Qwen/Qwen-Image-Bench](https://huggingface.co/datasets/Qwen/Qwen-Image-Bench) and writes ``train.jsonl``/``test.jsonl`` carrying ``dims_en``). The ``dims_en`` column flows to the reward via the per-sample ``metadata`` JSON, enabling faithful per-prompt scoring. For datasets **without** ``dims_en``, the reward falls back to the fixed ``dimensions`` config.

**Key config keys** (in addition to ``api_base_url`` / ``api_key`` / ``vlm_model``):

| Key | Default | Description |
|-----|---------|-------------|
| ``call_strategy`` | ``per_dimension`` | ``per_dimension`` issues one judge call per L1 dim (faithful, matches the benchmark); ``single_call`` scores all dims in one call (cheaper, slight deviation). |
| ``dimensions`` | all five L1 dims | Fallback L1 dimensions when a sample has no ``dims_en``. |
| ``score_dimension`` | ``total`` | ``total`` (overall) or a single L1 dim name. |
| ``max_tokens`` | ``4096`` | Generation cap (the judge uses thinking). |

**Cost.** ``per_dimension`` means up to 5 judge calls per image (each with a thinking trace), which dominates RL wall-clock; use ``async_reward: true`` with a high ``num_workers``/``max_concurrent``, trim ``dimensions``, or switch to ``single_call``.

**Implementation note**: Aggregation logic is vendored from upstream under ``src/flow_factory/rewards/qwen_image_bench/`` (``checklists.py``, ``score_utils.py``).

## Using Built-in Reward Models

For **VLM-as-Judge** rewards (e.g. ``rational_rewards_t2i``, ``rational_rewards_edit``, or ``vllm_evaluate``), set ``api_base_url``, ``vlm_model``, and any ``extra_kwargs`` as described in [VLM-as-Judge](#vlm-as-judge). For standard **local** GPU rewards (e.g. PickScore), use the following pattern.

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
        audio: Optional[List[torch.Tensor]] = None,
        condition_images: Optional[List[List[Image.Image]]] = None,
        condition_videos: Optional[List[List[List[Image.Image]]]] = None,
        **kwargs,
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
        audio: Optional[List[torch.Tensor]] = None,
        condition_images: Optional[List[List[Image.Image]]] = None,
        condition_videos: Optional[List[List[List[Image.Image]]]] = None,
        **kwargs,
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
        video: Optional[List[torch.Tensor]] = None, # List of (T, C, H, W) tensors, range in [0, 1]
        audio: Optional[List[torch.Tensor]] = None, # List of (C, T) waveforms
        condition_images: Optional[List[Union[torch.Tensor, List[torch.Tensor]]]] = None, # A batch of condition image list
        condition_videos: Optional[List[Union[torch.Tensor, List[torch.Tensor]]]] = None, # A batch of condition video list
        **kwargs,
    ) -> RewardModelOutput:
        # Stack and process directly on GPU
        rewards = torch.zeros(len(prompt), dtype=torch.float32, device=self.device)
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

### Dataset Metadata Convention

Reward models can declare extra parameters in `__call__` (beyond `prompt`/`image`/`video`) to receive **per-sample metadata** from the dataset. The data flows as:

```
JSONL field → Dataset "metadata" column → sample.extra_kwargs → reward __call__ kwargs
```

**Arrow serialization constraint:** Non-primitive metadata (nested dicts, variable-length lists) **must** be stored as JSON strings in the JSONL. Arrow cannot serialize heterogeneous nested structs — different rows with different sub-fields will crash `Dataset.map()`.

```jsonl
{"prompt": "a red car", "include": "[{\"class\":\"car\",\"count\":1,\"color\":\"red\"}]", "tag": "colors"}
```

All non-preprocess JSONL columns (here `include`, `tag`) are packed into a **single** per-sample `metadata` JSON string and delivered to the reward as `metadata: List[str]` (see `data_utils/dataset.py` `_preprocess_batch` and `BaseTrainer._inject_batch_metadata`). Declare `metadata` in `required_fields` and parse it internally:

```python
class MyMetadataReward(PointwiseRewardModel):
    required_fields = ("image", "prompt", "metadata")  # Receive the packed metadata JSON

    def __call__(self, prompt, image=None, metadata=None, **kwargs):
        for i in range(len(prompt)):
            meta = json.loads(metadata[i]) if isinstance(metadata[i], str) else metadata[i]
            spec = meta.get("include", [])  # original JSONL fields live inside `metadata`
            # ... use spec for evaluation
```

**Rules:**
1. Flat scalars (`str`, `int`, `float`) can be stored directly in JSONL.
2. Complex values (lists, nested dicts) → `json.dumps()` in JSONL; the packed `metadata` string is `json.loads()`-ed in the reward model.
3. Request `metadata` (not individual JSONL columns) in `required_fields`; missing fields raise errors during reward computation.

See `src/flow_factory/rewards/geneval.py` for a complete example (GenEval reads `include`/`exclude`/`tag` from the parsed `metadata`).

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

### Advantage Aggregation

When using multiple rewards, Flow-Factory supports the following aggregation strategies via `advantage_aggregation`:

| Strategy | Description |
|----------|-------------|
| `sum` | Advantage of the weighted sum of rewards |
| `gdpo` | Weighted sum of advantages from each reward |

> To use a customized aggregation algorithm, refer to `advantage/advantage_processor.py`, or override `BaseTrainer.compute_advantages` (`src/flow_factory/trainers/abc.py`).

**Weighted Sum (`sum`):**

Standard approach that *aggregates multiple rewards as a weighted sum* and computes the advantage from this weighted sum:

$$r_{total} = \sum_{i} w_i \cdot r_i$$

where $w_i$ is the `weight` specified for each reward model.

**GDPO (`gdpo`):**

Implements the advantage aggregation from [GDPO: Group Reward-Decoupled Normalization Policy Optimization](https://arxiv.org/abs/2601.05242), which computes *a weighted combination of individual advantages*:

$$A_{total} = \sum_{i} w_i \cdot A_i$$

It then applies *batch normalization*. To use this formula, set:
```yaml
train:
  trainer_type: 'grpo'
  advantage_aggregation: 'gdpo'  # Options: 'sum', 'gdpo'

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
    batch_size: 32
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


## Async Reward Computation

By default, reward computation happens synchronously after all samples are collected. When using IO-bound reward models (e.g., API calls to a remote server), this creates idle time where the training process waits for network responses.

**Async reward** enables reward computation to run concurrently with sampling via a `ThreadPoolExecutor`, reducing wall-clock time. Supported runtime-reward trainers can additionally consume complete reward tiles while unfinished reward requests continue running.

### When to Use

| Scenario | Recommended Setting |
|----------|-------------------|
| Remote API reward (HTTP calls) | `async_reward: true`, `num_workers: 4+` |
| Local GPU reward model | `async_reward: false` (default) |
| Remote reward server | `async_reward: true`, `num_workers: 4+` |

### Configuration

Enable per reward model in your YAML config:

```yaml
rewards:
  # Async: API-based reward with concurrent requests
  - name: "remote_aesthetic"
    reward_model: "flow_factory.rewards.my_reward_remote.RemotePointwiseRewardModel"
    server_url: "http://localhost:8000"
    batch_size: 16
    async_reward: true    # enable async computation
    num_workers: 4        # 4 concurrent API requests

  # Sync: local GPU reward (default, no benefit from async)
  - name: "pick_score"
    reward_model: "PickScore"
    batch_size: 16
    # async_reward defaults to false
```

> See [`examples/grpo/lora/sd3_5/nocfg.yaml`](../examples/grpo/lora/sd3_5/nocfg.yaml) for a complete training config.

### Per-Model Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `async_reward` | `bool` | `false` | Compute this reward asynchronously during sampling |
| `num_workers` | `int` | `1` | Number of concurrent workers. Set >1 for IO-bound models |

Training-level overlap controls live under `train`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reward_optimization_overlap` | `bool` | `false` | Consume complete async reward tiles during optimization |
| `reward_optimization_overlap_mode` | `ordered \| ready` | `ordered` | Preserve rollout order or bypass globally unready tiles |
| `reward_optimization_overlap_poll_interval` | `float` | `0.05` | Seconds between distributed readiness polls |

### How It Works

Async and sync reward models can coexist. The `RewardBuffer` partitions models automatically:

```
Sampling Loop (main thread):
  [Sample batch 0] → buffer.add_samples()  → [Sample batch 1] → ...
                          ↓
                   ThreadPoolExecutor dispatches ready async tasks
                   (non-blocking, returns immediately)

finalize():
  1. Compute sync rewards on main thread (standard path)
  2. Collect async results from completed futures
  3. Merge all rewards
```

With `train.reward_optimization_overlap: true`, `finalize()` is replaced by a cycle-aware
`seal → poll → resolve tile → finish` sequence. All training rewards must be async. Evaluation
continues to use the ordinary full-buffer path. Pointwise request tails are flushed at optimizer
tile boundaries, so one slow request cannot couple readiness of adjacent tiles even when
`reward.batch_size` does not divide the tile size.

All generation trainers that consume runtime group-relative rewards support this path: GRPO,
GRPO-Guard, DPPO, DiffusionNFT, AWM, CRD, DGPO, online DPO, and TDM-R1. DGPO and a
`group_distributed` TDM-R1 run support async **pointwise** rewards across rank-sharded groups.
Async groupwise rewards still require `group_contiguous`, because the reward implementation must
receive all K members on one process.

For IO-bound models with `num_workers > 1`, multiple API requests execute truly in parallel (Python's GIL is released during network IO):

```
num_workers=1 (serial):     [API call 500ms] [API call 500ms] [API call 500ms]  → 1500ms
num_workers=4 (concurrent): [API call 500ms]                                    → ~500ms
                            [API call 500ms]
                            [API call 500ms]
```

### Notes

- **Groupwise async rewards** require the `GroupContiguousSampler`, which ensures all samples of a group land on the same rank. Cross-rank overlap accepts pointwise rewards only.
- **Independent lanes**: each async reward owns a separate executor sized by its `num_workers`, so a slow component cannot occupy another component's client threads. A tile still waits for every applicable reward component.
- **`num_workers`** only affects async models. Sync models always compute on the main thread.
- **Error handling**: exceptions from worker threads are automatically re-raised on the main thread.
- **Remote output device**: set `device: cpu` for HTTP rewards so response tensors do not allocate on the training GPU from worker threads. Reward/optimization overlap requires this setting; local GPU rewards keep the ordinary phase boundary.
- **Protocol compatibility**: `RemotePointwiseRewardModel` expects `GET /health` and `POST /compute` returning `{"rewards": [...]}`. OpenAI-compatible or RewardService `/score` endpoints need a thin custom reward adapter or gateway; asynchronous scheduling does not translate wire protocols.

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
python example_server.py --port 8000
```

### Training Config

Combine with `async_reward` and `num_workers` for best performance (see [Async Reward Computation](#async-reward-computation)):

```yaml
rewards:
  - name: "remote_reward_1"
    reward_model: "flow_factory.rewards.my_reward_remote.RemotePointwiseRewardModel"
    server_url: "http://localhost:8000"
    batch_size: 16
    async_reward: true     # compute during sampling, not after
    num_workers: 4         # concurrent API requests
    timeout: 60.0          # optional, default 60s
    retry_attempts: 3      # optional, default 3
  - name: "remote_reward_2"
    reward_model: "flow_factory.rewards.my_reward_remote.RemotePointwiseRewardModel"
    server_url: "http://localhost:8001" # Use different ports if your have multiple reward servers
    batch_size: 16
    async_reward: true
    num_workers: 4
    timeout: 60.0          # optional, default 60s
    retry_attempts: 3      # optional, default 3
```

For groupwise rewards, use `RemoteGroupwiseRewardModel`:
```yaml
rewards:
  - name: "remote_ranking"
    reward_model: "flow_factory.rewards.my_reward_remote.RemoteGroupwiseRewardModel"
    server_url: "http://localhost:8000"
    timeout: 60.0        # optional, default 60s
    retry_attempts: 3    # optional, default 3
```

### Server Dependencies

The reward server runs in an **isolated environment**, separate from Flow-Factory.

```bash
# Create isolated environment
conda create -n reward_server_env
conda activate reward_server_env

# Install server framework
pip install fastapi uvicorn pillow

# Install your reward model's dependencies
pip install paddlepaddle paddleocr  # example
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
| VLM-based reward, same Python env as Flow-Factory | Prefer built-in [VLM-as-Judge](#vlm-as-judge) models (e.g. ``vllm_evaluate``, Rational Rewards) or a thin custom `PointwiseRewardModel` that calls vLLM/SGLang via the OpenAI SDK. |
| VLM-based reward, **incompatible** dependencies | Use this **Remote Reward Server** pattern (isolated process), or run the judge on another host and call it from a custom reward that matches that contract. |
| Closed-source API | Use `requests`, OpenAI SDK and official SDK directly in `__call__()` |
| Compatible dependencies | Implement as standard `PointwiseRewardModel` |
