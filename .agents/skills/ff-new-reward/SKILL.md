---
name: ff-new-reward
description: "Add a Flow-Factory reward model with pointwise/groupwise dispatch, media conversion, per-dataset routing, async execution, backend-safe loading, registry, configuration, and verification."
---

# New Reward Model Integration

Read `../../../guidance/rewards.md`, `../../knowledge/constraints.md` #13, and the current
`rewards/abc.py`, `rewards/reward_processor.py`, and `rewards/loader.py` contracts.

## 1. Choose the Dispatch Contract

- **Pointwise**: each input is scored independently. A call receives a non-empty chunk whose length
  is at most configured `batch_size`; tail chunks and per-dataset applicability gating can make it
  smaller.
- **Groupwise**: one call receives a complete sampler-assigned canonical
  `(source_id, unique_id)` group, either
  local or reconstructed across ranks. Configured pointwise `batch_size` does not define this
  call.

Return exactly one finite score per input passed to the call, in the same order. Do not return NaN
as a “not applicable” marker: `RewardProcessor` owns applicability masks, NaN padding outside model
calls, and `sample.applicable_rewards`.

Declare only fields needed from `BaseSample` in `required_fields`. Common fields include `prompt`,
`image`, `video`, `audio`, `condition_images`, and `condition_videos`. Additional JSONL metadata is
available as one JSON-encoded `metadata` string and must be parsed explicitly.

`required_fields` selects reward-consumer data; it does not replace a concrete sample class's
`reconstruction_required_fields`. If adding a sample field that its constructor or `__post_init__`
needs after a partial gather, update that inherited class contract separately. Collator
`_shared_fields` is a third, independent concern.

Set `use_tensor_inputs` deliberately:

- `False`: images/video frames arrive as PIL and audio as NumPy;
- `True`: media arrives as tensors.

Condition media retains nested per-sample item structure.

## 2. Implement the Model

```python
from typing import Any, List, Optional

import torch
from accelerate import Accelerator
from PIL import Image

from ..hparams import RewardArguments
from .abc import PointwiseRewardModel, RewardModelOutput


class MyRewardModel(PointwiseRewardModel):
    """Score prompt-image alignment for each generated sample."""

    required_fields = ("prompt", "image")
    use_tensor_inputs = False

    def __init__(self, config: RewardArguments, accelerator: Accelerator) -> None:
        """Load the reward network using the configured device and dtype."""
        super().__init__(config, accelerator)
        ...

    @torch.no_grad()
    def __call__(
        self,
        prompt: List[str],
        image: Optional[List[Image.Image]] = None,
        **kwargs: Any,
    ) -> RewardModelOutput:
        """Return one finite score per received prompt-media pair."""
        scores = ...
        return RewardModelOutput(rewards=scores)
```

Use `self.device` and `self.dtype`; do not hardcode CUDA. Keep public signatures typed and
Google-style. `@torch.no_grad()` is required for inference-only scoring.

Construction runs inside `ModelLoadCoordinator`'s REWARD load scope and reward resources remain
full per-rank replicas. A reward implementation must not call `accelerator.prepare()`, enter
target-only FSDP loading state, or mutate trainer component ownership.

## 3. Register and Configure

Add a lowercase canonical lazy path to `_REWARD_MODEL_REGISTRY`. Preserve direct Python-path
fallback; a decorator is optional and does not replace the static entry.

```yaml
rewards:
  - name: my_reward
    reward_model: my_reward
    model_path: org/model-name
    dtype: bfloat16
    device: cuda
    batch_size: 16
    applicable_datasets: [alignment]
    weight: 1.0
    async_reward: false
    num_workers: 1
```

`name` identifies the configured reward stream; `reward_model` resolves its implementation.
`applicable_datasets` is resolved to dataset names/source IDs, and `weight` may be one scalar or a
per-dataset mapping. Training and `eval_rewards:` are independent configurations. Runtime training
rewards are valid only for an execution contract with `feedback=runtime_reward`; reward-free and
offline trainers may still use evaluation rewards.

The loader deduplicates train/eval entries with the same model identity. Do not keep mutable
per-config-name state inside a shared model call.

When `async_reward` is enabled, calls run in worker threads and may use dedicated CUDA streams.
Implementations must be thread-safe or require `num_workers: 1`; tail pointwise batches still run
during finalize.

## 4. Verification

- Direct pointwise calls at full and tail lengths, or one complete groupwise call.
- Exact result shape/order and rejection of NaN/Inf.
- Source-gated partial and no-applicable subsets through `RewardProcessor`.
- Groupwise local and distributed reconstruction when applicable.
- Selective groupwise gathers retain every concrete sample `reconstruction_required_fields` entry.
- PIL/NumPy and tensor media conversion for every declared field.
- Scalar and per-dataset weighted multi-reward aggregation.
- Sync and async execution, including async tail flush and thread-safety assumptions.
- Train/eval deduplication and exactly one REWARD load scope per unique model.
- Registry lookup, direct-path fallback, device placement, and config parsing.
- Run `/ff-review` before commit.

## Common Failures

- Assuming every pointwise call has exactly `config.batch_size` inputs.
- Returning a raw scalar, wrong ordering, non-finite values, or framework-owned applicability NaNs.
- Omitting a required field or flattening nested condition media.
- Hardcoding a device or preparing/sharding a reward as a target component.
- Mutating shared model state by reward configuration name.
- Using a pointwise base for a group-dependent score, or enabling async workers for non-thread-safe
  code.
