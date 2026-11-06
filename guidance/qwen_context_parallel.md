# Experimental Qwen-Image context parallelism

`qwen-image` declares the native Diffusers masked joint image/text layout. Diffusers owns token
splitting, rotary-position slicing, masked Ulysses attention and the complete-image output gather.
Flow-Factory owns input validation, mask-preserving text padding, logical sample identity,
full-output backward normalization, model preparation and state lifecycle.

This is the ordinary text-to-image layout: no guidance embeddings, per-token conditional times,
additional timestep conditions or layer-3D rotary positions. `qwen-image-edit-plus` and
`qwen-image-2.1` remain separate capabilities. Qwen TP+CP composition is not admitted in this
slice; its TP/SP sequence boundaries still need their own composition proof.

## Configuration boundary

Qualification uses PyTorch 2.10.0, Accelerate 1.15.0 and Diffusers 0.41.0, FSDP2,
BF16, AdamW, `target_components: [transformer]`, `per_device_batch_size: 1`,
`dp_replicate_size: 1`, `dp_shard_size >= 2`, and one shared native attention backend. Disable
CPU-RAM-efficient loading in the FSDP2 launcher. The following fields are a delta for an existing
Qwen training configuration; paths, data and rewards remain workload-specific:

```yaml
num_processes: 4
mixed_precision: bf16
parallelism:
  dp_shard_size: 2
  tp_size: 1
  cp_size: 2
  sequence_parallel: false
model:
  model_type: qwen-image
  target_components: [transformer]
train:
  trainer_type: grpo-guard
  per_device_batch_size: 1
  guidance_scale: 2.0
  shuffle_samples: false
  reward_optimization_overlap: true
  reward_optimization_overlap_mode: ordered
acceleration:
  shared:
    - name: attention_backend
      params: {backend: native}
```

Attention heads and image token counts must divide by the CP degree. The adapter validates
`patch_size=2`, matching packed input/output channels, a single positive image geometry and native
`QwenDoubleStreamAttnProcessor2_0` before installing collective hooks. Special attention kwargs,
IP adapters and unsupported transformer metadata fail explicitly.

Text lengths may differ across positive/negative conditions and need not divide by CP. The adapter
preserves every position through the last live mask entry, pads both conditions to a common CP
multiple with zero mask entries and keeps the actual token positions unchanged. Native Ulysses
exchanges the local masks in the same order as image/text keys. The extra text tokens do not enter
the image loss. Rollout and replay use the identical preparation, including active CFG and its
norm rescaling.

## Acceptance and composition

The native acceptance case combines Qwen with full-weight Guard, ordered asynchronous reward
work, reference/EMA state, evaluation and drained fresh-process continuation on DP2 x CP2.
Its fixed-operand controls cover full and LoRA modes in FP32 and BF16 at CP1 and CP2, with holes
and unequal valid text lengths in CFG-shaped masks. They check actual outputs, every reachable
gradient, clipping, Adam moments and parameter updates under the existing shared bounds. Missing
CP normalization and zero/half/double/sign-corrupted gradients must fail those controls.

The final block's text-output projection and text MLP cannot influence the image output. Their
parameters remain part of the normal model and optimizer ownership. The fixture declares exactly
those inactive parameters before execution; their gradients must remain absent and their values
and optimizer state must remain unchanged. Any other missing gradient or inactive-state mutation
fails. No production parameter is frozen to accommodate a test.

The objective is the existing Guard implementation, checked by its deterministic CPU oracle and
actual native observations. Preparation, full-output backward, rewards, snapshots and checkpoint
ownership remain shared. Reusing those owners does not imply that every Qwen algorithm endpoint,
editing layout, pretrained-quality result or TP+CP setting has been executed. Exact evidence and
source identity belong in the implementation PR; the roadmap tracks the remaining closure work.

Tests: `test_qwen_context_parallel_contracts.py`, the shared model-control and optimizer-observer
contracts, and `test_qwen_context_parallel.py::test_cuda_qwen_guard_ordered_full` under
`tests/distributed/`. See [general CP behavior](context_parallel.md) and
[validation selection](automatic-validation.md).
