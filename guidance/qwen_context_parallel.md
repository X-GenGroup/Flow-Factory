# Experimental Qwen-Image context parallelism

`qwen-image` declares the native Diffusers masked joint image/text layout. Diffusers owns token
splitting, rotary-position slicing, masked Ulysses attention and the complete-image output gather.
Flow-Factory owns input validation, mask-preserving text padding, logical sample identity,
full-output backward normalization, model preparation and state lifecycle.

This is the ordinary text-to-image layout: no guidance embeddings, per-token conditional times,
additional timestep conditions or layer-3D rotary positions. `qwen-image-edit-plus` and
`qwen-image-2.1` remain separate capabilities. The ordinary layout admits TP+CP with explicit
image-only SP. SP remains disabled by default: set `sequence_parallel: true` to scatter image
activations while keeping text activations replicated over TP.

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

Attention heads must be divisible by `TP * CP`. Image tokens must be divisible by CP, or by `TP * CP`
when image SP is enabled. Text tokens need only divide by CP; local text length may be indivisible
by TP. For the qualified eight-rank composition, change `num_processes` to 8, `tp_size` to 2 and
`sequence_parallel` to true in the example. The adapter validates
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

Qwen's model-owned composition invokes the installed native CP image splitter at `img_in` before
SP scatters its output. Native text splitting remains at the first transformer block. Per-block
native mask/modulation-index splitting runs before SP argument hooks, including positional
checkpoint recomputation. The native output gather runs after SP gathers at `proj_out`. Native
rotary hooks, masked joint attention, slicing autograd and the shared CP loss factor retain their
existing owners. All moved hook metadata is validated before any installed callback changes.

The TP/SP witness compares DP2 x CP2 x TP2 full and checkpointed LoRA training, with SP on and
off, to an independent unsharded FP32 reference. Its CFG-shaped operands use 16 image tokens and
six masked text positions: CP-local text length three remains replicated over TP2. It checks all
active gradients, exactly the final unused text branches, three Adam updates/moments and the
actual omitted-CP-factor negative control. The TP2/SP Guard lifecycle covers checkpointed BF16
training, frozen reference/EMA, ordered overlap, evaluation and fresh-process continuation.
Its separate full/LoRA fixed-operand controls compare TP2 CP1/CP2 with SP disabled; the unsharded
probe owns the additional SP numerical proof.

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
editing layout, pretrained-quality result or arbitrary TP+CP setting has been executed. Exact evidence and
source identity belong in the implementation PR; the roadmap tracks the remaining closure work.

Changing Qwen's declared `tensor_parallel` capability from false to true changes the backend
identity stored in existing CP-enabled checkpoints, including CP>1/TP1. Old CP checkpoints do
not satisfy exact-resume identity after this change. CP1 identity is unchanged. Continuation
qualification uses the same final source, topology and enabled capability; no checkpoint identity
migration or relaxed comparison is provided.

Tests: `test_qwen_context_parallel_contracts.py`, `test_qwen_context_parallel_boundaries.py`,
`test_qwen_tp_context_parallel.py::test_cuda_qwen_tp_sp_cp_reference`, the shared model-control and optimizer-observer
contracts, and `test_qwen_context_parallel.py::test_cuda_qwen_guard_ordered_full` plus
`test_qwen_context_parallel.py::test_cuda_qwen_tp_sp_cp_guard_ordered_full` under
`tests/distributed/`. See [general CP behavior](context_parallel.md) and
[validation selection](automatic-validation.md).


## TP parameter coverage

The ordinary Qwen-Image and Edit-Plus adapters share one transformer plan. AdaLN
`img_mod.1`, `txt_mod.1` and `norm_out.linear` now shard **input columns**, following
H3's replicated-input row-parallel pattern. Output all-reduces preserve the native
shift/scale/gate order; no packed weight permutation or model-checkpoint conversion
is introduced. Under image SP, modulation output gradients are summed over token
shards before the row-parallel backward reconstructs the full time-embedding gradient.

For the native 60-layer, 24-head × 128, 3584-condition-width configuration, the
canonical transformer contains **20,430,401,088** parameters. TP covers
**20,405,772,288 (99.8795%)**, up from **13,592,125,440 (66.5289%)**. The denominator
includes all transformer weights and biases, including inactive final text branches;
it excludes text encoders, VAE and added LoRA parameters. Row-parallel biases,
normalization weights, boundary embeddings and `proj_out` remain replicated. These
are parameter-placement counts, not measured activation or peak-memory savings.
`tests/models/test_qwen_adaln_coverage.py` reproduces the counts on native meta modules
and verifies that Edit-Plus delegates to the same plan. Qwen-Image 2.1 has a separate
architecture and plan; these counts do not describe it.

The incremental acceptance uses one two-GPU TP2/DP1 component-level supervised
velocity update with image SP and checkpointing, for ordinary and `zero_cond_t`
Edit-2511 modulation. It checks dense output, every gradient, update and inactive
parameter against identical operands. This isolates the changed placement; the
previous CP/trainer lifecycle evidence remains recorded at its original commits.
