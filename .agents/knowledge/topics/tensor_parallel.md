# Tensor parallelism: owners and diagnostics

**Read when** changing TP/SP plans, composing TP with FSDP/CP, or investigating distributed
numerical/state errors. This page maps the implementation and invariants; current model regions
are in [model coverage](../../../guidance/parallelism/models.md), and measured histories are in
[the investigation archive](../../../docs/reference/parallelism/tensor-parallel-history.md).

## Authoritative owners

| Concern | Owner / next reference |
|---|---|
| Physical geometry and logical samples | `contracts/topology.py::ParallelTopology`, `utils/dist.py`; [topology](../../../docs/reference/parallelism/topology.md) |
| TP/SP declarations and defaults | Concrete adapters' `tensor_parallel_plan()`, `sequence_parallel_plan()` and `sequence_parallel_by_default`; resolve from `models/registry.py` |
| Styles, packed layouts, LoRA derivation and total plans | `models/tensor_parallel.py`; [TP/SP reference](../../../docs/reference/parallelism/tensor-parallel.md) |
| Materialization and preparation order | `trainers/common/model_preparation.py`, `loading/`; [component runtime](component_runtime.md) |
| Backend, optimizer and CP admission | `trainers/multirole/backend.py`, `trainers/common/context_parallel.py`, concrete model/trainer capabilities |
| CP/SP hook composition | Model family helpers and `models/context_parallel.py`; [CP reference](../../../docs/reference/parallelism/context-parallel.md) |
| Prepared parameter/optimizer ownership | `ModelBundle`, routed component proxies, variant registry; [variants](component_variants.md) |
| Exact-state identity | `trainers/common/runtime_identity.py`, prepared DTensor parameter layouts; [parameter state](../../../docs/development/parameter-state.md) |
| Evidence selection | [Selected validation](gpu_validation.md); tests and artifacts must match the changed owner and final source |

Resolve current registries instead of copying a model/algorithm admission list into this topic.
The user guide separates functional support, canonical parameter counts, SP regions and measured
performance. Those quantities must not be conflated.

## Sample and mesh invariants

- TP × CP peers process one logical sample: same identities, conditions, noise and time. Logical
  data replicas are `world / (TP × CP)`. SP reuses the TP group and adds no dimension.
- Sample statistics, reward groups, RNG and evaluation use the logical-DP scope, not the physical
  world or FSDP's parameter-shard scope. FSDP can flatten DP-shard and CP without creating samples.
- Read the actual flattened `dp` mesh through runtime helpers. It need not appear in root
  `mesh_dim_names`; an all-ones launcher plan must preserve ordinary DP behavior.
- `ParallelTopology` owns rank order `(dp_replicate, dp_shard, cp, tp)` and product validation.
  Adapter/backend gates decide which geometrically valid layouts are admitted.
- `verify_replicated_inputs` is a diagnostic bitwise check of flattened tensor values/counts.
  It is not a complete schema check for non-tensors or every leaf's shape and carries collective cost.

## Parameter and activation invariants

1. **Keep the plan total.** `build_total_plan` supplies replicated ownership for all remaining
   parameters, including bare root parameters. Mixed plain/DTensor ownership can break clipping
   even when an isolated forward appears correct.
2. **Retain trainability and canonical identity.** Torch partition styles can replace parameters;
   `apply_tensor_parallel` preserves `requires_grad`. Compare gradient presence as well as values,
   especially frozen LoRA base layers. Prepared registries and optimizers must reference the live
   replacement parameters.
3. **Keep logical packed weights canonical.** Packed column/row styles exchange activations around
   ordinary two-dimensional weight shards. Directly treating a contiguous packed shard as local
   Q/K/V or gate/up changes arithmetic. LoRA factors use the same logical packing.
4. **Assign partial gradients once.** Per-head norms on head shards need
   `ShardedInputReplicatedWeight`; affine weights on token shards need
   `SequenceBroadcastParameter`. Replicated conditioning consumed by token shards can need
   `SequenceBroadcastOutput`. Whole conditioning streams and token-indexed timesteps have different
   ownership; summing an already complete gradient multiplies it by TP.
5. **A TP weight shard does not imply SP token work.** Attention, MLP, text/image streams, prefixes
   and output heads have distinct axes and masks. Gather inputs before a TP contraction requires
   aligned tokens; retain shards only across model-owned residual regions.
6. **Validate token geometry before collectives.** Equal local lengths are required at the declared
   gather/scatter boundaries. Padding is valid only with the family's matching attention masks,
   rotary/index layout, crop and derivative rules. Native attention may ignore a supplied mask.
7. **Preserve complete PEFT branches.** Plain replicated SP parameter styles can derive LoRA
   ownership. Conditional/composed styles are not automatically supported. Activation hooks that
   must see base plus LoRA output belong around the complete wrapper, after both branches merge.

## Preparation, checkpoint and boundary order

`ModelPreparation` applies TP to canonical roots and variants before backend preparation, then
rebinds ownership to the prepared model. Frozen generation roots still need the declared layout.
Do not prepare a component outside its bundle or install duplicate hooks from a trainer.

For a composed token stream, CP splits precede SP splits and SP restoration precedes CP
restoration. Native forward wrappers can execute inside Torch hooks, so installation order alone
is not proof of execution order. Family composition helpers validate and retain the moved handles.
Observe real multi-block residuals and positional/keyword checkpoint calls in tests.

Checkpoint replay must retain the mask, valid length, roles and index metadata for its own forward,
including when later pending forwards have different lengths. Keep the model/backend checkpoint
owner singular. Model-specific in-forward policies are not generic recipes for another family.

Model-only export keeps canonical names, shapes and ordering. Exact continuation additionally
binds prepared TP parameter mesh dimensions, shape and placements in runtime identity; equal
world size and global parameter shape cannot distinguish changed sharding. Fail preflight before
mutating optimizer/state. Non-TP identities retain their existing schema. Fixed topology alone
is not permission to resume under a changed parameter plan.

## Proportionate evidence

Compare at fixed logical sample/update geometry and inspect every trainable tensor. At minimum,
check output values, input and parameter derivatives, gradient presence, frozen/inactive state,
optimizer updates/moments, canonical state and the affected checkpoint path. Observe both sides
of each new activation boundary; output parity alone cannot establish retained SP coverage.

Use native full/LoRA fixtures and checkpoint replay where those paths change. Mask, CP-loss-factor,
packing or partial-gradient negative controls should falsify the specific ownership assumption.
A CPU TP primitive does not prove FSDP composition; select the smallest actual backend mesh that
exercises the changed contract. Re-run the selected final-source gate after integration.

FP32 partitioned contractions change accumulation order. Diagnose against the same architecture's
native arithmetic and precision behavior; an FP64 control can separate layout errors from a
native FP32 norm. Do not silently widen a tolerance or replace a failed contract. Preserve the
failed result and explicitly justify any separately reviewed numerical oracle.

## Diagnostic index

| Symptom | First owner/check | Retained investigation |
|---|---|---|
| Correct loss, wrong tiny parameter gradients | Head/token partial-gradient style; compare every tensor | [Plan ownership](../../../docs/reference/parallelism/tensor-parallel-history.md#the-plan), [silent failures](../../../docs/reference/parallelism/tensor-parallel-history.md#two-silent-failures) |
| LoRA gains unexpected gradients | Partition-time trainability and derived base/A/B plan | [Verification](../../../docs/reference/parallelism/tensor-parallel-history.md#verifying-a-new-plan) |
| Mixed Tensor/DTensor clipping error | Total plan, root parameters and live optimizer references | [Plan completion](../../../docs/reference/parallelism/tensor-parallel-history.md#the-plan) |
| Hang after changing prompt/resolution | Equal local token lengths, masks and collective order | [Token divisibility](../../../docs/reference/parallelism/tensor-parallel-history.md#one-loud-limitation-the-token-count-must-divide-tp_size) |
| Correct TP1 but wrong TP+CP | CP-before-SP order, packed rotary order and CP gradient ownership | [CP reference](../../../docs/reference/parallelism/context-parallel.md) |
| OOM before first forward | Materialization/TP-before-FSDP residency and retained original storage | [Historical H3 failure](../../../docs/reference/parallelism/tensor-parallel-history.md#historical-low-coverage-h3-experiment), [current H3 evidence](../../../docs/reference/parallelism/models/minimax-h3.md) |
| OOM grows by acquisition | Retained samples/trajectory tensors and component-owned values | [Sample retention](../../../docs/reference/parallelism/tensor-parallel-history.md#per-epoch-sample-retention-which-tensor-parallelism-amplifies) |
| Wrong sample/reward count or duplicated evaluation | Logical-DP geometry, subgroup and singleton bypass | [Sample identity](../../../docs/reference/parallelism/tensor-parallel-history.md#sample-identity-is-the-precondition) |
| Checkpoint replay diverges | Per-call metadata, one checkpoint owner and FSDP live parameters | [Checkpoint history](../../../docs/reference/parallelism/tensor-parallel-history.md#checkpointing) |
| Tiny native probe passes, real workload fails | Actual cache/conditioning/precision and same-state oracle scope | [Qwen2.1 retained contracts](../../../docs/reference/parallelism/tensor-parallel.md#qwen-image-21-execution-success-and-numerical-qualification) |

Historical benchmarks do not establish current-plan speed or capacity. Read their source,
workload, failed attempts and qualification bounds before drawing a conclusion.

## Cross-refs

- UP: [constraints #18b](../constraints.md#18b-tensor-parallelism-scopes-samples-not-ranks),
  [#19](../constraints.md#19-fsdp-cpu-efficient-loading),
  [#26](../constraints.md#26-fail-fast-error-handling)
- PEER: [Component runtime](component_runtime.md), [Variants](component_variants.md),
  [Samplers](samplers.md), [MiniMax H3](minimax_h3.md)
- DETAIL: [Current TP/SP reference](../../../docs/reference/parallelism/tensor-parallel.md),
  [Historical investigations](../../../docs/reference/parallelism/tensor-parallel-history.md)
