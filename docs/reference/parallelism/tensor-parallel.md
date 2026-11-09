# Tensor and sequence parallelism

Tensor parallelism (TP) divides one denoiser across ranks. Sequence parallelism (SP) is an
adapter-specific overlay on that TP group: it divides selected token activations across the
same ranks. It does not introduce an additional independent mesh dimension. The supported
training backend for TP is FSDP2; DDP and DeepSpeed ZeRO-2 remain supported with TP disabled.

```yaml
parallelism:
  tp_size: 2
  sequence_parallel: true
```

On a four-rank launch, the omitted `dp_shard_size` resolves to two. Each pair of TP peers
receives the same sample; the two FSDP replicas receive different samples. Larger launches
derive a larger shard degree. Set `dp_shard_size` explicitly only to require a fixed product;
an inconsistent launch is an error. See [configuration rules](topology.md#configuration-and-runtime).
Compare layouts at fixed **logical replica count and
sample/update geometry**, rather than fixing physical rank count and inadvertently changing the
batch. A two-rank primitive test does not qualify four-rank FSDP2 plus TP training.

Omitting `parallelism` preserves the existing backend choice. With TP enabled, omitted
`sequence_parallel` follows the adapter default below. `false` selects TP alone. `true` requires
an explicit adapter SP plan and rejects an adapter without one before loading model weights.
Invalid head divisibility, unsupported backend/optimizer pairings and incompatible runtime
settings also fail through the normal preflight checks. Checkpoint-specific limits below still
apply even when the forward layout is supported.

Implementation ownership and contribution rules: [shared model parallelism](../../development/model-parallelism.md).

## Canonical packed projections

`PackedColwiseParallel(block_sizes)` and `PackedRowwiseParallel(block_sizes)` retain ordinary
2-D `Shard(0)` and `Shard(1)` weights. The styles exchange activations between contiguous linear
shards and each rank's slice of every logical block. For example, `(1, 1, 1, 4, 4)` describes
query/key/value plus two MLP streams; `(1, 4)` describes their attention/MLP output concatenation.
Each packed block width must be divisible by TP. Parameter names, global ordering, PEFT factors and native
model/optimizer checkpoint layouts stay canonical. Omitting `block_sizes` requires native
`_tp_packed_col_blocks` or `_tp_packed_row_blocks` metadata whose widths sum exactly to the
projection width. LoRA derives concrete immutable block widths from its base layer.

Each packed projection adds an activation all-to-all in forward and its inverse in backward.
LoRA applies the exchange to both affected branches. This is a correctness capability with an
explicit communication cost; no speedup is claimed. Packed column inputs are replicated. An SP
plan gathers at its model-owned entry boundary and may set a packed row output to `Shard(1)`.
Native adapters must declare their own packing and attention geometry before using these styles;
the helper alone does not enable a new model plan.

The selected `packed-tp-layout-contracts`, `packed-tp-native-contracts` and
`packed-tp-composed-state` cases check TP2/TP4, full/LoRA, checkpointed SP, all nonzero parameter
gradients, three Adam updates and moments, ordinary model export, and fresh-process distributed
checkpoint continuation. `packed-tp-cuda-state` adds native four-rank FSDP2/NCCL and exact BF16
transport, with FP64 arithmetic controls. The wrong-layout control omits the exchanges and must
fail numerically. These synthetic packed blocks do not qualify a model family or training recipe.

## Capability and qualification table

Current registry-wide admission, defaults and activation regions are maintained in the
[human model guide](../../../guidance/parallelism/models.md); each row links its native technical
contract. `tests/models/test_parallelism_capabilities.py` checks default/explicit plan resolution.
The [algorithm guide](../../../guidance/parallelism/algorithms.md) separates concrete trainer
admission from measured model/algorithm recipes. These are not interchangeable evidence claims.

Q0 means bounded native structures on CPU/Gloo, Q2 actual CUDA backend preparation and updates,
and Q3 the named pretrained weights; see [validation](../../development/validation/validation.md).
The [model closure ledger](../../../roadmap/handoffs/v0.5-model-closure.md) records tested sources
and outstanding obligations. Earlier TP-count/AdaLN increments used
[explicit incremental acceptance](../../development/validation/automatic-validation.md#explicit-incremental-acceptance);
those receipts retain their original scope. Expanded SP regions require the final selected
source-bound cases, including their actual prepared composition. A case definition is not a pass.

## SP ownership and state

SP plans keep token-local residuals where the native model layout allows it. Flux1 spans both
streams and the single stack; Flux2/3 retain shards within each stack and gather at native
packing boundaries. Z-Image and H3 refiners have independent token regions. SD3.5 and ordinary
Qwen preserve odd-text masks while sharding both residual streams. Output boundaries differ for
pointwise projections, packed role norms, cached prefixes and spatial heads. The
[activation table](../../../guidance/parallelism/models.md#sequence-parallel-coverage) is the current
inventory; no parameter-placement percentage expresses this activation coverage.

Replicated weights operating on token shards use partial-gradient ownership. The shared LoRA
mapper supports the exact plain `SequenceBroadcastParameter` style without exclusions; it does
not infer legality for conditional/composed styles. A family can install activation transfers
around a complete PEFT wrapper while leaving the parameter style independently derivable.
Qwen2.1's conditional final projection remains outside custom LoRA target admission.

Canonical weights and optimizer names remain native. Exact continuation additionally records
realized prepared TP DTensor mesh dimensions, shape and placements; changed parameter placement
is rejected even if TP degree and global tensor shapes match. See
[topology/state identity](topology.md#data-feedback-and-checkpoint-boundaries).

## Canonical parameter coverage

These native meta-model counts include every base weight and bias in the stated component,
including inactive/frozen parameters, and exclude added LoRA factors, buffers, optimizer state
and activations. They measure placement, not memory or throughput. Row-parallel biases and
normalization weights remain replicated.

| Model/plan | Sharded / total parameter elements | Coverage | Count definition |
|---|---:|---:|---|
| Flux1 / Kontext | 11,854,737,408 / 11,901,408,320 | 99.607854% | [Whole guidance transformer](models/flux.md#tensor-parallel-composition), both stacks plus AdaLN |
| Qwen Image / Edit-Plus | 20,405,772,288 / 20,430,401,088 | 99.879450% | [Whole shared transformer](models/qwen.md#tp-parameter-coverage); not Qwen-Image-2.1 |
| SD3.5 Medium | 2,227,745,280 / 2,243,171,520 | 99.312302% | [Whole Medium transformer](models/sd35.md) |
| SD3.5 Large / Large-Turbo | 8,027,428,864 / 8,056,627,520 | 99.637582% | [Whole Large transformer](models/sd35.md) |
| Bagel, SP off | 11,406,409,728 / 14,141,252,608 | 80.660533% | [Main language component](models/bagel.md#parameter-coverage-of-the-two-plans), including both MoT branches and untied vocabulary matrices |
| Bagel, SP on | 13,050,834,944 / 14,141,252,608 | 92.289101% | Same language denominator; numerator includes 258,048 Q/K/V bias elements |

Bagel excludes its separate ViT, VAE, connectors and timestep/latent projections; its percentages
are not whole-adapter coverage. Flux1 single-stream weights use TP and its optional SP plan
retains local residual tokens. Input-column AdaLN TP returns complete modulation vectors,
preserving native packing. The guides link reproducible CPU placement/count controls and each changed
model's bounded component-update evidence. H3's guide describes current packed FFN/AdaLN support
without assigning it an unmeasured aggregate coverage percentage.

## Qwen-Image-2.1: execution success and numerical qualification

The [merged TP PR](https://github.com/X-GenGroup/Flow-Factory/pull/270) reports completed
real-weight GRPO runs and isolated small-model parity. The
[historical investigation](https://github.com/X-GenGroup/Flow-Factory/blob/55dcd69433deb11159d5c03c22689518ce6f315e/.agents/knowledge/topics/tensor_parallel.md)
records a `2.7e-4` forward difference and `3.4%` gradient-norm difference versus a `0.19%`
reduction-order baseline. The new actual-weight experiment does not reproduce or erase that
historical measurement. Its original untouched-dense comparison failed 44 of 448 gradient
tensors, with global relative L2 about `0.00154`. Independent serial regrouping of canonical
linear/head contractions explains that first-step difference. A subsequent independently
evolving serial/TP experiment still failed its fixed update bound: Adam amplifies tiny
near-zero gradient differences. Both attempts and their analyses remain recorded failures.

A separately reviewed same-state contract passed at
`5b80b5379a90c3dd51f469bc34986662ddd47698` on all four ranks using hash-bound released
Qwen-Image-2.1 7B weights from download revision
`d26bb61231c349cf6b7896fa83353113880e1ba3` and native VL/VAE-conditioned fixed operands. Its scope is FP32
LoRA rank/alpha 8, FSDP2 DP2×TP2/CP1, with SP, cache and activation checkpointing disabled.
The actual TP model alone makes three AdamW advances; it is never reset from the reference.
Before each step, the canonical serial shadow receives the actual model's current LoRA
parameters. Untouched dense U and independently regrouped S then observe that same state;
neither has an optimizer trajectory.

Across all three steps, S/T outputs are bitwise equal and every one of the 448 gradient tensors
passes the unchanged `rtol=8e-4`, `atol=2e-6`, per-parameter relative-L2 `0.003` bounds. Maximum
observed per-parameter relative L2 is `9.89e-5`; maximum global relative L2 is approximately `8.64e-6`.
Every element of the actual optimizer history is checked against independent FP64 AdamW
recurrences and operation-count FP32 rounding. A hypothetical serial step from the same
validated actual prior moments uses input-gradient-derived error bounds. Learning rate `1e-5`,
epsilon `1e-8`, betas `(0.9, 0.999)` and weight decay `0.01` remain unchanged. The original
fixed cross-update bound is retained as a diagnostic and still fails on some first-step cells;
it is not relabeled as passed.

This replaces the blanket exclusion with the stated three-step, same-state qualification.
It does not establish equivalence of independently evolved serial training, pretrained
CP/SP/cache behavior, long-run convergence or media quality. The
[model CP/SP guide](models/qwen21.md) separately covers reduced-depth random-weight
native FP32 references and a BF16 conditioned Guard lifecycle. Its published `86b848bd`
gate retains its original checkpoint-load and continuation result. Later shared-runner
candidates use the [ordered-overlap contract](context-parallel.md#ordered-continuation-evidence):
exact restore, the first same-state update and each actual later objective/optimizer/state
trajectory, with subsequent cross-run numerical statistics recorded as diagnostics.
The source-bound summary is
`qwen-pretrained-shadow-summary.json`, SHA256
`16ac0f4c49dadfc985079c4ba1dc83e35ee0535daa37cd0d722a36ea0a668d5d`;
its published evidence link belongs in the [model handoff](../../../roadmap/handoffs/v0.5-model-closure.md).

## What the Wan qualification proves

[`test_tp_training_qualification.py`](../../../tests/distributed/test_tp_training_qualification.py)
uses the production TP applier and Flow-Factory's Accelerate subclass, followed by actual FSDP2
preparation on a `(dp_shard=2, tp=2)` CUDA mesh. Its reference processes the same two logical
samples without sharding. Four successive AdamW updates verify:

- Forward tensors and every gradient, including expected zero gradients and gradient membership.
- Parameter and optimizer identity after preparation, gradient clipping, every actual update,
  and populated Adam moments. Update deltas are checked separately from parameter values.
- Frozen LoRA base parameters remain bitwise unchanged and receive no gradients.
- Activation-checkpointed SP retains a bounded live CUDA allocation after repeated iterations.

The fixture uses AdamW with learning rate `1e-3`, epsilon `1e-4`, weight decay `0.01`,
and global gradient clipping at `0.5`. Elementwise checks use `rtol=2e-4, atol=2e-6`;
an additional per-parameter gradient check bounds maximum error by `5e-4` times that
gradient tensor's maximum magnitude. Q3 adds an absolute FP32 floor of `2e-7`, while Q2
adds none. Update deltas have the same scale bound plus a four-epsilon parameter-roundoff
allowance and Q3's `2e-7` floor. All parameter names and zero-gradient tensors remain checked.

The Q3 floor is calibrated to this fixture, not a universal TP tolerance: an unsharded
batched-versus-sequential control with identical weights already differs by `4.88e-8` on a
gradient whose maximum is `7.68e-5` (relative `6.35e-4`). After one AdamW update, the TP
trajectory differs by `1.30e-7` in that gradient and `1.40e-7` in its update. A pure relative
`5e-4` cutoff would reject the unsharded control. Q3 retains that batching control and records
its result with the worst TP gradient/update in each rank's JSON, while asserting the combined
bound. Recalibrate with evidence when changing precision, optimizer settings or geometry.

The Q2 fixture covers full tuning and LoRA, TP-only and checkpointed SP layouts. Its two Wan
blocks have real Diffusers structure and random weights; it is not a pretrained-model test.
The Q3 fixture loads only the locally staged transformer of
`Wan-AI/Wan2.1-T2V-1.3B-Diffusers`, snapshot
`0fad780a534b6463e45facd96134c9f345acfa5b`. It checks 600 LoRA parameter gradients/updates and
825 frozen base parameters per step. The inputs are small latent tensors and synthetic text
embeddings: VAE encoding, prompt processing, reward adapters, generated-media quality and full
training recipes are outside this Q3 claim. The test hashes the actual config and weight files
into each rank's JSON evidence; a model cache filename is not assumed to be a content hash.

Observed on 2026-11-04: Python 3.10.22, PyTorch 2.10.0+cu129, Diffusers 0.41.0,
Transformers 5.20.2, Accelerate 1.15.0 and PEFT 0.21.2, on four local NVIDIA L20Y 80 GB GPUs.
Bagel and FLUX.3 Q0 cases additionally passed in an isolated optional-model environment:
Python 3.13.16, PyTorch 2.10.0+cu129, Diffusers 0.42.0.dev0 at
`48cdbaef5ee3d718aaee392b44f2099ae511c4a5`, Transformers 5.20.2, Accelerate 1.15.0,
PEFT 0.21.2 and FlashAttention 2.8.3.post2. Their five previously unavailable baseline cases
(two Bagel plan cases, Bagel/FLUX.3 region parity, Bagel capability classification) were run
successfully there; no fake attention implementation or silent skip supplies that evidence.
These separate tuples must be preserved in reports instead of claiming one environment tested all rows.

The local CUDA run used FP32 without TF32. These results do not qualify arbitrary dependency upgrades,
BF16 numerical tolerances, larger TP degrees, other attention kernels or cross-node execution.

The [2026-11-05 merged-baseline follow-up](../../development/validation/integration-qualification.md) re-runs both local CUDA
cases on the merged source and adds two-node Q2 evidence: the four Wan full/LoRA TP/SP cases with
TP crossing nodes, plus checkpointed SP/LoRA with DP crossing nodes. The pretrained Q3 case stays
single-node; the follow-up does not expand its precision, dependency or full-pipeline claims.

Run the Q2 and Q3 cases separately so missing weights cannot be mistaken for a passed test:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 python -m pytest -q \
  tests/distributed/test_tp_training_qualification.py::test_wan_fsdp2_tp_updates_match_fixed_replica_reference \
  --basetemp=.scratch/tp-q2

FF_WAN_TP_MODEL_PATH=/absolute/path/to/transformer \
  PYTHONPATH=src OMP_NUM_THREADS=1 python -m pytest -q \
  tests/distributed/test_tp_training_qualification.py::test_pretrained_wan_lora_updates_match_fixed_replica_reference \
  --basetemp=.scratch/tp-q3
```

Select four visible GPUs before these commands. Each test writes per-rank JSON below its
`--basetemp` directory. A required case with insufficient GPUs or missing model files is blocked;
a pytest skip is not passing evidence. Keep the final source commit, catalog/plan identity,
dependency tuple and these artifacts together. Cross-node execution of the released-weight
Q3 canary remains unqualified; the two-node Q2 follow-up above retains its separate scope.

## Across-head normalization precision

Wan and LTX-2 normalize q/k over the head dimension split by TP. Both standalone and paired
norms share one FP32 activation view between their direct and statistic derivative branches.
They combine those gradients before converting back to FP16/BF16; separately rounding the
branches can corrupt nearly cancelling gradients despite matching forward outputs.
The [two-rank regression](../../../tests/distributed/test_across_head_norm_pairing.py) checks these
cases against FP64 and preserves the paired collective count and cleanup behavior.

This correction retains one FP32 view per local query/key slice for backward. Historical peak
memory measurements predate it; no new peak-memory or throughput guarantee is implied.

## Sharded target materialization

By default every rank keeps its loaded target root while TP and FSDP2 shard it, so the
initialization peak holds the TP-local root plus every new FSDP2 shard. With CPU-RAM-efficient
loading off, Accelerate 1.15 also retains an unused full state dict during `fully_shard`
([accelerate#4511](https://github.com/huggingface/accelerate/issues/4511)). On released
MiniMax H3 this is the difference between DP2×TP2 running out of memory and fitting.

```yaml
parallelism:
  tp_size: 2
  # Unset prefers sharded; set `replicated` to keep the previous path,
  # or `sharded` to make an unsupported run an error.
  target_materialization: sharded
```

When `target_materialization` is unset, the trainer resolves it before loading weights. It
chooses `sharded` inside the boundary below and otherwise falls back to `replicated`. Under
FSDP2 the fallback logs a warning naming the blocking setting; backends that never shard target
parameters, such as DDP and ZeRO-2, log the fallback at info level.

`sharded` runs these steps after initial-state synchronization:

1. The loaded target values are retained as a host-side source, and the target parameters move
   to the meta device. Buffers stay real.
2. TP/SP plans and Accelerate's FSDP2 preparation build the final layout on meta. TP on meta
   tensors performs no broadcast.
3. `FSDPBackendLoadRuntime` allocates each rank's local shards and copies into them the block of
   the source that the final mesh and placements select.
4. Optimizer groups are rebuilt onto the materialized parameters by canonical name. Accelerate's
   own remap keys parameters by `data_ptr`, which is zero for every meta tensor, so it cannot be
   used here.

Local values equal the replicated path bitwise. This holds on gloo for every opted-in registry
layout with LoRA: SD3.5, Z-Image, MiniMax H3, Flux1/Kontext, Flux2 dev and Klein 4B/9B, Flux3
base, Qwen-Image, Edit-2509/2511 and 2.1, Wan2.1 T2V/I2V, Wan2.2 dual/expanded and FLF2V, LTX-2.0
T2AV and LTX-2.3 I2AV, SenseNova, and Bagel MoT with and without its SP attention plan. It also
holds on four NCCL GPUs through Accelerate for one AdamW step. The runtime identity is unchanged.

With `fsdp_cpu_ram_efficient_loading: true`, only rank 0's source is used, broadcast one
parameter at a time. TP can therefore use efficient loading under `sharded`, while the
replicated path still rejects that combination.

An explicit `sharded` is rejected before loading in four cases, and an unset value falls back
in the same cases:

- a backend other than FSDP2;
- an adapter that does not declare `supports_sharded_target_materialization`;
- context parallelism, whose hooks on meta modules are not yet qualified;
- FSDP2 CPU offload.

### Checkpoint-backed targets

Adapters that also declare `supports_checkpoint_slice_loading` construct their target components
without reading weights once `sharded` is resolved. This covers every adapter that loads through
`BaseAdapter._load_diffusers_pipeline` (SD3.5, Wan2.1/2.2, Flux1/Kontext, Flux2/Klein,
Qwen-Image/Edit/2.1, Z-Image and LTX-2) and every adapter that loads through
`ModularPipelineRuntime` (MiniMax H3 and Flux3). The modular runtime builds requested targets from
their `ComponentSpec` and loads the remaining components natively.

- **Construction.** Each target parameter becomes a `CheckpointParameter`, a stride-0 placeholder
  with the dtype the native loader would give it. Buffers stay real. Every rank later reads only
  its own block of each tensor from the safetensors files.
- **Parity with the native loader.** Dtypes match `from_pretrained` in both loader modes:
  - `low_cpu_mem_usage` casts to the requested dtype and honors `_keep_in_fp32_modules`.
  - Eager loading assigns stored tensors only when the first state entry already has the stored
    dtype.

  Construction disables initialization, as the native loader does, so LoRA draws the same global
  RNG values and local shards equal native loading bitwise.
- **Safety.** Reading a placeholder's value before materialization raises.
- **Fallback.** These checkpoints keep native loading, with a warning: missing or reshaped keys,
  legacy attention renames, quantized or non-safetensors weights, and a resume that writes base
  weights first.

On released Wan2.1-T2V-14B stored in FP32 and loaded in BF16, native construction peaks at
80.53 GiB host RSS and takes 26.7 s per process. Checkpoint-backed construction peaks at 0.84 GiB
and takes 0.1 s. Spot checks of the `_keep_in_fp32_modules` tensors and BF16 tensors match
exactly. An FP32 checkpoint loaded in FP32 is already memory-mapped by the native loader, so the
saving appears when dtypes are converted, as with H3's FP32 originals of a BF16 checkpoint.
Checkpoint-backed targets also skip the FSDP2 initial-state broadcast, which otherwise sends every
target parameter from rank 0: about 66 GB per rank for H3 in BF16. Bagel and SenseNova build
their roots with handwritten and Transformers loaders, so they still load natively.

## Checkpoint and compatibility limits

The existing [TP checkpoint tests](../../../tests/distributed/test_tensor_parallel_checkpointing.py)
exercise model-only gathers and basic optimizer-state round trips over the composed mesh.
FSDP2 fixed-topology exact resume now also persists EMA, reference copies and algorithm
snapshots through rank-local auxiliary payloads; see [parameter state](../../development/parameter-state.md) for
the compatibility contract and two-/four-GPU next-update qualification. A successful model-only
save does not establish exact resume of auxiliary state.

With TP disabled, this qualification changes no production execution path. Reuse older evidence
only for its exact unchanged scope; do not relabel historical campaigns as a run of this revision.
The identity-seeded offline VAE change already merged with TP affects SD3.5 offline DPO even
when TP is disabled: it intentionally differs from the old process-RNG behavior. Universal
bitwise equivalence to a pre-TP checkpoint is not promised.

The historical H3 loading-capacity exclusion is separate from the new packed FFN/AdaLN TP
implementation. The recorded released-model FP32 DP2×TP2/CP1 attempt now completes TP at
63.39 GiB per rank, then exhausts memory during FSDP2 initialization because original TP
state remains live while FSDP shards are allocated. The eight-GPU DP2×TP4/CP1 loading/LoRA
capacity gate passed all ranks at `414d10acfac3c0173045e7fe2dfbde7c3c4f2379`: 33.225 GiB allocated before
FSDP and a 49.833 GiB initialization peak per rank. Tiny-native DP2×TP2×CP2 semantic controls
do not qualify released-model CP2 memory fit; see the [H3 guide](models/minimax-h3.md).

A separate final-source TP4 capacity run at `75e9fac069b1b7043ce1f4143a9a31db6d987bd9` passed all eight ranks. Its
[fresh measured record](models/minimax-h3.md) and [published evidence](https://github.com/X-GenGroup/Flow-Factory/actions/runs/42426359108)
retain the same narrow loading/LoRA scope; the historical `414d10ac` measurements above are unchanged.
[Wan CP and TP/SP+CP SFT](context-parallel.md) have separate experimental capabilities and
composition evidence. This TP/SP qualification alone does not imply those paths. EP remains
later roadmap work.

## Historical investigations

The [preserved investigation archive](tensor-parallel-history.md) retains earlier memory and
throughput measurements, low-coverage H3 preparation failures, partial-gradient and checkpoint
failures, sample-retention diagnostics, and the two-arm GPU campaign. Its old coverage tables and
phrases such as “current” refer to those historical sources. Current admission and plan regions
come from the model contracts linked above. A later pass does not erase a prior failed contract.
