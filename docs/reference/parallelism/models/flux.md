# Experimental Flux1 context parallelism

Flux1 declares the native Diffusers joint image/text layout. It uses both native block stacks,
the packed image codec, CLIP pooled conditions and T5 token conditions. CP splits image/text
streams and rotary IDs at the root. Model-owned composition restores SP before the native CP
output gather, then the adapter exposes complete image predictions.

Admission requires guidance embeddings, both stacks, native `FluxAttnProcessor`, patch size one,
equal packed input/output channels divisible by four, and heads divisible by CP × TP. Image tokens
must divide CP, and additionally TP with SP. With SP enabled, the root pads text to CP × TP and
masks introduced keys; without SP, text retains the CP divisibility requirement. Internal padding
is not admission for caller-supplied special masks, IP adapters or foreign processors. Runtime uses
one sample, native attention, BF16, AdamW and FSDP2 with replicated-DP=1 and sharded-DP>=2.
[Kontext](flux-kontext.md) owns reference-image packing and target-only cropping separately.

## Tensor-parallel composition

Flux1 shards attention and feed-forward weights in both native transformer stacks. Its single
stream combines attention and MLP channels in a 1:4 packed input; `PackedRowwiseParallel`
exchanges activations while retaining canonical two-dimensional weights and LoRA state.
The four AdaLN projection patterns, including `norm_out.linear`, shard input columns and return
full modulation vectors. Attention shards heads before native Ulysses exchanges them, so head
counts must divide by TP times CP.

`sequence_parallel: true` keeps image and text residuals sharded through the dual blocks and
keeps the joint residual sharded through the single blocks. Attention and MLP entries gather
matching operands; the packed single output reduce-scatters. Rank-major gathered text/image
pairs use matching rotary rows and key masks. The final image norm and projection also run on
local tokens, with output gathering after the complete native/PEFT projection wrapper.

Each AdaLN parent installs its conditioning-gradient reduction after native/LoRA linear branches
merge. Input-column TP and parameter ownership remain separate. The final projection uses plain
replicated SP parameter ownership, so custom final-projection LoRA preserves the complete wrapper
boundary. Native weights, parameter names and state ordering remain canonical. SP stays opt-in.

For the native 19-dual/38-single, 24-head by 128-width guidance model, the plan shards
11,854,737,408 of 11,901,408,320 parameter elements (99.608%), up from 36.165% with the
dual-stream-only plan. Embeddings, remaining biases and per-head norm weights stay replicated.
These are parameter counts, not measured GPU memory or throughput. Focused CPU controls check
actual placements, full/default and expanded block/AdaLN LoRA, SP off/on, input and parameter derivatives,
three Adam updates and canonical state reload. `flux-complete-tp-sft` is one two-GPU scenario
covering native Flux1 and reference-image Kontext supervised-flow components at DP1/TP2/CP1
with SP enabled. It uses genuine prompt/VAE conditions and target-only losses, not the production
SFTTrainer lifecycle (which requires DP sharding with FSDP). It does not replace or relabel
historical CP lifecycle evidence.

The root text-padding hook runs before native CP splitting. CP cuts precede SP cuts; the final
native CP gather is moved outside the projection wrapper so it runs after SP restoration.
Validation checks the installed callback metadata before relocation. No trainer-specific forward
or extra CP loss-scaling owner is introduced. Exact continuation also checks
[prepared parameter placements](../topology.md#data-feedback-and-checkpoint-boundaries).

Selected SP evidence includes `flux-sequence-contracts`, `flux-complete-sp-cp-contracts`,
`flux-complete-tp-sft`, the stable `flux-dual-sp-cp-reference` case and
`flux-kontext-sp-guard-ordered-full`. The historical “dual” case ID remains stable; its current
shape observations include the expanded text/single/output regions. CPU native full/expanded-LoRA
controls cover odd text, eager/checkpoint execution, every input/parameter derivative, three Adam
updates/moments and canonical reload. Four-rank CPU CP composition has its own case rather than
being described as a two-rank test. The prepared reference selects DP2 TP2 CP2 and retains its
missing-CP-factor control. The lifecycle retains reference encoding, ordered feedback, evaluation
and [continuation](../context-parallel.md#ordered-continuation-evidence).

These are selected contracts; the final source-bound results determine which passed. Earlier
partial-SP or SP-disabled receipts do not certify the expanded regions or a memory/speed gain.

The new qualification leaves select DP2 × TP2 × CP2: full Guard with ordered overlap and LoRA
DPPO with ready overlap. Both reuse the native acquisition, independent objective, evaluation,
checkpoint and fresh-process continuation checks below. Fixed-operand FP32/BF16 controls compare
CP1 and CP2 at TP2, matching logical-DP and TP rank coordinates, while the existing CPU TP
reference compares full/LoRA output, every gradient and optimizer updates against an unsharded
model. These leaves are qualification targets; the historical CP-only passes below do not certify
TP+CP or SP.

Strict CP resume identity includes the adapter capability. This extension changes its
`tensor_parallel` declaration from false to true, so Flux CP checkpoints created before the
extension cannot resume exactly under the new identity, including when TP remains one.
At CP=1, this model CP capability adds no identity fields. The
[shared scheduler compatibility note](bagel.md) applies independently to the full runtime
identity. The continuation cases qualify checkpoints created by the same final source and topology.

## Coupled evidence and its limits

The selected native intersections are Flux + GRPO-Guard full tuning with ordered overlap
(case D), and Flux + DPPO LoRA with ready overlap (case E), both TP1 × CP2 × DP2. The shared
loop owns actual acquisition, pending asynchronous CPU reward work, complete group/tile
selection, advantages, optimization, drained evaluation/checkpoint, and fresh-process
continuation. The Flux recipe supplies model/algorithm configuration and independent
objective observations through an explicit module interface.

Both native scenarios passed final qualification at `cb28a74cb1f4190c8801436a01a28764df04ccb6`
in [PR #306](https://github.com/X-GenGroup/Flow-Factory/pull/306), merged as `077f3a23`. The
[handoff](../../../../roadmap/handoffs/v0.5-gpu-handoff.md#completed-ae-delivery-layers) retains the original
GPU source, selected evidence and final CI provenance. These results do not claim pretrained
quality, convergence, performance, arbitrary masks, in-flight recovery, or topology-changing exact resume.

Each case also includes fixed-parameter, fixed-input native FSDP2 controls at CP1 and CP2 in
FP32 and BF16. These compare outputs, every gradient, clipping, Adam moments, and updates.
The FP32 control retains the existing strict component bounds. BF16 uses the shared precision
bounds: a `2/128` gradient norm cap, `4/128` update norm cap, and per-element `0.02` relative
plus `1e-3` cancellation floor. Independent FP64 clipping/Adam equations propagate accepted
gradient differences into moment and update bounds. Zero/half/double/sign corruption and the
actual omitted-CP-factor backward must fail. No SFT LoRA or DMD2 trajectory exemption applies.

Same-policy retained replay within one topology must remain exact before the first update.
Independent rerollouts under different CP layouts are not assumed to recreate identical
floating-point operands. Different ready schedules may change update order and accumulation;
no bitwise trajectory equality across schedules is required. Fixed-operand model controls
bind both parameters and conditioning inputs instead.

Checkpoint loading preserves the saved tensors, optimizer state, runtime identity and RNG exactly.
The current [ordered continuation contract](../context-parallel.md#ordered-continuation-evidence)
keeps work identity, RNG and progress exact, and compares the first resumed same-state update
with the original BF16 bounds. Every later actual objective/public-output derivative,
clipping/Adam, EMA and final-state check remains required; later cross-run numerical
statistics are diagnostic. Each final snapshot is bound to its own last observed update
for both ordered and ready schedules. Earlier published runs retain their original
comparison semantics and source identities.

## Objective boundary evidence

Deterministic CPU controls call the real GRPO, GRPO-Guard, and DPPO optimization paths. An
independent FP64 oracle checks loss and upstream derivatives, positive/negative/zero advantages,
clipping and its exact boundary, x/v mask spaces and DPPO's inclusive mask boundary, reference
KL in both spaces, CP normalization, and gradient accumulation. Packed three-dimensional
batch>1 controls expose wrong broadcasting. A fully clipped/masked policy contribution may
have zero derivative; reference KL remains a separate contribution. Native scenarios require
nontrivial learning over the run, without requiring every tile to produce policy gradients.

This replaces the old stochastic GPU requirement that every acquisition must happen to hit
both Guard clipping branches with a fixed margin. The previous Guard probes and their margin
failures remain historical failures; they are not relabeled passes. GPU runs continue to check
every actual objective, gradient, optimizer update, and state boundary. CPU branch controls
supply deterministic mathematical coverage without changing the native recipe to force it.

## Composition map

Admission is the intersection of existing model and concrete-trainer declarations, rather than
a second support registry. Adding Flux therefore reaches the existing SFT, GRPO, and DMD2
contracts as well as the new Guard/DPPO declarations.

| Boundary | Authoritative unchanged owner | Evidence and limit |
| --- | --- | --- |
| Native Flux token/rotary split, both block stacks, gathered output | Diffusers Flux `_cp_plan`, `ModelPreparation`, `ModelBundle` | D/E fixed-input FP32/BF16 CP controls; real wrapped blocks observed |
| Packed image encoding and conditioning | Flux output codec, `encode_prompt`, `forward` | CPU real CLIP/T5/VAE save/load and codec/transformer backward; D/E native generated images and exact retained replay |
| Complete-output backward and GAS | `BaseTrainer._backward_full_output_loss`, accelerator/role coordinator | Registered-trainer CPU owner tests; packed SFT loss/reduction and Guard/DPPO upstream controls; native D/E gradients |
| SFT offline objective | `SFTTrainer.optimize_batch`, shared flow-matching reduction | Packed CPU reduction/backward plus existing offline lifecycle proofs; no direct native Flux offline traversal in D/E |
| DMD2 live generator/fake roles | DMD2/distillation runtime, variant registry, role coordinator | Existing shared role/backend proofs and packed 3D independent DMD2 objective/upstream CPU controls; no direct native Flux multi-role endpoint in D/E |
| EMA, reference snapshots, checkpoint/resume | Existing parameter state and runtime identity owners | D/E single-policy native lifecycle; layout identity CPU regression. DMD2 role state is derived from its unchanged owner proofs, not directly rerun on Flux |

The two absent native endpoints are explicit coverage limits. This compositional argument does
not establish a full Flux algorithm-by-mode matrix and does not add a model-specific blacklist.
