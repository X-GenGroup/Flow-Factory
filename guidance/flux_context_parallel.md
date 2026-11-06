# Experimental Flux1 context parallelism

Flux1 declares the native Diffusers joint image/text layout. It uses real dual-stream and
single-stream blocks, the existing packed image codec, CLIP pooled conditions, and T5 token
conditions. Diffusers splits image/text tokens and their rotary IDs at the transformer root;
`proj_out` gathers the complete image output. There is no Wan sequence-boundary bridge and no
Wan cross-attention precision policy on this path.

The declaration requires guidance embeddings, both block stacks, native `FluxAttnProcessor`,
`patch_size=1`, equal packed input/output channels divisible by four, and attention heads
divisible by CP. Runtime image/text token counts must be positive and divisible by CP. The
initial training contract keeps batch size one, one transformer target, native attention,
BF16, AdamW, and FSDP2 with replicated DP=1 and sharded DP>=2. Added image KV, special masks,
IP-adapter processor replacements, uneven tokens, and Flux TP+CP remain rejected. CP2/DP2 is
the direct evidence geometry; other structurally valid degrees are not directly qualified
by these cases. No exact-degree model admission rule is introduced.

## Coupled evidence and its limits

The selected native intersections are Flux + GRPO-Guard full tuning with ordered overlap
(case D), and Flux + DPPO LoRA with ready overlap (case E), both TP1 × CP2 × DP2. The shared
loop owns actual acquisition, pending asynchronous CPU reward work, complete group/tile
selection, advantages, optimization, drained evaluation/checkpoint, and fresh-process
continuation. The Flux recipe supplies model/algorithm configuration and independent
objective observations through an explicit module interface.

Qualification remains pending until those exact-source cases pass. Preparing a fixture or
passing its CPU tests is not native CUDA evidence. This page does not claim pretrained quality,
convergence, performance, arbitrary masks, in-flight recovery, or topology-changing exact resume.

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
Ordered continuation keeps work identity, RNG and progress exact while checking subsequent
gradients and updates with the existing BF16 bounds and independent Adam/EMA equations. Each
final snapshot is bound to its own last observed update for both ordered and ready schedules.

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
