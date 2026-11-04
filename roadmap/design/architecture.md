# Architecture direction

This is the proposed direction for [P05-P16](../README.md#release-sequence). Existing contracts and
registries remain the implementation starting point. The design follows explicit domain ownership,
small verifiable changes, and behavior-based evidence, informed by
[ub-mode](https://github.com/unclebob/ub-mode). It does not require adding a generic framework around
every existing class.

## Ownership and upstream reuse

| Concern | Flow-Factory owns | Reuse |
|---|---|---|
| Distributed execution | Logical sample geometry, group meaning, capability validation, preparation order | Accelerate for existing DDP/ZeRO-2/FSDP2 execution; PyTorch meshes, DTensor, collectives and checkpoint primitives |
| Model computation | Adapter I/O, conditioning, trajectory/replay, model-specific training parity | Diffusers model blocks, encoders, VAE, attention implementations and usable CP plans/hooks |
| TP/SP | Training-aware plans, LoRA/freeze semantics, boundary layouts and gradient obligations | PyTorch parallelization primitives |
| CP | Mapping the training mesh into a model plan, sample/RNG identity, loss scaling and lifecycle | Ordinary Diffusers Ulysses/Ring mechanisms where their backward and layout contracts are verified |
| Model state | Component/variant identity, snapshot versions, writers, update cadence and persistence | PEFT adapters and backend-aware tensor/state storage |
| Algorithms | Objective, acquisition/feedback requirements, role-update semantics | Shared acquisition, feedback and optimization mechanisms |
| Rewards | Identity, batching, readiness, error propagation, ownership and cleanup | Reward implementations and remote services through existing interfaces |
| EP | Model router semantics, expert identity/placement, dense/expert optimizer and checkpoint ownership | NCCL all-to-all first; DeepEP as an optional transport |

Keep the full diffusion training loop in Flow-Factory: calling a Diffusers pipeline is not a
replacement for scheduler state, log-probability, replay, or train-inference consistency.
Maintain one owner for model preparation so attention selection, parallel hooks, parameter
transforms, backend preparation, and compilation cannot apply competing transformations.

Accelerate remains the default integration. P06 tests the narrow bridge needed for CP. Add a native
PyTorch FSDP2 implementation only if a reproducible limitation prevents that bridge from preserving
the required semantics; record the limitation and affected compatibility surface before that PR.

## Parallel geometry

For the initial dense-model design, let `D` be logical data replicas, `T` tensor parallel degree,
and `C` context parallel degree. Then `world_size = D * T * C`. SP reuses the TP group and adds no
independent mesh dimension. SP is available only with TP; defaults and supported layouts remain
adapter capabilities rather than a universal promise.

Ranks in one TP x CP group process the same logical sample and stochastic identity. Data replicas
process different samples. Global batch counts logical replicas, not physical ranks. A possible
FSDP2 composition shards parameters over `D * C` for each fixed TP coordinate; its gradient
reduction and CP loss normalization must be proved together.

The intended layout through a dense transformer region is:

1. A CP rank owns approximately `S/C` tokens, with global positions retained.
2. SP can further distribute residual/norm tokens over TP ranks, approximately `S/(C*T)`.
3. TP gathers the required tokens for column-parallel projections and produces local heads.
4. CP supplies full-context attention through its selected exchange. Ordinary Ulysses redistributes
   sequence/head ownership; for ordinary equal-head MHA, the head count must divide across `T*C`.
5. Row-parallel projection and SP reduce-scatter return the declared residual layout.

Those shapes describe the intended divisible dense case, not every packed or multimodal model.
Check conditioning broadcasts, text/image streams, masks, token order, and position tables at the
actual plan boundaries. GQA, uneven sequences, cached prefixes, and special attention masks need
their own capability decisions. The first CP implementation restores full denoiser outputs at the
adapter boundary so scheduler and trajectory semantics can remain unchanged.

## Parameter and execution state

Extend the existing [ModelBundle](../../src/flow_factory/models/model_bundle.py),
[component variants](../../src/flow_factory/models/variants.py), and
[named-state mechanisms](../../.agents/knowledge/topics/component_variants.md). They already
distinguish simultaneous trainable copies from temporal snapshots.

Every logical state needs a defined identity, writer, lifetime, and update cadence. A frozen
reference has no optimizer writes; an EMA/snapshot can be frozen for autograd while still changing
through an explicit update. Logical states do not each require a full module deepcopy. Live
trainable roles additionally own gradients and optimizer state.

Swap-and-restore snapshots are a serial execution mechanism. Concurrent rollout and training would
need an independently usable versioned execution instance and an algorithm policy for staleness.
That is deferred beyond the v1.0 scope.

Use the existing [ExecutionContract](../../src/flow_factory/contracts/execution.py) to separate
generation/dataset acquisition from runtime-reward/no-feedback execution. Use
[feedback contracts](../../src/flow_factory/contracts/feedback.py),
[overlap contracts](../../src/flow_factory/contracts/reward_overlap.py), and
[role optimization](../../src/flow_factory/trainers/role_optimization.py) for their existing
responsibilities. An algorithm defines its mathematics and update plan without owning backend
placement or model-copy storage.

Reward request batches, optimizer work tiles, accumulation windows, and optimizer steps are
different boundaries. Retain the current complete-rollout policy-version boundary while rewards
run asynchronously and overlap optimization. Ready-order scheduling can change update order;
recovery must preserve that order or checkpoint at a declared drained boundary. Functional overlap
needs a scheduling proof; a claim that reward latency is fully hidden needs a timing experiment.

## Dependency checkpoints

The investigation used Accelerate 1.15.0 and Diffusers 0.42.1 sources below. Recheck behavior and pin
a tested PyTorch/Accelerate/Diffusers/PEFT/attention combination during P06 and P13; these versions
are evidence anchors, not a new dependency requirement imposed by this document.

- [Accelerate CP](https://github.com/huggingface/accelerate/blob/v1.15.0/docs/source/concept_guides/context_parallelism.md)
  and [its implementation](https://github.com/huggingface/accelerate/blob/v1.15.0/src/accelerate/accelerator.py)
  require checking FSDP2 and hook assumptions. Do not activate two competing CP hook systems.
- [Diffusers parallel configuration](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/models/_modeling_parallel.py),
  [CP hooks](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/hooks/context_parallel.py),
  and [attention dispatch](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/models/attention_dispatch.py)
  are the integration boundary. Standalone CP support is not proof that FF TP, LoRA and CP compose.
  Verify backward for the chosen path; the investigated Anything variants were not training-ready.
- [Diffusers TP hooks](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/hooks/tensor_parallel.py)
  do not replace FF's training-aware LoRA/freeze handling. P10 must verify the composition directly.
- [DeepEP](https://github.com/deepseek-ai/DeepEP) supplies transport mechanisms, not the whole MoE
  training contract. P16 selects a concrete version and hardware/software compatibility tuple.

EP is not automatically another independent world-size multiplier. Dense and expert parameters may
use different groups, and expert-choice routing across tokens can change meaning if CP localizes
the routing decision. Preserve the selected model's router before claiming EP + CP support.
