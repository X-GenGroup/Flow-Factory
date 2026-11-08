# Maintaining model parallelism

Shared mechanisms live in `src/flow_factory/models/parallelism/`. Native module paths,
stream ordering, attention masks, padding policy and prefix/cache state remain in each model
family. The public adapter methods and `ModelPreparation` still own declaration and installation.

## Find the owner

| Module | Responsibility |
|---|---|
| `tensor.py` | Parameter placement and head-normalization styles |
| `sequence.py` | SP regions, token-partial parameter/conditioning derivatives and activation boundaries |
| `collectives.py` | Communication operators with explicit gradient and storage semantics |
| `packed.py` | Canonical packed projection weights and reversible activation exchange |
| `plan.py` | Adapter declarations, conditional patterns and total parameter coverage |
| `peft.py` | LoRA branch derivation and wrapper peeling |
| `guards.py` | Region checks, collective settling, norm pairing and replicated-input diagnostics |
| `runtime.py` | Apply the plan, retain trainability and localize native attention geometry |
| `context.py` | CP capabilities, prepared-root ownership and mesh mapping |
| `hooks.py` | Validate and relocate native CP callbacks around Torch SP boundaries |

Dependencies flow from styles/collectives to plan/PEFT, then guards/runtime. Shared code must not
import a model family. Existing `models/tensor_parallel.py`, `context_parallel.py` and
`packed_tensor_parallel.py` retain explicit compatibility exports; new code imports the actual owner.
Moving implementations does not change canonical parameter names or layouts.

## Declare layouts locally, install mechanisms centrally

Each family continues to declare its TP/SP plans and CP capability. It supplies native modules,
hook names and expected metadata to the shared hook helpers. Validate all required native hooks
before removing any of them. A `NativeContextHook` refuses to detach a callback replaced since
validation. Retain returned `ContextParallelBoundary` handles in the prepared contract.

Torch hooks surround Diffusers' wrapped `forward`; registration order across those systems alone
does not establish execution order. CP input splitting must precede SP scattering, and SP output
restoration must precede CP gathering. `register_context_input` can execute an explicitly ordered
native padding/split chain. `compose_context_output` handles an idempotent split/output composition
at model-selected paths and checks that its retained callback is still installed.

H3 still chooses its padding and per-block modulation indices. Qwen still splits image and text
at different producers. Flux still chooses stream and rotary ordering. Bagel, Qwen-Image 2.1 and
SenseNova still own prefix/cache and generation-state semantics. Do not encode these distinctions
as switches in a universal transformer processor.

## Preserve derivative and storage semantics

- `gather_tokens` gathers token shards and **reduce-scatters** consumer-partial gradients. Both
  directions settle their collectives before returning.
- `redistribute_sequence` uses DTensor's placement-derived backward and clones its local output.
  It is not interchangeable with `gather_tokens`; the clone also releases full producer storage
  after a scatter.
- `SequenceBroadcastSubmodule("linear")` installs the conditioning derivative boundary around
  the complete AdaLN projection, including PEFT branches. The parent owns this boundary while
  the child's parameter placement stays independently derivable.
- Model-owned padding still determines valid masks, RoPE/index rows, crop rules and checkpoint
  replay metadata. A tensor padding helper cannot decide these policies.

## Validation

Start with plan/LoRA and hook-order contracts. The hook tests use noncommuting transformations
and check gradients, positional/keyword calls, stale bindings and validation before mutation.
Existing native family fixtures remain the numerical oracle; do not replace them with mocks of
the shared implementation. Compare every parameter gradient and preserve frozen membership.

For changed communication, use focused two-rank Gloo/CUDA derivatives and checkpoint cases.
Add TP2/CP2 composition and a selected DP2/TP2/CP2 FSDP2 case when shared ownership is affected.
Choose cases by changed semantics, not the full model × algorithm × backend product. New shared
paths retain R4 routing; source-bound evidence and the selected-plan review remain required.

See [TP/SP contracts](../reference/parallelism/tensor-parallel.md),
[CP contracts](../reference/parallelism/context-parallel.md), and
[validation selection](validation/validation.md).
