# Parallelism support: where to inspect

**Read when** adding, reviewing or documenting a model's TP/SP/CP support, or composing an
accelerator with those paths. This is a navigation leaf, not an admission registry.

## Start at the authoritative owner

| Question | Inspect |
|---|---|
| Which implementation is selected? | [Model registry](../../../src/flow_factory/models/registry.py) and [trainer registry](../../../src/flow_factory/trainers/registry.py); resolve the concrete class and inherited family helpers |
| What TP/SP behavior is declared? | Adapter `tensor_parallel_plan()`, `sequence_parallel_plan()` and `sequence_parallel_by_default`; shared resolution and style semantics in [parallelism/plan.py](../../../src/flow_factory/models/parallelism/plan.py) |
| What CP layout can run? | Adapter `context_parallel_capability`, component names, metadata/input validators and dependency gate; [typed CP contract](../../../src/flow_factory/models/parallelism/context.py) |
| Which algorithm/backend can compose? | Concrete trainer's explicit CP declaration and [training gate](../../../src/flow_factory/trainers/common/context_parallel.py); [TP/backend gate](../../../src/flow_factory/trainers/multirole/backend.py) |
| Where are actual token boundaries? | Family-owned setup/composition hooks plus native model forwards; follow `configure_context_parallel_module`, `install_context_parallel_module` and `compose_context_parallel_sequence_boundaries` from [BaseAdapter](../../../src/flow_factory/models/abc.py) |
| Who installs and owns prepared state? | [ModelPreparation](../../../src/flow_factory/trainers/common/model_preparation.py), bundle/variant routes and [runtime identity](../../../src/flow_factory/trainers/common/runtime_identity.py) |

Read the native forward around each claimed boundary. Weight placement does not determine token
layout. Distinguish repeated residual stacks from conditioning, prefix/cache construction, packing,
spatial heads and final reconstruction. For gradient, checkpoint and CP-before-SP invariants, use
[the TP owner/diagnostic topic](tensor_parallel.md) and [component runtime](component_runtime.md).

## Select focused evidence

The default workflow generates a plan from the actual change, not from a fixed model × algorithm
matrix. [Automatic validation](../../../docs/development/validation/automatic-validation.md) binds
base/head, affected owners and transitive artifacts. The
[selection policy](../../../config/validation/selection.yaml) owns required coverage; the
[catalog](../../../config/validation/catalog.yaml) owns executable cases, topology and oracle scope.
The [manual plan example](../../../config/validation/plan.example.yaml) is a template, not a
preselected passing plan.

Inspect the test behind a case ID: native geometry, tuning targets, input/parameter derivatives,
checkpoint behavior, updates/state and observed token rows must answer the changed contract.
A component probe and a trainer lifecycle establish different claims. A selected case is not
execution evidence; the [validation contract](gpu_validation.md) requires source/result identity.
Do not infer new capacity or throughput from a numerical or local-row check.

## Route documentation by audience

| Need | Destination |
|---|---|
| Concepts, configuration and coverage overview | [Human parallelism guide](../../../guidance/parallelism/README.md), with model and algorithm indexes |
| Native layouts, ownership, precision and bounded evidence | [TP/SP](../../../docs/reference/parallelism/tensor-parallel.md), [CP](../../../docs/reference/parallelism/context-parallel.md) and their linked model/algorithm references |
| Current tested source and unresolved release obligations | [SP completion](../../../roadmap/handoffs/v0.5-sp-completion.md), [model ledger](../../../roadmap/handoffs/v0.5-model-closure.md) and [delivery handoff](../../../roadmap/handoffs/v0.5-gpu-handoff.md) |
| Earlier numerical failures and measurements | [TP investigation archive](../../../docs/reference/parallelism/tensor-parallel-history.md) and [campaign archive](gpu_campaign_legacy.md) |

Update the relevant owner and reference when support changes. Preserve historical source IDs and
failed outcomes; do not turn old receipts or a declaration into a current-source pass. Keep model
lists and percentages out of this navigation leaf.

## Cross-refs

- UP: [constraints #18b](../constraints.md#18b-tensor-parallelism-scopes-samples-not-ranks),
  [#30](../constraints.md#30-selected-validation-evidence)
- PEER: [TP owners/diagnostics](tensor_parallel.md), [Component runtime](component_runtime.md),
  [Selected validation](gpu_validation.md)
- WORKFLOW: [Model integration](../../skills/ff-new-model/SKILL.md),
  [Acceleration integration](../../skills/ff-new-accelerator/SKILL.md)
