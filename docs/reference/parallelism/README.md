# Parallelism implementation reference

Start with the [parallel training guide](../../../guidance/parallelism/README.md) for concepts,
configuration and model/algorithm choices. These pages describe the contracts behind those choices.

| Contract | Read |
|---|---|
| Physical ranks, logical samples and mesh ownership | [Topology](topology.md) |
| Parameter shards, activation layouts and LoRA | [TP and SP](tensor-parallel.md) |
| Context slicing, complete-output backward and lifecycle | [CP](context-parallel.md) |
| Model-specific layout and numerical evidence | [Model index](../../../guidance/parallelism/models.md) |
| Objective, feedback and role ownership | [Algorithm index](../../../guidance/parallelism/algorithms.md) |
| Canonical weights and exact continuation | [Parameter state](../../development/parameter-state.md) |
| Diff-driven checks and evidence provenance | [Automatic validation](../../development/validation/automatic-validation.md) |
| Earlier measurements and investigations | [Historical TP record](tensor-parallel-history.md) |

The model adapter and trainer capability declarations remain the admission authorities. The
validation catalog binds named cases to executable commands and semantic obligations; the roadmap
handoffs retain source commits, failures and published artifacts. Historical measurements do not
become measurements of a descendant commit when a document moves.

See [release acceptance](../../release/v0.5.md) for the distinction between implementation closure,
reviewed test evidence, merged integration and a published version.
