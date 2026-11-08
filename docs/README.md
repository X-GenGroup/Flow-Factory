# Developer documentation

For running training, start with the [user guide](../guidance/README.md). This directory holds
implementation contracts and validation/release procedures for contributors and agent tools.

| Task | Entry point |
|---|---|
| Add a model | [Model integration](development/new-model.md) |
| Change preparation or parameter ownership | [Preparation](development/model-preparation.md) · [Parameter state](development/parameter-state.md) |
| Set up local checks and CI | [Quality workflow](development/validation/quality.md) |
| Select evidence from a diff | [Automatic validation](development/validation/automatic-validation.md) · [Validation levels](development/validation/validation.md) |
| Understand catalog and consistency checks | [Catalog](development/validation/validation-catalog.md) · [Repository consistency](development/validation/repository-consistency.md) |
| Inspect a model's parallel contract | [Parallelism reference](reference/parallelism/README.md) |
| Prepare v0.5.0 | [Release acceptance](release/v0.5.md) · [Roadmap](../roadmap/README.md) |

## Documentation ownership

- `guidance/`: readable concepts, configuration choices and user-facing support boundaries.
- `docs/development/`: contributor procedures and shared implementation contracts.
- `docs/reference/parallelism/`: exact model/algorithm layouts, numerical obligations and evidence links.
- `roadmap/`: work items and historical source-bound handoffs. Historical execution identities stay intact.
- `.agents/`: task/path routing, conditional implementation rules and machine-readable evidence selection.
- Source docstrings: local shapes, units, ownership and failure behavior; link to the contract instead
  of repeating a model's history or qualification ledger.

Update the authoritative contract first, then its user summary and incoming links. Support tables
describe registered implementations; they are not a second runtime admission registry. The
repository checker validates local links and anchors across both documentation trees.

`guidance/acceleration.md` is a compatibility entry for existing runtime diagnostics and external
links. Its content lives only in `guidance/configuration/acceleration.md`.
