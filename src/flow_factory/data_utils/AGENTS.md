# Data Scope

Follow the repository root `AGENTS.md` and use `/ff-develop` or `/ff-debug` as appropriate.

- Typed schemas, cache identities, reconstruction sidecars, sampler geometry, and source provenance
  are persistent contracts.
- Preserve semantic fields across canonicalization, storage, projection, decoding, and adapter
  preprocessing.
- Sampler or multi-source scheduling changes are R4 candidates; local schema/cache changes require
  at least R3 and migration or invalidation evidence.
- Dataset epochs advance only after clean official-loader exhaustion.
