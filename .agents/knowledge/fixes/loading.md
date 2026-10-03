# Loading Fix Patterns

**Read when**: Debugging or changing loading behavior.

### Preprocessing cache identity includes precision policy
- **Date**: 2026-08-28
- **Symptom**: Changing `component_load_dtypes` or `frozen_parameters_dtype` could reuse condition embeddings computed under a different component precision policy.
- **Root Cause**: The offline cache fingerprint named the model but omitted the precision configuration that controls preprocessing component loading and storage.
- **Fix**: Offline condition-cache extras now include a sorted canonical JSON representation of both dtype policies, with regressions for load-policy changes, frozen-policy changes, and mapping-order stability.
- **Lesson**: Cache identity must include every configuration value that can change preprocessing numerics, even when that value is enforced during model loading rather than passed to the preprocessing function.
- **Related Constraint**: #20

### Adapter dtype manifests may span optional checkpoint components
- **Date**: 2026-08-30
- **Symptom**: Every Wan2.2 TI2V trainer rejected the Wan I2V adapter's `image_encoder` load-dtype
  default even though the checkpoint legitimately omits that optional component.
- **Root Cause**: The eager loader validated an adapter-wide dtype manifest only against components
  present in one checkpoint, conflating the checkpoint instance with the pipeline class's wider
  optional-component contract.
- **Fix**: Eager pipeline loading now validates adapter manifest selectors against the union of the
  checkpoint components and the pipeline class's declared optional components, while resolving
  dtype arguments only for components actually present. User overrides remain strict against the
  selected checkpoint. Regressions cover absent and present optional components, invalid manifest
  selectors, and explicit user selection of an absent component.
- **Lesson**: Adapter defaults may intentionally cover several checkpoint variants of one pipeline
  class. Optional class-level declarations belong to manifest validation, but they must not create
  components or weaken the fail-fast contract for checkpoint-specific user overrides.
- **Related Constraint**: #20

### Supported adapter paths require their upstream optional dependencies
- **Date**: 2026-08-30
- **Symptom**: Every Wan I2V trainer reached prompt preprocessing and then failed with
  `NameError: name 'ftfy' is not defined` inside Diffusers prompt normalization.
- **Root Cause**: Diffusers imports `ftfy` conditionally and declares it only in development/test
  extras, while its Wan I2V prompt helper calls the package unconditionally. Flow-Factory exposes
  Wan I2V as a core adapter but did not close that runtime dependency gap.
- **Fix**: `ftfy` is now a core project dependency, with a metadata regression that keeps exactly
  one install requirement for Wan prompt normalization.
- **Lesson**: A framework that promotes an upstream optional code path to a supported core feature
  also owns the path's transitive runtime dependencies; successful import of the upstream module
  does not prove its conditionally imported helpers are callable.
- **Related Constraint**: N/A

### Component runtime enumeration boundaries
- **Date**: 2026-08-10
- **Symptom**: Lazy stage-wide operations could materialize non-module specs, Bagel's nested
  transformer could be moved twice, and trainers bypassed adapter lifecycle overrides.
- **Root Cause**: The first runtime abstraction conflated declared specs, materialized modules,
  aliases, and prepared/replacement overrides under one component-name path.
- **Fix**: Split declared and materialized discovery, added non-enumerated pseudo aliases and
  generic device-excluded overrides, and restored trainer routing through adapter lifecycle APIs.
- **Lesson**: Discovery for explicit lookup and enumeration for lifecycle operations require
  separate contracts; aliases and overrides must remain addressable without becoming lifecycle
  roots.
- **Related Constraint**: #5.

### Structured trajectory bridge ownership boundaries
- **Date**: 2026-08-10
- **Symptom**: Batch-level state arguments could be forwarded twice, partial active-count
  overrides were rejected, and a plain mapping with structured trajectory data raised an
  incidental attribute error.
- **Root Cause**: The legacy bridge did not separate bridge-owned forward arguments from
  batch conditioning, and treated optional component metadata as a complete mapping.
- **Fix**: The bridge now strips state-owned batch keys, accepts ordered partial active-count
  overrides while rejecting unknown components, and validates the structured batch type before
  accessing batch metadata.
- **Lesson**: Bridge-owned values must have one authoritative source; optional component
  metadata should be consumed in authoritative component order without requiring every key.
- **Related Constraint**: #5, #26.

## Cross-refs

- UP: [Fix Pattern Router](README.md), [Hard Constraints](../constraints.md)
- WORKFLOW: [`ff-debug`](../../skills/ff-debug/SKILL.md)
