# Data Fix Patterns

**Read when**: Debugging or changing data behavior.

### Preprocessing cache identity must be explicit
- **Date**: 2026-08-11
- **Symptom**: Ref2VA could skip ordered-reference decoding or reuse a merged cache after source/geometry/helper changes.
- **Root Cause**: The real adapter lacked its capability flag, merged cache paths omitted source bytes, and reflective signature filtering could not see semantic fields hidden by `**kwargs`.
- **Fix**: Ref2VA opts into ordered references; cache paths hash dataset root, source bytes, and adapter versions; adapters declare hidden semantic fields via `preprocess_cache_fields`.
- **Lesson**: Reflection is only a default. Dynamic preprocessors need an explicit cache contract.
- **Related Constraint**: N/A

### Optional conditions require row-preserving empty sentinels
- **Date**: 2026-08-28
- **Symptom**: Legal batches mixing omitted and present negative prompts or condition images could reach tokenizers as `None`, lose their outer batch interpretation, or fail in an image encoder on an empty list.
- **Root Cause**: Optionality was validated per record, but preprocessing and collation lacked homogeneous representations for a missing value inside a mixed batch.
- **Fix**: Mixed optional negative prompts project missing values as empty strings; `is_multi_image_batch()` recognizes empty per-sample lists; Flux2/Klein emit aligned `None` latent/ID slots; and Bagel preserves empty condition-image slots. Arrow and offline-collator regressions cover the complete path.
- **Lesson**: A batch-level optional field still needs one explicit slot per row. Normalize at the preprocessing boundary while retaining the original record identity for provenance.
- **Related Constraint**: N/A

### Offline media bytes belong to the exact data identity
- **Date**: 2026-08-28
- **Symptom**: Replacing target, chosen, or rejected media in place left the same path-based record digest, so exact-resume preflight accepted a run whose next on-the-fly VAE inputs had changed. Input media replacement could likewise reuse stale condition embeddings.
- **Root Cause**: Offline identities included normalized media type, path, and rate metadata but not file content.
- **Fix**: `data_utils/offline_dataset.py` streams each unique normalized media path through SHA-256 once per source build. Input digests participate in condition IDs and cache fingerprints; supervision digests participate in full record IDs. No media, decoded pixels, or VAE latents are copied or cached.
- **Lesson**: A path is provenance, not immutable content. Exact future-data contracts and preprocessing caches must identify external file bytes when those bytes are decoded on demand.
- **Related Constraint**: #18

### Sparse media arguments require semantic input slots
- **Date**: 2026-08-29
- **Symptom**: A last-frame-only MiniMax H3 record could not be represented without pretending its
  image was the first frame, and heterogeneous Ref2VA cardinality rules could not be expressed by
  independent per-type counts.
- **Root Cause**: The public offline schema and pipeline contract treated media position and
  per-modality cardinality as the complete binding model.
- **Fix**: V2 input media now accepts an input-only semantic `slot`; contracts declare ordered and
  required slots plus aggregate count/required-any-type rules; projection resolves explicit slots
  first and fills remaining slots positionally. Outputs reject slots. Construction rejects
  multi-slot rules that claim order-insensitivity and aggregate bounds that cannot satisfy their
  per-type rules.
- **Lesson**: Use generic argument-binding metadata for sparse conditions and keep model-specific
  argument names in adapter contracts, not algorithm code or ad-hoc dataset columns.
- **Related Constraint**: #5

### Wan endpoint cardinality follows the checkpoint embedding path
- **Date**: 2026-08-30
- **Symptom**: The Wan I2V adapter advertised one optional last frame for every checkpoint even
  though standard Wan2.1 I2V and dedicated FLF2V weights require different exact image counts.
- **Root Cause**: The effective pipeline contract specialized only expanded-timestep checkpoints
  and ignored whether the loaded transformer used no CLIP image states, ordinary CLIP states, or
  learned first/last endpoint positional embeddings.
- **Fix**: Wan now resolves exact-one for expanded or ordinary CLIP-conditioned checkpoints,
  exact-two for endpoint-positioned FLF2V weights, and preserves one-or-two for Wan2.2's VAE-only
  condition path. The public FLF smoke profile independently requires both endpoint slots.
- **Lesson**: A shared adapter's public superset contract must be narrowed from realized checkpoint
  structure before dataset validation, cache identity, or exact-resume identity is derived.
- **Related Constraint**: #5

### Dataset media discriminators must survive projections unchanged
- **Date**: 2026-08-31
- **Symptom**: MiniMax H3 Ref2VA examples used a different media discriminator from strict V2
  records, and offline condition projection translated between the two representations before
  adapter preprocessing.
- **Root Cause**: Ordered-reference support introduced a private compatibility representation
  instead of preserving the public `MediaAsset.type` contract across canonicalization, decoding,
  cache projection, and adapter dispatch.
- **Fix**: Online Ref2VA manifests, canonical reference sidecars, decoded entries, offline
  projection, and MiniMax H3 dispatch now use `type` end to end. The condition-source and H3
  preprocessing cache versions were advanced so incompatible Arrow caches are rebuilt, and the
  dataset guide, fixtures, and contract tests follow the same schema.
- **Lesson**: A projection may change storage shape, such as list-of-struct to canonical JSON, but
  it should not rename semantic fields. Keep the public discriminator stable until the concrete
  third-party object-construction boundary and version every cache that stores the old shape.
- **Related Constraint**: #5

## Cross-refs

- UP: [Fix Pattern Router](README.md), [Hard Constraints](../constraints.md)
- WORKFLOW: [`ff-debug`](../../skills/ff-debug/SKILL.md)
