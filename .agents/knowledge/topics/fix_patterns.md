# Fix Patterns

**Read when**: After completing a bug fix.

---

This document defines the recording template and archival rules for fix experiences.

## Fix Entry Template

Each fix record uses the following format:

```markdown
### [Short Title]
- **Date**: YYYY-MM-DD
- **Symptom**: What the user observed (error message / abnormal behavior)
- **Root Cause**: Root cause analysis (one sentence)
- **Fix**: What was changed (files involved and key modifications)
- **Lesson**: Implications for future development (why this happened, how to prevent it)
- **Related Constraint**: If a new hard constraint was created, reference the constraint number (N/A if none)
```

## Archival Location Decision Table

Based on the fix type, write the fix entry to the appropriate document:

| Fix Type | Archival Location | Example |
|----------|------------------|---------|
| Violated an existing constraint | `constraints.md` — add "common violation case" under the relevant entry | Forgot to update registry path |
| Discovered a new hard constraint | `constraints.md` — new entry | Found ZeRO-2 + EMA incompatibility |
| Architecture / data-flow misunderstanding | `architecture.md` — relevant module section | Misunderstood preprocess_func call timing |
| Subsystem-specific pitfall | `topics/<topic>.md` — corresponding topic | Sampler boundary condition |
| Does not fit any of the above | This document's "Recorded Fix Patterns" section below | Append as a new record |

**Decision flow**: Check whether the fix matches the first four rows; if none match, fall back to this document.

## Recorded Fix Patterns

<!-- This section accumulates over time. Append new records at the end using the template above. -->

### Multi-modal batch homogeneity (R6)
- **Date**: 2026-04
- **Symptom**: Silent HF `Dataset.map` errors and inconsistent per-sample types in the `audios` column (sometimes `None`, sometimes `Tensor`, sometimes `List[Tensor]`); image/video columns had a latent batch-length mismatch when a sample contributed zero items.
- **Root Cause**: `_preprocess_batch` returned a mix of `None`, `Tensor`, and `List[Tensor]` for the same modality column, breaking Arrow's homogeneous-column requirement and forcing every downstream consumer to handle three input shapes.
- **Fix**: `data_utils/dataset.py:_preprocess_batch` now always emits `List[List[Media]]` per modality (`[]` for empty samples, `[item]` for single-item samples, multi as-is) and appends to BOTH `xx_args[xx]` and `batch[xx]` for every sample so the columns stay length-aligned. Mirrored the same shape on `models/abc.py:preprocess_func` (`audios` parameter) and `utils/audio.py` (`MultiAudioBatch` type alias).
- **Lesson**: HF Arrow demands homogeneous columns, and downstream consumers benefit from a single canonical type. When a column has variable cardinality per row, always represent it as `List[...]` even when the row is empty or has exactly one element. Never special-case "single item" by unwrapping.
- **Related Constraint**: N/A (codified in `topics/adapter_conventions.md` Gotcha #6 and the new "Multi-media batch homogeneity" bullet under Batch Dimension Convention).

### Non-abstract encoder defaults (R7)
- **Date**: 2026-04
- **Symptom**: Adding `encode_audio` as `@abstractmethod` on `BaseAdapter` would force one-line `pass` stubs on 11 existing concrete adapters, none of which consume audio. The first iteration of R6 actually shipped this — and the resulting "noise" diff dwarfed the real change.
- **Root Cause**: Incorrect default-discoverability assumption — abstract methods force every subclass to acknowledge a feature, even when the subclass doesn't use it.
- **Fix**: `models/abc.py` dropped `@abstractmethod` from all 4 encoders (`encode_prompt`, `encode_image`, `encode_video`, `encode_audio`); default body is `pass` returning `None`; `preprocess_func` skips integration when the called encoder returns `None`. The Round-6 stub overrides on 11 concrete adapters were reverted, leaving them byte-identical to `origin/main`.
- **Lesson**: When extending a base contract for a partial-coverage feature (where only some subclasses will participate), no-op default + opt-in override beats forcing every subclass to acknowledge it. Reserve `@abstractmethod` for invariants that ALL subclasses must implement (e.g. `load_pipeline`, `decode_latents`, `forward`, `inference`).
- **Related Constraint**: #12 (post-update text codifies "Optional encoder overrides (no-op default)").

### Distributed reward gathering must preserve reconstruction invariants
- **Date**: 2026-08-30
- **Symptom**: Two-rank H3 Ref2VA GRPO failed before reward execution because `MiniMaxH3Ref2VASample` was reconstructed with `reference_manifest=None`.
- **Root Cause**: The distributed group-reward path gathered only reward-consumed fields, although `gather_samples` reconstructs the concrete sample class and that class can require additional state.
- **Fix**: `BaseSample` now declares an empty `reconstruction_required_fields` contract,
  `OrderedReferenceConditionSample` adds `reference_manifest`, and `gather_samples` automatically
  unions that contract at the concrete reconstruction boundary. Reward callers therefore transport
  constructor state without forwarding it to the reward call or duplicating gather policy.
- **Lesson**: Communication payload requirements and reward-call requirements are distinct contracts; partial gathers must preserve constructor invariants even for fields that downstream computation does not consume.
- **Related Constraint**: N/A

### Preference-arm conditioning ownership
- **Date**: 2026-08-11
- **Symptom**: DPO evaluated the rejected H3 state with the chosen sample's prompt/reference conditioning.
- **Root Cause**: Shared-noise logic was incorrectly extended to the conditioning batch, even though only the forward-process noise is shared between arms.
- **Fix**: Rejected policy/reference forwards now receive `rejected_batch`, with a production-path regression using distinct prompt embeddings.
- **Lesson**: Preference arms may share stochastic coordinates but never model conditioning; replay state and conditioning must come from the same sample.
- **Related Constraint**: #7

### Dynamics and velocity conventions belong to adapters/schedulers
- **Date**: 2026-08-11
- **Symptom**: GRPO-Guard applied Flow-SDE `sqrt(-dt)` scaling to CPS, while NFT and OPD reconstructed H3 `x0` with the standard flow sign.
- **Root Cause**: Trainer-local formulas assumed one dynamics type and one velocity direction.
- **Fix**: Coupled trainers share a dynamics-aware transition-scale helper that rejects zero/non-finite variance, and `x0` projection routes through `adapter.project_velocity_to_clean_state()` using model-compute dtype.
- **Lesson**: Trainers should consume declared process semantics rather than infer them from historical single-model formulas.
- **Related Constraint**: #7

### Lazy components must re-enter lifecycle policy
- **Date**: 2026-08-11
- **Symptom**: Modular preprocessing modules materialized after adapter initialization missed explicit frozen dtype casting and the FSDP synchronization point.
- **Root Cause**: `_mix_precision()` and frozen synchronization enumerated only modules materialized at construction time.
- **Fix**: `on_load_components()` applies precision policy to newly materialized modules; preprocessing and inference synchronize explicit module parameters/buffers before use while skipping tokenizers/processors.
- **Lesson**: Laziness changes timing, not ownership or lifecycle obligations.
- **Related Constraint**: #19, #20

### Preprocessing cache identity must be explicit
- **Date**: 2026-08-11
- **Symptom**: Ref2VA could skip ordered-reference decoding or reuse a merged cache after source/geometry/helper changes.
- **Root Cause**: The real adapter lacked its capability flag, merged cache paths omitted source bytes, and reflective signature filtering could not see semantic fields hidden by `**kwargs`.
- **Fix**: Ref2VA opts into ordered references; cache paths hash dataset root, source bytes, and adapter versions; adapters declare hidden semantic fields via `preprocess_cache_fields`.
- **Lesson**: Reflection is only a default. Dynamic preprocessors need an explicit cache contract.
- **Related Constraint**: N/A

### Preserve contract-locked terminology during architecture rewrites
- **Date**: 2026-08-28
- **Symptom**: The SenseNova documentation regression test failed after the architecture table retained the correct behavior but dropped the contract phrase `ordered variable-count references`.
- **Root Cause**: A broad documentation rewrite paraphrased a tested semantic distinction without checking the existing documentation contract.
- **Fix**: `.agents/knowledge/architecture.md` now restores the exact phrase while retaining the grouped `images` and non-NaViT execution details; the documentation suite verifies both distinctions.
- **Lesson**: Search documentation tests before rewriting architecture terminology, especially where wording distinguishes adapters with superficially similar multi-reference inputs.
- **Related Constraint**: N/A

### Distributed exact-state phases need synchronized failure and one publisher
- **Date**: 2026-08-28
- **Symptom**: One rank could fail exact-resume RNG/hash preflight while a peer entered `Accelerator.load_state()` and hung; concurrent saves could also pass destination preflight together and write the same staging directory.
- **Root Cause**: Rank-local filesystem work was followed by raw barriers or core mutation without first gathering errors, and publication ownership was checked non-atomically before artifact creation.
- **Fix**: `trainers/abc.py` now gathers all-rank errors after barrier-free path resolution, runtime preflight, and core load; commits runtime progress only after every core load succeeds; atomically elects the global publisher before core save; and keeps every filesystem claim until all publishers install their final directory. `trainers/common/runtime_identity.py` hashes FSDP wrap/state-dict topology and the full DeepSpeed batch/accumulation plan, while `runtime_state.py` validates per-device RNG topology and state installability. Runtime manifests are written only after Accelerator artifacts finish. Targeted multi-rank failure simulations cover preflight, load, and publisher races.
- **Lesson**: A raw barrier is unsafe after rank-local I/O that can raise. Structure distributed checkpoints as preflight → synchronized error gather → backend mutation → synchronized error gather → manifest → atomic publication, acquire the publication claim before any process writes shared staging, and retain it until every filesystem reports success. Exact resume identity must cover backend topology, not only parameter names and shapes.
- **Related Constraint**: #18

### Validate nested media cardinality before flattening
- **Date**: 2026-08-28
- **Symptom**: Valid offline Flux1-Kontext batches shaped as `List[List[PIL]]` raised `TypeError` while checking whether a sample carried multiple condition images.
- **Root Cause**: `_standardize_image_input()` flattened the nested batch before running its per-sample cardinality check, so the check called `len()` on each PIL image.
- **Fix**: Flux1-Kontext now checks and warns on the original nested batch before selecting the first image, with a regression covering two offline single-image rows.
- **Lesson**: Perform shape and cardinality validation at the boundary where that structure still exists; flattening destroys the evidence needed to validate it safely.
- **Related Constraint**: N/A

### Optional conditions require row-preserving empty sentinels
- **Date**: 2026-08-28
- **Symptom**: Legal batches mixing omitted and present negative prompts or condition images could reach tokenizers as `None`, lose their outer batch interpretation, or fail in an image encoder on an empty list.
- **Root Cause**: Optionality was validated per record, but preprocessing and collation lacked homogeneous representations for a missing value inside a mixed batch.
- **Fix**: Mixed optional negative prompts project missing values as empty strings; `is_multi_image_batch()` recognizes empty per-sample lists; Flux2/Klein emit aligned `None` latent/ID slots; and Bagel preserves empty condition-image slots. Arrow and offline-collator regressions cover the complete path.
- **Lesson**: A batch-level optional field still needs one explicit slot per row. Normalize at the preprocessing boundary while retaining the original record identity for provenance.
- **Related Constraint**: N/A

### Preprocessing cache identity includes precision policy
- **Date**: 2026-08-28
- **Symptom**: Changing `component_load_dtypes` or `frozen_parameters_dtype` could reuse condition embeddings computed under a different component precision policy.
- **Root Cause**: The offline cache fingerprint named the model but omitted the precision configuration that controls preprocessing component loading and storage.
- **Fix**: Offline condition-cache extras now include a sorted canonical JSON representation of both dtype policies, with regressions for load-policy changes, frozen-policy changes, and mapping-order stability.
- **Lesson**: Cache identity must include every configuration value that can change preprocessing numerics, even when that value is enforced during model loading rather than passed to the preprocessing function.
- **Related Constraint**: #20

### Batched Arrow schemas must cover later optional values
- **Date**: 2026-08-28
- **Symptom**: A valid optional-image dataset ordered as two prompt-only rows followed by two image-conditioned rows failed in the second `Dataset.map` chunk while casting image bytes or latent tensors to the first chunk's empty/string schema.
- **Root Cause**: HuggingFace fixed the writer schema from the all-empty first output chunk, while Flux2 also omitted its image output keys whenever that individual chunk contained no images.
- **Fix**: `data_utils/dataset.py` now scans real map-sized slices, dropping resolved columns from the bounded scan until it finds each later typed representative chunk; it probes those chunks only when a source column transitions from empty in the first chunk to typed later, restores Python/NumPy/torch/MPS and explicit-generator RNG state after that schema-only probe, and passes the resulting explicit `Features` through the `datasets==3.3.2`-compatible map surface. Flux2 now emits aligned image columns whenever the source image field exists; Flux2 and Flux2-Klein regressions verify identical empty-chunk output structure, a spy rejects whole-column reads, and a four-row offline cache regression covers the real Arrow path.
- **Lesson**: A batched map's output key set and feature types are dataset-level contracts, not properties of whichever values happen to appear in the current chunk. Optional adapters must emit empty row slots consistently, and data writers must derive schemas from representative typed values before committing the first Arrow batch. A schema probe is intentionally narrow and restores RNG, but preprocessors still own their normal deterministic/cache-safe behavior; arbitrary adapter-owned mutable state is not rollback-safe.
- **Related Constraint**: N/A

### Exact resume must lock future execution and preserve acquisition boundaries
- **Date**: 2026-08-28
- **Symptom**: An exact checkpoint could pass preflight after objective, seed, scheduler, ordered training-data, or replayed evaluation semantics changed; a remote-rank runtime-child failure could let peers return from resume; an online resume retried its immutable source `checkpoint-N`; offline save-before-eval captured RNG that did not match the next uninterrupted epoch; and MPS could publish an exact checkpoint that its loader would always reject.
- **Root Cause**: Resume identity stopped at physical model/optimizer/backend structure and treated RNG-consuming evaluation as operational, runtime-child commit was outside the synchronized distributed phase, one generic checkpoint/evaluation order ignored the different replay boundaries of generated versus finite-dataset acquisition, and save preflight did not reject a device RNG unsupported by Accelerate.
- **Fix**: Runtime identity now includes trainer-extensible execution and rank-free data-contract digests derived from resolved objective/forward settings, realized training loader provenance/order/geometry, and the ordered realized evaluation path (cadence, arguments, dataset overrides, rewards, and prepared loaders). Runtime-child commit and attachment use a synchronized all-rank error phase. The first online boundary skips a save only when its resolved real path equals the exact-resume source, while still evaluating; offline boundaries evaluate before saving so exact checkpoints capture post-evaluation RNG, model-only saves retain the same observable order without claiming RNG restoration, and MPS exact save fails before adapter or filesystem mutation.
- **Lesson**: Exact resume compatibility covers every computation and ordered data stream that can affect future state, including evaluation replay, not only the state container. Every rank-local resume mutation needs a synchronized failure boundary, checkpoint placement must match whether acquisition resumes before a rollout or after a completed data epoch, and save must not publish a state the matching load path cannot restore.
- **Related Constraint**: #18

### Multi-source schedule seeds must be process-independent
- **Date**: 2026-08-28
- **Symptom**: The same multi-source counts, configured seed, and epoch could produce a different source order on separate ranks or after restart.
- **Root Cause**: `WeightedSourceBatchScheduler` seeded its generator with Python's salted `hash()` over a tuple containing a string, so the result depended on each process's `PYTHONHASHSEED`.
- **Fix**: `data_utils/multi_source.py` now derives a domain-separated unsigned 64-bit seed from SHA-256; a subprocess regression compares schedules under distinct `PYTHONHASHSEED` values.
- **Lesson**: Never use Python object hashes as distributed or persistent RNG seeds. Derive seeds from an explicitly versioned, stable byte representation.
- **Related Constraint**: #9

### Exact resume must include non-param-group optimizer semantics
- **Date**: 2026-08-28
- **Symptom**: An exact checkpoint could pass compatibility preflight after a role's gradient clipping threshold or update frequency changed.
- **Root Cause**: Runtime identity covered realized optimizer parameter groups, but `max_grad_norm` and `update_frequency` are consumed by role optimization outside those groups.
- **Fix**: `trainers/common/runtime_identity.py` now hashes resolved per-role optimizer arguments, including algorithm-provided defaults; runtime-identity regressions verify clipping and cadence drift change the execution digest while operational controls remain mutable.
- **Lesson**: Exact-resume identity must cover every value that controls whether and how an optimizer update occurs, not only values serialized in optimizer parameter groups.
- **Related Constraint**: #18

### Distillation rollout cursors derive from persisted progress
- **Date**: 2026-08-28
- **Symptom**: DMD2, TDM, and TDM-R1 exact resumes restarted prompt acquisition from dataloader epoch zero even though the checkpoint recorded completed rollout iterations.
- **Root Cause**: The trainers retained a live Python iterator and local dataloader epoch counter, while exact runtime state persisted only `TrainingProgress`; infinite grouped samplers never raised `StopIteration`, so the local epoch counter did not describe their real position either.
- **Fix**: `trainers/distillation/distillation_runtime.py` now reconstructs the global consumed-batch cursor as `rollout_iteration * gradient_accumulation_steps`, uses the realized finite loader length before sampler/config fallbacks, maps the cursor to sampler epoch and intra-epoch offset, and restores Python/NumPy/torch CPU/CUDA/MPS plus explicit loader-generator RNG around iterator reconstruction and skips. Regressions compare uninterrupted and resumed real `DataLoader`, infinite grouped-sampler, and finite multi-source sequences.
- **Lesson**: Do not serialize Python iterators or maintain a second checkpoint authority. At legal checkpoint boundaries, derive replayable loader position from persisted progress plus identity-locked realized geometry, and treat iterator construction and replayed skips as RNG-consuming side effects that must be neutralized.
- **Related Constraint**: #18

### Offline media bytes belong to the exact data identity
- **Date**: 2026-08-28
- **Symptom**: Replacing target, chosen, or rejected media in place left the same path-based record digest, so exact-resume preflight accepted a run whose next on-the-fly VAE inputs had changed. Input media replacement could likewise reuse stale condition embeddings.
- **Root Cause**: Offline identities included normalized media type, path, and rate metadata but not file content.
- **Fix**: `data_utils/offline_dataset.py` streams each unique normalized media path through SHA-256 once per source build. Input digests participate in condition IDs and cache fingerprints; supervision digests participate in full record IDs. No media, decoded pixels, or VAE latents are copied or cached.
- **Lesson**: A path is provenance, not immutable content. Exact future-data contracts and preprocessing caches must identify external file bytes when those bytes are decoded on demand.
- **Related Constraint**: #18

### Runtime identity excludes transport-only global source IDs
- **Date**: 2026-08-28
- **Symptom**: Inserting or reordering an eval-only dataset renumbered training sources and caused exact-resume rejection even though the ordered training data and reward mathematics were unchanged.
- **Root Cause**: Data identity hashed the full global name-to-ID registry and offline numeric `source_id`, which are transport metadata assigned across both train and eval entries.
- **Fix**: `trainers/common/runtime_identity.py` now locks ordered realized training source names and loader schemas while excluding global numeric IDs. Structural and real-`Arguments` regressions verify eval-only changes preserve both execution and data digests, while training-source order/count changes remain incompatible.
- **Lesson**: Compatibility identities should include semantic names and order, not remappable integer handles whose only purpose is runtime transport.
- **Related Constraint**: #18

### Offline epoch semantics follow the realized official sampler
- **Date**: 2026-08-28
- **Symptom**: Loader documentation claimed that every global sample appears exactly once and that an offline epoch is never padded, while the intentionally selected official `DistributedSampler(drop_last=False)` repeats tail indices when dataset size is not divisible by world size.
- **Root Cause**: The design correctly defined an epoch as a complete rank-local dataloader traversal, but documentation conflated that with global sample uniqueness and with the separate prohibition on inventing batches to close a gradient-accumulation window.
- **Fix**: Loader, workflow, dataset, sampler, and constraint documentation now preserve PyTorch's standard tail-equalization semantics and state that the framework adds no batches merely for accumulation. Existing uneven-geometry tests continue to lock official sampler behavior.
- **Lesson**: When delegating sharding to an official sampler, define epoch semantics over its realized finite loader. Distinguish sampler-level repeated indices from optimizer-level synthetic padding.
- **Related Constraint**: #9

### Recipe migrations must move tests and documentation with the config
- **Date**: 2026-08-28
- **Symptom**: Rebasing onto the precision-aware loading branch changed the MiniMax H3 T2VA default to the shared `vid_prompt` source, added ImageBind routing, and removed its old unvalidated warning, while the executable example test and user guides still required the prior dataset and wording.
- **Root Cause**: The recipe-only commit updated YAML semantics without treating example assertions, dataset links, dependency notes, and validation-status language as one public workflow contract.
- **Fix**: The T2VA default test now locks the shared TXT manifests and CLAP/ImageBind routing while retaining the dedicated JSONL fixture check for the validated debug recipe. The example and dataset guides now describe the shared source, ImageBind dependency, and the exact evidence boundary without claiming a completed long run.
- **Lesson**: An example configuration is executable documentation. Any recipe migration must update its production parse test, linked data provenance, optional dependency instructions, and validation claims in the same integration change.
- **Related Constraint**: #15

### Optional-kernel adapter tests must lazy-load behind the dependency seam
- **Date**: 2026-08-28
- **Symptom**: Collecting the Bagel TDM contract test on macOS failed before any test ran because `flash-attn>=2.5.8` was unavailable.
- **Root Cause**: The test imported the Bagel adapter at module scope instead of installing the existing fake optional-kernel modules before the adapter import.
- **Fix**: `tests/models/test_bagel_tdm_contracts.py` now lazily imports Bagel after stubbing `flash_attn`, OpenCV, and the availability probes; new Python files also receive the required Apache 2.0 headers.
- **Lesson**: Contract tests for CUDA-only optional adapters must exercise the adapter through its dependency boundary so CPU and macOS collection remains valid; importing such adapters at module scope turns an optional dependency into a repository-wide test dependency.
- **Related Constraint**: N/A

### Distillation exact cursors count rollout batches, not backend work items
- **Date**: 2026-08-29
- **Symptom**: After timestep-aligned TDM accumulation made each trajectory boundary one backend work item, exact resume skipped `num_inference_steps` times too many prompt batches.
- **Root Cause**: Cursor reconstruction still multiplied completed rollout iterations by backend `gradient_accumulation_steps`, even though one rollout now contributes multiple boundary losses to that accumulation window.
- **Fix**: `trainers/distillation/distillation_runtime.py` now derives consumed prompt batches through `resolve_rollout_accumulation_steps()`, and the cursor regression locks `gradient_accumulation_steps=8`, four losses per rollout, and two completed iterations to four consumed batches.
- **Lesson**: Persisted acquisition progress must be projected through the current acquisition-to-backend work-item ratio. Backend GAS is not a valid dataloader cursor when one acquired batch expands into multiple backward graphs.
- **Related Constraint**: #18a

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

### Offline velocity objectives must bypass unused scheduler transitions
- **Date**: 2026-08-29
- **Symptom**: LTX2 near-clean offline targets lost velocity precision after a velocity-to-x0-to-
  velocity round trip, while exact velocity-only Wan/LTX forwards still invoked scheduler steps that
  their loss never consumed. LTX2 I2AV also dropped cached negative prompts during preprocessing.
- **Root Cause**: Generation-oriented forward paths performed transition reconstruction before
  checking the requested offline component, and the I2AV preprocessing override failed to forward
  one base prompt argument.
- **Fix**: LTX2 retains official online reconstruction by default but opts offline forwards into raw
  model velocity; Wan and LTX return exact velocity requests before scheduler stepping; I2AV now
  forwards `negative_prompt` explicitly. Component, parity, and initialization regressions cover
  the split behavior.
- **Lesson**: Offline objectives may share a model forward with generation but must not inherit
  numerically lossy or RNG-consuming transition work that is outside their requested output.
- **Related Constraint**: #7

### Documented dataset tools must use their package invocation mode
- **Date**: 2026-08-30
- **Symptom**: Running the documented `python dataset/offline_smoke/prepare.py ...` command failed before argument parsing with `attempted relative import with no known parent package`.
- **Root Cause**: The package uses relative imports, while the documentation incorrectly advertised direct file execution instead of Python's module mode.
- **Fix**: All offline-smoke commands now use `python -m dataset.offline_smoke.<tool>` with unconditional package imports, and a subprocess regression executes the documented form.
- **Lesson**: A checked-in CLI example is part of the public interface; standardize on one package-aware invocation and test that exact process rather than adding conditional import fallbacks.
- **Related Constraint**: N/A

### Exact resume must lock the checkpoint-realized pipeline contract
- **Date**: 2026-08-29
- **Symptom**: Exact resume could accept a checkpoint after an in-place model configuration change
  switched a Wan adapter between first-only and first/last-frame semantics.
- **Root Cause**: Runtime identity hashed model arguments and trainer execution but omitted the
  adapter's resolved `effective_pipeline_io_contract`.
- **Fix**: The default execution identity now canonicalizes and hashes the realized pipeline I/O
  contract after adapter initialization; a regression changes only that contract and observes only
  the execution digest change.
- **Lesson**: Any checkpoint-dependent specialization that changes legal inputs or forward binding
  is future-execution state and belongs in exact-resume identity.
- **Related Constraint**: #18

### Offline condition caches need contract-stable schemas across sources
- **Date**: 2026-08-29
- **Symptom**: Changing only semantic slot order could reuse an Arrow cache with the old media
  projection, while a multi-source batch could fail because an all-empty optional source omitted
  columns that a populated source emitted.
- **Root Cause**: The source hash covered record identities but not the effective input projection
  contract, and projection decided column existence from each source's observed values.
- **Fix**: Condition source identity now includes the canonical effective input contract. With a
  contract, negative-prompt, declared media, and semantic-slot columns are projected consistently
  even when every row in one source is empty. A real two-source `DistributedSampler` loader
  regression mixes empty and populated optional conditions in one batch.
- **Lesson**: A concatenated cache schema is defined by the model contract, not by local source
  sparsity; cache identity must cover every declaration that can reorder or reshape projection.
- **Related Constraint**: #9

### Distributed Arrow schemas must be inferred before rank sharding
- **Date**: 2026-08-29
- **Symptom**: A distributed condition-cache build could write `List(null)` on an all-empty rank
  and `List(Image)` on a populated rank, then fail when the per-rank Arrow files were consolidated.
- **Root Cause**: Cross-chunk schema discovery ran after rank sharding, so each process inferred
  features from only its local value distribution. The standalone cache entry point also ignored a
  pipeline's single-sample preprocessing capability.
- **Fix**: Distributed preprocessing now derives one explicit feature schema from the full source
  before selecting rank-local rows, and both cache entry points force batch size one for ordered or
  `SINGLE_SAMPLE` contracts. A real two-part Arrow regression separates empty and populated rows
  across ranks, consolidates the files, and loads the merged dataset.
- **Lesson**: Distributed writers need a global serialization contract even when their data is
  disjoint. Batch capability is likewise part of the preprocessing boundary, not only trainer
  orchestration.
- **Related Constraint**: #9

### Validate output candidates before stochastic condition preparation
- **Date**: 2026-08-29
- **Symptom**: An invalid offline target correctly raised an exception but first consumed condition
  preparation RNG, so retrying with corrected media no longer reproduced the original encoding.
- **Root Cause**: `BaseAdapter.encode_output_state()` prepared raw conditions before validating the
  generator and exact output-media sequence.
- **Fix**: The lifecycle wrapper now validates generator type and candidate media before invoking
  any condition preparer or codec. A stochastic-preparer regression proves invalid media leaves the
  explicit generator unchanged and performs no preparation work.
- **Lesson**: Pure boundary validation must precede expensive or random transformations. Failed
  inputs should not mutate the state that determines a later valid retry.
- **Related Constraint**: #7

### Aggregate media guarantees must be canonical contract state
- **Date**: 2026-08-29
- **Symptom**: `INPUT_MEDIA` geometry rejected a valid contract whose aggregate `min_total_count` or
  `required_any_types` guaranteed a condition, while semantically identical required-type tuples
  in different orders produced different cache and resume identities.
- **Root Cause**: Geometry validation recognized only per-type minima, and the set-like aggregate
  field had no canonical ordering rule.
- **Fix**: Input-derived geometry now accepts every nonempty guarantee enforced by runtime
  validation, `required_any_types` must follow canonical media-type order, and required slots must
  follow their declaration order.
- **Lesson**: Declarative invariants should be interpreted consistently at construction and runtime,
  and set-like identity fields require one canonical representation.
- **Related Constraint**: #5

### ZeRO optimizer identity must use logical model groups
- **Date**: 2026-08-30
- **Symptom**: Every ZeRO-2 trainer failed during initialization because its optimizer schema
  contained a parameter not owned by the rebound component-variant registry.
- **Root Cause**: DeepSpeed ZeRO-1/2 replaces each public optimizer group with a rank-local flat
  FP32 master partition, while runtime identity incorrectly treated those partitions as the live
  model parameters owned by the registry.
- **Fix**: Runtime identity now maps stable parameter ownership through DeepSpeed's retained
  `bit16_groups` and continues to serialize settings from the public optimizer groups. It fails
  closed if logical groups are absent or do not match the partitioned group count.
- **Lesson**: A distributed optimizer's public parameter groups may be physical state partitions;
  exact-resume identity must separate logical model ownership from physical group settings.
- **Related Constraint**: #18a

### Custom Transformers models must implement their declared checkpointing seam
- **Date**: 2026-08-30
- **Symptom**: Every Bagel trainer failed during initialization when full gradient checkpointing
  called `Qwen2ForCausalLM.gradient_checkpointing_enable()` and Transformers reported that the
  architecture was incompatible.
- **Root Cause**: Bagel's custom Qwen2-NaViT model inherited
  `supports_gradient_checkpointing = True` but did not expose a `gradient_checkpointing` state or
  invoke the installed checkpoint function in its active decoder loop.
- **Fix**: The custom Qwen2 model now owns the standard checkpointing flag and routes each pure
  decoder layer through Transformers' installed non-reentrant checkpoint function while training.
  Cache-updating and TaylorSeer paths remain direct to avoid replaying mutations. A backward
  regression proves the layer is recomputed rather than merely accepting the API call.
- **Lesson**: A custom `PreTrainedModel` must pair its capability declaration with both the state
  seam expected by Transformers and an execution-time checkpoint boundary in the forward path.
- **Related Constraint**: #7

### FSDP wrap metadata must follow the instantiated architecture variant
- **Date**: 2026-08-30
- **Symptom**: Every Bagel FSDP2 trainer failed during `accelerator.prepare()` because Accelerate
  could not find the declared `Qwen2DecoderLayer` in the loaded model.
- **Root Cause**: Bagel's custom Qwen2 classes inherited fixed `_no_split_modules` metadata from the
  standard decoder even though `config.layer_module` instantiated a MoE or MoT decoder variant.
- **Fix**: Both the inner Qwen2 model and outer causal LM now derive their no-split class from the
  realized `layer_module`. CPU FSDP2 auto-wrap regressions verify all decoder variants resolve
  through the same PEFT wrapper used by LoRA training.
- **Lesson**: Distributed wrap metadata is realized model state. Config-selectable architectures
  must not inherit a fixed block class that may be absent from the instantiated module tree.
- **Related Constraint**: #9

### Parameter-sharded submodule work must remain inside the prepared root forward
- **Date**: 2026-08-30
- **Symptom**: Every Bagel FSDP2 trainer reached sampling or offline replay but failed at token
  embedding with a mixed `torch.Tensor` and `DTensor` operator error.
- **Root Cause**: Bagel's cache helpers and denoising path reached through the physical pipeline to
  `language_model.model.embed_tokens` before calling the routed transformer. Decoder layers owned
  nested FSDP groups, but embedding and final normalization belonged to the prepared `ModelBundle`
  root, whose unshard hook was bypassed by those direct calls.
- **Fix**: The outer Qwen forward now accepts raw packed token IDs and inserts their embeddings into
  an optional query-local auxiliary sequence. Bagel cache helpers accept an injected language-model
  forward, and the adapter supplies its routed transformer for text, VAE, ViT, denoising, and CFG
  passes. Each logical language-model pass therefore performs embedding, decoder execution, and
  final normalization inside one prepared-root call.
- **Lesson**: Under compositional FSDP, wrapping transformer blocks does not make arbitrary child
  access safe. Any computation using parameters owned by the prepared root must execute beneath
  that root's forward hooks; converting ordinary inputs to `DTensor` or manually unsharding only
  masks the first failure and breaks distributed lifecycle semantics.
- **Related Constraint**: #9

### FSDP2 activation checkpoints must replay inside the mixed-precision boundary
- **Date**: 2026-08-30
- **Symptom**: All four Wan FSDP2 trainers failed on their first backward because checkpointed
  tensors were saved as BF16 but recomputed as FP32.
- **Root Cause**: Model-level checkpointing captured FP32 block inputs before FSDP2's forward-input
  cast, while backward replay re-entered a block in `PRE_BACKWARD` state where PyTorch deliberately
  skips that cast.
- **Fix**: Before loading the model, the trainer resolves one checkpoint owner. Any FSDP2 full
  model policy is normalized to Accelerate's backend checkpoint wrappers, even when backend
  checkpointing was initially disabled, because those wrappers replay inside the fully-sharded
  mixed-precision boundary. Every selective FSDP2 model policy fails closed because its boundary
  remains outside the input cast. FSDP1 retains its existing owner, and direct trainer construction
  defensively reuses the same resolver after model realization. Wan FSDP2 GRPO, TDM, SFT, and
  offline DPO plus SD3.5 and Bagel regressions verify the shared path.
- **Lesson**: Checkpoint placement is part of distributed precision semantics. A recompute boundary
  outside a sharded module may not replay its forward hooks, so backend-aligned checkpoint wrappers
  must own FSDP2 full checkpointing instead of nesting model-level boundaries around sharded blocks.
- **Related Constraint**: #9, #20

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

### Absent optional components need distinct physical roots
- **Date**: 2026-08-30
- **Symptom**: Every Wan2.2 TI2V trainer rejected its load plan because the physical
  `image_encoder` root appeared to combine incompatible auxiliary and host roles.
- **Root Cause**: The eager runtime used object identity to collapse logical aliases, so multiple
  optional components whose value was the singleton `None` were mistaken for one shared physical
  object.
- **Fix**: Classic pipelines now preserve a declared optional `None` under its own logical root
  while retaining identity aliasing for real objects. The load coordinator finalizes only
  replicated roots that actually materialized, preventing FSDP replica checks from resolving an
  allowed absent component.
- **Lesson**: Object identity establishes physical aliasing only for materialized objects. Optional
  declarations retain distinct lifecycle identities, and backend finalization must follow observed
  materialization rather than the requested name set.
- **Related Constraint**: #9

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

### Condition-latent geometry follows the encoded source, not the rollout target
- **Date**: 2026-08-30
- **Symptom**: Wan2.2 TI2V rejected a one-frame VAE condition latent with temporal size one because
  the rollout noise and target video had temporal latent size two.
- **Root Cause**: The condition validator derived its expected temporal size from configured output
  frames. In the expanded-timestep pipeline, Diffusers intentionally encodes only the first input
  frame and broadcasts that one-frame condition through a full-length first-frame mask.
- **Fix**: Wan condition validation now derives temporal geometry from the actual video tensor sent
  to the VAE. The regression models one-frame encoding and proves the official broadcast produces
  the full rollout shape while non-expanded conditions keep their full temporal encoding.
- **Lesson**: Conditioning and generated states can share channels and spatial geometry without
  sharing sequence length. Validate each representation against its own source transform before
  relying on an explicitly defined broadcast or mask contract.
- **Related Constraint**: #7

### Internal immutable media rows must cross sample boundaries as public batch types
- **Date**: 2026-08-30
- **Symptom**: Wan I2V rollouts finished denoising but failed while constructing each sample because
  image canonicalization rejected a tuple of ordered first/last frames.
- **Root Cause**: Wan's internal condition normalizer intentionally returns immutable tuple rows,
  while `ImageConditionSample.condition_images` follows the public `ImageBatch` contract of lists,
  tensors, or arrays. The adapter passed the internal representation across that boundary unchanged.
- **Fix**: Wan now converts each ordered condition row to a list at sample construction. The
  regression proves first/last color order survives sample canonicalization and replay stacking.
- **Lesson**: Model-internal containers may enforce stronger invariants than shared sample APIs, but
  adapters must translate them explicitly at the ownership boundary instead of broadening a common
  media utility for one private representation.
- **Related Constraint**: #5

### Endpoint-conditioned checkpoints are not interchangeable with I2V checkpoints
- **Date**: 2026-08-30
- **Symptom**: Every Wan first/last-frame smoke reached the transformer but failed while concatenating
  image and text states because their leading dimensions were two and one.
- **Root Cause**: The FLF2V matrix profile selected a standard Wan2.1 I2V checkpoint, whose image
  projection lacks the learned endpoint positional embedding that folds two ordered CLIP image rows
  back into one logical sample.
- **Fix**: The public smoke profile and GPU validation plan now select the dedicated Wan2.1 FLF2V
  checkpoint, and the supported-model table documents that checkpoint explicitly.
- **Lesson**: Checkpoints that share a Diffusers pipeline class can still implement distinct
  conditioning contracts. Validation profiles must bind semantic modes to weights whose trained
  embedding layout realizes that mode instead of relying only on adapter-class compatibility.
- **Related Constraint**: N/A

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

### Ordered-reference preprocessors need the canonical manifest sidecar
- **Date**: 2026-08-30
- **Symptom**: MiniMax H3 Ref2VA failed during distributed dataset preprocessing because its
  strict workflow received decoded `references` but `reference_manifest=None`.
- **Root Cause**: `GeneralDataset` canonicalized and retained each ordered-reference manifest for
  Arrow output, but omitted that same manifest from the arguments passed to the adapter
  preprocessor.
- **Fix**: Ordered-reference preprocessing now forwards the canonical manifest beside the decoded
  transient media, and the real-media round-trip regression requires the preprocessor to receive
  the same canonical ordering that is stored in the cache.
- **Lesson**: Identity and reconstruction sidecars must cross the same preprocessing boundary as
  the transient inputs they describe. Preserve strict batch validation at the adapter instead of
  delaying a missing-sidecar failure until sample construction.
- **Related Constraint**: N/A

### Token-local H3 feed-forward work should bound packed-sequence temporaries
- **Date**: 2026-08-30
- **Symptom**: MiniMax H3 Ref2VA ZeRO-2 offline DPO and FSDP2 trainers exhausted a 95 GiB
  device while allocating one 380 MiB SwiGLU activation for a 13,889-token packed sequence.
- **Root Cause**: Diffusers' H3 blocks evaluated every feed-forward projection over the complete
  dynamic sequence, and its generic chunk helper rejected non-divisible lengths such as 13,889.
- **Fix**: H3 runtime setup now reuses each existing `ff.net` parameter tree inside a remainder-safe
  1,024-token executor for both token-refiner and main transformer blocks. Installation precedes
  Flow-Factory resume loading, LoRA, checkpointing, and distributed wrapping, so state-dict keys,
  parameter identities, and execution policy remain consistent across rollout and training.
- **Lesson**: A token-local operation does not need to inherit the peak allocation of a packed
  attention sequence. Apply memory bounds at the operation boundary while preserving the original
  parameter tree and accepting dynamic tail chunks.
- **Related Constraint**: N/A

### Head-space normalization should not upcast an entire packed sequence
- **Date**: 2026-08-30
- **Symptom**: MiniMax H3 Ref2VA FSDP2 GRPO exhausted a 95 GiB device while
  `attn.norm_q` requested one 380 MiB allocation before reaching the feed-forward layer.
- **Root Cause**: PyTorch RMSNorm promoted the complete `[1, 13889, 56, 128]` BF16 query to a
  379.8 MiB FP32 temporary even though normalization reduces only the final head dimension.
- **Fix**: H3 runtime setup now reuses each Q/K RMSNorm parameter inside a remainder-safe
  1,024-token executor on both repeated block stacks. Parameter identities and
  `attn.norm_q.weight` / `attn.norm_k.weight` state-dict paths remain unchanged.
- **Lesson**: Row-local normalization can preserve its exact reduction semantics while bounding
  the number of independent rows promoted at once. Chunk the non-reduced sequence dimension,
  never the normalized head dimension.
- **Related Constraint**: N/A

### Outer activation checkpoint replay can retain every inner token chunk
- **Date**: 2026-08-30
- **Symptom**: MiniMax H3 Ref2VA ZeRO-2 offline DPO reached backward replay but exhausted
  a 95 GiB device when the final feed-forward chunk requested a 28 MiB SiLU allocation.
- **Root Cause**: The block-level non-reentrant checkpoint replay rebuilt one autograd graph
  containing the intermediates from every sequential feed-forward chunk. Smaller chunks reduced
  each allocation but did not bound their cumulative saved activations.
- **Fix**: Long, grad-enabled H3 feed-forward chunks now use nested non-reentrant activation
  checkpoints. Each chunk is replayed independently during backward; no-grad rollout and short
  sequences retain their direct execution paths. Checkpoint replay preserves RNG state so later
  parameter-efficient adapters cannot silently change stochastic-gradient semantics.
- **Lesson**: Splitting a local operator bounds forward temporaries but not necessarily backward
  replay state. When an outer checkpoint recomputes a sequence of chunks, checkpoint each chunk
  as the lifetime boundary for its saved activations.
- **Related Constraint**: N/A

### FSDP2 overlap can retain an all-gather buffer beyond a memory-tight block
- **Date**: 2026-08-30
- **Symptom**: MiniMax H3 Ref2VA FSDP2 GRPO exhausted a 95 GiB device when a block
  pre-forward all-gather requested 1.21 GiB with only 1.03 GiB free.
- **Root Cause**: Each H3 transformer block formed one 1.21 GiB gather unit, while FSDP2's default
  implicit overlap also retained the current raw gather result through the next block's copy-in.
  The long Ref2VA packed sequence left too little headroom for either lifetime.
- **Fix**: Ref2VA extends the FSDP2 wrap policy with its attention, chunked feed-forward, and
  496 MiB AdaLN modulation modules, splitting each block into call-ordered gather units. It also
  opts into default-stream unshard immediately after distributed preparation so each raw gather
  result is released after copy-out, trading communication overlap for lower peak allocation.
- **Lesson**: When model activations nearly fill a device, communication overlap is also a memory
  policy. Apply the backend's explicit lifetime control at the prepared root instead of adding
  allocator flushes or weakening model semantics.
- **Related Constraint**: N/A

### PEFT projections should not materialize a full-sequence LoRA branch
- **Date**: 2026-08-31
- **Symptom**: MiniMax H3 Ref2VA FSDP2 GRPO passed every parameter all-gather but exhausted a
  95 GiB device when one adapted K projection requested a 190 MiB LoRA output with only
  114--134 MiB free.
- **Root Cause**: PEFT evaluated the base projection and low-rank branch over the complete 13,889-token
  packed sequence before adding them, so two full projection outputs overlapped at the memory peak.
- **Fix**: Ref2VA FSDP2 changes each existing adapted Q/K/V/output PEFT Linear to a token-chunked
  forward after LoRA injection. The complete PEFT contract runs per chunk and writes directly into
  one preallocated final output, preserving module and parameter identities, hooks, adapter behavior,
  and state-dict paths without a full-size concatenation copy.
- **Lesson**: Bound parameter-efficient adaptation at the outer adapted-module boundary. Chunking only
  the frozen base layer leaves the adapter branch unbounded, while concatenating chunk outputs can
  recreate the same peak through an avoidable second full-size result.
- **Related Constraint**: N/A

### Token chunk aggregation must not duplicate the complete packed output
- **Date**: 2026-08-31
- **Symptom**: After H3 Ref2VA FSDP2 crossed the adapted projection peak, training exhausted a
  95 GiB device when Q RMSNorm's chunk aggregation requested a 190 MiB output with only
  154--174 MiB free.
- **Root Cause**: The chunk executors retained a list whose outputs already totaled the complete
  packed tensor, then `torch.cat` allocated a second complete tensor to assemble that list.
- **Fix**: H3 feed-forward, Q/K RMSNorm, and PEFT projection chunk executors now allocate their final
  output once and copy each non-overlapping token slice into it. CopySlices preserves input and
  parameter gradients, including nested non-reentrant checkpoint replay, without a concatenation copy.
- **Lesson**: Bounding each operator invocation is insufficient if aggregation recreates a full-size
  peak. Treat chunk assembly as part of the memory contract and retain exactly one final output.
- **Related Constraint**: N/A

### Rotary embedding should rotate packed rows in bounded slices
- **Date**: 2026-08-31
- **Symptom**: MiniMax H3 Ref2VA FSDP2 GRPO crossed projection, normalization, and aggregation
  peaks but exhausted a 95 GiB device when rotary embedding requested a 144 MiB elementwise
  product with only 74--134 MiB free.
- **Root Cause**: Diffusers built rotate-half, cosine-product, sine-product, sum, and concatenation
  intermediates over all 13,889 query or key rows simultaneously.
- **Fix**: Ref2VA FSDP2 installs an instance-local H3 attention processor before distributed
  preparation. It preserves projection, backend, and output behavior while applying Q/K rotary
  embedding in aligned 1,024-token slices directly into one final output allocation.
- **Lesson**: Positional rotation is row-local even when the following attention is global. Bound
  its elementwise intermediates independently and keep cosine/sine slices aligned with token rows.
- **Related Constraint**: N/A

### Checkpointed attention should release QKV before its output projection
- **Date**: 2026-08-31
- **Symptom**: After bounded rotary embedding passed, H3 Ref2VA FSDP2 GRPO missed a 144 MiB
  output-projection allocation by roughly 10 MiB while the attention result was already available.
- **Root Cause**: The attention processor kept strong Python references to complete Q/K/V tensors
  through the output projection. FSDP2 activation checkpointing had discarded their saved-tensor
  storage requirements, but the local variables still extended their lifetime.
- **Fix**: The bounded H3 processor captures the output dtype, releases Q/K/V immediately after
  attention dispatch, and only then flattens and projects the attention result.
- **Lesson**: Under activation checkpointing, autograd may no longer own an intermediate while a
  Python local still does. End large tensor lifetimes at their last semantic use before allocating
  the next full-size result.
- **Related Constraint**: N/A

### FSDP2 checkpoint and backward-prefetch policies must have independent memory boundaries
- **Date**: 2026-08-31
- **Symptom**: MiniMax H3 Ref2VA FSDP2 GRPO reached the training forward only after several
  operator-level bounds, then exhausted each 95 GiB device during forward activation retention or
  the first backward all-gather. The final backward failure requested 286 MiB while the 442 MiB
  feed-forward unit was still resident.
- **Root Cause**: Accelerate reused the FSDP2 transformer wrap policy for activation checkpointing
  and checkpointed every direct child of each H3 block, retaining several complete packed-sequence
  inputs per block. After replacing those boundaries, PyTorch's implicit backward prefetch still
  overlapped the current feed-forward unit with the next attention unit's all-gather.
- **Fix**: Ref2VA now installs one non-reentrant checkpoint inside every materialized H3 block after
  the block's FSDP mixed-precision input cast, with its saved BF16 inputs held in pinned CPU memory.
  Backend preparation disables Accelerate's duplicate checkpoint owner and opts every prepared
  FSDP2 unit out of implicit next-unit backward prefetch, so the current unit reshards before its
  successor gathers. All variant instances are configured only after materialization. A two-rank
  GRPO sentinel completed two forward/backward/optimizer cycles with stable gradients.
- **Lesson**: FSDP wrap granularity, activation recomputation, saved-input placement, and collective
  prefetch are separate memory policies. Keep each boundary explicit; a policy that is correct for
  parameter sharding may multiply activation lifetimes or overlap adjacent full-parameter units.
- **Related Constraint**: #9, #20

### Nested FSDP2 checkpoint replay must verify that parameters remain unsharded
- **Date**: 2026-08-31
- **Symptom**: MiniMax H3 Ref2VA FSDP2 offline DPO completed both policy-arm forwards but
  failed during the second checkpoint replay with `got mixed torch.Tensor and DTensor`.
- **Root Cause**: PyTorch FSDP2 releases a nested unit after the first arm's post-backward while
  its state remains `PRE_BACKWARD`; the second arm's checkpoint replay therefore takes the
  pre-forward early return and uses sharded DTensor parameters with ordinary tensor inputs. This
  is the upstream PyTorch issue #153354, fixed only after PyTorch 2.10 by commit 6579652.
- **Fix**: Adapter-owned FSDP2 activation checkpointing now appends a post-prepare pre-forward
  hook to every prepared FSDP unit. The hook calls the public synchronous `unshard()` API after
  FSDP's own pre-forward hook, which is a no-op for normal forwards and restores parameters only
  when an earlier checkpoint graph has already resharded them. A two-rank nested-FSDP reproducer
  now completes two forward graphs and one combined backward without changing DPO semantics.
- **Lesson**: A distributed training state is not proof of parameter residency. Multiple
  checkpointed forwards may interleave one unit's post-backward with another graph's replay, so a
  compatibility backport must repair residency at the FSDP lifecycle boundary rather than split
  a coupled objective or expose the workaround in algorithm code.
- **Related Constraint**: #9, #20

### Optimizer/backend compatibility must fail before model loading
- **Date**: 2026-08-31
- **Symptom**: A Muon run configured with DeepSpeed ZeRO-2 failed with the intended compatibility
  error only after SD3.5 weights, LoRA adapters, and training data had been loaded.
- **Root Cause**: Optimizer/backend validation lived only in `_init_optimizer`, whose lifecycle
  position is necessarily after model adapter construction and data preprocessing.
- **Fix**: The compatibility contract is now one shared backend-plan validator called by the
  trainer loader before model construction and defensively called again before optimizer
  construction. The two lifecycle gates therefore cannot drift to different rules.
- **Lesson**: Validate compatibility from configuration as soon as the runtime backend is known.
  Keep a second check at the resource-construction boundary, but delegate both checks to one
  implementation so early rejection does not create a parallel source of truth.
- **Related Constraint**: N/A

### Optional optimizer APIs must be validated before model loading
- **Date**: 2026-08-31
- **Symptom**: A Muon configuration on the then-declared PyTorch 2.6 baseline could pass backend
  validation, load pretrained weights, and then fail when optimizer construction accessed the
  unavailable `torch.optim.Muon` attribute.
- **Root Cause**: Backend validation assumed that parsing a Muon optimizer configuration implied
  its optional PyTorch implementation existed. Flow-Factory's minimum PyTorch version predates
  that API, so configuration support and runtime capability are independent contracts.
- **Fix**: One optimizer capability validator now checks the concrete `torch.optim.Muon` API.
  The shared pre-load execution-plan validator calls it after rejecting intrinsically
  incompatible backends, so supported DDP/FSDP2 plans fail before model loading with an
  actionable upgrade message. Direct optimizer construction and the defensive pre-optimizer
  plan check reuse the same rule.
- **Lesson**: Optional APIs gated by dependency versions belong in the same early execution-plan
  validation as backend compatibility. Detect the capability itself instead of trusting a version
  string, while keeping a late defensive call at the construction boundary.
- **Related Constraint**: N/A

### No-feedback algorithms must keep monitoring rewards eval-only
- **Date**: 2026-08-31
- **Symptom**: Every shipped DiffusionOPD example failed argument loading even though its reward
  comments described monitoring rather than a training signal.
- **Root Cause**: The examples duplicated evaluation rewards under top-level `rewards`, but the
  algorithm declares a no-feedback execution contract that rejects every training reward.
- **Fix**: The DiffusionOPD examples now keep monitoring models only under `eval_rewards`; the
  algorithm guide states that evaluation scores never enter the distillation loss, and the shared
  distillation contract tests cover DiffusionOPD alongside DMD2 and TDM.
- **Lesson**: Monitoring intent does not change configuration semantics. A no-feedback algorithm
  must express quality metrics through the evaluation-only reward surface.
- **Related Constraint**: #7

### Repository dataset fixtures must not live inside the example-config tree
- **Date**: 2026-08-31
- **Symptom**: The checked-in SD3.5 SFT and offline-DPO manifests lived under `examples/data`, while
  every other repository dataset and the public dataset guide used the root `dataset/` hierarchy.
- **Root Cause**: The initial smoke fixtures were colocated with their configs without preserving
  the repository boundary between executable example configs and dataset assets.
- **Fix**: The manifests moved to `dataset/sft_sd3_5` and `dataset/offline_dpo_sd3_5`; their YAML,
  Markdown links, and directory-depth-sensitive asset paths moved with them. A production-parser
  regression now loads both configs and manifests and verifies every supervision asset exists.
- **Lesson**: Treat example configs and their datasets as separate public surfaces. When moving a
  manifest, recompute every dataset-root-relative media path and test the resolved files rather
  than checking only the configured directory string.
- **Related Constraint**: N/A

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

### Rebase conflict resolution includes downstream assertions
- **Date**: 2026-08-31
- **Symptom**: After rebasing onto a README correction for the MiniMax H3 parameter count, the
  merged table was accurate but a child-branch documentation test still required the superseded
  value.
- **Root Cause**: The factual conflict was resolved in the edited document, while its regression
  assertion lived in a cleanly rebased file and therefore received no conflict marker.
- **Fix**: The documentation assertion now checks the corrected `33B` value inherited from main.
- **Lesson**: Conflict markers identify overlapping text, not the complete semantic impact of a
  rebase. After resolving a factual conflict, search for and run downstream assertions that encode
  the same fact even when Git reports those files as clean.
- **Related Constraint**: N/A

### Dependency floors must preserve declared Python compatibility
- **Date**: 2026-08-31
- **Symptom**: Aligning the runtime metadata with `av>=18.0.0` raised Flow-Factory's Python floor
  from 3.10 to 3.11 even though the framework and its Muon dependency stack still supported 3.10.
- **Root Cause**: The media-decoder floor followed the latest tested PyAV release without checking
  whether Flow-Factory used a PyAV 18-only API or whether PyAV 17 covered the same contract.
- **Fix**: Restore `requires-python>=3.10`, set the tested decoder floor to `av>=17.0.0`, and align
  classifiers, formatter targets, runtime errors, installation guidance, and agent documentation.
- **Lesson**: A dependency-induced interpreter-floor increase is not automatically a framework
  requirement. Verify the used API surface and test the last compatible dependency line before
  dropping a supported Python version.
- **Related Constraint**: N/A

### Versioned model rows require independent size verification
- **Date**: 2026-09-21
- **Symptom**: The supported-model table listed Qwen-Image 2.1 as 20B even though its official
  model card identifies the visual generation component as 7B.
- **Root Cause**: The new version inherited the parameter count of earlier Qwen-Image entries
  without independently verifying the changed architecture.
- **Fix**: Correct the README row to 7B, add an exact-row documentation regression, and align
  existing documentation tests with the newly added model row and pinned Diffusers installation.
- **Lesson**: A versioned model name does not imply architectural continuity; verify parameter
  counts against the official model card and lock the specific table row rather than a global count.
- **Related Constraint**: N/A

### Distributed group identities must never share a floating-point payload with rewards
- **Date**: 2026-09-24
- **Symptom**: Cross-rank reward normalization could merge or misroute groups whose prompt hash
  exceeded the exact integer range of float32, and equal hashes from different datasets could be
  treated as the same comparison group.
- **Root Cause**: Reward values, `unique_id`, and `source_id` were packed into one float32 gather;
  grouping also treated `unique_id` alone as globally authoritative.
- **Fix**: Canonicalize group identity as the int64 pair `(source_id, unique_id)`, transport it in a
  separate integer collective, and reuse one acquisition-level dense mapping throughout streamed
  work units. Reward grouping, advantages, DPO pairing, DGPO noise, and TDM-R1 group loss all use
  the same key.
- **Lesson**: Communication coalescing cannot erase type semantics. Pack fields only when their
  exact representation and identity scope match; dataset provenance is part of a comparison-group
  key, not logging metadata.
- **Related Constraint**: #9

### Reward request batches and optimizer work units have independent ownership
- **Date**: 2026-09-24
- **Symptom**: Tying pointwise reward submissions to optimizer tile boundaries fragmented remote
  batches and reduced server utilization, even though a returned row can be routed to its stable
  sample index independently of when that sample becomes optimizable.
- **Root Cause**: The reward buffer was configured with `samples_per_tile` and flushed a short
  request whenever it reached an optimizer boundary.
- **Fix**: Let each reward model own its batch size and executor lane. Requests may cross optimizer
  work-unit boundaries; their futures populate stable acquisition rows, and every work unit waits
  only for the rows it contains. Optimizer geometry remains responsible for complete groups and
  gradient-accumulation closure.
- **Lesson**: Do not make one pipeline stage's batching policy an invariant of another stage.
  Connect stages through stable row identity and readiness dependencies instead.
- **Related Constraint**: #9

### Packed replay requires an acquisition manifest, not sampler-specific branches
- **Date**: 2026-09-24
- **Symptom**: Ready-order tile scheduling could preserve sample count yet silently split or repack
  a Bagel `bsz>1` NaViT forward, changing its packed-sequence numerics between rollout and replay.
- **Root Cause**: The scheduler knew group and accumulation geometry but did not own immutable
  evidence of the original rollout microbatch partition.
- **Fix**: Build an `AcquisitionManifest` containing sample object identity, canonical group keys,
  and rollout-batch boundaries. Pack-composition-dependent adapters validate every optimizer work
  unit against the manifest; work units may reorder, but recorded batches cannot split or repack.
- **Lesson**: Preserve batch-sensitive model semantics through a generic acquisition invariant.
  The special case belongs in an adapter capability plus manifest validation, not in sampler or
  model-name conditionals.
- **Related Constraint**: #7, #9

### Distributed readiness polling must back off as one synchronized schedule
- **Date**: 2026-09-24
- **Symptom**: Slow remote rewards caused tens of thousands of readiness all-reduces in one
  acquisition, making coordination itself a measurable part of the critical path.
- **Root Cause**: Every rank polled at one fixed short interval even after repeated globally empty
  results, so remote-service latency was converted into high-frequency collective traffic.
- **Fix**: Treat the configured interval as the initial delay, exponentially back off after each
  unsuccessful global poll to a bounded cap, and reset the delay whenever a work unit advances.
  Because every rank consumes the same readiness result, the backoff schedule stays symmetric.
- **Lesson**: A distributed poll is communication, not a free local status check. Adapt its cadence
  from globally observed progress while preserving identical collective order on every rank.
- **Related Constraint**: #9c, #18

### Multi-source overlap schedules must preserve group-complete source blocks
- **Date**: 2026-09-24
- **Symptom**: Interleaving one batch from each dataset could split a rank-local or global-tile
  comparison group across sources even though every underlying per-source sampler was valid.
- **Root Cause**: The source scheduler shuffled individual batches without knowing the selected
  sampler layout's minimum group-complete window.
- **Fix**: The sampler layout contract now reports its minimum synchronized window, and overlap
  loaders shuffle source blocks of exactly that length. Per-source alignment guarantees complete
  blocks; non-overlap scheduling retains the legacy one-batch behavior.
- **Lesson**: A composed loader must preserve the structural boundaries promised by each child
  sampler. Mix sources only at a boundary where every active group is closed.
- **Related Constraint**: #9, #9c

### Pairwise policy graphs need an explicit activation-storage policy
- **Date**: 2026-09-25
- **Symptom**: MiniMax H3 T2VA DDP offline and online DPO exhausted a 95 GiB device during the
  rejected policy forward even after retaining the offline recipe's rank-16 LoRA capacity.
- **Root Cause**: The pairwise objective retained the chosen and rejected trainable policy graphs
  until their joint loss backward, so both arms' saved activations coexisted; model-block
  checkpointing bounded one forward's intermediates but not this cross-arm graph lifetime.
- **Fix**: Pairwise trainers now wrap each trainable policy arm in one shared adapter-selected
  activation-storage context. MiniMax H3 opts in on DDP and ZeRO-2, where autograd saves tensors in
  pinned CPU memory until backward; FSDP2 retains its sharded backend path without duplicate
  offload. Offline and online DPO share the same policy, with CPU and FSDP2 no-op regressions.
- **Lesson**: Pairwise memory capacity depends on simultaneous graph count, not only trainable
  parameter size. Preserve the coupled objective and select activation storage at the shared
  trainer/adapter boundary instead of shrinking semantic geometry or splitting its backward.
- **Related Constraint**: #9, #20

### Decoded video bytes must cross one unit-pixel boundary
- **Date**: 2026-09-26
- **Symptom**: Offline Wan and LTX2 target encoding could pass white pixels to the VAE as 509
  instead of 1.
- **Root Cause**: Decoders return uint8 RGB arrays, but Diffusers treats NumPy video input as
  floating pixels already scaled to [0, 1]; its normalization only applies `2 * x - 1`.
- **Fix**: Add a strict shared `utils/video.py` boundary for C-contiguous uint8 RGB FHWC and convert
  sampled targets once into float32 unit pixels. Reuse it in Wan, LTX2, and the already-correct H3
  path while preserving family-specific temporal and VAE normalization semantics.
- **Lesson**: Share representation boundaries, not model-specific normalization. Test numeric
  endpoints with the real third-party processor, not only a shape stub.
- **Related Constraint**: #12, #20
- **Evidence**: Real VideoProcessor regressions fail before the fix for black, white, and mixed
  pixels and pass afterward. Utility and codec tests cover Wan/LTX2, while H3 parity remains exact.
- **Commit**: See the Git commit introducing this entry.

### Decoded image bytes must stay on the RGB PIL boundary
- **Date**: 2026-09-26
- **Symptom**: A real Diffusers image processor maps a white `uint8` NumPy target to 509 instead
  of 1, so a custom decoder or refactor that replaced the built-in PIL payload could reproduce the
  same silent scaling failure as video targets.
- **Root Cause**: PIL images and NumPy arrays carry different numerical semantics at the Diffusers
  preprocessing boundary, while image codecs independently checked only for a PIL base type and
  did not encode the complete RGB/positive-geometry contract in one shared utility.
- **Fix**: Add `require_decoded_rgb_image()` in `utils/image.py`, use it in the default decoder and
  every image output codec family, and preserve each family's released preprocessing: Diffusers
  image processors, Bagel `ToTensor` plus mean/std, and SenseNova `x/127.5-1`.
- **Lesson**: For decoded images, share and validate the byte-domain container rather than adding a
  universal float converter. Test black, midpoint, and white through the real family processors,
  because identical array values can mean different pixels when their container types differ.
- **Related Constraint**: #12, #20
- **Evidence**: Utility tests reject ambiguous payloads; real Diffusers and Bagel transforms plus
  the SenseNova codec map `(0,127,255)` to their expected model-pixel endpoints.
- **Commit**: See the Git commit introducing this entry.

### Model-pixel and decoded-audio boundaries must validate runtime representation
- **Date**: 2026-09-26
- **Symptom**: The offline media guide promised finite floating `BCHW`/`BCFHW` model pixels, but
  configured image codecs and Wan checked only partial geometry; an integer or non-finite processor
  result could therefore reach the VAE. LTX2 and MiniMax H3 also duplicated looser audio checks
  instead of enforcing the documented decoded CPU waveform boundary.
- **Root Cause**: The shared decoded byte-container contracts stopped before reusable tensor
  validators, so model families independently enforced different subsets of shape, dtype,
  finiteness, device, and ownership rules.
- **Fix**: Add dependency-neutral `MediaRepresentation`, `MediaFormat`, and `MediaGeometry`
  primitives plus one `utils/media.py` runtime validator. Keep the public image/video/audio helpers
  as thin wrappers and reuse them across grouped/ordered input decoding, default supervision
  decoding, configured image, Bagel, SenseNova, Wan, LTX2, and MiniMax H3. Input and output
  contracts now compose the same physical format, decoded output validation is central, cache
  identity includes the representation, and encoded-state validation rejects non-finite clean
  components while retaining adapter-owned latent intervals.
- **Lesson**: A common numerical boundary should standardize only facts shared by every model.
  Enforce container, layout, dtype, channel/color semantics, ownership, and finiteness centrally while leaving each
  adapter's released pixel/latent interval and packing convention explicit.
- **Related Constraint**: #12, #20
- **Evidence**: Contract tests cover modality/representation coherence and shared geometry/rates;
  runtime tests cover decoded and model-pixel container/layout/dtype/range enforcement; output
  state and condition-cache tests cover central validation and representation-sensitive identity.
- **Commit**: See the Git commit introducing this entry.

## Cross-refs

- UP: [Hard Constraints](../constraints.md), [Architecture](../architecture.md)
- WORKFLOW: [`ff-debug` Phase 5](../../skills/ff-debug/SKILL.md#5-capture-the-fix)
