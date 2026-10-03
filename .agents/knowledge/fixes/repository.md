# Repository Fix Patterns

**Read when**: Debugging or changing repository behavior.

### Preserve contract-locked terminology during architecture rewrites
- **Date**: 2026-08-28
- **Symptom**: The SenseNova documentation regression test failed after the architecture table retained the correct behavior but dropped the contract phrase `ordered variable-count references`.
- **Root Cause**: A broad documentation rewrite paraphrased a tested semantic distinction without checking the existing documentation contract.
- **Fix**: `.agents/knowledge/architecture.md` now restores the exact phrase while retaining the grouped `images` and non-NaViT execution details; the documentation suite verifies both distinctions.
- **Lesson**: Search documentation tests before rewriting architecture terminology, especially where wording distinguishes adapters with superficially similar multi-reference inputs.
- **Related Constraint**: N/A

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

### Documented dataset tools must use their package invocation mode
- **Date**: 2026-08-30
- **Symptom**: Running the documented `python dataset/offline_smoke/prepare.py ...` command failed before argument parsing with `attempted relative import with no known parent package`.
- **Root Cause**: The package uses relative imports, while the documentation incorrectly advertised direct file execution instead of Python's module mode.
- **Fix**: All offline-smoke commands now use `python -m dataset.offline_smoke.<tool>` with unconditional package imports, and a subprocess regression executes the documented form.
- **Lesson**: A checked-in CLI example is part of the public interface; standardize on one package-aware invocation and test that exact process rather than adding conditional import fallbacks.
- **Related Constraint**: N/A

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

### Agent policy drift requires structural validation
- **Date**: 2026-10-03
- **Symptom**: Commit authorization, inheritance exceptions, section ranges, and cross-reference
  headings disagreed across root instructions, skills, knowledge documents, and tool adapters.
- **Root Cause**: The harness documented structural requirements but had no executable validation
  for its duplicated indexes and adapter files.
- **Fix**: Align the canonical policies and add a repository validator that checks the structural
  invariants most likely to drift.
- **Lesson**: Keep policy in one canonical layer and test routing, registration, and adapter
  structure instead of relying on reviewers to compare every agent document manually.
- **Related Constraint**: #28

## Cross-refs

- UP: [Fix Pattern Router](README.md), [Hard Constraints](../constraints.md)
- WORKFLOW: [`ff-debug`](../../skills/ff-debug/SKILL.md)
