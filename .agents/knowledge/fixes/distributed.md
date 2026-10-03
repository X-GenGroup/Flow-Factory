# Distributed Fix Patterns

**Read when**: Debugging or changing distributed behavior.

### Multi-modal batch homogeneity (R6)
- **Date**: 2026-04
- **Symptom**: Silent HF `Dataset.map` errors and inconsistent per-sample types in the `audios` column (sometimes `None`, sometimes `Tensor`, sometimes `List[Tensor]`); image/video columns had a latent batch-length mismatch when a sample contributed zero items.
- **Root Cause**: `_preprocess_batch` returned a mix of `None`, `Tensor`, and `List[Tensor]` for the same modality column, breaking Arrow's homogeneous-column requirement and forcing every downstream consumer to handle three input shapes.
- **Fix**: `data_utils/dataset.py:_preprocess_batch` now always emits `List[List[Media]]` per modality (`[]` for empty samples, `[item]` for single-item samples, multi as-is) and appends to BOTH `xx_args[xx]` and `batch[xx]` for every sample so the columns stay length-aligned. Mirrored the same shape on `models/abc.py:preprocess_func` (`audios` parameter) and `utils/audio.py` (`MultiAudioBatch` type alias).
- **Lesson**: HF Arrow demands homogeneous columns, and downstream consumers benefit from a single canonical type. When a column has variable cardinality per row, always represent it as `List[...]` even when the row is empty or has exactly one element. Never special-case "single item" by unwrapping.
- **Related Constraint**: N/A (codified in `topics/adapter_conventions.md` Gotcha #6 and the new "Multi-media batch homogeneity" bullet under Batch Dimension Convention).

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

### Distributed exact-state phases need synchronized failure and one publisher
- **Date**: 2026-08-28
- **Symptom**: One rank could fail exact-resume RNG/hash preflight while a peer entered `Accelerator.load_state()` and hung; concurrent saves could also pass destination preflight together and write the same staging directory.
- **Root Cause**: Rank-local filesystem work was followed by raw barriers or core mutation without first gathering errors, and publication ownership was checked non-atomically before artifact creation.
- **Fix**: `trainers/abc.py` now gathers all-rank errors after barrier-free path resolution, runtime preflight, and core load; commits runtime progress only after every core load succeeds; atomically elects the global publisher before core save; and keeps every filesystem claim until all publishers install their final directory. `trainers/common/runtime_identity.py` hashes FSDP wrap/state-dict topology and the full DeepSpeed batch/accumulation plan, while `runtime_state.py` validates per-device RNG topology and state installability. Runtime manifests are written only after Accelerator artifacts finish. Targeted multi-rank failure simulations cover preflight, load, and publisher races.
- **Lesson**: A raw barrier is unsafe after rank-local I/O that can raise. Structure distributed checkpoints as preflight → synchronized error gather → backend mutation → synchronized error gather → manifest → atomic publication, acquire the publication claim before any process writes shared staging, and retain it until every filesystem reports success. Exact resume identity must cover backend topology, not only parameter names and shapes.
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

### Distillation exact cursors count rollout batches, not backend work items
- **Date**: 2026-08-29
- **Symptom**: After timestep-aligned TDM accumulation made each trajectory boundary one backend work item, exact resume skipped `num_inference_steps` times too many prompt batches.
- **Root Cause**: Cursor reconstruction still multiplied completed rollout iterations by backend `gradient_accumulation_steps`, even though one rollout now contributes multiple boundary losses to that accumulation window.
- **Fix**: `trainers/distillation/distillation_runtime.py` now derives consumed prompt batches through `resolve_rollout_accumulation_steps()`, and the cursor regression locks `gradient_accumulation_steps=8`, four losses per rollout, and two completed iterations to four consumed batches.
- **Lesson**: Persisted acquisition progress must be projected through the current acquisition-to-backend work-item ratio. Backend GAS is not a valid dataloader cursor when one acquired batch expands into multiple backward graphs.
- **Related Constraint**: #18a

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

## Cross-refs

- UP: [Fix Pattern Router](README.md), [Hard Constraints](../constraints.md)
- WORKFLOW: [`ff-debug`](../../skills/ff-debug/SKILL.md)
