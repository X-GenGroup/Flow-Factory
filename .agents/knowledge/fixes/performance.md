# Performance Fix Patterns

**Read when**: Debugging or changing performance behavior.

### Batched Arrow schemas must cover later optional values
- **Date**: 2026-08-28
- **Symptom**: A valid optional-image dataset ordered as two prompt-only rows followed by two image-conditioned rows failed in the second `Dataset.map` chunk while casting image bytes or latent tensors to the first chunk's empty/string schema.
- **Root Cause**: HuggingFace fixed the writer schema from the all-empty first output chunk, while Flux2 also omitted its image output keys whenever that individual chunk contained no images.
- **Fix**: `data_utils/dataset.py` now scans real map-sized slices, dropping resolved columns from the bounded scan until it finds each later typed representative chunk; it probes those chunks only when a source column transitions from empty in the first chunk to typed later, restores Python/NumPy/torch/MPS and explicit-generator RNG state after that schema-only probe, and passes the resulting explicit `Features` through the `datasets==3.3.2`-compatible map surface. Flux2 now emits aligned image columns whenever the source image field exists; Flux2 and Flux2-Klein regressions verify identical empty-chunk output structure, a spy rejects whole-column reads, and a four-row offline cache regression covers the real Arrow path.
- **Lesson**: A batched map's output key set and feature types are dataset-level contracts, not properties of whichever values happen to appear in the current chunk. Optional adapters must emit empty row slots consistently, and data writers must derive schemas from representative typed values before committing the first Arrow batch. A schema probe is intentionally narrow and restores RNG, but preprocessors still own their normal deterministic/cache-safe behavior; arbitrary adapter-owned mutable state is not rollback-safe.
- **Related Constraint**: N/A

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

## Cross-refs

- UP: [Fix Pattern Router](README.md), [Hard Constraints](../constraints.md)
- WORKFLOW: [`ff-debug`](../../skills/ff-debug/SKILL.md)
