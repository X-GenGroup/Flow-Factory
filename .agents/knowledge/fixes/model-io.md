# Model Io Fix Patterns

**Read when**: Debugging or changing model io behavior.

### Non-abstract encoder defaults (R7)
- **Date**: 2026-04
- **Symptom**: Adding `encode_audio` as `@abstractmethod` on `BaseAdapter` would force one-line `pass` stubs on 11 existing concrete adapters, none of which consume audio. The first iteration of R6 actually shipped this — and the resulting "noise" diff dwarfed the real change.
- **Root Cause**: Incorrect default-discoverability assumption — abstract methods force every subclass to acknowledge a feature, even when the subclass doesn't use it.
- **Fix**: `models/abc.py` dropped `@abstractmethod` from all 4 encoders (`encode_prompt`, `encode_image`, `encode_video`, `encode_audio`); default body is `pass` returning `None`; `preprocess_func` skips integration when the called encoder returns `None`. The Round-6 stub overrides on 11 concrete adapters were reverted, leaving them byte-identical to `origin/main`.
- **Lesson**: When extending a base contract for a partial-coverage feature (where only some subclasses will participate), no-op default + opt-in override beats forcing every subclass to acknowledge it. Reserve `@abstractmethod` for invariants that ALL subclasses must implement (e.g. `load_pipeline`, `decode_latents`, `forward`, `inference`).
- **Related Constraint**: #12 (post-update text codifies "Optional encoder overrides (no-op default)").

### Preference-arm conditioning ownership
- **Date**: 2026-08-11
- **Symptom**: DPO evaluated the rejected H3 state with the chosen sample's prompt/reference conditioning.
- **Root Cause**: Shared-noise logic was incorrectly extended to the conditioning batch, even though only the forward-process noise is shared between arms.
- **Fix**: Rejected policy/reference forwards now receive `rejected_batch`, with a production-path regression using distinct prompt embeddings.
- **Lesson**: Preference arms may share stochastic coordinates but never model conditioning; replay state and conditioning must come from the same sample.
- **Related Constraint**: #7

### Validate nested media cardinality before flattening
- **Date**: 2026-08-28
- **Symptom**: Valid offline Flux1-Kontext batches shaped as `List[List[PIL]]` raised `TypeError` while checking whether a sample carried multiple condition images.
- **Root Cause**: `_standardize_image_input()` flattened the nested batch before running its per-sample cardinality check, so the check called `len()` on each PIL image.
- **Fix**: Flux1-Kontext now checks and warns on the original nested batch before selecting the first image, with a regression covering two offline single-image rows.
- **Lesson**: Perform shape and cardinality validation at the boundary where that structure still exists; flattening destroys the evidence needed to validate it safely.
- **Related Constraint**: N/A

### Optional-kernel adapter tests must lazy-load behind the dependency seam
- **Date**: 2026-08-28
- **Symptom**: Collecting the Bagel TDM contract test on macOS failed before any test ran because `flash-attn>=2.5.8` was unavailable.
- **Root Cause**: The test imported the Bagel adapter at module scope instead of installing the existing fake optional-kernel modules before the adapter import.
- **Fix**: `tests/models/test_bagel_tdm_contracts.py` now lazily imports Bagel after stubbing `flash_attn`, OpenCV, and the availability probes; new Python files also receive the required Apache 2.0 headers.
- **Lesson**: Contract tests for CUDA-only optional adapters must exercise the adapter through its dependency boundary so CPU and macOS collection remains valid; importing such adapters at module scope turns an optional dependency into a repository-wide test dependency.
- **Related Constraint**: N/A

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

### Optional role discovery and lazy default materialization
- **Date**: 2026-08-10
- **Symptom**: A declared classic `transformer_2=None` entered the transformer role group and
  adapter freezing called `requires_grad_` on `None`; separately, `materialize_components(None)`
  eagerly loaded every modular spec.
- **Root Cause**: Role discovery filtered names rather than non-`None` values, and the default
  materialization request expanded declared names instead of materialized modules.
- **Fix**: Role discovery now excludes `None` values while retaining non-`None` modular specs;
  default materialization uses already-materialized module names, and normal canonical lookup
  returns direct materialized attributes before consulting the expensive declared component map.
- **Lesson**: Optional declarations are valid for explicit compatibility lookup but cannot imply
  role membership, and an omitted lazy-materialization selection must never mean "load all."
- **Related Constraint**: #5.

## Cross-refs

- UP: [Fix Pattern Router](README.md), [Hard Constraints](../constraints.md)
- WORKFLOW: [`ff-debug`](../../skills/ff-debug/SKILL.md)
