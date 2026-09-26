# Adapter Conventions

**Read when**: Adding or modifying a model adapter.

---

## Classifier-Free Guidance (CFG) Convention

All adapters that support CFG must follow a consistent two-stage pattern. Guidance-distilled models (FLUX.1, FLUX.1-Kontext, FLUX.2) do not use CFG — they pass `guidance_scale` as a guidance embedding directly to the transformer.

### Stage 1: `encode_prompt()` / data preprocessing

- **CFG condition**: `do_classifier_free_guidance = guidance_scale > 1.0` (exception: Z-Image uses `> 0.0`).
- `encode_prompt()` **must** accept `guidance_scale` and compute the CFG flag internally — callers should not need to decide.
- If `do_classifier_free_guidance` is true and `negative_prompt is None`, default to `""`.
- When CFG is active, encode the negative prompt and include `negative_prompt_embeds` (plus `negative_prompt_embeds_mask` or `negative_pooled_prompt_embeds` where applicable) in the returned dict.

### Stage 2: `forward()` / denoising step

- `forward()` receives `negative_prompt_embeds` (may be `None`).
- **CFG condition**: `do_classifier_free_guidance = guidance_scale > 1.0 and negative_prompt_embeds is not None`.
- If `guidance_scale > 1.0` but `negative_prompt_embeds is None`, emit `logger.warning(...)` and **fall back to the no-CFG path** (no error). The warning message must mention both the passed scale and the missing embeddings.
- CFG formula: `velocity = velocity_uncond + guidance_scale * (velocity_cond - velocity_uncond)`.

### Reference implementation

`flux/flux2_klein.py` — `encode_prompt()` and `_forward()`.

### Offline flow-matching guidance

Finite-data SFT and offline DPO must not reuse `train.guidance_scale`: that field controls
generation, and on a conventional CFG adapter it would turn the conditional velocity into a
conditional/unconditional composite whenever negative embeddings are cached. These trainers
expand the immutable `adapter.offline_training_forward_overrides` mapping into every SFT policy
and offline-DPO policy/reference forward. The mapping is layered after both configured training
arguments and dataset conditions, so adapter-owned model semantics always win.

The base mapping sets `guidance_scale=1.0`, the conventional CFG-off point. An adapter replaces
the complete mapping when its forward has different semantics or additional guidance branches:

- Z-Image uses `guidance_scale=0.0` because its CFG threshold is `> 0.0`, with normalization and
  truncation fixed to their neutral settings.
- Guidance-distilled FLUX.1, FLUX.1-Kontext, and FLUX.2 use the official Diffusers training
  condition `guidance_scale=3.5`; this is a learned model embedding, not classifier-free guidance.
- The currently supported Flux2-Klein forward always passes `guidance=None` into its transformer,
  so its `guidance_scale` remains conventional two-pass CFG and inherits the neutral `1.0`.
- Wan T2V/I2V neutralize both transformer stages with
  `guidance_scale=guidance_scale_2=1.0`.
- SenseNova neutralizes text and image guidance together and disables CFG normalization.
- Bagel replaces the base mapping with its actual `cfg_text_scale` / `cfg_img_scale` arguments;
  it must not inherit an irrelevant `guidance_scale` key through its permissive `**kwargs`.
- MiniMax H3 T2VA/FL2VA/Ref2VA inherit neutral `guidance_scale=1.0`; their strict forward validates
  that interface value even though the guidance-distilled checkpoint has no CFG branch.
- LTX2 sets video/audio CFG scales and modality scales to `1.0`, CFG rescale and STG scales to
  `0.0`, and neutralizes STG block selection with
  `spatio_temporal_guidance_blocks=None`.

These mappings are adapter-owned model conditioning, never sampling or algorithm knobs.

### Models with model-specific CFG extensions

| Model | Extension | Notes |
|---|---|---|
| Z-Image | `cfg_truncation`, `cfg_normalization` | Applied after standard CFG formula |
| Qwen-Image / Qwen-Image-Edit-Plus | Norm rescale after CFG | `comb_pred * (cond_norm / noise_norm)` |
| LTX2 | x0-space multi-guidance (CFG + STG + Modality Isolation) | CFG delta computed in x0-space, not velocity-space |
| SD3.5 | Requires `negative_pooled_prompt_embeds` in addition to `negative_prompt_embeds` | Two embedding checks in forward |
| SenseNova | Text `guidance_scale` + I2I `image_guidance_scale`, `cfg_norm`, `cfg_interval` | Raw prompts and ordered reference images are encoded lazily into NEO-Unify KV caches |

### If an algorithm ever needs the separate CFG branches

Some distillation objectives want the conditional and unconditional velocities on
their own rather than only the combined one. There is no such consumer today, so no
API exists. An earlier attempt shipped one anyway, with overrides in SD3.5 and
Z-Image and no caller, and it is not being ported. Build it this way when the
objective lands.

**Derive the guided branch arithmetically.** `guided == uncond + scale * (cond - uncond)`
by definition, so it costs no forward. The earlier attempt queried all three through
`forward()`, and because the third call ran its own batched CFG internally it spent
four transformer evaluations to produce two evaluations' worth of information.

**Declare branches as substitutions, not as a conditioning list.** A flat
"positive kwarg to negative kwarg" map only covers prompts. Real conditioning also
carries masks, and editing models carry image latents that may or may not differ
between branches. Declaring each branch as the set of kwargs that *change* relative
to the conditional one handles all three cases: named kwargs are substituted,
unnamed ones pass through unchanged, and a model with more than two branches simply
declares more entries.

```python
class SD3_5Adapter(BaseAdapter):
    guidance_branches = {
        "unconditional": {
            "prompt_embeds": "negative_prompt_embeds",
            "pooled_prompt_embeds": "negative_pooled_prompt_embeds",
        },
    }

class QwenImageEditPlusAdapter(BaseAdapter):
    guidance_branches = {
        "unconditional": {
            "prompt_embeds": "negative_prompt_embeds",
            "prompt_embeds_mask": "negative_prompt_embeds_mask",
            # Name image latents only when the branches genuinely differ; an editing
            # model that conditions both branches on the same reference omits them
            # and they pass through.
            "image_latents": "negative_image_latents",
        },
    }
```

**Keep resolution and the separate strategy shared; keep combination per adapter.**
Substituting kwargs and issuing one `forward(..., guidance_scale=1.0)` per branch
needs nothing model-specific beyond the declaration above, so SD3.5 and Z-Image need
no method at all; the earlier attempt cost them a few hundred lines each. Combining
the branches is model-specific and stays a hook: the default is the CFG formula,
LTX2 combines in x0-space, Z-Image applies truncation and normalization afterwards,
and Qwen-Image rescales the norm.

**Make the batched variant an opt-in override.** Concatenating along the batch axis
turns N passes into one at N times the activations. Which kwargs may be concatenated
and which must be repeated is genuinely model knowledge - a scalar, a list of image
shapes and a latent tensor each behave differently - so it belongs in an adapter
hook rather than in a generic batcher. Select between the two with a parameter,
defaulting to separate.

## `forward()` as the Consistency Boundary

`adapter.forward()` is the atomic unit for train-inference consistency (-> `train_inference_consistency.md`).

1. **Inference/forward identity**: `inference()` loop must call `forward()` — not duplicate its logic. Any code that affects model output belongs inside `forward()`.
2. **Argument preservation**: All arguments affecting `forward()` output must be stored on the Sample dataclass during rollout and replayed identically by `optimize()`. This includes `guidance_scale`, `stg_scale`, `connector_prompt_embeds`, `noise_level`, etc.

## Upstream Pipeline Alignment

- **Structural vs behavioral separation**: First commit matches the reference diffusers pipeline's numerical output; second commit cleans up style. Never combine both in a single change.
- **`inference()` must reproduce `Pipeline.__call__()` output** given the same seed, dtype, and parameters. Verify via parity testing (-> `parity_testing.md`).
- **Timestep convention**: Adapter receives `t` in `[0, 1000]`; converts internally per model needs. Detail: `topics/timestep_sigma.md`.

## Component Lifecycle

| Category | Property | Frozen | Offloadable | Examples |
|---|---|---|---|---|
| Preprocessing | `preprocessing_modules` | yes | yes | `text_encoders`, `vae` |
| Inference/Training | `inference_modules` | transformer: trainable; VAE: frozen | VAE: yes | `transformer`, `vae` |

Defined in `models/abc.py` (`preprocessing_modules` / `inference_modules` properties). Override in subclasses to add model-specific components (e.g., `connectors`, `image_encoder`).

## Batch Dimension Convention

- All adapter methods (`preprocess_func`, `encode_*`, `inference`, `forward`) receive tensors with batch dim `(B, ...)`.
- `BaseSample` fields are **per-sample** (no batch dim) — the sample collator handles stacking.
- `condition_images` is model-dependent: `Tensor(B,C,H,W)` for uniform shape, `List[List[Tensor]]` for variable shape.
- `inference()` condition parameters (`images`, `videos`, `audios`) arrive as `MultiImageBatch` / `MultiVideoBatch` / `MultiAudioBatch` (nested batch, e.g. `List[List[Image.Image]]`, `List[List[Tensor]]`) from the training pipeline collator (`data_utils/dataset.py` `collate_fn`). Type annotations on `inference()` must use the multi-form, not the bare `ImageBatch` / `VideoBatch` / `AudioBatch`.
- **Multi-media batch homogeneity**: `_preprocess_batch` (`data_utils/dataset.py`) guarantees `List[List[Media]]` for every modality column — empty samples contribute `[]`, single-item samples contribute `[item]`, multi-item samples contribute `[item1, ..., itemN]`. This keeps HF Arrow columns homogeneous and lets every `encode_*` consume a single shape.
- **Image-column persistence (HF Image feature)**: the raw `images` column and any `encode_image` output listed in `python_format_columns` (ClassVar on `BaseAdapter`, empty by default) are stored via the HuggingFace `Image` feature (PNG bytes) instead of raw tensors, and **read back as PIL** (`List[List[PIL.Image]]`). This is what lets ragged multi-reference batches (variable size/count) serialize — raw tensors are only Arrow-serializable when uniform. Opt in per adapter only for genuine RGB images (e.g. Bagel and SenseNova `condition_images`); never declare preprocessed/non-RGB tensors (VAE-ready video tensors, latents) — PIL conversion is lossy and breaks tensor consumers (e.g. LTX2-I2AV `condition_images` stays a tensor). Consumers must normalize via `_standardize_image_input` / `standardize_image_batch` before any tensor op. To keep PIL on the **sample** too (not just the dataset cache), the adapter's `ImageConditionSample` subclass must set `condition_images_as_pil = True` (else `__post_init__` re-canonicalizes to `List[Tensor(C,H,W)]` [0,1]); e.g. `BagelI2ISample` and `SenseNovaI2ISample`.
- Single-condition adapters must flatten internally via `_standardize_image_input` / `_standardize_video_input` using `is_multi_image_batch` / `is_multi_video_batch` to extract the first element per sample (e.g. `Wan2_I2V._standardize_image_input`, `LTX2_I2AV._standardize_image_input`). Multi-condition adapters (e.g. `Flux2`, Bagel, and SenseNova) consume the nested structure directly.

## Latent Geometry

Model-agnostic description of latent tensor **axis roles** on `BaseAdapter`. Defined in `models/latent_geometry.py` + `models/abc.py`. Additive and information-preserving — it only locates axes; it does not summarize or pool latents (a future consumer, e.g. a critic/value head, will add task-specific encoders on top).

### API

| Member | Role |
|---|---|
| `LATENT_AXES` (ClassVar) | Optional static `LatentAxes` override; `None` -> infer from ndim |
| `resolve_latent_axes(latents)` | Returns `LatentAxes` (override, else ndim-inferred) |
| `infer_latent_axes(ndim)` (module fn) | Maps rank 3/4/5 to canonical `LatentAxes`; fail-fasts on unsupported ranks |

### Layouts (axis roles, resolution-invariant)

| Layout | ndim | shape | channel | sequence | temporal | spatial |
|---|---|---|---|---|---|---|
| PACKED | 3 | `(B, Seq, C)` | -1 | 1 | - | () |
| CONV | 4 | `(B, C, H, W)` | 1 | - | - | (2,3) |
| VIDEO | 5 | `(B, C, T, H, W)` | 1 | - | 2 | (3,4) |

Only axis roles are stored, never dynamic sizes (Seq/H/W/T). Packed models fold H/W(/T) into `Seq` via patchify, so their spatial/temporal are empty by design.

### Per-adapter

All 14 adapters resolve correctly via default ndim inference — none override `LATENT_AXES`:

| Adapter(s) | Layout |
|---|---|
| FLUX.1/Kontext/2/Klein, Qwen-Image/Edit-Plus, Bagel, LTX2 T2AV/I2AV | PACKED |
| SD3.5, Z-Image, SenseNova | CONV |
| Wan2 T2V/I2V | VIDEO |

LTX2 packs `[video|audio]` into one `(B, Seq, C)` sequence, so it resolves as PACKED (the split point lives in the adapter's own `forward` via `video_seq_len`, not in the geometry layer). I2I/I2V/Edit store only the generated latent in `all_latents` (condition is concatenated inside `forward()` / kept in separate fields), so the standard layout applies and reference-image count is irrelevant. Override `LATENT_AXES` only for a genuinely non-standard rank/channel layout.

## Numbered Gotchas (append-only)

1. Never call `pipeline.__call__()` from `inference()` — decompose it into individual pipeline steps.
2. `encode_prompt()` must match the pipeline's tokenizer settings exactly (padding, truncation, max_length).
3. `_shared_fields` on Sample determines which fields are shared across batch in sampling. Missing fields cause silent data duplication.
4. `default_target_modules` must list all Linear layers to be LoRA'd; verify with `named_modules()`. Default is `['to_q', 'to_k', 'to_v', 'to_out.0']`.
5. `inference()` `images`/`videos` params are always `MultiImageBatch`/`MultiVideoBatch`. Single-condition adapters must flatten via `_standardize_*_input` with `is_multi_image_batch`/`is_multi_video_batch` (e.g. `Wan2_I2V._standardize_image_input`); annotate as `MultiImageBatch`/`MultiVideoBatch`, never `ImageBatch`/`VideoBatch`.
6. **Multi-media batch homogeneity** — `_preprocess_batch` always emits `List[List[Media]]` per modality. Do NOT unwrap single-element lists in `encode_*` and do NOT return a bare `Tensor` or `None` for empty samples — return `[]`. Returning a bare `Tensor` for single-audio samples (or `None` for empty image samples) breaks Arrow column homogeneity and forces downstream consumers to handle three input shapes. Applies symmetrically to `images`, `videos`, and `audios`.
7. **CFG two-stage consistency** — `encode_prompt()` and `forward()` must use the same threshold for CFG activation (`guidance_scale > 1.0`, or `> 0.0` for Z-Image). `forward()` must gracefully handle the case where `guidance_scale > threshold` but negative embeds are `None` (warn + fallback, never error). See "Classifier-Free Guidance (CFG) Convention" section above.
8. **Bagel batch handling (NaViT subset-round packing)** — Bagel uses sequence packing, not a leading batch dim. **Both T2I and I2I** pack all B samples into one block-diagonal forward (`_build_gen_context` + `_forward_packed`; the framework's `(B, num_tokens, dim)` latents reshape to packed `(B*num_tokens, dim)` and back). For I2I, reference images are added in per-image rounds (`num_rounds = max per-sample count`); a sample without an r-th image is passed as `None` to `prepare_vae_images` / `prepare_vit_images`, which keep its cached KV and add a **zero-length query segment**. So a **variable per-sample reference-image count** (and varying sizes) is handled by packing directly — there is no per-sample (`batch_size=1`) fallback. The cache merge requires every sample to remain on the key/value side, so only the query may be a subset. The prefill is `@torch.no_grad` and every round has >=1 active image (`max_seqlen_q > 1`), avoiding flash-attn zero-length pitfalls (no backward, no `max_seqlen_q==1`). `_is_i2i(condition_images)` depends only on condition-image presence (distributed-safe). CFG global renorm is computed **per sample** over `packed_seqlens - 2`, and `forward()` returns per-sample `(B,)` log-prob (not per-token). **Distributed**: the prefill makes a data-dependent number of `language_model` forward calls (`2*num_rounds + 2`); `language_model` is the only FSDP-sharded module (frozen ViT/VAE are unsharded, so they don't count). Under FSDP FULL_SHARD/HYBRID (and ZeRO-3) each call AllGathers `language_model`'s shard, so per-rank counts mismatch and deadlock — `_assert_variable_count_supported` fails fast there (`@torch.no_grad` does not help; FSDP still all-gathers to compute). DDP / DeepSpeed ZeRO-1/2 (the Bagel I2I backends) replicate params (local forward, fixed grad sync at backward), so variable counts are safe. The FSDP-safe alternative is to gather `language_model` once for the generation (`summon_full_params` / `reshard_after_forward=False`).
9. **Image columns persist via HF Image feature (variable-size/count I2I)** — preprocessing stores image data as PIL via the HF `Image` feature, not raw tensors; ragged tensor columns (multi-reference images of varying size/count) are NOT Arrow-serializable and otherwise crash in `Dataset.map` with `TypeError: a bytes-like object is required, not 'Tensor'` / `OverflowError`. The raw `images` column is always stored this way; an `encode_image` output is stored this way only when its name is listed in the adapter's `python_format_columns` ClassVar (default empty — opt in for RGB images only, e.g. Bagel and SenseNova `condition_images`). These columns **read back as PIL** (`List[List[PIL.Image]]`); the `torch` format excludes them (`_apply_torch_format` in `dataset.py`), and `collate_fn` keeps them as a `MultiImageBatch`. To keep PIL end-to-end on the **sample** (not just the cache), the adapter's `ImageConditionSample` subclass must also set `condition_images_as_pil=True` (else `ImageConditionSample.__post_init__` re-canonicalizes to `List[Tensor(C,H,W)]` [0,1]); e.g. `BagelI2ISample` and `SenseNovaI2ISample`. Bump `_PREPROCESS_FORMAT_VERSION` if the on-disk image format changes again.
10. **Latent geometry override is rarely needed** — `resolve_latent_axes` infers axis roles from latent ndim (3=packed, 4=conv, 5=video), correct for all 14 adapters. Set the `LATENT_AXES` ClassVar only for a genuinely non-standard rank/channel layout. LTX2's packed `[video|audio]` resolves as PACKED; SenseNova's generated pixels resolve as CONV. Modality splits and conditioning remain adapter-owned and do not change the generated trajectory layout. See "Latent Geometry".
11. **Diffusers cache readiness is explicit** — set `supports_diffusers_cache = True` only when every transformer forward branch, including CFG/STG variants and every transformer in a multi-transformer adapter, runs inside `cache_context`. The rollout accelerator rejects the default `False` before enabling any component. See `guidance/acceleration.md` "Model cache-readiness".
12. **LTX2 rollouts publish structured trajectories only** — `LTX2_T2AV_Adapter` / `LTX2_I2AV_Adapter` `inference()` fill `BaseSample.trajectory` with one `StructuredTrajectory` per sample (per-component states, full per-component schedules, joint + per-component log probabilities, and the latent-shaped callbacks in `LTX2_STRUCTURED_CALLBACK_FIELDS`) and leave every legacy field (`timesteps`, `all_latents`, `latent_index_map`, `log_probs`, `log_prob_index_map`) `None`. Non-latent callbacks (e.g. `std_dev_t`, `noise_level`) stay in `extra_kwargs` with their `callback_index_map`, which is present only when such a callback was actually collected. Trainers must read the trajectory through the adapter bridge (`get_terminal_state`, `get_replay_step`, `get_replay_callback`), never by indexing the legacy fields. I2AV additionally carries a video `active_mask` derived from `~conditioning_mask`, so the conditioning frame is excluded from every reduction, log-prob weighting and forward-process noising.

13. **SenseNova ragged I2I is per-sample, not NaViT-packed** — SenseNova-U1 1.0/1.5 accepts ordered, variable-size and variable-count reference images. Each sample's references become one variable-length NEO-Unify prefix and remain PIL across preprocessing, rollout and replay. A framework batch may contain several such samples, but `SenseNovaAdapter.inference()` / `forward()` iterate them and call `SenseNovaDenoiser` with B=1; unlike Bagel, independent samples are not concatenated into a packed attention sequence. The native model's `batch_size>1` path only expands one shared prompt/reference KV cache to generate multiple noises for the same condition and is not a ragged multi-sample batch API.

14. **Offline forward overrides are model conditioning, not sampling CFG** — SFT and offline DPO expand the adapter's complete immutable `offline_training_forward_overrides` mapping into every policy and reference forward, after batch conditions and configured sampling arguments. Conventional CFG adapters use their CFG-off point; guidance-distilled adapters use the value expected by their learned guidance embedder; multi-branch adapters neutralize every active branch under its real forward argument names. Replace the base mapping rather than adding unrelated keys, and never infer it from cached negative embeddings or expose it as an algorithm knob.

15. **Runtime condition realization has one input owner** — A conditioned offline adapter declares
    `build_condition_state_preparer()` only when cached fields are not already the exact forward
    condition. `prepare_condition_state()` runs once per batch. SFT reuses that realization for
    target binding; offline DPO reuses the same tensor leaves for chosen/rejected and
    policy/reference forwards. The input-owned `forward_context` and `output_context` may share
    prepared tensors. Require collision-free keys only within each consumer's merged view: cached
    plus forward fields, and input-owned plus candidate-owned output fields. The preparer declaration
    lists logical components but must not materialize, move, replace, or cast them. Checkpoint
    variants narrow the class-level I/O superset through `_resolve_pipeline_io_contract()` rather
    than branching inside the dataset or algorithm.

16. **Diffusers cache readiness may be policy-specific** — When an opted-in adapter supports only
    a subset of upstream cache policies, declare the exact user-facing ids in
    `supported_diffusers_cache_policies`; `None` retains all-policy behavior for existing
    cache-ready adapters. Model-specific `prepare_diffusers_cache()` shims run only after the
    accelerator validates both policy and config, and before cache enablement. Unsupported
    adapters, policies, and configs must fail before component or global-registry mutation.

17. **Exact native KV caches are symmetric adapter state, not rollout acceleration** — A model may
    reuse a mathematically exact causal prefix without opting into `diffusers_cache`. Qwen-Image
    2.1 prefills text and condition-image K/V with a minimal 2x2 target block, then uses cached
    decode for every real denoising prediction. Rollout keeps one detached cache per CFG branch;
    replay never imports those tensors and rebuilds differentiable K/V from current parameters.
    Both prefill and decode must call the routed prepared component, carry identical LoRA attention
    kwargs, and use the same grad-enabled/checkpointed transformer path during rollout and replay.
    Rollout cache tensors become graph-free `requires_grad=True` leaves so attention dispatch still
    matches differentiable replay, while returned predictions are detached immediately. Keep
    `supports_diffusers_cache = False` so the lossy feature-cache plugin cannot be enabled
    accidentally. Replay K/V leave each block from inside attention rather than through a block
    output, so under FSDP2 the adapter registers every owning unit's pre-backward hook on them;
    any new side-output cache needs the same bridge before it can train under parameter sharding.

18. **Decoded video bytes cross one explicit unit-pixel boundary** — Built-in offline video
    decoders return C-contiguous CPU `uint8` RGB `FHWC`; output codecs validate that representation
    and call `decoded_video_to_unit_float()` exactly once. Unit pixels are shared, but model pixels
    are not: Wan/LTX2 use Diffusers `[-1,1]`, while MiniMax H3 applies checkpoint mean/std. Reject
    ambiguous float payloads instead of guessing their range, and keep temporal geometry,
    posterior policy, and latent packing adapter-owned.

19. **Decoded image bytes stay inside one explicit RGB PIL boundary** — Built-in offline image
    decoders return detached positive-size RGB PIL images, and every image output codec validates
    that representation with `require_decoded_rgb_image()` before model preprocessing. Container
    type carries numerical meaning: Diffusers converts PIL bytes to unit pixels but treats NumPy
    arrays as already unit-scaled. Reject array/tensor payloads and non-RGB modes instead of
    guessing or silently converting them. Share this decoded-media boundary while keeping resize,
    posterior policy, model pixel normalization, and latent packing adapter-owned.

## Fix Records

### Sampling CFG leaked into finite-data velocity matching

- **Date**: 2026-08-28
- **Symptom**: SFT and offline DPO could optimize a CFG-composite velocity when
  `train.guidance_scale > 1.0`, while their target remained the conditional flow-matching
  velocity.
- **Root Cause**: The shared forward helper copied the generation-oriented training arguments
  into offline forwards without an adapter-owned model-conditioning override.
- **Fix**: Added the immutable `BaseAdapter.offline_training_forward_overrides` mapping, declared
  complete model-specific neutral or guidance-distilled mappings (including Wan T2V, SenseNova,
  and Bagel multi-branch CFG), and expanded it into every offline policy/reference forward after
  configured and batch arguments.
- **Lesson**: Generation controls and finite-data model conditioning may share a low-level
  argument name but must have separate semantic owners.
- **Related Constraint**: #7

### Active CFG requires a materialized default negative condition

- **Date**: 2026-09-20
- **Symptom**: Qwen-Image 2.1 silently ran only the positive branch when
  `guidance_scale > 1.0` and the caller omitted `negative_prompt`.
- **Root Cause**: Its prompt encoder treated an optional raw negative prompt as permission to
  disable CFG instead of applying the framework's empty-prompt default.
- **Fix**: Qwen-Image 2.1 now normalizes a missing negative prompt to `""` whenever CFG is active,
  with a regression that requires both encoded branches and their masks.
- **Lesson**: Raw-input optionality and branch activation are separate contracts; once a guidance
  scale activates CFG, preprocessing must materialize every required branch.
- **Related Constraint**: N/A

### Tuple condition geometry preserves its area budget

- **Date**: 2026-09-20
- **Symptom**: A Qwen-Image 2.1 `condition_image_size` such as `[384, 512]` used a `512 * 512`
  pixel budget instead of the configured `384 * 512` budget.
- **Root Cause**: The adapter widened Diffusers' scalar `output_resolution` to a tuple by squaring
  its largest side rather than preserving Flow-Factory's tuple-as-area convention.
- **Fix**: Integer sizes still produce a square area, while tuple/list sizes now use the product
  of their height and width; a non-square 4:3 regression locks the resulting geometry.
- **Lesson**: When extending a scalar upstream control to a framework tuple, preserve the
  framework's established semantic unit rather than deriving a new one from one coordinate.
- **Related Constraint**: N/A

### Side-output KV caches must join the FSDP2 backward lifecycle

- **Date**: 2026-09-24
- **Symptom**: Qwen-Image 2.1 GRPO with FSDP2 finished rollout and the first replay forward, then
  failed in backward checkpoint replay of `attn.to_q` with `aten.mm.default got mixed
  torch.Tensor and DTensor`. The same recipe trained under DeepSpeed ZeRO-2.
- **Root Cause**: FSDP2 gathers a unit for backward only when a gradient reaches one of its
  forward outputs. The replay prefill stores post-RoPE K/V from inside attention, so the cached
  target pass sends gradients into each block through those tensors after the block's main-path
  post-backward has already resharded it. For the first block, whose prefill inputs require no
  gradient, no later hook would even schedule a post-backward reduce-scatter for that path.
- **Fix**: `_register_fsdp2_prefix_cache_pre_backward()` registers every FSDP2 unit that owns a
  block's parameters (its nested units plus its nearest FSDP ancestor) on that layer's replay K/V
  through the unit state's own `_register_pre_backward_hook`, giving the tensors the lifecycle of
  block outputs. It fails closed if a PyTorch build lacks that unit-state hook. Two-rank gloo
  regressions use Accelerate's FSDP2 layout (per-child checkpoint wrappers, per-block units,
  frozen non-block parameters, CFG) and require bit-exact gradients versus an unsharded replica,
  while the same run without the bridge must reproduce the DTensor failure.
- **Lesson**: A differentiable tensor that leaves a sharded unit by any path other than its
  forward outputs bypasses FSDP2's pre-backward gather and post-backward scheduling. Re-entering
  the unit's own lifecycle hook is required; a replay-time `unshard()` does not help when
  checkpointing wraps the unit's children, and cannot schedule the missing gradient reduction.
- **Related Constraint**: #9

## Cross-refs

- UP: [`constraints.md` #5](../constraints.md#5-adapter-component-runtime-contract), [`constraints.md` #11](../constraints.md#11-basetrainer-execution-contract), [`constraints.md` #12](../constraints.md#12-baseadapter-abstract-methods), [Architecture Adapter Pattern](../architecture.md#adapter-pattern-models)
- PEER: [Train/Inference Consistency](train_inference_consistency.md), [Parity Testing](parity_testing.md)
- WORKFLOW: [`ff-new-model`](../../skills/ff-new-model/SKILL.md)
