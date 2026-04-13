# Adapter Conventions

**Read when**: Adding or modifying a model adapter.

---

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

Defined in `models/abc.py` L380-387. Override in subclasses to add model-specific components (e.g., `connectors`, `image_encoder`).

## Batch Dimension Convention

- All adapter methods (`preprocess_func`, `encode_*`, `inference`, `forward`) receive tensors with batch dim `(B, ...)`.
- `BaseSample` fields are **per-sample** (no batch dim) — the sample collator handles stacking.
- `condition_images` is model-dependent: `Tensor(B,C,H,W)` for uniform shape, `List[List[Tensor]]` for variable shape.

## Numbered Gotchas (append-only)

1. Never call `pipeline.__call__()` from `inference()` — decompose it into individual pipeline steps.
2. `encode_prompt()` must match the pipeline's tokenizer settings exactly (padding, truncation, max_length).
3. `_shared_fields` on Sample determines which fields are shared across batch in sampling. Missing fields cause silent data duplication.
4. `default_target_modules` must list all Linear layers to be LoRA'd; verify with `named_modules()`. Default is `['to_q', 'to_k', 'to_v', 'to_out.0']`.

## Cross-refs

- UP: `architecture.md` "Adapter Pattern", `constraints.md` #5 #11-12
- PEER: `train_inference_consistency.md`, `parity_testing.md`
