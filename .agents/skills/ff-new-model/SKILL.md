---
name: ff-new-model
description: "Complete workflow for adding a new model adapter. Covers analysis, sample dataclass, adapter implementation (7 abstract methods), registry, example YAML, and verification. Trigger: 'add model', 'support new model', 'integrate model', 'new adapter'."
---

# New Model Adapter Integration

> **Authoritative reference**: `guidance/new_model.md` — read it first.

## Prerequisites

Before starting, ensure you understand:
1. The target model's diffusers pipeline (or that you'll need a pseudo-pipeline)
2. The task type: Text-to-Image, Image-to-Image, Text-to-Video, Image-to-Video
3. Which Sample dataclass to extend

## Phase 1: Analysis

1. **Identify the diffusers pipeline** for the target model
   - Check if it exists in `diffusers`: `from diffusers import <Pipeline>`
   - If not, you'll need a pseudo-pipeline (see `guidance/new_model.md` advanced section)
2. **Study an existing adapter** of the same task type:
   - T2I: `models/flux/flux1.py` or `models/stable_diffusion/sd3_5.py`
   - I2I: `models/flux/flux1_kontext.py` or `models/qwen_image/qwen_image_edit_plus.py`
   - T2V: `models/wan/wan2_t2v.py`
   - I2V: `models/wan/wan2_i2v.py`
3. **Map pipeline components** to adapter responsibilities:
   - Text encoders → `encode_prompt()`, `preprocessing_modules`
   - VAE → `encode_image()` / `decode_latents()`, `preprocessing_modules`
   - Transformer/UNet → `forward()`, `target_module_map`, `inference_modules`
4. **Also read**: `topics/adapter_conventions.md` for upstream alignment rules; `topics/dtype_precision.md` for precision handling in `cast_latents()`.

## Phase 2: Implementation

### Step 1 — Define Sample Dataclass

```python
# src/flow_factory/models/<family>/<model>.py
@dataclass
class MyModelSample(T2ISample):  # or appropriate base
    _shared_fields: ClassVar[frozenset[str]] = frozenset({})
    # Add model-specific fields if needed
```

### Step 2 — Create Adapter Class

```python
class MyModelAdapter(BaseAdapter):

    @property
    def preprocessing_modules(self) -> List[str]:
        return ["text_encoder", "vae"]  # Components for Stage 1

    @property
    def inference_modules(self) -> List[str]:
        return ["vae"]  # Components needed at inference time

    @property
    def target_module_map(self) -> Dict[str, str]:
        return {"transformer": "transformer"}  # Trainable components
```

### Step 3 — Implement Required Methods

| Method | Purpose | Stage | Abstract? |
|--------|---------|-------|-----------|
| `load_pipeline()` | Load diffusers pipeline | Init | Yes |
| `encode_prompt()` | Text → embeddings | 1 | Yes |
| `encode_image()` | Image → latents | 1 | Yes |
| `encode_video()` | Video frames → latents | 1 | Yes |
| `decode_latents()` | Latents → pixels | 3 | Yes |
| `inference()` | Full multi-step denoising | 3 | Yes |
| `forward()` | Single-step denoising loss | 6 | Yes |
| `preprocess_func()` | Raw inputs → cached tensors (calls encode methods) | 1 | No (concrete, override only if needed) |

### Step 4 — Register

Add to `_MODEL_ADAPTER_REGISTRY` in `src/flow_factory/models/registry.py`:
```python
'my-model': 'flow_factory.models.<family>.<model>.MyModelAdapter',
```

## Phase 3: Configuration

Create example YAML config in `examples/grpo/lora/<model>.yaml`:
```yaml
model:
  model_type: "my-model"
  model_path: "org/model-name"
  finetune_type: "lora"
  target_components: ["transformer"]
```

## Phase 4: Verification

Also read: `topics/parity_testing.md` for the 4-layer verification protocol.

- [ ] `load_pipeline()` successfully loads the model
- [ ] `preprocess_func()` produces correct cached tensors
- [ ] `inference()` generates valid images/videos
- [ ] `forward()` computes loss without errors
- [ ] Training runs end-to-end with GRPO for ≥2 steps
- [ ] LoRA weights save and reload correctly
- [ ] Registry entry resolves correctly: `get_model_adapter_class('my-model')`
- [ ] Example YAML config is valid and complete

## Common Pitfalls

1. **Forgetting to set `preprocessing_modules`** — causes text encoder to stay on GPU, OOM during training
2. **Wrong `target_module_map`** — LoRA applied to wrong components, no training effect
3. **Mismatched `_shared_fields`** — data corruption during batch collation
4. **Not handling `enable_preprocess=False`** — encoding components not loaded at inference time
5. **Inconsistent custom field types across samples** — if a custom sample field is `Tensor` on some samples and `List[Tensor]` on others, `gather_samples` will fall back to slow pickle-based `gather_object`. Always canonicalize to a single type in `__post_init__`; prefer `List[Tensor]` for variable-length data.
6. **Wrong `images`/`condition_images` convention** — `preprocess_func()`, `encode_image()`, and `inference()` all operate at **batch level**: `images` is `List[List[Image.Image]]` and `condition_images` is `List[List[Tensor(C,H,W)]]`, where the outer list indexes samples in the batch and the inner list holds each sample's condition images. Never pass a flat `List[Image]` or `List[Tensor]`.
