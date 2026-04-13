# Dependency Management

**Read when**: Changing `pyproject.toml`, deps, install commands.

---

## Package Manager

Flow-Factory uses **pip** with `setuptools` as the build backend. Install with:

```bash
pip install -e "."              # Core only
pip install -e ".[all]"         # Core + DeepSpeed + quantization
pip install -e ".[deepspeed]"   # Core + DeepSpeed only
```

## Dependency Layout

```
pyproject.toml
├── [project.dependencies]              Core deps (always installed)
├── [project.optional-dependencies]
│   ├── deepspeed       DeepSpeed >= 0.15.4
│   ├── quantization    bitsandbytes >= 0.45.3
│   ├── wandb           Weights & Biases tracking
│   ├── swanlab         SwanLab tracking
│   ├── nvidia          xformers, nvidia-ml-py
│   └── all             deepspeed + quantization
└── [tool.*]            black, isort config
```

## Core Dependencies

| Package | Min Version | Purpose |
|---------|-------------|---------|
| `torch` | >= 2.6.0 | PyTorch core |
| `torchvision` | >= 0.19.0 | Vision utilities |
| `transformers` | >= 4.57.1 | Text encoders, tokenizers |
| `diffusers` | >= 0.36.0 | Diffusion pipelines, schedulers |
| `accelerate` | >= 1.11.0 | Distributed training, mixed precision |
| `peft` | >= 0.17.0 | LoRA, parameter-efficient fine-tuning |
| `datasets` | >= 3.3.2 | Dataset loading |
| `pydantic` | >= 2.8.0 | Config dataclass validation |

## Key Compatibility Notes

### DeepSpeed
- Only **ZeRO-1** and **ZeRO-2** are supported. ZeRO-3 is broken for reward model sharding (constraint #10).
- DeepSpeed is optional — Accelerate alone handles most distributed scenarios.

### diffusers
- Model adapters depend on specific pipeline classes from diffusers. Major version bumps may rename or remove pipeline classes.
- `load_pipeline()` in each adapter returns a `DiffusionPipeline`-compatible object; breaking changes in diffusers' pipeline API require adapter updates.

### transformers
- Used for text encoders (T5, CLIP). Flow-Factory does not patch transformers internals — standard HuggingFace loading is used.

### accelerate
- Primary distributed backend. `accelerator.prepare()` is used for trainable modules and optimizer only (constraint #9).
- The dataloader uses custom samplers and is NOT prepared via accelerate.

### peft
- Provides LoRA functionality. Applied via `BaseAdapter.apply_lora()` to components listed in `target_components`.

## Common Dependency Issues

1. **torch / torchvision version mismatch** — torchvision must match the torch major version. Check the PyTorch compatibility matrix.
2. **diffusers pipeline rename** — When diffusers renames a pipeline class, `load_pipeline()` in the affected adapter must be updated.
3. **CUDA version mismatch** — torch wheels are CUDA-version-specific. Ensure the installed torch matches the system CUDA.
4. **DeepSpeed + accelerate conflict** — Certain DeepSpeed versions may conflict with newer accelerate releases. Pin both if issues arise.
5. **protobuf / sentencepiece** — Required for T5 text encoders (used by Flux, SD3.5). Missing these causes import errors during preprocessing.

## Updating Dependencies

1. Edit version constraint in `pyproject.toml` under `[project.dependencies]` or `[project.optional-dependencies]`.
2. Reinstall: `pip install -e ".[all]"`
3. Run tests: `pytest`
4. Verify training runs end-to-end with at least one model adapter.


## Cross-refs

- `constraints.md` #10 (DeepSpeed ZeRO-3 unsupported)
- `architecture.md` "Configuration Hierarchy" (hparams structure)
