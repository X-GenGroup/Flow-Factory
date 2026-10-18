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
│   ├── deepspeed       DeepSpeed >= 0.18.3
│   ├── quantization    bitsandbytes >= 0.45.3
│   ├── wandb           Weights & Biases tracking
│   ├── swanlab         SwanLab tracking
│   ├── nvidia          xformers >= 0.0.35, nvidia-ml-py
│   ├── bagel           flash-attn, opencv-python
│   ├── geneval         mmcv, mmengine, mmdet, open_clip_torch
│   ├── geneval2        scipy (exact GenEval2 GM parity)
│   └── all             deepspeed + quantization
└── [tool.*]            black, isort config
```

## Core Dependencies

The authoritative list is `pyproject.toml` `[project.dependencies]` (20+ packages). Key ones:

| Package | Supported Version | Purpose |
|---------|-------------|---------|
| `torch` | >= 2.10.0 | PyTorch core, AdamW, and the native `torch.optim.Muon` API |
| `torchvision` | >= 0.25.0 | Vision utilities; paired baseline for PyTorch 2.10 |
| `torchaudio` | >= 2.10.0 | Audio I/O (audio / audio-video models, CLAP) |
| `torchcodec` | >= 0.10.0 | Runtime used by TorchAudio 2.10 `load` / `save` APIs |
| `transformers` | >= 4.57.1 | Text encoders, tokenizers |
| `diffusers` | >= 0.40.0 | Diffusion pipelines, schedulers, MiniMax H3 and LTX2 APIs |
| `accelerate` | >= 1.14.0 | Distributed training, mixed precision, and max reduction |
| `av` | >= 17.0.0 | CPU video/audio decoding with Python 3.10 support |
| `peft` | >= 0.17.0 | LoRA, parameter-efficient fine-tuning |
| `datasets` | >= 3.3.2 | Dataset loading |
| `huggingface-hub` | >= 0.35.3 | Model/dataset downloads |
| `protobuf` / `sentencepiece` | >= 6.33.2 / >= 0.2.1 | T5 tokenizer (Flux, SD3.5) |
| `pydantic` | >= 2.8.0 | Config dataclass validation |

## Key Compatibility Notes

### DeepSpeed
- Only **ZeRO-1** and **ZeRO-2** are supported. ZeRO-3 is broken for reward model sharding (constraint #10).
- DeepSpeed is optional — Accelerate alone handles most distributed scenarios.

### Muon
- The `torch>=2.10.0` baseline includes `torch.optim.Muon`. Runtime capability detection remains
  authoritative for vendor or custom builds that may omit the API.
- Muon is supported with DDP and FSDP2. The pre-load optimizer/backend validator rejects
  DeepSpeed and FSDP1 before pretrained weights are loaded.

### TorchAudio and TorchCodec
- TorchAudio 2.10 delegates `torchaudio.load` and `torchaudio.save` to TorchCodec. The matching
  baseline is TorchCodec 0.10, and its native extension also needs FFmpeg 4–8 shared libraries.
- The CUDA Docker image installs FFmpeg. Non-container installations must provide compatible
  system FFmpeg libraries; the `imageio[ffmpeg]` executable bundle does not provide those shared
  libraries.

### PyAV
- The tested floor is `av>=17.0.0`. Python 3.10 resolves to a compatible 17.x release because
  PyAV 18 requires Python 3.11; Python 3.11 and newer may resolve to later PyAV releases.
- Flow-Factory uses the container, stream, frame, and resampler APIs available in PyAV 17; it does
  not require a PyAV 18-only API.

### diffusers
- Use the released `diffusers>=0.40.0` package as the authoritative API. The repository submodule
  may be used for upstream development, but must not silently override the declared runtime dependency.
- `flux3` and `qwen-image-2.1` need the pinned submodule (`7ecb1e1b`), which requires Python 3.11+
  and `huggingface-hub>=1.32`; released diffusers keeps Python 3.10 support for the other adapters.
- Model adapters depend on specific pipeline classes from diffusers. Major version bumps may rename or remove pipeline classes.
- `load_pipeline()` in each adapter returns a `DiffusionPipeline`-compatible object; breaking changes in diffusers' pipeline API require adapter updates.

### transformers
- Used for text encoders (T5, CLIP). Flow-Factory does not patch transformers internals — standard HuggingFace loading is used.

### bagel extra
- Bagel remains a registered model adapter, but its heavyweight runtime packages are optional. Install with `pip install -e ".[bagel]"` before using `model_type: "bagel"`.

### geneval / geneval2 rewards
- `geneval` needs `pip install -e ".[geneval]"` (mmcv / mmengine / mmdet + open_clip_torch) for Mask2Former detection.
- `geneval2` (`geneval2_soft_tifa`) needs `pip install -e ".[geneval2]"` (scipy) for exact GenEval2 geometric-mean parity; Qwen3-VL ships with core `transformers`, and a pure-Python gmean is used when scipy is absent.

### HPSv2 reward
- The PyPI `hpsv2` package pins `protobuf<4`, conflicting with Flow-Factory's `protobuf>=6`. It is **not** an optional extra. Install it without dependencies after Flow-Factory: `uv pip install hpsv2 --no-deps` (runtime works with protobuf 6+).

### accelerate
- Primary distributed backend. `accelerator.prepare()` wraps a single `ModelBundle` (all target components) plus the optimizer as one root (constraint #9).
- Version 1.14 or newer is required because distributed runtime-identity checks use
  `Accelerator.reduce(..., reduction="max")`; earlier releases do not implement max reduction.
- Online generation uses framework samplers. Finite SFT/offline-DPO data uses PyTorch's official
  `DistributedSampler`; that already-sharded loader is not prepared via Accelerate.

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

- UP: [`constraints.md` #10](constraints.md#10-deepspeed-zero-3-is-unsupported), [Architecture Configuration Hierarchy](architecture.md#configuration-hierarchy)
- PEER: [Component Variants](topics/component_variants.md)
