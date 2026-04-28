# Docker (CUDA) — Flow-Factory Training Image

Pre-built GPU training image for Flow-Factory: CUDA 12.9, Python 3.12, PyTorch 2.8, DeepSpeed, and W&B — ready to run `ff-train` out of the box.

## Prerequisites

| Requirement | Minimum |
|---|---|
| **OS** | Linux (x86_64) |
| **GPU** | NVIDIA with Compute Capability ≥ 7.0 |
| **NVIDIA Driver** | ≥ 535 |
| **Docker** | ≥ 24.0 |
| **NVIDIA Container Toolkit** | [Install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) |

> **Apple Silicon (M1–M4):** The image is `linux/amd64` only. You can use it for smoke tests and CLI checks via Docker Desktop's QEMU emulation, but there is no CUDA-capable GPU on macOS — real training requires a Linux + NVIDIA host.

## Quick start

### A. Pull a prebuilt image (recommended)

```bash
docker pull ghcr.io/x-gengroup/flow-factory:0.1.0
# or: docker pull ghcr.io/x-gengroup/flow-factory:latest
```

Run:

```bash
docker run --rm -it --gpus all ghcr.io/x-gengroup/flow-factory:0.1.0
```

### B. Build locally

Clone with the `diffusers` submodule (required):

```bash
git clone --recursive https://github.com/X-GenGroup/Flow-Factory.git
cd Flow-Factory
# or, if already cloned: git submodule update --init --recursive
```

Build from the **repository root**:

```bash
docker buildx build --platform linux/amd64 \
  -f docker/docker-cuda/Dockerfile \
  -t flow-factory:local --load .
```

Run:

```bash
docker run --rm -it --gpus all flow-factory:local
```

## Running a training job

### Single-GPU example

```bash
docker run --rm -it --gpus all \
  ghcr.io/x-gengroup/flow-factory:0.1.0 \
  ff-train examples/grpo/lora/flux1/default.yaml
```

### With custom configs and output directory

```bash
docker run --rm -it --gpus all \
  -v /path/to/my-configs:/app/configs:ro \
  -v /path/to/outputs:/app/outputs \
  ghcr.io/x-gengroup/flow-factory:0.1.0 \
  ff-train /app/configs/my_experiment.yaml
```

### Multi-GPU with DeepSpeed

```bash
docker run --rm -it --gpus all --ipc=host --shm-size=16g \
  ghcr.io/x-gengroup/flow-factory:0.1.0 \
  ff-train examples/grpo/lora/flux1/default.yaml
```

> **Note:** `--ipc=host` (or `--shm-size=16g`) is required for DeepSpeed / NCCL multi-GPU communication.

## Configuration

| Option | Example |
|---|---|
| **W&B tracking** | `-e WANDB_API_KEY=your_key` |
| **Mount data** | `-v /host/data:/app/data:ro` |
| **Mount outputs** | `-v /host/outputs:/app/outputs` |
| **Shared memory** | `--ipc=host` or `--shm-size=16g` |
| **PyTorch mirror** (build-time) | `--build-arg PYTORCH_INDEX_URL=https://your-mirror/whl/cu129` |

The working directory inside the image is `/app` (a copy of the repo at build time). For live-editing, mount a bind volume over `/app` or a subdirectory.

Do not commit secrets; use environment variables or your orchestrator's secret store.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `nvidia-smi` not found in container | NVIDIA Container Toolkit not installed | [Install the toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) and restart Docker |
| `CUDA out of memory` | Batch size too large for GPU VRAM | Reduce batch size, enable DeepSpeed ZeRO-3, or use FSDP |
| Build fails on `diffusers` install | Submodule not initialized | Run `git submodule update --init --recursive` |
| Every source change triggers full rebuild | Expected with `COPY . /app` | The Dockerfile uses two-phase COPY for layer caching; ensure `pyproject.toml` is unchanged for cache hits |

## Building tips

- **Layer caching**: The Dockerfile copies `pyproject.toml` first and installs dependencies, then copies the full source. As long as `pyproject.toml` doesn't change, PyTorch/DeepSpeed layers are cached.
- **Fast iteration**: For development, mount your source as a volume (`-v $(pwd):/app`) instead of rebuilding the image.
- **PyTorch index override**: Pass `--build-arg PYTORCH_INDEX_URL=...` to use a mirror or different CUDA version.
- **Apple Silicon**: Always pass `--platform linux/amd64` when building or running on Mac.

---

<details>
<summary><strong>For maintainers: publishing to GHCR</strong></summary>

After building locally (e.g. as `flow-factory:local`), tag and push:

```bash
docker tag flow-factory:local ghcr.io/x-gengroup/flow-factory:0.1.0
docker tag flow-factory:local ghcr.io/x-gengroup/flow-factory:latest
echo $GITHUB_PAT | docker login ghcr.io -u $GITHUB_USER --password-stdin
docker push ghcr.io/x-gengroup/flow-factory:0.1.0
docker push ghcr.io/x-gengroup/flow-factory:latest
```

Requires a GitHub PAT with **`write:packages`** scope and appropriate org access. Prefer CI (e.g. GitHub Actions) for repeatable releases.

Users who need reproducible deploys should pin by **digest** rather than moving tags.

</details>
