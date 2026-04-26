# Build from repository root (includes diffusers submodule):
#   docker buildx build --platform linux/amd64 -f docker/docker-cuda/Dockerfile -t flow-factory:local --load .
#
# Apple Silicon: always pass --platform linux/amd64 for this training image.
# Python 3.12 + uv: dependency installs use `uv pip` (same resolution model as pip, faster).
FROM nvidia/cuda:12.9.1-devel-ubuntu22.04

LABEL org.opencontainers.image.source="https://github.com/X-GenGroup/Flow-Factory"
LABEL org.opencontainers.image.version="0.1.0"
LABEL org.opencontainers.image.description="Flow-Factory GPU training image (CUDA 12.9 + Python 3.12)"

# Pin uv version; see https://docs.astral.sh/uv/guides/docker/
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /uvx /usr/local/bin/

ENV DEBIAN_FRONTEND=noninteractive
# No pip cache inside image layers
ENV UV_NO_CACHE=1
ENV UV_LINK_MODE=copy
# Skip .pyc generation: smaller layers; both uv and Python runtime agree
ENV UV_COMPILE_BYTECODE=0
ENV PYTHONDONTWRITEBYTECODE=1

SHELL ["/bin/bash", "-eu", "-o", "pipefail", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    ffmpeg \
    build-essential \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN uv venv /opt/ff-venv --python 3.12
ENV PATH="/opt/ff-venv/bin:$PATH"
ENV VIRTUAL_ENV=/opt/ff-venv
ENV UV_PROJECT_ENVIRONMENT=/opt/ff-venv

WORKDIR /app

# --- Phase 1: install dependencies (cached unless pyproject.toml changes) ---
COPY pyproject.toml README.md /app/
# Stub src dir so setuptools find_packages can resolve without full source
RUN mkdir -p /app/src/flow_factory && touch /app/src/flow_factory/__init__.py

ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu129
RUN uv pip install \
    torch==2.8.0 \
    torchvision==0.23.0 \
    torchaudio==2.8.0 \
    --index-url "${PYTORCH_INDEX_URL}"

RUN uv pip install ".[deepspeed,wandb]"

# --- Phase 2: copy full source and re-link editable installs ---
COPY . /app
RUN uv pip install --no-deps -e .
RUN uv pip install -e ./diffusers

# Fail fast if any layer is broken
RUN python -c "import deepspeed, torch, wandb; print('torch', torch.__version__)" && ff-train --help >/dev/null

# Intentionally root: DeepSpeed / NCCL may need elevated IPC access.
CMD ["/bin/bash"]
