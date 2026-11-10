# Build from repository root:
#   docker buildx build --platform linux/amd64 -f docker/docker-cuda/Dockerfile -t flow-factory:local --load .
#
# Apple Silicon: always pass --platform linux/amd64 for this training image.
# Python 3.12 + uv: dependency installs use `uv pip` (same resolution model as pip, faster).
FROM nvidia/cuda:12.9.1-devel-ubuntu22.04

ARG VERSION=dev
ARG VCS_REF=unknown
ARG BUILD_RECIPE_REF=unknown
LABEL org.opencontainers.image.source="https://github.com/X-GenGroup/Flow-Factory"
LABEL org.opencontainers.image.version="${VERSION}"
LABEL org.opencontainers.image.revision="${VCS_REF}"
LABEL io.flow-factory.build-recipe-revision="${BUILD_RECIPE_REF}"
LABEL org.opencontainers.image.description="Flow-Factory GPU training image (CUDA 12.9 + Python 3.12)"

# Pin uv version; see https://docs.astral.sh/uv/guides/integration/docker/
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /uvx /usr/local/bin/

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
COPY docker/docker-cuda/constraints.txt /app/docker/docker-cuda/constraints.txt
# Stub src dir so setuptools find_packages can resolve without full source
RUN mkdir -p /app/src/flow_factory && touch /app/src/flow_factory/__init__.py

ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu129
RUN uv pip install \
    torch==2.10.0 \
    torchvision==0.25.0 \
    torchaudio==2.10.0 \
    torchcodec==0.10.0 \
    --index-url "${PYTORCH_INDEX_URL}"

RUN uv pip install --constraint docker/docker-cuda/constraints.txt ".[deepspeed,wandb]"

# --- Phase 2: install the complete application as a wheel ---
COPY . /app
RUN uv build --wheel --out-dir /tmp/ff-dist \
    && uv pip install --no-deps --reinstall /tmp/ff-dist/*.whl \
    && rm -rf /tmp/ff-dist

# Fail fast if any layer is broken
RUN uv pip check && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python docker/smoke.py

# Intentionally root: DeepSpeed / NCCL may need elevated IPC access.
CMD ["/bin/bash"]
