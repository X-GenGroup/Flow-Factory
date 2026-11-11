# CUDA training image

## Published v0.5.0 image

The [v0.5.0 release](https://github.com/X-GenGroup/Flow-Factory/releases/tag/v0.5.0) publishes the
CUDA training image built from source and recipe commit `ced2cd9b0465a2c9d733b9b8aed4d4351f950a22`.
It uses Python 3.12, CUDA 12.9, Torch 2.10.0, Accelerate 1.15.0 and Diffusers 0.42.1, and installs
a built Flow-Factory wheel. Its release evidence bundle records this registry digest:

```bash
docker pull ghcr.io/x-gengroup/flow-factory@sha256:d11cf62a22622f570111225bd18af797f4fe3505bad11f8ea4ad461d400610c4
# Versioned alias:
docker pull ghcr.io/x-gengroup/flow-factory:0.5.0

docker run --rm -it --gpus all --ipc=host \
  -v /path/to/configs:/configs:ro \
  -v /path/to/outputs:/outputs \
  -v /path/to/huggingface:/cache/huggingface \
  -e HF_HOME=/cache/huggingface \
  ghcr.io/x-gengroup/flow-factory:0.5.0 \
  ff-train /configs/train.yaml
```

Set output paths under `/outputs`; weights and optional reward services are not bundled. Use an
NVIDIA driver compatible with the image's CUDA runtime and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
See [parallel training](../guidance/parallelism/quickstart.md), [installation](../guidance/installation.md)
and [publishing](../docs/release/publishing.md) for layout constraints, current-source environments
and artifact verification. The later v0.5.1 source tag and main's Torch 2.10–2.14 work do not
replace this image or its pinned dependencies. Prefer a digest for an immutable deployment;
`latest` is a moving alias.

## Historical v0.2.0 environment

The published v0.2.0 image contains Python 3.12, CUDA 12.9.1, PyTorch 2.10, the released Diffusers 0.41
package, Accelerate, DeepSpeed, W&B and FFmpeg shared libraries. It targets Linux `amd64` with an
NVIDIA GPU and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
Use a driver compatible with CUDA 12.9; CUDA minor-version compatibility on older drivers has
[feature restrictions](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).

Current-source builds use Python 3.12 and Diffusers 0.42.1, matching the project dependency floor.
The published v0.2.0 image and its immutable build records retain their original dependencies.

## Use v0.2.0

Pull the published image:

```bash
docker pull ghcr.io/x-gengroup/flow-factory:0.2.0
```

The `0.2.0` and `v0.2.0` aliases identify the same image. At its original publication, `latest` also pointed to
it; that historical observation does not identify the current `latest` image. For a fixed deployment identity, use
`ghcr.io/x-gengroup/flow-factory@sha256:f83d18bc98e9b3a5c5ccf522b242519fd05b66643bf9606fe9c33612e9814e56`.

Download the tested source build bundle and verification records from the
[v0.2.0 Release](https://github.com/X-GenGroup/Flow-Factory/releases/tag/v0.2.0):

```bash
gh release download v0.2.0 --repo X-GenGroup/Flow-Factory --dir release-v0.2.0
cd release-v0.2.0
sha256sum -c SHA256SUMS
# Follow BUILDING.md to build the local flow-factory:0.2.0 image.
```

The [publication run](https://github.com/X-GenGroup/Flow-Factory/actions/runs/41805515570)
passed the image build and network-disabled CPU smoke before pushing. Anonymous manifest/config
reads and access to every layer were verified; this is not a full second image download or a GPU
test. The existing `0.1.0` image and `v0.1.0` source tag remain unchanged.

For a custom configuration and persistent output/cache:

```bash
docker run --rm -it --gpus all --ipc=host \
  -v /path/to/configs:/configs:ro \
  -v /path/to/outputs:/outputs \
  -v /path/to/huggingface:/cache/huggingface \
  -e HF_HOME=/cache/huggingface \
  ghcr.io/x-gengroup/flow-factory:0.2.0 \
  ff-train /configs/train.yaml
```

Set the configuration's output path under `/outputs`. Supply tracking/service credentials through
your runtime environment or secret store. Model weights and optional reward services are not part
of the image. See [multi-node instructions](../multinode_examples/README.md) for distributed launch.

The v0.2.0 application source is the immutable commit `55dcd694`. Its historical Python package
metadata still reports `0.1.0`; the container version and revision labels identify this release.
The build recipe is recorded separately because the source tag predates these packaging fixes.
The release includes the recipe, source build bundle, resolved packages and verification record.

In the v0.2.0 environment, TP requires FSDP2 and at least two FSDP shards; CP/EP were future work. The released Diffusers
package does not include the pinned development APIs required by FLUX.3 and Qwen-Image 2.1; those
adapters require their documented development environment. Bagel and reward-specific optional
extras are also separate installs. See [the baseline limitations](https://github.com/X-GenGroup/Flow-Factory/blob/main/roadmap/milestones/v0.2.md).

## Verify an image

The commands below use the current v0.5.0 image; the preceding v0.2.0 record is historical.

The publish workflow runs the CPU check inside the built image with networking disabled. It tests
CLI entry points, DeepSpeed/TorchCodec imports and a small SD3 denoiser forward/backward/update;
it downloads no weights and is not a GPU training qualification matrix.

```bash
docker run --rm --network none ghcr.io/x-gengroup/flow-factory:0.5.0 \
  python docker/smoke.py

docker run --rm --network none --gpus 'device=0' \
  ghcr.io/x-gengroup/flow-factory:0.5.0 \
  python docker/smoke.py --device cuda
```

## Build locally

Build the current checkout from the repository root:

```bash
docker buildx build --platform linux/amd64 \
  -f docker/docker-cuda/Dockerfile \
  --build-arg VERSION=dev \
  --build-arg VCS_REF="$(git rev-parse HEAD)" \
  --build-arg BUILD_RECIPE_REF="$(git rev-parse HEAD)" \
  -t flow-factory:local --load .
```

To rebuild v0.2.0, download the release's `flow-factory-v0.2.0-docker.tar.gz` build bundle and
verify it with `SHA256SUMS`. It contains the tagged application source with the recorded Docker
recipe overlaid. Extract into an empty directory, use the build command above with `VERSION=0.2.0`,
and use the source/recipe revisions from the release's `provenance.txt` instead of `git rev-parse`.
The dependency constraints pin the key CUDA/model packages; `requirements-resolved.txt` records
the complete resolved environment of the validated build. Once published, the image digest is
the reproducible deployment identity; rebuilding against live package/OS indexes can change bytes.

Dependency installation is cached separately from application source. The build context excludes
local checkpoints, scratch worktrees, environments and output directories. You may override
`PYTORCH_INDEX_URL` with a mirror for the same CUDA 12.9 wheels; changing the CUDA family requires
changing and validating the base image and paired package versions together.

## Publish a release image

The [container workflow](../.github/workflows/container.yml) accepts an existing source tag and
checks it out independently of the build recipe. After this workflow is on main:

```bash
gh workflow run container.yml -f tag=v0.2.0 -F publish=false
# For a new version whose source tag already exists:
gh workflow run container.yml -f tag=v0.3.0 -F publish=true
```

The job checks registry write access by opening an empty blob upload before an expensive build.
It cancels that upload when supported; GHCR may return HTTP 405 for cancellation after accepting
the upload. This cleanup limitation does not invalidate the successful write check. No image data
or tag is published by the probe, and authorization failures still stop publication.
It records source and recipe commits, builds `linux/amd64`, checks the image without network
access, then publishes `VERSION` and `vVERSION` aliases. Stable versions also update `latest`;
prereleases do not. Existing version aliases are never overwritten. Publishing uses the workflow's
repository-scoped `GITHUB_TOKEN`; no personal registry token is placed in the Dockerfile or image.
The GHCR package must grant this repository Actions write access and remain public for anonymous
community pulls. The job uploads the build bundle, resolved environment, image identity and smoke
result as an artifact. Attach the verified artifact/checksums to the corresponding GitHub Release.

## Troubleshooting

| Symptom | Check |
|---|---|
| CUDA unavailable | Host driver, Container Toolkit and `--gpus`; the CPU smoke intentionally does not require a GPU |
| TorchCodec import fails | Use the paired constraints and FFmpeg shared libraries; the `imageio` executable is not a replacement |
| Model class absent from Diffusers | Verify the adapter's released-versus-development dependency requirements |
| NCCL shared-memory error | Provide `--ipc=host` or a workload-appropriate `--shm-size` |
| Registry push denied | Grant the repository Actions access in the GHCR package settings |
