# CUDA training image

The release image contains Python 3.12, CUDA 12.9.1, PyTorch 2.10, the released Diffusers 0.41
package, Accelerate, DeepSpeed, W&B and FFmpeg shared libraries. It targets Linux `amd64` with an
NVIDIA GPU and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
Use a driver compatible with CUDA 12.9; CUDA minor-version compatibility on older drivers has
[feature restrictions](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).

## Use v0.2.0

Download the tested build bundle and instructions from the
[v0.2.0 Release](https://github.com/X-GenGroup/Flow-Factory/releases/tag/v0.2.0):

```bash
gh release download v0.2.0 --repo X-GenGroup/Flow-Factory --dir release-v0.2.0
cd release-v0.2.0
sha256sum -c SHA256SUMS
# Follow BUILDING.md to build the local flow-factory:0.2.0 image.
```

Prebuilt GHCR publication is currently blocked by package write authorization. The image builds
and CPU smoke checks passed; registry publication did not. Check the release page for updated
availability before using the intended `ghcr.io/x-gengroup/flow-factory:0.2.0` / `v0.2.0` aliases.
Existing `0.1.0`, `latest` and the `v0.1.0` source tag have not been changed by this release.

For a custom configuration and persistent output/cache:

```bash
docker run --rm -it --gpus all --ipc=host \
  -v /path/to/configs:/configs:ro \
  -v /path/to/outputs:/outputs \
  -v /path/to/huggingface:/cache/huggingface \
  -e HF_HOME=/cache/huggingface \
  flow-factory:0.2.0 \
  ff-train /configs/train.yaml
```

Set the configuration's output path under `/outputs`. Supply tracking/service credentials through
your runtime environment or secret store. Model weights and optional reward services are not part
of the image. See [multi-node instructions](../multinode_examples/README.md) for distributed launch.

The v0.2.0 application source is the immutable commit `55dcd694`. Its historical Python package
metadata still reports `0.1.0`; the container version and revision labels identify this release.
The build recipe is recorded separately because the source tag predates these packaging fixes.
The release includes the recipe, source build bundle, resolved packages and verification record.

TP requires FSDP2 and at least two FSDP shards. CP/EP remain future work. The released Diffusers
package does not include the pinned development APIs required by FLUX.3 and Qwen-Image 2.1; those
adapters require their documented development environment. Bagel and reward-specific optional
extras are also separate installs. See [the baseline limitations](https://github.com/X-GenGroup/Flow-Factory/blob/main/roadmap/milestones/v0.2.md).

## Verify an image

The publish workflow runs the CPU check inside the built image with networking disabled. It tests
CLI entry points, DeepSpeed/TorchCodec imports and a small SD3 denoiser forward/backward/update;
it downloads no weights and is not a GPU training qualification matrix.

```bash
docker run --rm --network none flow-factory:0.2.0 \
  python docker/smoke.py

docker run --rm --network none --gpus 'device=0' \
  flow-factory:0.2.0 \
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

The job checks registry write access with a cancelled empty blob upload before an expensive build.
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
