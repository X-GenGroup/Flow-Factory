# Installation and compatible environments

Use Python 3.11 or newer and PyTorch 2.11 or newer. Install Torch and torchvision wheels for your
CUDA environment, then install Flow-Factory. The project's minimum dependencies describe API
requirements; they do not qualify every later dependency combination for distributed training.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
ff-train --help
```

For reproducible release qualification, use the exact constraints in
[`docker/docker-cuda/constraints.txt`](../docker/docker-cuda/constraints.txt), which also back the CUDA image recipe:

```bash
python -m pip install -c docker/docker-cuda/constraints.txt torch torchvision \
  --index-url https://download.pytorch.org/whl/cu129
python -m pip install -c docker/docker-cuda/constraints.txt torchaudio torchcodec \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -c docker/docker-cuda/constraints.txt -e '.[deepspeed,wandb]'
python -m pip check
```

The release recipe uses Python 3.12, Torch 2.11.0, Accelerate 1.15.0 and Diffusers 0.42.1.
CP also requires Accelerate >=1.15 and Diffusers >=0.42.1. The measured Torch versions within
the current package floor are 2.11.0, 2.12.1, 2.13.0 and 2.14.1, with Accelerate 1.15.0 and
Diffusers 0.42.1. The [precision qualification](../docs/development/validation/torch210-214.md)
records the exact environments and selected CPU/GPU cases. Torch keeps its native RMSNorm
behavior, and CP backward recomputation restores the forward autocast policy. Untested newer
versions emit a warning for CP. Model, topology, and precision contracts still apply. Report
failures with the configuration, package versions, and traceback.
On Torch 2.11 Hopper workers, the framework warns and disables cuDNN SDPA after reproducing
incorrect backward gradients; other native SDPA backends remain available.
The release recipe uses the TorchAudio 2.11.1 and TorchCodec 0.12.0 CPU wheels from the
[Torch 2.11 qualification](../docs/development/validation/torch211.md).
Flux3 and Bagel use the released Diffusers package.
Historical receipts using Python 3.10 or development Diffusers builds describe those executions,
not the current installation recipe.

| Need | Extra or system dependency |
|---|---|
| DDP or FSDP2, including declared TP/SP/CP paths | Core install; CP's qualified tuple above |
| DeepSpeed ZeRO-1/2 | `.[deepspeed]`; ZeRO-3 is not supported |
| Bagel | `.[bagel]`; a compatible FlashAttention build and OpenCV |
| Quantization | `.[quantization]`; not automatic admission for every parallel placement |
| Tracking | `.[wandb]` or `.[swanlab]` |
| GenEval detection rewards | `.[geneval]`, including its external detector dependencies |
| Audio via TorchAudio/TorchCodec | Compatible FFmpeg shared libraries; the imageio executable alone is insufficient |

`.[all]` currently means DeepSpeed plus quantization, not every optional model/reward dependency.
Install only the extras needed by the selected recipe. Optional packages may require native builds;
their absence does not make an unrelated adapter unavailable.

## Published artifacts versus current source

| Source/artifact | Environment and scope |
|---|---|
| [v0.5.0 release](https://github.com/X-GenGroup/Flow-Factory/releases/tag/v0.5.0) | Published wheel, sdist, CUDA image and durable evidence; Torch 2.10.0 release constraints |
| [v0.5.1 source tag](https://github.com/X-GenGroup/Flow-Factory/tree/v0.5.1) | Subsequent Torch 2.10/2.11 support; no separate GitHub Release or image |
| [v0.6.0 Release](https://github.com/X-GenGroup/Flow-Factory/releases/tag/v0.6.0) | Wheel, sdist, CUDA image and evidence. Torch >=2.11, with targeted checks on 2.11–2.14 |

The [container guide](../docker/README.md) records the published image digest and source identity.
Use that immutable image for the release environment, or install current source with the selected
qualification's compatible dependency tuple. Historical v0.2 artifacts retain their own environment.
