# Installation and compatible environments

Use Python 3.11 or newer. Install PyTorch and its matching torchvision/torchaudio wheels for your
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
python -m pip install -c docker/docker-cuda/constraints.txt -e '.[deepspeed,wandb]'
python -m pip check
```

The release recipe uses Python 3.12, Torch 2.10.0, Accelerate 1.15.0 and Diffusers 0.42.1.
CP explicitly checks **Torch 2.10.x, Accelerate 1.15.x and Diffusers 0.42.x (at least 0.42.1)**.
Installing an unconstrained future version can therefore satisfy package metadata while failing
CP preflight. Flux3 and Bagel use the released Diffusers package; neither needs a Git revision pin.
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

The [container guide](../docker/README.md) records image/source identities. The existing v0.2 image
is a historical release, not a v0.5 environment. A v0.5 image must be built and published from its
reviewed release source before it is advertised for use.
