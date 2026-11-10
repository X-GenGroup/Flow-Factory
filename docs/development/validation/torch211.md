# Torch 2.11 qualification

This qualification extends the v0.5 implementation to Torch 2.11 without changing the
published v0.5.0 image or its immutable evidence. Torch 2.14 is outside this change.
The package floor remains Torch 2.10. Torch 2.11 is an additional qualified environment;
a microbenchmark is not grounds for ending 2.10 support.

## Reproduce the environment

Use a separate Python 3.12 environment. Do not upgrade an active training environment.

```bash
python -m venv .venv-torch211
. .venv-torch211/bin/activate
python -m pip install torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install torchaudio==2.11.1 torchcodec==0.12.0 \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -c config/validation/dependencies/torch211.txt -e '.[deepspeed,wandb]'
python -m pip check
```

Run these commands from the repository root. Use matching FFmpeg shared libraries, as described
in [installation](../../../guidance/installation.md). An FFmpeg executable alone is insufficient.
The qualification host uses FFmpeg 7 libraries from an existing Conda environment through a
process-local library path; no system or driver upgrade is required.

The CPU media wheels are deliberate: default PyPI TorchAudio 2.11.1 requires `libcudart.so.13`,
which is absent from the selected CUDA 12.8 environment. TorchCodec's CPU build serves the
framework's host media I/O; Torch remains CUDA-enabled. Native GPU media decoding is not a
claim of this tuple. TorchAudio 2.11 and TorchCodec >=0.12 have an upstream Torch >=2.11
compatibility contract, but each selected binary still needs compatible native libraries.

## Version policy and ownership

CP's required API floors remain Torch >=2.10, Accelerate >=1.15 and Diffusers >=0.42.1.
The measured Torch lines are 2.10 and 2.11. Unmeasured versions above the floors warn and
continue. An advisory is not a numerical certificate: report reproducible failures with the
configuration, traceback and complete dependency versions. The model, topology, precision,
objective and state checks still apply; runtime errors are not swallowed. Explicit model-owned
source pins retain their meaning.

Torch 2.11 fixes nested FSDP activation-checkpoint replay after a preceding backward reshards
shared units. The backend now installs its replay-unshard backport only below 2.11. Two-rank
regression coverage creates two graphs before either backward, compares every gradient and three
AdamW updates to a dense reference, and checks that 2.11 installs zero backport hooks. The
Accelerate preparation-cache release and adapter memory policies solve different problems and
remain in place.

DTensor's 2.11 `Partial` backward changes do not justify deleting the framework's explicit TP
submesh and gradient-reduction semantics. SP numerical and packed projection witnesses check
those boundaries, including a negative control where the wrong reduction leaves loss unchanged.

## Hopper attention policy

Torch 2.11.0+cu128 selects cuDNN SDPA for the short-sequence Flux BF16 fixture on this
SM90 host. The CP1 fixed-operand reference had gradient norm **14.0543**, versus **0.030727**
for CP2; the same 2.10 control passed at approximately **0.03070** on both layouts.
A separate, non-distributed native Flux probe reproduced incorrect gradients or NaNs
with cuDNN, while Flash/Efficient/Math produced finite gradients of approximately **0.0369**.
This is not accepted asynchronous accumulation noise. The original failure is retained.

`Accelerator` now applies a version-bounded policy after selecting its worker device:
on Torch 2.11 and Hopper (CUDA capability major 9), call the public
`torch.backends.cuda.enable_cudnn_sdp(False)` and warn. The setting persists for the worker
process, covering rollout, training, activation recomputation and Diffusers CP backward
replay. It leaves other SDPA backends and cuDNN convolutions enabled. No import-time CUDA
initialization or third-party monkeypatch is added. Explicit cuDNN backend contexts can
override the switch and are **not** part of this qualification.

The native attention regression compares every trainable Flux BF16 gradient and the output
exactly against explicit Flash SDPA. An independent strided SDPA probe compares output and
all Q/K/V derivatives against FP64 math with operator-local BF16 bounds and zero/half/double
negative controls. The distributed controls retain their
FP32 negative controls and BF16 per-element/whole-signal bounds. No tolerance was relaxed.
Other GPU architectures and arbitrary dependency tuples are not newly qualified here.

Upstream also reports cuDNN backward problems in
[masked Hopper inputs](https://github.com/pytorch/pytorch/issues/196678). That is related
context, not proof that the local unmasked fixture has the same root cause.

## Selected evidence

The source of each run, dependency tuple, commands, JUnit results and raw logs are recorded in
the task's `.scratch/torch211/` directory. The following scenarios cover independent contracts;
they are not a model-by-algorithm Cartesian matrix. Tiny native fixtures prove functional and
numerical behavior, not pretrained convergence or end-to-end throughput.

| Witness | Topology | Contract |
|---|---|---|
| CP admission, backend loading, runtime policy, media and SP/packed numerical tests | CPU/Gloo | Required floors, advisory-only newer versions, gradient ownership and real codec/features |
| Native Flux attention | One GPU | Exact native/Flash model gradients; BF16 SDPA derivatives against FP64 |
| Nested FSDP checkpoint replay | DP2 | Two outstanding graphs, dense gradients/updates, no 2.11 backport hooks |
| Backend preparation | DDP2 / ZeRO2 | Single prepared root and both trainable role routes |
| Wan SFT LoRA state | DP2 x CP2 | Exact loaded state/RNG, EMA and continued updates |
| Snapshot forward | DP2 x TP2 | Full training, checkpointing and pending policy graph across reference/EMA writes |
| Kontext Guard full ordered overlap | DP2 x TP2 x CP2, SP on | Reference input, packed projections, full SP, native replay and lifecycle |
| Flux DPPO LoRA ready overlap | DP2 x TP2 x CP2, SP off | Alternate parameter mode and ready-order ownership |

Performance results are reported separately from correctness. A fixed native SD3 model uses
four width-512 blocks, 1024 image tokens, 128 text tokens, BF16 autocast and full AdamW updates.
Both Torch versions use CUDA 12.8 and the same major framework dependencies. Each run warms up
for ten updates and measures three groups of twenty updates; compilation startup is reported
separately. GPU competition is paused only for the measurement and restored afterward.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_torch_training.py --mode eager --report eager.json
CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_torch_training.py --mode compile --report compile.json
```

Run each command in both environments. Compilation warmup includes the first ten updates;
steady-state latency excludes it. Use fresh Inductor cache directories when comparing cold
compilation. The 2.11 benchmark applies the same Hopper SDPA policy as training.

## Measured performance

On this H800-equivalent SM90 node (CUDA runtime reports capability 9.0), with CUDA 12.8
wheels and the same non-Torch dependencies:

| Torch | Eager median | Regional compile median | Compile warmup (10 steps) | Peak allocation (eager / compile) |
|---|---:|---:|---:|---:|
| 2.10.0 | 15.754 ms | 12.190 ms | 9.873 s | 764.16 / 765.03 MiB |
| 2.11.0 | 15.779 ms | 11.725 ms | 9.888 s | 764.16 / 765.03 MiB |

The eager difference is about 0.15%, which does not establish an improvement. The compiled
workload is about 3.8% faster in 2.11; measured peak allocation is unchanged. This is one fixed
native workload, not evidence of a universal training speedup. Compile uses the public
`Module.compile()` on each transformer block; SD3 does not declare Diffusers
`_repeated_blocks`, so calling `compile_repeated_blocks()` directly is not a valid benchmark.
The failed setup attempts are retained separately from the successful measurements.

Keep Torch 2.10 as the package minimum. Use the 2.11 tuple when its native FSDP checkpoint
handling or measured workload benefit is useful. No `torch.compile` + CP, FlashAttention-4,
GPU media decoding, arbitrary CUDA wheel family or other GPU architecture is newly certified
by these experiments. Audio coverage here is actual CPU I/O/features beside CUDA training.

## Upstream references

- [Torch 2.11 changes](https://github.com/pytorch/pytorch/releases/tag/v2.11.0)
- [Nested FSDP checkpoint fix](https://github.com/pytorch/pytorch/pull/171779)
- [TorchAudio compatibility](https://github.com/pytorch/audio)
- [TorchCodec compatibility](https://github.com/meta-pytorch/torchcodec)
