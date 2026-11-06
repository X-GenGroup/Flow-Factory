# Tensor and sequence parallelism

Tensor parallelism (TP) divides one denoiser across ranks. Sequence parallelism (SP) is an
adapter-specific overlay on that TP group: it divides selected token activations across the
same ranks. It does not introduce an additional independent mesh dimension. The supported
training backend for TP is FSDP2; DDP and DeepSpeed ZeRO-2 remain supported with TP disabled.

```yaml
parallelism:
  tp_size: 2
  sequence_parallel: true
```

On a four-rank launch, the omitted `dp_shard_size` resolves to two. Each pair of TP peers
receives the same sample; the two FSDP replicas receive different samples. Larger launches
derive a larger shard degree. Set `dp_shard_size` explicitly only to require a fixed product;
an inconsistent launch is an error. See [configuration rules](parallelism.md#configuration-and-runtime).
Compare layouts at fixed **logical replica count and
sample/update geometry**, rather than fixing physical rank count and inadvertently changing the
batch. A two-rank primitive test does not qualify four-rank FSDP2 plus TP training.

Omitting `parallelism` preserves the existing backend choice. With TP enabled, omitted
`sequence_parallel` follows the adapter default below. `false` selects TP alone. `true` requires
an explicit adapter SP plan and rejects an adapter without one before loading model weights.
Invalid head divisibility, unsupported backend/optimizer pairings and incompatible runtime
settings also fail through the normal preflight checks. Checkpoint-specific limits below still
apply even when the forward layout is supported.

## Capability and qualification table

A declared plan is a capability, not a claim that every checkpoint and workload has been
qualified. Q0 here means random-weight real model structures on CPU/Gloo, Q2 means actual CUDA
backend preparation and updates, and Q3 means the named pretrained weights. These levels are
defined in [validation guidance](validation.md). Historical runs remain historical evidence.

| Model identifier | TP / SP default | Current qualification and limits |
|---|---|---|
| `wan2_t2v` | TP; SP **on** | Q0 per-sample and per-token timestep layouts, including LoRA. Q2 tiny real Wan blocks under FSDP2 + TP2, with SP and activation checkpointing. Q3 Wan2.1-T2V-1.3B **denoiser LoRA updates**, TP2 + SP + FSDP2; see the bounded claim below. |
| `wan2_i2v` | TP; SP off | Q0 plan expansion covers Wan2.1 added image KV projections and Wan2.2 without them. Shared T2V block evidence does not qualify a full I2V pipeline or the complete two-transformer schedule. |
| `sd3-5` | TP; SP **on** | Q0 real SD3 blocks, dual attention, LoRA, gradients and updates. Earlier Medium real-weight runs are recorded in PR #270; they do not certify this revision or Large. Large has 38 heads and therefore supports TP2, not TP4/8. |
| `qwen-image`, `qwen-image-edit-plus` | TP; SP off | Q0 real dual-stream blocks, Edit-2511 per-token modulation, checkpointing, gradients and updates. No new pretrained-weight qualification. |
| `qwen-image-2.1` | TP; SP off | **Excluded from qualified TP training** while the real-layout numerical discrepancy below remains unresolved. A declared plan, isolated parity, and successful GRPO execution do not close that gap. |
| `ltx2_t2av`, `ltx2_i2av` | TP; SP off | Q0 LTX-2 / 2.3 real blocks, per-token timesteps, gated attention and prompt modulation, gradients and updates. No pretrained-weight qualification. |
| `sensenova` | TP; SP off | Q0 real generation branch behind odd-length T2I/I2I prefixes, gradients and updates. SP leaves the prefix whole; image-token divisibility is required. No pretrained-weight qualification. |
| `z-image` | TP; SP off | Q0 real blocks and updates; TP2 ceiling from 30 attention heads. No pretrained-weight qualification. |
| `flux1`, `flux1-kontext` | TP; no SP plan | Q0 FLUX.1 blocks and LoRA updates; Kontext shares its plan. Packed single-stream projections remain replicated. No full Kontext/pipeline qualification. |
| `flux3` | TP; no SP plan | Q0 real FLUX.3 blocks, full/LoRA gradients and updates in the separate pinned Diffusers tuple below. No pretrained-weight qualification; another FLUX model cannot substitute for this case. |
| `flux2`, `flux2-klein` | No TP or SP plan | TP is refused. Do not infer TP support from FLUX.3 or from upstream inference parallelism. |
| `bagel` | TP; no SP plan | Q0 real vendored feed-forward branches, gradients and updates, plus plan expansion on the real tree. The separate optional-dependency tuple below supplies FlashAttention. Attention remains replicated and is outside the feed-forward numerical case; no pretrained-weight qualification. |
| `minimax-h3-t2va`, `minimax-h3-fl2va`, `minimax-h3-ref2va` | TP; SP off | Chunked qk-norm primitives have Q0 evidence. Full TP preparation has a known materialization capacity limit on 80 GB cards; use FSDP2 without TP for the recorded supported workloads. |

The registry-wide default/explicit-mode contract is executable in
[`test_parallelism_capabilities.py`](../tests/models/test_parallelism_capabilities.py).
The qualified training statement in this revision is intentionally bounded to the named Wan
layouts and weights. Other rows retain their measured structural evidence and stated gaps.

## Qwen-Image-2.1: execution success and numerical qualification

The [merged TP PR](https://github.com/X-GenGroup/Flow-Factory/pull/270) reports real-weight GRPO
runs with TP2 and SP disabled that complete, plus an isolated small-model gradient parity test.
The [merged numerical investigation](https://github.com/X-GenGroup/Flow-Factory/blob/55dcd69433deb11159d5c03c22689518ce6f315e/.agents/knowledge/topics/tensor_parallel.md)
still records a `2.7e-4` forward difference and a `3.4%` gradient-norm difference in a real
`ff-train` input layout, versus a `0.19%` reduction-order baseline. Those are different oracles:
completion and isolated parity do not establish per-parameter parity for real prefix KV caches,
block-causal segments and scattered image tokens. This qualification does not claim a fix.

To remove the exclusion, retain a real-layout reproducer and compare forward outputs, gradient
membership, every affected gradient and updates against the unsharded reference on the final
revision with fixed logical samples. Explicit SP rejection of odd joint-prefix lengths is a
separate shape limitation. An OOM is a separate capacity outcome. Neither explains the numerical
discrepancy, and turning SP off does not by itself establish numerical qualification.

## What the Wan qualification proves

[`test_tp_training_qualification.py`](../tests/distributed/test_tp_training_qualification.py)
uses the production TP applier and Flow-Factory's Accelerate subclass, followed by actual FSDP2
preparation on a `(dp_shard=2, tp=2)` CUDA mesh. Its reference processes the same two logical
samples without sharding. Four successive AdamW updates verify:

- Forward tensors and every gradient, including expected zero gradients and gradient membership.
- Parameter and optimizer identity after preparation, gradient clipping, every actual update,
  and populated Adam moments. Update deltas are checked separately from parameter values.
- Frozen LoRA base parameters remain bitwise unchanged and receive no gradients.
- Activation-checkpointed SP retains a bounded live CUDA allocation after repeated iterations.

The fixture uses AdamW with learning rate `1e-3`, epsilon `1e-4`, weight decay `0.01`,
and global gradient clipping at `0.5`. Elementwise checks use `rtol=2e-4, atol=2e-6`;
an additional per-parameter gradient check bounds maximum error by `5e-4` times that
gradient tensor's maximum magnitude. Q3 adds an absolute FP32 floor of `2e-7`, while Q2
adds none. Update deltas have the same scale bound plus a four-epsilon parameter-roundoff
allowance and Q3's `2e-7` floor. All parameter names and zero-gradient tensors remain checked.

The Q3 floor is calibrated to this fixture, not a universal TP tolerance: an unsharded
batched-versus-sequential control with identical weights already differs by `4.88e-8` on a
gradient whose maximum is `7.68e-5` (relative `6.35e-4`). After one AdamW update, the TP
trajectory differs by `1.30e-7` in that gradient and `1.40e-7` in its update. A pure relative
`5e-4` cutoff would reject the unsharded control. Q3 retains that batching control and records
its result with the worst TP gradient/update in each rank's JSON, while asserting the combined
bound. Recalibrate with evidence when changing precision, optimizer settings or geometry.

The Q2 fixture covers full tuning and LoRA, TP-only and checkpointed SP layouts. Its two Wan
blocks have real Diffusers structure and random weights; it is not a pretrained-model test.
The Q3 fixture loads only the locally staged transformer of
`Wan-AI/Wan2.1-T2V-1.3B-Diffusers`, snapshot
`0fad780a534b6463e45facd96134c9f345acfa5b`. It checks 600 LoRA parameter gradients/updates and
825 frozen base parameters per step. The inputs are small latent tensors and synthetic text
embeddings: VAE encoding, prompt processing, reward adapters, generated-media quality and full
training recipes are outside this Q3 claim. The test hashes the actual config and weight files
into each rank's JSON evidence; a model cache filename is not assumed to be a content hash.

Observed on 2026-11-04: Python 3.10.22, PyTorch 2.10.0+cu129, Diffusers 0.41.0,
Transformers 5.20.2, Accelerate 1.15.0 and PEFT 0.21.2, on four local NVIDIA L20Y 80 GB GPUs.
Bagel and FLUX.3 Q0 cases additionally passed in an isolated optional-model environment:
Python 3.13.16, PyTorch 2.10.0+cu129, Diffusers 0.42.0.dev0 at
`48cdbaef5ee3d718aaee392b44f2099ae511c4a5`, Transformers 5.20.2, Accelerate 1.15.0,
PEFT 0.21.2 and FlashAttention 2.8.3.post2. Their five previously unavailable baseline cases
(two Bagel plan cases, Bagel/FLUX.3 region parity, Bagel capability classification) were run
successfully there; no fake attention implementation or silent skip supplies that evidence.
These separate tuples must be preserved in reports instead of claiming one environment tested all rows.

The local CUDA run used FP32 without TF32. These results do not qualify arbitrary dependency upgrades,
BF16 numerical tolerances, larger TP degrees, other attention kernels or cross-node execution.

The [2026-11-05 merged-baseline follow-up](integration-qualification.md) re-runs both local CUDA
cases on the merged source and adds two-node Q2 evidence: the four Wan full/LoRA TP/SP cases with
TP crossing nodes, plus checkpointed SP/LoRA with DP crossing nodes. The pretrained Q3 case stays
single-node; the follow-up does not expand its precision, dependency or full-pipeline claims.

Run the Q2 and Q3 cases separately so missing weights cannot be mistaken for a passed test:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 python -m pytest -q \
  tests/distributed/test_tp_training_qualification.py::test_wan_fsdp2_tp_updates_match_fixed_replica_reference \
  --basetemp=.scratch/tp-q2

FF_WAN_TP_MODEL_PATH=/absolute/path/to/transformer \
  PYTHONPATH=src OMP_NUM_THREADS=1 python -m pytest -q \
  tests/distributed/test_tp_training_qualification.py::test_pretrained_wan_lora_updates_match_fixed_replica_reference \
  --basetemp=.scratch/tp-q3
```

Select four visible GPUs before these commands. Each test writes per-rank JSON below its
`--basetemp` directory. A required case with insufficient GPUs or missing model files is blocked;
a pytest skip is not passing evidence. Keep the final source commit, catalog/plan identity,
dependency tuple and these artifacts together. Cross-node validation remains pending.

## Across-head normalization precision

Wan and LTX-2 normalize q/k over the head dimension split by TP. Both standalone and paired
norms share one FP32 activation view between their direct and statistic derivative branches.
They combine those gradients before converting back to FP16/BF16; separately rounding the
branches can corrupt nearly cancelling gradients despite matching forward outputs.
The [two-rank regression](../tests/distributed/test_across_head_norm_pairing.py) checks these
cases against FP64 and preserves the paired collective count and cleanup behavior.

This correction retains one FP32 view per local query/key slice for backward. Historical peak
memory measurements predate it; no new peak-memory or throughput guarantee is implied.

## Checkpoint and compatibility limits

The existing [TP checkpoint tests](../tests/distributed/test_tensor_parallel_checkpointing.py)
exercise model-only gathers and basic optimizer-state round trips over the composed mesh.
FSDP2 fixed-topology exact resume now also persists EMA, reference copies and algorithm
snapshots through rank-local auxiliary payloads; see [parameter state](parameter_state.md) for
the compatibility contract and two-/four-GPU next-update qualification. A successful model-only
save does not establish exact resume of auxiliary state.

With TP disabled, this qualification changes no production execution path. Reuse older evidence
only for its exact unchanged scope; do not relabel historical campaigns as a run of this revision.
The identity-seeded offline VAE change already merged with TP affects SD3.5 offline DPO even
when TP is disabled: it intentionally differs from the old process-RNG behavior. Universal
bitwise equivalence to a pre-TP checkpoint is not promised.

H3's TP capacity failure happens while materializing/sharding parameters before FSDP preparation,
not while allocating long-sequence activations. Lower resolution or SP cannot repair that bound;
meta initialization plus sharded loading is a separate future loading change. [Wan CP and TP/SP+CP SFT](context_parallel.md) have separate experimental capabilities and
composition evidence. This TP/SP qualification alone does not imply those paths. EP remains
later roadmap work.
