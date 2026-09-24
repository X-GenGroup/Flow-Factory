# GPU Validation Matrix

This document defines the real-weight GPU validation contract and records the PR #220 execution
result. The historical direct runs establish successful launch and training completion; current
framework-wide changes are governed by the exact-commit merge gate below.

## Framework-upgrade merge gate

Broad changes to the execution kernel, training dataflow, distributed backend, model loading or
preparation, sampler or batch geometry, reward or advantage pipeline, or optimizer/checkpoint
infrastructure must pass this gate before merge. A localized adapter, reward, or algorithm change
may use a narrower affected matrix only when it does not alter one of those shared layers.

The machine-readable source of truth is
`config/gpu_validation/framework_upgrade.yaml`. Validate it and list the exact jobs with:

```bash
python scripts/validate_gpu_campaign.py
python scripts/validate_gpu_campaign.py --list-jobs
```

The current manifest contains **63 mandatory jobs**: 54 core end-to-end jobs plus nine focused
reward-overlap jobs. The core is **18 model/algorithm pairs x 3 backends**:

| Profile | Algorithms | Backend cells | Purpose |
|---|---:|---:|---|
| Qwen-Image-2.1 OCR text-to-image | GRPO | 3 | Coupled reward training, GDPO aggregation, async AES + HY-OCR overlap, subgroup tiles |
| SD3.5 text-to-image | NFT, SFT, offline DPO, online DPO | 12 | Decoupled reward, both dataset paradigms, and rank-local online pairing |
| Bagel ordered multi-image editing with `per_device_batch_size=2` | TDM | 3 | Packed-sequence microbatches and two-role reward-free distillation |
| FLUX.2 Klein Base 4B ordered multi-image editing | all six algorithms | 18 | One image adapter across every acquisition/feedback paradigm |
| MiniMax H3 text-to-audio-video | all six algorithms | 18 | Structured video/audio trajectories, rewards, codecs, and multi-role replay |

The core algorithm axis is exactly `{grpo, nft, sft, offline-dpo, online-dpo, tdm}` and every row runs
with DDP, DeepSpeed ZeRO-2, and FSDP2. A generated-acquisition job completes one rollout followed
by every optimizer/role update declared in the manifest. An offline job traverses one finite
dataset containing exactly 32 records at `per_device_batch_size=1` and
`gradient_accumulation_steps=1`, producing exactly one optimizer step on 32 ranks. TDM must report
both its generator and fake-role updates.

The nine supplemental jobs do not repeat the backend Cartesian product. Together with four reused
DDP core cells, they form a constrained pairwise critical-path matrix across all overlap-capable
trainers (`grpo`, `grpo-guard`, `dppo`, `nft`, `awm`, `crd`, `dgpo`, `online-dpo`, and `tdm-r1`),
the four compatible placements (`subgroup_tile`, `global_batch`, `global_tile`, and `rank_local`),
single- and multi-source rewards, `sum` and `gdpo`, and `ready` and `ordered` scheduling. It also
contains a Bagel GRPO case at `per_device_batch_size=2`. This is intentionally not a blind
reward x layout x trainer x backend Cartesian product: online DPO requires rank-local complete
groups, DGPO requires complete groups in a global microbatch, local GPU rewards cannot satisfy the
CPU-async overlap contract, and the three core Qwen/NFT/online-DPO rows already cross the overlap
kernel with every backend.

### Reward deployment and overlap topology

The manifest selects rewards by task and by the capability being tested:

| Profile | Deployment | Server topology | Use |
|---|---|---|---|
| `remote-aes-async` | External pointwise HTTP service; CPU client | 9 replicas x TP8 = 72 GPUs | Image quality and single-source async paths |
| `remote-hy-ocr-async` | External pointwise OpenAI-compatible service; CPU client | DP8 = 8 GPUs | OCR-sensitive fast-service and packed-batch boundary path |
| `remote-aes-plus-hy-ocr-async` | Both remote services, independently scheduled | 80 reward GPUs total | Multi-reward tail latency, GDPO, and ready scheduling |
| `local-clap-plus-imagebind-sync` | In-process CUDA rewards | Training ranks | H3 audio/video semantics; no overlap claim |

Tracked configuration contains endpoint environment-variable names, never cluster addresses or
temporary client implementations. The campaign workspace supplies those values and clients, and
they must remain outside the pull request. Evidence must include a real health/inference probe,
the exact deployed topology, per-source score cardinality and p50/p95/max latency, and the
resolved reward configuration. Writing `async_reward: true` without proving the service actually
ran is not a pass.

Every enabled overlap job uses pointwise remote rewards, CPU clients, an independent executor pool
per reward source, `global_std=false`, and at least two independently ready work units. Jobs marked
`overlap_observation: required` must record a positive
`timing/reward_overlap/optimization_overlap_seconds` and prove optimization began while some reward
requests were still pending. `ready` is the production default and may select complete
work units out of acquisition order. The ordered Qwen canary holds model, data, rewards, sampler,
and seed fixed while changing only the selection policy; it verifies the supported deterministic
fallback without weakening the ready-path requirement. Out-of-order completion is recorded but is
not itself mandatory because external service timing is nondeterministic.

The Bagel `per_device_batch_size=2` HY-OCR cell is explicitly `observe_only`: it still must execute
the async tile-stream path, score every sample, expose at least two work units, and report the full
timing series, but a healthy DP8 OCR service can drain all requests before optimization starts.
That zero-overlap outcome is a measured fast-service boundary, not a failed concurrency claim.
DPPO uses the slower AES deployment so every overlap-capable trainer still has a separate
`required` cell that proves physical reward/optimization concurrency. This distinction prevents
artificial sleeps or deliberately under-provisioned reward servers from becoming part of the gate.

The minimum is enforced twice: manifest construction rejects an overlap cell whose resolved cycle
contains fewer than two optimizer work units, and result validation rejects runtime evidence with
fewer than two completed units. Optimizer counts are therefore selected per concrete profile, not
silently changed in the trainer: production-shaped NFT/AWM/DGPO overlap canaries use two updates,
and rank-local online DPO uses three because the 96 groups place three complete pairs on each rank.
The H3 synchronous boundary keeps its one-update NFT/online-DPO cycle because it makes no overlap
claim. Multi-role TDM-R1 must report one update for each of its fake, surrogate, and generator
roles; omitting the reward-trained surrogate is a failed cycle, not a two-role TDM equivalent.

### Gate geometry

All jobs use 32 training GPUs (four 8-GPU nodes), BF16, evaluation disabled, checkpoint saving
disabled, and a 1024-pixel output long edge. Image reward jobs retain the production-shaped
`unique_sample_num_per_epoch=96`, `group_size=16`, and `per_device_batch_size=1` geometry. The
Qwen and FLUX subgroup-tile jobs use eight-rank subgroups; online DPO remains rank-local because
pair construction requires rank-local complete groups. Bagel TDM uses
`per_device_batch_size=2`, `group_size=1`, and `unique_sample_num_per_epoch=128`, with sample
shuffling disabled so every packed microbatch retains its original composition.

H3 retains the expensive semantic constraints rather than the production sample count: a
`[576, 1024]` output, 124 frames at 24 fps (5.17 seconds), neutral guidance, and two denoising
steps. Its reward algorithms use the smallest non-degenerate `group_size=2` and
`unique_sample_num_per_epoch=32`; TDM uses one sample per rank. CLAP and ImageBind run
synchronously and cover audio/text and audio/video semantics respectively. H3 does not claim
reward/optimization overlap. Image profiles remain the production-scale overlap benchmark; the
H3 slice is an end-to-end structured-media correctness and explicit non-overlap boundary gate.

Most production image paths retain `unique_sample_num_per_epoch=96` and `group_size=16`. One
supplemental `global_tile` job deliberately uses `group_size=24`: with a 32-rank global forward
batch, a complete work unit spans three global batches and contains four groups. Keeping K=16 in
that cell would collapse `global_tile` into `global_batch` and would not test tiled closure.

FLUX.2 Klein must use `black-forest-labs/FLUX.2-klein-base-4B`, not the 9B default found in some
starting recipes, and every record must contain exactly two ordered reference images. Offline
FLUX jobs use the `multi_image_to_image` fixture contract. H3 offline jobs use the
`text_to_audio_video` fixture contract and retain ordered `(video, audio)` supervision.
The recipe path in each run is only the algorithm-specific starting point: profile, geometry,
workload, backend, and one-cycle fields from the manifest take precedence in the resolved config.

### Launch and evidence contract

Launch every job through `ff-train`/Accelerate with the backend config declared in the manifest.
Raw `torchrun` bypasses Accelerate plugin construction and therefore counts only as DDP; naming a
raw launch `zero2` or `fsdp2` is not evidence. Each job must record the observed runtime backend:
`MULTI_GPU` for DDP, `DEEPSPEED` with `zero_stage=2`, or `FSDP` with `fsdp_version=2`.

Results must be bound to the full tested commit SHA and the SHA-256 digest of the manifest. Every
job attaches its command, resolved configuration, environment manifest, complete logs, and compact
metrics. The observations must prove all ranks completed, losses/gradient metrics were finite,
the intended parameters changed, the world size was 32, the runtime backend matched, and the exact
one-cycle update counts completed. They also echo the exact task, model/checkpoint, geometry,
starting recipe, workload, reward profile, and offline fixture contract from the manifest; changing
only the job label cannot satisfy the validator. Validate the final bundle with:

```bash
python scripts/validate_gpu_campaign.py --results /path/to/results.json
```

Core result job IDs are `{profile}__{backend}__{algorithm}`; supplemental IDs are declared
verbatim in the manifest. The result set must exactly equal all 63 jobs. `skipped`, `capacity`,
`infrastructure`, timeout, or any other non-passing state
blocks merge; it remains useful failure evidence but is not a pass. If the code changes after the
campaign, the exact-commit requirement makes the evidence stale and the affected jobs must be run
again.

This gate establishes one-cycle end-to-end correctness, not convergence or throughput. Keep
reward-overlap timing and scalability benchmarks as separate campaigns with their full timing
series. The historical PR #220 matrix below remains model-family semantic certification and does
not replace the current exact-commit gate.

## PR #220 result

The complete dynamic smoke scope reached its successful terminal marker for **144/144** unique
jobs:

- **120/120 main jobs**: 10 semantic modes x DDP/DeepSpeed ZeRO-2/FSDP2 x
  GRPO/SFT/offline DPO/TDM.
- **22/22 formal variant-gate jobs**: Wan 2.1/2.2 T2V and I2V routing variants (including
  corrected redundant FLF2V coverage), all six strict Wan A14B dual-transformer routing jobs,
  LTX 2.3 T2AV/I2AV, and H3 FL2VA first-plus-last coverage.
- **2/2 supplemental jobs**: Flux1-Kontext image-to-image SFT and offline DPO.

All **132/132** checkpoint-variant backend/algorithm cells also passed static configuration and
contract validation. Four positive Muon jobs passed for DDP/FSDP2 with SFT and mixed-role TDM;
the DeepSpeed ZeRO-2 negative gate rejected Muon before model loading as intended.

These are reduced-geometry, finite-length execution smokes. They establish archived real-weight
execution coverage for model loading, distributed backends, optimizer paths, and finite-data
termination. Capability-sensitive instrumented gates additionally checked routing and updates,
but earlier direct runs did not uniformly capture the complete command, environment, and metrics
artifact set required below. These results do not claim convergence, long-run reward improvement,
quality parity, or numerical parity.

## Environment gate

Record the following once for every validation campaign:

- Flow-Factory commit SHA and parent stacked-PR SHA.
- Python, CUDA, PyTorch, Accelerate, DeepSpeed, and Diffusers versions.
- `diffusers>=0.40.0`; do not mix results from an older official release.
- GPU model, GPU count, per-GPU memory, driver, and NCCL versions.
- Exact model revision and dataset-content hash.
- `PYTHONPATH` and editable-install source must resolve to the tested worktree.

Run a configuration/import preflight before allocating model weights:

```bash
python -m compileall -q src/flow_factory
python -c "import diffusers; assert tuple(map(int, diffusers.__version__.split('.')[:2])) >= (0, 40)"
```

## Main experiment matrix

The main campaign is the Cartesian product of the ten semantic modes, three
distributed backends, and four algorithms below: **10 x 3 x 4 = 120 jobs**.
Do not collapse first-frame and first/last-frame Wan rows: they exercise
different condition layouts and active masks.

### Semantic modes

| ID | Mode | Representative checkpoint | Required input | Required output |
|---|---|---|---|---|
| `sd35-t2i` | SD3.5 text-to-image regression anchor | `stabilityai/stable-diffusion-3.5-medium` | prompt | image |
| `bagel-mri2i` | Bagel ordered multi-reference-images-to-image regression anchor | `ByteDance-Seed/BAGEL-7B-MoT` | prompt plus exactly two ordered images per sample | image |
| `wan-t2v` | Wan text-to-video | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | prompt | video |
| `wan-i2v-first` | Wan first-frame-to-video | `Wan-AI/Wan2.2-TI2V-5B-Diffusers` | exactly one first-frame image | video |
| `wan-flf2v` | Wan first/last-frame-to-video | `Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers` | ordered first and last images | video |
| `ltx2-t2av` | LTX2 text-to-audio-video | `Lightricks/LTX-2` | prompt | ordered video and audio |
| `ltx2-i2av` | LTX2 image-to-audio-video | `Lightricks/LTX-2` | prompt plus one image | ordered video and audio |
| `h3-t2va` | MiniMax H3 text-to-video-audio | `MiniMaxAI/MiniMax-H3` | prompt | ordered video and audio |
| `h3-fl2va` | MiniMax H3 sparse first/last-frame-to-video-audio | `MiniMaxAI/MiniMax-H3` | `first_frame`, `last_frame`, or both image slots | ordered video and audio |
| `h3-ref2va` | MiniMax H3 ordered-reference-to-video-audio | `MiniMaxAI/MiniMax-H3` | 1-12 ordered heterogeneous references, including image or video | ordered video and audio |

The public V2 schema uses the `type` discriminator and an optional input-only
`slot`. An explicit slot reserves its adapter-declared semantic argument;
unslotted media is only a positional shorthand that fills remaining slots in
declaration order. Standard Wan2.1 I2V checkpoints require `first_frame`, the
dedicated FLF2V checkpoint requires both endpoints, and Wan2.2 I2V-A14B retains
an optional VAE-only `last_frame`; H3 FL accepts either slot or both. Supervision
media does not repeat independent condition-image objects. Its video is
nevertheless the full configured output sequence, whose first and/or last
endpoint must correspond to the supplied endpoint conditions.

### Backends

| ID | `config_file` | Required observation |
|---|---|---|
| `ddp` | `config/accelerate_configs/multi_gpu.yaml` | Every rank executes the same branch and step count. |
| `zero2` | `config/deepspeed/deepspeed_zero2.yaml` | Optimizer state is partitioned and both policy/reference scopes complete. |
| `fsdp2` | `config/accelerate_configs/fsdp2.yaml` | Adapter wrap plan, DTensor parameters, checkpointing, and component routing remain valid. |

### Algorithms and stopping rules

| ID | `train.trainer_type` | Exact smoke length | Evaluation |
|---|---|---|---|
| `grpo` | `grpo` | two training epochs, one optimizer step per epoch | `eval.eval_freq: 0` |
| `sft` | `sft` | exactly two rank-local dataloader batches | `eval.eval_freq: 0` |
| `offline-dpo` | `offline-dpo` | exactly two rank-local dataloader batches | `eval.eval_freq: 0` |
| `tdm` | `tdm` | two training epochs, one generator/fake update cycle per epoch | `eval.eval_freq: 0` |

For SFT and offline DPO, build a finite dataset whose official
`DistributedSampler` yields exactly two batches per rank, set
`gradient_accumulation_steps: 1`, and run one complete dataloader epoch. One
offline epoch means one complete dataloader traversal. Sampler tail padding is
standard PyTorch behavior and does not change that definition.

For GRPO and TDM, use `max_epochs: 2`,
`gradient_step_per_epoch: 1`, the smallest valid group size, and no evaluation.
Reduce resolution, frame count, and inference steps only within each adapter's
declared geometry constraints. MiniMax H3 must retain at least five seconds of
24-fps output even in a low-resolution smoke run.

### Concrete smoke profiles

Use these geometry and sampling overlays unless a real checkpoint rejects the
reduced geometry. Any fallback must stay valid for the adapter contract and be
recorded in the resolved YAML; do not silently return to a large quality recipe.

| Mode | Starting recipe | Train geometry | Steps | Algorithm-specific notes |
|---|---|---|---|---|
| `sd35-t2i` | `examples/grpo/lora/sd3_5/default.yaml` | `resolution: 256` | `num_inference_steps: 2` | Use the checked-in SD3.5 SFT/offline-DPO/TDM recipe as the algorithm overlay. |
| `bagel-mri2i` | `examples/grpo/lora/bagel/i2i.yaml` | `resolution: 256` | `num_inference_steps: 2` | Every row has exactly two ordered references; keep `shuffle_samples: false`. |
| `wan-t2v` | `examples/grpo/lora/wan21/t2v.yaml` | `resolution: 240`, `num_frames: 5` | `num_inference_steps: 2` | Target video has at least five frames and carries its source `fps`. |
| `wan-i2v-first` | `examples/grpo/lora/wan22/i2v.yaml` | `resolution: 240`, `num_frames: 5` | `num_inference_steps: 2` | Use TI2V-5B and exactly one condition image. |
| `wan-flf2v` | `examples/grpo/lora/wan21/i2v.yaml` | `resolution: 240`, `num_frames: 5` | `num_inference_steps: 2` | Use the dedicated FLF2V checkpoint with two ordered condition images. |
| `ltx2-t2av` | `examples/grpo/lora/ltx2/t2av.yaml` | `resolution: [128, 192]`, `num_frames: 9`, `frame_rate: 24.0` | `num_inference_steps: 2` | AV targets cover the exact 9-frame clock; audio carries `sample_rate`. |
| `ltx2-i2av` | `examples/grpo/lora/ltx2/i2av.yaml` | `resolution: [128, 192]`, `num_frames: 9`, `frame_rate: 24.0` | `num_inference_steps: 2` | One condition image; verify the first latent frame is inactive in the loss. |
| `h3-t2va` | `examples/grpo/lora/minimax_h3_t2va/debug.yaml` | `resolution: [64, 96]`, `num_frames: 124`, `frame_rate: 24.0` | `num_inference_steps: 2` | Preserve the released five-second minimum and neutral guidance. |
| `h3-fl2va` | `examples/grpo/lora/minimax_h3_fl2va/default.yaml` plus H3 debug geometry | `resolution: [64, 96]`, `num_frames: 124`, `frame_rate: 24.0` | `num_inference_steps: 2` | The two offline records are explicit `first_frame`-only and `last_frame`-only cases. Cover both slots together in the additional variant gate. |
| `h3-ref2va` | `examples/grpo/lora/minimax_h3_ref2va/default.yaml` plus H3 debug geometry | `resolution: [64, 96]`, `num_frames: 124`, `frame_rate: 24.0` | `num_inference_steps: 2` | Preserve heterogeneous global reference order; include image, video, and audio references. |

For the two offline algorithms, construct each ready-to-use alias from its independent pinned
public dataset before launching distributed workers:

```bash
python -m dataset.offline_smoke.prepare \
  --algorithm sft \
  --profile "${MODE}" \
  --world-size "${WORLD_SIZE}" \
  --per-device-batch-size 1 \
  --batches-per-rank 2

python -m dataset.offline_smoke.prepare \
  --algorithm offline-dpo \
  --profile "${MODE}" \
  --world-size "${WORLD_SIZE}" \
  --per-device-batch-size 1 \
  --batches-per-rank 2
```

Run the command once per shared filesystem rather than once per distributed rank. The SFT and DPO
repositories have the same alias/input distribution but different supervision and self-contained
media. `image-i2i` is an additional contract gate outside the 120-job main matrix. The catalog and
materializer operate on arbitrary ordered output media sequences; currently published fixtures are
limited to the image, video, and `(video, audio)` outputs implemented by registered adapters.

For GRPO set `group_size: 2`, `unique_sample_num_per_epoch: 1`, and
`gradient_accumulation_steps: auto`; this avoids a degenerate one-candidate
advantage while retaining one optimizer step per epoch. For TDM set
`group_size: 1` and `gradient_accumulation_steps: auto`; with the two-step smoke
profile the resolved accumulation count must be divisible by two. SFT and
offline DPO use `gradient_accumulation_steps: 1`. Use at least two distributed
ranks for every backend; increase the rank count only for checkpoint memory
capacity, without changing the two iteration/batch stopping rule.

The offline fixture must contain exactly `2 * world_size` records and use
`per_device_batch_size: 1`, so `DistributedSampler(drop_last=False)` yields two
non-padded batches on every rank. Demonstration and preference fixtures use the
same input distribution. Preference rows must use distinct chosen/rejected
media files with identical geometry. Before launch, validate conditioned target
endpoints against their inputs: LTX2 I2AV and first-conditioned Wan/H3 targets
must begin with the supplied first frame within the fixture tolerance; a target
with a last-frame condition must end with that supplied frame. Store the
comparison metric and tolerance in the fixture manifest.

The campaign generator should materialize one job ID for every Cartesian-product
cell as `{mode}__{backend}__{algorithm}` and assert that the set has exactly 120
unique IDs before submission. A skipped or infrastructure-blocked cell remains
in the result table with its failure classification; it must not silently reduce
the matrix.

## Checkpoint-variant coverage

The 120-job main matrix uses one checkpoint per semantic mode. Run the following
additional variant gate before declaring a family generally supported. The
minimum gate is DDP plus SFT and offline DPO for two batches each; configuration
construction and static contract validation must also pass under all three
backends and all four algorithms.

| Family mode | Additional checkpoint variants |
|---|---|
| Wan T2V/TI2V family | `Wan2.1-T2V-14B-Diffusers`, `Wan2.2-TI2V-5B-Diffusers`, `Wan2.2-T2V-A14B-Diffusers` |
| Wan I2V first-only | `Wan2.1-I2V-14B-480P-Diffusers`, `Wan2.1-I2V-14B-720P-Diffusers`, `Wan2.2-I2V-A14B-Diffusers` |
| Wan first/last | `Wan2.2-I2V-A14B-Diffusers` |
| LTX2 T2AV and I2AV | `dg845/LTX-2.3-Diffusers` |
| MiniMax H3 | The same checkpoint is covered separately by T2VA, FL2VA, and Ref2VA inputs. Add an FL2VA first-plus-last fixture to complement the first-only/last-only main jobs. |

Wan2.2 TI2V-5B uses expanded timesteps, and standard CLIP-conditioned Wan2.1 I2V
checkpoints lack endpoint positional embeddings; both are first-frame only. The
dedicated Wan2.1 FLF2V checkpoint requires both endpoints. Wan2.2 I2V-A14B does
not use CLIP image embeddings, so its VAE-only optional-last path remains a
separate first/last variant gate.

For the Wan2.2 A14B dual-transformer gate, force or instrument two offline
timestep samples so that one routes below the transformer boundary and one
routes at or above it. Record both sampled timesteps and the selected component.
Two unconstrained random batches are not sufficient evidence that both intended
trainable transformers updated.

## Per-job overrides

Start from the closest checked-in example and apply a reviewable overlay. Every
generated job must make these values explicit:

```yaml
log:
  logging_backend: none
  save_freq: 0

train:
  per_device_batch_size: 1
  ema_decay: 0
  enable_gradient_checkpointing: true
  seed: 42
  max_epochs: 2  # Use 1 for the finite two-batch offline fixture.

eval:
  eval_freq: 0
```

Offline jobs additionally require a homogeneous V2 manifest, unit source
weights, on-the-fly target media encoding, and `num_train_timesteps: 1`.
Offline DPO chosen/rejected media must share one prepared condition state, the
same component times, and the same diffusion noise. Do not add a target-latent
preprocessing cache.

GRPO jobs should use the smallest reward model that still exercises reward
routing. TDM is reward-free and needs both generator and fake optimizer
entries. Disable W&B and checkpoint saving for the smoke campaign.

Bagel workers require the project `bagel` extra in addition to the selected
distributed backend, including a compatible `flash-attn` build and OpenCV. For
example, a DeepSpeed worker should install `.[deepspeed,bagel]`. Record the
installed optional reward extras in the environment manifest as well; a missing
Bagel or reward dependency is an `import` failure, not a model-support result.

Keep the two-reference Bagel cardinality identical on every rank in the FSDP2
main-matrix job. Ragged per-rank reference-round counts are a separate stress
case because a sharded language model must execute the same number of parameter
all-gathers on every rank.

## Acceptance criteria

A job passes only when all of the following are captured:

1. Configuration parsing, registry lookup, and all imports succeed from the
   tested worktree.
2. Model loading preserves the adapter's component-dtype manifest and produces
   no unexpected FP32/BF16/FP16 coercion.
3. Every rank reports identical branch routing, batch count, optimizer-step
   count, and terminal status; no rank hangs at a collective.
4. Both iterations/batches produce finite loss, finite gradient norm, and a
   confirmed parameter update for every intended trainable component.
5. SFT target encoding runs on the fly. Offline DPO reuses one prepared input
   realization for chosen and rejected outputs and keeps paired noise/time
   coupling exact.
6. Conditioned video/audio jobs retain their pinned condition regions through
   noising and forward loss masks. Unconditioned target regions remain active.
7. Joint audio-video jobs report both components, aligned durations, valid
   component-specific scheduler times, and the intended objective reduction.
8. GRPO replay likelihood and TDM replay checks satisfy the configured
   tolerances. Condition media and derived prefixes are retained in every
   conditioned mode. TDM may intentionally skip decoding generated output media
   when its algorithm contract does not consume it.
9. Peak allocated/reserved GPU memory and wall time are recorded for capacity
   planning.
10. The command, resolved YAML, stdout/stderr log, environment manifest, and a
    compact metrics JSON are attached to the job result.

## Failure classification

Classify failures before retrying:

- `contract`: schema, media cardinality/order, geometry, or capability mismatch.
- `import`: dependency, registry, optional component, or worktree-resolution error.
- `model`: official pipeline/adapter semantic mismatch.
- `algorithm`: loss, replay, reference-policy, or objective coupling error.
- `backend`: DDP/DeepSpeed/FSDP2 wrapping, collective, dtype, or checkpoint issue.
- `capacity`: reproducible OOM after the allowed geometry reductions.
- `infrastructure`: download, permission, driver, filesystem, or cluster failure.

Do not mark `capacity` or `infrastructure` failures as unsupported model/algorithm
combinations. Preserve the first failing artifact and link any follow-up fix to
the exact job ID `{mode}__{backend}__{algorithm}`.
