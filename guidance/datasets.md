# Dataset Guide

Flow-Factory datasets describe generation conditions, prompts, and optional reward metadata. The
model adapter determines which columns it consumes; a column supported by the generic loader is
not necessarily meaningful to every model or workflow.

## Directory and split layout

Each dataset is a directory with a required training split and an optional evaluation split:

```text
dataset/example/
├── train.txt or train.jsonl
├── test.txt or test.jsonl
├── images/                 # optional
├── videos/                 # optional
└── audios/                 # optional
```

When both formats are absent, loading fails. JSONL is required when a row contains media paths,
metadata, negative prompts, or heterogeneous references. Plain text files contain one prompt per
line.

Configure a dataset under `data.datasets`:

```yaml
data:
  datasets:
    - name: example
      dataset_dir: "dataset/example"
      train: {weight: 1, max_dataset_size: 1024}
      eval: {}
  preprocessing_batch_size: 8
  cache_dir: "~/.cache/flow_factory/datasets"
```

For ordinary media columns, `image_dir`, `video_dir`, and `audio_dir` default to
`{dataset_dir}/images`, `{dataset_dir}/videos`, and `{dataset_dir}/audios`. Override them when JSONL
paths use a different root:

```yaml
data:
  datasets:
    - name: example
      dataset_dir: "dataset/example"
      image_dir: "dataset/example"
      video_dir: "/shared/videos"
      audio_dir: "/shared/audio"
```

Relative paths in the ordinary `image`/`images`, `video`/`videos`, and `audio`/`audios` columns are
resolved against their corresponding media directory. MiniMax H3 Ref2VA is different:
relative `references[*].path` and `references[*].audio_path` values are resolved against
`dataset_dir`; absolute paths are accepted unchanged.

## Offline V2 records

SFT and offline DPO require strict JSONL records with `"schema_version": 2`. In this release, V2 is
a supervised offline format: every record must carry either demonstration or preference
supervision. It is not an alternative input format for the existing online generation loader.
V2 separates the model input from its supervision, so dataset rows remain model- and
algorithm-neutral:

```text
record
├── input                    # prompt plus optional condition media
├── supervision             # demonstration or preference
└── metadata                # optional JSON provenance; never a model input
```

Every V2 media object uses `type` as its only public discriminator. The accepted values are
`image`, `video`, and `audio`:

```json
{"type":"image","path":"images/source.png"}
{"type":"image","path":"images/end.png","slot":"last_frame"}
{"type":"video","path":"videos/clip.mp4","fps":24.0}
{"type":"audio","path":"audios/clip.wav","sample_rate":48000}
```

`slot` is an optional, input-only semantic binding. It is useful when position alone is ambiguous:
an explicitly slotted item reserves that adapter-declared slot, while unslotted items fill the
remaining slots in declaration order. Duplicate, unknown, and wrong-media-type slots fail during
contract validation. Supervision outputs reject `slot` because their order is declared by the
pipeline output contract rather than by condition-argument names.

The same `type` discriminator is preserved when V2 media are projected into ordered-reference
adapter inputs, so public records and adapter preprocessing use one media-entry contract.

### Demonstration supervision

One SFT row has a shared input and one target candidate:

```jsonl
{"schema_version":2,"input":{"prompt":"Restyle the source as a watercolor.","media":[{"type":"image","path":"conditions/source.png"}]},"supervision":{"type":"demonstration","target":{"media":[{"type":"image","path":"targets/watercolor.png"}]}},"metadata":{"license":"example"}}
```

Use it with `train.trainer_type: sft`. `target.media` is an ordered sequence because a pipeline
may eventually emit several modalities. Adapter-level codec availability and explicit blockers
are checked before model weights are loaded. The offline loader then validates each record's exact
output sequence, rates, input cardinality, and batch capability before condition preprocessing or
training; adapter-specific encoded geometry is validated at the output-codec boundary.

### Preference supervision

One offline-DPO row shares the input across a chosen and rejected candidate:

```jsonl
{"schema_version":2,"input":{"prompt":"A clean typographic poster.","media":[]},"supervision":{"type":"preference","chosen":{"media":[{"type":"image","path":"pairs/chosen.png"}]},"rejected":{"media":[{"type":"image","path":"pairs/rejected.png"}]}},"metadata":{"annotator":"example"}}
```

Use it with `train.trainer_type: offline-dpo`. A training source must be homogeneous: every row
must carry the supervision type required by its trainer. Prompt-only rows, mixed demonstration and
preference rows, unknown keys, and non-V2 records fail during manifest loading.

All V2 media paths are resolved against that source's `dataset_dir`; an absolute path is retained.
Images, videos, and audio have built-in CPU decoders. Video targets require PyAV 17 or newer.
Decoded audio is a detached CPU `float32` waveform shaped `(channels, samples)`. A manifest
`sample_rate` is a logical source-clock override and does not pre-resample the decoded samples;
source-clock truncation, channel conversion, the single model-rate conversion, posterior selection,
and latent packing remain adapter-owned.

Tiny schema-complete fixtures live under the repository `dataset/` root:

- [SFT demonstration manifest](../dataset/sft_sd3_5/train.jsonl) with its
  [SD3.5 config](../examples/sft/lora/sd3_5/default.yaml).
- [Offline-DPO preference manifest](../dataset/offline_dpo_sd3_5/train.jsonl) with its
  [SD3.5 config](../examples/offline_dpo/lora/sd3_5/default.yaml).

### Public offline smoke datasets

The repository includes one builder for two independent, self-contained public mini datasets:

- [Jayce-Ping/Flow-Factory-SFT-Smoke](https://huggingface.co/datasets/Jayce-Ping/Flow-Factory-SFT-Smoke)
- [Jayce-Ping/Flow-Factory-Offline-DPO-Smoke](https://huggingface.co/datasets/Jayce-Ping/Flow-Factory-Offline-DPO-Smoke)

Their runtime aliases cover the currently implemented image, video, and ordered `(video, audio)`
output adapters. The catalog is expressed with the general `PipelineIOContract` output sequence;
it does not encode audio-video as a special media type. A future audio-only or other ordered output
can therefore use the same V2 schema and preparation path, although no fixture is published for a
model family that does not exist in the framework today.

Before a GPU smoke run, materialize exactly two rank-local batches from the pinned Hub revision:

```bash
python -m dataset.offline_smoke.prepare \
  --algorithm sft \
  --profile ltx2-t2av \
  --world-size 8

python -m dataset.offline_smoke.prepare \
  --algorithm offline-dpo \
  --profile h3-ref2va \
  --world-size 8
```

The default output is `dataset/_prepared_offline_smoke/<algorithm>/<profile>/train.jsonl` with
profile-local media. The preparation tool uses the locked immutable dataset revision, selects
`world_size * per_device_batch_size * batches_per_rank` rows, and validates through the existing
V2 and canonical task-profile contract boundaries. It does not preprocess target media or create
latent caches.
See [`dataset/offline_smoke`](../dataset/offline_smoke/README.md) for construction, provenance, and
publication details.

Evaluation still uses generation acquisition, including when the trainer is SFT or offline DPO.
Consequently, a split enabled through `data.datasets[*].eval` must use one of the legacy
prompt/condition formats documented under [Common task formats](#common-task-formats), not a
supervised V2 record. A single dataset directory may therefore contain a V2 `train.jsonl` and a
legacy prompt-only `test.jsonl` without conflating their roles.

### Condition cache and target lifecycle

Offline preprocessing intentionally caches only the input side:

```text
V2 input
  -> project prompt and condition media
  -> adapter.preprocess_func under no-grad
  -> Arrow cache of prompt/condition tensors

V2 target, chosen, or rejected
  -> decode from the source file in Dataset.__getitem__
  -> collate decoded CPU media
  -> adapter.prepare_condition_state once under no-grad on every training microbatch
  -> adapter.encode_output_state with that prepared state under no-grad
  -> clean latent state for the objective
```

The prepared condition keeps immutable cached condition fields and exposes explicit model-forward
and output-codec views. Those two views may intentionally reference the same prepared input tensor;
the framework prevents accidental key overwrites rather than requiring artificial tensor copies.
SFT uses the same object for target binding and the forward.
Offline DPO uses one object for both chosen/rejected encodes and both policy/reference forwards, so
stochastic condition realization cannot drift between preference arms.

Target, chosen, and rejected payloads, their VAE latents, and supervision metadata are never
written to the Arrow condition cache. There is no target-VAE preprocessing cache. This avoids a
second large media-derived dataset on disk and keeps output geometry and posterior semantics owned
by the adapter. The frozen VAE (or other output codec component) remains available at training
time and performs the comparatively small on-the-fly encode; evaluation already needs the decoder.

During offline dataset construction, every unique normalized input and supervision media path is
streamed once through SHA-256, with digests memoized only for that source build. These digests are
identity metadata: media payloads, decoded pixels, and output latents are neither copied nor cached.
Replacing an input condition file in place therefore changes its condition identity and invalidates
the Arrow cache automatically. The checkpoint-realized input projection contract is also part of
the cache key, so changes to slot order, binding, aggregate rules, negative-prompt policy, or batch
capability cannot reuse an incompatible Arrow schema. Replacing target, chosen, or rejected media
in place changes the
full record identity used by exact-resume checks; supervision is still decoded afresh and requires
no target-cache invalidation step.

### Offline dataloader and epoch semantics

Offline sources are concatenated once and sharded with PyTorch's official
`torch.utils.data.DistributedSampler`, including a one-process run. The trainer calls
`sampler.set_epoch(data_epoch)` and advances `data_epoch` only after the finite dataloader is
exhausted successfully. Therefore one offline epoch has the standard meaning: one complete
dataloader traversal. An exception or interruption during a partial traversal does not publish a
completed epoch.

Keep these configuration rules:

```yaml
data:
  datasets:
    - name: demonstrations
      dataset_dir: "dataset/demonstrations"
      train: {weight: 1}
  enable_preprocess: true
  sampler_type: auto
train:
  max_epochs: 4
  per_device_batch_size: 1
  gradient_accumulation_steps: 4
```

- Each offline source must use `train.weight: 1`; replacement weighting would make full traversal
  stop meaning one data epoch.
- `gradient_accumulation_steps` must be an explicit positive integer. The number of rank-local
  batches must divide evenly by it; the framework never adds batches merely to close a partial
  accumulation window or silently flushes one.
- PyTorch's official `DistributedSampler` owns cross-rank tail handling. With its default
  `drop_last=False`, it may repeat tail indices when the global dataset size is not divisible by
  world size so every rank traverses the same number of samples. This is standard sampler behavior;
  an offline epoch is the resulting complete dataloader traversal, not a global uniqueness claim.
- The offline loader is already rank-sharded and is not passed to `Accelerator.prepare()`.
- `num_train_timesteps` controls independently sampled Monte Carlo loss terms averaged inside a
  microbatch. It does not multiply gradient accumulation or change epoch length.

### Decoded supervision media contract

The built-in decoders expose one model-neutral CPU representation per modality. Custom decoders
may return adapter-specific payloads, but every built-in output codec validates its own public
boundary before model preprocessing.

| Modality | Canonical decoded payload | Rate metadata |
|---|---|---|
| Image | Detached RGB `PIL.Image.Image` with positive size | Not applicable |
| Video | C-contiguous `np.uint8` RGB, `(F,H,W,3)`, positive `F/H/W`, byte domain `[0,255]` | Positive finite source `fps` when required by the output contract |
| Audio | Detached contiguous CPU `torch.float32`, `(C,S)`, positive `C/S`, finite values | Positive source `sample_rate` when required by the output contract |

Video target pixels cross these named stages:

| Stage | Shape/layout | Numeric convention |
|---|---|---|
| `decoded_frames` | NumPy `FHWC`, RGB | `uint8` bytes in `[0,255]` |
| `unit_frames` | NumPy `FHWC`, RGB | `float32`, exactly one `x / 255` conversion into `[0,1]` |
| `pixel_values` | Torch `BCFHW`, finite floating point | Model-specific normalization before the VAE |
| clean latents | Adapter-specific state/layout | Adapter-specific latent normalization and packing |

The common contract ends at unit pixels. Wan and LTX2 pass unit-range NumPy frames through
Diffusers `VideoProcessor`, which applies `2*x-1` for VAE input in `[-1,1]`. MiniMax H3 instead
applies its released `(x-mean)/std` pixel convention. Output codecs must not infer whether a float
payload means `[0,1]` or `[0,255]`, silently accept both, or normalize the same payload twice.
Temporal sampling, spatial resizing, posterior `sample`/`argmax`, and latent packing remain
adapter-owned.

Video paths are relative to the dataset directory, and manifest `fps` is the actual source rate,
not the configured training rate. Adapters deterministically project source timestamps onto their
model clock and reject insufficient duration; they do not stretch a whole clip to the requested
frame count. For example, a generic demonstration target is:

```jsonl
{"schema_version":2,"input":{"prompt":"A person jumps over a small puddle.","media":[]},"supervision":{"type":"demonstration","target":{"media":[{"type":"video","path":"videos/jump.mp4","fps":24.0}]}},"metadata":{}}
```

Wan and LTX2 checkpoints trained through the previous unscaled-NumPy target path are not repaired
by this code change, resume, or LoRA merging. Start corrected validation from the original base
model. MiniMax H3 target encoding already crossed the unit-pixel boundary and is numerically
unchanged by the shared helper refactor.

### Offline model support

The V2 schema is broader than the codecs currently implemented by adapters. Static capability
validation fails before heavyweight model loading when a selected pipeline cannot preserve its
output semantics.

| Offline status | Model types | Notes |
|---|---|---|
| Supported | `sd3-5`, `flux1`, `flux1-kontext`, `flux2`, `flux2-klein`, `qwen-image`, `qwen-image-edit-plus`, `qwen-image-2.1`, `z-image`, `bagel`, `sensenova` | Image-output codecs with adapter-specific geometry and packing. Qwen-Image 2.1 keeps ordered condition blocks before the target block and runs offline batches one sample at a time. SenseNova uses the existing grouped `images` input with within-type order, not heterogeneous references. |
| Supported | `wan2_t2v` | Video targets require `fps`; the codec resamples to configured frames/rate and samples the Wan VAE posterior on the fly. |
| Supported | `wan2_i2v` | Input media uses a checkpoint-specific contract. Expanded-timestep TI2V and standard Wan2.1 I2V accept exactly `first_frame`; dedicated Wan2.1 FLF2V requires both `first_frame` and `last_frame`; Wan2.2 I2V-A14B accepts `first_frame` plus an optional VAE-only `last_frame`. Condition pixels are cached at configured geometry and encoded with VAE posterior mode once per batch. Video targets require `fps`; offline execution is B=1. |
| Supported | `ltx2_t2av`, `ltx2_i2av` | Every candidate is an exact ordered `(video, audio)` pair with required `fps` and `sample_rate`. Both streams are aligned to the official LTX2 clock and encoded/packed on the fly. I2AV requires the `first_frame` image slot, substitutes its posterior-mode first latent into each target, and excludes the pinned tokens with an active mask. |
| Supported | `minimax-h3-t2va`, `minimax-h3-fl2va`, `minimax-h3-ref2va` | Every candidate is an exact ordered `(video, audio)` pair. FL2VA accepts `first_frame`, `last_frame`, or both slots; Ref2VA accepts 1-12 globally ordered references and requires at least one image or video. Conditioned workflows realize one official prefix per batch, shared by both offline-DPO candidates and policy/reference forwards. H3 remains B=1. |

Wan endpoint semantics use generic semantic slots, not model-specific schema keys. The realized
slot cardinality is checkpoint-specific: expanded-timestep TI2V and standard Wan2.1 I2V accept
exactly `first_frame`; dedicated Wan2.1 FLF2V requires both endpoints; Wan2.2 I2V-A14B accepts
`first_frame` plus an optional VAE-only `last_frame`. Unslotted input remains a positional
convenience, but explicit slots are recommended for sparse or generated manifests. The target is
the complete generated video, whose conditioned endpoint or endpoints must correspond to the
supplied images.

```jsonl
{"schema_version":2,"input":{"prompt":"A paper boat crosses the pond.","media":[{"type":"image","path":"conditions/first.png","slot":"first_frame"}]},"supervision":{"type":"demonstration","target":{"media":[{"type":"video","path":"targets/first-only.mp4","fps":24.0}]}},"metadata":{}}
{"schema_version":2,"input":{"prompt":"Interpolate the changing sky.","media":[{"type":"image","path":"conditions/first.png","slot":"first_frame"},{"type":"image","path":"conditions/last.png","slot":"last_frame"}]},"supervision":{"type":"preference","chosen":{"media":[{"type":"video","path":"pairs/chosen.mp4","fps":24.0}]},"rejected":{"media":[{"type":"video","path":"pairs/rejected.mp4","fps":24.0}]}},"metadata":{}}
```

LTX2 and MiniMax H3 supervision list video first and audio second. LTX2 aligns audio duration with
the official `num_frames / frame_rate` clock. H3 target video must cover the configured 24-fps
duration; it is deterministically sampled onto that frame grid and resized to the configured
canvas. H3 audio is converted to stereo at the audio-VAE rate, then trimmed or right-padded to the
exact aligned latent duration. For example:

```jsonl
{"schema_version":2,"input":{"prompt":"Ocean waves beneath an aurora.","media":[]},"supervision":{"type":"demonstration","target":{"media":[{"type":"video","path":"targets/aurora.mp4","fps":24.0},{"type":"audio","path":"targets/aurora.wav","sample_rate":32000}]}},"metadata":{}}
{"schema_version":2,"input":{"prompt":"Ocean waves beneath an aurora.","media":[]},"supervision":{"type":"preference","chosen":{"media":[{"type":"video","path":"pairs/chosen.mp4","fps":24.0},{"type":"audio","path":"pairs/chosen.wav","sample_rate":32000}]},"rejected":{"media":[{"type":"video","path":"pairs/rejected.mp4","fps":24.0},{"type":"audio","path":"pairs/rejected.wav","sample_rate":32000}]}},"metadata":{}}
```

For LTX2 I2AV, bind one image to `first_frame`. H3 FL2VA supports first-only, last-only, and
first-plus-last records; use explicit slots for the last-only form. H3 Ref2VA puts the complete
ordered image/video/audio reference sequence in `input.media`; the offline projection preserves
those ordered `type` entries at the adapter preprocessing boundary.

```jsonl
{"schema_version":2,"input":{"prompt":"Reveal the scene before this ending.","media":[{"type":"image","path":"conditions/end.png","slot":"last_frame"}]},"supervision":{"type":"demonstration","target":{"media":[{"type":"video","path":"targets/story.mp4","fps":24.0},{"type":"audio","path":"targets/story.wav","sample_rate":32000}]}},"metadata":{}}
```

## Common task formats

The following compact formats remain supported for generation acquisition. They are separate from
strict V2 offline records and do not carry output supervision.

### Text-conditioned generation

Text-to-image, text-to-video, and text-to-audio-video datasets may use plain text:

```text
A hill at sunset.
An astronaut riding a horse on Mars.
```

The equivalent JSONL form is:

```jsonl
{"prompt":"A hill at sunset."}
{"prompt":"An astronaut riding a horse on Mars."}
```

Adapters that implement classifier-free guidance may also consume `"negative_prompt"`. Do not add
it to workflows that explicitly reject CFG, including MiniMax H3.

Examples:

- [T2I prompts](../dataset/t2is/train.jsonl)
- [T2I prompts with negative prompts](../dataset/t2is_neg/train.jsonl)
- [MiniMax H3 T2VA prompts](../dataset/minimax_h3_t2va/train.jsonl)

### Image-conditioned generation

Use `"image"` for one condition image or `"images"` for an ordered list:

```jsonl
{"prompt":"Restyle this scene as a watercolor.","image":"input.png"}
{"prompt":"Combine these visual references.","images":["first.png","second.png"]}
```

The loader normalizes the singular form to the batched image path and decodes images as RGB. The
adapter decides whether multiple images are valid and what their order means.

### Video-conditioned generation

Use `"video"` for one condition video or `"videos"` for a list:

```jsonl
{"prompt":"Restyle this motion.","video":"input.mp4"}
{"prompt":"Use both motion references.","videos":["motion-a.mp4","motion-b.mp4"]}
```

Videos are decoded before adapter preprocessing. As with images, list cardinality and semantics
belong to the selected adapter.

### Audio columns

The generic loader accepts `"audio"` or `"audios"` and resolves them against `audio_dir`:

```jsonl
{"prompt":"Generate visuals matching this sound.","audio":"sound.wav"}
{"prompt":"Use these sound references.","audios":["voice.wav","ambience.wav"]}
```

Only use these columns with an adapter that declares an audio input. MiniMax H3 Ref2VA instead
places audio entries inside the ordered `"references"` array.

## SenseNova-U1 datasets

SenseNova-U1 1.0/1.5 supports prompt-only T2I and ordered multi-reference I2I.
T2I can use the ordinary text or JSONL formats above; the
[default GRPO recipe](../examples/grpo/lora/sensenova/default.yaml) uses OCR prompts.

For I2I, each row uses the ordinary `"images"` list. List order is preserved and
each sample may contain a different number and size of reference images:

```jsonl
{"prompt":"Combine these images together.","images":["first.png","second.png"]}
{"prompt":"Combine these images together.","images":["style.png","subject.png","layout.png"]}
```

Prepare the 2–3-reference example dataset with:

```bash
python dataset/multi_ref_image/prepare.py
```

Then launch the
[multi-reference GRPO recipe](../examples/grpo/lora/sensenova/multi_reference_image.yaml).
The dataset cache stores ragged reference lists as PIL images. SenseNova builds a
separate variable-length NEO-Unify prefix for each generated sample; it does not
NaViT-pack independent samples like Bagel.

## MiniMax H3 datasets

All three MiniMax H3 workflows require preprocessing and training batch size `B=1`. They do not use
CFG, so both training and evaluation must keep:

```yaml
data:
  preprocessing_batch_size: 1
train:
  per_device_batch_size: 1
  guidance_scale: 1.0
eval:
  per_device_batch_size: 1
  guidance_scale: 1.0
```

MiniMax H3 accepts 5–15 seconds at 24 fps. `num_frames` is rounded up to
`17*n+5`, so 124 is the smallest explicit frame count that satisfies both the
duration and VAE chunking contracts.

T2VA `debug.yaml` is real-weight validated on 1 and 16 GPUs, including LoRA
checkpoint save/resume. Its 64x96 canvas validates
correctness and memory fit, not visual quality or reward improvement.
`quality_720p_fsdp2.yaml` has real-weight initialization, checkpoint, native-resolution
decode, and evaluation coverage; no long-run reward trend is claimed. The aligned default uses
the shared `dataset/vid_prompt` source, LoRA rank 64, and CLAP plus ImageBind rewards. The PR #220
smoke campaign completed all 36 H3 main cells: T2VA, FL2VA, and Ref2VA across
DDP/DeepSpeed ZeRO-2/FSDP2 and GRPO/SFT/offline DPO/TDM. The FL2VA first-plus-last
SFT/offline-DPO variant gate also passed. This is execution coverage, not a published long-run
reward trend, convergence, or numerical-parity claim.

### T2VA: `minimax-h3-t2va`

T2VA is prompt-only:

```jsonl
{"prompt":"A small paper windmill turns beside a quiet stream with synchronized birdsong."}
```

Do not include negative prompts, images, or references. The
[T2VA default GRPO configuration](../examples/grpo/lora/minimax_h3_t2va/default.yaml) uses the
shared [`vid_prompt` TXT dataset](../dataset/vid_prompt/train.txt). The dedicated
[T2VA JSONL fixture](../dataset/minimax_h3_t2va/train.jsonl) remains the compact input for the
real-weight validated `debug.yaml` recipe.

### FL2VA: `minimax-h3-fl2va`

FL2VA uses an ordered `"images"` list:

```jsonl
{"prompt":"Animate this scene with matching sound.","images":["images/first.png"]}
{"prompt":"Interpolate between these scenes.","images":["images/first.png","images/last.png"]}
```

- One image is the first frame.
- Two images are the first frame followed by the last frame.
- Any other cardinality is invalid.
- Order must not be sorted, deduplicated, or inferred from filenames.

This legacy generation form cannot express last-only conditioning. Strict V2 offline records use
the generic `slot: "last_frame"` binding shown above, so FL2VA can train on a final frame without a
synthetic first frame.

The example stores paths relative to the dataset root, so its YAML sets `image_dir` to the dataset
directory. See the [FL2VA dataset fixture](../dataset/minimax_h3_fl2va/train.jsonl) and
[FL2VA GRPO configuration](../examples/grpo/lora/minimax_h3_fl2va/default.yaml).

### Ref2VA: `minimax-h3-ref2va`

The online Ref2VA loader uses a non-empty ordered `"references"` array containing image, video,
and audio entries. Each entry uses the same `type` discriminator as strict V2 media objects:

```jsonl
{"prompt":"Create a coherent scene using the references in order.","references":[{"type":"image","path":"references/style.png"},{"type":"video","path":"references/motion.mp4","fps":12.0},{"type":"audio","path":"references/ambience.wav","sample_rate":16000}]}
```

Array order is semantically significant. It is preserved during validation, encoding, caching, and
sample identity hashing. The array accepts 1-12 entries and requires at least one image or video;
an audio-only array is invalid.

Supported entries:

| `type` | Required keys | Optional keys | Decoded value |
|---|---|---|---|
| `image` | `type`, `path` | none | RGB image |
| `video` | `type`, `path` | `fps`, `audio_path`, `sample_rate` | frames and optional soundtrack |
| `audio` | `type`, `path` | `sample_rate` | waveform |

`fps` and `sample_rate` overrides must be finite positive numbers. A video may use its embedded
soundtrack or a separate dataset-relative `audio_path`; a video `sample_rate` override requires
`audio_path`. Unknown keys and unsupported `type` values fail before preprocessing.

The compact online manifest and strict V2 format share the same media-entry discriminator. Offline
condition projection preserves `input.media[*].type` when it constructs the ordered-reference
manifest consumed by the adapter.

See the [Ref2VA dataset fixture](../dataset/minimax_h3_ref2va/train.jsonl), its
[local fixture notes](../dataset/minimax_h3_ref2va/README.md), and the
[Ref2VA GRPO configuration](../examples/grpo/lora/minimax_h3_ref2va/default.yaml).

## Preprocessing and cache flow

Normal datasets follow this path:

```text
TXT/JSONL row
  -> resolve paths and decode media
  -> adapter.preprocess_func under no-grad
  -> move encoded tensors to CPU
  -> write Arrow cache
  -> collate cached fields for rollout
```

Prompt encoders, condition VAEs, and processors are preprocessing components. Online RL can
offload them after cache creation because optimization consumes cached conditions. Offline SFT and
offline DPO reload any condition-preparer and output-codec components declared by the adapter.
They realize one prepared condition per batch, then encode target, chosen, and rejected media on
the fly; those output states are never cached.

Ref2VA adds an ordered-reference path:

```text
references array
  -> validate and canonicalize as reference_manifest
  -> decode images/video/audio with PIL and PyAV
  -> construct transient pinned Diffusers reference objects
  -> run Ref2VA setup, text, reference-encoder, and layout blocks under no-grad
  -> cache Arrow-safe prompt embeddings, condition latents, layout, geometry, and manifest
```

Reference media is not stored in the Arrow cache. The decoded PIL/PyAV values and upstream
reference objects exist only during preprocessing. Cache identity includes the configured dataset
root, TXT/JSONL source bytes, semantic preprocessing fields, adapter preprocessing version, and
model identity.
`reference_manifest` is the canonical JSON string retained for reproducibility and sample
identity. Replacing a media file in place without changing its path or manifest does not change
the source hash; set `force_reprocess: true` after such a replacement.

## How Ref2VA references affect training

At rollout time, the cached reference encodings become immutable video/audio
`condition_prefixes`. Fresh random video and audio target states are created separately. The
packed layout presented to `transformer_ref` is conceptually:

```text
[reference condition rows | generated target rows]
```

Joint attention lets the references change the predicted target video/audio velocity and therefore
the rollout distribution, generated media, reward, and replay log-probability. After each forward
pass, condition-row predictions are discarded and only target rows are stepped by the video and
audio schedulers.

Reference condition rows are not trajectory states. Reference condition rows do not contribute
policy log-probability degrees of freedom. They are fixed observations, not sampled actions.
Gradients flow through `transformer_ref` (or its LoRA parameters) because its target prediction
depends on those observations, but gradients do not update the source media, text encoder,
reference encoder, or VAE.

The resulting sample retains:

- the target-only video/audio structured trajectory;
- cached prompt embeddings;
- `condition_prefixes`, packed layout, and geometry for identical replay conditions;
- `reference_manifest` for identity and auditability.

This separation keeps rollout and optimization mathematically aligned without treating reference
tokens as generated state.
