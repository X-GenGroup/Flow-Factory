# Examples

Curated training configs for representative supported algorithm–model combinations.

## Directory Structure

```
examples/{algorithm}/{finetune_type}/{model_type}/{variant}.yaml
```

| Level | Description | Examples |
|-------|-------------|---------|
| `algorithm` | Training algorithm | `sft`, `offline_dpo`, `grpo`, `dppo`, `nft`, `awm`, `dgpo`, `dpo`, `crd`, `opd`, `dmd2`, `tdm`, `tdm_r1` |
| `finetune_type` | Parameter-efficient or full | `lora`, `full` |
| `model_type` | Model family (underscore-separated) | `flux1`, `sd3_5`, `wan2_t2v`, `ltx2_t2av` |
| `variant` | Config variant | `default.yaml`, `nocfg.yaml`, `t2v.yaml` |

**Naming rules**:
- Model directory names use underscores matching the config's `model_type` field (e.g., `sd3-5` → `sd3_5`, `flux1-kontext` → `flux1_kontext`).
- `default.yaml` is the baseline config for a model. Use descriptive names for variants (`nocfg.yaml`, `rational_rewards_t2i.yaml`, `t2v.yaml`, `i2v.yaml`).

**Quick start**:
```bash
ff-train examples/grpo/lora/flux1/default.yaml
```

## Offline examples

- [`sft` with SD3.5](sft/lora/sd3_5/default.yaml) consumes V2
  `demonstration` records from [`dataset/sft_sd3_5`](../dataset/sft_sd3_5/train.jsonl).
- [`offline-dpo` with SD3.5](offline_dpo/lora/sd3_5/default.yaml) consumes V2
  `preference` records from
  [`dataset/offline_dpo_sd3_5`](../dataset/offline_dpo_sd3_5/train.jsonl).
- [`sft` with Wan2.2 TI2V-5B](sft/lora/wan2_t2v/wan22_ti2v5b.yaml) consumes V2
  `demonstration` records with one video target and its actual source `fps`. The 49-frame, 10-FPS
  recipe requires roughly 4.8 seconds of source coverage, is configured for eight-process BF16
  ZeRO-2, and does not bundle a dataset.

The two tiny manifests reuse repository images so their paths resolve without a separate dataset
download. They are configuration and smoke-test fixtures, not quality-training datasets. Offline
training requires an explicit integer `gradient_accumulation_steps`; the number of rank-local
dataloader batches must be divisible by it. See the [dataset guide](../guidance/datasets.md#offline-v2-records)
for the production schema and media requirements.

## DMD2 and TDM

- [`dmd2` SD3.5 OCR recipe](dmd2/lora/sd3_5/ocr.yaml) — validated in a
  distributed OCR training run.
- [`tdm` SD3.5 OCR recipe](tdm/lora/sd3_5/ocr.yaml) — official conditional-noise objective.
- [`tdm` MiniMax H3 T2VA 6-step recipe](tdm/lora/minimax_h3_t2va/default.yaml) —
  validated 512×768×124 run; train over 200 steps to approach convergence. It includes a disabled,
  optional FirstBlockCache snippet whose threshold still requires target-GPU reward/quality
  calibration; do not combine it with `torch_compile`.
- [`tdm-r1` SD3.5 OCR recipe](tdm_r1/lora/sd3_5/ocr.yaml) — official G24
  fake-surrogate-generator objective initialized from the released TDM adapter.

## MiniMax H3 examples

- [`minimax-h3-t2va`](grpo/lora/minimax_h3_t2va/default.yaml)
- [`minimax-h3-t2va` real-weight debug recipe](grpo/lora/minimax_h3_t2va/debug.yaml)
- [`minimax-h3-t2va` native-quality FSDP2 recipe](grpo/lora/minimax_h3_t2va/quality_720p_fsdp2.yaml)
- [`minimax-h3-t2va` TDM 6-step recipe](tdm/lora/minimax_h3_t2va/default.yaml)
- [`minimax-h3-fl2va`](grpo/lora/minimax_h3_fl2va/default.yaml)
- [`minimax-h3-ref2va`](grpo/lora/minimax_h3_ref2va/default.yaml)

## SenseNova-U1 examples

- [`sensenova` U1.5 T2I + OCR GRPO](grpo/lora/sensenova/default.yaml)
- [`sensenova` U1.5 ordered multi-reference I2I + PickScore GRPO](grpo/lora/sensenova/multi_reference_image.yaml)

Both recipes default to `sensenova/SenseNova-U1.5-8B-MoT-SFT`; set `model.model_name_or_path` to
`sensenova/SenseNova-U1-8B-MoT` for U1.0. For I2I, provide the dataset JSONL
column `images` as an ordered list per sample. Preprocessing maps it to the
adapter's `condition_images`; variable-size/count references remain PIL through
the HF Image feature.
Prepare the example's 2–3-image dataset with
`python dataset/multi_ref_image/prepare.py` before launching the multi-reference recipe.

<details>
<summary>U1.5 T2I OCR GRPO validation curves (4 nodes × 8 GPUs)</summary>

These curves validate the default T2I OCR recipe only; they are not evidence for
the multi-reference I2I recipe.

![SenseNova OCR train reward](../docs/assets/sensenova-u15-ocr-train-reward-ocr-mean.png)
![SenseNova OCR eval reward](../docs/assets/sensenova-u15-ocr-eval-reward-ocr-mean.png)
![SenseNova OCR train ratio](../docs/assets/sensenova-u15-ocr-train-ratio-mean.png)

</details>

## Prompt enhancement examples

These configs run `ff-prompt-enhance`, not `ff-train`. Each writes a new dataset whose
`prompt` column is the enhanced text that training then uses verbatim:

- [`flux3`](prompt_enhance/flux3.yaml) — FLUX.3 prompt upsampler (`flux-3-image-klein-pup`), T2I and image-conditioned prompts
- [`flux3_geneval`](prompt_enhance/flux3_geneval.yaml) — the same upsampler over GenEval; its output, `dataset/geneval_flux3_pe`, trains [`grpo/lora/flux3/geneval.yaml`](grpo/lora/flux3/geneval.yaml)
- [`flux2`](prompt_enhance/flux2.yaml) — FLUX.2-dev caption upsampling
- [`ltx2`](prompt_enhance/ltx2.yaml) — LTX-2 Gemma prompt enhancement
- [`qwen_image_2_1_t2i`](prompt_enhance/qwen_image_2_1_t2i.yaml) — Qwen-Image-2.1 T2I rewriter (`Qwen-Image-2.1-PE-T2I`) for prompt-only datasets
- [`qwen_image_2_1_geneval`](prompt_enhance/qwen_image_2_1_geneval.yaml) — the T2I rewriter over GenEval; its output, `dataset/geneval_qwen_image_21_pe`, trains [`grpo/lora/qwen_image_2_1/geneval.yaml`](grpo/lora/qwen_image_2_1/geneval.yaml)
- [`qwen_image_2_1_i2i`](prompt_enhance/qwen_image_2_1_i2i.yaml) — Qwen-Image-2.1 I2I rewriter (`Qwen-Image-2.1-PE-I2I`) for image-editing datasets
- [`openai_compatible`](prompt_enhance/openai_compatible.yaml) — any served chat model

See the [prompt enhancement guide](../guidance/prompt_enhance.md).

## Qwen-Image 2.1 example

- [`qwen-image-2.1` ordered-image editing GRPO](grpo/lora/qwen_image_2_1/default.yaml)

The same adapter also accepts prompt-only records. It currently requires the bundled Diffusers
submodule and enforces `per_device_batch_size: 1`. The recipe enables the model's exact prefix KV
cache (`train.use_kv_cache: true`); replay always follows the graph recorded by rollout, as described
in the [Acceleration guide](../guidance/acceleration.md).

## FLUX.3 example

- [`flux3` text-to-image & multi-reference editing GRPO](grpo/lora/flux3/default.yaml)

Supports both `black-forest-labs/flux-3-image-klein-base` (true CFG) and
`black-forest-labs/flux-3-image-klein` (step-distilled flow map, no CFG) via the bundled
Diffusers submodule.

## MiniMax H3 recipes and validation

The T2VA `debug.yaml` recipe uses a 64x96 correctness geometry. `quality_720p_fsdp2.yaml` is the
native-quality path: 768x1344, 124 frames, 24 denoising steps, LoRA rank 64 / alpha 128, and two
updates from 48 prompt groups per epoch. The T2VA default scores with the `clap` and `imagebind`
rewards, and FL2VA and Ref2VA with `imagebind`; install ImageBind first (see
[ImageBind setup](../guidance/rewards.md#imagebind-setup)). Real-weight execution coverage for the H3
recipes is recorded in [MiniMax H3 validation status](../guidance/gpu_validation.md#minimax-h3-validation-status).

## Contributing

We welcome community contributions! Here's what you can contribute and how:

### Verified Training Configs

If you've tested a model–algorithm combination and confirmed reward improvement, submit a PR with:
- The config YAML following the directory structure above
- A brief note in the PR description about hardware used and observed reward trend

> **Example**: [#145 — LTX-2.3 + PickScore](https://github.com/X-GenGroup/Flow-Factory/pull/145) added a GRPO + LoRA config for text-to-audio-video, with a training curve (8×H200, 18h) confirming reward improvement.

### Custom Reward Models

New reward models are welcome — add the implementation under `src/flow_factory/rewards/` and include an example config that uses it. Please ensure your reward model's dependencies are compatible with the existing environment (check `pyproject.toml`).

### New Model Adapters

See the [New Model Guide](../guidance/new_model.md) for how to add a new diffusion/flow-matching model. Include at least one example config with your PR.

### Guidelines

- Configs should be self-contained and runnable with `ff-train`
- Include comments for non-obvious parameter choices
- If your config requires a specific dataset, document how to obtain it
- Test on at least one hardware configuration before submitting
