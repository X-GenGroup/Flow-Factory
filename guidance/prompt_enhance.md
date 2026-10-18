# Prompt Enhancement

Flow-Factory separates two jobs:

1. **Model training** (`ff-train`) feeds every dataset `prompt` to the model **verbatim**. No
   adapter rewrites prompts during preprocessing, rollout, or evaluation.
2. **Prompt enhancement** (`ff-prompt-enhance`) rewrites a dataset's prompts with a prompt
   enhancer model and writes a **new dataset**. Training on it is an ordinary training run.

Whether to enhance, with which model, and how many rewrites per prompt are decisions you make
before training. If your deployment enhances prompts, train and evaluate on enhanced datasets
produced by the same enhancer configuration.

## Quick Start

```bash
ff-prompt-enhance examples/prompt_enhance/flux3.yaml
```

Then point a training config at the result:

```yaml
data:
  datasets:
    - name: default
      dataset_dir: "dataset/sharegpt4o_image_mini_flux3_pe"
```

`ff-prompt-enhance` uses the same launcher as `ff-train`: `num_processes`, `num_machines`,
`main_process_ip`, and the other launcher keys in the YAML start one enhancer replica per
process, and each rank enhances a disjoint shard of rows.

## Output Dataset

For every selected split, `output_dir/{split}.jsonl` contains `num_variants` rows per source row,
with the variants of one source row adjacent:

| Field | Value |
|---|---|
| `prompt` | The enhanced prompt. This is the text the model trains on. |
| `source_prompt` | The prompt the row was enhanced from. |
| `pe_variant` | Variant index in `[0, num_variants)`. |
| every other field | Copied unchanged from the source row (`images`, `negative_prompt`, metadata, ...). |

Each output row is an independent training prompt; variants of one source prompt are not grouped.
`source_prompt` and `pe_variant` reach training samples through the standard `metadata` column.
The source `images`, `videos`, and `audios` directories are symlinked into `output_dir`, so
relative media paths keep resolving. `prompt_enhance_manifest.json` records the configuration,
enhancer details, and row counts.

`.txt` splits are read as one prompt per line; the output is always JSONL. A run resumes from the
per-rank results under `output_dir/.parts/` and only enhances the `(row, variant)` pairs that are
missing, so raising `num_variants` adds variants to a finished run. Resuming after changing the
enhancer, model, seed, dtype, system prompt, generation settings, or source rows fails; use a new
`output_dir` or delete `.parts/` to start over.

## Configuration

```yaml
enhancer: flux3              # registry key or package.module.Class
model_name_or_path: "black-forest-labs/flux-3-image-klein-pup"
input_dir: "dataset/sharegpt4o_image_mini"
output_dir: "dataset/sharegpt4o_image_mini_flux3_pe"
splits: null                 # null = every train/test split present
num_variants: 2              # rewrites per source prompt (default 1)
batch_size: 8                # prompts per enhancer call
seed: 42
dtype: bfloat16              # enhancer load dtype
system_prompt: null          # null = the enhancer's default
generation: {}               # enhancer-specific; unsupported keys are rejected
```

`generation` keys are checked against the enhancer's supported list in the table below, and an
unsupported key raises. Other top-level keys that are not configuration fields are collected into
`extra_kwargs` with a warning; this is how `openai-compatible` receives `base_url`, and it also
means a misspelled top-level key is not rejected. The launcher keys can be overridden on the
command line (`--num_processes`, `--num_machines`, `--machine_rank`, `--main_process_ip`,
`--main_process_port`), with the same command line > cluster environment > YAML priority as
`ff-train` (see [Multi-node training](../multinode_examples/README.md)).

Each `(row, variant)` is seeded with `seed + variant_index * num_source_rows + row_index`, which
does not depend on `batch_size` or the number of processes. `flux3`, `ltx2`, the two
`qwen-image-2.1` enhancers, and `openai-compatible` seed every prompt with it; `flux2` samples a
whole batch in one call, seeded
with the batch's first row. LLM decoding is still not bit-reproducible across hardware, so the
written dataset, not a re-run, is the record of what was generated.

## Built-in Enhancers

| `enhancer` | Model | `generation` keys | Notes |
|---|---|---|---|
| `flux3` | `black-forest-labs/flux-3-image-klein-pup` | `max_new_tokens`, `do_sample`, `temperature`, `top_p`, `top_k`, `repetition_penalty` | Uses the upsampler's `t2i` / `i2i` system prompts; image rows pass their references in order. Default decoding is greedy (`max_new_tokens: 3584`), so set `do_sample: true` for `num_variants > 1`. |
| `flux2` | `black-forest-labs/FLUX.2-dev` | `temperature` | Loads only the Mistral3 text encoder and runs `Flux2Pipeline.upsample_prompt`. The T2I/I2I system messages are fixed, so `system_prompt` must be `null`. |
| `ltx2` | `diffusers/LTX-2.3-Diffusers` and other LTX-2 Diffusers checkpoints with a Gemma3 processor | `max_new_tokens`, `do_sample`, `temperature`, `top_p`, `top_k`, `repetition_penalty`, `no_repeat_ngram_size` | Runs `LTX2Pipeline.enhance_prompt` with the upstream T2V/I2V default system prompt; LTX-2.5 uses its dedicated Gemma4 enhancer. Requires a `processor`, declared in `model_index.json` or shipped as a `processor/` subfolder; `Lightricks/LTX-2` has neither. Image rows use their first image. A non-empty `generation` replaces the upstream generation kwargs. |
| `qwen-image-2.1-t2i` | `Qwen/Qwen-Image-2.1-PE-T2I` | `max_new_tokens`, `do_sample`, `temperature`, `top_p`, `top_k`, `presence_penalty` | Prompt-only rows; a row with reference images fails. The rewriter thinks, then answers with JSON whose `rewritten_prompt` becomes the prompt; its `wh_ratio` is not recorded. The default system prompt is the checkpoint's `system_prompt.txt`. Default decoding is the official T2I profile: sampling with `temperature: 1.0`, `top_p: 0.95`, `top_k: 20`, vLLM-style `presence_penalty: 1.5`, and `max_new_tokens: 16256`. A sampled rewrite that loops until `max_new_tokens` or answers without a parseable `rewritten_prompt` is resampled from the continuing RNG stream of its seed; the run fails after three attempts (one under greedy decoding). |
| `qwen-image-2.1-i2i` | `Qwen/Qwen-Image-2.1-PE-I2I` | `max_new_tokens`, `do_sample`, `temperature`, `top_p`, `top_k`, `presence_penalty` | Image-editing rows; a row without a reference image fails. Images are downscaled to at most 1024×1024 pixels and passed in order, so multi-image prompts can refer to `<image1>`, `<image2>`, .... Answer handling, resampling, and system prompt as for `qwen-image-2.1-t2i`; `wh_ratio` and `ratio_follow` are not recorded. Default decoding is the official edit profile: the same sampling without a presence penalty and with `max_new_tokens: 24000`. |
| `openai-compatible` | Any served chat model | `temperature`, `top_p`, `max_new_tokens` | Requires `system_prompt` and `extra_kwargs.base_url`; optional `api_key_env`, `timeout`, `max_concurrency`. |

## Training on Long Enhanced Prompts

Enhanced prompts are often several hundred tokens long. Make sure the training adapter's
`max_sequence_length` covers them; otherwise the text encoder truncates the prompt the model
trains on. Long prompts also enlarge the preprocessing cache for models with wide text
embeddings such as FLUX.3. If the cache becomes too large, set `data.enable_preprocess: false`
to encode prompts during rollout instead.

## Adding an Enhancer

Subclass `flow_factory.prompt_enhance.BasePromptEnhancer` and point `enhancer` at its python path,
or register it in `prompt_enhance/registry.py`:

```python
from flow_factory.prompt_enhance import BasePromptEnhancer


class MyEnhancer(BasePromptEnhancer):
    uses_images = False                      # True to receive each row's reference images
    supported_generation_keys = frozenset({"temperature"})

    def load(self) -> None:
        ...                                  # load onto self.device

    def enhance(self, prompts, images, seeds):
        ...                                  # return one string per prompt, in order
```

The runner owns dataset I/O, sharding, seeding, and resume. `enhance` runs inside an isolated RNG
scope seeded with `seeds[0]`; an enhancer that generates prompts one at a time should seed each
prompt with its own entry of `seeds`.
