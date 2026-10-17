# GenEval prompts rewritten by the FLUX.3 prompt upsampler

`ff-prompt-enhance` output for `dataset/geneval`, used by
`examples/grpo/lora/flux3/geneval.yaml`. It covers the full test split (553 rows) and the first 1024
train rows, which is the slice that config trains on (`max_dataset_size: 1024` keeps the first rows).

- Config: `examples/prompt_enhance/flux3_geneval.yaml` (`flux3` enhancer, one variant, seed 42,
  greedy decoding with `max_new_tokens: 3584`).
- Model: `black-forest-labs/flux-3-image-klein-pup` at revision
  `3f3edb52b2c55de0d49feab28598b68f7f8e4830`.
- `prompt` holds the rewritten caption, `source_prompt` the GenEval prompt, and `pe_variant` is 0. The
  GenEval `include`, `exclude`, and `tag` fields are copied unchanged, so the reward scores the same
  object, count, color, and position constraints.
- Test row 85 (`a photo of a black tv remote`) degenerated under greedy decoding: it listed button
  labels until it hit `max_new_tokens`. That row was regenerated with `do_sample: true`,
  `temperature: 0.7`, `top_p: 0.9`, and its runner seed 127 (`seed + row`).
