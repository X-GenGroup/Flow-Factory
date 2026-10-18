# GenEval prompts rewritten by the Qwen-Image-2.1 T2I rewriter

`ff-prompt-enhance` output for `dataset/geneval`, used by
`examples/grpo/lora/qwen_image_2_1/geneval.yaml`. It covers the full test split (553 rows) and the
first 1024 train rows, which is the slice that config trains on (`max_dataset_size: 1024` keeps the
first rows).

- Config: `examples/prompt_enhance/qwen_image_2_1_geneval.yaml` (`qwen-image-2.1-t2i` enhancer, one
  variant, seed 42, and the official T2I profile: thinking, then sampling with `temperature: 1.0`,
  `top_p: 0.95`, `top_k: 20`, `presence_penalty: 1.5`, and `max_new_tokens: 16256`).
- Model: `Qwen/Qwen-Image-2.1-PE-T2I` at revision `f3ed7985c788ad75b3ab7223e0c4c51e2a43545b`.
- `prompt` holds the rewritten prompt, `source_prompt` the GenEval prompt, and `pe_variant` is 0. The
  GenEval `include`, `exclude`, and `tag` fields are copied unchanged, so the reward scores the same
  object, count, color, and position constraints.
- The rewriter's recommended `wh_ratio` is not recorded; training renders at the configured
  resolution.
- Every row was answered on its first attempt, so no row was resampled. Rewrites run 207-1039
  words (median about 440).
