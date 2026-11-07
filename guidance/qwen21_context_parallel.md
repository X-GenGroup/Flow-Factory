# Qwen-Image 2.1 tensor, sequence and context parallelism

Qwen-Image 2.1 declares a separate prefix/block-causal layout. Its repeated attention
and SwiGLU projections use the existing tensor plan. Native Ulysses CP supports both
the default exact prefix cache and the public `train.use_kv_cache: false` joint path.
Sequence parallelism is optional and remains disabled by default.

The qualified native components are QwenImage21Transformer2DModel with causal
conditioning, patch size one and the segmented attention processor. Attention heads
must be divisible by TP times CP. Model validation rejects unsupported processors
and noncausal layouts before CP installation. The native released Diffusers 0.41
implementation includes these components; no global dependency change is needed.

For cached execution, prefix extraction retains its original ordered text/image
segments, positions and differentiable cache. Target decoding partitions target
tokens and cached keys consistently. Prefix key padding is masked; loss-free dummy
queries satisfy the native Ulysses backward requirement that local Q/K lengths match.
Neither padding changes a model parameter, image position or cache entry. Rollout
retains detached exact caches, and replay rebuilds them from current parameters.

The joint path preserves native text-causal and per-image block visibility. Internal
token padding permits prompt-dependent lengths without changing token positions;
the attention mask excludes padded keys. Native gather is followed by removal of
padding before the adapter selects target-image output. SP slices residual/modulation
tokens, gathers for native attention and MLP projections, and restores tokens before
the output projection. Prefix extraction remains whole. Each model/block invocation
owns its layout scope, including activation-checkpoint recomputation.

The checkpoint identity contains `qwen21_prefix_block_causal` only when CP is enabled.
Existing CP1 checkpoint identity remains unchanged. Earlier versions did not admit
Qwen-2.1 CP checkpoints. Exact continuation is qualified on one immutable source and
the same realized topology, native dependency versions and runtime configuration.

Validation uses independent native joint forwards as the oracle for both cache modes,
full and LoRA parameter gradients, three Adam steps and an omitted-CP-loss-factor
negative control. The numerical probe uses AdamW with learning rate `1e-3` and epsilon
`1e-4`, matching the conditioned Guard fixture's optimizer scale. Its forward,
gradient, update and optimizer-state tolerances remain unchanged. CPU native
collectives complement an eight-rank DP2/TP2/CP2 gate
with native 32 by 128 attention heads. The conditioned Guard lifecycle uses the real
Qwen3-VL processor/encoder, ordered RGBA reference images and the native VAE, including
evaluation, EMA, exact state load and fresh-process continuation.

See [context parallelism](context_parallel.md) and [tensor parallelism](tensor_parallel.md)
for shared preparation and logical-sample ownership.
