# Experimental Flux2 and Klein TP/SP/CP

`flux2` and `flux2-klein` share the native `Flux2Transformer2DModel` layout. The dev
adapter retains guidance embeddings; Klein retains separate positive/negative CFG calls.
Both expose canonical packed TP and optional block-local token SP. SP remains off by default.
The model-owned CP layout is `flux2_target_reference_text`; Diffusers splits the combined
target/reference image sequence, text sequence and their batched four-axis rotary IDs at
the transformer root, then gathers `proj_out`. The adapter crops reference predictions only
after that gather.

TP shards the dual-stream attention projections and packed SwiGLU inputs, as well as the
single-stream packed Q/K/V/MLP and attention/MLP output projections. The shared packed styles
retain ordinary two-dimensional parameters and canonical saved ordering. Differentiable
activation exchanges route each logical packed region to its TP owner and reverse that
routing at row projections. This adds communication; the qualification makes no performance
claim. Native attention heads, inner width and single-stream `mlp_hidden_dim` become local
once, while saved config and native packed-block metadata remain global. LoRA A/B ownership
comes from the same packed styles, preserving canonical PEFT and optimizer state.

With `sequence_parallel: true`, both image and text residuals are sharded inside each native
dual block, and the joint residual is sharded inside each single block. Native token gathers
at attention/FF inputs and reduce-scattered projection outputs retain their autograd owners.
Every block restores its CP-local stream before the next native block or concatenation.
Shared modulation derivatives sum token contributions once. The adapter validates native CP
split/gather metadata; it neither moves those hooks nor replaces a transformer forward.

CP admission requires both block stacks, native dense processors, four positive even rotary
axes summing to head width, patch size one, equal packed input/output width divisible by four,
and heads divisible by CP times TP. The guidance flag must match the adapter; an omitted dev
flag uses the native default `true`. Runtime uses one sample per model call, paired reference
latents/IDs, and paired negative text/IDs for Klein. The combined target/reference token count
and each text sequence must be divisible by CP, or by CP times TP with SP. Individual target and
reference spans need not divide independently. Masks, added attention arguments, reference KV
cache/extraction processors and malformed layouts fail before attention collectives. The
existing framework requires native attention, BF16 training, AdamW and FSDP2 with sharded DP.

The singleton I2I forward preserves the scheduler's original tensor endpoints and public
output subclass. Previously, concatenating even one result disconnected the returned mean
and velocity from the returned log probability graph. Values and parameter gradients are
unchanged; native full/LoRA Guard controls check exact loss/gradient parity, graph ancestry
and optional fields. Multi-sample aggregation retains its existing behavior.

The eight-rank numerical reference uses AdamW `lr=1e-3`, matching the native
lifecycle fixture's learning-rate scale. It retains the probe's `eps=1e-4`,
`weight_decay=0.01`, raw-gradient objective and independent optimizer histories;
the lifecycle separately uses clipping and its configured weight decay. Every
rank records actual/reference optimizer settings. Output, gradient, update and
moment bounds, all-element checks and negative controls remain unchanged.

## Selected evidence

The source-bound cases are qualification requirements for the reviewed candidate, not a claim
that pending GPU runs have passed:

- `flux2-parallel-contracts`: native saved encoder/VAE components, guidance versus CFG, real
  64×128 references resized to 32×64, nontrivial VAE batch-normalization state, packed positions,
  native cached-inference transitions/decoded pixels, singleton Guard derivatives and early guards.
- `flux2-tp-sp-contracts`: full/LoRA TP2 and checkpointed SP output, every active parameter
  gradient, three Adam updates and moments against an independent unsharded model.
- `flux2-tp-sp-cp-reference`: six full/LoRA DP2 TP2 CP2 native reference parameterizations with
  dev 48, Klein 4B 24 and Klein 9B 32 attention heads, separate logical-DP operands, an omitted-CP-factor
  negative control, actual residual lengths and three gradient/Adam comparisons.
- `flux2-guard-ordered-full` and `flux2-klein-dppo-ordered-lora`: real native text/VAE reference
  conditioning, BF16 ordered reward overlap, evaluation, independent objective/Adam/EMA checks,
  exact checkpoint loading and tolerance-checked fresh-process continuation at the same topology.

Tiny fixtures retain published head counts, 128 packed latent channels, four rotary axes,
MLP ratio 3, both block types and native encoder layer taps. They reduce denoiser depth to one
block per stack, head width to 8 and text width to 24. This is bounded structural evidence,
not pretrained quality, convergence, throughput, arbitrary degree/mask/cache support, or
in-flight/topology-changing exact resume. Flux3 has a separate transformer layout and gate.
