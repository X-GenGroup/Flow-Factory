# Experimental Flux1-Kontext context parallelism

Flux1-Kontext owns the `flux_reference_image_text` layout. The existing native Flux transformer
processes `[target image tokens | reference image tokens]` alongside the text stream. Diffusers
splits the combined image sequence and its rotary IDs at the transformer root and gathers the
whole image output at `proj_out`. The adapter then crops predictions to the target token count.
Reference latents stay conditioning; they never become generated output or scheduler state.

The model reuses Flux1's guidance-distilled transformer metadata/processor checks and TP plan
for both transformer stacks, including canonical packed single-stream output and AdaLN
projections. Attention heads must be divisible by TP times CP. Both transformer stacks and
native `FluxAttnProcessor` are required.
`sequence_parallel: true` opts into the [Flux dual-stream image SP region](flux_context_parallel.md#tensor-parallel-composition)
for the combined target/reference stream. Image tokens restore their CP-local order before the
first single block concatenates text and image. Text and single-stream activations stay TP
replicated; SP is off by default. No CP/SP boundary relocation, trainer fork or additional
gradient scaling owner is introduced.

CP inputs require batch one, positive target and reference sequences with the native packed
channel width, and combined image/text token counts divisible by CP. With SP, combined image
tokens must also be divisible by TP times CP; individual target/reference spans need not be.
The target/reference boundary
need not align with a CP split. Combined rotary IDs must match the image sequence, with finite
coordinates, target tag zero and single-reference tag one. Multiple/empty reference-image lists, missing conditions, mismatched
widths, invalid ID geometry, IP adapters, special masks and substituted attention processors are
rejected before collectives. The ordinary CP=1 input policy is unchanged.

Use the shared FSDP2/AdamW/BF16/native-attention configuration with one transformer target,
replicated DP=1 and sharded DP>=2. Admission composes with the existing concrete trainer contracts;
it is not a full model-by-algorithm qualification. The directly selected topology is DP2 TP2 CP2.
The new layout has a separate CP runtime identity from ordinary Flux1. Existing CP=1 identities
are unchanged; exact continuation is qualified only for the same source and topology.

The selected evidence consists of:

- `flux-kontext-cp-contracts`: real serialized tiny CLIP/T5/VAE/Flux components, independent VAE
  posterior-mode encoding and target-only crop, image and rotary-tag ablations, full/LoRA active
  gradients, frozen LoRA base parameters, loader/configuration/ID rejection and opt-in SP admission.
- `flux-kontext-tp-cp-reference`: native full and LoRA FP32 preparation with DP2 TP2 CP2 and
  activation checkpointing. An unsharded module computes the mean over the two logical samples.
  Every gradient, three Adam updates including warm-state moments, and inactive parameters are
  compared. Omitting the CP loss factor must produce half-sized gradients and fail equivalence.
- `flux-kontext-guard-ordered-full`: native BF16 Guard generation and ordered asynchronous CPU
  reward overlap, independent objective derivatives and clipping/Adam/EMA, drained evaluation,
  exact checkpoint loading and fresh-process continuation with the existing BF16 update bounds.
  Actual reference-image operands and retained same-policy replay are checked throughout.

The fixtures encode a real 64x64 PNG at an explicit 32x64 condition size, yielding eight reference
tokens alongside sixteen generated target tokens. The native split therefore crosses the target
segment. This exercises explicit fixed-size resizing and reference encoding; it does not qualify
automatic preferred-resolution selection, arbitrary masks, multiple references, pretrained image
quality, convergence, throughput or topology-changing resume. Source-bound selected results,
rather than historical ordinary-Flux receipts, determine acceptance.

The SP extension adds `flux-sequence-contracts`, `flux-dual-sp-cp-reference`, and
`flux-kontext-sp-guard-ordered-full`. The first two retain native dual/single geometry and check
all gradients and three Adam updates for full and LoRA tuning. The lifecycle uses the same real
reference encoding and ordered Guard/evaluation/continuation owners with SP enabled, and records
actual TP/CP coordinates plus image-token lengths before and after the SP region. Existing
SP-disabled case definitions remain intact; their receipts alone do not qualify the new region.
