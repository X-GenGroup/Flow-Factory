# Flux3 base and distilled TP/SP/CP

`flux3` owns a native Ulysses CP plan for the base and step-distilled flow-map
transformers. CP splits target, reference and text streams and their four-axis
position IDs at the transformer input, before native per-token modulation. The
native final projection gathers only target outputs. Model and trainer forwards
continue to use native Diffusers attention, rotary arithmetic and autograd.

This path uses released `diffusers>=0.42.1` and Python 3.11 or newer, as declared in
`pyproject.toml`. It does not require a Git installation or an exact source revision.
CP uses the shared qualified 0.42.x release-line gate. Reference preprocessing follows the released native processor: one image is reflect-padded
onto its token-grid canvas; multiple references are center-cropped to that grid.
`condition_image_size: null` (the default) retains native limits: 4 megapixels for a single
reference canvas and 1 megapixel per reference for multiple images. An integer sets a
maximum area of `size²`; `[height, width]` sets `height × width`, preserving aspect ratio.
For example, `condition_image_size: [768, 1024]` caps each reference at that area.
The budget replaces the native area limit; native resizing, padding/cropping and normalization
remain authoritative. A single-reference budget must fit the native minimum 64×64 canvas;
the native 4096-pixel side limit remains in force. The pipeline processor is never mutated.

Generation follows the native output rule too. With both `height` and `width` omitted,
I2I uses the first reference canvas; a single reference is then cropped back to its valid
region (after any area downscale). Supplying either dimension disables crop-back; the other
omitted dimension still follows the first reference canvas. T2I defaults to 1024×1024.
Training and evaluation normally supply explicit configured dimensions. Offline supervised
targets keep their configured output geometry.

Preprocessing caches retain the native `reference_crop`; the new cache version excludes old
crop-only entries. `Flux3Sample.height/width` describe the denoising canvas, while
`reference_crop` identifies the visible output. It survives sample transport and can be passed
to `decode_latents` to reconstruct the same cropped pixels. Bare reference latents need explicit
sizes unless their prepared condition images and crop metadata are also supplied.

TP shards separate Q/K/V projections and the native packed `[gate, up]` SwiGLU
projection in every image, reference-image, text and joint block. Packed weights
remain canonical two-dimensional parameters; the shared packed style exchanges
activations reversibly. PEFT names and canonical checkpoint tensor ordering remain
unchanged. This adds communication and makes no speed or memory-saving claim.

`parallelism.sequence_parallel: true` keeps residual shards between consecutive blocks in each
image, reference-image, text and joint stack. Each stack has one residual scatter and one gather.
The native root consumes complete CP-local streams at mode-to-joint concatenation and target/reference
cropping. Joint blocks retain local text/content pairs and reorder rotary rows to the corresponding
rank-major attention gather. Attention/FF projections still gather inputs and reduce-scatter outputs.

Early per-token modulation is sharded once at its producer and cloned to avoid retaining the full
producer storage. Shared joint modulation stays complete because the root concatenates target and
reference modulation; each joint block selects its own rows. Final target norm/projection executes
on local tokens after the crop, then restores SP before CP. Whole PEFT output wrappers preserve
custom projection LoRA. SP remains opt-in.

Metadata validation happens before weight loading. Dense native attention is
required; masks, custom processors and extra attention/cache arguments are not
admitted. Every native forward uses one sample. Each positive/negative text,
target and present reference stream must divide CP, or CP×TP with SP enabled.
The early per-mode attention stacks require individual divisibility even when the
combined target/reference length divides. Position IDs must be finite and match
the corresponding token count; orphan reference IDs are rejected.

FSDP2 peers must follow the same native module-call schedule. The adapter checks
per-sample call counts before rollout/replay fallback loops, then reference
presence and the effective CFG branch before each native transformer call. It
compares the actual FSDP shard and replication mesh axes, excluding TP, and
finishes every axis comparison before raising on all affected peers. Uniform
T2I and I2I remain supported; reference token counts may differ across logical DP
peers. A mixed reference-present/reference-free shard group is rejected before
its condition stack can diverge. Serial and DDP execution add no guard collective.

The base checkpoint's omitted `flow_map` field retains the native `false`
default. It uses conventional positive/negative CFG. The distilled checkpoint
sets `flow_map: true`, accepts no negative prompt and supplies the next timestep
to the native target-time embedding for every mode. Reference image encoding
keeps native VAE normalization, packed channels and ordered reference positions.

For singleton image-conditioned/ragged forward calls, the public scheduler output
now retains the original tensor objects. Previously unconditional concatenation
disconnected the returned mean/velocity from the returned log probability.
Outputs, real Guard loss and parameter gradients remain exactly equal; the public
endpoint derivatives now expose their complete upstream graph. Multi-sample
aggregation retains its previous path.

The selected structural fixtures retain 24 attention heads, four rotary axes,
128 packed channels, SwiGLU ratio3 and all four native block stacks. Head width
and depth are reduced; these are random-weight numerical fixtures, not pretrained
quality or performance claims. Real Qwen3-VL uses the native 80-token window and
eight default hidden-layer taps; real VAE references use the native 64×128 canvas; separate pixel oracles verify custom area budgets.
CPU tests compare native conditioning, all denoising transitions and decoded
outputs, full/LoRA gradients, three Adam updates/moments and canonical reload.
A subsequent reference-free control checks native T2I output/gradients and keeps
only the unused image-condition trainables and their warm optimizer state unchanged.

The eight-rank numerical reference uses AdamW `lr=1e-3`, matching the native
lifecycle fixture's learning-rate scale. It retains the probe's `eps=1e-4`,
`weight_decay=0.01`, raw-gradient objective and independent optimizer histories;
the lifecycle separately uses clipping and its configured weight decay. Every
rank records actual/reference optimizer settings. Output, gradient, update and
moment bounds, all-element checks and negative controls remain unchanged.

Native qualification selects four full/LoRA base/distilled DP2 TP2 CP2 reference
variants and two BF16 lifecycles: base Guard full and distilled DPPO LoRA. Each
lifecycle exercises actual image conditioning, ordered reward overlap, evaluation,
exact checkpoint-loaded state and fresh-process continuation under the
[ordered contract](../context-parallel.md#ordered-continuation-evidence). The first same-state
update retains its original bounds; each later actual objective/derivative, optimizer, EMA
and state check remains required, while later cross-run numerical statistics are diagnostic.
Delivery requires those cases on the reviewed immutable candidate; source and
CPU checks alone do not establish a GPU pass. The evidence does not claim Ring,
reference masks, reference KV caches, pretrained weights or variable batch CP.

## Continuous-stack SP evidence

`flux3-continuous-sp-cp-contracts` observes multiple blocks in every active stack, per-token
modulation and final target rows. `flux3-continuous-sp-cuda` and
`flux3-continuous-sp-cp-reference` select component and prepared TP/CP compositions. Base/distilled
conditioning, full/expanded-LoRA derivatives, checkpoint replay, three updates/moments and canonical
state remain explicit obligations. Historical block-local-SP receipts retain their original scope;
new boundary coverage needs its final source-bound results. No throughput or peak-memory gain is
claimed from the local-row observations.
