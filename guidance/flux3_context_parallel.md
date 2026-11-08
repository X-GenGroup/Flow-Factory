# Flux3 base and distilled TP/SP/CP

`flux3` owns a native Ulysses CP plan for the base and step-distilled flow-map
transformers. CP splits target, reference and text streams and their four-axis
position IDs at the transformer input, before native per-token modulation. The
native final projection gathers only target outputs. Model and trainer forwards
continue to use native Diffusers attention, rotary arithmetic and autograd.

This path requires Diffusers `0.41.0.dev0` from commit
`7ecb1e1b4fe500be7fb7381a7bb79a24d7fa3941`, with a recorded VCS installation.
That revision requires Python 3.11 or newer. A wheel with the same version string
or a different optional-model revision does not satisfy CP admission.

TP shards separate Q/K/V projections and the native packed `[gate, up]` SwiGLU
projection in every image, reference-image, text and joint block. Packed weights
remain canonical two-dimensional parameters; the shared packed style exchanges
activations reversibly. PEFT names and canonical checkpoint tensor ordering remain
unchanged. This adds communication and makes no speed or memory-saving claim.

`parallelism.sequence_parallel: true` additionally shards each block's residual
and per-token modulation. Joint blocks concatenate local text and image tokens;
the model-owned boundary reorders the corresponding rotary rows to the same
rank-major TP gather order, then gathers each output stream separately. SP is off
by default. The CP root splits and final target gather enclose those block-local
SP boundaries.

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
eight default hidden-layer taps; real VAE references resize 64×128 images to 32×64.
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
[ordered contract](context_parallel.md#ordered-continuation-evidence). The first same-state
update retains its original bounds; each later actual objective/derivative, optimizer, EMA
and state check remains required, while later cross-run numerical statistics are diagnostic.
Delivery requires those cases on the reviewed immutable candidate; source and
CPU checks alone do not establish a GPU pass. The evidence does not claim Ring,
reference masks, reference KV caches, pretrained weights or variable batch CP.
