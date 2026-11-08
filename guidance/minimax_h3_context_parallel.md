# MiniMax H3 packed audiovisual TP/SP/CP

The three registered native workflows support Ulysses context parallelism:
`minimax-h3-t2va` and `minimax-h3-fl2va` prepare `transformer`;
`minimax-h3-ref2va` prepares `transformer_ref`. The family owns its typed
`minimax_h3_packed_audio_video_text` capability and native split/gather composition.
Use the canonical component in `target_components`. Batch size remains one,
guidance remains 1.0, and negative prompts remain unsupported.

The released structure has 56 attention heads with width 128, video 24 latent
channels patched by (1,2,2), audio 32 channels and packed SwiGLU projections. TP
and CP degrees must divide the head count together. Packed FFN TP keeps the
native two-dimensional checkpoint weight order. With SP, each existing FFN
chunk gathers at `net.0` and reduce-scatters at `net.2`, preserving its local
input/output token count even for uneven final chunks. Text refiners remain
replicated and retain native attention. AdaLN input-column shards retain complete
native modulation tables; SP sums each table's token contributions before rowwise
backward. This also restores full time-embedding gradients without a second sum.
The timestep embedding width must be divisible by TP.

Natural prompts need no divisibility adjustment when CP is active. After the
native text refiner, family hooks append dummy packed rows, modulation indices
and identity rotary rows up to a multiple of TP × CP. Native attention masks
those rows as keys. Native output indices select only original video/audio rows,
so dummy rows never become generated or conditioned media. Each block derives
its own valid length from that invocation's original index argument, including
checkpoint recomputation. CP cuts precede SP cuts; both output heads gather SP
before native CP gathers. CP=1 with SP retains the existing requirement that the
packed sequence divide TP.

Ref2VA's in-forward FSDP2 checkpoint runs inside the native block body, after
CP/SP cuts and backend input casts. CP-only uses the same verified hook relocation.
Each checkpoint captures its invocation's valid-row count and restores it during
replay, so pending forwards with different packed lengths keep independent masks.
Only the exact native bound forward restored by Diffusers is admitted; foreign
wrappers and changed boundary callbacks remain errors. CPU controls compare FP32
and BF16 execution with and without this checkpoint, including all gradients,
and reject the deliberately wrong snapshot across two pending lengths.

All conditioning remains native: FL2VA first/last anchors; Ref2VA ordered image,
audio and video references, their source rates and shared rotary clock; Qwen3-VL
hidden_states[50]; independent video shift 12/audio shift 3 schedules; immutable
condition prefixes; data-ward velocity; target-only joint likelihood. Unknown
attention processors, fused layouts, incompatible native geometry, malformed
packed partitions and invalid timestep indices fail before collectives or hook
mutation. No transformer forward fork or generic transport change is introduced.

The selected native reference case retains actual 56×128 attention geometry,
96-wide packed video and 32-wide audio, all three layouts and full/LoRA modes.
Odd sequence lengths cross CP/SP boundaries. It checks outputs, every parameter
gradient, three Adam updates and moments against an independent unsharded model,
with omitted-CP-reduction and unmasked-padding negative controls. CPU development
controls additionally exercise CP-only and TP+CP without SP; their real Gloo
rank counts are distinct from the selected CPU1 contract case.

Native component fixtures retain real VL/vision and both VAE encoders/decoders,
51 conditioner layers, source-rate-aware ordered references, 124 generated frames
at 24 fps and stereo 32 kHz audio. Selected FL2VA LoRA and Ref2VA full GRPO lifecycles
cover finite varying rewards, joint AV scheduler derivative checks, evaluation,
ordered overlap, clipping/Adam/EMA, exact checkpoint load/work identity and
tolerance-checked BF16 continuation. Tiny fixtures reduce model depth and internal
width; they do not claim released checkpoint quality, memory fit or throughput.
Exact GPU qualification is recorded by the reviewed commit's selected evidence.

The released 33.123-billion-parameter transformer loads on CPU before TP and
FSDP preparation. Attention-only TP2 would retain about 109.04 GiB per rank in
FP32; packed FFN plus AdaLN TP reduces the estimated pre-FSDP resident parameters
to 63.23 GiB. With DP2×TP2×CP2, Accelerate flattens DP and CP into the four-rank
`dp_shard_cp` FSDP axis. Estimated FP32 parameters, gradients and two Adam moments
then total about 63.23 GiB per rank, excluding activations and preparation
transients. Actual-width two-rank CPU controls cover the 2688→96768 AdaLN
projection, full/LoRA gradients and three optimizer steps. These estimates and
controls do not establish actual-weight GPU capacity; that requires a separate
prepared-checkpoint measurement.

The recorded DP2×TP2/CP1 FP32 capacity attempt exhausted memory during FSDP2
initialization on 79.19 GiB GPUs. TP preparation completed with 63.39 GiB allocated
per rank. Accelerate 1.15 retains `original_sd = model.state_dict()` while native
FSDP allocates new shards, so the original TP storage and approximately 31.70 GiB
of new shards coexist. The failed attempt and its all-rank receipts remain separate
evidence; neither the parameter estimates nor the tiny CP controls establish that
the released model fits DP2×TP2×CP2 on eight GPUs.

The opt-in Q3 case `minimax-h3-pretrained-tp4-loading-capacity` uses the same released
Ref2VA checkpoint at revision `42ed227ee7df40d41602854ae760620d6eb651fe` on eight
GPUs: DP2×TP4, SP enabled, CP1. It preserves FP32 original parameters and BF16
compute. Estimated base parameter residency is 33.14 GiB before FSDP and 16.57 GiB
after FSDP per rank, excluding LoRA, buffers and preparation transients. These are
estimates until the selected case measures the reviewed candidate. The case records
all actual weight hashes, every rank's memory peaks, 14 local attention heads and
51 native modulation placements, then checks one finite native AV LoRA backward/Adam
step. It does not claim released-model CP2 capacity, CP numerical equivalence or
full-model training capacity. Set
`FF_RUN_H3_CAPACITY=1` and `FF_H3_MODEL_PATH` to a staged snapshot; without the path,
the opt-in test resolves that pinned revision from the ordinary Hugging Face cache
without downloading. The selected runner requires the model path explicitly.

H3 TP/SP-plan or native materialization changes select this capacity obligation;
ordinary conditioning/configuration changes retain their focused native controls.
Reviewers must add this case when shared loading, original-dtype materialization
or TP application changes can alter H3 pre-FSDP residency.

See [model integration](new_model.md), [parallelism](parallelism.md), and
[validation](validation.md).
