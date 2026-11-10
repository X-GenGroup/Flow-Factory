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
input/output token count even for uneven final chunks. The text refiner also shards native
attention and packed SwiGLU weights. Its SP region pads and masks text, retains residual shards
through every refiner block and final norm, then gathers and removes padding before multimodal
assembly. AdaLN input-column shards retain complete
native modulation tables; SP sums each table's token contributions before rowwise
backward. This also restores full time-embedding gradients without a second sum.
The timestep embedding width must be divisible by TP.

Natural prompts need no divisibility adjustment with SP, with or without CP.
After the native text refiner, family hooks append dummy packed rows, modulation
indices and identity rotary rows up to a multiple of TP × CP (TP when CP=1).
Native attention masks those rows as keys. Native output indices select only
original video/audio rows, so dummy rows never become generated or conditioned
media. Each block derives its own valid length from that invocation's original
index argument, including checkpoint recomputation. CP cuts precede SP cuts; both
output heads gather SP before native CP gathers. Without CP, the SP plan's
`H3PackedSequencePadding` installs these hooks ahead of the SP scatters and their
divisibility guards; CP removes them and installs its own TP × CP padding. A
packed length that already divides TP keeps the unmasked native attention.

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
mutation. Main-stack forward and transport ownership stay native. Model-owned refiner subclasses
preserve native residual operations while carrying an explicit padding mask as a checkpoint
argument; they do not replace parameter objects.

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
ordered overlap, clipping/Adam/EMA, exact checkpoint load/work identity and the
first same-state resumed update under the original BF16 bounds. Each later actual
trajectory retains independent objective/public-output derivative and optimizer/state
checks; later cross-run numerical comparisons are diagnostic. Tiny fixtures reduce model
depth and internal width; they do not claim released checkpoint quality, memory fit or throughput.
The earlier H3 head `75e9fac069b1b7043ce1f4143a9a31db6d987bd9` passed its selected report and imported evidence in [#344](https://github.com/X-GenGroup/Flow-Factory/pull/344). It predates refiner SP; results below retain their original SHAs and do not certify the expanded plan.

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

A repeated DP2×TP2/CP1 run of the same canary at main `dc17e8f6` still exhausted memory
during FSDP2 initialization: TP preparation left 61.94 GiB allocated per rank, and
77.21 GiB were allocated when `fully_shard` failed. Eight-GPU DP4×TP2/CP1 completes
FSDP2 initialization at a 77.26 GiB peak and leaves 15.45 GiB allocated, but the
allocator then still reserved 77.28 GiB. Without a cache release, the first forward
failed in cuBLAS handle creation and NCCL allocation. The FSDP2 backend now calls
`torch.cuda.empty_cache()` once after preparation. With that release, all eight ranks
went from 77.28 to 15.54 GiB reserved and completed the bounded native AV LoRA step at a
28.33 GiB peak. Each rank had 28 local attention heads, 51 modulation placements and
416 finite gradient tensors, 208 of them nonzero. DP shards 0-1 updated 416 tensors and
shards 2-3 updated 208. The release does not lower the 77.26 GiB initialization peak,
which leaves under 2 GiB of headroom on 79.19 GiB devices. DP2×TP2 remains unsupported.
Run this placement with `FF_H3_CAPACITY_DP=4 FF_H3_CAPACITY_TP=2`.

With `parallelism.target_materialization` unset or `sharded` (H3 resolves an unset value to
`sharded` under FSDP2 without CPU offload, including with CP), target parameters are on meta until
FSDP2 preparation finishes. Each rank then allocates and fills only its own local shards.
The same canary, run with `FF_H3_TARGET_MATERIALIZATION=sharded` on unpatched Accelerate
1.15.0, passed every rank. Per-rank memory:

| Placement | Before FSDP2 | Preparation peak | After preparation | Step peak |
|---|---|---|---|---|
| DP2×TP2, 4 GPUs | 0 GiB | 30.98 GiB | 30.98 GiB | 43.86 GiB |
| DP2×TP2 with `fsdp_cpu_ram_efficient_loading`, 4 GPUs | 0 GiB | 32.24 GiB | 30.98 GiB | 43.86 GiB |
| DP4×TP2, 8 GPUs | 0 GiB | 15.48 GiB | 15.48 GiB | 28.36 GiB |

In every run, each rank's loss, finite and nonzero gradient counts and updated-tensor count
equal the replicated path. Host RSS stays at 133 GB per process, because every rank still
loads the root on CPU first. These runs qualify the measured LoRA loading and bounded-step
placements only. They make no claim about full-model training memory, CP or media quality.

The opt-in Q3 case `minimax-h3-pretrained-tp4-loading-capacity` uses the same released
Ref2VA checkpoint at revision `42ed227ee7df40d41602854ae760620d6eb651fe` on eight
GPUs: DP2×TP4, SP enabled, CP1. It preserves FP32 original parameters and BF16
compute. Estimated base parameter residency is 33.14 GiB before FSDP and 16.57 GiB
after FSDP per rank, excluding LoRA, buffers and preparation transients. The actual
case passed all 8 ranks at `414d10acfac3c0173045e7fe2dfbde7c3c4f2379`. Measured PyTorch allocation
was 33.225 GiB before FSDP and 16.608 GiB afterward; initialization peaked at 49.833 GiB
and the bounded native AV step at 23.849 GiB per rank. Maximum reserved memory was
50.646 GiB. All ranks retained actual weight hashes, 14 local attention heads and
51 native modulation placements, then completed finite LoRA backward/AdamW: 416
active gradient tensors, 208 nonzero gradients and 416 updated tensors per rank.
AdamW weight decay can update tensors with zero gradients. These PyTorch figures
exclude non-PyTorch device consumption. It does not claim released-model CP2 capacity, CP numerical equivalence or
full-model training capacity. Set
`FF_RUN_H3_CAPACITY=1` and `FF_H3_MODEL_PATH` to a staged snapshot; without the path,
the opt-in test resolves that pinned revision from the ordinary Hugging Face cache
without downloading. The selected runner requires the model path explicitly.

The separate fresh final-source capacity run at `75e9fac069b1b7043ce1f4143a9a31db6d987bd9` passed all eight ranks: 33.225 GiB allocated before FSDP, 16.608 GiB after FSDP, 49.833 GiB initialization peak and 23.849 GiB native-step peak per rank. Maximum reserved memory was 50.646 GiB. Each rank checked 51 modulation placements, 416 finite gradient tensors, 208 nonzero gradients and 416 updated tensors. The scope is released Ref2VA DP2×TP4/SP/CP1 LoRA loading and one bounded native AV update, with FP32 originals and BF16 compute. It does not establish released CP2 fit, full-model training capacity, serial numerical parity, media quality or performance. [H3 PR](https://github.com/X-GenGroup/Flow-Factory/pull/344) · [Fresh evidence](https://github.com/X-GenGroup/Flow-Factory/actions/runs/42426359108/artifacts/13595548123).

H3 TP/SP-plan or native materialization changes select this capacity obligation;
ordinary conditioning/configuration changes retain their focused native controls.
Reviewers must add this case when shared loading, original-dtype materialization
or TP application changes can alter H3 pre-FSDP residency.

See [model integration](../../../development/new-model.md), [parallelism](../topology.md), and
[validation](../../../development/validation/validation.md).

## Text-refiner SP selection

`minimax_h3-refiner-sp-cpu`, `minimax_h3-refiner-sp-cuda` and
`minimax_h3-refiner-sp-cp-reference` select the new refiner attention/packed-FFN TP and masked SP
region. The refiner's valid-key mask is an explicit block argument so checkpoint replay retains
its own length across later pending forwards. Main-stack CP padding remains a separate owner;
CP1 main packed-token divisibility is unchanged. Full/expanded-LoRA controls compare derivatives,
three updates/moments and canonical state, and observe actual local refiner rows.

The current plan has no newly reproduced aggregate TP coverage percentage or measured refiner
memory benefit. The released-weight capacity observations above remain historical results for
their exact source, not a fresh qualification of the refiner extension. The selected plan determines
whether the final source also requires a repeated capacity run.
