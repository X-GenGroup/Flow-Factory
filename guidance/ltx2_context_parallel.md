# LTX2 joint audio/video context parallelism

The `ltx2_t2av` and `ltx2_i2av` adapters admit the native LTX-2 and LTX-2.3 joint
transformers with `tp_size: 2`, `cp_size: 2`, `sequence_parallel: true` and FSDP2.
The layout identity is `ltx2_joint_audio_video`. CP1 retains the existing defaults;
sequence parallelism remains an explicit option for LTX2.

The family helper owns the native CP plan for both video and audio projections,
all four split rotary tables, per-token timesteps and both output gathers. CP
splits before SP and gathers after SP. Text and text masks remain replicated.
Both attention head counts must divide evenly across TP × CP. Video and audio
sequence lengths must each be divisible by CP × TP with SP, or by CP otherwise.
Packed training inputs contain one sample with exact video/audio geometry. I2AV
keeps its first latent frame immutable and excludes it from the training objective.

LTX-2.3 attention gates, prompt modulation, perturbed attention and cross-timestep
conditioning remain native. CFG, spatio-temporal guidance and modality isolation
retain the adapter's existing branches. The two native schedulers and ordered
video/audio trajectory own replay, likelihood weighting and active-element masks.

Diffusers 0.41 Ulysses backward assumes equal query and key/value lengths. LTX2's
family processor pads projected cross-modal operands to their local maximum
length after Q/K normalization and rotary embedding, masks padded keys, and trims
dummy query outputs before native gating/projection. Both directions use unchanged
native Ulysses autograd. This avoids dropping remote-query contributions to source
key/value gradients. Self-attention keeps the native processor.

Validation uses native 32-head video/audio transformers, 128 latent channels per
stream and unequal 12-video/8-audio token sequences. The four-token first-frame
condition ends inside a six-token CP chunk and crosses the three-token SP boundary. Reduced
layers and internal widths keep the fixture local; video VAE scale32/8, audio
8×16 packed channels, 128 connector registers, native Gemma tokenization and
stereo24k decoding retain their structural contracts. Both 2.0/2.3 and T2AV/I2AV
forms have independent full/LoRA output, every-gradient and three-Adam-step controls.

The CPU contracts exercise real conditioning, saved pipeline reload, guided AV
rollouts and the actual GRPO loss. A separate four-rank Gloo development control
compares native CP+TP/SP gradients in FP64. Exact selected GPU evidence uses
DP2 × TP2 × CP2 on eight devices, including conditioned acquisition, varying video
rewards, joint AV likelihood derivatives, ordered overlap, evaluation, model/EMA
updates and exact checkpoint loading/work identity. Fresh-process continuation follows the
[ordered contract](context_parallel.md#ordered-continuation-evidence): the first same-state
update retains the original BF16 bounds, and each later actual objective/derivative, optimizer,
EMA and state check remains required. Later cross-run numerical statistics are diagnostic.
The objective oracle observes native scheduler tensor endpoints
before I2AV repacks its conditioned frame; it independently checks component and
active-element-weighted joint likelihoods and both component derivatives.

The eight-rank numerical reference uses AdamW `lr=1e-3`, matching the native
lifecycle fixture's learning-rate scale. It retains the probe's `eps=1e-4`,
`weight_decay=0.01`, raw-gradient objective and independent optimizer histories;
the lifecycle separately uses clipping and its configured weight decay. Every
rank records actual/reference optimizer settings. Output, gradient, update and
moment bounds, all-element checks and negative controls remain unchanged.

These are model-structure proofs. They do not establish pretrained generation
quality, production-scale memory requirements or interleaved-rotary variants.
The latter fail preflight because published LTX-2/2.3 metadata uses split rotary.
