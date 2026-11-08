# SD3.5 TP/SP and context parallelism

SD3.5 owns a joint image/text CP layout over native Diffusers Ulysses. Both Medium's second image-only attention and Large's ordinary joint blocks retain their native projection, normalization, modulation and residual owners. TP, image-only SP and CP can compose on independent rank axes.

```yaml
model:
  model_type: sd3-5
  target_components: [transformer]
parallelism:
  dp_shard_size: 2
  tp_size: 2
  sequence_parallel: true
  cp_size: 2
mixed_precision: bf16
acceleration:
  shared:
    - name: attention_backend
      params: {backend: native}
```

Use the existing FSDP2 launcher and ordinary algorithm/data/model settings. The eight-rank shape is DP2 x TP2/SP x CP2. The [shared admission and checkpoint rules](context_parallel.md) remain authoritative: one logical example per device, AdamW, explicit algorithm capability, the pinned release dependency lines, and drained fixed-topology continuation.

## Native layout ownership

The image patch embedding and text projection split along CP. SP further splits only image residuals inside each CP shard; the text residual remains replicated across TP peers. SP gathers before the native CP output gather, restoring image order before unpatchification. Medium keeps the native second image-only attention.

Real head counts are 24 for Medium and 38 for Large. Large has 19 heads per TP2 rank. The model-local processor pads only independent Q/K/V attention heads from 19 to 20 for Ulysses, then discards the dummy head before the original output projection. Parameter names, tensor shapes, declared head counts and checkpoints do not change. This is Ulysses; it does not claim a model-level Ring path.

Native CLIP/T5 conditioning can contain 333 or 589 text positions. Only temporary text activations are padded for CP divisibility, with padded keys masked in the exact rank-major Ulysses order. Original text, pooled conditioning and image output lengths remain unchanged. Image patches must divide evenly by CP and enabled SP. Unsupported custom attention kwargs, ControlNet residuals and skipped blocks fail before native attention collectives.

## AdaLN tensor placement

The TP plan shards `norm1.linear`, `norm1_context.linear` and `norm_out.linear`
on their input-feature axis. Their conditioning input is replicated; the native
rowwise contraction slices that input and sums its output. All packed nine-way
(Medium dual attention), six-way and two-way shift/scale/gate outputs keep their
original ordering and full width. Biases stay replicated. Image SP composes the
existing modulation-output gradient sum after the rowwise projection; text
modulation has no extra SP reduction. CP installation and default attention-only
LoRA targets are unchanged. Explicit LoRA on a modulation linear remains
unsupported with SP, because the shared plan rejects composed activation styles
on LoRA wrappers before preparation; TP-only uses the ordinary rowwise pairing.

The following counts use every transformer parameter as the denominator, excluding
buffers and auxiliary encoders. The numerator counts parameters whose TP placement
is sharded, including column-parallel biases but excluding row-parallel biases.
These are architecture counts, not measured memory savings.

| Native config | Transformer parameters | Sharded before | Sharded after | Coverage before → after |
|---|---:|---:|---:|---:|
| Medium | 2,243,171,520 | 1,460,974,080 | 2,227,745,280 | 65.13% → 99.31% |
| Large / Large-Turbo | 8,056,627,520 | 5,342,189,568 | 8,027,428,864 | 66.31% → 99.64% |

`test_sd35_adaln_meta_layout_and_parameter_coverage` reconstructs these complete
configs on the meta device and checks every modulation shape and added sharded
weight. Its config inputs come from the published [Medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium/blob/main/transformer/config.json),
[Large](https://huggingface.co/stabilityai/stable-diffusion-3.5-large/blob/main/transformer/config.json)
and [Large-Turbo](https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo/blob/main/transformer/config.json)
transformer metadata. Medium has 24 blocks and 24 heads, with dual attention in
blocks 0–12; Large and Large-Turbo share the same 38-block, 38-head layout. All
use 64-wide attention heads.

The incremental CUDA selector `test_cuda_sd35_adaln_tp2_supervised_update` uses
only two devices (DP1, TP2, SP enabled, CP1), reduced depth and the native head
geometry. It compares one component-level supervised flow/MSE update against the
unsharded model, including every parameter gradient, Adam state and parameter.
It covers Medium and Large norm paths; it is not a full `SFTTrainer` lifecycle or
a pretrained-model capacity measurement. Focused CPU controls additionally cover
SP disabled, default LoRA and existing TP/SP+CP composition. The source describes
the test contract; execution results remain attached to the reviewed candidate.

## Evidence

- CPU controls preserve ordinary no-CP outputs and every gradient, native CLIP/T5/VAE save-reload conditioning, metadata rejection before weight loading and invalid-input rejection before collectives.
- Four-rank Gloo development controls retain Medium/Large head counts, 589 text positions, full/LoRA and eager/checkpointed execution; all active gradients and three Adam updates/moments match an unsharded reference.
- `sd35-tp-sp-cp-reference` uses eight CUDA ranks, FSDP2, real 64-wide attention heads, full/LoRA and backend recomputation. It compares full outputs, every gradient, three Adam updates/moments and a missing-CP-loss-factor negative control with independent logical-DP means.
- `sd35-dppo-ordered-full` exercises actual Large conditioning and decoded-image rewards through DPPO, ordered reward overlap, on-policy replay, evaluation, EMA and fresh-process drained continuation. It requires exact checkpoint/work restoration, the first same-state update under the original bounds, and each later actual run's objective/derivative, optimizer, EMA and state checks under the [ordered continuation contract](context_parallel.md#ordered-continuation-evidence). Later cross-run numerical statistics are diagnostic.

Native model depth and encoder widths are bounded in these fixtures. The catalog and exact-source PR evidence identify the executed selectors; they do not claim pretrained convergence or a throughput measurement.
