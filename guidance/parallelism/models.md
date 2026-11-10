# Model coverage

All **19 registered model adapters** declare TP, an optional SP plan, and Ulysses CP with TP
composition. These are model-specific capabilities: the loaded variant, token geometry, tuning
mode, algorithm and backend must also satisfy their contracts. A declaration does not establish
that every model × algorithm × GPU layout has been measured.

Read coverage at three levels:

1. **Functional admission:** which native model layout the adapter accepts.
2. **Parameter placement:** which weights TP shards, and the denominator of any coverage count.
3. **Activation regions:** where SP retains token shards and where complete streams return.

Here, “replicated” means across TP peers. CP and FSDP can independently partition work or state.

## Supported model layouts

SP is on by default for `sd3-5` and `wan2_t2v` when TP is enabled. It is opt-in for every other
registered adapter. All CP rows below use Ulysses; none admits Ring.

| Registry key | Native layout and important bounds | Detailed contract |
|---|---|---|
| `sd3-5` | Medium, Large and Large-Turbo joint image/text streams; Medium's second image attention is retained. Large has 38 heads, so TP2 is valid while TP4/8 are not. | [SD3.5](../../docs/reference/parallelism/models/sd35.md) |
| `flux1`, `flux1-kontext` | Dual and single streams; Kontext retains reference-image packing and target cropping. Odd text is internally padded with an attention mask. | [Flux1](../../docs/reference/parallelism/models/flux.md) · [Kontext](../../docs/reference/parallelism/models/flux-kontext.md) |
| `flux2`, `flux2-klein` | Native dual/joint stacks with target, reference and text streams; per-stream lengths and heads must satisfy the declared TP/CP divisibility rules. | [Flux2](../../docs/reference/parallelism/models/flux2.md) |
| `flux3` | Base/distilled modes retain separate target, reference and text streams, per-token modulation and native packing. FSDP peers must agree on module-call schedules. | [Flux3](../../docs/reference/parallelism/models/flux3.md) |
| `qwen-image`, `qwen-image-edit-plus` | Joint image/text attention; Edit-2509/2511 preserve ordered references and their respective timestep layouts. Odd text and mask holes are preserved. | [Qwen Image](../../docs/reference/parallelism/models/qwen.md) · [Edit Plus](../../docs/reference/parallelism/models/qwen-edit.md) |
| `qwen-image-2.1` | Native block-causal joint calls and cached target calls, with separate prefix extraction. | [Qwen Image 2.1](../../docs/reference/parallelism/models/qwen21.md) |
| `z-image` | Flat T2I layout, two native refiners and the unified image/text stack. Native-padded stream lengths must divide TP; Omni/ControlNet layouts need separate support. | [Z-Image](../../docs/reference/parallelism/models/z-image.md) |
| `wan2_t2v`, `wan2_i2v` | Wan2.1/2.2 variants keep text/image conditioning, scalar or per-token timesteps and one/two-transformer schedules distinct. CP requires `cross_attention_fp32_v1`. | [Wan T2V](../../docs/reference/parallelism/context-parallel.md) · [I2V](../../docs/reference/parallelism/models/wan-i2v.md) · [Other variants](../../docs/reference/parallelism/models/wan-remaining.md) |
| `ltx2_t2av`, `ltx2_i2av` | Video and audio streams, text cross-attention and cross-modal attention; both token lengths must satisfy TP/CP divisibility. The optional absolute keyframe-position path is outside the SP plan. | [LTX-2](../../docs/reference/parallelism/models/ltx2.md) |
| `sensenova` | U1.0/U1.5 generation branches with whole prefix/cache preparation. Ordinary MLP and spatial pixel heads have distinct output boundaries; the optional deep-head variant is unqualified. | [SenseNova](../../docs/reference/parallelism/models/sensenova.md) |
| `bagel` | Packed generation queries with detached prefixes and role-specific branches. SP enables attention TP as well as residual sharding; supported attention TP degrees are 2/4. | [Bagel](../../docs/reference/parallelism/models/bagel.md) |
| `minimax-h3-t2va`, `minimax-h3-fl2va`, `minimax-h3-ref2va` | Packed video/audio/text, independent text refinement and ordered heterogeneous references. Ref2VA prepares its declared reference transformer. | [MiniMax H3](../../docs/reference/parallelism/models/minimax-h3.md) |

Model loading remains subject to each adapter's dependencies. In particular, optional model
packages and native attention kernels remain model-specific. Current adapters use released
Diffusers >=0.42.1; the measured CP dependency lines are Diffusers 0.42, Torch 2.10/2.11 and Accelerate 1.15.
Unmeasured newer versions warn instead of being rejected solely by release line. See the linked contract before choosing a checkpoint or environment.

## Parameter coverage

TP covers attention and feed-forward projections in the repeated stacks. Packed projections use
model-owned layouts that preserve canonical weight ordering. Flux1, SD3.5, ordinary Qwen and H3
also shard their declared modulation projections. Z-Image covers both refiners as well as its
main stack; H3 covers the text refiner and main stack. LTX-2 covers all six attention paths and
both modality FFNs. SenseNova covers both decoder branches.

Bagel is a deliberate exception to the usual SP toggle: without SP, its TP plan covers both
branches' MLPs; enabling SP also enables attention TP. Embeddings, output heads and conditioning
weights can remain replicated even when the surrounding token activations are sharded.

The following are existing reproducible counts of **canonical base parameter elements**, including
weights and biases, in the stated component. They exclude LoRA additions, buffers, optimizer state
and activations. They are placement counts, not memory reductions or speedups.

| Model / plan | Sharded / total elements | Coverage | Denominator |
|---|---:|---:|---|
| Flux1 / Kontext | 11,854,737,408 / 11,901,408,320 | 99.607854% | Whole standard guidance transformer |
| Qwen Image / Edit Plus | 20,405,772,288 / 20,430,401,088 | 99.879450% | Whole shared transformer; excludes Qwen 2.1 |
| SD3.5 Medium | 2,227,745,280 / 2,243,171,520 | 99.312302% | Whole Medium transformer |
| SD3.5 Large / Large-Turbo | 8,027,428,864 / 8,056,627,520 | 99.637582% | Whole Large transformer |
| Bagel, SP off | 11,406,409,728 / 14,141,252,608 | 80.660533% | Main language component, including both branches and vocabulary matrices |
| Bagel, SP on | 13,050,834,944 / 14,141,252,608 | 92.289101% | Same language component; includes sharded Q/K/V biases |

Bagel's denominator excludes its separate ViT, VAE, connectors and timestep/latent projections.
These counts and their controls are recorded in the
[parameter coverage reference](../../docs/reference/parallelism/tensor-parallel.md#canonical-parameter-coverage).
An aggregate H3 percentage has not been established for the current plan. This guide assigns no
new aggregate percentage to H3 or the other uncounted families.

## Sequence-parallel coverage

“Through the stack” means residuals stay sharded between consecutive blocks in that stack.
Attention and MLP operations may still gather their inputs. A gather at a native packing or
output boundary does not mean every block stores a full residual.

| Family | Activations retained as SP token shards | Where complete streams return or remain |
|---|---|---|
| SD3.5 | Image and text residuals, norms and modulation through the joint blocks; text FFNs where present; final image norm and projection | Embedding/conditioning production precedes the scatters. Odd text uses family-owned masked padding. Image output gathers after projection, before unpatchification. |
| Flux1 / Kontext | Both dual streams, the joint single stream, and final image norm/projection | Timestep/guidance conditioning and embedding production remain before SP. Joint attention preserves padded text/image order; final image output gathers before native output handling. |
| Flux2 / Klein | Both dual residual streams through the dual stack; joint residual through the single stack; final image norm/projection | Streams gather at stack exits for native concatenation and text cropping, then scatter into the next region. Modulation production remains replicated. |
| Flux3 | Each target/reference/text mode stack, then the joint stack; early modulation rows and final target norm/projection | Native mode-to-joint packing and target/reference cropping use complete CP-local streams. Shared joint modulation remains complete and each block selects its local rows. |
| Qwen Image / Edit Plus | Image and text residuals, norms, modulation and FFNs through all blocks; final image norm/projection | Text scatters after native rotary construction. Masks and rotary tables stay complete for gathered attention; image/reference outputs return in native order. |
| Qwen Image 2.1 | Active joint-query or cached-target residuals through all blocks and the final projection | Prefix extraction stays whole. Cached K/V, causal segment metadata and validity masks retain their model-owned layouts. Output gathers after projection. |
| Z-Image | Noise and caption refiners independently; unified main stack; final image projection | Each refiner gathers before native joint assembly. Input embeddings and conditioning production precede SP; final output gathers before unpatchification. |
| Wan T2V / I2V | Video residuals, norms and modulation through every denoising block and final projection | Text/image conditioning tokens remain whole across TP; attention receives the required complete operands. Output gathers after projection. |
| LTX-2 T2AV / I2AV | Both video and audio residuals, norms and modulation through all blocks and both final projections | Cross-modal attention gathers both required streams. Text conditioning and prompt modulation remain whole; outputs gather before native reconstruction. |
| SenseNova | Generation residuals and norms through the decoder; U1.0 ordinary pointwise MLP head also runs on local tokens | Prefix/understanding calls remain whole. Pointwise output gathers after projection; the pixel head receives complete decoder output for its spatial forward. |
| Bagel | Packed generation residuals through all decoder blocks and both final role norms | Detached prefix construction remains whole. Role indexes are local to each shard; tokens gather after final norms before packed-output restoration and downstream projections. |
| MiniMax H3 | Text refiner through its final norm; packed multimodal main stack, final norm and both output heads | Refiner padding is masked, gathered and removed before multimodal packing. Both output heads gather before native global video/audio selection. |

The current plans include the formerly omitted Flux1 single/text streams, Flux2/3 inter-block
residuals, Z-Image refiners, H3 text refiner, SD3.5/Qwen text streams and the output regions listed
above. This is a description of implemented regions, not a claim of maximum possible SP.

## Boundaries that remain intentional

Prefix extraction builds caches with model-specific causal and packing rules. Sharding that work
would need its own cache contract; it is not achieved by adding an SP style to a projection.
Similarly, SenseNova's pixel head uses spatial convolution and a complete image grid. Token slices
alone do not supply the neighboring pixels required by that operation.

Input embeddings, text encoders, VAE work, conditioning production and some native assembly stages
remain outside denoiser SP. Per-sample conditioning may stay replicated while its gradient sums
the contributions from token shards. LTX prompt modulation instead remains on a whole text stream
and must retain its existing gradient ownership.

Default LoRA targets and arbitrary custom targets are different contracts. Plain replicated
weights on token shards support the declared LoRA path; conditional or composed layouts need
specific support. For example, Qwen2.1's conditional final projection is not an admitted custom
LoRA target. Use the family contract when extending a target list.

Numerical tests observe local residual rows and compare derivatives and updates. They do not
supply a universal SP memory reduction or throughput gain; see [performance](performance.md).
