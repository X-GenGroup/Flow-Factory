# Condition dropout in supervised training

SFT can train on a demonstration with part of its instruction removed. The output image, video or audio remains the target. Flow-Factory samples the condition once per record and data epoch and reuses it for every flow-matching time sampled for that record.

```yaml
train:
  trainer_type: sft
  condition_dropout: 0.1
```

| Setting | Effect on an eligible record |
| --- | --- |
| Omitted or `0.1` | Use the model's numeric policy. About one record in ten receives a dropped condition. |
| `0` | Use the original condition and skip preparation of alternatives. Set this explicitly to continue a checkpoint created before condition dropout. |
| A number in `(0, 1]` | Draw a nonempty drop mask with that probability. Use `1` to exercise the dropped branches on every eligible record. |
| `native` | Use the published training sampler and rates for a qualified checkpoint. Admission checks run before model loading. |

### What each model drops

| Adapter or family | Dropped presentation | Retained inputs |
| --- | --- | --- |
| `sd3-5`, `flux1`, `z-image`, `qwen-image` | Encode the empty caption through the model's text encoders, including pooled outputs and masks where applicable. | Time, image geometry and embedded guidance follow the existing SFT recipe. |
| `flux1-kontext`, `flux2`, `flux2-klein`, `flux3` | Empty caption for generation and editing. For Klein Base this matches its empty-text CFG arm before mixing. On guidance-distilled variants it is a training augmentation. | The ordered edit references and their token positions. |
| `qwen-image-edit-plus`, `qwen-image-2.1` | Encode an absent instruction with the original reference images through the multimodal encoder. | Reference order, image tokens and masks. Two different reference sets can produce different empty-instruction embeddings. |
| `wan2_t2v`, `wan2_i2v`, `ltx2_t2av`, `ltx2_i2av` | Omit the caption within the selected generation task. | First/last frames as required by the checkpoint, AV targets and existing supervision masks. |
| `minimax-h3-t2va`, `minimax-h3-fl2va`, `minimax-h3-ref2va` | T2VA presents a single checkpoint EOS token. Keyframe and ordered-reference workflows remove caption text from their respective templates. These are numeric augmentations. | Keyframes, labels, reference video/audio prefixes, ordering and the generated AV region. |
| `bagel` | Independently eligible caption, ViT and VAE reference segments. A reference can retain either visual representation by itself. | Target generation segments and packed row membership. Existing FSDP requirements on reference counts still apply. |
| `sensenova` | T2I can omit the instruction. Editing can keep only the ordered images or omit both images and instruction, using the corresponding native context. | Supervised target image and its target slot. |

**Numeric and native give different editing recipes.** On the qualified SenseNova 8B I2I recipes, `native` gives 80% full conditioning, 10% images only and 10% neither. Numeric `0.1` gives 90%, 5% and 5%. On BAGEL, the published recipe uses independent drop probabilities of 0.1 per eligible caption and 0.3 per eligible ViT or VAE reference segment. Numeric mode first chooses whether to drop anything, then samples among the nonempty masks in the recipe's proportions.

Native recipes are qualified for `ByteDance-Seed/BAGEL-7B-MoT`, `sensenova/SenseNova-U1-8B-MoT` and `sensenova/SenseNova-U1.5-8B-MoT-SFT`. Other built-in checkpoints can use a numeric probability. A third-party adapter can set `train.condition_dropout: 0` until it declares a policy. The [P23 source table](../../roadmap/milestones/v0.6x-sft.md#p23b) links the training recipes and inference implementations behind these choices.

### Cache, resume and diagnostics

With `data.enable_preprocess: true`, positive inputs stay in the usual input cache. For independent text encoders such as Klein, one deterministic absent-text entry can serve different references within a source. The cached reference tokens stay on their positive rows. For joint encoders such as Qwen Edit, the alternate includes the original references, order and template. Packed models assemble the selected prefix during training. Set `data.force_reprocess: false` to reuse existing compatible caches.

With `data.enable_preprocess: false`, workers load condition media and targets. The training rank encodes only the selected condition with frozen preprocessing components. Offline DPO supports the same cached and on-demand paths. Both preference arms share one complete input condition. Condition dropout applies only to SFT.

A fixed training seed, data epoch and record identity reproduce each draw across workers and model-parallel peers. Changing the probability or policy revision changes exact-resume identity. Metrics under `condition_dropout/prompt/` and `condition_dropout/reference/` report observed records, eligible records, dropped records and segment counts per optimizer window. Divide `dropped` by `eligible` to inspect the realized rate. Evaluation uses its configured conditions.
