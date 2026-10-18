# Changelog

## Unreleased — `fix/config-validation-and-empty-shards`

- `scheduler.sde_steps` and `num_sde_steps` are validated against `train.num_inference_steps`
  when the config loads. An index the rollout never reaches previously failed only at replay.
- `eval.num_frames` and `eval.frame_rate` default to the training values. An unset value previously
  reached test-split preprocessing and evaluation as `None`, breaking every video adapter; the
  MiniMax-H3 examples also set them explicitly.
- Distributed preprocessing of a dataset with fewer rows than ranks writes empty Arrow parts for
  row-less ranks instead of failing consolidation.

### Behavior changes

- Configs with out-of-range, repeated, or non-integer `sde_steps`, or `num_sde_steps` larger than
  the list, now raise at load time.

---

## Unreleased — `fix/media-uint8-range-detection`

- Sample media follow one explicit pixel contract: `BaseSample` canonicalizes `image`, `video`,
  `condition_images`, and `condition_videos` to floating `[0, 1]` tensors, and pixel conversions
  take their range from dtype (integer is bytes, floating is `[0, 1]`) or an explicit `PixelRange`
  instead of min/max detection. One pixel drifting just outside `[0, 1]` no longer turns reward or
  logged media near-black or washed out, and near-black `uint8` input is no longer rescaled.
- Wan2 I2V and LTX2 I2AV cache `[0, 1]` condition pixels and derive the `[-1, 1]` VAE input
  internally, bit-identical to Diffusers preprocessing. LTX2 I2AV condition images reach rewards,
  logs, and sample identity with their real colors. Both preprocess caches rebuild once.

### Behavior changes

- Floating sample media more than 1/64 outside `[0, 1]` raise `ValueError` naming the sample field;
  smaller edge drift is clamped.
- `normalize_to_uint8` / `normalize_video_to_uint8` and the PIL/NumPy conversions built on them
  read floating input as `[0, 1]`; pass `value_range=PixelRange.SIGNED_UNIT` or `PixelRange.BYTE`
  for other intervals. Non-finite pixels and integer pixels outside `[0, 255]` raise.
- `hash_tensor` quantizes floats at byte resolution instead of detecting their range.

---

## Unreleased — SenseNova-U1

- Added SenseNova-U1 1.0/1.5 T2I and ordered variable-count multi-reference
  I2I support through the unified NEO-Unify adapter.
- Added GRPO recipes for OCR T2I and PickScore multi-reference I2I.
- See [PR #217](https://github.com/X-GenGroup/Flow-Factory/pull/217).

---

## Unreleased — `feature/structured-trajectory-runtime`

- Added structured multi-component trajectories, ordered heterogeneous references,
  component variants, named snapshots, and role-aware checkpoints.
- Added explicit per-variant optimizer configuration, including AdamW and Muon.
- Split shared trainer, coupled replay, decoupled replay, and multi-role backend
  responsibilities into focused runtime modules.
- Added packed distributed metric/sample collectives with benchmark-gated fallbacks.
- Added scalar or selector-based `frozen_parameters_dtype` policies.
- Added adapter defaults and scalar/selector overrides for load-time component dtypes.
- Added physical-root load planning and backend-owned target/auxiliary/reward lifecycles.

### Breaking changes

- Multi-role configurations use top-level `optimizers:` entries keyed by variant name.
- ZeRO-3 is unsupported; use DDP, ZeRO-1/2, FSDP1, or FSDP2 as documented.
- Structured trajectories replace single-tensor trajectory assumptions in adapter hooks.

---

## Unreleased — `feature/opd-loss-targets`

- Added `train.loss_target` with `xt`, `v`, and `x0` target spaces for
  DiffusionOPD. The `v` and `x0` targets are restricted to ODE dynamics;
  `xt` continues to support ODE and SDE.
- Added independent `train.self_normalize`. When enabled, each sample's
  target-space MSE is divided by its detached mean absolute student-teacher
  error plus `1e-8`, enabling `xt_norm`, `v_norm`, and `x0_norm`.
- Generalized the teacher cache from transition means to projected teacher
  targets. No adapter, scheduler-output, or base-trainer interface changed.

### Breaking changes

- Removed the historical `0.5` multiplier from the DiffusionOPD objective.
  Existing YAML remains valid and defaults to `loss_target: xt` with
  `self_normalize: false`, but its loss and gradients are now 2x their
  previous values.
- Renamed DiffusionOPD training metrics from `train/kl_div` and
  `train/kl_div_<teacher_name>` to `train/distill_loss` and
  `train/distill_loss_<teacher_name>`, because the configurable targets and
  objective scale are not generally a Gaussian KL.

---

## Unreleased — `feat/diffusion-opd` (merge to `main`)

**Full range:** `652f7315..e4496c6` (2026-05-26 … 2026-06-02)
**Repo:** [X-GenGroup/Flow-Factory](https://github.com/X-GenGroup/Flow-Factory)
**Branch:** `feat/diffusion-opd`

This branch stacks two features onto `main` (at `652f7315`):

| Part | Range on branch | PR / feature |
|------|-----------------|--------------|
| A — Multi-dataset training & eval | `652f7315..02c581a` | [#168](https://github.com/X-GenGroup/Flow-Factory/pull/168) (squash commit `02c581a`) |
| B — DiffusionOPD | `02c581a..e4496c6` | On-policy multi-teacher distillation (this branch) |

Part B requires Part A (`data.datasets`, multi-source train loaders, per-dataset eval).

---

## Part A — Multi-dataset training & eval (#168)

**Range:** `652f7315..02c581a` (2026-05-26 … 2026-05-31)
**Squash commit on branch:** `02c581a`
**PR:** [#168](https://github.com/X-GenGroup/Flow-Factory/pull/168)

Unified configuration and runtime support for **multiple named datasets** in one
training run: weighted multi-source training, per-dataset eval with independent
reward routing, and source-aware reward gating with NaN-padded cross-rank transport.

> The commit table below documents **PR #168 development history** (individual
> commits before squash). On `feat/diffusion-opd` the entire feature lands as a
> single squash commit `02c581a`. A post-merge review round (still within `02c581a`)
> removed legacy `data.dataset_dir` canonicalization, renamed reward routing fields,
> and migrated all example YAMLs — see [Post-review hardening](#post-review-hardening-after-15c3943).

### Commit ranges by phase (PR #168 development)

| Phase | Range | Date | What |
|-------|-------|------|------|
| Pre-existing multi-eval foundation | `8e573cc..33d678b` | 2026-05-26 | Initial multi-eval + per-dataset reward routing + per-dataset eval overrides |
| Plan steps 1–12 (multi-training-dataset infra) | `f2b2100..cf3a3f7` | 2026-05-30 | Unified `data.datasets` schema, exact partitioning, source-aware gate, NaN-pad transport, `multi_source.yaml` |
| Review items 1–9 | `56c1b67..3d1d098` | 2026-05-30 | Naming, typed `source`/`source_id`, integer weights, eager reward resolution, cleanup |
| Eval-path unification | `5a00d9b..15c3943` | 2026-05-30 | Single `evaluate()`; `get_dataloader` → `get_train_dataloader` + `get_eval_dataloaders` |

### Commit list (PR #168 development)

| Hash | Date | Subject |
|------|------|---------|
| `15c3943` | 2026-05-30 | `[review,merge-6]` CHANGELOG.md: document eval-merge breaking changes |
| `9e38246` | 2026-05-30 | `[review,merge-5]` drop `self.test_dataloader` and `self.eval_reward_buffer` |
| `b4717f4` | 2026-05-30 | `[review,merge-4]` delete `_evaluate_single_dataset`; unify into `evaluate()` |
| `b360984` | 2026-05-30 | `[review,merge-3]` route legacy test through `get_eval_dataloaders`; drop `_build_legacy_test_dataloader` |
| `5a00d9b` | 2026-05-30 | `[review,merge-1]` rename: `get_dataloader` → `get_train_dataloader`; train-only return |
| `3d1d098` | 2026-05-30 | `[review,item-8]` cleanups: explicit batch_size, deepcopy shim, inverse routing check |
| `94f28ba` | 2026-05-30 | `[review,items-2+3]` feat: per-source spec fields + unified single/multi path |
| `933beae` | 2026-05-30 | `[review,item-7]` feat: integer weights + exact-divisibility partition |
| `0a612bb` | 2026-05-30 | `[review,item-6]` feat: promote `source`/`source_id` to first-class `BaseSample` fields |
| `6be080f` | 2026-05-30 | `[review,item-9]` feat: eager `RewardArguments.applicable_datasets` resolution |
| `fd2e2bd` | 2026-05-30 | `[review,item-4]` reorder: training-before-eval parameters and accessors |
| `521a9bb` | 2026-05-30 | `[review,item-1]` rename: `dl` → `loader` |
| `56c1b67` | 2026-05-30 | `[review,item-5]` cleanup: drop `object.__getattribute__` paranoia in gate/aggregation |
| `cf3a3f7` | 2026-05-30 | `[examples]` feat: `multi_source.yaml` smoke config |
| `610a80f` | 2026-05-30 | `[trainers]` feat: `BaseTrainer` `__source__` injection + `set_epoch` propagation |
| `da62f47` | 2026-05-30 | `[advantage]` feat: applicability-aware aggregation |
| `993d144` | 2026-05-30 | `[rewards,samples]` feat: source-aware reward gate with NaN-padded transport |
| `de40eec` | 2026-05-30 | `[rewards,trainers]` feat: `MultiRewardLoader` `training_dataset_names` plumbing |
| `b3e74c7` | 2026-05-30 | `[data_utils,trainers]` feat: multi-source train dataloader infra |
| `4662279` | 2026-05-30 | `[data_utils]` feat: `get_data_sampler` `unique_sample_num` override (later removed) |
| `e072d1c` | 2026-05-30 | `[hparams]` feat: shared `_align_unique_sample_num` + multi-source partition |
| `33a9349` | 2026-05-30 | `[hparams,trainers,data_utils]` feat: switch eval consumers to `data.datasets` |
| `5f84e05` | 2026-05-30 | `[hparams]` feat: top-level `eval_datasets` deprecation shim |
| `c01909e` | 2026-05-30 | `[hparams]` feat: `data.datasets` field + per-split properties + validators |
| `f2b2100` | 2026-05-30 | `[hparams]` feat: unified `DatasetArguments` schema |
| `33d678b` | 2026-05-26 | refactor: simplify multi-eval implementation after code review |
| `cb3c434` | 2026-05-26 | feat: add per-dataset eval generation overrides |
| `8e573cc` | 2026-05-26 | feat: support multiple eval datasets with per-dataset reward routing |
| `02c581a` | 2026-05-31 | **Squash merge** of #168 onto branch (includes post-review hardening) |

### Breaking changes (#168)

#### Eval metric key rename

All eval flows through the unified per-dataset `evaluate()`. Configs declare
datasets via `data.datasets` (each entry has a `name`, e.g. `default`); eval
metric keys are namespaced by that name.

Keys move from:

```
eval/reward_<name>_mean
eval/reward_<name>_std
eval_samples
```

to:

```
eval/<dataset_name>/reward_<name>_mean
eval/<dataset_name>/reward_<name>_std
eval/<dataset_name>/samples
```

W&B / TensorBoard dashboards must update (e.g. `eval/reward_` →
`eval/default/reward_`; `eval_samples` → `eval/default/samples`).

Landed in: `b4717f4` (eval merge).

#### Eval cache one-time reprocess

The unified eval path adds an `eval_<name>` token to the preprocessing-cache
fingerprint. Existing `~/.cache/flow_factory/datasets/...` entries from the old
path do not match, so the test split is reprocessed once on the next run.
Training caches are unaffected.

Landed in: `b360984` (eval merge).

#### Removed `BaseTrainer` attributes

Used only by the deleted legacy single-eval path:

- `self.test_dataloader` → `self.eval_dataloaders` (`Dict[str, DataLoader]`).
- `self.eval_reward_buffer` → `self.eval_dataset_reward_buffers[name]`.
- `self.eval_reward_processor` → `self.eval_dataset_reward_processors[name]`.
- `_evaluate_single_dataset` / `_evaluate_multi_dataset` removed; `evaluate()`
  is the single entry point.

Landed in: `9e38246` (eval merge).

#### Renamed `data_utils.get_dataloader` → `get_train_dataloader`

Returns `(train_loader, train_loaders_by_source)`; eval is owned by
`get_eval_dataloaders`.

```python
# Before
from flow_factory.data_utils.loader import get_dataloader
train, test, by_source = get_dataloader(config, accelerator, ...)

# After
from flow_factory.data_utils.loader import get_train_dataloader, get_eval_dataloaders
train, by_source = get_train_dataloader(config, accelerator, ...)
eval_dict = get_eval_dataloaders(config.data_args.eval_datasets, config, accelerator, ...)
```

Landed in: `5a00d9b` (eval merge).

#### Top-level `eval_datasets:` YAML key deprecated

YAMLs using the brief-lived top-level `eval_datasets:` field are auto-migrated
to `data.datasets[*].eval` with a `DeprecationWarning` per config load. Scheduled
for removal one release after #168 ships.

Landed in: `5f84e05` (plan step 3); migration shim: `_migrate_legacy_eval_datasets`.

#### `RewardArguments.applicable_datasets` semantic change

The routing field (named `datasets` when first introduced, renamed to
`applicable_datasets` in post-review hardening) is eagerly resolved at config
load: `None` becomes the explicit list of applicable dataset names. Empty list
`[]` is honored as "never fires" with a warning.

Landed in: `6be080f` (review item 9); rename in post-review hardening.

#### Integer `train.weight` required

`DatasetTrainSpec.weight` must be a positive integer (integer-valued floats like
`1.0` are coerced; non-integer floats raise). With `num_batches_per_epoch %
sum(weights) == 0`, every batch is guaranteed to come from a single source.

Landed in: `933beae` (review item 7).

#### Legacy `data.dataset_dir` rejected (post-review)

After post-review hardening, **bare `data.dataset_dir` without `data.datasets` is
rejected** (mutual exclusion when both are set). All example YAMLs use
`data.datasets` with per-entry `dataset_dir`. The earlier
`_canonicalize_legacy_dataset_dir` shim was removed.

### Added (#168)

- **Unified `data.datasets:` schema** — each entry has `name`, `dataset_dir`,
  optional media roots, and optional `train:` / `eval:` sub-blocks.
- **Multi-source training** — integer `train.weight`, exact batch partitioning,
  `WeightedSourceBatchScheduler`, per-source DataLoaders, homogeneous batches.
- **Per-dataset eval** — `get_eval_dataloaders`, `DatasetEvalSpec` overrides
  (`resolution`, `num_inference_steps`, `guidance_scale`), namespaced metrics.
- **Source-aware reward routing** — `RewardArguments.applicable_datasets` on
  training and eval rewards; `_datasets_resolved: frozenset[int]` for hot-path gate.
- **Sample bookkeeping** — `BaseSample.source` / `BaseSample.source_id`;
  `MultiSourceTrainDataLoader` injects `__source__` / `__source_id__`.
- **NaN-padded reward transport** — full-length tensors with NaN at non-applicable
  positions for deadlock-free `accelerator.gather`.
- **Applicability-aware aggregation** — `AdvantageProcessor` uses
  `sample.applicable_rewards`; `_validate_every_source_has_a_reward` at config load.
- **`train_dataloaders_by_source`** — exposed on every trainer (consumed by
  DiffusionOPD for per-teacher rollout routing).

### Notable internal refactors (#168)

- Single `evaluate()` implementation; legacy eval helpers deleted (`b4717f4`).
- `get_train_dataloader` + `get_eval_dataloaders`; legacy test loader deleted
  (`b360984`).
- Config validation in `Arguments.__post_init__` (`_validate_dataset_routing` +
  resolvers).
- **Latent bug fix:** `_evaluate_multi_dataset` called `get_merged_eval_kwargs` on
  the parent `DatasetArguments` instead of `DatasetEvalSpec`; fixed in eval merge
  (`b4717f4`). (A related eval-preprocess gap — per-dataset `guidance_scale` not
  merged at dataset-cache time — was fixed on the OPD branch; see Part B.)

### Post-review hardening (after `15c3943`, included in squash `02c581a`)

- **`_canonicalize_legacy_dataset_dir` removed** — bare `data.dataset_dir` rejected;
  all example YAMLs migrated to `data.datasets`.
- **`RewardArguments.datasets` → `applicable_datasets`** — renamed across gate,
  loader, and advantage routing.
- **Per-dataset reward weights** — `RewardArguments.weight` as scalar or
  `{dataset: weight}` dict, expanded at config load.
- **Metadata transport** — per-sample JSONL metadata as `sample.metadata` (JSON
  string); GenEval `required_fields` updated to `("image", "prompt", "metadata")`.
- **Communication optimizations** — merged advantage-stage gathers; packed M
  groupwise reductions; vectorized group normalization.
- **Dead code removed** — `eval_dataset_args.py`, `_partition_unique_sample_num`,
  `_per_source_unique_sample_num`, `_encode_prompts`.
- **Eval-only guard** — `generate_samples()` raises if no training dataloader.

---

## Part B — DiffusionOPD

**Range:** `02c581a..e4496c6` (2026-05-31 … 2026-06-02)
**Builds on:** Part A (#168)
**Paper:** [On-Policy Distillation of Diffusion Models](https://arxiv.org/abs/2605.15055)

Multi-teacher on-policy distillation: each **training dataset** is distilled by
**exactly one** LoRA teacher; a single teacher may cover **multiple** datasets
(teachers must not overlap on the same dataset). Distillation runs along the
student's rollout trajectories via per-step mean-matching KL, supporting **ODE
and SDE** dynamics.

### Commit list (Part B)

| Hash | Date | Subject |
|------|------|---------|
| `aa9b639` | 2026-05-31 | `[trainer,hparams,scheduler]` feat: add DiffusionOPD on-policy distillation trainer |
| `34a043f` | 2026-05-31 | `[examples]` feat: add DiffusionOPD SD3.5 example configs |
| `51ec3c4` | 2026-06-01 | Update teacher path to HF |
| `fc97e83` | 2026-06-01 | `[examples,docs]` fix: point teachers at `quanhaol/DiffusionOPD` subfolders |
| `0fbbc59` | 2026-06-01 | `[trainer,hparams,examples]` fix: distillation steps via `timestep_range` |
| `a6f8dab` | 2026-06-01 | `[trainer,hparams,examples]` feat: per-teacher KL logging + gradient accumulation fix |
| `33519e6` | 2026-06-01 | `[hparams]` fix: gate `_validate_teacher_sources` on `isinstance` (PR #170) |
| `425b443` | 2026-06-01 | Refactor `data_utils` |
| `e7544fc` | 2026-06-01 | Fix eval preprocessing kwargs (per-dataset `guidance_scale`) |
| `28afc90` | 2026-06-01 | `[data,hparams]` chore: black/isort on new OPD files |
| `ed91b21` | 2026-06-02 | Update readme |
| `e4496c6` | 2026-06-02 | Teacher coverage validation in `Arguments._validate_teacher_sources` |

### Added (Part B)

- **`trainer_type: diffusion-opd`** — `DiffusionOPDTrainer`
  (`trainers/distillation/opd/trainer.py`), registered in `trainers/registry.py`. Two-pass
  `optimize()`: PASS 1 (`no_grad`) caches each teacher's `mu_T` with one weight
  swap per teacher; PASS 2 student-only gradient loop matching `mu_S` to cached
  `mu_T`. Rewards used only for eval monitoring. Logs `train/kl_div_{teacher_name}`
  and overall `train/kl_div`.
- **`DiffusionOPDTrainingArguments` + `TeacherConfig`**
  (`hparams/training_args/opd.py`): `teachers` (`path`, `name`,
  `applicable_datasets`, `guidance_scale`), `teacher_param_device`, `timestep_range`.
  Overrides `get_preprocess_guidance_scale()` (max of student and per-teacher CFG)
  and `get_num_train_timesteps()` (one backward per distilled step). Routing:
  one teacher may list several datasets; `DiffusionOPDTrainer` rejects overlap.
- **`SDESchedulerMixin.get_kl_divergence_denominator(std_dev_t, dt)`**
  (`scheduler/abc.py`): dynamics-agnostic transition variance for the KL denominator.
- **`trainers/distillation/opd/common.py`** — `load_teachers`: named-parameter snapshots for
  teacher LoRAs (architecture must match student LoRA slot).
- **`Arguments._validate_teacher_sources`** (`hparams/args.py`, OPD-only):
  (1) every teacher `applicable_datasets` entry is a declared training dataset;
  (2) every active training dataset appears in some teacher's `applicable_datasets`.
  Overlap enforcement stays in `DiffusionOPDTrainer`.
- **Examples** — `examples/opd/lora/sd3_5/DiffusionOPD_aligned.yaml` (HF teachers,
  upstream `mopd` recipe) and `geneval_pickscore_ocr.yaml` (GenEval/PickScore/OCR
  with eval-only `*_no-cfg` ablation tracks).

### Fixed (Part B)

- **Eval CFG preprocessing** (`data_utils/loader.py`): `get_eval_dataloaders`
  merges per-dataset eval overrides via `DatasetEvalSpec.get_merged_eval_kwargs`,
  matching `BaseTrainer.evaluate`. Fixes silent CFG disable when per-dataset
  `guidance_scale > 1.0` but shared `eval.guidance_scale` was used at preprocess time.
- **Distillation step selection** — steps from `train.timestep_range`, not
  SDE-only `scheduler.train_timesteps` (empty under ODE). `resolve_distill_step_band()`
  shared by `sample()`, `optimize()`, and `get_num_train_timesteps()`.
- **Gradient accumulation** — `get_num_train_timesteps()` returns distilled step
  count so `gradient_step_per_epoch` math is correct.
- **Teacher validation gate** — `_validate_teacher_sources` gated on
  `isinstance(DiffusionOPDTrainingArguments)`.

### Docs (Part B)

- `guidance/algorithms.md`: "DiffusionOPD: On-Policy Distillation" + reference [14].
- `README.md` / `examples/README.md`: `diffusion-opd` trainer and `opd` examples.
