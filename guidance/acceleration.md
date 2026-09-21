# Acceleration

Flow-Factory's acceleration layer is a **model-agnostic, registry-based plugin system**
that speeds up training without touching trainer or model math. It lives in
`src/flow_factory/acceleration/` and mirrors the reward/model/trainer registries.

The dominant cost of online RL fine-tuning is **Stage 3 — rollout** (multi-step denoising
under `torch.no_grad()`), so that is where most accelerators apply.

## Safety model (read this first)

The correctness axis is **symmetric application** (`stage`), not numerical bit-exactness.
Every accelerator declares two markers, and a validator (`acceleration/validator.py`)
enforces them against the trainer's `paradigm` before training starts (fail-fast):

| marker | values | meaning |
|--------|--------|---------|
| `stage` | `both` / `rollout` | `both`: persistent transform applied to both rollout `inference()` and training `forward()` (the `shared` slot). `rollout`: per-epoch context, torn down before training (the `rollout` slot). |
| `safety` | `lossless` / `lossy` | Numerical consistency class. `lossless` = bit-identical; `lossy` = not bit-identical. It gates rollout-only accelerators and warns for lossy `stage='both'` accelerators. |

The single restriction: a **`lossy` rollout** accelerator is allowed **only on
`decoupled` / `distillation`** trainers. Why: for **coupled** algorithms (GRPO,
GRPO-Guard, DPPO) the rollout's per-step log-prob becomes the PPO "old log-prob";
changing the rollout while the training forward stays exact biases the importance ratio
and silently corrupts gradients (`.agents/knowledge/constraints.md` #7). For
**decoupled** (NFT, AWM, DGPO, DPO, CRD) and **distillation** (diffusion-opd), the rollout
log-prob does not enter the loss, so a lossy rollout only shifts the generated-sample
distribution — acceptable and tunable. Monitor the reward mean/std when enabling it.

`stage='both'` means the transform persists on the module used by both rollout and
training; it does not imply numerical exactness. The separate `safety` marker records
measured cross-stage divergence. A lossy shared accelerator such as `torch_compile` is
allowed on coupled trainers when its residual stays within `clip_range`, but the validator
warns because the ratio is approximately 1 rather than bit-exact.

## Configuration

Add an optional `acceleration:` block to any config. Two independent slots, each an
**ordered list** of `{name, params}` entries (both empty by default). **List order is the
application order**:

```yaml
acceleration:
  # Persistent stage='both' accelerators, applied to rollout and training in list order.
  shared:
    - name: attention_backend                # set the diffusers backend first...
      params: { backend: _flash_3_hub }
    - name: torch_compile                    # ...so the compiled graph captures it
      params: { mode: auto }                 # auto (default) | regional | full

  # Rollout-only (Stage 3), nested in list order. May be lossy (paradigm-gated).
  rollout:
    - name: diffusers_cache                   # diffusers_cache | <lossless id>
      params: { policy: first_block, threshold: 0.08 }
```

Either slot may be omitted or left empty. A single entry dict (without the list dashes) is
accepted as shorthand for a one-element list. A direct python path (e.g.
`my_pkg.accel.MyAccelerator`) is accepted in place of a registered id.

## Available accelerators

| id | safety | stage | Notes |
|----|--------|-------|-------|
| `attention_backend` | lossless | both | Sets the diffusers attention backend on every transformer. Requires a `backend` param. Forwards any backend (`native` / `flash` / `_flash_3` / `_flash_3_hub` / `sage` / `xformers`) to `set_attention_backend`. List it in `shared` **before** `torch_compile` so the compiled graph captures the backend. |
| `torch_compile` | lossy | both | `torch.compile` of the shared transformer. `mode: auto` (default) selects regional compilation when the base transformer declares `_repeated_blocks`, otherwise full compilation. Explicit `regional` forces diffusers' `compile_repeated_blocks`; explicit `full` compiles the whole module. Extra `compile_kwargs` are forwarded to the selected compile call. Compiles in place (checkpoint- and EMA/ref-safe), applied after `post_init`. Marked **lossy** because it is applied symmetrically but is **not bit-exact across rollout vs training** (grad/no-grad graph split → intermittent ~1e-5 on-policy residual, within `clip_range`); allowed on coupled algos, but the validator warns. |
| `diffusers_cache` | lossy | rollout | Diffusers-native feature caching (no extra dependency). Requires an adapter with `supports_diffusers_cache = True`; an adapter may further restrict the accepted policies with `supported_diffusers_cache_policies`. `policy`: `first_block` (default) / `faster` / `pyramid` / `taylorseer` / `magcache`; remaining params are forwarded to the policy config. |

Model-native exact caches are not acceleration plugins. For example, Qwen-Image 2.1 owns a
single-stream block-causal prefix KV cache inside its adapter. Rollout reuses detached prefix K/V,
while gradient replay rebuilds the same cache from the current trainable parameters. It therefore
keeps `supports_diffusers_cache = False`: that flag describes the lossy rollout-only feature-cache
plugin above, not an adapter's symmetric train/inference mechanism.

### Attention backend

Attention-backend selection is a `shared` accelerator (it transforms the module shared by
rollout and training, so it is consistent for any algorithm — even an approximate kernel
like Sage int8). It used to live under the dedicated `model.attn_backend` knob; that knob
was **removed** and folded into the acceleration layer:

```yaml
acceleration:
  shared:
    - name: attention_backend
      params: { backend: _flash_3_hub }   # native | flash | flash_hub | _flash_3 | _flash_3_hub | sage | xformers
    - name: torch_compile                 # optional; if present, list it AFTER attention_backend
      params: { mode: auto }              # auto (default) | regional | full
```

It is applied through `AttentionBackendAccelerator` by the trainer
(`BaseTrainer._apply_shared_acceleration`) — after `accelerator.prepare` / `post_init`, in
list order. Place it **before** `torch_compile` so the compiled graph captures the chosen
backend. This is the single code path for backend selection (the old
`BaseAdapter._set_attention_backend` and `model.attn_backend` were both removed). A config
that still sets `model.attn_backend` fails fast with a migration error.

> **Bagel** forces `flash_attention_2` at model load (requires `pip install -e ".[bagel]"`)
> and its custom transformer has no `set_attention_backend`, so it does **not** take an
> `attention_backend` entry — omit it (the accelerator raises if applied to bagel).

### torch.compile modes

`mode: auto` is the default and resolves each unwrapped base transformer independently.
Models with a non-empty `_repeated_blocks` declaration use regional compilation; models
without one use full compilation. This lets multi-transformer adapters mix strategies when
their components expose different capabilities. The selected strategy is logged per component.
With the current diffusers model declarations, FLUX/FLUX.2, Qwen-Image, Z-Image, Wan, and
LTX2 select regional; SD3.5 and the custom Bagel transformer select full.

Use explicit `regional` only when you want to require `compile_repeated_blocks`; it fails fast
when `_repeated_blocks` is missing or empty. Regional compilation usually has much lower
cold-start cost and handles changing image/sequence lengths more robustly. Explicit `full`
always compiles the whole transformer and is the compatibility override for models such as
SD3.5, whose diffusers transformer currently does not declare `_repeated_blocks`.

### torch.compile train-inference consistency (coupled algorithms)

For coupled algorithms (GRPO / GRPO-Guard / DPPO), the first-inner-step PPO ratio must remain
**numerically on-policy around 1 and within `clip_range`**. `torch.compile` (Inductor)
threatens this because it compiles a
**separate, numerically non-identical graph for grad vs no-grad mode** (Dynamo guards on
`grad_mode`): rollout normally runs the transformer under `torch.no_grad()` and the training
forward under grad, so a naive compiled rollout would diverge from training and bias the ratio.

`CompileAccelerator` handles this automatically (no user action needed): rollout remains
under an outer `torch.no_grad()`, while the wrapper runs only the compiled transformer under
`torch.enable_grad()`. Scheduler/CFG operations therefore do not extend the graph, and
trajectory/callback collectors detach rollout-old tensors before storage. The accelerator
also compiles the **base** transformer under any PEFT/LoRA wrapper (not the wrapper itself).
With this, the
on-policy ratio is driven to **≈1, well within `clip_range` (1e-4)** — but **not strictly
bit-exact**: an intermittent **~1e-5** residual remains on a minority of samples/ranks.
Measurements covered 16×H20 with ZeRO-2/FSDP2, SD3.5 full compilation, Qwen-Image
regional/full compilation, and CFG/no-CFG. Forcing grad removes the *dominant*
grad-vs-no-grad graph split, but rollout and training are still distinct Inductor kernel
invocations (different latent stride/contiguity + autograd-graph context), and bf16's
non-associative accumulation surfaces a last-bit difference for some inputs. So compile is
**numerically on-policy, not bit-exact** — if you need a strictly bit-exact ratio, use eager or
the `attention_backend` accelerator (a backend's forward is grad-mode-independent and stays
exactly 0). See `CompileAccelerator._wrap_forward_grad_consistent` for details.

Two implementation notes that matter for correctness:
- The grad-force wrapper must **not** detach its own output — an inner detach lets Inductor
  pick a divergent inference-optimized kernel and re-introduces ~1e-5 drift. The outer
  no-grad boundary stops graph propagation; collectors detach direct outputs before storage.
- Determinism knobs (`cudnn.deterministic`, `fallback_random`) are irrelevant here — the cause
  was the grad-vs-no-grad graph split, not RNG (a `train-vs-train` recompute is exactly 0.0).

See `.scratch/torch_compile_consistency_report.md` for the full analysis.


## Model cache-readiness (lossy caching)

Feature caching reuses block outputs across denoising steps via the transformer's
`cache_context(...)`. Readiness is explicit: an adapter opts in with
`supports_diffusers_cache = True` only when every transformer forward branch has a context.
Models with a narrower upstream surface also declare
`supported_diffusers_cache_policies = frozenset({...})`; `None` preserves the existing
all-policy behavior for cache-ready adapters. The cache accelerator validates the policy and its
config before preparing or enabling any component. An adapter that needs upstream compatibility
may override `prepare_diffusers_cache(policy, component_name, transformer)` with an idempotent
shim; it must not enable the cache itself.

Cache-ready adapters are FLUX.2-Klein, Qwen-Image, Qwen-Image-Edit-Plus, Wan T2V/I2V, and
LTX2 T2AV/I2AV. MiniMax H3 T2VA/FL2VA/Ref2VA are cache-ready for `first_block` only.
Qwen merged CFG uses a shared `cond_uncond` context; no-CFG uses `cond`. FLUX.1/Kontext,
FLUX.2, SD3.5, Z-Image, Bagel, and Qwen-Image 2.1 are not feature-cache-ready. Qwen-Image 2.1
instead enables its exact native KV cache by default. Validate the reward distribution before and
after enabling lossy feature caching on a supported model.

MiniMax H3 support bridges a missing diffusers 0.40.0 transformer-block registry entry with the
FirstBlockCache metadata required by the main block stack. It registers the actual runtime block
class, including an FSDP2-generated subclass, before enabling the cache. Each independent B=1
inference resets state once; every denoising forward reuses a workflow-specific context while the
call itself continues through the prepared component route (`transformer_ref` for Ref2VA).
Invalid policies and configurations fail before cache enablement. H3 FirstBlockCache cannot be
combined with Flow-Factory's `torch_compile`: its forced-grad rollout path would make diffusers
0.40.0 retain cross-step autograd graphs, so that combination fails fast. This path has exact
diffusers-0.40.0 CPU lifecycle coverage, including a real cache hit and reset. Because
FirstBlockCache is lossy and its threshold is workload-dependent, tune `threshold` and recheck
reward/quality on the target prompt distribution and hardware.

As a lossy rollout accelerator, H3 caching remains restricted to decoupled or distillation
trainers such as TDM. Coupled GRPO/GRPO-Guard/DPPO configurations are rejected by the paradigm
validator.

`torch_compile` is model-agnostic and applies to every adapter.

## Multi-role backend contract

DMD2, TDM, and TDM-R1 keep one prepared model root and one optimizer root.
Roles update in exclusive sequential phases (`fake` × `R`, optional
`surrogate`, then `generator`). DDP, FSDP2, and DeepSpeed ZeRO-1/2 support that
program. ZeRO-3 remains globally unsupported.

The CPU suite validates chronology and role-local optimizer state. It does not
validate CUDA collectives or sharded parameter swaps; no GPU pass is implied
by CPU results.

## Adding a new accelerator

1. Subclass `acceleration/abc.py::BaseAccelerator`; set the `safety` and `stage` class
   attributes. Implement `setup()` (one-time mutation, e.g. compile) and/or
   `rollout_context()` (per-epoch context, e.g. caching).
2. Register the id → class path in `acceleration/registry.py`.
3. Optional dependencies must be imported defensively (`try/except ImportError`) per
   constraint #22(a).
