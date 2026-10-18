# Component Variants

**Read when**: Adding an algorithm that trains more than one copy of the model at once
(distillation, adversarial critics), or changing parameter ownership, per-variant LoRA/full
storage, variant-scoped checkpointing, or the multi-role optimizer contract.

## The layering rule

`BaseAdapter` is infrastructure. It supplies named mechanisms and holds no opinion about what an
algorithm calls things or why. The algorithm's trainer owns the vocabulary.

| Layer | Speaks | Owns |
|---|---|---|
| `models/` | component variants | the mechanism: create a copy, activate one, collect its parameters, save a scope |
| `trainers/` | roles | the meaning: which variants exist, what they are called, which is trained when, which one ships |

Concretely: the model layer never contains the words *generator*, *fake* or *surrogate*. A DMD
trainer names its own variants and keeps a one-line `_add_generator()` if it wants one. Two
distillation trainers duplicating that line is fine and preferred; a shared abstraction that
teaches the adapter what a generator is, is not.

This mirrors how other frameworks draw the line: verl's `Role` enum lives in the PPO trainer, not
in `FSDPWorker`; diffusers exposes `set_adapters(names)` with no privileged adapter name; TRL's
`DPOTrainer` owns `ref_model` rather than the model class owning it.

## Two mechanisms, and how to choose

Both live on `BaseAdapter`. They answer different questions.

**Named parameter snapshots** are *temporal*: one set of weights installed at a time.

```python
adapter.add_named_parameters("old_policy")          # snapshot the live weights
with adapter.use_named_parameters("old_policy"):    # temporarily install them
    ...
adapter.update_named_parameters("old_policy", decay)  # blend toward live weights
```

Use this for anything that is *the same model at another point in time*: a reference policy, an
EMA, an old snapshot, a sampling model. CRD's old and sampling models are exactly this.

**Component variants** are *spatial*: several copies live at once, each with its own optimizer
group.

```python
adapter.declare_component_variants(("generator", "fake"))   # names are yours
with adapter.use_component_variant("fake"):
    velocity = adapter.forward_state(...)
params = adapter.variant_parameters("fake")                  # build your own param group
```

Use this only when copies must coexist and hold gradients simultaneously. That is the one thing a
trainer cannot build for itself, because it needs PEFT adapters on a shared base, one prepared
`ModelBundle` root, and forward routing.

**A variant is always a live trainable copy.** A frozen reference is not one: ask for
`use_ref_parameters()` (the pre-finetune weights) or a named snapshot instead. DMD2's reference
score goes through `use_ref_parameters()` for exactly this reason, so only `generator` and `fake`
are declared variants.

The distinction is not about the weights, which are the cheap part; it is about what has to sit
next to them. A trainable variant carries gradients and optimizer state (roughly 3x the parameters
for Adam), both of which must be co-located with the parameters during backward and step. A
snapshot carries neither, stores values only, and its shadow can live on CPU (`EMAModuleWrapper`
has a cross-device copy path). Activating a snapshot copies values into the *existing* live
parameters, so it allocates no second module and composes with whichever variant is active.

This is also why "just offload the idle variant to CPU" does not collapse the two mechanisms. For a
read-only copy it would, which is why no frozen storage mode exists. For a trainable one it does
not: you could move only the weights, the gradients and optimizer state would have to follow, and
in DMD2 the fake score is touched twice per step, so there is no idle window. Offload belongs to
the distributed backend (FSDP `cpu_offload`, ZeRO-2 offload), which reaches variants automatically
because they are bundle members. Moving a wrapped module's parameters behind FSDP's back would
break its flat-parameter invariants.

## Declaring variants

`declare_component_variants(trainable_variants)` runs before `accelerator.prepare` so the bundle
sees every member; reconfiguring an existing registry raises rather than silently rebuilding
ownership under a prepared root.

The **base variant is positional**: whichever name comes first owns the adapter's canonical
components, and every later variant is layered on it. The registry never recognises a particular
name, so an algorithm may call its base `generator` and the model layer stays ignorant of the word.

| `ComponentVariantSpec` field | Meaning |
|---|---|
| `storage_mode` | `lora` or `full` |
| `adapter_name` | PEFT adapter under `lora`; `None` under `full` |
| `component_routes` | canonical component name -> the module this variant uses |

The storage modes give the memory trade-off directly. Under `lora`, N variants cost one base plus
N adapters and the base takes the `default` adapter. Under `full`, N variants cost N copies routed
as `{variant}__{component}`. `RoutedComponentProxy` resolves a canonical name through the active
variant, so adapter code keeps saying `self.transformer`.

Two variants may share a route only under `lora` with a named adapter, because a LoRA adapter
layers on the base weights rather than copying them. A `full` variant sharing a route would
silently alias the base, so the registry rejects it.

## Checkpointing

Scope is the caller's, passed explicitly:

```python
adapter.save_checkpoint(path, variant=None)        # the base components
adapter.save_checkpoint(path, variant="generator")  # one named variant
```

There is no rule inside the adapter about which variant ships. An algorithm that exports one
variant's EMA composes it from primitives, which is the whole point of the split:

```python
tensors = adapter.get_variant_snapshot("generator_ema")
with adapter.use_variant_snapshot("generator_ema"):
    adapter.save_checkpoint(path, variant=None, model_only=True)
```

Variant metadata is written before `accelerator.save_state` and validated before
`accelerator.load_state`. Both orderings matter: accelerate mutates optimizer state in place, so a
resume that discovers a changed layout afterwards has already restored state onto the wrong groups.
The on-disk key stays `roles` because the checkpoint is the trainer's artifact; the registry's own
key is `variants`, and the two are translated at the boundary.

## Optimizer configuration

Optimizer hyperparameters live in a top-level `optimizers:` list, one entry per
trainable variant, resolved by name. `optimizer:` selects both the implementation and
the arguments subclass (`hparams/optimizer_args/_registry.py`), so AdamW and Muon
hyperparameters never share a class:

```yaml
optimizers:
  - name: generator
    optimizer: muon
    learning_rate: 2.0e-5
  - name: fake
    optimizer: adamw
    learning_rate: 1.0e-5
    update_frequency: 5
```

`update_frequency` is the single source of role cadence. `resolve_role_optimizer_args`
(`hparams/optimizer_args/_base.py`) fills an unset value from the algorithm's default
entry (`default_role_optimizers()`), `role_update_plan(update_frequencies)` on the
training arguments orders the phases and rejects repeats for roles outside
`repeatable_roles`, and `BaseTrainer._role_update_repeats` feeds the trainer loops. The
same plan is stored in multi-role checkpoint metadata, and the resolved value is part
of the exact-resume identity. `Arguments.__post_init__` runs the resolution once at
parse time, on the user's entries before the flat `train.*` fallback is synthesized,
so an invalid cadence or an entry that names no trainable role fails before any model
loads.

Omitting `optimizers:` keeps the flat `train.learning_rate` family working: one default
AdamW configuration is synthesized in `Arguments.__post_init__`, which is the single
place that knows the legacy spelling.

`torch.optim.Muon` orthogonalizes matrices and **rejects any parameter that is not
2D**, so a Muon variant is really optimized by two algorithms: Muon for its matrices
and AdamW for its biases, normalization scales and embeddings (`fallback_betas` /
`fallback_eps` configure that half). Since the framework prepares exactly one optimizer
root, `optimizer/loader.py` returns a `CompositeOptimizer` in that case, which exposes
its children's groups as one list. An all-AdamW run still gets a plain
`torch.optim.AdamW`, unchanged.

Muon therefore gives one variant **two** parameter groups, which is why
`OptimizationRole.optimizer_group_ids` is a tuple. It requires a PyTorch build that
exposes `torch.optim.Muon` (included in standard releases from PyTorch 2.9); the runtime feature
check, rather than the version string alone, is authoritative. DeepSpeed is rejected as unverified because it rebuilds
its own optimizer wrapper, while FSDP1 is rejected because its flat parameters erase
the required matrix rank; the supported distributed plans are DDP and FSDP2.

## Optimizer and backend contract

`BaseTrainer._init_prepared_role_optimization` always constructs a `RoleOptimizationCoordinator`
(`trainers/role_optimization.py`) once `prepare` has run, so `self.role_optimization` exists even
for a single-policy trainer, where it degenerates to one role. It drives disjoint role updates
through one physical optimizer; each role owns one or more `param_groups`, listed in
`OptimizationRole.optimizer_group_ids`, since a Muon role contributes two.
`_validate_multirole_backend` rejects the layouts multi-role cannot support:

- more than one prepared model root or more than one prepared optimizer;
- a prepared root or optimizer that is not the tracked `model_bundle` / `optimizer`;
- an optimizer group role mapping that changed during `prepare`;
- DeepSpeed outside ZeRO-1/2 (see [`constraints.md` #10](../constraints.md));
- FSDP2 with `use_orig_params=False`, or optimizer parameters that are not identities from the
  prepared root.

CPU tests cover phase chronology and role-local optimizer state, not CUDA collectives or sharded
parameter swaps; real-weight TDM coverage is recorded in [GPU Validation](gpu_validation.md).

The primary role, whose optimizer step paces the public `self.step`, is simply the first declared.
Single-policy runs skip all of it: one variant, one group, and the flat `train.learning_rate`
family describes it.

## Review checklist

- [ ] No algorithm word (`generator`, `fake`, `surrogate`) appears under `src/flow_factory/models/`.
- [ ] A time-shifted copy uses named parameter snapshots, not a variant.
- [ ] A frozen reference goes through `use_ref_parameters()`, never a declared variant.
- [ ] Test fakes reject undeclared variant names, so they cannot hide a production `KeyError`.
- [ ] Variants are declared once, before `accelerator.prepare`.
- [ ] `lora` variants declare an `adapter_name`; `full` variants do not.
- [ ] Forwards that depend on a variant run inside `use_component_variant`.
- [ ] Checkpoint scope is passed explicitly rather than inferred.
- [ ] Variant metadata is written before, and validated ahead of, accelerate's state mutation.

## Cross-refs

- UP: [`constraints.md` #10](../constraints.md#10-deepspeed-zero-3-is-unsupported), [`architecture.md` Component Management](../architecture.md#component-management)
- PEER: [Component Runtime](component_runtime.md), [Structured Trajectory](structured_trajectory.md), [Autocast and Parameter Swaps](autocast_param_swap.md), [GPU Validation](gpu_validation.md)
