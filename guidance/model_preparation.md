# Model preparation

`BaseTrainer` keeps the initialization sequence visible. Its `ModelPreparation` owner performs
the destructive model transforms, one backend preparation, parameter rebinding and route
installation. `ModelLoadCoordinator` and its backend runtime retain loading and wrapping policy.
Adapters retain precision, attention and activation-checkpoint capabilities.

| Order | Owner and operation |
|---|---|
| 1 | Adapter: load targets, freeze weights, inject base LoRA, apply dtype policy |
| 2 | Trainer and variant registry: materialize every role, restore role weights, align dtypes |
| 3 | Preparation: apply TP/SP to every eligible physical module once, then rebind ownership |
| 4 | Trainer: construct optimizer groups from the resulting parameters |
| 5 | Preparation: check exact optimizer membership; ask the existing loading backend to prepare one `ModelBundle`; rebind registry parameters |
| 6 | Trainer: validate prepared role/backend ownership; preparation installs canonical routing proxies |
| 7 | Trainer: load auxiliary inference components, initialize adapter/algorithm snapshots and restore runtime state |
| 8 | Trainer: apply shared acceleration entries in configured order, including attention selection before compilation |

Full variants have distinct module routes. Named LoRA variants share a module and have disjoint
parameter ownership; TP therefore de-duplicates by physical module identity. Frozen bundled
members of the trained architecture receive the same TP plan; another architecture keeps its
existing FSDP-only behavior. The optimizer is constructed after TP and rebound by Accelerate
during FSDP preparation. ZeRO's optimizer partitions remain backend-owned storage, while the
registry describes live model parameters.

Preparation is single-use. Calling transforms, backend preparation or route installation out of
order, or calling them twice, fails before repeating the operation. A failed destructive stage
cannot be retried on the same owner: construct a fresh model instead. Snapshots and exact-state
resume require completed routes. Existing public configuration and checkpoint formats are
unchanged; the preparation phase is transient initialization state.

Custom trainers overriding protected `_initialization` or `_apply_tensor_parallel` hooks must
call the inherited hook or delegate through `self.model_preparation` in the documented order.
Bypassing this owner is rejected by `require_ready()` before snapshot/runtime initialization.
All registered trainers use the inherited preparation path; this guard makes the migration
requirement explicit for external subclasses with custom initialization.

## Bounded CP bridge result

The reproducible probe is
[`test_context_parallel_probe.py`](../tests/distributed/test_context_parallel_probe.py). It uses
random-weight, real Diffusers Wan blocks on one node with `DP-shard=2, CP=2`, native attention and
Ulysses degree 2. The tested tuple is Python 3.10, PyTorch `2.10.0+cu129`, Accelerate `1.15.0`,
Diffusers `0.41.0`, PEFT `0.21.2`, on four NVIDIA L20Y GPUs.

The bridge retains Accelerate's FSDP2 `dp_shard_cp` mesh and gives Diffusers a mesh with matching
CP members under its required `ring` and `ulysses` names, including the singleton ring axis.
A **test-only** Accelerator override bypasses `_prepare_cp`: Diffusers alone installs CP hooks,
and Accelerate continues to own FSDP2 preparation. Attention selection must precede Diffusers
`enable_parallelism()`. The probe verifies full outputs, every gradient, actual AdamW updates
and optimizer state against an unsharded reference processing the same two logical samples.

**Loss normalization is part of the bridge.** Diffusers gathers the complete output but its
gather backward returns only the local CP slice. FSDP2 averages parameter gradients over
`DP-shard × CP`. For this full-output mean loss, multiplying the distributed loss by `CP=2`
recovers the mean over distinct logical samples. An unscaled control demonstrates the erroneous
half-sized gradients. This is a measured rule for this probe's reduction contract; a future
token-local, masked or weighted objective must derive its own normalization explicitly.

This evidence supports retaining Accelerate for the next CP integration. It does **not** enable
public CP, establish TP+CP composition, or qualify CP checkpoint/resume, rollout/reward plumbing,
LoRA, activation checkpointing, mixed precision, arbitrary lengths, other models or cross-node
execution. Native Ring is outside this probe because its path needs LSE output that the selected
native attention forward does not return. General CP support remains P09 work.

P09 must place compatible attention selection before CP hook installation while keeping final
compilation after preparation and state restoration. It must also make logical-sample loss
normalization and checkpoint identity explicit; the probe-local override is not a public backend.
MiniMax H3's eager TP materialization peak remains a separate loading limitation. Neither SP nor
this Wan CP result demonstrates bounded H3 loading.

## Focused verification

The CPU order/identity failures and all backend paths live in
[`test_model_preparation.py`](../tests/distributed/test_model_preparation.py). The four-rank
variant cases use real Wan blocks, `DP-shard=2, TP=2`, SP and activation checkpointing. Both full
copies and shared LoRA execute alternating role updates and compare every gradient, update and
warm optimizer-state tensor, including unchanged inactive and frozen parameters.

```bash
# CPU owner contracts (use a Torch-capable test environment)
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python -m pytest -q \
  tests/distributed/test_model_preparation.py -k 'not cuda'

# Four visible GPUs; workers are spawned by the tests
PYTHONPATH=src OMP_NUM_THREADS=1 python -m pytest -q -s \
  tests/distributed/test_model_preparation.py::test_cuda_variant_preparation_matches_unsharded_updates
PYTHONPATH=src OMP_NUM_THREADS=1 python -m pytest -q -s \
  tests/distributed/test_context_parallel_probe.py

# Two GPUs; repeat with fsdp2 or zero2 and the matching backend config
PYTHONPATH=src python -m accelerate.commands.launch \
  --config_file config/validation/backends/ddp.yaml --module pytest -q -s \
  tests/distributed/test_model_preparation.py::test_cuda_backend_prepares_one_root_and_updates_both_routes \
  --backend ddp
```

The CP case records `execution_scope: model_probe`. The evidence validator accepts only bounded
FSDP2 Q2 numerical CP probes without TP, EP, replicated-DP or reward execution in this scope.
Default runtime cases continue to reject CP. Rank records include the actual backend, device,
world size and dependency tuple; these records are evidence for the stated case, not a blanket
capability declaration.
