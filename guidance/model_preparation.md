# Model preparation

`BaseTrainer` keeps the initialization sequence visible. Its `ModelPreparation` owner performs
the destructive model transforms, one backend preparation, parameter rebinding and route
installation. `ModelLoadCoordinator` and its backend runtime retain loading and wrapping policy.
Adapters retain precision, attention and activation-checkpoint capabilities.

| Order | Owner and operation |
|---|---|
| 1 | Adapter: load targets, freeze weights, inject base LoRA, apply dtype policy |
| 2 | Trainer and variant registry: materialize every role, restore role weights, align dtypes |
| 3 | Preparation: synchronize FSDP2 replicated initial state; apply TP/SP, or select native attention and install declared CP hooks, once per physical root |
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

<a id="bounded-cp-bridge-result"></a>
## Context-parallel preparation

The P06 bridge is now the guarded production path described in
[context parallelism](context_parallel.md). Diffusers owns native Ulysses hooks; Accelerate owns
the FSDP2 `dp_shard_cp` mesh. The Accelerator bridge checks one prepared FF bundle and its installed
plan before bypassing Accelerate's separate CP setup. Native attention selection happens before
hooks; the normal CP=1 acceleration order is unchanged.

The retained FP32 probe checks three updates, warm optimizer state and an unscaled-loss negative
control. Complete BF16 SFT tests cover full/LoRA tuning and backend recomputation. Unsupported
algorithms, evaluation, EMA, exact resume and TP composition fail at the capability boundary.

The [merged-baseline qualification](integration-qualification.md) remains historical evidence for
`fbb038e3`: its two-node probe did not expose public CP. Later CP results must name their own source
revision. Neither CP nor SP resolves MiniMax H3's separate eager-loading memory peak.

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
Runtime CP cases additionally require the explicit `wan-sft-ulysses-bf16-v1` capability. Rank records include the actual backend, device,
world size and dependency tuple; these records are evidence for the stated case, not a blanket
capability declaration.
