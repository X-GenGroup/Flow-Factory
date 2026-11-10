# Loading, memory and checkpoint migration

Flow-Factory prepares one bundle of target components for the distributed backend. Loading the
weights, placing target parameters and restoring optimizer state are separate operations.

## FSDP2 target loading

```yaml
parallelism:
  tp_size: 2
  sequence_parallel: true
  target_materialization: sharded
```

Leaving `target_materialization` unset prefers `sharded` for qualified FSDP2 adapters, including
CP compositions. Every current registered adapter admits this target-materialization path.
Parameters move to meta after initial-state synchronization; TP/SP/CP and FSDP2 establish their
final placements before each rank allocates and fills its local shards. This lowers preparation
memory without changing the original-parameter precision policy or optimizer ownership.

| Setting | Behavior |
|---|---|
| Unset | Prefer sharded; fall back to replicated outside its capability boundary, with a warning under FSDP2 |
| `sharded` | Require the path; reject unsupported backends/adapters or FSDP2 CPU offload |
| `replicated` | Keep the loaded root while TP and FSDP2 shard it; useful as an explicit comparison/fallback |

DDP and ZeRO-2 retain their backend loading behavior. CPU-RAM-efficient loading can compose with
TP on the sharded path. Auxiliary encoders, VAEs and reward modules keep their own loading and
residency owners; target sharding does not imply sharding every pipeline component.

Checkpoint-backed safetensors slices additionally reduce host memory for supported Diffusers and
modular loaders, including H3 and Flux3. Bagel and SenseNova retain their native host loaders and
still use sharded GPU materialization. The loader falls back when native construction transforms
or weight restoration require ordinary loading; GPU materialization and host slice loading are
not the same capability.

The [H3 capacity record](../../docs/reference/parallelism/models/minimax-h3.md) distinguishes
historical replicated-loading failures from later sharded-loading successes. For its recorded
Ref2VA LoRA canary, DP2×TP2 preparation fell to 30.98 GiB and the native-step peak was 43.86 GiB.
Those numbers are specific to that model revision, geometry and precision; they are not a general
limit for H3 training or a throughput claim.

## Choosing a resume mode

| Intent | Required compatibility |
|---|---|
| Continue training state exactly | Same physical world size/rank order, DP/TP/CP geometry, placements, roles, algorithm, precision and saved runtime contract |
| Start new training from saved model weights | Canonical model weights and compatible model configuration; initialize new optimizer/scheduler state |
| Restore while changing topology | Use weights-only restart; topology-changing optimizer continuation is outside v0.5 |

Current TP runtime identities fingerprint tensor placements as well as canonical names and shapes.
Older TP training-state checkpoints without that fingerprint require a weights-only restart or
an explicitly reviewed migration. A newer package version does not silently upgrade incompatible
optimizer state. Replicated and sharded materialization can share a state contract when their final
placements and other recorded identities match.

Exact restore and the first same-state update are distinct from later asynchronous scheduling.
Ready-order reward overlap may change subsequent floating-point accumulation order; valid logical
samples, gradients and optimizer updates remain required. Drain the pipeline at supported save
boundaries: arbitrary in-flight reward/pipeline restoration is not supported.

See [parameter-state ownership](../../docs/development/parameter-state.md) for the serialized
contract and [reward overlap](../training/workflow/reward_optimization_overlap.md) for lifecycle
boundaries. Keep the original receipt's commit, dependency tuple and topology when reusing evidence.
