# Parallel topology and logical samples

`ParallelTopology` in `src/flow_factory/contracts/topology.py` is the shared authority for
configuration geometry, runtime scope helpers and checkpoint layout identity. It describes
physical dimensions and sample ownership; adapter/backend capability checks decide which layouts
can execute training.

## Dimensions and ownership

The canonical physical rank order is `(dp_replicate, dp_shard, cp, tp)`, with TP varying fastest.
This matches Accelerate's mesh construction. Singleton axes are omitted from the actual root mesh.
SP reuses TP and therefore does not multiply the number of ranks.

| Quantity or scope | Meaning |
|---|---|
| Physical world | `dp_replicate * dp_shard * cp * tp` processes |
| Logical data replicas | `dp_replicate * dp_shard` distinct sample batches |
| Sample peers | All TP x CP coordinates of one logical data replica |
| TP group | Fixed data and CP coordinates; vary TP |
| CP group | Fixed data and TP coordinates; vary CP |
| Data/statistics group | Fixed TP and CP coordinates; vary the data coordinates |

For a planned `dp_shard=2, cp=2, tp=2` layout, ranks 0–3 share one sample batch and ranks 4–7
share another. Rank 3's TP group is `(2, 3)`, CP group is `(1, 3)`, and data/statistics group is
`(3, 7)`. A sample count reduces over two logical replicas. Reducing it over all eight processes
would count every sample four times.

This example describes topology, not a supported CP training configuration. Production CP remains
unavailable. Its preparation/attention probe belongs to P06 and its adapter qualification to P09.
In particular, a backend's gradient synchronization scope is not necessarily its sample-statistics
scope: FSDP can shard over the joint `dp_shard_cp` dimension while sample counts still use only DP.
The topology contract does not choose a CP loss-scaling or attention implementation.

## Configuration and runtime

The existing public `parallelism` keys remain unchanged:

```yaml
parallelism:
  tp_size: 2
  # dp_shard_size: derived from the remaining launched ranks
  # dp_replicate_size: defaults to 1
  # sequence_parallel: unset keeps the adapter's qualified default
  verify_replicated_inputs: false
```

Only the shard degree is inferred. All explicitly supplied dimensions must multiply to the
launched world size; inconsistent dimensions fail during configuration parsing, before model
loading. TP training still requires FSDP2 with at least two shard ranks. For example TP=2 requires
at least four ranks under the current training contract. A two-rank TP test qualifies a primitive,
not a complete FSDP2 training run.

When an explicit plan is built, all Accelerate dimensions are supplied explicitly. Environment
variables cannot silently add CP or independent DeepSpeed SP. Launcher-only CP/SP overrides on an
otherwise unconfigured run are rejected before weights load. Flow-Factory's `sequence_parallel`
flag refers to its TP-bound overlay, not Accelerate's separate `sp_size` dimension.

With no plan, each physical rank remains one logical sample replica. DDP and ZeRO-1/2 replicate
weights; ZeRO's optimizer/gradient partitioning does not become an FSDP mesh axis. FSDP uses a DP
shard dimension. An all-ones Accelerate plan is handled as this unconfigured case, preserving the
existing FSDP2 launch behavior.

Runtime helpers continue to obtain the local data coordinate from the actual flattened `dp` mesh.
The name `dp` need not appear in the root `mesh_dim_names`. Root axis sizes/order and physical rank
placement are checked once per mesh against the canonical contract; a custom incompatible mesh
fails instead of silently selecting another sample's coordinates. Flow-Factory does not create a
second mesh or additional process groups for these checks.

## Data, feedback and checkpoint boundaries

The generated sampling planner, offline `DistributedSampler`, replica RNG, reward coordinator and
replica reductions retain their existing public interfaces. TP x CP peers receive identical
logical sample identities and seeds; separate data replicas receive their own partitions.
`GroupCoordinator` may further narrow feedback to an existing sampler subgroup or local group.
The existing prohibition on `subgroup_tile` combined with TP remains in place.

A single logical replica bypasses sample gathers and reductions even if several physical TP/CP
ranks share it. A missing subgroup handle must not accidentally dispatch such an operation to the
physical world. This rule covers tensor, object, and heterogeneous sample-field transport.

Existing supported checkpoint identity payloads remain byte-for-byte schema compatible: no extra
field is added for an unconfigured/data-only run, and TP records the same resolved dimensions and
SP flag. CP metadata changes the existing `cp_size` field, so two different CP/data layouts cannot
share an exact-state identity. This does not promise CP checkpoint support.

## Evidence

- `tests/contracts/test_topology.py`: dimensions, group membership, invalid products/ranks and
  logical counts, without process groups.
- `tests/distributed/test_topology_scopes.py`: actual Gloo TP/CP meshes, generated/offline sample
  identities, replica RNG, statistics and singleton-replica bypass. Its selected two-device NCCL
  cases cover scope routing, not model CP execution.
- Existing data-geometry, sample-collective and replica-seeding tests cover compatibility with
  the previously supported TP layouts.
- Configuration and runtime-identity tests pin early rejection and previous checkpoint schemas.

See [selected validation plans](validation.md) for QA levels and evidence requirements, and
[the v0.4 milestone](../roadmap/milestones/v0.4.md#p05) for integration acceptance.

Model-specific TP/SP qualification and exclusions are recorded in the
[TP/SP guide](tensor_parallel.md). Topology-only CP probes do not extend that support table.
