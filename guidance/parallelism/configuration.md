# Configure a parallel layout

Choose a model and algorithm, then enable a layout admitted by both. These snippets are overrides
for an otherwise complete training configuration.

## Choose the GPU geometry

Use the [FSDP2 launcher configuration](../../config/accelerate_configs/fsdp2.yaml) for TP or CP.
FSDP2 requires `dp_shard_size >= 2`; CP additionally requires `dp_replicate_size: 1`.
The examples below use one replicated-DP coordinate. Disable FSDP CPU-efficient loading for
either TP or CP.

| Launched GPUs | Configuration | Logical data replicas |
|---:|---|---:|
| 4 | `tp_size: 2`, `cp_size: 1` | 2 |
| 4 | `tp_size: 1`, `cp_size: 2` | 2 |
| 8 | `tp_size: 2`, `cp_size: 2` | 2 |

For the eight-GPU layout:

```yaml
mixed_precision: bf16
parallelism:
  tp_size: 2
  cp_size: 2
  sequence_parallel: true
```

Leaving `dp_shard_size` unset infers it from the launched process count:

```text
world = dp_replicate_size × dp_shard_size × cp_size × tp_size
logical data replicas = dp_replicate_size × dp_shard_size
```

An explicit shard size must match that product. The framework rejects a mismatch instead of
silently changing the layout. `world` means launched processes, not the number of visible GPUs.
A two-GPU TP component test has a different scope from four-GPU FSDP2 training.

## Choose SP explicitly

SP requires `tp_size > 1` and uses the same TP peers. Its setting has three meanings:

| `sequence_parallel` | Meaning |
|---|---|
| Omitted | Use the adapter default: on for SD3.5 and Wan T2V, off elsewhere |
| `true` | Require the model's declared SP plan |
| `false` | Keep SP off, including on models where it defaults on |

Set it explicitly when comparing runs. For CP-only training (`tp_size: 1`), omit SP or set it to
`false`. Bagel also changes attention TP placement when SP is enabled; see its
[parameter coverage](models.md#parameter-coverage).

SP divides only the regions in the [activation coverage table](models.md#sequence-parallel-coverage).
TP weight sharding alone does not imply that all token activations divide evenly or use the same
layout.

## Satisfy the CP contract

The current CP training path uses FSDP2 on CUDA/NCCL, BF16, AdamW and one example per device.
Disable FSDP CPU-efficient loading for this path. Backend activation checkpointing may be enabled;
model-specific checkpoint ownership still applies.

All current registered model capabilities use Ulysses with one shared native attention backend:

```yaml
train:
  per_device_batch_size: 1
acceleration:
  shared:
    - name: attention_backend
      params: {backend: native}
```

Other shared or rollout accelerators are excluded by the current CP gate. Ring is not admitted
by any current registered model. Wan T2V/I2V additionally require:

```yaml
model:
  compute_precision_policy: cross_attention_fp32_v1
```

Use only the canonical target components declared by the adapter. Some models have multiple
scheduled denoisers or a distinct reference target; replacing these with a generic `transformer`
assumption can change which component trains.

The ordinary dependency gate uses Torch 2.10.x, Accelerate 1.15.x and Diffusers 0.42.x. Optional
model-specific development revisions require their declared version and commit. See the
[model contracts](models.md) for those pins and optional packages.

## Check geometry and state

Head divisibility applies after composition. Ulysses usually needs attention heads divisible by
TP × CP; grouped-query attention adds KV-head constraints. Some adapters explicitly pad
communication heads, so use the family rule rather than inferring one from the global head count.

Token constraints are also per stream. Flux1, SD3.5, ordinary Qwen and H3's text refiner have
mask-aware text padding. Other paths may require divisible lengths. Changing resolution, prompt
length, references or video/audio duration can therefore change whether a layout is valid.

| Boundary | What to check |
|---|---|
| Model | Exact variant, target roots, conditioning and packed layout |
| Algorithm | Required outputs, roles, tuning mode and sampler semantics |
| Environment | Core versions plus model-specific dependencies |
| Geometry | Head, token and patch divisibility; supported padding and masks |
| Feedback | Logical-DP grouping and overlap settings |
| Resume | Recorded topology, parameter placement, precision, roles and runtime identity |

Compare layouts at fixed logical sample count and optimizer-update geometry. Moving from eight
DP ranks to DP2 × TP2 × CP2 reduces independent data replicas from eight to two; adjust sampling
and accumulation deliberately.

Model-only export and exact training continuation are different operations. Exact continuation
requires the recorded preparation and a drained acquisition or finite-data epoch boundary.
Topology changes, changed TP placements and arbitrary in-flight recovery are outside that contract.
Ready reward scheduling may change accumulation order; identical asynchronous timing is not a
correctness requirement.

See the [topology reference](../../docs/reference/parallelism/topology.md) and
[TP contract](../../docs/reference/parallelism/tensor-parallel.md) for implementation details.
