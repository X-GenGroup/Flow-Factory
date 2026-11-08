# Coverage, memory and performance

Choose parallelism for the resource that limits your run. Parameter placement, retained
activations, peak memory and throughput are separate measurements.

| Limiting resource | Useful starting point | Cost to inspect |
|---|---|---|
| Optimizer/gradient state | ZeRO-2 or FSDP2 | State synchronization and parameter gathers |
| Large matrix weights or contractions | TP with FSDP2 | Activation collectives and fewer independent samples |
| Residual/norm/modulation activations | SP on supported regions | Gather/scatter boundaries and replicated work |
| Long image/video/audio context | CP | Attention transport, head divisibility and padding |
| Reward wait time | Async reward and optimization overlap | Service capacity and ready training work |

## What the coverage numbers mean

The [TP percentages](models.md#parameter-coverage) count canonical base parameter elements in a
specified component. They exclude LoRA additions, activations, optimizer state and other pipeline
components. A 99.6% placement figure is not a 99.6% memory reduction, speedup or SP coverage figure.

SP can reduce retained activations in the [listed regions](models.md#sequence-parallel-coverage).
Current Flux2/3 plans, for example, keep residual shards between blocks in each stack and restore
complete streams at native packing boundaries. Embeddings, conditioning, caches, gathered operands,
attention workspaces and output reconstruction still consume memory.

A smaller local residual is direct evidence of a changed layout. A smaller peak allocation or
faster step requires a separate measurement. Checkpointing, precision, batch geometry and component
residency can change which allocation determines the peak. There is no comparable per-family
memory or throughput benchmark for the latest SP extensions, so this guide assigns them no
numerical memory reduction or speedup.

Canonical packed TP can require additional activation exchanges to retain native weight ordering.
SP also needs reductions for some replicated parameters and conditioning. Communication is
model-dependent; replacing an all-reduce with gather/scatter operations does not establish a
universal equal-traffic or equal-speed result.

## Read numerical evidence at its stated scope

Native component tests compare model outputs, individual gradients, parameter updates and
optimizer state, and observe actual token shard lengths. Selected trainer tests additionally
exercise logical samples, rewards, replay, role ownership and continuation. These are correctness
checks; neither test size nor a successful exit is a throughput or convergence measurement.

Released-weight evidence has narrower scopes. The
[qualification reference](../../docs/reference/parallelism/tensor-parallel.md) records named Wan
LoRA paths, Qwen-Image-2.1 FP32 same-state LoRA updates with SP/cache/checkpointing disabled, and a
separate H3 Ref2VA TP4/SP/CP1 loading/update capacity result. Those historical runs do not establish
released-weight capacity or update parity for every newer SP region or CP composition. Retained
failures and the exact conditions remain part of the references.

Reward overlap can hide service wait time when training work is ready. Its benefit depends on
rollout cost, reward latency, tile size and optimization time. Functional overlap does not imply
a guaranteed speedup.

## Compare equivalent runs

Hold model revision, inputs, logical sample/update geometry, precision, checkpointing and optimizer
settings fixed. Record:

- Full DP/TP/CP geometry and the SP regions enabled.
- Peak allocated and reserved GPU memory, including preparation if capacity is the concern.
- Step time after warmup and throughput per logical sample.
- Communication time and time waiting for rewards.
- The numerical update comparison used to establish that the runs perform equivalent work.

Report both the benefit and its cost. Increasing model parallelism can make a sample fit while
reducing independent data throughput; the useful choice depends on the workload.
