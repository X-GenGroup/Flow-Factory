# Parameter state and exact resume

Live trainable variants and temporal parameter copies have different owners. Existing variants
remain in one `ModelBundle`, with disjoint optimizer groups. A fixed reference, EMA or named
snapshot is a detached value copy, with no optimizer membership or gradient accumulation.
Algorithms choose when those copies update; the checkpoint layer only stores their values and
counters.

| State | Writer and lifetime | Exact checkpoint contents |
|---|---|---|
| Trainable variant | Its active role and optimizer; lifetime of the prepared bundle | Prepared model, optimizer moments, role counters |
| Fixed reference | Explicit initialization or copy; unchanged by optimizer steps | Parameter values and disabled update cadence |
| EMA | Configured decay and `ema_update_interval` | Parameter values, schedule, interval and update count |
| Algorithm snapshot | The algorithm's named copy/update operation | Declared name, parameter values and update count |

An EMA interval of two updates after steps 2, 4, 6, and so on. Interval zero disables automatic
updates. Snapshot/EMA contexts install values temporarily and restore the live parameters even
when their body raises. Exact checkpoints reject an open parameter-swap context or an incomplete
acquisition boundary. Loading snapshots validates the complete tree before copying any value.

## Storage and compatibility

Replicated state retains the existing v1 JSON plus safetensors format. FSDP2 writes a v2 storage
manifest containing one auxiliary payload per physical rank. Each DTensor leaf records its
local shard, global shape and stride, dtype, device mesh, placements and storage device type.
This includes FSDP2's composed `_StridedShard` placements under TP. No full-model gather is
required, and a CPU-offloaded EMA retains its CPU storage with the existing CUDA mesh.

Every rank writes its payload before the publisher commits the root manifest. The manifest
hashes model/optimizer/RNG artifacts and every auxiliary rank file. Resume checks the complete
file set, hashes, runtime identity and local tensor layout before calling Accelerate to restore
model or optimizer state. It then commits progress and auxiliary children. A checksum is an
integrity check, not a signature; checkpoints must still come from a trusted producer.

Existing supported v1 state checkpoints remain readable. Auxiliary child schemas remain v1;
v2 identifies the distributed storage envelope. Old Flow-Factory revisions that only understand
the v1 storage envelope cannot read new FSDP2 v2 checkpoints. Model-only export remains separate
and does not carry optimizer or auxiliary state.

Supported resume keeps the same physical world, rank ordering, backend mesh, tensor placements,
model/algorithm configuration and auxiliary storage placement. Changing those requires a model
weights restart or an explicit migration. FSDP1 auxiliary persistence remains unsupported.
FSDP2 exact state requires a shared checkpoint directory and `save_on_each_node=false`; per-node
auxiliary checkpoint directories are rejected before save/load. The bounded two-node qualification
below uses shared checkpoint storage and preserves the same topology at resume.

## Checkpoint timing

Trainer checkpoints require at least one completed optimizer update. Periodic, direct and final
saves skip optimizer step zero for every backend, algorithm and checkpoint kind, including
model-only saves. Skipping creates no checkpoint directory and does not initialize optimizer
state or enter a save barrier. Evaluation at the initial boundary remains available.

Eligibility uses restored global optimizer progress, not the acquisition counter or the age of
an individual role. Existing checkpoint directory names still use the acquisition counter.
`BaseTrainer.save_checkpoint()` returns whether it saved; custom overrides must preserve that
result so the cycle owner records only completed saves. Adapter weight export remains a separate
model API and is not a trainer checkpoint policy.

## Qualification

The retained [GPU resume test](../tests/distributed/test_parameter_state_resume.py) exercises
actual Accelerate FSDP2 with two local GPUs and FSDP2 DP2 x TP2 with four local GPUs. A small
random-weight two-role module keeps the test focused on storage and role ownership. It uses
AdamW (learning rate 1e-3, epsilon 1e-4, weight decay 0.01, clipping norm 1), FP32 and no rewards.
It performs three updates per role before saving, then compares the next update against an
uninterrupted run. The loss consumes the saved reference, EMA, old-policy and variant snapshot
values, so omitting a copy affects the oracle.

Assertions cover every parameter and populated AdamW moment, inactive-role isolation, fixed
reference immutability, EMA interval two, snapshot cadence, CPU offload, and exact next-update
values (zero absolute and relative tolerance in this deterministic fixture). A self-consistent
manifest with a wrong DTensor layout is rejected on every rank before any backend load.
The environment is Python 3.10, PyTorch 2.10.0+cu129 and Accelerate 1.15.0. This is Q2 runtime
qualification; it does not claim new real-weight, mixed-precision or cross-node model coverage.

The [2026-11-05 merged-baseline follow-up](integration-qualification.md) also runs this fixture on
two physical nodes, covering FSDP2 and FSDP2+TP. For the four-rank case, both TP-crossing and
DP-crossing placement reproduce the exact next update and populated optimizer moments. This is
fixed-topology runtime-state evidence; it does not qualify topology changes or a real-weight model.
