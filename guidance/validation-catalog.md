# Focused validation catalog

[`config/validation/catalog.yaml`](../config/validation/catalog.yaml) describes reusable proof cases.
It inventories existing suites and a few focused fixtures; it is not a model × algorithm × backend
matrix. The [selected-plan contract](validation.md) still decides which obligations a change must
prove. Catalog presence is not a support claim or evidence that a case has run.

Inspect the catalog without importing Torch or allocating a GPU:

```bash
python3 scripts/validation_catalog.py
python3 scripts/validation_catalog.py --case reward-worker-device
```

The script needs PyYAML, available in the isolated quality environment. `make test-cpu` checks
catalog invariants. A small source reader callback lets the P04 planner validate artifact paths
and test selectors against immutable Git blobs instead of the current working tree.

## Case fields

| Field | Meaning |
|---|---|
| `id`, `covers` | Stable case and proof-obligation identifiers; coverage is limited to the stated oracle |
| `facets`, `owners` | Candidate selection hints; reviewers still establish the affected contracts |
| `command` | Argument vector with an explicitly bound Python interpreter, never a shell string |
| `fixture`, `fixture_kind` | Synthetic tensors, an actual model class with generated weights, or pretrained weights |
| `qa`, `backend`, `topology` | Evidence level, actual backend and process-group geometry |
| `oracle` | Assertions and reward mode; the description states the claim and exclusions |
| `artifacts` | Test/config inputs whose identities belong in a selected evidence plan |
| `cost_seconds` | Rough scheduling estimate, not a benchmark or test timeout |
| `resources` | GPUs visible per node, package prerequisites, environment and required model-path variables |

Topology and hardware are separate. `reward-worker-device` runs one fresh interpreter, uses
`cuda:1`, and checks that `cuda:0` has no primary context. It therefore requests two visible GPUs
while retaining one process in its Q1 topology. CPU/Gloo cases need the training environment but
zero GPUs. The composed checkpoint fixture uses four CPU ranks for TP2 × DP2; it does not certify
CUDA FSDP2 state persistence.

Per-model TP/SP entries distinguish real model structures from pretrained weights. Their gradient
oracles compare membership and values against an unsharded reference. Optional model imports may
still be unavailable in a particular Diffusers build; a skipped case is missing evidence, not a
successful qualification. No case promises the entire corresponding model family is supported.

## Running a selected case

Use a verified training environment for Torch cases and bind its interpreter explicitly:

```bash
CUDA_VISIBLE_DEVICES=0,1 python3 scripts/validation_catalog.py \
  --case reward-worker-device --run --python /path/to/training/bin/python
```

`--run` performs package/GPU/model-path preflight and reports unmet prerequisites as `blocked`
(exit 2). It then returns the command's exit status. This convenience runner does not produce an
accepted evidence bundle: inspect pytest skip/collection results and use the selected-plan result
validator before recording a pass. The P04 planner/report workflow owns automatic selection,
exact-source identities and the final gate. The interpreter, installed dependency versions,
logical geometry, actual completed ranks and full logs belong in that evidence.

The backend cases use small two-rank configurations under
[`config/validation/backends/`](../config/validation/backends/ddp.yaml). They select one existing
production TDM-R1 phase test with LoRA-shaped storage and GAS 2. They assert one prepared root,
logical model role updates, checkpoint counters and optimizer steps. ZeRO-2 can expose a
padding-only optimizer shard on a rank, so the update oracle reads actual model weights while
state-isolation checks inspect local optimizer storage, including padding. During a second cycle, inactive parameters
and every populated Adam state tensor must remain unchanged. DDP, ZeRO-2 and FSDP2 are separate
cases; a CPU role test cannot replace a selected backend obligation. Pass a free
`--main_process_port` to the underlying Accelerate command when concurrent jobs share a node.
There is no cross-node claim from a same-node run.

The Q3 canary is deliberately narrow:

```bash
FF_VALIDATION_WAN_TRANSFORMER=/local/Wan2.1-T2V-1.3B-Diffusers/transformer \
CUDA_VISIBLE_DEVICES=0 python3 scripts/validation_catalog.py \
  --case pretrained-wan-canary --run --python /path/to/training/bin/python
```

The checkpoint must be a local ordinary `WanTransformer3DModel` T2V transformer. The test rejects
Animate/I2V configuration and never downloads weights. Record the upstream revision and hashes of
all local checkpoint files alongside the result. The test verifies a finite forward, exact
trainable gradient membership, finite nonzero gradients, actual LoRA updates and exact frozen
parameter values. It does not validate text encoding, VAE decode, output quality, TP/SP or a reward
model. A model-path variable missing at selection time remains blocked.

## Controlled feedback fixtures

The overlap suite releases a chosen tile's reward gate and waits for that request's Future,
rather than hoping it completes before a short wall-clock deadline. A generous timeout only
catches deadlocks. Cleanup releases all remaining gates, shuts down executors and asserts worker
threads have terminated even if the test body fails. Existing trainer cases cover ordered and
ready scheduling, group closure and source identity. These are functional-order assertions;
they do not measure fully hidden reward latency.

The [local protocol fixture](../tests/validation/conftest.py) binds an ephemeral loopback port and
serves the existing remote reward client's `/health` and `/compute` endpoints. Pixel-derived
scores vary within one prompt group. Its [tests](../tests/validation/test_reward_protocol.py)
check full/tail request sizes, response ordering, reconstruction, malformed requests and cleanup.
It needs neither pretrained reward weights nor a shared deployment. A sandbox that forbids
loopback sockets must report the resource restriction; it cannot treat the case as passed.

To extend the catalog, first reuse a test with the required assertions. Add a new fixture only
for a missing contract. Keep backend-specific coverage IDs distinct, use the smallest valid mesh,
and add metadata regression tests for any new schema boundary.
