# Validation plans

For generated plans and the shared local/CI gate, see [automatic validation](automatic-validation.md).

Select evidence for the changed contract. Risk, affected owners, QA level and execution topology
are separate decisions. This is the P01b manual workflow; P02 supplies reusable fixtures and P04
automates selection through the linked planner. The validator checks declared coverage and identities. Reviewers must
still check that the plan covers the diff and that logs substantiate the recorded observations.

## Inspect and select

1. Run `python3 scripts/agent_scope.py --base origin/main`. It compares tracked worktree content,
   including staged additions, with the base. Stage intended new files first. Untracked directory
   names are reported without recursively reading them. Binary, symlink, oversized and unverified
   source inputs remain visible; incomplete collection exits with code 2. Resolve their relevance
   in review, rather than interpreting a partial classification as a low-risk pass.
2. For reproducible review, use `--base <base-sha> --head <tested-sha>`. Both revisions are resolved
   to full commit IDs. The immutable mode ignores subsequent working-tree edits and unrelated
   artifacts. Renames are inspected as deletion plus addition. Reads are bounded to 1 MiB per
   source and 16 MiB total; no external Git diff/textconv driver runs.
3. Enumerate affected owners, reached registry implementations, invariants and interactions.
   Explain why representative cases cover the unselected implementations. Path rules are a
   starting point; reviewers add dependencies that the classifier cannot infer.
4. Choose tests and record each selection's reason. For R3/R4, record a manual plan using
   [the template](../config/validation/plan.example.yaml). For R1/R2, focused commands and results
   in the review suffice; the same format is available for more involved evidence.

Python comment/docstring exemptions use token equivalence. The separate logging exemption only
recognizes literal calls to the existing module console logger, with unchanged remaining AST and
tool directives. Formatting arguments, f-strings, `self.logger`, custom/shadowed loggers and
changed control flow retain normal owner rules. This does not certify handler performance or
downstream log consumers; inspect those when affected. Edits outside an exempt file still escalate.

## QA and topology

| QA | Proof | Resource choice |
|---|---|---|
| Q0 | CPU contracts, numerical references, Gloo group/geometry tests | CPU; some tests need Torch |
| Q1 | Local CUDA, autograd, dtype, adapters and recomputation | One GPU with a small valid model structure |
| Q2 | Actual distributed backend preparation, updates and state | Smallest valid interacting mesh |
| Q3 | Real pretrained weight/configuration compatibility | Selected model and backend; record revision |

These are evidence types, not a mandatory ladder. A backend-independent objective normally needs
a numerical oracle, not three backends. A model-specific sharding plan must preserve that model's
structure. A shared backend or ownership change covers every affected backend behavior. Add known
interactions directly instead of relying only on pairwise coverage. Capacity and throughput are
separate experiments. The lightweight quality environment does not contain Torch; run Q0 Gloo/math
tests that require it in the appropriate test environment.

Plans record `nodes`, `ranks_per_node`, `dp_replicate`, `dp_shard`, `tp`, `cp` and `ep`. The product
of mesh dimensions must equal the placement's rank count. The current schema describes a uniform
placement; CPU/Gloo cases use CPU processes, all other backends use one GPU per rank.
DDP and ZeRO-2 use `dp_replicate` for their full model copies. ZeRO-2 optimizer-state
partitioning is backend storage and does not create a `dp_shard` model dimension; FSDP2
uses `dp_shard` for its model shard group.

| Claim | Example minimum |
|---|---|
| Local distributed group | 1 node x 2 ranks |
| Cross-node path | 2 nodes x 1 rank |
| TP primitive (`backend: tp`) | TP=2; does not certify production training |
| Current FSDP2+TP training | DP-shard=2, TP=2; 4 ranks |
| Multiple nontrivial subgroups | Often 4 ranks; justify the needed dimensions |

Runtime cases reject smaller FSDP2 sharding, DDP/ZeRO-2 TP and EP. CP requires the explicit
`execution_capability: wan-sft-ulysses-bf16-v1`: FSDP2 Q2, TP=1, replicated-DP=1, sharded-DP>=2,
a numerical oracle and no rewards. `wan-sft-tp-ulysses-bf16-v1` admits TP>=2 with the same
remaining constraints. Results must repeat the observed capability. SP follows the
current TP implementation and is not an independent mesh axis. Further CP/EP layouts must update
the contract and tests together. Compare layouts on identical logical samples and
effective update geometry; physical world size is not the logical data-replica count.

A case may explicitly use `execution_scope: topology_only` for process-group geometry without
model execution. This narrow scope accepts `backend: gloo` (Q0) or `backend: nccl` (Q2), including
CP axes, with an `assertions` oracle and `reward: none`. It does not permit EP, real-model Q3
claims, model numerical qualification or FSDP/DDP/ZeRO preparation claims. Raw NCCL is available
only under this scope. Passing results must repeat `execution_scope: topology_only` and report
the actual raw process-group backend and every completed rank. Existing cases default to
`execution_scope: runtime` and retain all prior production-capability restrictions.

A bounded experimental model case may use `execution_scope: model_probe` for FSDP2 Q2 with
CP and a numerical oracle, without EP, replicated-DP or rewards. TP>=2 additionally requires
`execution_capability: wan-tp-ulysses-fp32-probe-v1`. It must report elementwise
comparison and nonzero gradients. This scope records the [P06 bridge experiment](model_preparation.md),
not complete CP training. The separate runtime capability qualifies the bounded
[Wan SFT path](context_parallel.md); a passing model probe alone cannot establish it.

## Plan and result contract

Keep generated plans, logs and results under `.scratch/`, and attach them to the review. The
template's zero identities are placeholders. A structural pass is explicitly `plan_valid` and
does not validate source availability or execution.

```bash
python3 scripts/validate_validation_plan.py .scratch/plan.yaml
python3 scripts/validate_validation_plan.py .scratch/plan.yaml \
  --results .scratch/results.json --expected-commit <full-reviewed-commit-sha>
```

The plan includes full immutable `base_commit` and `tested_commit`, `risk`, `owners`, `facets`,
SHA-256 `artifacts` for relevant tests/configs, obligations and cases. Each obligation has a unique
`id`, `description` and boolean `required`. Each case has a unique `id`, `covers`, boolean
`selected`, `reason`, `fixture`, command argv, `qa`, `backend`, `topology` and `oracle`.
Include helper/fixture files and dependency locks that affect the test in `artifacts`; record
external weight/dataset revisions in `fixture`. Human review checks completeness of this list.

All required obligations need selected coverage. Every selected case is gating; optional
experiments outside the diff are `selected: false` with a reason. Empty, duplicate, unknown or
uncovered obligations are rejected. The parsed plan's canonical JSON hash is printed as
`plan_sha256`; YAML whitespace does not change it.

Results use this structure (the values below describe fields, not execution evidence):

```text
schema_version: 1
tested_commit: full tested commit
plan_sha256: hash printed by plan validation
artifacts: same test/config path -> SHA-256 mapping as the plan
environment:
  python: observed Python version
  hardware: observed CPU/GPU model, driver and node details as relevant
  packages: observed dependency name -> version mapping
cases:
  - id: planned case ID
    status: pass | fail | blocked | not_selected
    reason: required for any non-pass
    backend: observed backend (required for pass)
    topology: observed dimensions/placement (required for pass)
    command: executed argv (required for pass)
    runtime: observed device, process/plugin identity and rank completion (required for pass)
    evidence: log/report path -> SHA-256 mapping (required for pass)
    observations: oracle-specific facts (required for pass)
```

Missing selected results become `blocked`; they never pass. Unselected cases may be omitted or
recorded as `not_selected`. Wrong commit, plan hash, test/config hash, command, backend or topology
is rejected. For result validation, the tool also checks source commit availability and hashes
test/config blobs from that commit (not the current worktree). `--repo` selects a different local
repository if necessary. Result validation returns 0 only for a fully passing selection, 1 for
failed/blocked evidence, and 2 for malformed/stale inputs. The tool never launches commands.

Record runtime backend/plugin observations in the hashed logs: a `torchrun` label does not prove
ZeRO-2 or FSDP2. `runtime` must contain integer `world_size`, all `ranks_completed` as rank IDs,
`device` (`cpu` or `cuda`) and `distributed_type`. Local CPU/CUDA uses `NO`; DDP uses `MULTI_GPU`;
ZeRO-2 uses `DEEPSPEED` plus integer `zero_stage: 2`; FSDP2 uses `FSDP` plus integer
`fsdp_version: 2`. Raw Gloo and TP primitive tests use `PROCESS_GROUP` plus
`process_group_backend: gloo` or `nccl`, respectively. Artifact hashes do not authenticate
measurements; reviewers inspect the evidence
files and assertions. Re-run after source/test/config changes. Do not copy an old passing result
onto a new commit. An immutable local candidate commit can be tested before advancing the PR
branch; keep its generated evidence outside the tested source commit.

## Numerical and overlap oracles

Every oracle has a `kind`, explanatory `description`, and reward mode `none`, `constant`,
`varying` or `real`. Constant fixtures can test scheduling or a zero-signal boundary; they cannot
establish a learning update or measured reward performance.
All passing observations require integer `exit_code: 0`.

- `assertions`: focused tests must assert the actual contract; exit success alone is insufficient.
- `numerical`: record `elementwise_comparison: true`, `nonzero_gradients: true`, and
  `varying_rewards: true` when rewards contribute. The description/logs identify tensors,
  tolerances, reference precision, parameter updates and inactive optimizer-state checks. These
  fields record passed assertions; they do not replace their numerical reports. Constant rewards
  cannot establish a learning update.
- `overlap_order`: record an ordered `events` list of `{unit, event}`. Each work unit follows
  `submit`, `ready`, `optimize_start`, `optimize_end`. At least two units complete, no work remains,
  and optimization starts while another submitted reward remains pending. Controlled events may
  force readiness ordering; additional assertions cover affected backpressure, failure cleanup,
  multi-source identities and ready/ordered selection semantics.
- `overlap_timing`: additionally record `artificial_delay: false`, a positive finite
  `overlap_seconds`, `timing_method` and `workload`. Artificial sleeps and a functional event trace
  do not establish production latency hiding. Use a separate measured workload.

The [historical GPU campaign](../.agents/knowledge/topics/gpu_campaign_legacy.md), its manifest
and its reader remain available for explicit qualification requests. Its 45 jobs, 32 training
GPUs and reward deployment are not an automatic requirement for any risk level.

Reusable executable cases and controlled fixtures are listed in the
[validation catalog](validation-catalog.md). Catalog inspection is independent of execution.
