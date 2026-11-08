# Selected validation evidence

**Read when**: Selecting or reviewing semantic evidence under `constraints.md` #30, including R3/R4
changes and GPU qualification. The canonical procedure and schema are in
[Validation plans](../../../docs/development/validation/validation.md).

## Selection contract

Risk R0-R4 describes semantic impact. Facets identify affected owners/contracts. Q0-Q3 describes
the proof needed. Topology supplies the minimum resources for that proof. These are separate
choices: R4 requires coverage of every affected obligation, not a fixed GPU count or Cartesian
matrix. Review every reached registry implementation; use representative executions only when
unselected implementations have the same relevant behavior and the plan explains why.

R3/R4 changes require a selected plan and results for the reviewed commit. R1/R2 can report their
focused commands and results in the review; use the same plan format when composition or GPU
claims benefit from it. Missing hardware leaves required evidence blocked. A safe verdict cannot
skip that obligation. Scope automation is advisory: reviewers add semantic dependencies that the
path/content rules cannot see.

Use two local ranks for a basic distributed boundary and two nodes with one rank each for a
changed cross-node path. The current FSDP2+TP training contract needs dp_shard >= 2 and TP >= 2,
so it requires at least four ranks. A two-rank TP primitive proves only that primitive. The bounded [Wan CP SFT capability](../../../docs/reference/parallelism/context-parallel.md) requires an explicit
runtime capability in the evidence plan. Wan TP+CP uses a separate capability and requires
DP2 × TP2 × CP2 for the direct logical-DP interaction case. EP remains separate work.
No fixed reward deployment is required for all tests.

Prefer Q0 contract/numerical tests; add Q1 local CUDA, Q2 actual backend preparation and updates,
or Q3 selected real weights as the claim requires. Q0 Torch/Gloo tests may need the training
environment; they are not implicitly part of the dependency-light `make test-cpu` suite.
Keep effective logical samples fixed when comparing parallel layouts. Check elementwise
gradients/updates, trainable membership, inactive state isolation, and checkpoint continuation
where those contracts change. Finite loss or a successful exit alone cannot prove equivalence.

For tile streams, exercise multiple work units, controlled readiness/order, source identity,
backpressure, cancellation, and cleanup as affected. Event ordering can prove functional overlap;
controlled waits cannot quantify hidden reward latency. Use varying rewards for learning oracles.
A real reward model is needed when its adapter or measured workload is the subject of the test.

## Historical experiments

The [retained campaign and PR #220 archive](gpu_campaign_legacy.md) records the old deployment
and model/algorithm results. `config/gpu_validation/framework_upgrade.yaml` and
`scripts/validate_gpu_campaign.py` remain usable for explicitly selected historical campaigns.
Their fixed matrix is not the default merge gate; old results do not certify a new source commit.

## Cross-refs

- UP: [`constraints.md` #30](../constraints.md#30-selected-validation-evidence)
- PROCEDURE: [Validation plans](../../../docs/development/validation/validation.md)
- CODE: [`agent_scope.py`](../../../scripts/agent_scope.py), [`validate_validation_plan.py`](../../../scripts/validate_validation_plan.py)
- WORKFLOW: [`ff-develop`](../../skills/ff-develop/SKILL.md), [`ff-review`](../../skills/ff-review/SKILL.md)
