# Automatic validation plans

P04 builds on the [manual evidence contract](validation.md) and the P02 test catalog. Impact
classification, required coverage, execution resources, and result acceptance remain separate.
The planner never starts a GPU test as a side effect of inspecting a diff.

## Review a plan

First inspect an uncommitted change with `python3 scripts/agent_scope.py --base origin/main`.
After creating a reviewable commit, generate a plan for its immutable source:

```bash
python3 scripts/quality.py validation generate --base origin/main --head HEAD \
  --output .scratch/validation/plan.json
```

For a stacked PR, use its actual parent branch as `--base`. Also inspect the cumulative diff
against main when reviewing the integrated stack. The plan records the resolved commits, changed
symbols, risk facets, owner rules, required obligations, selected and unselected cases, reasons,
and hashes of the catalog, policy, tests, configurations, and dependency declarations.

`config/validation/selection.yaml` owns coverage policy. The catalog owns executable cases and
cost estimates. Mandatory interactions are covered before cost selection. Backend-specific
obligations prevent a cheap CPU or DDP case from substituting for FSDP2 state isolation. Unknown
source owners and incomplete source collection remain explicit uncovered review obligations.
They require inspecting the owner and adding real coverage; they are not silently waived.
A changed cataloged test is itself selected, even if a cheaper case shares its coverage label.

`--case CASE_ID` and `--require OBLIGATION_ID` add local review requirements. Add durable
requirements to the checked-in policy when they must also apply to CI. There is no option to
remove an inferred requirement. Proven comment/docstring and literal console-log edits retain
the scope classifier's fast path. A parsed TOML comparison also recognizes a change to only
`project.version`; dependency or build-system changes do not receive that release-metadata
exemption.

Historical changes can be routed against the current policy without presenting them as newly
executed evidence:

```bash
python3 scripts/quality.py validation replay --base BASE_COMMIT --head OLD_COMMIT \
  --output .scratch/validation/history.json
```

A replay is explicitly marked non-executable. It keeps the historical source identities separate
from the current policy commit.

## Execute and inspect results

```bash
python3 scripts/quality.py validation run-cpu --plan .scratch/validation/plan.json \
  --output .scratch/validation/cpu.json --logs .scratch/validation/logs
python3 scripts/quality.py validation report --plan .scratch/validation/plan.json \
  --results .scratch/validation/cpu.json --expected-commit "$(git rev-parse HEAD)" \
  --output .scratch/validation/report.json
```

The CPU runner requires a clean checkout at the planned commit. It executes selected local
assertion cases when their dependencies are installed, records the actual environment, and
hashes their logs. Pytest cases also need a nonempty JUnit report without skips. A missing Torch
installation in the independent quality environment produces a blocked result, not a successful
CPU test. Run the same tool with the intended training environment for those cases.

GPU and distributed cases use their catalog commands and the [manual result schema](validation.md).
Keep assertion logs under `.scratch/` and record actual backend/plugin identity, every completed
rank, numerical observations, and any required overlap trace. A zero exit code or a backend label
alone does not supply those observations. The report accepts multiple result files; conflicting
pass/fail observations are rejected. Passing evidence from a complete environment can fill a
case blocked only by a missing dependency in another environment.

The final report also requires the clean planned checkout and distinguishes `pass`, `fail`,
`blocked`, and `not_selected`. Every required obligation must be covered by a selected passing
case. Reports re-create the canonical plan and check log contents, so editing a plan to deselect
GPU work or replacing a log cannot create a pass.
Exact-source evidence remains the default. An ancestor bundle can also cover unchanged execution
inputs after edits limited to existing Markdown under `roadmap/` or `guidance/`, or the CPU evidence
tools `scripts/validation_plan.py`, `scripts/validation_evidence.py` and their regression file
`tests/scripts/test_validation_planner.py`. The gate checks Git ancestry, the same comparison base,
file modes, the complete remaining tracked tree, original canonical plan and log hashes, and
unchanged case definitions and test/config artifacts. Additions, deletions, runtime, dependency,
policy, workflow, and native test changes reject reuse. A changed input declared by a reused case
also rejects reuse, even inside the allowlist.

Both `harness-contracts` and `repository-consistency` must pass freshly on the current head;
ancestor results cannot satisfy them or override a current failure. The original bundle and its
`tested_commit` stay unchanged. The report records `evidence_reuse` with the actual tested commit,
original plan digest, accepted target, changed paths and reused case IDs. This is acceptance of
unchanged inputs, not a claim that GPU tests executed again on the new head. Artifact discovery
prefers an exact match, then checks at most 19 earlier first-parent commits in the same PR range.

## CI and externally executed GPU evidence

The stable `cpu-contracts` required check generates the same plan for the PR's actual head,
executes available CPU cases, and applies the same final report gate. It preserves the plan,
results, and logs as an Actions artifact. Missing selected GPU evidence fails this check explicitly;
it does not become a skipped green job. The separate `quality` check continues to enforce hooks.

To supply already executed evidence, first create a bounded bundle:

```bash
python3 scripts/quality.py validation bundle --plan .scratch/validation/plan.json \
  --results .scratch/validation/gpu.json --output .scratch/validation/bundle.json
```

Dispatch `quality.yml` on the exact PR branch using GitHub CLI's JSON stdin input.
Its `bundle` input is the JSON file's text. The workflow verifies the branch head, re-generates the
canonical plan, checks observations and log hashes, and uploads a commit-and-plan-specific
artifact. Re-run the PR's `cpu-contracts` job after upload. The gate reads only matching successful
manual-evidence workflow artifacts and validates their contents again. It never executes shell
commands received in a bundle. A recorded GPU result remains operator-supplied evidence;
reviewers should inspect the assertions and logs, not infer that hosted CI ran the GPU job.

This first CI integration consumes the default generated plan. Keep reviewer-added `--case` or
`--require` selections in a separate local plan and report; their different plan hash does not
replace evidence for the default CI plan. Reusable mandatory coverage belongs in the reviewed
selection policy so both local and CI generation select it.

The bundle limit is 60 KB, including concise assertion logs. Keep large traces in the original
experiment archive and select the meaningful assertion artifact for the bundle. The evidence job
extends the already registered quality workflow so its branch version can be dispatched while
reviewing a stack. It only records evidence; its skipped jobs use different names so they cannot
satisfy or replace the required PR checks.

Merging or rebasing creates a new source identity. Re-run selection and the required evidence
for that commit before recording a release qualification; a green check on an earlier PR head
does not qualify the integration commit automatically.

Keep branch protection requiring both `quality` and `cpu-contracts`; the latter's name is retained
for compatibility, although its final gate now includes all selected evidence levels. Repository
administrators configure required checks; this implementation does not mutate protection rules.

## Explicit incremental acceptance

When an authorized reviewer explicitly limits a restack or TP extension to incremental tests,
record that decision separately from test qualification. This is an opt-in evidence-dispatch
protocol, not automatic source equivalence or a way to mark historical GPU cases as current passes.
Risk classification, prior passing tests, and permission to edit code do not authorize this scope;
the reviewer must have an explicit instruction limiting the required new execution.
The canonical plan, case statuses, missing obligations, historical bundles, and original tested
commits stay unchanged. A report can therefore remain `status: blocked` while its separate
`acceptance.status: accepted_incremental` allows the CI job to succeed. This does not establish
full current-head qualification or release qualification.

The bounded JSON manifest has `schema_version: 1`, `kind: reviewed_incremental`, the repository,
exact target `base_commit`, `tested_commit` and canonical `plan_sha256`, plus:

- `baseline`: original base/head/plan digest, original bundle byte SHA256, and its successful
  `quality.yml` dispatch `artifact_id` and `run_id`. The original canonical plan is reconstructed
  and every supplied original assertion and log is verified. Each claimed historical GPU case
  must explicitly pass; missing historical CPU entries remain visible as blocked and are never
  manufactured (current CPU evidence is mandatory). Any supplied historical failure rejects the review. Nonancestor sources and changed bases are
  permitted only through this explicit review, without changing the baseline identity.
- `delta`: the complete ordered output of `validation_acceptance.delta(root, old_head, new_head)`.
  Every changed tracked path binds its before/after mode and SHA256, including additions/deletions.
  Unsupported file types, omitted paths, and later edits reject the review.
- `review`: `decision: approved_incremental`, `scope` (`restack_only`, `model_tp_extension`, or
  `validation_or_documentation`), reviewer, verbatim `authorization_quote`, and concrete reason.
  The dispatching repository writer attests this review; the tool does not authenticate a quoted
  user statement or infer approval from an ordinary source change.
- `required_fresh_cases` and `historical_cases`: disjoint lists covering every selected case.
  Every CPU selection needs actual target-head evidence. New GPU cases and all declared incremental
  witnesses also require target-head passes. A model TP extension must name a GPU witness.
  Historical cases must have complete passing proofs in the baseline with unchanged command, backend, topology,
  oracle and coverage contract. Their model-source changes are explicitly reviewed, never claimed
  equivalent by the tool. A current failure always rejects acceptance.

When the historical cases were originally qualified in different PRs, an optional
`supplemental_baselines` list may supply up to four additional original sources. Each entry
contains exactly `baseline` (the same six identity/artifact fields) and `delta` (that source's
complete reviewed change to the target). Every source passes the same source, plan, dispatch,
bundle, assertion and failure checks independently. Each historical case must have an explicit
PASS with an unchanged declared contract in at least one source. The report lists the actual
source head(s) for each case and preserves each source's own qualification status. Baselines are
never merged into a manufactured current-head result, and no source-equivalence claim is implied.
Duplicate source heads, extra entries, omitted deltas and missing case proofs reject acceptance.
For offline validation, repeat `--supplemental-bundle ORIGINAL-BUNDLE.json` in manifest order;
if supplied, the local list must cover every supplemental entry. Current CPU and incremental GPU
requirements are unchanged.

Create the current partial evidence bundle normally. Dispatch it and the acceptance manifest
**separately** using the existing `bundle` input and the optional `acceptance` input to avoid
combining their payload sizes. Each is limited to 60 KB. The manifest artifact is named
`validation-acceptance-HEAD-PLAN`; recording it validates source/review/baseline only and does not
assert that the fresh tests have passed. CI fetches both artifacts and enforces the fresh proofs
when reporting. An absent manifest retains the strict default gate. No shell commands are taken
from the manifest. Old commits outside fetched branch history are retrieved by their validated
full commit IDs from the checkout's fixed `origin`.

For an offline review with the original bundle retained locally:

```bash
python3 scripts/quality.py acceptance record --plan PLAN.json \
  --acceptance ACCEPTANCE.json --repository OWNER/REPO --baseline-bundle ORIGINAL-BUNDLE.json
python3 scripts/quality.py validation report --plan PLAN.json --results CURRENT-CPU.json \
  --bundle CURRENT-BUNDLE.json --expected-commit FINAL_HEAD --acceptance ACCEPTANCE.json \
  --repository OWNER/REPO --baseline-bundle ORIGINAL-BUNDLE.json --output REPORT.json
```

Local validation verifies bundle contents but cannot authenticate GitHub artifact ownership;
the dispatch and CI re-check the referenced artifact/run against the repository API. Any changed
head, base, plan, reviewed file delta, test contract, or required incremental witness requires a
new explicit review. Preserve original failures and the distinction between historical proof,
actual incremental execution, and the acceptance decision in PR prose.
