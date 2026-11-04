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
Reuse is deliberately limited to the identical source commit, plan hash, and test/config hashes.
A new head needs fresh evidence, even if its description says it is a documentation-only change.

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
