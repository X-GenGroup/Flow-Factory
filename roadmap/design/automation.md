# Developer automation: hooks and CI

This proposal adds fast local feedback and reproducible PR checks to the early
[P01 work](../milestones/v0.3.md#p01). It complements the [validation design](validation.md): hooks
catch inexpensive consistency failures; the harness selects semantic and GPU evidence. This
document does not install hooks, change repository settings, or enable a workflow.

## Starting point

Initially inspected on **2026-11-04**, at roadmap commit `50b747e4` based on main `2a115133`.
Rechecked after #274/#270 at main `55dcd694`: those merges add reward/TP behavior and tests, but
do not install the hook/CI workflow proposed here.

- [pyproject.toml](../../pyproject.toml) defines Black/isort with a 100-character line length, but
  no separate quality-tool environment. Installing the project normally pulls its training stack.
- No tracked pre-commit configuration, GitHub workflow, Makefile, `.editorconfig`, or root
  `.gitattributes` was present. This does not establish the state of server-side branch rules.
- [Harness validation](../../scripts/validate_agent_harness.py),
  [scope classification](../../scripts/agent_scope.py),
  [docstring auditing](../../scripts/audit_docstrings.py), and focused tests already exist. Reuse
  their authoritative behavior rather than implementing another copy in CI shell snippets.
- Registries contain lazy Python paths; examples and documentation can become stale when a key,
  module, or class is renamed. Importing every model to catch that would burden a basic hook with
  optional runtime dependencies.

## What the reference projects actually do

The links below pin the inspected revisions. These are examples to learn from, not dependencies or
policies imported into Flow-Factory.

| Project and inspected sources | Observed pattern | Application here |
|---|---|---|
| UniRL: [hooks](https://github.com/Tencent-Hunyuan/UniRL/blob/0d07b80b952a8ea3ae6c4d2df3a42a3628cec096/.pre-commit-config.yaml), [lint CI](https://github.com/Tencent-Hunyuan/UniRL/blob/0d07b80b952a8ea3ae6c4d2df3a42a3628cec096/.github/workflows/lint.yml), [recipe checker](https://github.com/Tencent-Hunyuan/UniRL/blob/0d07b80b952a8ea3ae6c4d2df3a42a3628cec096/lint/check_recipe_targets.py) | Basic hygiene and Ruff plus import-free recipe-target and dependency-boundary checks; explicit vendored-data exclusions | Validate FF registry/recipe references without loading Torch or model weights; protect upstream and prompt fixtures from automatic rewriting |
| Diffusers: [Makefile](https://github.com/huggingface/diffusers/blob/35087a644038eb531f48581708d4763044cb4852/Makefile), [PR checks](https://github.com/huggingface/diffusers/blob/35087a644038eb531f48581708d4763044cb4852/.github/workflows/pr_tests.yml), [test fetcher workflow](https://github.com/huggingface/diffusers/blob/35087a644038eb531f48581708d4763044cb4852/.github/workflows/pr_test_fetcher.yml) | No root pre-commit config at this revision; shared quality/style commands, modified-file fixup, repository consistency and test-selection tools. The fetched-test workflow is manually dispatched | One local/CI command surface and explicit consistency checks; connect selection to P04 without assuming upstream selects tests automatically on every PR |
| verl: [hooks](https://github.com/verl-project/verl/blob/745432653d15af14954af1e055c8aeed72afa064/.pre-commit-config.yaml), [hook CI](https://github.com/verl-project/verl/blob/745432653d15af14954af1e055c8aeed72afa064/.github/workflows/pre-commit.yml), [sanity CI](https://github.com/verl-project/verl/blob/745432653d15af14954af1e055c8aeed72afa064/.github/workflows/sanity.yml) | Ruff/mypy plus generated-config, device-API, documentation, license and test-structure checks; separate sanity and backend test workflows | Encode established repository contracts, with each check placed according to dependency and execution cost |
| verl-omni: [hook CI](https://github.com/verl-project/verl-omni/blob/ac96451566a71d444b751e6e30dc94b1f88b459e/.github/workflows/pre-commit.yml), [GPU groups](https://github.com/verl-project/verl-omni/blob/ac96451566a71d444b751e6e30dc94b1f88b459e/.github/workflows/gpu_smoke.yml), [label invalidation](https://github.com/verl-project/verl-omni/blob/ac96451566a71d444b751e6e30dc94b1f88b459e/.github/workflows/drop-ci-labels.yml) | Label-driven PR CI, distinct core/omni/diffusion GPU groups, and label removal on new commits. Even its hook workflow installs parts of the runtime stack | Group expensive jobs by capability and bind evidence to revisions; keep FF's cheap PR checks automatic and independent of that runtime installation |
| vLLM-Omni: [hooks](https://github.com/vllm-project/vllm-omni/blob/37d6299491e474d6a8b34942873acab9a0da1819/.pre-commit-config.yaml), [CI](https://github.com/vllm-project/vllm-omni/blob/37d6299491e474d6a8b34942873acab9a0da1819/.github/workflows/pre-commit.yml), [test-mark checker](https://github.com/vllm-project/vllm-omni/blob/37d6299491e474d6a8b34942873acab9a0da1819/tools/pre_commit/check_test_marks.py) | Formatting, workflow checks and project-specific rules; test level/hardware metadata; gradual handling of historical lint debt. Several full-tree hooks are skipped in CI while changed test marks have a separate check | Use test capabilities/topology rather than fixed GPU SKUs; migrate debt explicitly so a local rule is not silently absent from PR enforcement |

## Execution boundaries

| Entry point | Default work | Cost and behavior |
|---|---|---|
| `pre-commit` | Staged-file format, syntax, merge-marker and basic file checks; relevant fast repository validators | Warm-run target of a few seconds for ordinary small diffs; measure rather than promise a fixed time. No models, GPU, reward services or full test suite |
| Optional `pre-push` or an explicit check command | Branch-diff plan and selected small CPU contract tests | Read-only; clear command and timeout. Do not unexpectedly format committed files or launch distributed/GPU jobs |
| PR CI | Same quality rules, repository consistency, selected CPU tests and an evidence summary | Automatic for every PR update, including forks and docs-only PRs; expensive jobs selected inside the workflow |
| Selected GPU jobs | Q1-Q3 cases required by the diff and support claims | P02/P04 choose valid fixtures and topology; execution stays within authorized resource scope |
| Scheduled/manual jobs | Broader compatibility, external-link checks, capacity and performance | Separate signals with explicit ownership; not hidden costs of a commit |

Install only `pre-commit` by default. Leave `pre-push` opt-in initially, and do not install
`post-checkout`, `post-merge`, or `post-commit` actions that fetch models, mutate dependencies or
create commits. Validate the existing `[modules] type: description` PR-title convention in CI.
Strict commit-message checks and automatic sign-off insertion are not part of the first rollout;
the repository does not currently require them, and intermediate commits need not be PR titles.

## Initial hook set

| Check | Initial scope and choice |
|---|---|
| Formatting/import order | Keep pinned Black and isort with the current `pyproject.toml` settings; one formatter and one import-order authority |
| Python syntax and obvious mistakes | AST parsing and debug-statement detection. Evaluate a small pinned Ruff error-only rule set in P01c without also enabling Ruff format/import sorting |
| Structured files | Safe YAML, TOML and JSON syntax; actionlint for workflow changes. Treat format syntax separately from FF configuration semantics |
| Text hygiene | EOF/newline/whitespace checks on owned source and docs; retain intentional Markdown hard breaks and fixture/prompt bytes |
| Accident prevention | Merge markers, case collisions, private-key patterns and unexpected large added files; choose a size limit from actual assets rather than copying another project's number |
| Harness/docs | Existing harness validator when its inputs change; local documentation links and roadmap IDs/dependencies; existing docstring rules when Python documentation changes |

Formatting scopes exclude the Diffusers submodule, generated outputs and identified upstream or
byte-sensitive fixtures. Artifact checks still inspect staged additions: a formatting exclusion
must not let a force-added model checkpoint bypass the large-file check. Enumerate Git paths,
including deletions/renames where relevant, rather than recursively reading all untracked files.

Do not apply a new global formatter to make the initial workflow pass. Inventory the existing
Black/isort baseline; enable the same changed-file or clean-directory policy locally and in CI.
List remaining debt with owners and a removal condition. A later Ruff migration, if worthwhile,
gets a dedicated mechanical PR and removes the replaced tools atomically. Retain the current
license-header convention and documentation style rather than copying another project's SPDX or
one-line-docstring rules.

## Flow-Factory consistency checks

P01c's implementation is documented in
[repository consistency](../../guidance/repository-consistency.md). It uses small, deterministic
checks tied to actual repository contracts:

1. Parse built-in registry declarations and verify module/symbol references and duplicate keys.
   Check example model/trainer/reward/acceleration identifiers against the relevant registry using
   its real normalization rules. Handle aliases and re-exports explicitly. External plugin paths
   remain a supported extension mode and require their own environment-backed contract evidence;
   static analysis must not claim to have validated them.
2. Validate examples structurally without constructing every training Arguments object or loading
   model weights. Deeper configuration/optional-dependency checks belong in the selected CPU job.
3. Reuse harness validation and docstring auditing; add local Markdown target/anchor and roadmap
   dependency validation as one maintained helper with focused negative fixtures.
4. Enforce only dependency directions that are explicitly owned and already verified. Historical
   import-cycle regressions need import tests as well; an AST scan is not numerical or runtime
   correctness evidence. Start type checking with a proven-clean contract/helper subset if useful.
5. Validate P02's test-catalog metadata once it exists: what is covered, CPU/CUDA, backend, logical
   mesh, placement, required assets and oracle. Share this source with P04 rather than copying a
   second resource matrix into pytest markers or CI YAML.

Trigger a full registry-reference pass when a registry or referenced Python module changes, not
only when an example changes. A rename/deletion can break unchanged examples. Scope genuinely
global checks by their inputs and keep them read-only; moving them to every commit is conditional
on measured cost.

## One toolchain, usable from a worktree

Proposed implementation files are `.pre-commit-config.yaml`, a small `Makefile`,
`requirements/quality.txt`, scoped `.editorconfig`/`.gitattributes`, and thin workflows under
`.github/workflows/`. Use `scripts/` for FF-specific validators. These are future files, not
commands provided by this roadmap PR.

- Pin hook revisions in the hook config and the small runner/test dependencies in the quality
  environment. Keep rule settings in `pyproject.toml` and do not duplicate formatter versions or
  rules in shell, CI and agent prompts. Cache environments by their configuration identity.
- Quality bootstrap must work without `pip install -e .` or the training dependencies. Prepare
  hook environments once; normal warmed checks need neither downloads nor a running GPU.
- Provide quality/check commands that preserve the contributor's checkout, plus a separate
  explicit format/fix command. Use check modes where available and a disposable checkout for
  fix-only utilities. Local hooks may format the selected staged files through pre-commit's
  normal staging behavior; show the diff for review rather than automatically staging additional
  files or committing it.
- Exercise partially staged files, spaces in paths, renames/deletions, a new branch without an
  upstream, and two linked worktrees. Use Git's root/path commands rather than assuming `.git` is
  a directory. Resolve configuration from the active worktree, not an absolute path to the first
  checkout. Handle older branches without a hook config explicitly during rollout.
- Git configuration is shared across worktrees by default. Bootstrap must preserve existing hooks
  and avoid global `core.hooksPath` changes or silently replacing another checkout's setup. Record
  installation/recovery instructions for human and agent contributors.

The [pre-commit stage/staging documentation](https://pre-commit.com/#supported-git-hooks) and
[Git worktree details](https://git-scm.com/docs/git-worktree#_configuration_file) define these
mechanics; verify them against the tool versions selected during P01a.

## PR checks and evidence

Keep lightweight workflows automatic on open, synchronize, reopen and ready-for-review events;
metadata validation also handles title edits. Use shared implementations for local and CI checks.
CI must not write back to the PR branch. A fix-only hook may change its disposable checkout, but
must then fail the check and report the repair diff rather than committing or pushing it.

The implemented stable checks are `quality` (including P01c repository consistency) and
`cpu-contracts`. Consistency shares the existing hook/CI implementation; a separate
`repo-consistency` job would currently repeat the same scan and environment setup. Add a final
`validation-gate` when its corresponding planner implementation lands. Run
lightweight planning on every PR. A final always-evaluated gate must distinguish intentionally
unselected jobs from required failed, cancelled, skipped or missing evidence. Avoid a required
workflow disappearing behind a top-level path filter and leaving the PR pending indefinitely.

For each plan record the comparison base, PR head, actual tested commit (including a synthetic
merge commit when used), test/config identity, dependencies and selected obligations. New commits
invalidate results by default; narrowly justified reuse follows P04. A GPU label requests work on
a specific revision, not a permanent approval for future pushes. Labels may add evidence but may
not silently waive a required obligation.

Use ordinary read-only PR jobs for contributor code. Metadata-only privileged workflows must not
execute PR code. Pin external actions, cancel superseded PR runs, set timeouts and publish readable
summaries/artifacts. GPU/self-hosted execution remains explicitly authorized. Document which stable
checks maintainers should require on main after the workflows are healthy; YAML alone does not
activate server-side branch protection. See [GitHub's required-check guidance](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches).

## Early delivery sequence

- [P01a](../milestones/v0.3.md#p01a): local hook bootstrap and minimal matching CPU CI, preserving
  current formatting. Start immediately; it does not depend on TP, CP or a GPU runner.
- [P01b](../milestones/v0.3.md#p01b): migrate risk/evidence policy and provide the manual selected
  plan, building on that executable entry point.
- [P01c](../milestones/v0.3.md#p01c): add the focused repository-consistency checks and documented
  debt handling. It can progress alongside P01b after P01a.
- P02 supplies test/resource metadata; P04 connects the planner, CPU selection and GPU evidence
  gate. All P01 children are part of v0.3.0; no additional release tag is required just for hooks.
