# Fast quality workflow

Quality tooling is independent of the training environment. Use Python 3.10+ and Git on Linux or
macOS (x86_64 or arm64); no Torch, model weights, GPU, reward service, or editable project install
is needed.
Black and isort retain the settings in `pyproject.toml`. Python tool versions, including transitive
dependencies, are pinned in `requirements/quality.txt`; official actionlint release archives and
SHA-256 values are pinned in `requirements/actionlint.json`. Pre-commit stays on the Python-3.10-compatible 4.6 series.

## Setup and everyday commands

Run from the repository root, including a linked worktree:

```bash
make quality-env                 # Install the isolated Python tools; network needed once
make quality-warm                # Prepare the checksummed actionlint binary once
make hooks                       # Optional: install the native pre-commit hook
# make hooks-pre-push            # Optional: also check committed changes before pushing

git add <intended-new-files>      # New files become visible to checks after staging
make quality                     # Read-only, changed files relative to origin/main
make consistency                 # Read-only repository contracts for that same range
make format                      # Explicitly fix those working files; never stages them
make test-cpu                    # Harness/docstring and real Git/hook integration tests
```

To select a different interpreter for bootstrap, use `make quality-env PYTHON=/path/to/python3`.
Warming tools downloads the small official actionlint binary; no Go installation is needed.
Ordinary warmed commits perform local checks only.
The first install is not part of commit-latency measurements. `make quality` returns failure and a
repair diff if a formatter would change a file. Review fixes and stage them yourself. The explicit
`format` command can also return failure after applying fixes; rerun the check after reviewing them.

`quality` compares the merge base with `origin/main` against the current tracked working files,
including staged additions and unstaged tracked edits. It does not require a branch upstream and
does not walk arbitrary untracked/checkpoint directories. Fetch `origin/main` when the base is
stale. Override it or check an immutable commit explicitly:

```bash
make quality QUALITY_ARGS='--base origin/main --head HEAD'
make quality QUALITY_ARGS='--all-files'  # Inventory existing debt; never fixes the working tree
```

Checks execute the same `.pre-commit-config.yaml` as hooks, in a disposable local Git checkout under
`.scratch/`. The original index, staged/unstaged content and existing hooks remain unchanged. Git
hooks are disabled for the snapshot's checkout/staging operations using command-local settings.
Formatting scopes and tool settings are not reimplemented in CI. Optional pre-push checks use the
pushed commit range, falling back to `origin/main` for a new remote branch; they do not format the
checkout, test an unstaged repair instead of the pushed commit, or launch GPUs. Run `make test-cpu`
explicitly for the small CPU suite.

## Staging, worktrees, and existing hooks

Pre-commit owns partial-staging behavior: it temporarily stashes unstaged edits, checks staged
content, and restores the edits. If an automatic repair conflicts with the unstaged patch, it rolls
back repairs and restores the work. A failed hook does not stage files or create a commit. Avoid
simultaneously editing the same files while a commit/check is in progress.

The environment lives under `git rev-parse --git-common-dir`, in `ff-quality/<requirements-hash>/`.
Bootstrap is serialized across worktrees. Each checkout resolves its own requirements and hook
configuration; the generated hook's interpreter path survives deleting the worktree that installed
it. Re-run `make quality-env` when switching to requirements that have not been prepared; run
`make quality-warm` after actionlint pin changes. The actionlint binary is cached by its archive
checksum under the same common Git directory.
Its installer extracts only the regular `actionlint` member after checksum verification.

Installation is explicit. Native pre-commit migration mode preserves an existing hook as
`pre-commit.legacy` and runs it alongside the new hook. No global Git setting is changed. If
`core.hooksPath` already belongs to another manager, setup refuses to replace it: integrate
pre-commit with that manager using its documented mechanism, or continue using `make quality`
without installing hooks. Do not use `--overwrite` or change a shared `core.hooksPath` casually.
Git hooks are shared by linked worktrees. Older branches without `.pre-commit-config.yaml` are
allowed to commit via `--allow-missing-config`; CI on branches with this workflow still enforces
checks. To uninstall, use the quality environment's `python -m pre_commit uninstall` (and
`--hook-type pre-push` if installed); native uninstall restores the preserved legacy hook. Coordinate
that shared-hook removal with other worktree users. Do not delete the common environment while a
shared hook still points to it.

## Check scopes and existing debt

| Check | Scope |
|---|---|
| Black/isort | Changed Python files; current settings, no repository-wide reformat |
| Python syntax / debug statements | Changed Python; debug checks share formatter ownership exclusions |
| YAML/TOML/JSON syntax | Changed structured files; no training config/model construction |
| EOF/whitespace repair | Owned Markdown, workflow files and automation text; Markdown hard breaks remain |
| Merge markers, case collisions, private keys | Changed relevant files, independent of formatting exclusions |
| File size / generated artifacts | All changed regular files, including force-added ignored files |
| Repository consistency | Registry/example references, local docs/roadmap, established dependency boundaries; global incoming references include unchanged files |
| Harness/docstring structure | Existing validators selected by their inputs; full inventory also in the CPU CI suite |
| actionlint | Changed workflows, pinned actionlint; optional ambient shellcheck/pyflakes disabled |

The size ceiling is 12 MiB: the largest existing tracked asset is the 10,817,461-byte GenEval2
metadata file. Generated checkpoint extensions and output directories are rejected even below that
ceiling. Do not hide model artifacts in a formatting exclusion. If a legitimate larger asset is
necessary, review its storage/fixture ownership and update policy with evidence.

Formatters exclude the Diffusers submodule, Bagel/SenseNova upstream `modeling`/`data` trees,
Qwen-Image-Bench's two vendored aggregation files, and explicit `fixtures`/`snapshots` directories.
Text fixers do not rewrite datasets, YAML prompt blocks or Python string literals. Syntax and
accident checks can still inspect these files. Tests cover preserved prompt bytes and Markdown
hard breaks. See the [baseline inventory](quality-baseline.md) for the 26 pre-existing Black/isort
debt files and their cleanup owners. An edited debt file must satisfy the same rule locally and
in CI; there is no CI-only skip list.

## CI and maintainer setup

Every PR update (including title edits, forks and documentation-only changes) receives stable
`quality` and `cpu-contracts` checks. `quality` uses Python 3.12 and checks the tested merge commit
against the PR comparison base; `cpu-contracts` uses Python 3.10. Pushes to main are also checked.
Jobs have read-only repository permissions, pinned actions, timeouts and superseded-run
cancellation. They never commit or push repairs, execute privileged `pull_request_target` code, or
request GPU runners. PR titles use the existing `[modules] type: description` convention; commit
messages are not gated. Dependency updates must retain the Python 3.10/3.12 checks.

After this workflow is green on main, a maintainer should require **quality** and **cpu-contracts**
in the repository's main-branch ruleset/protection settings. Creating the workflow does not enable
server-side required checks. Keep these workflows free of top-level path filters so a required
check never disappears for a documentation change. First-time fork workflows may still need the
repository's normal maintainer approval; never use privileged PR execution to bypass that control.

These entry points implement [P01a](../roadmap/milestones/v0.3.md#p01a) and
[P01c repository consistency](repository-consistency.md). The CPU suite also checks P01b scope
routing, selected-plan validation and the retained historical campaign reader. Follow
[validation plans](validation.md) for semantic evidence; fast hook success does not prove training
or distributed correctness. P02/P04 own reusable fixtures and automated selection.
