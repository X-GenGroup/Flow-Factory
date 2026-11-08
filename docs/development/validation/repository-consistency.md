# Repository consistency checks

P01c extends the [quality workflow](quality.md) with static checks of repository contracts. Use
`make quality` for the usual PR-range checks, or run only consistency checks in the same read-only
snapshot:

```bash
make consistency
make consistency QUALITY_ARGS='--base origin/main --head HEAD'
make consistency QUALITY_ARGS='--all-files --head HEAD'
```

New files must be staged to enter the tracked-file inventory. The default command includes
unstaged edits to tracked files; `--head` checks only the specified commit. The native pre-commit
hook checks staged content using pre-commit's temporary stash and restores unstaged work. Both
paths execute `scripts/check_repository.py` with the active checkout's pinned quality environment.
No Flow-Factory import, Torch installation, model download or network service is needed.

## Checked contracts

| Check | Authoritative input | Failure examples |
|---|---|---|
| Built-in registries | Literal dictionaries in the nine registry modules | Missing module/symbol, duplicate lookup key, unsupported dynamic declaration |
| Training argument parity | Trainer and training-argument registry keys | A built-in trainer has no matching argument configuration |
| Example references | Explicit fields in tracked `examples/**/*.yaml` and `*.yml` | Unknown model/trainer/reward/accelerator/logger/enhancer/optimizer identifier; missing launcher config |
| Local documentation | Root Markdown, `guidance/`, `roadmap/`, `.agents/`, `examples/`, nested `AGENTS.md` | Missing tracked target or Markdown anchor, including incoming links from unchanged documents |
| Roadmap graph | Work-item headings, stable anchors, Status and Depends on fields in `roadmap/milestones/` | Duplicate ID, unknown dependency, wrong dependency link, cycle or undocumented status |
| Direct dependency boundaries | AST imports in `contracts/` and `scheduler/` | Contracts import concrete runtime owners; scheduler imports trainers |
| Harness structure | Existing `validate_agent_harness.py` | Invalid routing, references, permissions or startup budget |
| Docstring hygiene | Existing `audit_docstrings.audit_source` error rules | Stale documented argument, incorrect path header or other existing enforced documentation error |

The registry list records declaration locations, not a second catalog of supported models or
algorithms. Entries come from source on each run. Add a checker specification and a failing fixture
if introducing another registry. Trainer and argument registries are checked together.

Lookup normalization follows the current runtime: identifiers are lowercased, with no automatic
underscore/hyphen conversion or whitespace stripping. Scheduler class-name keys remain case
sensitive. The logger's `none`/null behavior is explicit. Multiple keys may deliberately point to
one class. Explicit imports, re-exports, module aliases and simple symbol aliases are followed;
cycles, missing symbols and non-static values fail. Built-in dictionaries must remain statically
enumerable; runtime registration functions and decorators are not executed by this check.

An external `package.module.Class` is allowed where the runtime accepts a Python path and is
reported as **UNVERIFIED**, with a requirement for plugin/runtime evidence.
The hook displays these notices even when its static checks pass. A dotted path under
`flow_factory` is checked against tracked source. Optimizer identifiers use their registry only.
Symbol existence is not proof that imports, inheritance, initialization or runtime behavior work.

Recipe checks preserve the current list/single-mapping reward and acceleration forms. Prompt
enhancement examples use their own schema. They inspect explicit registry references and basic
container shapes; they do not construct Arguments, reject all unknown extension fields, expand
runtime defaults, or decide model/backend compatibility. P02/P04 own the reusable test catalog and
semantic evidence selection.

Markdown is parsed with pinned `markdown-it-py`; fenced/inline code and comments are not links.
Reference links, images, HTML links/anchors, percent-encoded paths and repeated heading anchors are
handled. Remote links are not fetched. Only Markdown fragments are checked as heading/HTML
anchors; fragments in other file formats remain the target format's responsibility. Roadmap
**linked** work-item IDs define dependency edges; prose may qualify separate release conditions.

Dependency checks guard two existing directions, not a proposed whole-framework layering:
`contracts` may refer to other contracts but not concrete Flow-Factory runtime owners; `scheduler`
must not depend on `trainers`. The latter preserves the import-cycle regression documented in
`tests/dependencies/test_module_import_isolation.py`. Static checks inspect direct imports and
literal absolute dynamic imports. They do not prove transitive import isolation or replace that
suite's fresh-interpreter runtime tests.

## Selection, cost and limits

The hook always receives a chance to run, including deletion-only commits. It selects checks from
the complete changed-path set rather than just files still present on disk:

- Source, example or launcher-config changes recheck all registry targets and all examples, so a
  rename/deletion cannot leave an unchanged recipe pointing at removed code.
- Any tracked path can be a local link target; documentation links and the roadmap graph are
  checked globally. A changed heading can invalidate links in unchanged documents.
- Changes to dependency owners select both existing dependency checks. Harness input changes
  select the existing harness validator. Source changes audit only changed owned Python files.
- Changes to consistency/quality infrastructure or an explicit `--all-files` run select the full
  inventory. Full docstring enforcement also remains in the CPU suite.

`quality.py` passes the comparison paths to the hook through a process-local environment value.
This preserves deleted paths and base-to-head changes even when the disposable snapshot's index
equals HEAD. Direct pre-commit invocations combine supplied paths with staged deletions. The
checker enumerates tracked paths, reads regular source files through the existing bounded reader,
and refuses unreadable/oversized input instead of silently passing it. Limits are 1 MiB per source
file and 32 MiB of source text per run. Untracked checkpoint trees are not traversed or read.

Initial inventory on the P01c base (`7dd84ab8`, merged #279) found no failures under these rules;
there is no suppression baseline to maintain. The existing
[formatting debt inventory](quality-baseline.md) remains separate. Fix new consistency failures at
their owner. Any future unavoidable exception needs a reviewed exact scope, owner, rationale and
removal condition; do not add directory-wide exemptions to make CI green.

The existing `quality` CI check runs the same configured hook as a local commit. `cpu-contracts`
runs the negative fixtures, Git integration tests and full-inventory regression. These remain the
two required-check candidates; a third job solely repeating the same consistency scan is not
needed. GPU evidence is independent of these checks.

When changing these tools, measure warm small-diff latency separately from environment setup.
The P01c Python 3.12 measurements (median of three warm scanner runs on the development host) were:

| Input | Selected work | Seconds |
|---|---|---|
| One documentation file | Global local links and roadmap graph | 0.49 |
| `utils/dist.py` | Registries/examples, local links and changed-file docstrings | 1.56 |
| Harness rules | Local links and the existing harness validator | 0.59 |
| Full inventory | All five checks, including all owned source docstrings | 4.42 |

These are scanner timings, excluding installation, Git snapshots and other formatting hooks. The
inventory covered nine registries (78 entries), 102 examples and approximately 6 MB of source
text. Run times vary by host; use these as a baseline rather than a performance guarantee.

If a global check pushes ordinary commits beyond a few seconds, move that check to the explicit
command and CPU CI while retaining a fast hook equivalent or a documented narrower scope. Do not
silently remove enforcement. Inspect `--json` output from `scripts/check_repository.py` in the
quality environment for selected checks, source bytes, findings and elapsed time.
