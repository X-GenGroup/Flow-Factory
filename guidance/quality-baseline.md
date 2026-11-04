# Formatting baseline

Measured on 2026-11-04 against main `98567df971edc44c7c301d549c71bd0fdd32c017`, with Black
26.10.1 and isort 9.0.2 using the unchanged `pyproject.toml` settings and the P01a ownership
exclusions. There are 12 Black failures and 18 isort failures across 26 distinct existing files.
This is an inventory, not an executable exemption list; changed files are checked identically
in local hooks and CI. P01a does not reformat these unrelated files.

| Path | Existing failure | Cleanup owner |
|---|---|---|
| `dataset/multi_ref_image/prepare.py` | Black | dataset maintainers |
| `dataset/sudoku/generate_dataset.py` | Black/isort | dataset maintainers |
| `inference/example_full.py` | Black/isort | inference maintainers |
| `inference/example_lora.py` | Black/isort | inference maintainers |
| `reward_server/example_server.py` | Black/isort | reward_server maintainers |
| `scripts/convert_geneval_dataset.py` | Black | scripts maintainers |
| `scripts/convert_tdm_checkpoint.py` | isort | scripts maintainers |
| `src/flow_factory/loading/__init__.py` | isort | loading maintainers |
| `src/flow_factory/models/flux/flux3.py` | isort | models maintainers |
| `src/flow_factory/models/qwen_image/qwen_image.py` | isort | models maintainers |
| `src/flow_factory/models/qwen_image/qwen_image_edit_plus.py` | isort | models maintainers |
| `src/flow_factory/models/qwen_image_21/qwen_image_21.py` | isort | models maintainers |
| `src/flow_factory/models/variants.py` | Black | models maintainers |
| `src/flow_factory/trainers/common/group_noise.py` | isort | trainers maintainers |
| `src/flow_factory/trainers/role_optimization.py` | Black | trainers maintainers |
| `tests/distributed/test_region_entry_parity.py` | isort | distributed test maintainers |
| `tests/examples/test_multirole_examples.py` | Black | examples test maintainers |
| `tests/models/test_flux3.py` | isort | models test maintainers |
| `tests/models/test_qwen_image_21.py` | isort | models test maintainers |
| `tests/models/test_sensenova.py` | Black | models test maintainers |
| `tests/prompt_enhance/test_prompt_enhancers.py` | isort | prompt_enhance test maintainers |
| `tests/scheduler/minimax_h3/test_scheduler.py` | Black | scheduler test maintainers |
| `tests/scripts/test_convert_tdm_checkpoint.py` | Black | scripts test maintainers |
| `tests/trainers/test_dgpo_crd_production_paths.py` | isort | trainers test maintainers |
| `tests/trainers/test_distillation_guidance.py` | isort | trainers test maintainers |
| `tests/trainers/test_multirole_production_paths.py` | isort | trainers test maintainers |

The owning contributor fixes an entire touched file before merging its next change. Remaining
entries should be resolved in bounded mechanical cleanup PRs during P01c, before v0.3.0; keep
training behavior changes separate. Remove a row after both tools pass for that file. Re-inventory
after changing tool versions or formatting scope. Once every owned path passes, remove this debt
inventory and consider a full-tree format gate in a separate reviewed change. Upstream/fixture
exclusions remain ownership boundaries, independent of debt removal.

Run `make quality QUALITY_ARGS='--all-files'` for a read-only inventory of all configured rules.
It can report additional historical text/structured-file issues; resolve touched files under the
same policy. The focused Black/isort baseline above is not a claim that every new hygiene rule
already passes across all historical files. See [the quality workflow](quality.md) for scopes.
