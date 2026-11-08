# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Baseline oracle for the unchanged public concatenated LTX2 ``forward``.

``legacy_forward_golden.json`` was captured by running the pre-Task-4A
implementation against the shared fakes; see ``generate_legacy_forward_golden.py``
in this directory. The expectations therefore never come from the new
component-return branch, which this module never calls.

Provenance is checked in two layers. The recorded identity is always compared with
the fixed commit/blob constants below, and every output/order/RNG comparison reads
only the JSON, so the whole oracle runs from a source export or a shallow clone.
The live ``git rev-parse`` cross-check that ties those constants to real objects
runs only where Git and the referenced objects are available.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import pytest
import torch
from ltx2_fakes import (
    AUDIO_SCHEDULER_OFFSET,
    BATCH_SIZE,
    TIMESTEP,
    VIDEO_SCHEDULER_OFFSET,
    VIDEO_SEQ_LEN,
    PipelineFake,
    SchedulerFake,
    TransformerFake,
    audio_latents,
    conditioning_mask,
    forward_conditioning_kwargs,
    video_latents,
)

from flow_factory.models.ltx2.ltx2_i2av import LTX2_I2AV_Adapter
from flow_factory.models.ltx2.ltx2_t2av import LTX2_T2AV_Adapter
from flow_factory.scheduler import SDESchedulerOutput

TESTS_DIR = Path(__file__).resolve().parent
REPO = TESTS_DIR.parents[2]
GOLDEN = json.loads((TESTS_DIR / "legacy_forward_golden.json").read_text())
ADAPTERS = {"t2av": LTX2_T2AV_Adapter, "i2av": LTX2_I2AV_Adapter}
ORACLE_COMMIT = "ee6d247e14b0192a741d236a0a9d491aa10042fa"
ORACLE_COMMIT_REF = "ee6d247"
ORACLE_BLOBS = {
    "src/flow_factory/models/ltx2/ltx2_t2av.py": "4c8105b5ba84afb000677ca1349f49621481ce6a",
    "src/flow_factory/models/ltx2/ltx2_i2av.py": "228fa1df199a2d02e635fc2c9d66f4e838450a05",
}
SEED = 20260810
NOISE_SCALE = 0.125
# The oracle holds fp32 values captured on the machine that ran the pre-Task-4A
# implementation. Replaying the same arithmetic elsewhere reassociates it differently,
# so an exact comparison reports a machine change rather than a code change: on a second
# machine every field lands within 1 ULP (1.19e-07 absolute, 1.0e-07 relative). A
# behavioral change in the concatenated forward -- a dropped term, a different reduction
# order across components, a rescaled sigma -- moves these values by orders of magnitude
# more, so the window below still fails on the drift this oracle exists to catch. The RNG
# position below stays exact: it is a draw, not arithmetic, and must reproduce bit for bit.
FORWARD_ATOL = 1e-6
FORWARD_RTOL = 1e-6
OUTPUT_FIELDS = (
    "next_latents",
    "next_latents_mean",
    "std_dev_t",
    "dt",
    "log_prob",
    "velocity",
)


def _adapter(cls: type) -> Tuple[Any, List[Tuple[str, Any]]]:
    log: List[Tuple[str, Any]] = []
    transformer = TransformerFake(noise_scale=GOLDEN["noise_scale"])
    adapter = object.__new__(cls)
    adapter.accelerator = SimpleNamespace(device=torch.device("cpu"))
    adapter.pipeline = PipelineFake(SchedulerFake(VIDEO_SCHEDULER_OFFSET, log), transformer)
    adapter.component_runtime = SimpleNamespace(get_component=lambda name: transformer)
    adapter.load_scheduler = lambda: SchedulerFake(AUDIO_SCHEDULER_OFFSET, log)
    adapter.scheduler_group = adapter.build_scheduler_group()
    return adapter, log


def _expected(entry: Optional[Dict[str, Any]]) -> Optional[torch.Tensor]:
    if entry is None:
        return None
    return torch.tensor(entry["values"], dtype=torch.float32).reshape(entry["shape"])


def _run(name: str, *, compute_log_prob: bool, fields: Tuple[str, ...]) -> Dict[str, Any]:
    cls = ADAPTERS[name]
    adapter, log = _adapter(cls)
    extra = {"conditioning_mask": conditioning_mask()} if cls is LTX2_I2AV_Adapter else {}
    torch.manual_seed(GOLDEN["seed"])
    output = adapter.forward(
        t=torch.full((BATCH_SIZE,), TIMESTEP),
        t_next=torch.zeros(BATCH_SIZE),
        latents=torch.cat([video_latents(), audio_latents()], dim=1),
        video_seq_len=VIDEO_SEQ_LEN,
        compute_log_prob=compute_log_prob,
        return_kwargs=list(fields),
        **extra,
        **forward_conditioning_kwargs(),
    )
    return {
        "output": output,
        "log": log,
        "rng_state_sum": int(torch.get_rng_state().sum().item()),
        "post_draw": torch.randn(4),
    }


def _cases() -> List[Tuple[str, str, bool, Tuple[str, ...]]]:
    cases = []
    for key in sorted(GOLDEN["cases"]):
        name, log_prob_part, fields_part = key.split("|")
        compute_log_prob = log_prob_part.split("=")[1] == "True"
        fields = tuple(fields_part.split("=")[1].split(","))
        cases.append((key, name, compute_log_prob, fields))
    return cases


def test_t2av_guidance_calls_expose_optional_audio_time_to_native_cp_metadata() -> None:
    from diffusers.hooks.context_parallel import ModuleForwardMetadata
    from diffusers.models.transformers.transformer_ltx2 import LTX2VideoTransformer3DModel

    adapter, _ = _adapter(LTX2_T2AV_Adapter)
    transformer = adapter.pipeline.transformer
    metadata = ModuleForwardMetadata(_cls=LTX2VideoTransformer3DModel)
    observed = []

    def native_call(*args, **kwargs):
        # This is the native CP lookup that previously failed before model entry;
        # it cannot resolve a Python default omitted from both args and kwargs.
        value, is_kwarg, _ = metadata._get_parameter_from_args_kwargs(
            "audio_timestep", args, kwargs
        )
        assert value is None and is_kwarg
        observed.append(
            (kwargs.get("isolate_modalities"), kwargs.get("spatio_temporal_guidance_blocks"))
        )
        return transformer(*args, **kwargs)

    adapter.component_runtime = SimpleNamespace(get_component=lambda name: native_call)
    conditioning = forward_conditioning_kwargs()
    conditioning.update(
        guidance_scale=2.0,
        negative_connector_prompt_embeds=conditioning["connector_prompt_embeds"],
        negative_connector_audio_prompt_embeds=conditioning["connector_audio_prompt_embeds"],
        negative_connector_attention_mask=conditioning["connector_attention_mask"],
        stg_scale=1.0,
        spatio_temporal_guidance_blocks=[0],
        modality_scale=2.0,
    )
    adapter.forward(
        t=torch.full((BATCH_SIZE,), TIMESTEP),
        t_next=torch.zeros(BATCH_SIZE),
        latents=torch.cat([video_latents(), audio_latents()], dim=1),
        video_seq_len=VIDEO_SEQ_LEN,
        compute_log_prob=False,
        return_kwargs=["velocity"],
        **conditioning,
    )
    assert observed == [(None, None), (False, [0]), (True, None)]


@pytest.mark.parametrize(
    "key,name,compute_log_prob,fields",
    _cases(),
    ids=[case[0] for case in _cases()],
)
def test_legacy_concatenated_forward_matches_the_pre_task_4a_oracle(
    key: str, name: str, compute_log_prob: bool, fields: Tuple[str, ...]
) -> None:
    golden = GOLDEN["cases"][key]

    result = _run(name, compute_log_prob=compute_log_prob, fields=fields)
    output = result["output"]

    assert isinstance(output, SDESchedulerOutput)
    for field in OUTPUT_FIELDS:
        expected = _expected(golden[field])
        received = getattr(output, field)
        if expected is None:
            assert received is None, f"{key}: expected {field} to stay unset"
            continue
        assert received is not None, f"{key}: expected {field} to be returned"
        drift = (received.double() - expected.double()).abs().max().item()
        assert torch.allclose(received, expected, atol=FORWARD_ATOL, rtol=FORWARD_RTOL), (
            f"{key}: {field} drifted by {drift:.3e}, outside "
            f"atol={FORWARD_ATOL} rtol={FORWARD_RTOL}"
        )


@pytest.mark.parametrize(
    "key,name,compute_log_prob,fields",
    _cases(),
    ids=[case[0] for case in _cases()],
)
def test_legacy_forward_keeps_the_scheduler_order_and_rng_position(
    key: str, name: str, compute_log_prob: bool, fields: Tuple[str, ...]
) -> None:
    golden = GOLDEN["cases"][key]

    result = _run(name, compute_log_prob=compute_log_prob, fields=fields)

    assert [list(entry) for entry in result["log"]] == golden["dispatch_log"]
    assert result["rng_state_sum"] == golden["rng_state_sum"]
    assert torch.allclose(
        result["post_draw"],
        torch.tensor(golden["post_forward_draw"], dtype=torch.float32),
        atol=0,
        rtol=0,
    )


def _git_object(ref: str) -> Optional[str]:
    """Return the object hash ``ref`` resolves to, or ``None`` when Git cannot resolve it.

    Source exports have no Git binary or repository and shallow clones lack the oracle
    objects; the caller skips its cross-check there instead of failing the whole oracle.
    """
    if shutil.which("git") is None or not (REPO / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", ref],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _resolved_or_skip(ref: str) -> str:
    resolved = _git_object(ref)
    if resolved is None:
        pytest.skip(f"Git object {ref!r} cannot be resolved in this checkout")
    return resolved


def test_the_golden_file_matches_the_fixed_oracle_identity() -> None:
    assert GOLDEN["oracle_commit"] == ORACLE_COMMIT
    assert GOLDEN["oracle_commit_ref"] == ORACLE_COMMIT_REF
    assert ORACLE_COMMIT.startswith(ORACLE_COMMIT_REF)
    assert GOLDEN["oracle_blobs"] == ORACLE_BLOBS
    assert GOLDEN["seed"] == SEED
    assert GOLDEN["noise_scale"] == NOISE_SCALE


def test_the_golden_file_covers_every_recorded_case() -> None:
    assert len(GOLDEN["cases"]) == 12
    assert {key.split("|")[0] for key in GOLDEN["cases"]} == {"t2av", "i2av"}


def test_the_recorded_commit_matches_the_git_objects() -> None:
    assert _resolved_or_skip(f"{ORACLE_COMMIT}^{{commit}}") == ORACLE_COMMIT
    assert _resolved_or_skip(f"{ORACLE_COMMIT_REF}^{{commit}}") == ORACLE_COMMIT


def test_the_recorded_blobs_match_the_git_objects() -> None:
    for path, blob in ORACLE_BLOBS.items():
        assert _resolved_or_skip(f"{ORACLE_COMMIT}:{path}") == blob
        # The adapters changed in Task 4A, so the oracle blobs must not be the current ones.
        assert _resolved_or_skip(f"HEAD:{path}") != blob


def test_the_generator_pins_the_recorded_identity_and_refuses_other_checkouts(
    tmp_path: Path,
) -> None:
    _resolved_or_skip(f"{ORACLE_COMMIT}^{{commit}}")
    import generate_legacy_forward_golden as generator

    identity = generator.resolve_oracle_identity()

    assert identity["oracle_commit"] == ORACLE_COMMIT
    assert identity["oracle_blobs"] == ORACLE_BLOBS
    with pytest.raises(ValueError, match=r"oracle worktree.*HEAD.*ee6d247.*received"):
        generator.require_oracle_worktree(REPO, identity["oracle_commit"])
    with pytest.raises(FileNotFoundError, match=r"expected an oracle worktree"):
        generator.require_oracle_worktree(tmp_path / "missing", identity["oracle_commit"])


GIT_DEPENDENT_TESTS = (
    test_the_recorded_commit_matches_the_git_objects,
    test_the_recorded_blobs_match_the_git_objects,
)


def _run_git_independent_assertions() -> int:
    """Execute every golden comparison that must survive without Git, returning the case count."""
    checked = 0
    for key, name, compute_log_prob, fields in _cases():
        test_legacy_concatenated_forward_matches_the_pre_task_4a_oracle(
            key, name, compute_log_prob, fields
        )
        test_legacy_forward_keeps_the_scheduler_order_and_rng_position(
            key, name, compute_log_prob, fields
        )
        checked += 1
    test_the_golden_file_matches_the_fixed_oracle_identity()
    test_the_golden_file_covers_every_recorded_case()
    return checked


def _simulate_unavailable_git(monkeypatch: pytest.MonkeyPatch, mode: str, tmp_path: Path) -> None:
    """Make ``_git_object`` see a source export, a non-repository, or a shallow clone."""
    if mode == "no_git_binary":
        monkeypatch.setattr(shutil, "which", lambda name: None)
    elif mode == "no_repository":
        monkeypatch.setattr(sys.modules[__name__], "REPO", tmp_path)
    elif mode == "unresolvable_objects":
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *args, **kwargs: SimpleNamespace(returncode=128, stdout="", stderr="fatal"),
        )
    else:
        raise ValueError(f"expected a known simulation mode, received {mode!r}")


@pytest.mark.parametrize("mode", ["no_git_binary", "no_repository", "unresolvable_objects"])
def test_golden_comparisons_still_run_when_git_lookup_is_unavailable(
    monkeypatch: pytest.MonkeyPatch, mode: str, tmp_path: Path
) -> None:
    _simulate_unavailable_git(monkeypatch, mode, tmp_path)

    assert _git_object(f"{ORACLE_COMMIT}^{{commit}}") is None
    assert _run_git_independent_assertions() == 12


@pytest.mark.parametrize("mode", ["no_git_binary", "no_repository", "unresolvable_objects"])
@pytest.mark.parametrize("test", GIT_DEPENDENT_TESTS, ids=lambda test: test.__name__)
def test_provenance_cross_checks_skip_when_git_lookup_is_unavailable(
    monkeypatch: pytest.MonkeyPatch, mode: str, tmp_path: Path, test: Any
) -> None:
    _simulate_unavailable_git(monkeypatch, mode, tmp_path)

    with pytest.raises(pytest.skip.Exception, match=r"cannot be resolved"):
        test()
