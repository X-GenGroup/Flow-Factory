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

"""Integration tests for trainer-owned safe exact-resume lifecycle."""

import json
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from accelerate.utils import DistributedType

from flow_factory.trainers.abc import BaseTrainer
from flow_factory.trainers.common.runtime_state import (
    TRAINER_RUNTIME_METADATA_FILENAME,
    TrainerRuntimeState,
)
from flow_factory.trainers.execution import TrainingProgress


def _identity(*, world_size: int = 1) -> dict[str, Any]:
    """Return one strict realized-runtime identity for file tests."""
    return {
        "trainer": "tests.TinyTrainer",
        "adapter": "tests.TinyAdapter",
        "algorithm": "sft",
        "model": "tiny:tests/tiny",
        "finetune_type": "full",
        "optimizer_roles": ("base",),
        "parameter_schema_digest": "a" * 64,
        "optimizer_schema_digest": "b" * 64,
        "execution_contract_digest": "d" * 64,
        "data_contract_digest": "e" * 64,
        "distributed_type": "NO",
        "backend_schema_digest": "c" * 64,
        "mixed_precision": "no",
        "gradient_scaler": "none",
        "world_size": world_size,
    }


def _write_accelerate_artifacts(directory: Path) -> None:
    """Write the minimal exact-resume artifact names validated by runtime state."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "model.safetensors").write_bytes(b"prepared-model")
    torch.save({"state": {}, "param_groups": []}, directory / "optimizer.bin")
    _write_rng_artifact(directory / "random_states_0.pkl")


def _write_rng_artifact(path: Path) -> None:
    """Write one parseable per-rank Accelerate RNG state."""
    torch.save(
        {
            "step": 0,
            "random_state": random.getstate(),
            "numpy_random_seed": np.random.get_state(),
            "torch_manual_seed": torch.get_rng_state(),
        },
        path,
    )


class _SaveAccelerator:
    """Expose the synchronization and publication ownership used by the trainer."""

    def __init__(self) -> None:
        self.is_main_process = True
        self.is_local_main_process = True
        self.device = torch.device("cpu")
        self.project_configuration = SimpleNamespace(save_on_each_node=False)
        self.wait_calls = 0

    def wait_for_everyone(self) -> None:
        """Record each lifecycle barrier."""
        self.wait_calls += 1


class _StateSavingAdapter:
    """Represent the adapter's Accelerator-artifact save delegation."""

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[str, bool, bool]] = []

    def save_checkpoint(
        self,
        *,
        save_directory: str,
        model_only: bool,
        include_training_roles: bool,
    ) -> None:
        """Write core artifacts before optionally simulating an interrupted save."""
        self.calls.append((save_directory, model_only, include_training_roles))
        _write_accelerate_artifacts(Path(save_directory))
        if self.fail:
            raise RuntimeError("accelerator save failed")


class _StateLoadingAdapter:
    """Record state mutation and enforce the trainer's preflight ordering."""

    def __init__(self, trainer: Any, checkpoint: Path, *, fail_load: bool = False) -> None:
        self.trainer = trainer
        self.checkpoint = checkpoint
        self.post_init_paths: list[str | None] = []
        self.load_calls = 0
        self.core_mutated = False
        self.fail_load = fail_load

    def post_init(self) -> None:
        """Observe that the adapter's legacy automatic resume was suppressed."""
        self.post_init_paths.append(self.trainer.model_args.resume_path)

    def _resolve_checkpoint_path(self, path: str) -> str:
        """Return the already-local checkpoint directory."""
        assert path == str(self.checkpoint)
        return path

    def _load_training_state(self, path: str) -> None:
        """Mutate only after runtime validation has staged a payload."""
        assert path == str(self.checkpoint)
        assert self.trainer.runtime_state.validated_load_pending
        assert self.trainer.progress == TrainingProgress()
        self.load_calls += 1
        self.core_mutated = True
        if self.fail_load:
            raise RuntimeError("accelerator load failed")


class _RuntimeChild:
    """Checkpointable child that can fail only during the commit phase."""

    def __init__(self, value: int = 0, *, fail_load: bool = False) -> None:
        self.value = value
        self.fail_load = fail_load
        self.load_calls = 0

    def state_dict(self) -> dict[str, Any]:
        """Return one scalar child payload."""
        return {"value": self.value}

    def validate_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Validate without mutating the child."""
        if set(state_dict) != {"value"} or type(state_dict["value"]) is not int:
            raise TypeError(f"invalid runtime child payload: {state_dict!r}")

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore after core load, optionally simulating a rank-local failure."""
        self.load_calls += 1
        if self.fail_load:
            raise RuntimeError("runtime child load failed")
        self.value = state_dict["value"]


def _saving_trainer(adapter: _StateSavingAdapter) -> Any:
    """Build the narrow host consumed by BaseTrainer's atomic save helper."""
    trainer = object.__new__(BaseTrainer)
    trainer.accelerator = _SaveAccelerator()
    trainer.adapter = adapter
    trainer.runtime_state = TrainerRuntimeState(
        TrainingProgress(optimizer_step=4, rollout_iteration=0, data_epoch=2),
        identity=_identity(),
    )
    return trainer


def _loading_trainer(
    checkpoint: Path,
    *,
    fail_load: bool = False,
    num_processes: int = 1,
) -> Any:
    """Build the narrow host consumed by the adapter-finalization resume helper."""
    trainer = object.__new__(BaseTrainer)
    trainer.model_args = SimpleNamespace(
        resume_path=str(checkpoint),
        resume_type="state",
    )
    trainer.runtime_state = TrainerRuntimeState(identity=_identity())
    trainer._runtime_children_attached = False
    trainer.accelerator = SimpleNamespace(
        process_index=0,
        device=torch.device("cpu"),
        num_processes=num_processes,
    )
    trainer.adapter = _StateLoadingAdapter(trainer, checkpoint, fail_load=fail_load)
    return trainer


def _resume_loading_trainer(trainer: Any) -> None:
    """Run the constructor's post-init then safe exact-resume phases."""
    trainer._initialize_adapter_runtime()
    trainer._finalize_adapter_runtime()


def test_progress_property_has_one_runtime_source_of_truth() -> None:
    """Compatibility counters replace the immutable runtime value without a copy."""
    trainer = object.__new__(BaseTrainer)
    trainer.runtime_state = TrainerRuntimeState()

    trainer.step = 5
    trainer.progress = TrainingProgress(
        optimizer_step=trainer.step,
        rollout_iteration=3,
        data_epoch=2,
    )

    assert trainer.progress is trainer.runtime_state.progress
    assert trainer.step == 5
    assert "progress" not in trainer.__dict__


def test_exact_state_save_publishes_only_after_runtime_manifest(tmp_path: Path) -> None:
    """Accelerator artifacts and runtime manifest become visible as one directory."""
    final = tmp_path / "checkpoint-2"
    adapter = _StateSavingAdapter()
    trainer = _saving_trainer(adapter)

    trainer._save_exact_training_state(str(final))

    staging = tmp_path / ".checkpoint-2.flow-factory-staging"
    claim = tmp_path / ".checkpoint-2.flow-factory-publish-claim"
    assert final.is_dir()
    assert not staging.exists()
    assert not claim.exists()
    assert (final / TRAINER_RUNTIME_METADATA_FILENAME).is_file()
    metadata = json.loads((final / TRAINER_RUNTIME_METADATA_FILENAME).read_text(encoding="utf-8"))
    assert [entry["path"] for entry in metadata["state_files"]] == [
        "model.safetensors",
        "optimizer.bin",
        "random_states_0.pkl",
    ]
    assert adapter.calls == [(str(staging), False, True)]
    assert trainer.accelerator.wait_calls == 4


def test_atomic_publish_claim_elects_one_writer_on_a_shared_path(tmp_path: Path) -> None:
    """Multiple local-main candidates cannot race the manifest or directory rename."""
    claim = tmp_path / ".checkpoint.flow-factory-publish-claim"
    first = _saving_trainer(_StateSavingAdapter())
    second = _saving_trainer(_StateSavingAdapter())

    assert first._claim_state_checkpoint_publication(str(claim))
    assert not second._claim_state_checkpoint_publication(str(claim))


def test_global_publisher_claim_loss_aborts_before_core_save(tmp_path: Path) -> None:
    """A concurrent job cannot make this global main write the shared staging path."""
    adapter = _StateSavingAdapter()
    trainer = _saving_trainer(adapter)
    trainer._claim_state_checkpoint_publication = lambda path: False

    with pytest.raises(FileExistsError, match="claimed by a concurrent writer"):
        trainer._save_exact_training_state(str(tmp_path / "checkpoint"))

    assert adapter.calls == []


def test_manifest_failure_keeps_staging_and_claim_without_entering_next_barrier(
    tmp_path: Path,
) -> None:
    """A publisher error exits before peers could wait at the next raw barrier."""
    final = tmp_path / "checkpoint"
    trainer = _saving_trainer(_StateSavingAdapter())

    def fail_manifest(path: str) -> None:
        raise OSError(f"manifest failed at {path}")

    trainer.runtime_state.prepare_save = fail_manifest

    with pytest.raises(OSError, match="manifest failed"):
        trainer._save_exact_training_state(str(final))

    assert not final.exists()
    assert (tmp_path / ".checkpoint.flow-factory-staging").is_dir()
    assert (tmp_path / ".checkpoint.flow-factory-publish-claim").is_file()
    assert trainer.accelerator.wait_calls == 2


def test_remote_publication_failure_keeps_successful_node_claim(
    tmp_path: Path,
) -> None:
    """Node-local finals remain visibly claimed until every publisher succeeds."""
    final = tmp_path / "checkpoint"
    trainer = _saving_trainer(_StateSavingAdapter())

    def synchronize(phase: str, error: Exception | None) -> None:
        if error is not None:
            raise error
        if phase == "atomic publication":
            raise RuntimeError("remote publisher replace failed")

    trainer._synchronize_checkpoint_phase_error = synchronize

    with pytest.raises(RuntimeError, match="remote publisher replace failed"):
        trainer._save_exact_training_state(str(final))

    assert final.is_dir()
    assert not (tmp_path / ".checkpoint.flow-factory-staging").exists()
    assert (tmp_path / ".checkpoint.flow-factory-publish-claim").is_file()


def test_interrupted_accelerator_save_never_publishes_partial_destination(
    tmp_path: Path,
) -> None:
    """A failed core-state save leaves only its explicit staging directory."""
    final = tmp_path / "checkpoint-2"
    trainer = _saving_trainer(_StateSavingAdapter(fail=True))

    with pytest.raises(RuntimeError, match="accelerator save failed"):
        trainer._save_exact_training_state(str(final))

    assert not final.exists()
    assert (tmp_path / ".checkpoint-2.flow-factory-staging").is_dir()
    assert (tmp_path / ".checkpoint-2.flow-factory-publish-claim").is_file()


def test_exact_state_save_refuses_to_overwrite_before_adapter_mutation(
    tmp_path: Path,
) -> None:
    """An existing immutable destination is rejected before any core save call."""
    final = tmp_path / "checkpoint-2"
    final.mkdir()
    adapter = _StateSavingAdapter()
    trainer = _saving_trainer(adapter)

    with pytest.raises(FileExistsError, match="cannot overwrite"):
        trainer._save_exact_training_state(str(final))

    assert adapter.calls == []


def test_exact_state_save_rejects_partial_acquisition_before_adapter_mutation(
    tmp_path: Path,
) -> None:
    """Offline state cannot claim exact resume while a dataloader epoch is partial."""
    adapter = _StateSavingAdapter()
    trainer = _saving_trainer(adapter)
    trainer._acquisition_cycle_active = True

    with pytest.raises(RuntimeError, match="complete acquisition boundary"):
        trainer._save_exact_training_state(str(tmp_path / "checkpoint"))

    assert adapter.calls == []


def test_exact_state_save_rejects_mps_before_adapter_mutation(tmp_path: Path) -> None:
    """MPS cannot publish an exact checkpoint that Accelerate cannot restore."""
    adapter = _StateSavingAdapter()
    trainer = _saving_trainer(adapter)
    trainer.accelerator.device = torch.device("mps")

    with pytest.raises(RuntimeError, match="unsupported on MPS.*does not serialize"):
        trainer._save_exact_training_state(str(tmp_path / "checkpoint"))

    assert adapter.calls == []
    assert not (tmp_path / ".checkpoint.flow-factory-staging").exists()
    assert not (tmp_path / ".checkpoint.flow-factory-publish-claim").exists()


def test_sharded_auxiliary_state_fails_before_core_checkpoint_mutation(
    tmp_path: Path,
) -> None:
    """FSDP EMA shards are never serialized as if rank-zero tensors were replicated."""
    adapter = _StateSavingAdapter()
    adapter.component_variant_registry = SimpleNamespace(_snapshots={})
    trainer = _saving_trainer(adapter)
    trainer.accelerator.distributed_type = DistributedType.FSDP
    trainer.runtime_state = TrainerRuntimeState(
        child_names=(trainer._ADAPTER_EMA_RUNTIME_CHILD,),
        identity=_identity(),
    )

    with pytest.raises(RuntimeError, match="distributed-aware gather.*adapter_ema"):
        trainer._save_exact_training_state(str(tmp_path / "checkpoint"))

    assert adapter.calls == []


def test_fsdp2_per_node_auxiliary_save_rejects_before_core_mutation(tmp_path: Path) -> None:
    """A rank-local manifest must not claim peers on a node-private directory."""
    adapter = _StateSavingAdapter()
    trainer = _saving_trainer(adapter)
    trainer.accelerator.distributed_type = DistributedType.FSDP
    trainer.accelerator.is_fsdp2 = True
    trainer.accelerator.project_configuration.save_on_each_node = True
    with pytest.raises(RuntimeError, match="shared checkpoint directory.*save_on_each_node=false"):
        trainer._save_exact_training_state(str(tmp_path / "checkpoint"))
    assert adapter.calls == []
    assert not (tmp_path / "checkpoint").exists()


def test_untracked_named_snapshot_fails_before_core_checkpoint_mutation(
    tmp_path: Path,
) -> None:
    """An exact checkpoint cannot silently omit an algorithm-owned reference."""
    adapter = _StateSavingAdapter()
    adapter._named_parameters = {
        "old_policy": SimpleNamespace(ema_wrapper=object()),
    }
    trainer = _saving_trainer(adapter)

    with pytest.raises(RuntimeError, match="omit.*old_policy.*runtime_child_names"):
        trainer._save_exact_training_state(str(tmp_path / "checkpoint"))

    assert adapter.calls == []


def test_exact_resume_preflights_then_loads_then_commits_progress(tmp_path: Path) -> None:
    """Runtime progress changes only after the core loader returns successfully."""
    checkpoint = tmp_path / "checkpoint"
    _write_accelerate_artifacts(checkpoint)
    source = TrainerRuntimeState(
        TrainingProgress(optimizer_step=9, rollout_iteration=4, data_epoch=0),
        identity=_identity(),
    )
    source.prepare_save(checkpoint)
    trainer = _loading_trainer(checkpoint)
    commit_calls = 0
    commit = trainer.runtime_state.commit_validated_load

    def commit_once() -> None:
        nonlocal commit_calls
        commit_calls += 1
        commit()

    trainer.runtime_state.commit_validated_load = commit_once

    _resume_loading_trainer(trainer)

    assert trainer.adapter.post_init_paths == [None]
    assert trainer.adapter.load_calls == 1
    assert trainer.adapter.core_mutated
    assert trainer.progress == TrainingProgress(
        optimizer_step=9,
        rollout_iteration=4,
        data_epoch=0,
    )
    assert trainer.runtime_state.load_received
    assert trainer._runtime_children_attached
    assert commit_calls == 1


def test_online_exact_resume_skips_duplicate_source_checkpoint_save(tmp_path: Path) -> None:
    """The resumed pre-rollout boundary evaluates without rewriting checkpoint N."""
    checkpoint = tmp_path / "run" / "checkpoints" / "checkpoint-4"
    _write_accelerate_artifacts(checkpoint)
    TrainerRuntimeState(
        TrainingProgress(optimizer_step=9, rollout_iteration=4),
        identity=_identity(),
    ).prepare_save(checkpoint)
    trainer = _loading_trainer(checkpoint)
    _resume_loading_trainer(trainer)
    trainer.log_args = SimpleNamespace(
        save_freq=1,
        save_dir=str(tmp_path),
        run_name="run",
    )
    trainer.eval_args = SimpleNamespace(eval_freq=1)
    events: list[str] = []
    trainer.save_checkpoint = lambda *args, **kwargs: pytest.fail(
        f"duplicate save attempted: {args!r}, {kwargs!r}"
    )
    trainer.evaluate = lambda: events.append("eval")

    trainer._run_periodic_cycle_boundaries()

    assert events == ["eval"]


def test_online_exact_resume_does_not_skip_same_basename_at_different_path(
    tmp_path: Path,
) -> None:
    """Duplicate suppression compares resolved full paths and propagates save errors."""
    trainer = object.__new__(BaseTrainer)
    trainer.progress = TrainingProgress(rollout_iteration=4)
    trainer.log_args = SimpleNamespace(
        save_freq=1,
        save_dir=str(tmp_path / "new-output"),
        run_name="run",
    )
    trainer.eval_args = SimpleNamespace(eval_freq=1)
    trainer._exact_resume_source_checkpoint = trainer._canonical_checkpoint_path(
        str(tmp_path / "old-output" / "run" / "checkpoints" / "checkpoint-4")
    )
    trainer._exact_resume_boundary_pending = True

    def fail_save(*args: Any, **kwargs: Any) -> None:
        raise FileExistsError("different destination already exists")

    trainer.save_checkpoint = fail_save
    trainer.evaluate = lambda: pytest.fail("evaluation must not run after save failure")

    with pytest.raises(FileExistsError, match="different destination already exists"):
        trainer._run_periodic_cycle_boundaries()

    assert trainer._exact_resume_boundary_pending


def test_remote_rank_preflight_failure_aborts_before_local_core_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All ranks leave together when another rank cannot validate its RNG artifact."""
    checkpoint = tmp_path / "checkpoint"
    _write_accelerate_artifacts(checkpoint)
    TrainerRuntimeState(identity=_identity()).prepare_save(checkpoint)
    trainer = _loading_trainer(checkpoint, num_processes=2)
    remote_failure = {
        "rank": 1,
        "type": "RuntimeError",
        "message": "missing random_states_1.pkl",
    }
    monkeypatch.setattr(
        "flow_factory.trainers.abc.gather_object",
        lambda payload: [None, remote_failure],
    )

    with pytest.raises(RuntimeError, match="resume preflight failed across ranks"):
        _resume_loading_trainer(trainer)

    assert trainer.adapter.load_calls == 0
    assert not trainer.adapter.core_mutated
    assert not trainer.runtime_state.load_received


def test_remote_exact_state_resolution_has_no_adapter_barrier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Remote state resolution defers synchronization to the all-rank preflight."""
    calls: list[tuple[str, str | None, str | None]] = []

    def download(repo_id: str, subfolder: str | None, revision: str | None) -> str:
        calls.append((repo_id, subfolder, revision))
        return "/local/checkpoint"

    monkeypatch.setattr("flow_factory.trainers.abc.download_hf_checkpoint", download)

    assert (
        BaseTrainer._resolve_exact_state_checkpoint_path("hf://owner/repo/state@revision")
        == "/local/checkpoint"
    )
    assert calls == [("owner/repo", "state", "revision")]


def test_remote_rank_core_load_failure_prevents_local_runtime_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime progress commits only after every rank reports a successful core load."""
    checkpoint = tmp_path / "checkpoint"
    _write_accelerate_artifacts(checkpoint)
    TrainerRuntimeState(
        TrainingProgress(optimizer_step=9, data_epoch=3),
        identity=_identity(),
    ).prepare_save(checkpoint)
    trainer = _loading_trainer(checkpoint, num_processes=2)
    gathered = iter(
        (
            [None, None],
            [
                None,
                {
                    "rank": 1,
                    "type": "RuntimeError",
                    "message": "accelerator load failed",
                },
            ],
        )
    )
    monkeypatch.setattr(
        "flow_factory.trainers.abc.gather_object",
        lambda payload: next(gathered),
    )

    with pytest.raises(RuntimeError, match="Accelerator artifact load failed across ranks"):
        _resume_loading_trainer(trainer)

    assert trainer.adapter.load_calls == 1
    assert trainer.adapter.core_mutated
    assert trainer.progress == TrainingProgress()
    assert trainer.runtime_state.validated_load_pending
    assert not trainer.runtime_state.load_received


def test_remote_rank_runtime_child_failure_synchronizes_after_every_core_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No rank returns from finalization when another rank cannot attach a child."""
    checkpoint = tmp_path / "checkpoint"
    _write_accelerate_artifacts(checkpoint)
    source_child = _RuntimeChild(7)
    source = TrainerRuntimeState(
        TrainingProgress(optimizer_step=9, data_epoch=3),
        child_names=("algorithm_state",),
        identity=_identity(),
    )
    source.attach_child("algorithm_state", source_child)
    source.prepare_save(checkpoint)

    trainer = _loading_trainer(checkpoint, num_processes=2)
    target_child = _RuntimeChild()
    trainer.runtime_state = TrainerRuntimeState(
        child_names=("algorithm_state",),
        identity=_identity(),
    )
    trainer._trainer_runtime_children = {"algorithm_state": target_child}
    gathered = iter(
        (
            [None, None],
            [None, None],
            [
                None,
                {
                    "rank": 1,
                    "type": "RuntimeError",
                    "message": "runtime child load failed",
                },
            ],
        )
    )
    monkeypatch.setattr(
        "flow_factory.trainers.abc.gather_object",
        lambda payload: next(gathered),
    )

    with pytest.raises(RuntimeError, match="runtime child commit failed across ranks"):
        _resume_loading_trainer(trainer)

    assert trainer.adapter.load_calls == 1
    assert trainer.adapter.core_mutated
    assert target_child.load_calls == 1
    assert getattr(trainer, "_exact_resume_source_checkpoint", None) is None
    assert not getattr(trainer, "_exact_resume_boundary_pending", False)


@pytest.mark.parametrize(
    "identity_field",
    ("execution_contract_digest", "data_contract_digest"),
)
def test_execution_or_data_contract_drift_is_rejected_before_core_load(
    tmp_path: Path,
    identity_field: str,
) -> None:
    """Objective and loader digests participate in the manifest preflight gate."""
    checkpoint = tmp_path / "checkpoint"
    _write_accelerate_artifacts(checkpoint)
    TrainerRuntimeState(identity=_identity()).prepare_save(checkpoint)
    trainer = _loading_trainer(checkpoint)
    target_identity = _identity()
    target_identity[identity_field] = "f" * 64
    trainer.runtime_state = TrainerRuntimeState(identity=target_identity)

    with pytest.raises(ValueError, match=f"identity mismatch.*{identity_field}"):
        _resume_loading_trainer(trainer)

    assert trainer.adapter.load_calls == 0
    assert not trainer.adapter.core_mutated


def test_corrupt_core_artifact_is_rejected_before_adapter_mutation(tmp_path: Path) -> None:
    """Manifest hashes gate the adapter's model/optimizer load call."""
    checkpoint = tmp_path / "checkpoint"
    _write_accelerate_artifacts(checkpoint)
    source = TrainerRuntimeState(identity=_identity())
    source.prepare_save(checkpoint)
    (checkpoint / "optimizer.bin").write_bytes(b"corrupt")
    trainer = _loading_trainer(checkpoint)

    with pytest.raises(RuntimeError, match="size mismatch|SHA-256 mismatch"):
        _resume_loading_trainer(trainer)

    assert trainer.adapter.load_calls == 0
    assert not trainer.adapter.core_mutated
    assert trainer.progress == TrainingProgress()


def test_unmanifested_backend_artifact_is_rejected_before_adapter_mutation(
    tmp_path: Path,
) -> None:
    """A stale file cannot become an unhashed input to Accelerator.load_state."""
    checkpoint = tmp_path / "checkpoint"
    _write_accelerate_artifacts(checkpoint)
    source = TrainerRuntimeState(identity=_identity())
    source.prepare_save(checkpoint)
    (checkpoint / "optimizer_1.bin").write_bytes(b"stale")
    trainer = _loading_trainer(checkpoint)

    with pytest.raises(RuntimeError, match="unmanifested=.*optimizer_1.bin"):
        _resume_loading_trainer(trainer)

    assert trainer.adapter.load_calls == 0
    assert not trainer.adapter.core_mutated


def test_preflight_recognizes_sharded_fsdp_and_nonzero_rank_rng_artifacts(
    tmp_path: Path,
) -> None:
    """Exact resume does not assume plain Accelerator filenames or global rank zero."""
    checkpoint = tmp_path / "checkpoint"
    (checkpoint / "pytorch_model_fsdp_0").mkdir(parents=True)
    (checkpoint / "optimizer_0").mkdir()
    (checkpoint / "pytorch_model_fsdp_0" / ".metadata").write_bytes(b"model")
    (checkpoint / "optimizer_0" / ".metadata").write_bytes(b"optimizer")
    _write_rng_artifact(checkpoint / "random_states_4.pkl")
    source = TrainerRuntimeState(identity=_identity(world_size=5))
    source.prepare_save(checkpoint)
    restored = TrainerRuntimeState(identity=_identity(world_size=5))

    restored.validate_load(
        checkpoint,
        expected_process_index=4,
        expected_device_type="cpu",
    )
    restored.commit_validated_load()

    assert restored.load_received


def test_preflight_requires_the_current_rank_rng_before_core_mutation(
    tmp_path: Path,
) -> None:
    """Another rank's valid RNG payload cannot make this rank's resume exact."""
    checkpoint = tmp_path / "checkpoint"
    _write_accelerate_artifacts(checkpoint)
    source = TrainerRuntimeState(identity=_identity(world_size=2))
    source.prepare_save(checkpoint)
    restored = TrainerRuntimeState(identity=_identity(world_size=2))

    with pytest.raises(RuntimeError, match="current rank RNG.*random_states_1.pkl"):
        restored.validate_load(
            checkpoint,
            expected_process_index=1,
            expected_device_type="cpu",
        )

    assert not restored.validated_load_pending


def test_preflight_requires_the_active_device_rng_payload(tmp_path: Path) -> None:
    """Accelerate's silent CUDA RNG fallback is rejected before state mutation."""
    checkpoint = tmp_path / "checkpoint"
    _write_accelerate_artifacts(checkpoint)
    source = TrainerRuntimeState(identity=_identity())
    source.prepare_save(checkpoint)
    restored = TrainerRuntimeState(identity=_identity())

    with pytest.raises(ValueError, match="missing required keys.*torch_cuda_manual_seed"):
        restored.validate_load(
            checkpoint,
            expected_process_index=0,
            expected_device_type="cuda",
        )

    assert not restored.validated_load_pending


def test_preflight_rejects_device_rng_topology_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A saved per-device RNG list must match the current visible device count."""
    checkpoint = tmp_path / "checkpoint"
    _write_accelerate_artifacts(checkpoint)
    rng_path = checkpoint / "random_states_0.pkl"
    rng_state = torch.load(rng_path, map_location="cpu", weights_only=False)
    rng_state["torch_cuda_manual_seed"] = [torch.zeros(16, dtype=torch.uint8)]
    torch.save(rng_state, rng_path)
    source = TrainerRuntimeState(identity=_identity())
    source.prepare_save(checkpoint)
    restored = TrainerRuntimeState(identity=_identity())
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    with pytest.raises(ValueError, match="device-state topology mismatch.*expected 2"):
        restored.validate_load(
            checkpoint,
            expected_process_index=0,
            expected_device_type="cuda",
        )

    assert not restored.validated_load_pending


def test_backend_identity_drift_is_rejected_before_core_state_validation(
    tmp_path: Path,
) -> None:
    """A checkpoint cannot cross prepared-state backend layouts by accident."""
    checkpoint = tmp_path / "checkpoint"
    _write_accelerate_artifacts(checkpoint)
    source = TrainerRuntimeState(identity=_identity())
    source.prepare_save(checkpoint)
    target_identity = _identity()
    target_identity["distributed_type"] = "FSDP"
    target_identity["backend_schema_digest"] = "d" * 64
    restored = TrainerRuntimeState(identity=target_identity)

    with pytest.raises(ValueError, match="identity mismatch.*distributed_type.*FSDP"):
        restored.validate_load(
            checkpoint,
            expected_process_index=0,
            expected_device_type="cpu",
        )

    assert not restored.validated_load_pending


def test_preflight_requires_scaler_artifact_before_model_mutation(tmp_path: Path) -> None:
    """An fp16 runtime cannot discover a missing scaler after loading model state."""
    checkpoint = tmp_path / "checkpoint"
    _write_accelerate_artifacts(checkpoint)
    identity = _identity()
    identity["mixed_precision"] = "fp16"
    identity["gradient_scaler"] = "torch.amp.grad_scaler.GradScaler"
    source = TrainerRuntimeState(identity=identity)
    source.prepare_save(checkpoint)
    restored = TrainerRuntimeState(identity=identity)

    with pytest.raises(RuntimeError, match="missing the gradient scaler artifact"):
        restored.validate_load(
            checkpoint,
            expected_process_index=0,
            expected_device_type="cpu",
        )

    assert not restored.validated_load_pending


def test_failed_core_load_does_not_commit_runtime_progress(tmp_path: Path) -> None:
    """A backend load exception leaves the validated runtime payload uncommitted."""
    checkpoint = tmp_path / "checkpoint"
    _write_accelerate_artifacts(checkpoint)
    source = TrainerRuntimeState(
        TrainingProgress(optimizer_step=9, data_epoch=3),
        identity=_identity(),
    )
    source.prepare_save(checkpoint)
    trainer = _loading_trainer(checkpoint, fail_load=True)

    with pytest.raises(RuntimeError, match="accelerator load failed"):
        _resume_loading_trainer(trainer)

    assert trainer.adapter.load_calls == 1
    assert trainer.adapter.core_mutated
    assert trainer.progress == TrainingProgress()
    assert trainer.runtime_state.validated_load_pending
    assert not trainer.runtime_state.load_received


def test_public_late_state_load_is_rejected_before_adapter_call() -> None:
    """Exact state restore cannot bypass constructor-time identity preflight."""
    trainer = object.__new__(BaseTrainer)
    trainer.adapter = SimpleNamespace(load_checkpoint=lambda **kwargs: pytest.fail(str(kwargs)))

    with pytest.raises(RuntimeError, match="must be configured.*before trainer construction"):
        trainer.load_checkpoint("checkpoint", resume_type="state")
