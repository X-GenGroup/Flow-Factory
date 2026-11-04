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

"""Format compatibility and failure atomicity for rank-local runtime manifests."""

import hashlib
import json
from pathlib import Path

import pytest
import torch

from flow_factory.ema.ema import EMAModuleWrapper
from flow_factory.trainers.common.runtime_state import (
    TRAINER_RUNTIME_METADATA_FILENAME,
    TrainerRuntimeState,
)


def _runtime(value: float = 1.0):
    child = EMAModuleWrapper(
        [torch.tensor([value])], decay=0.5, update_step_interval=2, decay_schedule="constant"
    )
    identity = {**TrainerRuntimeState().identity, "world_size": 2}
    runtime = TrainerRuntimeState(child_names=("ema",), identity=identity)
    return runtime, child


def _write_ranks(path: Path):
    for rank in range(2):
        runtime, child = _runtime(float(rank + 1))
        runtime.attach_child("ema", child)
        runtime.prepare_rank_save(path, process_index=rank)
    runtime.prepare_save(path, rank_count=2)


def test_rank_payload_restores_selected_rank_and_retains_deferred_mutation(tmp_path):
    _write_ranks(tmp_path)
    assert json.loads((tmp_path / TRAINER_RUNTIME_METADATA_FILENAME).read_text())["version"] == 2
    runtime, child = _runtime(9.0)
    runtime.validate_load(tmp_path, children={"ema": child}, expected_process_index=1)
    assert child.ema_parameters[0].item() == 9.0
    runtime.commit_validated_load()
    runtime.attach_child("ema", child)
    assert child.ema_parameters[0].item() == 2.0


@pytest.mark.parametrize("damage", ["missing_peer", "corrupt_peer", "wrong_rank", "extra_backend"])
def test_every_rank_is_preflighted_before_any_child_mutation(tmp_path, damage):
    _write_ranks(tmp_path)
    peer = tmp_path / "flow_factory_trainer_runtime.rank00001.json"
    if damage == "missing_peer":
        peer.unlink()
    elif damage == "corrupt_peer":
        payload = json.loads(peer.read_text())
        tensor_path = tmp_path / payload["tensor_file"]["path"]
        tensor_path.write_bytes(tensor_path.read_bytes()[:-1])
    elif damage == "wrong_rank":
        peer.write_text(peer.read_text().replace('"rank": 1', '"rank": 0'))
        root = tmp_path / TRAINER_RUNTIME_METADATA_FILENAME
        manifest = json.loads(root.read_text())
        descriptor = {
            "path": peer.name,
            "size": peer.stat().st_size,
            "sha256": hashlib.sha256(peer.read_bytes()).hexdigest(),
        }
        manifest["rank_files"][1] = descriptor
        manifest["state_files"] = [
            descriptor if entry["path"] == peer.name else entry for entry in manifest["state_files"]
        ]
        root.write_text(json.dumps(manifest))
    else:
        (tmp_path / "optimizer_99.bin").write_bytes(b"foreign")
    runtime, child = _runtime(9.0)
    with pytest.raises(
        (ValueError, RuntimeError, FileNotFoundError),
        match="topology mismatch" if damage == "wrong_rank" else None,
    ):
        runtime.validate_load(tmp_path, children={"ema": child}, expected_process_index=0)
    assert child.ema_parameters[0].item() == 9.0
    assert not runtime.validated_load_pending
    assert not runtime.load_received


def test_incomplete_shard_set_cannot_publish(tmp_path):
    runtime, child = _runtime()
    runtime.attach_child("ema", child)
    runtime.prepare_rank_save(tmp_path, process_index=0)
    with pytest.raises(FileNotFoundError):
        runtime.prepare_save(tmp_path, rank_count=2)
    assert not (tmp_path / TRAINER_RUNTIME_METADATA_FILENAME).exists()


def test_ema_cadence_frozen_reference_and_installed_scope_boundary():
    live = torch.nn.Parameter(torch.tensor([1.0]))
    ema = EMAModuleWrapper([live], decay=0.5, update_step_interval=2, decay_schedule="constant")
    reference = EMAModuleWrapper([live], decay=0.0, update_step_interval=0)
    for step in range(4):
        live.data.add_(1)
        ema.step([live], step)
        reference.step([live], step)
    assert ema.num_updates == 2
    assert ema.ema_parameters[0].item() == 3.5
    assert reference.num_updates == 0
    assert reference.ema_parameters[0].item() == 1.0
    assert ema.ema_parameters[0].grad is None
    assert not ema.ema_parameters[0].requires_grad
    before = live.detach().clone()
    with pytest.raises(RuntimeError, match="injected"):
        with ema.use_ema_parameters([live]):
            with pytest.raises(RuntimeError, match="installed"):
                ema.state_dict()
            raise RuntimeError("injected")
    torch.testing.assert_close(live, before, rtol=0, atol=0)
