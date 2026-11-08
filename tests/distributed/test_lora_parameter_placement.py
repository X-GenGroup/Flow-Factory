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

"""CPU DTensor placement proof for strict prepared LoRA imports."""

import json
import traceback
from pathlib import Path

import torch
import torch.distributed as dist
from conftest import run_gloo_ranks
from peft import LoraConfig, get_peft_model
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Shard, distribute_tensor
from torch.distributed.tensor.placement_types import _StridedShard

from flow_factory.trainers.distillation.opd.common import _load_one_teacher
from tests.models.test_lora_parameter_loading import TinyComponent, _owner


def _component():
    module = TinyComponent()
    module.block = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 8))
    return get_peft_model(module, LoraConfig(r=4, lora_alpha=4, target_modules=["0", "1"]))


def _worker(rank, store_path, result_dir, options):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{store_path}", rank=rank, world_size=4)
    result = {"rank": rank}
    try:
        teacher = _component()
        with torch.no_grad():
            for index, parameter in enumerate(p for p in teacher.parameters() if p.requires_grad):
                parameter.copy_(
                    torch.arange(parameter.numel()).reshape(parameter.shape) / 17 + index
                )
        expected = [p.detach().clone() for p in teacher.parameters() if p.requires_grad]
        path = Path(result_dir) / "teacher"
        if rank == 0:
            teacher.save_pretrained(path)
        dist.barrier()
        torch.manual_seed(17)
        student = _component()
        mesh = DeviceMesh("cpu", torch.arange(4).reshape(2, 2))
        placements = (_StridedShard(0, split_factor=2), Shard(0))
        for name, parameter in tuple(student.named_parameters()):
            if parameter.requires_grad:
                parent, leaf = name.rsplit(".", 1)
                sharded = distribute_tensor(
                    parameter.detach(), mesh, placements, src_data_rank=None
                )
                setattr(student.get_submodule(parent), leaf, torch.nn.Parameter(sharded))
        owner = _owner(student)
        parameters = owner._get_component_parameters(["transformer"])
        identities = [id(parameter) for parameter in parameters]
        local_before = [parameter.to_local().detach().clone() for parameter in parameters]
        _load_one_teacher(owner, "teacher", str(path), ["transformer"], "cpu")
        for value, reference in zip(
            owner._named_parameters["teacher"].ema_wrapper.ema_parameters, expected, strict=True
        ):
            assert value.placements == placements and value.device_mesh == mesh
            torch.testing.assert_close(value.full_tensor(), reference, rtol=0, atol=0)
        assert identities == [
            id(parameter) for parameter in owner._get_component_parameters(["transformer"])
        ]
        for parameter, reference in zip(parameters, local_before, strict=True):
            torch.testing.assert_close(parameter.to_local(), reference, rtol=0, atol=0)
        result.update(
            status="pass", strided_placements=True, restored_student=True, parameter_identity=True
        )
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result, indent=2))
        dist.destroy_process_group()


def test_cpu_strided_mesh_teacher_import_preserves_placement_and_student(tmp_path):
    for result in run_gloo_ranks(_worker, tmp_path, 4, {}):
        assert "error" not in result, result.get("error")
        assert result["status"] == "pass"
