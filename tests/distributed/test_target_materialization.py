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

"""Meta-first TP/FSDP2 preparation materializes the replicated path's exact local shards.

Each case prepares the same model twice on a DP2 x TP2 gloo mesh: once from
loaded values (today's path) and once from meta parameters filled afterwards by
``ParameterSource.materialize``. Every rank's local shard, every full tensor,
the forward output and the optimizer's parameter identities must agree exactly.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pytest
import torch
import torch.distributed as dist
from conftest import requires_gloo, run_gloo_ranks
from torch import nn

WORLD_SIZE = 4


class _Block(nn.Module):
    def __init__(self, width: int, hidden: int) -> None:
        super().__init__()
        self.up = nn.Linear(width, hidden)
        self.down = nn.Linear(hidden, width)
        self.norm = nn.LayerNorm(width)
        # FSDP2 requires a shared module to stay inside one sharding unit.
        self.tied = self.norm
        self.register_buffer("table", torch.arange(width, dtype=torch.float32), persistent=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = self.down(torch.relu(self.up(self.norm(tokens))))
        return self.tied(tokens + hidden) + self.table


class _Net(nn.Module):
    """Uneven head rows, a frozen block, tied modules and non-persistent buffers."""

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(_Block(12, 36) for _ in range(3))
        self.head = nn.Linear(12, 7)
        for parameter in self.blocks[2].parameters():
            parameter.requires_grad_(False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            tokens = block(tokens)
        return self.head(tokens)


def _plain_case() -> Tuple[nn.Module, Callable[[nn.Module], torch.Tensor], Callable, List[str]]:
    from torch.distributed.tensor.parallel import (
        ColwiseParallel,
        RowwiseParallel,
        parallelize_module,
    )

    torch.manual_seed(0)
    model = _Net()

    def forward(module: nn.Module) -> torch.Tensor:
        return module(torch.randn(4, 12, generator=torch.Generator().manual_seed(1)))

    def tensor_parallel(module: nn.Module, mesh) -> None:
        plan = {}
        for index in range(len(module.blocks)):
            plan[f"blocks.{index}.up"] = ColwiseParallel()
            plan[f"blocks.{index}.down"] = RowwiseParallel()
        parallelize_module(module, mesh, plan)

    return model, forward, tensor_parallel, ["blocks.0", "blocks.1", "blocks.2"]


def _shard_only_fixture(architecture: str) -> Tuple[nn.Module, type]:
    """Native tiny roots whose inputs need a pipeline; only their shards are compared."""
    if architecture == "wan_t2v":
        from test_wan_sequence_parallel_parity import _build, _with_lora

        from flow_factory.models.wan.wan2_t2v import Wan2_T2V_Adapter

        return _with_lora(_build()), Wan2_T2V_Adapter
    if architecture == "wan_i2v":
        from wan_i2v_cp_fixture import build_transformer, install_lora

        from flow_factory.models.wan.wan2_i2v import Wan2_I2V_Adapter

        return install_lora(build_transformer()), Wan2_I2V_Adapter
    if architecture.startswith("wan_"):
        from wan_remaining_cp_fixture import adapter_class, build_transformer, install_lora

        variant = architecture[len("wan_") :]
        return install_lora(build_transformer(variant), variant), adapter_class(variant)
    if architecture.startswith("flux2_"):
        from flux2_cp_fixture import adapter_class, build_transformer, install_lora

        variant = architecture[len("flux2_") :]
        return install_lora(build_transformer(variant), variant), adapter_class(variant)
    if architecture == "flux3_base":
        from flux3_cp_fixture import build_transformer, install_lora

        from flow_factory.models.flux.flux3 import Flux3Adapter

        return install_lora(build_transformer("base"), "base"), Flux3Adapter
    if architecture == "flux1_kontext":
        from test_region_entry_parity import _ARCHITECTURES, _with_lora

        from flow_factory.models.flux.flux1_kontext import Flux1KontextAdapter

        model, _, _ = _ARCHITECTURES["flux1"]()
        return _with_lora(model, Flux1KontextAdapter), Flux1KontextAdapter
    if architecture == "qwen_image":
        from qwen_cp_fixture import adapter_class, build_transformer, install_lora

        return install_lora(build_transformer()), adapter_class()
    if architecture.startswith("qwen_edit_"):
        from qwen_edit_cp_fixture import adapter_class, build_transformer, install_lora

        zero_cond_t = architecture == "qwen_edit_2511"
        return install_lora(build_transformer(zero_cond_t=zero_cond_t)), adapter_class()
    if architecture.startswith("ltx2"):
        from ltx2_cp_fixture import adapter_class, build_transformer, install_lora

        ltx23 = architecture == "ltx23_i2av"
        return install_lora(build_transformer(ltx23=ltx23), ltx23), adapter_class(ltx23)
    if architecture == "sensenova":
        from sensenova_cp_fixture import build_denoiser, install_lora

        from flow_factory.models.sensenova.sensenova import SenseNovaAdapter

        return install_lora(build_denoiser()), SenseNovaAdapter
    raise ValueError(f"unknown materialization fixture {architecture!r}")


def _fixture(
    architecture: str,
) -> Tuple[nn.Module, Optional[Callable[[nn.Module], torch.Tensor]], type]:
    if architecture == "h3":
        from h3_cp_fixture import adapter_class, build_transformer, install_lora, model_inputs

        def forward(module: nn.Module) -> torch.Tensor:
            video, audio = module(**model_inputs("t2va"))
            return torch.cat((video.flatten(), audio.flatten()))

        return install_lora(build_transformer()), forward, adapter_class("t2va")
    from test_region_entry_parity import _ARCHITECTURES, _with_lora

    if architecture not in _ARCHITECTURES:
        model, adapter = _shard_only_fixture(architecture)
        return model, None, adapter
    model, forward, adapter = _ARCHITECTURES[architecture]()
    return _with_lora(model, adapter), forward, adapter


def _sharding_units(native: nn.Module) -> List[str]:
    """Every item of each outermost ``ModuleList``, as FSDP2 per-block wrapping would see."""
    units: List[str] = []
    for name, module in native.named_modules():
        if isinstance(module, nn.ModuleList) and not any(
            name == unit or name.startswith(unit + ".") for unit in units
        ):
            units.extend(f"{name}.{index}" for index in range(len(module)))
    return units


def _adapter_case(
    architecture: str,
) -> Tuple[nn.Module, Callable[[nn.Module], torch.Tensor], Callable, List[str]]:
    from flow_factory.models.tensor_parallel import (
        apply_tensor_parallel,
        peel_peft_wrapper,
        resolve_parallel_plan,
    )

    model, forward, adapter = _fixture(architecture)
    native = peel_peft_wrapper(model)
    prefix = next(name for name, item in model.named_modules() if item is native)
    units = [f"{prefix}.{unit}" for unit in _sharding_units(native)]

    def tensor_parallel(module: nn.Module, mesh) -> None:
        apply_tensor_parallel(
            peel_peft_wrapper(module),
            mesh,
            resolve_parallel_plan(adapter, sequence_parallel=False),
        )

    return model, forward, tensor_parallel, units


def _case(name: str):
    return _plain_case() if name == "plain" else _adapter_case(name)


def _prepare(module: nn.Module, mesh, tensor_parallel: Callable, units: List[str]) -> nn.Module:
    from torch.distributed.fsdp import fully_shard

    tensor_parallel(module, mesh["tp"])
    for unit in units:
        fully_shard(module.get_submodule(unit), mesh=mesh["dp"])
    fully_shard(module, mesh=mesh["dp"])
    return module


def _materialize_worker(rank: int, store_path: str, result_dir: str, options: Dict[str, Any]):
    store = dist.FileStore(store_path, WORLD_SIZE)
    dist.init_process_group("gloo", store=store, rank=rank, world_size=WORLD_SIZE)
    result: Dict[str, Any] = {"rank": rank}
    try:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import DTensor

        from flow_factory.loading.materialization import detach_parameters_to_meta

        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "tp"))
        model, forward, tensor_parallel, units = _case(options["case"])
        source_rank = options["source_rank"]
        replicated = _prepare(copy.deepcopy(model), mesh, tensor_parallel, units)

        candidate = copy.deepcopy(model)
        if source_rank is not None and rank != source_rank:
            with torch.no_grad():
                for parameter in candidate.parameters():
                    parameter.zero_()
        source = detach_parameters_to_meta(candidate, source_rank=source_rank)
        candidate = _prepare(candidate, mesh, tensor_parallel, units)
        optimizer = torch.optim.AdamW(p for p in candidate.parameters() if p.requires_grad)
        before = [id(p) for group in optimizer.param_groups for p in group["params"]]
        report = source.materialize(candidate, torch.device("cpu"))
        after = [id(p) for p in candidate.parameters() if p.requires_grad]

        differing, placements = [], set()
        expected = dict(replicated.named_parameters())
        for name, parameter in candidate.named_parameters():
            reference = expected[name]
            placements.add(str(getattr(parameter, "placements", "plain")))
            local = parameter.to_local() if isinstance(parameter, DTensor) else parameter
            reference_local = reference.to_local() if isinstance(reference, DTensor) else reference
            if (
                getattr(parameter, "placements", None) != getattr(reference, "placements", None)
                or parameter.requires_grad != reference.requires_grad
                or not torch.equal(local, reference_local)
            ):
                differing.append(name)
        buffers_equal = all(
            torch.equal(buffer, dict(replicated.named_buffers())[name])
            for name, buffer in candidate.named_buffers()
        )
        result.update(
            differing=differing,
            parameters=report.parameters,
            placements=sorted(placements),
            optimizer_identity=before == after,
            buffers_equal=buffers_equal,
            output_equal=forward is None or torch.equal(forward(replicated), forward(candidate)),
            tied=(
                candidate.blocks[0].tied.weight is candidate.blocks[0].norm.weight
                if options["case"] == "plain"
                else True
            ),
        )
    except Exception as error:  # reported to the parent, which asserts on it
        result["error"] = f"{type(error).__name__}: {error}"
    Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result), encoding="utf-8")
    dist.destroy_process_group()


_CASES = [
    ("plain", None),
    ("plain", 0),
    ("sd3_5", None),
    ("z_image", None),
    ("h3", None),
    ("flux1", None),
    ("qwen_image_21", None),
    ("wan_t2v", None),
    ("wan_i2v", None),
    ("wan_t2v_dual", None),
    ("wan_t2v_expanded", None),
    ("wan_i2v_dual", None),
    ("wan_flf2v", None),
    ("flux1_kontext", None),
    ("flux2_dev", None),
    ("flux2_klein4b", None),
    ("flux2_klein9b", None),
    ("flux3_base", None),
    ("qwen_image", None),
    ("qwen_edit_2509", None),
    ("qwen_edit_2511", None),
    ("ltx20_t2av", None),
    ("ltx23_i2av", None),
    ("sensenova", None),
]


@requires_gloo
@pytest.mark.parametrize("case, source_rank", _CASES, ids=[f"{c}-src{s}" for c, s in _CASES])
def test_meta_preparation_materializes_the_replicated_local_shards(
    tmp_path: Path, case: str, source_rank
) -> None:
    import diffusers

    if case == "flux3_base" and not hasattr(diffusers, "Flux3Transformer2DModel"):
        pytest.skip("this diffusers has no Flux3Transformer2DModel")

    results = run_gloo_ranks(
        _materialize_worker, tmp_path, WORLD_SIZE, {"case": case, "source_rank": source_rank}
    )
    for result in results:
        assert "error" not in result, result["error"]
        assert result["differing"] == [], result
        assert result["optimizer_identity"] and result["buffers_equal"], result
        assert result["output_equal"] and result["tied"], result
    if case == "plain":
        assert "(_StridedShard(dim=0, sf=2), Shard(dim=0))" in results[0]["placements"]
        assert "(Shard(dim=0), Shard(dim=1))" in results[0]["placements"]


def test_detach_moves_parameters_and_keeps_buffers_and_aliases() -> None:
    from flow_factory.loading.materialization import detach_parameters_to_meta

    model = _Net()
    table = model.blocks[0].table
    frozen = model.blocks[2].up.weight.requires_grad
    source = detach_parameters_to_meta(model)

    assert all(parameter.is_meta for parameter in model.parameters())
    assert model.blocks[0].table is table and not table.is_meta
    assert model.blocks[0].tied.weight is model.blocks[0].norm.weight
    assert model.blocks[2].up.weight.requires_grad is frozen
    assert source.entries["blocks.0.tied.weight"] is source.entries["blocks.0.norm.weight"]


def test_materialize_upcasts_meta_parameters_to_their_final_dtype() -> None:
    from flow_factory.loading.materialization import detach_parameters_to_meta

    model = nn.Linear(3, 2).to(torch.bfloat16)
    loaded = model.weight.detach().clone()
    source = detach_parameters_to_meta(model)
    model.weight.data = model.weight.data.to(torch.float32)

    source.materialize(model, torch.device("cpu"))

    assert model.weight.dtype == torch.float32
    assert torch.equal(model.weight, loaded.to(torch.float32))


def test_materialize_rejects_missing_and_unconsumed_sources() -> None:
    from flow_factory.loading.materialization import detach_parameters_to_meta

    model = nn.Linear(3, 2)
    source = detach_parameters_to_meta(model)
    with pytest.raises(ValueError, match="no loaded source for prepared parameter 'weight'"):
        source.materialize(model, torch.device("cpu"), canonical_name=lambda name: f"x.{name}")

    model = nn.Sequential(nn.Linear(3, 2), nn.Linear(2, 2))
    source = detach_parameters_to_meta(model)
    del model[1]
    with pytest.raises(ValueError, match="not consumed .*'1.weight'"):
        source.materialize(model, torch.device("cpu"))


def test_materialize_rejects_resident_parameters_and_reuse() -> None:
    from flow_factory.loading.materialization import detach_parameters_to_meta

    model = nn.Linear(3, 2)
    source = detach_parameters_to_meta(model)
    with pytest.raises(ValueError, match="every parameter on the meta device"):
        source.materialize(nn.Linear(3, 2), torch.device("cpu"))
    source.materialize(model, torch.device("cpu"))
    with pytest.raises(RuntimeError, match="already released"):
        source.materialize(model, torch.device("cpu"))


def test_detach_rejects_distributed_and_valueless_parameters() -> None:
    from flow_factory.loading.materialization import detach_parameters_to_meta

    with torch.device("meta"):
        model = nn.Linear(3, 2)
    with pytest.raises(ValueError, match="no loaded value on the meta device"):
        detach_parameters_to_meta(model)
