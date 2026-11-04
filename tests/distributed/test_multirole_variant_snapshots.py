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

"""Multi-role parameter snapshots on a tensor-parallel mesh.

TDM-R1 keeps two trainable roles and a slow EMA snapshot of one of them,
advanced every step and installed around a trust-region forward. A snapshot is a
detached clone of live parameters, so once tensor parallelism makes those
parameters DTensors the snapshot becomes one too, and every operation the path
performs -- ``clone``, ``copy_``, ``lerp_``, ``to(device=, dtype=)`` -- has to
hold on a sharded tensor and on the *composed* 2-D mesh.

Three things are pinned here, each a separate way the path could go wrong while
staying quiet:

* the snapshot carries its parameter's mesh and placements. A snapshot that
  degraded to a plain local tensor would make ``lerp_`` mix a shard with a whole
  tensor, which raises -- but one that kept DTensor-ness on the *wrong* mesh
  would not;
* the EMA arithmetic matches an unsharded reference after several updates, which
  a per-rank-only update would not;
* role-scoped clipping sees one mesh. ``role_optimization`` hands
  ``clip_grad_norm_`` only the active role's parameters, and Accelerate's
  DTensor path groups them by ``grad.device_mesh``; a role whose module was left
  unsharded lands in a second group and raises "cross-mesh operation".

Exact-resume persistence is covered separately by
``test_parameter_state_resume.py`` on actual CUDA FSDP2 and FSDP2+TP backends.
This suite isolates in-memory snapshot arithmetic and clipping.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from conftest import requires_gloo, run_gloo_ranks
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

HEADS, HEAD_DIM = 4, 8
DIM = HEADS * HEAD_DIM
SEQ, BATCH = 8, 2
DECAYS = (0.9, 0.5, 0.99)
SNAPSHOT = "slow_surrogate"


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.heads = HEADS
        self.to_q = nn.Linear(DIM, DIM, bias=False)
        self.to_out = nn.Linear(DIM, DIM, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.to_out(F.gelu(self.to_q(hidden_states)))


class _Denoiser(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_Block() for _ in range(2)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return hidden_states


class _StubAdapter:
    """The least an adapter has to expose for the registry to materialize."""

    trainable_component_names = ("transformer",)
    target_module_map = {"transformer": "all"}

    def __init__(self, module: nn.Module) -> None:
        self._module = module

    def get_component(self, name: str) -> nn.Module:
        return self._module


def _plan() -> Dict[str, Any]:
    return {"blocks.*.to_q": ColwiseParallel(), "blocks.*.to_out": RowwiseParallel()}


def _registry():
    """Two full trainable roles under distinct routes, as TDM-R1 declares them."""
    from flow_factory.models.variants import ComponentVariantRegistry, ComponentVariantSpec

    torch.manual_seed(1234)
    registry = ComponentVariantRegistry(_StubAdapter(_Denoiser()))
    registry.declare(
        ComponentVariantSpec(
            name="generator",
            storage_mode="full",
            component_routes={"transformer": "transformer"},
        )
    )
    registry.declare(
        ComponentVariantSpec(
            name="surrogate",
            storage_mode="full",
            component_routes={"transformer": "transformer_surrogate"},
        )
    )
    registry.materialize(("generator", "surrogate"))
    return registry


def _whole(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor


def _perturb(registry: Any, update: int) -> None:
    """Move the live surrogate weights by a rank-independent amount.

    One additive scalar per (update, registration ordinal). Adding the same
    number everywhere means a shard receives exactly its slice of the global
    perturbation, so the comparison against the unsharded reference isolates the
    snapshot arithmetic instead of measuring the perturbation itself.
    """
    for ordinal, record in enumerate(registry.parameter_records("surrogate")):
        parameter = record.parameter
        local = parameter.to_local() if isinstance(parameter, DTensor) else parameter
        with torch.no_grad():
            local.add_(0.01 * (update + 1) * (ordinal + 1))


def _worker(rank: int, store_path: str, result_dir: str, options: Dict[str, Any]) -> None:
    from flow_factory.models.tensor_parallel import apply_tensor_parallel

    dp_size, tp_size = options["dp_size"], options["tp_size"]
    world_size = dp_size * tp_size
    dist.init_process_group(
        "gloo", store=dist.FileStore(store_path, world_size), rank=rank, world_size=world_size
    )
    result: Dict[str, Any] = {"rank": rank}
    try:
        mesh = init_device_mesh("cpu", (dp_size, tp_size), mesh_dim_names=("dp_shard", "tp"))
        registry = _registry()
        routes = registry.bundle_members()

        # What `_apply_tensor_parallel` does: every variant's route, once per
        # distinct module.
        seen: set[int] = set()
        for variant_name in registry.variant_names:
            module = routes[registry.resolve_route(variant_name, "transformer")]
            if id(module) in seen:
                continue
            seen.add(id(module))
            if tp_size > 1:
                apply_tensor_parallel(module, mesh["tp"], _plan())
        if dp_size > 1:
            for module in routes.values():
                fully_shard(module, mesh=mesh["dp_shard"])
        result["sharded_routes"] = len(seen)

        # Both backends replace the Parameter objects recorded at
        # materialization, so the trainer rebinds before any snapshot is taken.
        registry.rebind_parameters(routes)

        registry.add_snapshot("surrogate", SNAPSHOT)
        stored = registry._get_snapshot(SNAPSHOT)["parameters"]
        result["spec_mismatches"] = [
            key
            for key, snapshot in stored.items()
            for parameter in [
                next(
                    r.parameter
                    for r in registry.parameter_records("surrogate")
                    if f"{r.component_name}.{r.parameter_name}" == key
                )
            ]
            if isinstance(parameter, DTensor) != isinstance(snapshot, DTensor)
            or (
                isinstance(parameter, DTensor)
                and (
                    parameter.device_mesh != snapshot.device_mesh
                    or parameter.placements != snapshot.placements
                )
            )
        ]

        for update, decay in enumerate(DECAYS):
            _perturb(registry, update)
            registry.update_snapshot(SNAPSHOT, decay)
        result["ema"] = {
            key: round(float(_whole(value).double().sum()), 6) for key, value in stored.items()
        }

        before = {
            f"{r.component_name}.{r.parameter_name}": _whole(r.parameter.detach()).clone()
            for r in registry.parameter_records("surrogate")
        }
        installed: Dict[str, torch.Tensor] = {}
        with registry.use_snapshot(SNAPSHOT):
            for record in registry.parameter_records("surrogate"):
                key = f"{record.component_name}.{record.parameter_name}"
                installed[key] = _whole(record.parameter.detach()).clone()
        result["install_error"] = max(
            float((installed[key] - _whole(value)).abs().max()) for key, value in stored.items()
        )
        result["restore_error"] = max(
            float((_whole(r.parameter.detach()) - before[key]).abs().max())
            for r in registry.parameter_records("surrogate")
            for key in [f"{r.component_name}.{r.parameter_name}"]
        )

        # Role-scoped clipping: one mesh for the active role, and a norm that
        # reduces over every mesh dimension.
        registry.activate("surrogate")
        hidden = torch.randn(BATCH, SEQ, DIM, generator=torch.Generator().manual_seed(7))
        routes[registry.resolve_route("surrogate", "transformer")](hidden).pow(2).mean().backward()
        records = [
            r for r in registry.parameter_records("surrogate") if r.parameter.grad is not None
        ]
        result["grad_meshes"] = sorted(
            {str(getattr(r.parameter.grad, "device_mesh", None)) for r in records}
        )
        # Across *every* role, not just the active one: `ModelBundle` hands
        # `prepare` one root over all routes, so a route left unsharded makes
        # that root mix meshes. The active role alone is homogeneous either way
        # and would not notice.
        result["parameter_meshes"] = sorted(
            {
                str(getattr(parameter, "device_mesh", None))
                for module in routes.values()
                for parameter in module.parameters()
            }
        )
        result["inactive_role_grads"] = [
            f"{r.component_name}.{r.parameter_name}"
            for r in registry.parameter_records("generator")
            if r.parameter.grad is not None
        ]
        truth = float(
            torch.stack([_whole(r.parameter.grad).double().pow(2).sum() for r in records])
            .sum()
            .sqrt()
        )
        reported = float(
            torch.nn.utils.clip_grad_norm_([r.parameter for r in records], max_norm=1.0e9)
        )
        result["clip_relative_error"] = abs(reported - truth) / max(truth, 1e-12)
    except Exception as error:  # reported to the parent, which asserts on it
        result["error"] = f"{type(error).__name__}: {str(error)[:400]}"
    Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result), encoding="utf-8")
    dist.destroy_process_group()


def _run(tmp_path: Path, **options: Any) -> List[Dict[str, Any]]:
    """Spawn the ranks this file's worker needs; the harness is shared."""
    options.setdefault("dp_size", 1)
    return run_gloo_ranks(
        _worker, tmp_path, options["dp_size"] * options["tp_size"], options, timeout=300
    )


pytestmark = requires_gloo


@pytest.mark.parametrize(
    ("dp_size", "tp_size"),
    [(1, 2), (2, 2)],
    ids=["tp2", "dp2_tp2"],
)
def test_a_role_snapshot_survives_tensor_parallelism(
    tmp_path: Path, dp_size: int, tp_size: int
) -> None:
    """The EMA of a sharded role equals the EMA of the same role unsharded."""
    reference = _run(tmp_path / "ref", tp_size=1)
    sharded = _run(tmp_path / f"dp{dp_size}tp{tp_size}", dp_size=dp_size, tp_size=tp_size)

    assert "error" not in reference[0], reference[0]
    for row in sharded:
        assert "error" not in row, row
        # Both roles' modules are distinct objects under a full finetune, so both
        # are sharded; a shared LoRA route would report one.
        assert row["sharded_routes"] == 2, row
        assert row["spec_mismatches"] == [], row
        assert row["ema"] == reference[0]["ema"], row
        assert row["install_error"] == 0.0, row
        assert row["restore_error"] == 0.0, row


@pytest.mark.parametrize(
    ("dp_size", "tp_size"),
    [(1, 2), (2, 2)],
    ids=["tp2", "dp2_tp2"],
)
def test_role_scoped_clipping_sees_one_mesh(tmp_path: Path, dp_size: int, tp_size: int) -> None:
    """Every route of the active role lands on the same mesh, and the norm is exact.

    ``role_optimization`` passes only the active role's parameters, and
    Accelerate's DTensor-aware clip groups them by ``grad.device_mesh``. Leaving
    one role's module unsharded produces a second group and raises "DTensor does
    not support cross-mesh operation" at the first optimizer step.
    """
    rows = _run(tmp_path, dp_size=dp_size, tp_size=tp_size)
    for row in rows:
        assert "error" not in row, row
        assert len(row["grad_meshes"]) == 1, row
        assert len(row["parameter_meshes"]) == 1, row
        # One role's backward must not deposit gradients on the other's copy.
        assert row["inactive_role_grads"] == [], row
        assert row["clip_relative_error"] < 1e-6, row
