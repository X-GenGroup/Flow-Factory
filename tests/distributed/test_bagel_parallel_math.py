# Copyright 2026 Jayce-Ping
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""High precision packed attention oracle, independent of Bagel's BF16 projections."""

import json
import traceback
from pathlib import Path

import torch
import torch.distributed as dist
from conftest import requires_gloo, run_gloo_ranks


def _worker(rank, store_path, result_dir, options):
    result = {"rank": rank}
    dist.init_process_group("gloo", init_method=f"file://{store_path}", rank=rank, world_size=2)
    try:
        from diffusers.hooks.context_parallel import EquipartitionSharder
        from diffusers.models._modeling_parallel import ContextParallelConfig, ParallelConfig
        from torch.distributed.device_mesh import init_device_mesh

        from flow_factory.models.bagel._parallel import PackedLayout, packed_attention_kernel

        mesh = init_device_mesh("cpu", (1, 1, 2), mesh_dim_names=("replicate", "ring", "ulysses"))
        cp = ContextParallelConfig(ring_degree=1, ulysses_degree=2, mesh=mesh)
        native = ParallelConfig(context_parallel_config=cp)
        native.setup(rank, 2, torch.device("cpu"), mesh=mesh)
        # Actual Bagel Q/KV ratio and head width, odd prefixes and query padding.
        layout = PackedLayout((0, 7, 9), 2, 2)
        lengths = (3, 5, 9)
        generator = torch.Generator().manual_seed(711)
        values = [
            torch.randn(16, heads, 128, dtype=torch.float64, generator=generator).requires_grad_()
            for heads in (28, 4, 4)
        ]
        reference = [value.detach().clone().requires_grad_() for value in values]
        prefix = [
            torch.randn(sum(lengths), 4, 128, dtype=torch.float64, generator=generator)
            for _ in range(2)
        ]
        local = [
            EquipartitionSharder.shard(layout.reorder(value), 0, cp._flattened_mesh)
            for value in values
        ]
        output = packed_attention_kernel(*local, *prefix, lengths, layout, cp, native)
        actual = layout.restore(EquipartitionSharder.unshard(output, 0, cp._flattened_mesh))
        independent = []
        for q, k, v, pk, pv in zip(
            *(value.split(layout.lengths) for value in reference),
            *(value.split(lengths) for value in prefix),
        ):
            key = torch.cat([pk, k]).repeat_interleave(7, dim=1)
            value = torch.cat([pv, v]).repeat_interleave(7, dim=1)
            independent.append(
                torch.nn.functional.scaled_dot_product_attention(
                    q.transpose(0, 1)[None], key.transpose(0, 1)[None], value.transpose(0, 1)[None]
                )[0].transpose(0, 1)
            )
        expected = torch.cat(independent)
        torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)
        weight = torch.randn(expected.shape, dtype=torch.float64, generator=generator)
        (actual * weight).sum().backward()
        (expected * weight).sum().backward()
        result["derivatives"] = {}
        for name, actual_input, reference_input in zip(
            ("query", "key", "value"), values, reference
        ):
            gradient = actual_input.grad.clone()
            dist.all_reduce(gradient)
            torch.testing.assert_close(gradient, reference_input.grad, rtol=2e-11, atol=2e-11)
            assert not torch.allclose(gradient / 2, reference_input.grad, rtol=0.01, atol=1e-10)
            result["derivatives"][name] = float((gradient - reference_input.grad).abs().max())
        result.update(
            status="pass",
            precision="float64",
            q_heads=28,
            kv_heads=4,
            head_dim=128,
            omitted_cp_factor_rejected=True,
        )
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result, indent=2))
        dist.destroy_process_group()


@requires_gloo
def test_bagel_native_packed_attention_matches_float64_dense_oracle(tmp_path):
    results = run_gloo_ranks(_worker, tmp_path, 2, {})
    assert all(row.get("status") == "pass" and "error" not in row for row in results), results
