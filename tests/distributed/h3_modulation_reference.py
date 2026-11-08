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

"""Released-width H3 modulation gradients and optimizer-state reference."""

import copy
import json
import traceback
from pathlib import Path

import torch
import torch.distributed as dist


class NativeModulationTables(torch.nn.Module):
    """Exercise every native modality/time table at the released channel widths."""

    def __init__(self):
        super().__init__()
        from diffusers.models.transformers.transformer_minimax_h3 import (
            MiniMaxH3AdaLayerNormModulation,
            MiniMaxH3AdaLayerNormOut,
        )

        self.main = MiniMaxH3AdaLayerNormModulation(2688, 5376)
        self.norm = MiniMaxH3AdaLayerNormOut(5376, 2688, 1e-5)

    def forward(self, temb, hidden, indices):
        tables = self.main(temb)
        rows = indices * 3 + torch.arange(indices.numel(), device=indices.device) % 3
        values = hidden + sum(
            (index + 1) / 6 * table.index_select(0, rows) for index, table in enumerate(tables)
        )
        return self.norm(values, temb, indices)


def run_native_modulation_reference(rank, store_path, result_dir, options):
    """Compare actual H3 TP/SP styles with an independent unsharded native pair."""
    dist.init_process_group("gloo", store=dist.FileStore(store_path, 2), rank=rank, world_size=2)
    result = {
        "rank": rank,
        "mode": options["mode"],
        "time_width": 2688,
        "hidden_width": 5376,
        "modulation_output_width": 96768,
        "objective": "element_mean_square",
    }
    try:
        from torch.distributed.device_mesh import DeviceMesh
        from torch.distributed.tensor import DTensor

        from flow_factory.models.minimax_h3.adapters import MiniMaxH3T2VAAdapter
        from flow_factory.models.tensor_parallel import (
            apply_tensor_parallel,
            peel_peft_wrapper,
            resolve_parallel_plan,
        )

        torch.manual_seed(71)
        reference = NativeModulationTables()
        module = copy.deepcopy(reference)
        if options["mode"] == "lora":
            from peft import LoraConfig, get_peft_model

            config = LoraConfig(
                r=2, lora_alpha=2, target_modules=["main.linear"], init_lora_weights=False
            )
            torch.manual_seed(72)
            reference = get_peft_model(reference, config)
            torch.manual_seed(72)
            module = get_peft_model(module, config)
        native = peel_peft_wrapper(module)
        mesh = DeviceMesh("cpu", torch.arange(2), mesh_dim_names=("tp",))
        plan = resolve_parallel_plan(MiniMaxH3T2VAAdapter, sequence_parallel=True)
        apply_tensor_parallel(
            native,
            mesh,
            {
                "main.linear": plan["transformer_blocks.*.adaln_proj.linear"],
                "main": plan["transformer_blocks.*.adaln_proj"],
                "norm.linear": plan["norm_out.linear"],
                "norm.norm": plan["norm_out.norm"],
            },
        )
        optimizers = [
            torch.optim.AdamW(model.parameters(), lr=0.001, eps=1e-4)
            for model in (reference, module)
        ]
        parameters = dict(reference.named_parameters())
        updates = []

        def full(value):
            return value.full_tensor() if isinstance(value, DTensor) else value

        for step in range(3):
            torch.manual_seed(80 + step)
            temb = torch.randn(4, 2688, requires_grad=True)
            actual_temb = temb.detach().clone().requires_grad_()
            hidden = torch.randn(13, 5376)
            indices = torch.arange(13) % 4
            # An uneven 6/7-row cut retains global modality tags because the
            # second rank starts at a multiple of the three native modalities.
            chosen = slice(0, 6) if rank == 0 else slice(6, 13)
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            expected = reference(temb, hidden, indices)
            actual = module(actual_temb, hidden[chosen], indices[chosen])
            torch.testing.assert_close(actual, expected[chosen], rtol=3e-4, atol=2e-6)
            expected.square().mean().backward()
            (actual.square().sum() / expected.numel()).backward()
            torch.testing.assert_close(actual_temb.grad, temb.grad, rtol=3e-4, atol=2e-6)
            count = 0
            residual = 0.0
            for name, parameter in module.named_parameters():
                target = parameters[name]
                assert (parameter.grad is None) == (target.grad is None), name
                if parameter.grad is None:
                    assert not parameter.requires_grad, name
                    continue
                grad = full(parameter.grad)
                torch.testing.assert_close(grad, target.grad, rtol=3e-4, atol=2e-6, msg=name)
                residual = max(residual, (grad - target.grad).abs().max().item())
                count += 1
            assert count > 0
            for optimizer in optimizers:
                optimizer.step()
            for name, parameter in module.named_parameters():
                target = parameters[name]
                torch.testing.assert_close(full(parameter), target, rtol=3e-4, atol=2e-6, msg=name)
                if parameter.requires_grad:
                    for key in ("step", "exp_avg", "exp_avg_sq"):
                        torch.testing.assert_close(
                            full(optimizers[1].state[parameter][key]),
                            optimizers[0].state[target][key],
                            rtol=3e-4,
                            atol=2e-6,
                            msg=name + key,
                        )
            updates.append({"step": step, "gradients": count, "max_gradient_residual": residual})

        # Remove only the main table's SP gradient sum, retaining its TP shards
        # and every native projection, then require the oracle to reject it.
        assert len(native.main._forward_hooks) == 1
        native.main._forward_hooks.clear()
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        expected = reference(temb, hidden, indices)
        actual = module(actual_temb, hidden[chosen], indices[chosen])
        expected.square().mean().backward()
        (actual.square().sum() / expected.numel()).backward()
        mismatches = sum(
            int(
                not torch.allclose(
                    full(parameter.grad), parameters[name].grad, rtol=3e-4, atol=2e-6
                )
            )
            for name, parameter in module.named_parameters()
            if parameter.grad is not None
        )
        assert mismatches > 0
        result.update(status="pass", updates=updates, missing_modulation_sum_mismatches=mismatches)
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result, indent=2))
        dist.destroy_process_group()
