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

"""Native packed Flux3 full/LoRA TP and checkpointed token-SP numerical controls."""

import copy
import json
import traceback
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from conftest import run_gloo_ranks
from flux3_cp_fixture import (
    VARIANTS,
    build_transformer,
    fixed_inputs,
    install_lora,
    native_velocity,
)
from torch.distributed.device_mesh import init_device_mesh

from flow_factory.models.flux.flux3 import Flux3Adapter
from flow_factory.models.tensor_parallel import (
    apply_tensor_parallel,
    peel_peft_wrapper,
    resolve_parallel_plan,
)


def _worker(rank, store_path, result_dir, options):
    dist.init_process_group("gloo", store=dist.FileStore(store_path, 2), rank=rank, world_size=2)
    result = {"rank": rank, **options}
    try:
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("tp",))
        variant, sp = options["variant"], options["sequence_parallel"]
        reference = build_transformer(variant).double()
        if options["mode"] == "lora":
            reference = install_lora(reference, variant).double()
        reference.set_attention_backend("native")
        model = copy.deepcopy(reference)
        if sp:
            model.enable_gradient_checkpointing()
            reference.enable_gradient_checkpointing()
        apply_tensor_parallel(
            peel_peft_wrapper(model),
            mesh,
            resolve_parallel_plan(Flux3Adapter, sequence_parallel=sp),
        )
        native = peel_peft_wrapper(model)
        observed = set()
        for label, module in (
            ("image", native.mode_blocks["image"][0].norm),
            ("reference", native.mode_blocks["image_cond"][0].norm),
            ("text", native.txt_mode_blocks[0].norm),
            ("joint", native.single_transformer_blocks[0].norm),
        ):
            module.register_forward_pre_hook(
                lambda _module, args, name=label: observed.add((name, args[0].shape[1]))
            )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-4)
        actual_parameters, expected_parameters = dict(model.named_parameters()), dict(
            reference.named_parameters()
        )
        assert actual_parameters.keys() == expected_parameters.keys()
        for step in range(3):
            inputs = fixed_inputs(variant, step=step, dtype=torch.float64)
            actual, expected = native_velocity(model, inputs, variant), native_velocity(
                reference, inputs, variant
            )
            torch.testing.assert_close(actual, expected, rtol=2e-8, atol=1e-10)
            actual.square().mean().backward()
            expected.square().mean().backward()
            active = 0
            for name, target in expected_parameters.items():
                parameter = actual_parameters[name]
                if target.requires_grad:
                    assert target.grad is not None and parameter.grad is not None, name
                    torch.testing.assert_close(
                        parameter.grad.full_tensor(), target.grad, rtol=2e-7, atol=1e-9, msg=name
                    )
                    active += int(bool(torch.count_nonzero(target.grad)))
                else:
                    assert target.grad is None and parameter.grad is None, name
            assert active == sum(value.requires_grad for value in expected_parameters.values())
            optimizer.step()
            reference_optimizer.step()
            for name, target in expected_parameters.items():
                parameter = actual_parameters[name]
                torch.testing.assert_close(
                    parameter.full_tensor(), target, rtol=2e-7, atol=1e-9, msg=name
                )
                if target.requires_grad:
                    for key in ("exp_avg", "exp_avg_sq"):
                        torch.testing.assert_close(
                            optimizer.state[parameter][key].full_tensor(),
                            reference_optimizer.state[target][key],
                            rtol=2e-7,
                            atol=1e-9,
                            msg=name + key,
                        )
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
        factor = 2 if sp else 1
        assert observed == {
            ("image", 16 // factor),
            ("reference", 8 // factor),
            ("text", 80 // factor),
            ("joint", 24 // factor),
            ("joint", 80 // factor),
        }, observed
        assert native.single_transformer_blocks[0].attn.heads == 12
        assert native.config.num_attention_heads == 24
        canonical = {
            name: (value.full_tensor() if hasattr(value, "full_tensor") else value).detach().cpu()
            for name, value in model.state_dict().items()
        }
        reloaded = build_transformer(variant).double()
        if options["mode"] == "lora":
            reloaded = install_lora(reloaded, variant).double()
        reloaded.load_state_dict(canonical, strict=True)
        with torch.no_grad():
            torch.testing.assert_close(
                native_velocity(reloaded, inputs, variant),
                native_velocity(reference, inputs, variant),
                rtol=2e-7,
                atol=1e-9,
            )
        no_reference = {**inputs, "image_latents": None, "image_latent_ids": None}
        actual = native_velocity(model, no_reference, variant)
        expected = native_velocity(reference, no_reference, variant)
        torch.testing.assert_close(actual, expected, rtol=2e-8, atol=1e-10)
        actual.square().mean().backward()
        expected.square().mean().backward()
        inactive = {}
        for name, target in expected_parameters.items():
            parameter = actual_parameters[name]
            if target.requires_grad:
                assert (target.grad is None) == (".image_cond." in name), name
            assert (parameter.grad is None) == (target.grad is None), name
            if target.grad is None:
                inactive[name] = (
                    parameter.full_tensor().detach().clone(),
                    {
                        key: (
                            value.full_tensor().detach().clone()
                            if hasattr(value, "full_tensor")
                            else value.clone()
                        )
                        for key, value in optimizer.state.get(parameter, {}).items()
                    },
                )
            else:
                torch.testing.assert_close(
                    parameter.grad.full_tensor(), target.grad, rtol=2e-7, atol=1e-9, msg=name
                )
        assert any(".image_cond." in name for name in inactive)
        optimizer.step()
        reference_optimizer.step()
        for name, target in expected_parameters.items():
            parameter = actual_parameters[name]
            torch.testing.assert_close(
                parameter.full_tensor(), target, rtol=2e-7, atol=1e-9, msg=name
            )
            for key, value in optimizer.state.get(parameter, {}).items():
                actual = value.full_tensor() if hasattr(value, "full_tensor") else value
                torch.testing.assert_close(
                    actual,
                    reference_optimizer.state[target][key],
                    rtol=2e-7,
                    atol=1e-9,
                    msg=name + key,
                )
            if name in inactive:
                before, state = inactive[name]
                torch.testing.assert_close(
                    parameter.full_tensor(), before, rtol=0, atol=0, msg=name
                )
                for key, value in state.items():
                    after = optimizer.state[parameter][key]
                    torch.testing.assert_close(
                        after.full_tensor() if hasattr(after, "full_tensor") else after,
                        value,
                        rtol=0,
                        atol=0,
                        msg=name + key,
                    )
        result.update(
            status="pass",
            updates=3,
            t2i_warm_state_control=True,
            active_parameters=active,
            observed=sorted(observed),
            canonical_parameter_names=True,
            native_head_count=24,
            tp_size=mesh.size(),
        )
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result, indent=2))
        dist.destroy_process_group()


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("mode", ["full", "lora"])
@pytest.mark.parametrize("sequence_parallel", [False, True], ids=["tp", "checkpointed-sp"])
def test_native_flux3_tensor_sequence_gradients(tmp_path, variant, mode, sequence_parallel):
    results = run_gloo_ranks(
        _worker,
        tmp_path,
        2,
        {"variant": variant, "mode": mode, "sequence_parallel": sequence_parallel},
    )
    assert all(row.get("status") == "pass" and "error" not in row for row in results), results
