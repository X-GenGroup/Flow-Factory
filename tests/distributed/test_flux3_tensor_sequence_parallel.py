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

"""Native Flux3 multi-block residual residency and full/LoRA numerical controls."""

import copy
import json
import traceback
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

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
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from flow_factory.models.flux.flux3 import Flux3Adapter
from flow_factory.models.tensor_parallel import (
    apply_tensor_parallel,
    peel_peft_wrapper,
    resolve_parallel_plan,
)


def _worker(rank, store_path, result_dir, options):
    cp_size = options.get("cp_size", 1)
    world_size = 2 * cp_size
    device = torch.device("cuda", rank) if options.get("cuda") else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dist.init_process_group(
        "nccl" if options.get("cuda") else "gloo",
        store=dist.FileStore(store_path, world_size),
        rank=rank,
        world_size=world_size,
    )
    result = {"rank": rank, **options}
    try:
        topology = init_device_mesh(device.type, (cp_size, 2), mesh_dim_names=("cp", "tp"))
        mesh = topology["tp"]
        variant, sp = options["variant"], options["sequence_parallel"]
        reference = build_transformer(variant).to(device=device, dtype=torch.float64)
        if options["mode"] == "lora":
            targets = ("norm_out.linear", "proj_out") if options.get("boundary_lora") else ()
            reference = install_lora(reference, variant, extra_targets=targets).double()
        reference.set_attention_backend("native")
        model = copy.deepcopy(reference)
        cls = Flux3Adapter
        if cp_size > 1:
            cls.validate_context_parallel_module(peel_peft_wrapper(model), cp_size, 2)
        if options["checkpoint"]:
            model.enable_gradient_checkpointing()
            reference.enable_gradient_checkpointing()
        apply_tensor_parallel(
            peel_peft_wrapper(model),
            mesh,
            resolve_parallel_plan(Flux3Adapter, sequence_parallel=sp),
        )
        native = peel_peft_wrapper(model)
        if cp_size > 1:
            from diffusers.models._modeling_parallel import ContextParallelConfig

            cp_mesh = DeviceMesh(
                device.type,
                topology.mesh.T.reshape(1, 2, 1, cp_size),
                mesh_dim_names=("data", "tensor", "ring", "ulysses"),
            )
            cls.configure_context_parallel_module(native)
            # The pinned native config resolver assumes an accelerator exists
            # even with an explicit CPU mesh. Change discovery only for this
            # Gloo control; native hooks, attention and collective math remain.
            with ExitStack() as setup:
                if device.type == "cpu":
                    setup.enter_context(
                        patch("torch._C._get_accelerator", return_value=torch.device("cpu"))
                    )
                    setup.enter_context(patch("torch.accelerator.device_count", return_value=1))
                cls.install_context_parallel_module(
                    native,
                    ContextParallelConfig(ring_degree=1, ulysses_degree=cp_size, mesh=cp_mesh),
                )
            boundaries = cls.compose_context_parallel_sequence_boundaries(native)
            assert len(boundaries) == 1

        def complete_cp_gradients():
            # Native CP output backward supplies a local slice. FSDP performs
            # this parameter-gradient reduction in the eight-rank GPU control.
            if cp_size > 1:
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        dist.all_reduce(parameter.grad.to_local(), group=topology["cp"].get_group())

        observed = set()
        edges = set()
        modulation = set()
        if sp:

            def record_modulation(_module, _args, output, *, name):
                assert output.untyped_storage().nbytes() == output.numel() * output.element_size()
                modulation.add((name, output.shape[1]))

            for name, module in native.early_stream_modulations.items():
                module.register_forward_hook(
                    lambda module, args, output, name=name: record_modulation(
                        module, args, output, name=name
                    )
                )
        for label, blocks in (
            ("image", native.mode_blocks["image"]),
            ("reference", native.mode_blocks["image_cond"]),
            ("text", native.txt_mode_blocks),
            ("joint", native.single_transformer_blocks),
        ):
            assert len(blocks) >= 2
            for index, block in enumerate(blocks):
                block.norm.register_forward_pre_hook(
                    lambda _module, args, name=(label, index): observed.add(
                        (*name, args[0].shape[1])
                    )
                )
                block.register_forward_hook(
                    lambda _module, _args, output, name=(label, index): edges.add(
                        (
                            *name,
                            (
                                tuple(x.shape[1] for x in output)
                                if isinstance(output, tuple)
                                else (output.shape[1],)
                            ),
                        )
                    )
                )
        native.proj_out.register_forward_pre_hook(
            lambda _module, args: observed.add(("output", 0, args[0].shape[1]))
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-4)
        actual_parameters, expected_parameters = dict(model.named_parameters()), dict(
            reference.named_parameters()
        )
        assert actual_parameters.keys() == expected_parameters.keys()
        for step in range(3):
            inputs = fixed_inputs(variant, step=step, device=device, dtype=torch.float64)
            input_names = (
                "latents",
                "image_latents",
                "prompt_embeds",
                "negative_prompt_embeds",
                "t",
            )
            if variant == "distilled":
                input_names += ("t_next",)
            if step == 0:
                for name in input_names:
                    if name in inputs:
                        inputs[name].requires_grad_()
            reference_inputs = {
                name: (
                    value.detach().clone().requires_grad_()
                    if isinstance(value, torch.Tensor) and value.requires_grad
                    else value
                )
                for name, value in inputs.items()
            }
            actual, expected = native_velocity(model, inputs, variant), native_velocity(
                reference, reference_inputs, variant
            )
            torch.testing.assert_close(actual, expected, rtol=2e-8, atol=1e-10)
            actual.square().mean().backward()
            complete_cp_gradients()
            expected.square().mean().backward()
            if step == 0:
                for name in input_names:
                    if name not in inputs:
                        continue
                    gradient, expected_gradient = inputs[name].grad, reference_inputs[name].grad
                    assert gradient is not None and expected_gradient is not None, name
                    if cp_size > 1:
                        dist.all_reduce(gradient, group=topology["cp"].get_group())
                    torch.testing.assert_close(
                        gradient, expected_gradient, rtol=2e-7, atol=1e-9, msg=name
                    )
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
        factor = (2 if sp else 1) * cp_size
        assert observed == {
            (label, index, tokens // factor)
            for index in range(2)
            for label, tokens in (
                ("image", 16),
                ("reference", 8),
                ("text", 80),
                ("joint", 24),
                ("joint", 80),
            )
        } | {("output", 0, 16 // factor)}, observed
        assert edges == {
            (label, index, tuple(token // (factor if index == 0 else cp_size) for token in tokens))
            for index in range(2)
            for label, tokens in (
                ("image", (16,)),
                ("reference", (8,)),
                ("text", (80,)),
                ("joint", (80, 24)),
            )
        }, edges
        assert modulation == (
            {("image", 8 // cp_size), ("image_cond", 4 // cp_size), ("txt", 40 // cp_size)}
            if sp
            else set()
        )
        assert native.single_transformer_blocks[0].attn.heads == 12
        assert native.config.num_attention_heads == 24
        canonical = {
            name: (value.full_tensor() if hasattr(value, "full_tensor") else value).detach().cpu()
            for name, value in model.state_dict().items()
        }
        reloaded = build_transformer(variant).to(device=device, dtype=torch.float64)
        if options["mode"] == "lora":
            reloaded = install_lora(reloaded, variant, extra_targets=targets).double()
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
        complete_cp_gradients()
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
            residual_edges=sorted(edges),
            early_modulation=sorted(modulation),
            canonical_parameter_names=True,
            native_head_count=24,
            tp_size=mesh.size(),
            cp_size=cp_size,
            input_gradient_names=[name for name in input_names if name in inputs],
        )
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result, indent=2))
        dist.destroy_process_group()


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("mode", ["full", "lora"])
@pytest.mark.parametrize(
    "sequence_parallel,checkpoint",
    [(False, False), (True, False), (True, True)],
    ids=["tp", "eager-sp", "checkpointed-sp"],
)
def test_native_flux3_tensor_sequence_gradients(
    tmp_path, variant, mode, sequence_parallel, checkpoint
):
    results = run_gloo_ranks(
        _worker,
        tmp_path,
        2,
        {
            "variant": variant,
            "mode": mode,
            "sequence_parallel": sequence_parallel,
            "checkpoint": checkpoint,
        },
    )
    assert all(row.get("status") == "pass" and "error" not in row for row in results), results


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="TP2 requires two CUDA ranks")
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("mode", ["full", "lora"])
@pytest.mark.parametrize("checkpoint", [False, True], ids=["eager", "checkpointed"])
def test_cuda_flux3_continuous_sequence_gradients(tmp_path, variant, mode, checkpoint):
    results = run_gloo_ranks(
        _worker,
        tmp_path,
        2,
        {
            "variant": variant,
            "mode": mode,
            "sequence_parallel": True,
            "checkpoint": checkpoint,
            "cuda": True,
        },
    )
    assert all(row.get("status") == "pass" and "error" not in row for row in results), results


@pytest.mark.parametrize("checkpoint", [False, True], ids=["eager", "checkpointed"])
@pytest.mark.parametrize(
    "cuda",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.skipif(
                torch.cuda.device_count() < 2, reason="TP2 requires two CUDA ranks"
            ),
        ),
    ],
    ids=["cpu", "cuda"],
)
def test_flux3_custom_output_lora_gradients(tmp_path, checkpoint, cuda):
    results = run_gloo_ranks(
        _worker,
        tmp_path,
        2,
        {
            "variant": "distilled",
            "mode": "lora",
            "sequence_parallel": True,
            "checkpoint": checkpoint,
            "boundary_lora": True,
            "cuda": cuda,
        },
    )
    assert all(row.get("status") == "pass" and "error" not in row for row in results), results


@pytest.mark.parametrize("mode", ["full", "lora"])
@pytest.mark.parametrize("checkpoint", [False, True], ids=["eager", "checkpointed"])
def test_flux3_native_tp_sp_cp_gradients(tmp_path, mode, checkpoint):
    results = run_gloo_ranks(
        _worker,
        tmp_path,
        4,
        {
            "variant": "distilled",
            "mode": mode,
            "sequence_parallel": True,
            "checkpoint": checkpoint,
            "boundary_lora": mode == "lora",
            "cp_size": 2,
        },
    )
    assert all(row.get("status") == "pass" and "error" not in row for row in results), results
