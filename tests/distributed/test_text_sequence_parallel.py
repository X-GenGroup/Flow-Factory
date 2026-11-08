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

"""Native SD3/Qwen text SP, complete derivatives, three updates and canonical state."""

import copy
import json
import traceback
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from conftest import run_gloo_ranks
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from flow_factory.models.tensor_parallel import (
    apply_tensor_parallel,
    peel_peft_wrapper,
    resolve_parallel_plan,
)


def _full(value):
    return value.full_tensor() if isinstance(value, DTensor) else value


def _case(family, mode):
    from peft import LoraConfig, get_peft_model

    if family.startswith("sd35"):
        from test_sd35_context_parallel import build_transformer

        from flow_factory.models.stable_diffusion.sd3_5 import SD3_5Adapter

        adapter = SD3_5Adapter
        model = build_transformer(family.split("-")[1])
    else:
        from test_qwen_image_sequence_parallel_parity import _build

        from flow_factory.models.qwen_image.qwen_image import QwenImageAdapter
        from flow_factory.models.qwen_image.qwen_image_edit_plus import QwenImageEditPlusAdapter

        adapter = QwenImageEditPlusAdapter if family == "qwen-edit" else QwenImageAdapter
        model = _build(family == "qwen-edit")
    if mode != "full":
        targets = adapter.default_target_modules.fget(object.__new__(adapter))
        if mode == "expanded_lora":
            targets = [
                name
                for name, child in model.named_modules()
                if isinstance(child, torch.nn.Linear)
                and (name == "proj_out" or name.startswith(("transformer_blocks.", "norm_out.")))
            ]
        torch.manual_seed(143)
        model = get_peft_model(
            model, LoraConfig(r=2, lora_alpha=2, target_modules=targets, init_lora_weights=False)
        )
    return adapter, model


def _inputs(family, step, device, dtype):
    generator = torch.Generator().manual_seed(991 + step)
    if family.startswith("sd35"):
        from test_sd35_context_parallel import model_inputs

        values = model_inputs(step)
        values["encoder_hidden_states"] = values["encoder_hidden_states"][:, :5].clone()
    else:
        image_tokens = 32 if family == "qwen-edit" else 16
        values = dict(
            hidden_states=torch.randn(2, image_tokens, 16, generator=generator),
            encoder_hidden_states=torch.randn(2, 7, 32, generator=generator),
            encoder_hidden_states_mask=torch.tensor(
                [[1, 0, 1, 1, 0, 1, 1], [1, 1, 0, 0, 0, 0, 1]], dtype=torch.bool
            ),
            timestep=torch.tensor([0.5, 0.4]),
            img_shapes=[[(1, 4, 4)] * (2 if family == "qwen-edit" else 1)] * 2,
            return_dict=False,
        )
    return {
        name: (
            value.to(device=device, dtype=dtype if value.is_floating_point() else value.dtype)
            if torch.is_tensor(value)
            else value
        )
        for name, value in values.items()
    }


def _assert_predictions(actual, expected, reference, inputs, tolerance, fp64_oracle):
    if not fp64_oracle:
        torch.testing.assert_close(actual, expected, **tolerance)
        return None
    # Different head/token GEMM decompositions round near-zero Qwen outputs
    # differently. Compare both FP32 paths to the same independent native FP64
    # arithmetic, rather than relax gradient or optimizer-state gates.
    with torch.no_grad():
        precise = copy.deepcopy(reference).double()
        precise_inputs = {
            name: value.double() if torch.is_tensor(value) and value.is_floating_point() else value
            for name, value in inputs.items()
        }
        golden = precise(**precise_inputs)[0]
        scale = golden.abs().max().item()
        dense_error = (expected.detach().double() - golden).abs().max().item()
        parallel_error = (actual.detach().double() - golden).abs().max().item()
        eps = torch.finfo(expected.dtype).eps
        assert (
            dense_error <= 64 * eps * scale + eps
        ), "native FP32 oracle exceeds its rounding envelope"
        budget = 2 * dense_error + 2 * eps * scale
        assert parallel_error <= budget, (parallel_error, dense_error, budget)
        torch.testing.assert_close(actual, expected, rtol=tolerance["rtol"], atol=budget)
        return {
            "dense_fp64_error": dense_error,
            "parallel_fp64_error": parallel_error,
            "budget": budget,
        }


def _worker(rank, store_path, result_dir, options):
    cp_size = options.get("cp_size", 1)
    world = 2 * cp_size
    cuda = options.get("cuda", False)
    device = torch.device("cuda", rank) if cuda else torch.device("cpu")
    dtype = torch.float32 if cuda else torch.float64
    if cuda:
        torch.cuda.set_device(rank)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group(
        "nccl" if cuda else "gloo",
        store=dist.FileStore(store_path, world),
        rank=rank,
        world_size=world,
    )
    result = {"rank": rank, "cp_size": cp_size, "scenarios": []}
    try:
        mesh = DeviceMesh(
            device.type,
            torch.arange(world).reshape(cp_size, 2).T.unsqueeze(1),
            mesh_dim_names=("tp", "ring", "ulysses"),
        )
        for case in options["scenarios"]:
            family, mode, checkpointing = case["family"], case["mode"], case["checkpointing"]
            adapter, model = _case(family, mode)
            model = model.to(device=device, dtype=dtype)
            reference = copy.deepcopy(model)
            if checkpointing:
                model.enable_gradient_checkpointing()
                reference.enable_gradient_checkpointing()
            native = peel_peft_wrapper(model)
            if family.startswith("qwen"):
                native.set_attention_backend("native")
                peel_peft_wrapper(reference).set_attention_backend("native")
            apply_tensor_parallel(
                native, mesh["tp"], resolve_parallel_plan(adapter, sequence_parallel=True)
            )
            if cp_size > 1:
                from diffusers.models._modeling_parallel import ContextParallelConfig

                adapter.validate_context_parallel_module(native, cp_size, 2)
                adapter.configure_context_parallel_module(native)
                native.set_attention_backend("native")
                adapter.install_context_parallel_module(
                    native, ContextParallelConfig(ulysses_degree=cp_size, mesh=mesh)
                )
                adapter.compose_context_parallel_sequence_boundaries(native)
            observed = set()
            for index, block in enumerate(native.transformer_blocks):
                text_norm = block.norm1_context if family.startswith("sd35") else block.txt_norm1
                text_norm.register_forward_pre_hook(
                    lambda _module, args, index=index: observed.add((index, args[0].shape[1]))
                )
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, eps=1e-4)
            reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.001, eps=1e-4)
            tolerance = dict(rtol=3e-4, atol=3e-6) if cuda else dict(rtol=2e-7, atol=1e-8)
            for step in range(3):
                result["active_case"] = {**case, "step": step}
                inputs = _inputs(family, step, device, dtype)
                differentiable = (
                    ("hidden_states", "encoder_hidden_states", "pooled_projections")
                    if family.startswith("sd35")
                    else ("hidden_states", "encoder_hidden_states")
                )
                for name in differentiable:
                    inputs[name].requires_grad_(True)
                expected_inputs = {
                    name: (
                        value.detach().clone().requires_grad_(value.requires_grad)
                        if torch.is_tensor(value)
                        else copy.deepcopy(value)
                    )
                    for name, value in inputs.items()
                }
                actual = model(**inputs)[0]
                expected = reference(**expected_inputs)[0]
                rounding = _assert_predictions(
                    actual,
                    expected,
                    reference,
                    expected_inputs,
                    tolerance,
                    cuda and family.startswith("qwen"),
                )
                if rounding is not None:
                    result.setdefault("forward_rounding", []).append(
                        {**case, "step": step, **rounding}
                    )
                actual.square().mean().backward()
                expected.square().mean().backward()
                for name in differentiable:
                    if cp_size > 1:
                        dist.all_reduce(inputs[name].grad, group=mesh["ulysses"].get_group())
                    torch.testing.assert_close(
                        inputs[name].grad,
                        expected_inputs[name].grad,
                        msg=f"input:{name}",
                        **tolerance,
                    )
                parameters, expected_parameters = dict(model.named_parameters()), dict(
                    reference.named_parameters()
                )
                assert parameters.keys() == expected_parameters.keys()
                active = 0
                nonzero = 0
                for name, parameter in parameters.items():
                    target = expected_parameters[name]
                    assert (parameter.grad is None) == (target.grad is None), name
                    if target.grad is not None:
                        if cp_size > 1:
                            dist.all_reduce(
                                (
                                    parameter.grad.to_local()
                                    if isinstance(parameter.grad, DTensor)
                                    else parameter.grad
                                ),
                                group=mesh["ulysses"].get_group(),
                            )
                        full_gradient = _full(parameter.grad)
                        torch.testing.assert_close(
                            full_gradient, target.grad, msg=name, **tolerance
                        )
                        assert (
                            torch.isfinite(full_gradient).all()
                            and torch.isfinite(target.grad).all()
                        )
                        active += 1
                        nonzero += int(
                            bool(torch.count_nonzero(full_gradient))
                            and bool(torch.count_nonzero(target.grad))
                        )
                assert nonzero > 0
                result.setdefault("nonzero_parameter_gradients", []).append(
                    {**case, "step": step, "count": nonzero, "active_parameters": active}
                )
                optimizer.step()
                reference_optimizer.step()
                for name, parameter in parameters.items():
                    target = expected_parameters[name]
                    torch.testing.assert_close(_full(parameter), target, msg=name, **tolerance)
                    assert (
                        optimizer.state.get(parameter, {}).keys()
                        == reference_optimizer.state.get(target, {}).keys()
                    )
                    for key, value in optimizer.state.get(parameter, {}).items():
                        torch.testing.assert_close(
                            _full(value),
                            reference_optimizer.state[target][key],
                            msg=f"{name}:{key}",
                            **tolerance,
                        )
                optimizer.zero_grad(set_to_none=True)
                reference_optimizer.zero_grad(set_to_none=True)
            text_length = 5 if family.startswith("sd35") else 7
            local_text = (text_length + 2 * cp_size - 1) // (2 * cp_size)
            assert observed == {(index, local_text) for index in range(2)}, observed
            canonical = {name: _full(value) for name, value in model.state_dict().items()}
            assert canonical.keys() == reference.state_dict().keys()
            checkpoint = Path(result_dir, f"{family}-{mode}-{checkpointing}-{rank}.pt")
            torch.save(canonical, checkpoint)
            restored = copy.deepcopy(reference)
            restored.load_state_dict(torch.load(checkpoint, weights_only=True))
            _assert_predictions(
                model(**inputs)[0],
                restored(**inputs)[0],
                restored,
                inputs,
                tolerance,
                cuda and family.startswith("qwen"),
            )
            result["scenarios"].append(
                {
                    **case,
                    "updates": 3,
                    "active_parameters": active,
                    "text_shapes": sorted(observed),
                    "input_gradients": True,
                    "canonical_reload": True,
                }
            )
        result["status"] = "pass"
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result, indent=2))
        dist.destroy_process_group()


@pytest.mark.parametrize("family", ["sd35-medium", "sd35-large", "qwen", "qwen-edit"])
@pytest.mark.parametrize("cp_size", [1, 2])
def test_native_text_sp_complete_derivatives_updates_and_state(tmp_path, family, cp_size):
    scenarios = [
        {"family": family, "mode": mode, "checkpointing": checkpointing}
        for mode in ("full", "lora", "expanded_lora")
        for checkpointing in (False, True)
    ]
    rows = run_gloo_ranks(
        _worker, tmp_path, 2 * cp_size, {"cp_size": cp_size, "scenarios": scenarios}
    )
    assert all(row.get("status") == "pass" and "error" not in row for row in rows), rows


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="native text SP requires two CUDA ranks")
def test_cuda_native_text_sp_three_updates(tmp_path):
    scenarios = [
        {"family": family, "mode": mode, "checkpointing": True}
        for family in ("sd35-medium", "sd35-large", "qwen", "qwen-edit")
        for mode in ("full", "expanded_lora")
    ]
    rows = run_gloo_ranks(_worker, tmp_path, 2, {"cuda": True, "scenarios": scenarios})
    assert all(row.get("status") == "pass" and "error" not in row for row in rows), rows


@pytest.mark.parametrize("family", ["sd35-medium", "sd35-large", "qwen", "qwen-edit"])
def test_text_padding_mask_is_required_for_native_image_predictions(family):
    """An unmasked padding control must disagree even when native tensor shapes are legal."""
    import torch.nn.functional as F

    adapter, model = _case(family, "full")
    model = model.double()
    inputs = _inputs(family, 0, "cpu", torch.float64)
    with torch.no_grad():
        expected = model(**inputs)[0]
        padded = copy.deepcopy(inputs)
        length = inputs["encoder_hidden_states"].shape[1]
        padding = (-length) % 4
        padded["encoder_hidden_states"] = F.pad(inputs["encoder_hidden_states"], (0, 0, 0, padding))
        if family.startswith("sd35"):
            adapter.configure_context_parallel_module(model)
            padded["joint_attention_kwargs"] = {"cp_text_tokens": length}
        else:
            model.set_attention_backend("native")
            padded["encoder_hidden_states_mask"] = F.pad(
                inputs["encoder_hidden_states_mask"], (0, padding), value=0
            )
        torch.testing.assert_close(model(**padded)[0], expected, rtol=2e-7, atol=1e-8)
        if family.startswith("sd35"):
            padded.pop("joint_attention_kwargs")
        else:
            padded["encoder_hidden_states_mask"][:, length:] = True
        assert (model(**padded)[0] - expected).abs().max().item() > 1e-5


def test_sd35_sp_rejects_foreign_processor_before_mutating_any_attention():
    from types import SimpleNamespace

    from flow_factory.models.stable_diffusion._sequence_parallel import SD35SequenceParallelInput

    _, model = _case("sd35-medium", "full")
    first = model.transformer_blocks[0].attn.processor
    model.transformer_blocks[-1].attn.processor = object()
    with pytest.raises(ValueError, match="native joint attention processor"):
        SD35SequenceParallelInput()._apply(model, SimpleNamespace(size=lambda: 2))
    assert model.transformer_blocks[0].attn.processor is first
    assert not hasattr(model, "_ff_sd35_sp_degree")


def test_qwen_sp_rejects_mask_shape_before_padding():
    from types import SimpleNamespace

    from flow_factory.models.qwen_image._sequence_parallel import QwenSequenceParallelInput

    _, model = _case("qwen", "full")
    QwenSequenceParallelInput()._apply(model, SimpleNamespace(size=lambda: 2))
    inputs = _inputs("qwen", 0, "cpu", torch.float64)
    inputs["encoder_hidden_states_mask"] = inputs["encoder_hidden_states_mask"][:, :6]
    with pytest.raises(ValueError, match="mask must match original"):
        model(**inputs)
