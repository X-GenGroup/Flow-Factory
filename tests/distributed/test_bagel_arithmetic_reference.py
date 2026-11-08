# Copyright 2026 Jayce-Ping
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""CPU proofs for Bagel independent arithmetic reference and own optimizer history."""

import contextvars
import copy
import json
import traceback
from pathlib import Path
from types import MethodType, SimpleNamespace

import bagel_arithmetic_reference as _norm_reference
import pytest
import torch
import torch.distributed as dist
from bagel_arithmetic_operators import attention, column, mlp, padded_geometry, row
from bagel_own_adam_reference import own_adam, same_state_gradients, synchronize_shadow
from native_adam_window_oracle import _capture_update, _records

from flow_factory.models.bagel.modeling.qwen2.modeling_qwen2 import Qwen2RMSNorm


def gradients(output, inputs, parameters, target):
    return torch.autograd.grad((output * target).sum(), [*inputs, *parameters], allow_unused=False)


def assert_values(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    return float((actual - expected).detach().abs().max()) if actual.numel() else 0.0


def linear_controls(lora):
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(481)
    module = torch.nn.Module()
    for name, source, target in (
        ("gate_proj", 3584, 1024),
        ("up_proj", 3584, 1024),
        ("down_proj", 1024, 3584),
    ):
        projection = torch.nn.Linear(source, target, bias=True, dtype=torch.float64)
        setattr(module, name, projection)
    module.act_fn = torch.nn.functional.silu
    if lora:
        wrapper = get_peft_model(
            module,
            LoraConfig(
                r=2,
                lora_alpha=2,
                lora_dropout=0.0,
                init_lora_weights=False,
                target_modules=["gate_proj", "up_proj", "down_proj"],
            ),
        )
        module = wrapper.base_model.model.double()
    reference = copy.deepcopy(module)
    value = torch.randn(6, 3584, dtype=torch.float64, requires_grad=True)
    refvalue = value.detach().clone().requires_grad_()
    target = torch.randn(6, 3584, dtype=torch.float64)
    actual = mlp(module, value)
    expected = reference.down_proj(
        reference.act_fn(reference.gate_proj(refvalue)) * reference.up_proj(refvalue)
    )
    output_error = assert_values(actual, expected)
    params = [(name, p) for name, p in module.named_parameters() if p.requires_grad]
    refparams = [(name, p) for name, p in reference.named_parameters() if p.requires_grad]
    assert [n for n, _ in params] == [n for n, _ in refparams]
    left = gradients(actual, [value], [p for _, p in params], target)
    right = gradients(expected, [refvalue], [p for _, p in refparams], target)
    errors = [assert_values(a, b) for a, b in zip(left, right)]
    assert all(torch.count_nonzero(g) for g in right)
    if lora:
        assert any("lora_A" in n for n, _ in params) and any("lora_B" in n for n, _ in params)
        assert all(p.grad is None for p in module.parameters() if not p.requires_grad)
    projection = module.gate_proj
    parts = [column(projection, value.detach(), rank) for rank in (0, 1)]
    assert_values(torch.cat(parts, -1), projection(value.detach()))
    assert not torch.allclose(
        torch.cat(parts[::-1], -1), projection(value.detach()), rtol=1e-8, atol=1e-8
    )
    hidden = torch.randn(6, 1024, dtype=torch.float64)
    full = row(module.down_proj, hidden.chunk(2, -1))
    assert_values(full, module.down_proj(hidden))
    base = getattr(module.down_proj, "base_layer", module.down_proj)
    assert not torch.allclose(full + base.bias, module.down_proj(hidden), rtol=1e-8, atol=1e-8)
    assert not torch.allclose(left[0] / 2, right[0], rtol=1e-8, atol=1e-8)
    return {
        "mode": "lora" if lora else "full",
        "width": 3584,
        "intermediate": 1024,
        "output_max_abs": output_error,
        "vjp_max_abs": max(errors),
        "active_parameters": [n for n, _ in params],
        "bias_count_wrong_half_and_gradient_factor_negatives": True,
    }


def attention_controls(sp):
    torch.manual_seed(812)
    lengths, prefix_lengths = [0, 7, 9], [3, 5, 9]
    heads, kv = (14, 2) if sp else (28, 4)
    values = [
        torch.randn(sum(lengths), h, 128, dtype=torch.float64, requires_grad=True)
        for h in (heads, kv, kv)
    ]
    prefixes = [
        torch.randn(sum(prefix_lengths), kv, 128, dtype=torch.float64, requires_grad=True)
        for _ in range(2)
    ]
    reference = [x.detach().clone().requires_grad_() for x in [*values, *prefixes]]
    padded, original_ids, reverse_ids, membership = padded_geometry(lengths, 2, 2 if sp else 1)
    padded_values = []
    for value in values:
        # Build canonical padded storage without using the production layout.
        items, offset = [], 0
        for n, size in zip(lengths, padded):
            items += [value[offset : offset + n], value.new_zeros(size - n, *value.shape[1:])]
            offset += n
        padded_values.append(torch.cat(items))
    parts = [[value[indices] for indices in membership] for value in padded_values]
    prefix_parts = [list(value.split(prefix_lengths)) for value in prefixes]
    actual_parts = attention(*parts, *prefix_parts, lengths, padded, sp=sp, dense_fp64=True)
    canonical_order = torch.tensor(membership[0] + membership[1]).argsort()
    actual = torch.cat(actual_parts)[canonical_order][original_ids]
    independent = []
    for q, k, v, pk, pv in zip(
        *(x.split(lengths) for x in reference[:3]),
        *(x.split(prefix_lengths) for x in reference[3:]),
    ):
        key = torch.cat([pk, k]).repeat_interleave(7, 1).transpose(0, 1)
        val = torch.cat([pv, v]).repeat_interleave(7, 1).transpose(0, 1)
        scores = q.transpose(0, 1) @ key.transpose(-2, -1) / (128**0.5)
        independent.append((scores.softmax(-1) @ val).transpose(0, 1))
    expected = torch.cat(independent)
    error = assert_values(actual, expected)
    target = torch.randn_like(expected)
    left = gradients(actual, [*values, *prefixes], [], target)
    right = gradients(expected, reference, [], target)
    errors = [assert_values(a, b) for a, b in zip(left, right)]
    assert all(torch.count_nonzero(g) for g in right)
    wrong = attention(
        *parts, *prefix_parts, lengths, padded, sp=sp, dense_fp64=True, mask_padding=False
    )
    wrong = torch.cat(wrong)[canonical_order][original_ids]
    assert not torch.allclose(wrong, expected, rtol=1e-8, atol=1e-8)
    assert not torch.allclose(actual.flip(1), expected, rtol=1e-8, atol=1e-8)
    assert not torch.allclose(left[0] / 2, right[0], rtol=1e-8, atol=1e-8)
    native = attention(*parts, *prefix_parts, lengths, padded, sp=sp, dense_fp64=False)
    assert_values(torch.cat(native)[canonical_order][original_ids], expected)
    text = {0, 6, 7, 15}
    mapped_text = {original_ids[index] for index in text}
    mapped_generation = set(range(sum(padded))) - mapped_text
    assert mapped_text.isdisjoint(mapped_generation)
    assert mapped_text | mapped_generation == set(range(sum(padded)))
    assert all(index in mapped_generation for index, old in enumerate(reverse_ids) if old < 0)
    return {
        "sp": sp,
        "query_heads": heads,
        "kv_heads": kv,
        "head_width": 128,
        "lengths": lengths,
        "prefix_lengths": prefix_lengths,
        "padded_lengths": padded,
        "output_max_abs": error,
        "vjp_max_abs": max(errors),
        "wrong_mask_head_gradient_factor_negatives": True,
        "padding_roles_partition": True,
    }


def rejects(fn):
    try:
        fn()
    except AssertionError:
        return True
    raise AssertionError("Corruption was accepted")


def test_bagel_actual_adam_and_shadow_isolation_reject_corruption():
    torch.set_num_threads(1)
    p = torch.nn.Parameter(torch.linspace(-0.1, 0.1, 262145))
    shadow = torch.nn.Parameter(torch.zeros_like(p))
    opt = torch.optim.AdamW([p], lr=1e-3, eps=1e-4, weight_decay=1e-4)
    records = _records({"p": p})
    sync = []
    for step in range(3):
        snapshot = p.detach().clone()
        states = copy.deepcopy(opt.state_dict())
        sync.append(synchronize_shadow({"p": p}, {"p": shadow}, opt))
        torch.testing.assert_close(p, snapshot, rtol=0, atol=0)
        torch.testing.assert_close(shadow, snapshot, rtol=0, atol=0)
        for key, value in states["state"].get(0, {}).items():
            torch.testing.assert_close(opt.state_dict()["state"][0][key], value, rtol=0, atol=0)
        p.grad = torch.linspace(-0.2, 0.3, 262145) * (1 - 0.3 * step)
        _capture_update({"p": p}, opt, records, torch.tensor(1.0), clipping_world_size=1)
    own = own_adam(records, world_size=1)
    assert own["p"]["windows"] == 2
    rejections = {}
    for field in ("exp_avg", "exp_avg_sq", "step"):
        bad = copy.deepcopy(records)
        value = bad["updates"][1]["moments"]["p"][field]
        value.reshape(-1)[-1] += 0.01 if field != "step" else 1
        rejections[field] = rejects(lambda: own_adam(bad, world_size=1))
    bad = copy.deepcopy(records)
    bad["updates"][1]["parameters"]["p"][-1] += 0.01
    rejections["parameter"] = rejects(lambda: own_adam(bad, world_size=1))
    bad = copy.deepcopy(records)
    bad["updates"].pop()
    rejections["missing_update"] = rejects(lambda: own_adam(bad, world_size=1))
    bad = copy.deepcopy(records)
    bad["updates"][1]["gradients"]["p"][-1] += 1
    rejections["wrong_gradient"] = rejects(lambda: own_adam(bad, world_size=1))
    rejections["aliased_shadow"] = rejects(lambda: synchronize_shadow({"p": p}, {"p": p}, opt))
    gradients = {"p": torch.ones(262145)}
    assert (
        same_state_gradients(gradients, copy.deepcopy(gradients))["parameters"]["p"]["elements"]
        == 262145
    )
    rejections["cp_factor"] = rejects(
        lambda: same_state_gradients({"p": gradients["p"] / 2}, gradients)
    )
    wrong = copy.deepcopy(gradients)
    wrong["p"][-1] += 1
    rejections["last_window"] = rejects(lambda: same_state_gradients(wrong, gradients))
    result = {
        "status": "pass",
        "scope": "CPU diagnostic controls",
        "own_adam": own,
        "synchronizations": sync,
        "negative_rejections": rejections,
    }
    assert result["status"] == "pass" and all(rejections.values())


def _arithmetic_gloo_worker(rank, store, directory, options):
    result = {"rank": rank, **options, "scope": "CPU Gloo prototype, not native CUDA qualification"}
    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=4)
    result["reference_contract"] = "bagel-canonical-arithmetic-global-signal-v1"
    result["runtime"] = {
        "world_size": dist.get_world_size(),
        "backend": dist.get_backend(),
        "device": "cpu",
    }
    try:
        from bagel_arithmetic_reference import ArithmeticReference
        from bagel_cp_fixture import (
            build_pipeline,
            build_tokenizer,
            forward_calls,
            install_lora,
            make_input_adapter,
            model_call_inputs,
        )
        from diffusers.models._modeling_parallel import ContextParallelConfig
        from test_bagel_parallel import _cpu_flash_reference
        from test_context_parallel_training import _bf16_tensors
        from test_model_preparation import parameter_map
        from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
        from torch.distributed.tensor import DTensor

        from flow_factory.models.bagel.bagel import BagelAdapter
        from flow_factory.models.bagel.modeling.bagel import qwen2_navit, siglip_navit
        from flow_factory.models.tensor_parallel import apply_tensor_parallel

        qwen2_navit.flash_attn_varlen_func = _cpu_flash_reference
        siglip_navit.flash_attn_varlen_func = _cpu_flash_reference
        pipeline = build_pipeline()
        if options["mode"] == "lora":
            pipeline.bagel.language_model = install_lora(pipeline.bagel.language_model)
            pipeline.transformer = pipeline.bagel.language_model
        shadow_pipeline = copy.deepcopy(pipeline)
        actual = pipeline.transformer
        shadow = shadow_pipeline.transformer
        raw = actual.base_model.model if options["mode"] == "lora" else actual
        raw_shadow = shadow.base_model.model if options["mode"] == "lora" else shadow
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("cp", "tp"))
        plan = BagelAdapter.tensor_parallel_plan()
        if options["sp"]:
            plan = {**plan, **BagelAdapter.sequence_parallel_plan()}
        apply_tensor_parallel(raw, mesh["tp"], plan)
        BagelAdapter.validate_context_parallel_module(raw, 2, 2)
        BagelAdapter.configure_context_parallel_module(raw)
        raw.set_attention_backend("native")
        cp_mesh = DeviceMesh(
            "cpu", mesh.mesh.T.unsqueeze(1), mesh_dim_names=("tensor", "ring", "ulysses")
        )
        BagelAdapter.install_context_parallel_module(
            raw, ContextParallelConfig(ring_degree=1, ulysses_degree=2, mesh=cp_mesh)
        )
        independent = ArithmeticReference(raw_shadow, sequence_parallel=options["sp"]).install()
        if options["mode"] == "lora":
            actual.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        counts = [0]

        def observe(*args):
            counts[0] += 1

        for block in raw.model.layers:
            block.register_forward_pre_hook(observe)
        tokenizer = build_tokenizer(Path(directory) / f"tokenizer-{rank}")
        actual_adapter = make_input_adapter(pipeline, tokenizer, "cpu")
        shadow_adapter = make_input_adapter(shadow_pipeline, tokenizer, "cpu")
        actual_inputs, actual_metadata = model_call_inputs(actual_adapter, 0, 0, ragged=False)
        shadow_inputs, shadow_metadata = model_call_inputs(shadow_adapter, 0, 0, ragged=False)
        assert actual_metadata == shadow_metadata
        cache_residuals = []
        for left, right in zip(actual_inputs, shadow_inputs):
            for kind in ("key_cache", "value_cache"):
                for layer, expected in getattr(right["past_key_values"], kind).items():
                    received = getattr(left["past_key_values"], kind)[layer]
                    if options["sp"]:
                        expected = expected.chunk(2, 1)[rank % 2]
                    cache_residuals.append(
                        _bf16_tensors({"cache": received}, {"cache": expected}, atol=1e-3)
                    )
        before_generation = counts[0]
        actual_output = forward_calls(actual, actual_inputs)
        shadow_output = forward_calls(shadow, shadow_inputs)
        result["output_relative_l2"] = _bf16_tensors(
            {"output": actual_output.detach()}, {"output": shadow_output.detach()}, atol=1e-3
        )
        target = torch.randn(actual_output.shape, generator=torch.Generator().manual_seed(337))
        (actual_output.float() - target).square().mean().backward()
        (shadow_output.float() - target).square().mean().backward()
        actual_parameters, expected_parameters = parameter_map(actual), parameter_map(shadow)
        assert actual_parameters.keys() == expected_parameters.keys()
        gradients, reference_gradients, inactive, frozen = {}, {}, [], []
        factor_rejections = 0
        for name, parameter in actual_parameters.items():
            expected = expected_parameters[name]
            assert parameter.requires_grad == expected.requires_grad
            assert (parameter.grad is None) == (expected.grad is None), name
            if parameter.grad is None:
                (inactive if parameter.requires_grad else frozen).append(name)
                continue
            gradient = (
                parameter.grad.full_tensor()
                if isinstance(parameter.grad, DTensor)
                else parameter.grad.detach().clone()
            )
            dist.all_reduce(gradient, group=mesh["cp"].get_group())
            gradients[name] = gradient
            reference_gradients[name] = expected.grad.detach()
            try:
                _bf16_tensors({name: gradient / 2}, {name: expected.grad}, atol=1e-3)
            except AssertionError:
                factor_rejections += 1
        assert gradients and factor_rejections > 0
        gradient_checks = same_state_gradients(gradients, reference_gradients)
        try:
            same_state_gradients(
                {name: gradient / 2 for name, gradient in gradients.items()}, reference_gradients
            )
        except AssertionError:
            global_factor_rejected = True
        else:
            global_factor_rejected = False
        assert global_factor_rejected
        assert all(name.endswith("lm_head.weight") for name in inactive)
        if options["mode"] == "lora":
            assert frozen and counts[0] - before_generation > 6
        assert independent.context.get() is None
        result.update(
            status="pass",
            global_gradient_relative_l2=gradient_checks["global_relative_l2"],
            gradient_relative_l2_scope=gradient_checks["relative_l2_scope"],
            gradient_region_statistics="diagnostic_only",
            gradients=gradient_checks["parameters"],
            global_cp_factor_rejected=global_factor_rejected,
            cache_max_relative_l2=max(cache_residuals),
            inactive=inactive,
            frozen_count=len(frozen),
            cp_factor_rejections=factor_rejections,
            generation_block_calls=counts[0] - before_generation,
            metadata=actual_metadata,
        )
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        Path(directory, f"rank-{rank}.json").write_text(json.dumps(result, indent=2))
        dist.destroy_process_group()


@pytest.mark.parametrize("lora", [False, True])
def test_bagel_arithmetic_linear_fp64_vjp(lora):
    linear_controls(lora)


@pytest.mark.parametrize("sp", [False, True])
def test_bagel_arithmetic_attention_fp64_vjp(sp):
    attention_controls(sp)


@pytest.mark.parametrize("mode", ["full", "lora"])
@pytest.mark.parametrize("sp", [False, True])
def test_bagel_arithmetic_reference_actual_gloo(tmp_path, mode, sp):
    from conftest import run_gloo_ranks

    rows = run_gloo_ranks(
        _arithmetic_gloo_worker, tmp_path, 4, {"mode": mode, "sp": sp}, timeout=1800
    )
    assert all(row.get("status") == "pass" and "error" not in row for row in rows), rows


def _norm_exact(left, right):
    torch.testing.assert_close(left, right, atol=0, rtol=0)


def _norm_must_reject(function):
    try:
        function()
    except AssertionError:
        return
    raise AssertionError("negative control was accepted")


def _norm_warm_norm():
    torch.manual_seed(705)
    norm = Qwen2RMSNorm(3584)
    optimizer = torch.optim.AdamW(norm.parameters(), lr=0.001, eps=0.0001)
    for step in range(3):
        value = torch.randn(6, 3584)
        target = torch.randn_like(value) + 0.2 * (step + 1)
        optimizer.zero_grad(set_to_none=True)
        (norm(value) - target).square().mean().backward()
        optimizer.step()
    assert torch.all(norm.weight != 1)
    assert int(optimizer.state[norm.weight]["step"]) == 3
    norm.zero_grad(set_to_none=True)
    return norm


def _norm_geometry(sp, role_mode):
    lengths = [0, 7, 9]
    padded, original, _, cp_rows = padded_geometry(lengths, 2, 2 if sp else 1)
    if role_mode == "mixed":
        text = [original[index] for index in (0, 6, 7, 15)]
    elif role_mode == "empty_text":
        text = []
    else:
        assert role_mode == "empty_generation"
        text = list(range(sum(padded)))
    gen = [row for row in range(sum(padded)) if row not in text]
    return _norm_reference.Call(
        True,
        lengths,
        padded,
        original,
        cp_rows,
        torch.tensor(text, dtype=torch.long),
        torch.tensor(gen, dtype=torch.long),
    )


def _norm_independent_groups(call, sp, residual_sp, role):
    owner = {}
    degree = 2 if sp and residual_sp else 1
    for cp, rows in enumerate(call.cp_rows):
        for offset, row in enumerate(rows):
            owner[row] = cp * degree + offset // (len(rows) // degree)
    return [
        [index for index, row in enumerate(role.tolist()) if owner[row] == group]
        for group in range(2 * degree)
    ]


def _norm_run_case(template, sp, residual_sp, suffix, dtype, role_mode):
    call = _norm_geometry(sp, role_mode)
    role = call.text if suffix == "" else call.gen
    values = torch.randn(role.numel(), 3584, generator=torch.Generator().manual_seed(91))
    if role.numel():
        values[:, 0] = role.float() + 0.25
    values = values.to(dtype).requires_grad_()
    expected_values = values.detach().clone().requires_grad_()
    norm, oracle = (copy.deepcopy(template), copy.deepcopy(template))
    original_identity = id(norm.weight)
    before = norm.weight.detach().clone()
    native = norm.forward
    seen = []

    def record(owner, hidden):
        seen.append(hidden.detach().clone())
        assert id(owner.weight) == original_identity
        return native(hidden)

    norm.forward = MethodType(record, norm)
    wrapper = object.__new__(_norm_reference.ArithmeticReference)
    wrapper.sp = sp
    wrapper.context = contextvars.ContextVar("test_norm_call", default=None)
    wrapper._install_role_norm(norm, suffix, residual_sp=residual_sp)
    token = wrapper.context.set(call)
    try:
        output = norm(values)
    finally:
        wrapper.context.reset(token)
    assert wrapper.context.get() is None
    groups = _norm_independent_groups(call, sp, residual_sp, role)
    assert len(seen) == len(groups)
    expected = torch.zeros_like(output)
    for observed, positions in zip(seen, groups):
        ids = torch.tensor(positions, dtype=torch.long)
        _norm_exact(observed, expected_values[ids])
        expected = expected.index_copy(0, ids, oracle(expected_values[ids]))
    _norm_exact(output, expected)
    assert output.dtype == expected.dtype == torch.float32
    target = torch.randn(output.shape, generator=torch.Generator().manual_seed(377))
    left = torch.autograd.grad((output * target).sum(), (values, norm.weight))
    right = torch.autograd.grad((expected * target).sum(), (expected_values, oracle.weight))
    for received, wanted in zip(left, right):
        _norm_exact(received, wanted)
    _norm_exact(norm.weight, before)
    assert id(norm.weight) == original_identity
    if role.numel():
        assert torch.count_nonzero(left[0]) and torch.count_nonzero(left[1])
        for factor in (0, 0.5, 2):
            _norm_must_reject(lambda factor=factor: _norm_exact(left[1] * factor, right[1]))
        if role.numel() > 1:
            _norm_must_reject(lambda: _norm_exact(output.flip(0), expected))
    else:
        assert not torch.count_nonzero(left[1])
    seen.clear()
    prefix = copy.copy(call)
    prefix.generation = False
    token = wrapper.context.set(prefix)
    try:
        prefix_output = norm(values)
    finally:
        wrapper.context.reset(token)
    assert len(seen) == 1
    _norm_exact(seen[0], values)
    _norm_exact(prefix_output, oracle(values))
    prefix_left = torch.autograd.grad((prefix_output * target).sum(), (values, norm.weight))
    prefix_right = torch.autograd.grad(
        (oracle(expected_values) * target).sum(), (expected_values, oracle.weight)
    )
    for received, wanted in zip(prefix_left, prefix_right):
        _norm_exact(received, wanted)
    return {
        "sp": sp,
        "scope": "block" if residual_sp else "final_root",
        "role": "text" if suffix == "" else "generation",
        "dtype": str(dtype),
        "role_mode": role_mode,
        "native_call_rows": [[int(role[index]) for index in group] for group in groups],
        "empty_calls": sum((not group for group in groups)),
        "native_output_and_vjp_exact": True,
        "prefix_single_native_call_output_and_vjp_exact": True,
        "original_parameter_identity_and_values": True,
    }


def _norm_wiring_control(template):
    calls = []
    wrapper = object.__new__(_norm_reference.ArithmeticReference)
    wrapper.sp = True
    wrapper.context = contextvars.ContextVar("wiring", default=None)
    wrapper.observations = {"prefix_calls": 0, "generation_calls": 0}
    root = SimpleNamespace(layers=[])
    root.forward_inference = lambda **kwargs: None
    for suffix in ("", "_moe_gen"):
        setattr(root, "norm" + suffix, copy.deepcopy(template))
    for _ in range(2):
        block = SimpleNamespace(self_attn=SimpleNamespace(forward_inference=lambda **kwargs: None))
        for suffix in ("", "_moe_gen"):
            setattr(
                block, "mlp" + suffix, SimpleNamespace(forward=lambda hidden_state: hidden_state)
            )
            for kind in ("input_layernorm", "post_attention_layernorm"):
                setattr(block, kind + suffix, copy.deepcopy(template))
        root.layers.append(block)
    wrapper.model = SimpleNamespace(model=root)
    original_installer = wrapper._install_role_norm

    def observe(module, suffix, *, residual_sp):
        calls.append((id(module), suffix, residual_sp))
        original_installer(module, suffix, residual_sp=residual_sp)

    wrapper._install_role_norm = observe
    wrapper.install()
    assert len(calls) == 10 and len({item[0] for item in calls}) == 10
    assert all(item[2] for item in calls)
    assert {id(root.norm), id(root.norm_moe_gen)} <= {item[0] for item in calls}
    return {
        "norms_installed": 10,
        "block_norms_split_sp": 8,
        "final_root_norms_split_sp": 2,
        "qk_norms_untouched": True,
    }


def _norm_grouping_negative_controls():
    call = _norm_geometry(True, "mixed")
    block = _norm_independent_groups(call, True, True, call.text)
    final = _norm_independent_groups(call, True, False, call.text)
    assert [len(group) for group in block] == [2, 0, 1, 1]
    assert [len(group) for group in final] == [2, 2]
    _norm_must_reject(
        lambda: _norm_exact(
            torch.tensor([len(x) for x in block]), torch.tensor([len(x) for x in final])
        )
    )
    _norm_must_reject(
        lambda: _norm_exact(torch.tensor([len(x) for x in block]), torch.tensor([1, 1, 1, 1]))
    )
    return {
        "wrong_final_tp_split_rejected": True,
        "wrong_block_cp_only_rejected": True,
        "role_before_residual_split_rejected": True,
    }


@pytest.fixture(scope="module")
def _norm_warm_fixture():
    return _norm_warm_norm()


@pytest.mark.parametrize("sp", [False, True])
@pytest.mark.parametrize("residual_sp", [False, True], ids=["final-root", "block"])
@pytest.mark.parametrize("suffix", ["", "_moe_gen"], ids=["text", "generation"])
@pytest.mark.parametrize(
    "dtype,role_mode",
    [
        (torch.float32, "mixed"),
        (torch.bfloat16, "mixed"),
        (torch.float32, "empty_text"),
        (torch.float32, "empty_generation"),
    ],
    ids=["fp32", "bf16-input", "empty-text", "empty-generation"],
)
def test_bagel_arithmetic_residual_norm_native_geometry_and_vjp(
    _norm_warm_fixture, sp, residual_sp, suffix, dtype, role_mode
):
    _norm_run_case(_norm_warm_fixture, sp, residual_sp, suffix, dtype, role_mode)


def test_bagel_arithmetic_residual_norm_installation_and_grouping_reject_wrong_scope(
    _norm_warm_fixture,
):
    _norm_wiring_control(_norm_warm_fixture)
    _norm_grouping_negative_controls()


def test_bagel_global_gradient_gate_is_partition_and_window_invariant():
    # A low-energy region can exceed the global relative limit while every
    # element and the complete gradient signal satisfy the original contract.
    high = torch.ones(262144)
    low = torch.full((262144,), 1e-3)
    expected = torch.cat([high, low])
    actual = torch.cat([high, low * 1.04])
    from test_context_parallel_training import _bf16_tensors

    with pytest.raises(AssertionError):
        _bf16_tensors({"low": actual[262144:]}, {"low": low}, atol=1e-3)
    partitions = [
        ({"all": actual}, {"all": expected}),
        ({"a": actual[:262144], "z": actual[262144:]}, {"a": high, "z": low}),
        ({"z": actual[262144:], "a": actual[:262144]}, {"z": low, "a": high}),
        (
            {"a": actual[:12345], "b": actual[12345:300001], "c": actual[300001:]},
            {"a": expected[:12345], "b": expected[12345:300001], "c": expected[300001:]},
        ),
    ]
    metrics = []
    regional_exceeds = False
    for left, right in partitions:
        for window in (4096, 262144, 1048576):
            result = same_state_gradients(left, right, diagnostic_window_size=window)
            assert result["relative_l2_scope"] == "all_trainable_gradients"
            metrics.append(result["global_relative_l2"])
            regional_exceeds |= any(
                row["diagnostic_full_relative_l2"] > 2 / 128
                or row["diagnostic_max_window_relative_l2"] > 2 / 128
                for row in result["parameters"].values()
            )
    assert len(set(metrics)) == 1 and metrics[0] < 2 / 128 and regional_exceeds


@pytest.mark.parametrize("factor", [0.0, 0.5, 2.0])
def test_bagel_global_gradient_gate_rejects_wrong_factor(factor):
    expected = {"small": torch.full((31,), 1e-5), "other": torch.linspace(2e-5, 3e-5, 47)}
    actual = {name: value * factor for name, value in expected.items()}
    # Absolute floors alone would admit this; the unchanged global norm rejects it.
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0.02, atol=1e-3)
    with pytest.raises(AssertionError):
        same_state_gradients(actual, expected)


def test_bagel_global_gradient_gate_rejects_missing_gradient():
    expected = {"high": torch.ones(17), "small": torch.full((19,), 1e-5)}
    with pytest.raises(AssertionError):
        same_state_gradients({"high": expected["high"]}, expected)


def test_bagel_global_gradient_gate_rejects_pointwise_outlier_below_global_cap():
    expected = torch.ones(4096)
    actual = expected.clone()
    actual[-1] += 0.1
    assert float((actual - expected).norm() / expected.norm()) < 2 / 128
    with pytest.raises(AssertionError):
        same_state_gradients({"all": actual}, {"all": expected})


def test_bagel_global_gradient_gate_calls_shared_owner_once(monkeypatch):
    import bagel_own_adam_reference as owner

    original = owner._bf16_tensors
    expected = {"a": torch.full((29,), 0.1), "z": torch.full((31,), 1e-5)}
    actual = {name: value.clone() for name, value in expected.items()}
    calls = []

    def observe(left, right, *, atol):
        assert left is actual and right is expected and atol == 1e-3
        calls.append(tuple(sorted(left)))
        return original(left, right, atol=atol)

    monkeypatch.setattr(owner, "_bf16_tensors", observe)
    result = same_state_gradients(actual, expected, diagnostic_window_size=7)
    assert calls == [("a", "z")] and result["global_relative_l2"] == 0
