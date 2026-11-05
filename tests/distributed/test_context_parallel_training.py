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

"""Complete SFT CP equivalence at fixed logical DP and TP/SP geometry."""

import copy
import json
import os
import subprocess
import sys
import traceback
from contextlib import contextmanager, nullcontext
from functools import wraps
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from conftest import run_gloo_ranks
from test_model_preparation import full_value, initialize_cuda, runtime_observation


def _config(root, result_dir, options):
    from flow_factory.hparams import Arguments

    return Arguments.from_dict(
        {
            "mixed_precision": "bf16",
            "parallelism": {
                "cp_size": options["cp_size"],
                "dp_shard_size": 2,
                "tp_size": options.get("tp_size", 1),
                "sequence_parallel": options.get("sequence_parallel"),
            },
            "model": {
                "model_type": "wan2_t2v",
                "model_name_or_path": str(root / "model"),
                "target_components": ["transformer"],
                "target_modules": "all" if options["mode"] == "full" else "default",
                "finetune_type": options["mode"],
                "component_load_dtypes": "float32",
                "lora_rank": 2,
                "lora_alpha": 2,
            },
            "data": {
                "datasets": [
                    {"name": "tiny", "dataset_dir": str(root / "data"), "train": {"weight": 1}}
                ],
                "cache_dir": str(root / "cache"),
                "dataloader_num_workers": 0,
                "preprocessing_batch_size": 1,
                "enable_preprocess": True,
            },
            "train": {
                "trainer_type": "sft",
                "max_epochs": 2,
                "gradient_accumulation_steps": 2,
                "ema_decay": 0,
                "resolution": [32, 32],
                "num_frames": 5,
                "frame_rate": 4.0,
                "guidance_scale": 1.0,
                "num_train_timesteps": 2,
                "enable_gradient_checkpointing": options["checkpointing"],
                "seed": 17,
            },
            "log": {
                "save_model_only": True,
                "save_freq": 2,
                "save_dir": str(Path(result_dir) / "saves"),
                "run_name": "wan-sft",
                "logging_backend": "none",
                "verbose": False,
            },
            "eval": {"eval_freq": 0},
            "acceleration": {
                "shared": [{"name": "attention_backend", "params": {"backend": "native"}}]
            },
            "scheduler": {"dynamics_type": "ODE"},
            "optimizers": [
                {"name": "default", "learning_rate": 1e-3, "max_grad_norm": 1.0, "eps": 1e-4}
            ],
        }
    )


@contextmanager
def _capture_cross_attention(trainer):
    """Retain actual BF16 SDPA endpoints for a common-input head-split control."""
    endpoints, active, handles = {}, [None], []
    original = F.scaled_dot_product_attention

    def attention(*args, **kwargs):
        output = original(*args, **kwargs)
        name = active[0]
        if name is not None and output.requires_grad:
            rows = endpoints.setdefault(name, [])
            # Non-reentrant checkpoint recomputations may execute without their
            # temporary output receiving a gradient. Keep enough to select four
            # completed calls after backward, including the first GAS window.
            if len(rows) < 8:
                assert len(args) in (0, 3), "attention controls require named optional arguments"
                query, key, value = (
                    args[:3] if args else (kwargs["query"], kwargs["key"], kwargs["value"])
                )
                assert kwargs.get("attn_mask") is None and not kwargs.get("is_causal", False)
                assert kwargs.get("dropout_p", 0) == 0 and kwargs.get("scale") is None
                assert not kwargs.get("enable_gqa", False)
                assert all(tensor.dtype == torch.bfloat16 for tensor in (query, key, value, output))
                row = {
                    key: tensor.detach().clone()
                    for key, tensor in zip(("q", "k", "v", "output"), (query, key, value, output))
                }
                rows.append(row)
                for key, tensor in (("dq", query), ("dk", key), ("dv", value), ("dy", output)):
                    tensor.register_hook(
                        lambda gradient, key=key, row=row: row.update(
                            {key: gradient.detach().clone()}
                        )
                    )
        return output

    for name, module in trainer.model_bundle.named_modules():
        if name.endswith(".attn2"):
            handles.append(
                module.register_forward_pre_hook(
                    lambda module, args, name=name: active.__setitem__(0, name)
                )
            )
            handles.append(
                module.register_forward_hook(
                    lambda *args: active.__setitem__(0, None), always_call=True
                )
            )
    F.scaled_dot_product_attention = attention
    try:
        yield endpoints
    finally:
        F.scaled_dot_product_attention = original
        for handle in handles:
            handle.remove()


def _check_cross_attention_endpoints(accelerator, endpoints):
    """Compare production local derivatives to native attention on joined heads.

    BF16 attention can have ill-conditioned small q/k gradients. Joining exactly
    the same inputs and upstream gradient isolates TP head splitting from that
    native arithmetic; FP64 residuals are diagnostics, not a relaxed pass bound.
    """
    group = accelerator.torch_device_mesh["tp"].get_group()
    degree = dist.get_world_size(group)
    checked, diagnostics = 0, []
    assert len(endpoints) == 2
    for name, rows in endpoints.items():
        complete = [row for row in rows if {"dq", "dk", "dv", "dy"} <= row.keys()]
        assert len(complete) >= 4, (name, len(complete))
        for row in complete[:4]:
            joined = {}
            # Backward hooks may populate the dictionary in different orders on
            # peers. Every rank must gather the same named endpoint at each call.
            for key in ("q", "k", "v", "output", "dy", "dq", "dk", "dv"):
                value = row[key]
                shards = [
                    torch.empty_like(value, memory_format=torch.contiguous_format)
                    for _ in range(degree)
                ]
                dist.all_gather(shards, value.contiguous(), group=group)
                joined[key] = torch.cat(shards, dim=1)
            values = [joined[key].detach().requires_grad_() for key in ("q", "k", "v")]
            output = F.scaled_dot_product_attention(*values)
            gradients = torch.autograd.grad(output, values, joined["dy"])
            torch.testing.assert_close(joined["output"], output, rtol=0, atol=0)
            for key, gradient in zip(("dq", "dk", "dv"), gradients):
                torch.testing.assert_close(joined[key], gradient, rtol=0, atol=0)
                assert gradient.count_nonzero()
                # A wrong derivative in one small tensor must fail even when a
                # whole-model gradient norm could hide it.
                for factor in (0.0, 0.5, 2.0):
                    with pytest.raises(AssertionError):
                        torch.testing.assert_close(joined[key] * factor, gradient, rtol=0, atol=0)
            precise = [value.detach().double().requires_grad_() for value in values]
            precise_output = F.scaled_dot_product_attention(*precise)
            precise_gradients = torch.autograd.grad(precise_output, precise, joined["dy"].double())
            diagnostics.append(
                {
                    "module": name,
                    **{
                        key: float((gradient.double() - reference).norm() / reference.norm())
                        for key, gradient, reference in zip(
                            ("dq", "dk", "dv"), gradients, precise_gradients
                        )
                    },
                }
            )
            checked += 1
    return {
        "common_input_calls": checked,
        "negative_factors": [0.0, 0.5, 2.0],
        "bf16_vs_fp64": diagnostics,
    }


def _worker(rank, store_path, result_dir, options):
    # Configure the launcher before NCCL starts its native monitoring threads.
    # Mutating environ concurrently with native getenv is not thread-safe.
    os.environ.update(
        FSDP_VERSION="2",
        FSDP_AUTO_WRAP_POLICY="TRANSFORMER_BASED_WRAP",
        FSDP_TRANSFORMER_CLS_TO_WRAP="WanTransformerBlock",
        FSDP_CPU_RAM_EFFICIENT_LOADING="false",
        FSDP_RESHARD_AFTER_FORWARD="true",
        FSDP_ACTIVATION_CHECKPOINTING=str(options["checkpointing"]).lower(),
    )
    model_degree = options["cp_size"] * options.get("tp_size", 1)
    initialize_cuda(rank, store_path, world_size=2 * model_degree)
    result = {"rank": rank, **options}
    try:
        from flow_factory.trainers.loader import load_trainer
        from flow_factory.utils.dist import data_parallel_geometry

        trainer = load_trainer(_config(Path(options["fixture"]), result_dir, options))
        result["runtime"] = runtime_observation(trainer.accelerator)
        data_size, data_rank = data_parallel_geometry(trainer.accelerator)
        assert data_size == trainer.dataloader.sampler.num_replicas == 2
        assert data_rank == trainer.dataloader.sampler.rank == rank // model_degree
        assert getattr(trainer.accelerator, "_cp_context", None) is None
        assert (
            trainer.accelerator.state.fsdp_plugin.activation_checkpointing
            == options["checkpointing"]
        )
        records = {
            "initial_parameters": {
                name: full_value(p).cpu().clone()
                for name, p in trainer.model_bundle.named_parameters()
            },
            "batches": [],
            "forwards": [],
            "updates": [],
        }
        optimize = trainer.optimize_batch
        forward = trainer.adapter.forward
        step = trainer._apply_optimizer_step
        optimizer_step = trainer.optimizer.step
        clipped = []

        def observed_optimizer_step(*args, **kwargs):
            clipped.append(
                {
                    name: full_value(p.grad).cpu().clone()
                    for name, p in trainer.model_bundle.named_parameters()
                    if p.grad is not None
                }
            )
            return optimizer_step(*args, **kwargs)

        def observed_batch(batch):
            records["batches"].append(batch.record_ids)
            return optimize(batch)

        @wraps(forward)
        def observed_forward(*args, **kwargs):
            output = forward(*args, **kwargs)
            records["forwards"].append(
                {
                    "t": kwargs["t"].detach().cpu().clone(),
                    "latents": kwargs["latents"].detach().cpu().clone(),
                    "prompt_embeds": kwargs["prompt_embeds"].detach().cpu().clone(),
                    "velocity": output.velocity.detach().cpu().clone(),
                }
            )
            return output

        def observed_step(loss_info):
            optimizer = trainer.optimizer.optimizer
            assert len(optimizer.param_groups) == 1
            group = optimizer.param_groups[0]
            assert not group["amsgrad"] and not group["maximize"]
            settings = {key: group[key] for key in ("lr", "betas", "eps", "weight_decay")}
            settings["max_grad_norm"] = trainer.training_args.max_grad_norm
            parameters = dict(trainer.model_bundle.named_parameters())
            before = {name: full_value(p).cpu().clone() for name, p in parameters.items()}
            gradients = {
                name: full_value(p.grad).cpu().clone()
                for name, p in parameters.items()
                if p.grad is not None
            }
            expected_names = {name for name, p in parameters.items() if p.requires_grad}
            assert set(gradients) == expected_names
            assert gradients and all(torch.isfinite(g).all() for g in gradients.values())
            assert any(torch.count_nonzero(g) for g in gradients.values())
            losses = torch.stack(loss_info["loss"]).cpu().clone()
            returned = step(loss_info)
            after = {name: full_value(p).cpu().clone() for name, p in parameters.items()}
            for name, p in parameters.items():
                if not p.requires_grad:
                    torch.testing.assert_close(after[name], before[name], rtol=0, atol=0)
            moments = {
                name: {
                    key: full_value(value).cpu().clone()
                    for key, value in optimizer.state[p].items()
                }
                for name, p in parameters.items()
                if p.requires_grad
            }
            records["updates"].append(
                {
                    "gradients": gradients,
                    "parameters": after,
                    "moments": moments,
                    "loss": losses,
                    "deltas": {name: after[name] - before[name] for name in expected_names},
                    "clipped_gradients": clipped.pop(),
                    "optimizer": settings,
                }
            )
            return returned

        trainer.optimize_batch = observed_batch
        trainer.adapter.forward = observed_forward
        trainer._apply_optimizer_step = observed_step
        trainer.optimizer.step = observed_optimizer_step
        capture = (
            _capture_cross_attention(trainer)
            if options.get("tp_size", 1) > 1
            and options["cp_size"] == 1
            and options["mode"] == "lora"
            else nullcontext(None)
        )
        with capture as endpoints:
            trainer.start()
        if endpoints is not None:
            result["attention_endpoints"] = _check_cross_attention_endpoints(
                trainer.accelerator, endpoints
            )
        assert trainer.step == 4 and len(records["updates"]) == 4
        assert len(records["batches"]) == 8 and len(records["forwards"]) == 16
        exports = sorted(Path(result_dir, "saves").rglob("*.safetensors"))
        assert exports, "the full trainer must publish its configured final model export"
        from safetensors.torch import load_file

        exported = load_file(str(exports[-1]))
        expected_export = {
            name.removeprefix("members.transformer.")
            .replace("._checkpoint_wrapped_module", "")
            .replace(".default.", "."): value.to(
                torch.bfloat16 if options["mode"] == "full" else torch.float32
            )
            for name, value in records["updates"][-1]["parameters"].items()
            if options["mode"] == "full" or name in records["updates"][-1]["gradients"]
        }
        # The public save_checkpoint default stores full weights as BF16;
        # PEFT LoRA export preserves the prepared FP32 adapter parameters.
        _compare(exported, expected_export, exact=True)
        torch.save(records, Path(result_dir, f"observations-{rank}.pt"))
        result.update(
            status="pass",
            optimizer_steps=trainer.step,
            data_rank=data_rank,
            exports=len(exports),
            execution_capability=(
                "wan-sft-tp-ulysses-bf16-v1"
                if options.get("tp_size", 1) > 1
                else "wan-sft-ulysses-bf16-v1"
            ),
        )
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result, indent=2))
        dist.destroy_process_group()


def _compare(actual, expected, *, exact=False):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(
            actual, expected, rtol=0 if exact else 0.04, atol=0 if exact else 5e-4
        )
    elif isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key in expected:
            _compare(actual[key], expected[key], exact=exact)
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _compare(left, right, exact=exact)
    else:
        assert actual == expected


def _bf16_tensors(actual, expected, *, atol, relative_norm=2 / 128):
    """Bound every element and the full signal, including near-zero cancellation.

    BF16 reduce-scatter changes addition order when DP2 becomes DP2 x CP2.
    Near-zero derivatives need an absolute floor; a global relative bound keeps
    that floor from accepting missing gradients or a wrong reduction factor.
    The strict FP32 probe separately checks every parameter and its negative control.
    """
    assert set(actual) == set(expected) and expected
    names = sorted(expected)
    for name in names:
        torch.testing.assert_close(actual[name], expected[name], rtol=0.02, atol=atol, msg=name)
    left = torch.cat([actual[name].float().flatten() for name in names])
    right = torch.cat([expected[name].float().flatten() for name in names])
    residual = float((left - right).norm() / right.norm().clamp_min(1e-12))
    assert residual <= relative_norm, (residual, relative_norm)
    return residual


def _gamma(operations):
    """Standard relative error bound for a finite sequence of FP32 operations."""
    unit_roundoff = torch.finfo(torch.float32).eps / 2
    assert operations * unit_roundoff < 1
    return operations * unit_roundoff / (1 - operations * unit_roundoff)


def _assert_error_bound(actual, expected, error, label):
    actual = actual.double()
    assert actual.shape == expected.shape == error.shape, label
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all(), label
    assert torch.isfinite(error).all() and (error >= 0).all(), label
    assert torch.all((actual - expected).abs() <= error + torch.finfo(torch.float32).tiny), label


def _relative_l2(actual, expected, limit=2 / 128):
    assert set(actual) == set(expected) and expected
    names = sorted(expected)
    left = torch.cat([actual[name].double().flatten() for name in names])
    right = torch.cat([expected[name].double().flatten() for name in names])
    residual = float((left - right).norm() / right.norm().clamp_min(1e-12))
    assert residual <= limit, (residual, limit)
    return residual


def _check_clipping(update):
    """Bound parallel norm/reduction roundoff before verifying optimizer input."""
    assert all(value.dtype == torch.float32 for value in update["gradients"].values())
    assert all(value.dtype == torch.float32 for value in update["clipped_gradients"].values())
    gradients = {name: value.double() for name, value in update["gradients"].items()}
    observed = update["clipped_gradients"]
    assert set(observed) == set(gradients)
    norm = torch.stack([value.square().sum() for value in gradients.values()]).sum().sqrt()
    count = sum(value.numel() for value in gradients.values())
    # A serial sum is a conservative bound for any tree reduction of nonnegative
    # squares. Include squaring, sqrt, division, coefficient and product rounding.
    norm_error = _gamma(2 * count + 4)
    maximum = update["optimizer"]["max_grad_norm"]
    coefficient = (maximum / (norm + 1e-6)).clamp(max=1)
    low = (maximum / (norm * (1 + norm_error) + 1e-6)).clamp(max=1)
    high = (maximum / (norm * (1 - norm_error) + 1e-6)).clamp(max=1)
    errors = {}
    for name, gradient in gradients.items():
        expected = gradient * coefficient
        errors[name] = gradient.abs() * torch.maximum(coefficient - low, high - coefficient)
        errors[name] += _gamma(3) * gradient.abs() * high
        _assert_error_bound(observed[name], expected, errors[name], f"clipping: {name}")
    return coefficient, errors


def _check_adam_trajectory(records):
    """Independently verify FP32 clipping and AdamW against FP64 arithmetic.

    Reference moments use validated optimizer-input gradients, never gradients
    inferred from observed moments. Bounds propagate FP32 operation roundoff,
    including signed cancellation; they are independent of CP error tolerances.
    """
    parameters = records["initial_parameters"]
    assert all(value.dtype == torch.float32 for value in parameters.values())
    moments, rounding = {}, {}
    contexts = []
    tiny = torch.finfo(torch.float32).tiny
    for index, update in enumerate(records["updates"], start=1):
        coefficient, clip_errors = _check_clipping(update)
        settings = update["optimizer"]
        beta1, beta2 = settings["betas"]
        lr, eps, decay = settings["lr"], settings["eps"], settings["weight_decay"]
        assert set(update["moments"]) == set(update["gradients"])
        assert set(update["deltas"]) == set(update["gradients"])
        context = {
            "coefficient": coefficient,
            "clip_errors": clip_errors,
            "rounding": {},
            "before": {},
            "delta_rounding": {},
        }
        for name, gradient in update["clipped_gradients"].items():
            h = gradient.double()
            m, v = moments.get(name, (torch.zeros_like(h), torch.zeros_like(h)))
            rm, rv = rounding.get(name, (torch.zeros_like(h), torch.zeros_like(h)))
            previous = records["updates"][index - 2]["moments"][name] if index > 1 else None
            observed_m = previous["exp_avg"].double() if previous else torch.zeros_like(h)
            observed_v = previous["exp_avg_sq"].double() if previous else torch.zeros_like(h)
            rm = (
                beta1 * rm
                + _gamma(6) * (observed_m.abs() + (1 - beta1) * (h - observed_m).abs())
                + tiny
            )
            rv = (
                beta2 * rv
                + _gamma(8) * (beta2 * observed_v.abs() + (1 - beta2) * h.square())
                + tiny
            )
            m = beta1 * m + (1 - beta1) * h
            v = beta2 * v + (1 - beta2) * h.square()
            actual = update["moments"][name]
            assert actual["exp_avg"].dtype == actual["exp_avg_sq"].dtype == torch.float32
            assert update["parameters"][name].dtype == torch.float32
            assert float(actual["step"]) == index
            _assert_error_bound(actual["exp_avg"], m, rm, f"Adam first moment: {name}")
            _assert_error_bound(actual["exp_avg_sq"], v, rv, f"Adam second moment: {name}")
            # Validate the update using independently checked observed moments;
            # their storage-rounding errors must not be amplified through eps.
            before = parameters[name].double()
            after = update["parameters"][name]
            assert update["deltas"][name].dtype == torch.float32
            torch.testing.assert_close(
                update["deltas"][name], after - parameters[name], rtol=0, atol=0
            )
            context["before"][name] = before
            direction = (actual["exp_avg"].double() / (1 - beta1**index)) / (
                (actual["exp_avg_sq"].double() / (1 - beta2**index)).sqrt() + eps
            )
            delta = lr * direction
            expected = before * (1 - lr * decay) - delta
            parameter_error = (
                _gamma(32) * (before.abs() + (lr * decay * before).abs() + delta.abs()) + tiny
            )
            _assert_error_bound(
                update["parameters"][name],
                expected,
                parameter_error,
                f"Adam parameter update: {name}",
            )
            # The observed delta subtracts two FP32 stored parameters. Include
            # this separate rounding after the independently checked update.
            context["delta_rounding"][name] = (
                parameter_error + _gamma(1) * (after.double().abs() + before.abs()) + tiny
            )
            moments[name], rounding[name] = (m, v), (rm, rv)
            context["rounding"][name] = (rm, rv)
        parameters = update["parameters"]
        contexts.append(context)
    return contexts


def _adam_direction_difference_bound(left, right, errors, settings, step):
    """Propagate input-derived moment errors through positive Adam denominators.

    Each actual moment already passed its independent FP64 recurrence. Using
    those checked denominators retains their correlation with small first
    moments, unlike an interval that independently lets the variance reach zero.
    The square-root difference identity handles either sign of the first moment.
    """
    beta1, beta2 = settings["betas"]
    first_error, second_error = errors
    first_error = first_error / (1 - beta1**step)
    second_error = second_error / (1 - beta2**step)
    first = [state["exp_avg"].double() / (1 - beta1**step) for state in (left, right)]
    roots = [(state["exp_avg_sq"].double() / (1 - beta2**step)).sqrt() for state in (left, right)]
    denominators = [value + settings["eps"] for value in roots]
    root_sum = roots[0] + roots[1]
    # When both variances are exactly zero, their square roots are equal.
    root_error = torch.where(
        root_sum > 0,
        second_error / torch.where(root_sum > 0, root_sum, torch.ones_like(root_sum)),
        torch.zeros_like(root_sum),
    )
    bounds = [
        first_error / denominators[index]
        + first[1 - index].abs() * root_error / (denominators[0] * denominators[1])
        for index in (0, 1)
    ]
    return torch.minimum(*bounds)


def _compare_bf16_updates(actual, expected):
    left_contexts, right_contexts = _check_adam_trajectory(actual), _check_adam_trajectory(expected)
    residuals, moment_errors, input_errors = [], {}, {}
    assert len(actual["updates"]) == len(expected["updates"])
    for step, (left, right, lc, rc) in enumerate(
        zip(actual["updates"], expected["updates"], left_contexts, right_contexts), start=1
    ):
        assert left["optimizer"] == right["optimizer"]
        measured = {
            "gradient_relative_l2": _bf16_tensors(left["gradients"], right["gradients"], atol=1e-3),
            "update_relative_l2": _relative_l2(left["deltas"], right["deltas"], limit=4 / 128),
        }
        _bf16_tensors(left["parameters"], right["parameters"], atol=1e-3)
        _bf16_tensors({"loss": left["loss"]}, {"loss": right["loss"]}, atol=2e-3)
        beta1, beta2 = right["optimizer"]["betas"]
        for name, state in right["moments"].items():
            g = right["gradients"][name].double()
            h = right["clipped_gradients"][name].double()
            gradient_error = 1e-3 + 0.02 * g.abs()
            clip_error = (
                lc["coefficient"] * gradient_error
                + (lc["coefficient"] - rc["coefficient"]).abs() * g.abs()
            )
            clip_error += lc["clip_errors"][name] + rc["clip_errors"][name]
            em, ev = moment_errors.get(name, (torch.zeros_like(g), torch.zeros_like(g)))
            em = beta1 * em + (1 - beta1) * clip_error
            ev = beta2 * ev + (1 - beta2) * clip_error * (2 * h.abs() + clip_error)
            moment_errors[name] = (em, ev)
            for index, (key, error) in enumerate((("exp_avg", em), ("exp_avg_sq", ev))):
                bound = error + lc["rounding"][name][index] + rc["rounding"][name][index]
                _assert_error_bound(
                    left["moments"][name][key], state[key].double(), bound, f"CP {key}: {name}"
                )
            # Only gradients accepted above and independently checked clipping
            # feed this tighter update bound; no error is inferred from deltas.
            actual_h = left["clipped_gradients"][name].double()
            dm, dv = input_errors.get(name, (torch.zeros_like(h), torch.zeros_like(h)))
            dm = beta1 * dm + (1 - beta1) * (actual_h - h).abs()
            dv = beta2 * dv + (1 - beta2) * (actual_h.square() - h.square()).abs()
            input_errors[name] = (dm, dv)
            input_bounds = tuple(
                error + lc["rounding"][name][index] + rc["rounding"][name][index]
                for index, error in enumerate((dm, dv))
            )
            direction_error = _adam_direction_difference_bound(
                left["moments"][name], state, input_bounds, right["optimizer"], step
            )
            settings = right["optimizer"]
            delta_error = settings["lr"] * (
                direction_error
                + settings["weight_decay"] * (lc["before"][name] - rc["before"][name]).abs()
            )
            delta_error += lc["delta_rounding"][name] + rc["delta_rounding"][name]
            _assert_error_bound(
                left["deltas"][name],
                right["deltas"][name].double(),
                delta_error,
                f"CP update: {name}",
            )
            _compare(left["moments"][name]["step"], state["step"], exact=True)
        for key in ("exp_avg", "exp_avg_sq"):
            measured[key + "_relative_l2"] = _relative_l2(
                {name: state[key] for name, state in left["moments"].items()},
                {name: state[key] for name, state in right["moments"].items()},
            )
        residuals.append(measured)
    return residuals


def _optimizer_fixture(gradients):
    """Record real CPU AdamW steps for oracle failure/cancellation controls."""
    parameter = torch.nn.Parameter(torch.full_like(gradients[0], 0.1))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3, eps=1e-4, weight_decay=1e-4)
    records = {"initial_parameters": {"p": parameter.detach().clone()}, "updates": []}
    settings = {
        key: optimizer.param_groups[0][key] for key in ("lr", "betas", "eps", "weight_decay")
    }
    settings["max_grad_norm"] = 1.0
    for gradient in gradients:
        before = parameter.detach().clone()
        parameter.grad = gradient.clone()
        torch.nn.utils.clip_grad_norm_([parameter], 1.0)
        clipped = parameter.grad.clone()
        optimizer.step()
        records["updates"].append(
            {
                "gradients": {"p": gradient.clone()},
                "clipped_gradients": {"p": clipped},
                "parameters": {"p": parameter.detach().clone()},
                "deltas": {"p": parameter.detach() - before},
                "moments": {
                    "p": {key: value.clone() for key, value in optimizer.state[parameter].items()}
                },
                "optimizer": settings,
                "loss": torch.tensor([1.0]),
            }
        )
    return records


@pytest.mark.parametrize("corruption", ["warm_moment", "no_clipping", "missing_update"])
def test_adam_oracle_rejects_corrupted_state(corruption):
    gradients = [torch.linspace(-2, 3, 64) * factor for factor in (1.0, -0.9, 0.7)]
    records = _optimizer_fixture(gradients)
    _check_adam_trajectory(records)
    corrupted = copy.deepcopy(records)
    update = corrupted["updates"][1]
    if corruption == "warm_moment":
        update["moments"]["p"]["exp_avg"][0] += 1e-5
        _relative_l2(
            {"p": update["moments"]["p"]["exp_avg"]},
            {"p": records["updates"][1]["moments"]["p"]["exp_avg"]},
        )
    elif corruption == "no_clipping":
        update["clipped_gradients"]["p"] = update["gradients"]["p"].clone()
    else:
        update["parameters"]["p"] = corrupted["updates"][0]["parameters"]["p"].clone()
    with pytest.raises(AssertionError):
        _check_adam_trajectory(corrupted)


def test_adam_error_bound_preserves_signed_gradient_cancellation():
    gradients = [torch.full((64,), 0.01), torch.full((64,), 0.01)]
    gradients[0][0], gradients[1][0] = 0.3, -0.27
    perturbed = [gradient.clone() for gradient in gradients]
    for gradient in perturbed:
        gradient[0] += 8e-4
    reference, actual = _optimizer_fixture(gradients), _optimizer_fixture(perturbed)
    m_ref = reference["updates"][1]["moments"]["p"]["exp_avg"]
    m_actual = actual["updates"][1]["moments"]["p"]["exp_avg"]
    assert (m_actual - m_ref).abs()[0] > 1e-4 + 0.02 * m_ref.abs()[0]
    _compare_bf16_updates(actual, reference)


def test_adam_update_bound_preserves_near_zero_gradient_sign_changes():
    # Exact values from the retained CP-only full-training failure. Both raw
    # gradients meet the existing criterion, but Adam amplifies their sign change.
    gradients = [torch.full((2048,), 0.01)]
    gradients[0][0] = -6.103515625e-5
    perturbed = [value.clone() for value in gradients]
    perturbed[0][0] = 0.000244140625
    reference, actual = _optimizer_fixture(gradients), _optimizer_fixture(perturbed)
    delta, expected = actual["updates"][0]["deltas"]["p"], reference["updates"][0]["deltas"]["p"]
    assert (delta - expected).abs()[0] > 1e-3 + 0.02 * expected.abs()[0]
    _compare_bf16_updates(actual, reference)


@pytest.mark.parametrize("corruption", ["delta_only", "parameter_and_delta"])
def test_adam_oracle_rejects_local_update_corruption_below_global_cap(corruption):
    reference = _optimizer_fixture([torch.full((2048,), 0.01)])
    actual = copy.deepcopy(reference)
    update = actual["updates"][0]
    if corruption == "parameter_and_delta":
        update["parameters"]["p"][0] += 1e-4
        update["deltas"]["p"] = update["parameters"]["p"] - actual["initial_parameters"]["p"]
    else:
        update["deltas"]["p"][0] += 1e-4
    _relative_l2(update["deltas"], reference["updates"][0]["deltas"], limit=4 / 128)
    with pytest.raises(AssertionError):
        _compare_bf16_updates(actual, reference)


@pytest.mark.parametrize("sign", [-1.0, 0.0, 1.0])
def test_adam_direction_bound_handles_signed_moments_and_zero_variance(sign):
    settings = {"betas": (0.9, 0.999), "eps": 1e-4}
    left = {"exp_avg": torch.tensor([sign * 2e-5, 0.0]), "exp_avg_sq": torch.tensor([5e-11, 0.0])}
    right = {"exp_avg": torch.tensor([-sign * 1e-5, 0.0]), "exp_avg_sq": torch.tensor([1e-11, 0.0])}
    errors = tuple(
        (left[key].double() - right[key].double()).abs() for key in ("exp_avg", "exp_avg_sq")
    )
    bound = _adam_direction_difference_bound(left, right, errors, settings, 1)
    directions = [
        state["exp_avg"].double() / 0.1 / ((state["exp_avg_sq"].double() / 0.001).sqrt() + 1e-4)
        for state in (left, right)
    ]
    torch.testing.assert_close(bound[1], torch.tensor(0.0, dtype=torch.float64), rtol=0, atol=0)
    assert torch.all(
        (directions[0] - directions[1]).abs() <= bound * (1 + 16 * torch.finfo(torch.float64).eps)
    )


@pytest.mark.parametrize("factor", [0.0, 0.5, 2.0])
def test_bf16_oracle_rejects_wrong_reduction_even_below_absolute_floor(factor):
    expected = {"small_gradient": torch.tensor([1e-5, -2e-5])}
    actual = {name: value * factor for name, value in expected.items()}
    with pytest.raises(AssertionError):
        _bf16_tensors(actual, expected, atol=1e-3)


@pytest.mark.skipif(
    torch.cuda.device_count() < 4, reason="complete CP2 x DP2 SFT requires four CUDA ranks"
)
@pytest.mark.parametrize(
    "mode,checkpointing", [("full", False), ("full", True), ("lora", False), ("lora", True)]
)
def test_cuda_wan_sft_cp_equivalence(tmp_path, mode, checkpointing):
    _compare_sft_layouts(tmp_path, mode, checkpointing)


def _compare_sft_layouts(tmp_path, mode, checkpointing, tp_size=1, sequence_parallel=None):
    root = tmp_path.resolve() / "fixture"
    # Keep codec/model fixture construction out of the process spawning CUDA
    # ranks. Native initialization in that parent caused a later worker SIGSEGV.
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("wan_sft_fixture.py")),
            str(root),
            str(2 * tp_size),
        ],
        check=True,
    )
    groups = {}
    for degree in (1, 2):
        directory = tmp_path / f"cp{degree}"
        # CP composition is compared at fixed TP/SP. The separate strict FP32
        # probe compares all composed modes to an unsharded reference. Changing
        # TP also changes native BF16 kernel arithmetic in small q/k gradients;
        # the common-input endpoint control above distinguishes that from a
        # wrong head split without weakening any gradient/update tolerance.
        current_tp = tp_size
        results = run_gloo_ranks(
            _worker,
            directory,
            2 * degree * current_tp,
            {
                "fixture": str(root),
                "cp_size": degree,
                "tp_size": current_tp,
                "sequence_parallel": sequence_parallel if current_tp > 1 else None,
                "mode": mode,
                "checkpointing": checkpointing,
            },
        )
        for row in results:
            assert "error" not in row, row.get("error")
            assert row["status"] == "pass"
        groups[degree] = [
            torch.load(directory / f"observations-{rank}.pt", weights_only=True)
            for rank in range(2 * degree * current_tp)
        ]
    residuals = []
    for rank, actual in enumerate(groups[2]):
        expected = groups[1][rank // (2 * tp_size) * tp_size]
        _compare(actual["initial_parameters"], expected["initial_parameters"], exact=True)
        _compare(actual["batches"], expected["batches"], exact=True)
        for left, right in zip(actual["forwards"], expected["forwards"]):
            for name in ("t", "latents", "prompt_embeds"):
                _compare(left[name], right[name], exact=True)
            _bf16_tensors(
                {"velocity": left["velocity"]}, {"velocity": right["velocity"]}, atol=2 / 128
            )
        residuals.append(_compare_bf16_updates(actual, expected))
        if rank % (2 * tp_size):
            _compare(actual, groups[2][rank - rank % (2 * tp_size)], exact=True)
    print(
        json.dumps(
            {
                "mode": mode,
                "checkpointing": checkpointing,
                "optimizer_steps": 4,
                "logical_replicas": 2,
                "tp_size": tp_size,
                "sequence_parallel": sequence_parallel,
                "ranks": results,
                "residuals": residuals,
            }
        )
    )


@pytest.mark.skipif(
    torch.cuda.device_count() < 8, reason="complete DP2 x CP2 x TP2 SFT requires eight CUDA ranks"
)
@pytest.mark.parametrize("sequence_parallel", [False, True])
@pytest.mark.parametrize(
    "mode,checkpointing", [("full", False), ("full", True), ("lora", False), ("lora", True)]
)
def test_cuda_wan_sft_tp_cp_equivalence(tmp_path, mode, checkpointing, sequence_parallel):
    _compare_sft_layouts(
        tmp_path, mode, checkpointing, tp_size=2, sequence_parallel=sequence_parallel
    )
