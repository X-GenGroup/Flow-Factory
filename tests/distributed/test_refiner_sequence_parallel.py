# Copyright 2026 Jayce-Ping
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Native refiner activation, derivative and warm Adam controls across TP/SP/CP."""

import copy
import json
import traceback
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from conftest import run_gloo_ranks


def _build(family, mode, *, chunk=False, bounded_padding=False):
    if family == "z-image":
        from test_z_image_context_parallel import build_transformer, install_lora

        model = build_transformer()
        model = type(model).from_config(model.config, n_refiner_layers=2)
        if bounded_padding:
            # Match the native pipeline fixture's bounded learned padding values.
            with torch.no_grad():
                model.x_pad_token.normal_(0, 0.05)
                model.cap_pad_token.normal_(0, 0.05)
    else:
        from h3_cp_fixture import install_lora

        from diffusers import MiniMaxH3Transformer3DModel

        torch.manual_seed(59)
        model = MiniMaxH3Transformer3DModel(
            num_attention_heads=8,
            attention_head_dim=8,
            hidden_size=32,
            num_layers=2,
            num_refiner_layers=2,
            ffn_dim=64,
            in_channels=24,
            audio_in_channels=32,
            patch_size=(1, 2, 2),
            text_dim=64,
            freq_dim=16,
            time_embed_hidden_dim=32,
            time_embed_dim=16,
            rope_freq_dim=1,
        )
    if chunk and family != "z-image":
        from flow_factory.models.minimax_h3._chunking import (
            install_h3_attention_norm_chunking,
            install_h3_feed_forward_chunking,
        )

        install_h3_feed_forward_chunking(model, max_tokens=3)
        install_h3_attention_norm_chunking(model, max_tokens=3)
    return install_lora(model) if mode == "lora" else model


def _inputs(family, step, device, dtype):
    if family == "z-image":
        from test_z_image_context_parallel import model_inputs

        values = model_inputs(step, dtype)
    else:
        from h3_cp_fixture import model_inputs

        values = model_inputs(family, step=step, odd=bool(step % 2))
        if step % 2:
            # Keep the main SP length even while independently exercising odd text.
            # CP still sees its own padding when the total is not divisible by four.
            rows = values["position_ids"].shape[0]
            values["hidden_states"] = torch.cat(
                (values["hidden_states"], values["hidden_states"][:, :1] * 0.37), dim=1
            )
            values["video_indices"] = torch.cat((values["video_indices"], torch.tensor([rows])))
            values["position_ids"] = torch.cat(
                (values["position_ids"], torch.tensor([[3.0, 1.0, 0.0]]))
            )
            values["token_tags"] = torch.cat((values["token_tags"], torch.tensor([0])))
            values["timestep_indices"] = torch.cat((values["timestep_indices"], torch.tensor([0])))
    leaves = {}

    def convert(value, name):
        if isinstance(value, torch.Tensor):
            value = value.to(
                device=device, dtype=dtype if value.is_floating_point() else value.dtype
            )
            # Position IDs are fixed geometry, not differentiable model inputs.
            if value.is_floating_point() and name != "position_ids":
                value = value.detach().requires_grad_()
                leaves[name] = value
            return value
        if isinstance(value, list):
            return [convert(item, f"{name}.{index}") for index, item in enumerate(value)]
        return value

    return {name: convert(value, name) for name, value in values.items()}, leaves


def _worker(rank, store_path, result_dir, options):
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.tensor import DTensor, Shard

    from flow_factory.models.tensor_parallel import (
        apply_tensor_parallel,
        peel_peft_wrapper,
        resolve_parallel_plan,
    )

    cp = options.get("cp", 1)
    cuda = options.get("cuda", False)
    device = torch.device("cuda", rank) if cuda else torch.device("cpu")
    if cuda:
        torch.cuda.set_device(device)
    dist.init_process_group(
        "nccl" if cuda else "gloo",
        store=dist.FileStore(store_path, 2 * cp),
        rank=rank,
        world_size=2 * cp,
    )
    result = {"rank": rank, **options}
    try:
        family = options["family"]
        if family == "z-image":
            from flow_factory.models.z_image.z_image import ZImageAdapter as adapter
        else:
            from h3_cp_fixture import adapter_class

            adapter = adapter_class(family)
        mesh = DeviceMesh(
            device.type, torch.arange(2 * cp).reshape(cp, 2), mesh_dim_names=("cp", "tp")
        )
        dtype = torch.float32 if cuda and not options.get("double") else torch.float64
        build_options = {"bounded_padding": options.get("bounded_padding", False)}
        module = _build(family, options["mode"], chunk=True, **build_options).to(
            device=device, dtype=dtype
        )
        reference = _build(family, options["mode"], **build_options).to(device=device, dtype=dtype)
        native = peel_peft_wrapper(module)
        apply_tensor_parallel(
            native, mesh["tp"], resolve_parallel_plan(adapter, sequence_parallel=options["sp"])
        )
        if cp > 1:
            from diffusers.models._modeling_parallel import ContextParallelConfig

            native_mesh = DeviceMesh(
                device.type,
                mesh.mesh.T.reshape(1, 2, 1, cp),
                mesh_dim_names=("data", "tensor", "ring", "ulysses"),
            )
            adapter.configure_context_parallel_module(native)
            native.set_attention_backend("native")
            adapter.install_context_parallel_module(
                native, ContextParallelConfig(ulysses_degree=cp, mesh=native_mesh)
            )
            adapter.compose_context_parallel_sequence_boundaries(native)
        checkpoint = options["checkpoint"]
        if checkpoint == "native":
            native.enable_gradient_checkpointing()
        elif checkpoint == "in-forward":
            from flow_factory.models.minimax_h3._chunking import (
                install_h3_in_forward_block_checkpointing,
            )

            assert install_h3_in_forward_block_checkpointing(native) == 4
            assert install_h3_in_forward_block_checkpointing(native) == 4
        shapes = []
        if family == "z-image":
            stacks = (native.noise_refiner, native.context_refiner)
        else:
            stacks = (native.token_refiner.refiner_blocks,)
            if options["sp"]:
                assert type(stacks[0][0]).__name__ == "H3SequenceParallelTokenRefinerBlock"
                assert type(stacks[0][0]).__name__ in native._no_split_modules
                assert type(stacks[0][0]).__name__ in native._repeated_blocks
                from accelerate import FullyShardedDataParallelPlugin

                from flow_factory.models.model_bundle import ModelBundle

                plugin = FullyShardedDataParallelPlugin(
                    fsdp_version=2, auto_wrap_policy="TRANSFORMER_BASED_WRAP"
                )
                plugin.set_auto_wrap_policy(ModelBundle({"transformer": module}))
                assert (
                    type(stacks[0][0]) in plugin.auto_wrap_policy.keywords["transformer_layer_cls"]
                )
        for stack in stacks:
            for block in stack:

                def observe(owner, args, kwargs):
                    shapes.append(int(args[0].shape[1]))

                block.register_forward_pre_hook(observe, with_kwargs=True)
        actual_parameters = dict(module.named_parameters())
        reference_parameters = dict(reference.named_parameters())
        assert actual_parameters.keys() == reference_parameters.keys()
        for name, value in reference_parameters.items():
            actual = actual_parameters[name]
            assert value.requires_grad == actual.requires_grad, name
            torch.testing.assert_close(actual.full_tensor(), value, rtol=0, atol=0)
        if family != "z-image":
            for name, value in actual_parameters.items():
                if (
                    "token_refiner.refiner_blocks" in name
                    and ".attn.to_q." in name
                    and "lora_A" not in name
                ):
                    assert Shard(0) in value.placements, name
        optimizer = torch.optim.AdamW(
            [p for p in module.parameters() if p.requires_grad], lr=0.001, eps=1e-4
        )
        reference_optimizer = torch.optim.AdamW(
            [p for p in reference.parameters() if p.requires_grad], lr=0.001, eps=1e-4
        )
        tolerance = (
            dict(rtol=5e-4, atol=5e-6)
            if family == "z-image" or cuda
            else dict(rtol=2e-8, atol=2e-10)
        )
        result["updates"] = []
        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
            pending = []
            # Unequal text lengths coexist before backward; replay must own its mask.
            for micro in range(2):
                values, leaves = _inputs(family, 2 * step + micro, device, dtype)
                expected_values, expected_leaves = _inputs(family, 2 * step + micro, device, dtype)
                outputs = module(**values)
                expected_outputs = reference(**expected_values)
                if family == "z-image":
                    outputs, expected_outputs = outputs[0], expected_outputs[0]
                for actual, expected in zip(outputs, expected_outputs):
                    torch.testing.assert_close(actual, expected, **tolerance)
                pending.append((outputs, expected_outputs, leaves, expected_leaves))
            for outputs, expected_outputs, leaves, expected_leaves in reversed(pending):
                sum(x.square().mean() for x in outputs).backward()
                sum(x.square().mean() for x in expected_outputs).backward()
                for name, expected in expected_leaves.items():
                    actual = leaves[name]
                    assert (actual.grad is None) == (expected.grad is None), name
                    if actual.grad is not None:
                        if cp > 1:
                            dist.all_reduce(actual.grad, group=mesh["cp"].get_group())
                        # Native sinusoidal time embedding explicitly computes in FP32.
                        input_tolerance = (
                            dict(rtol=2e-6, atol=2e-9)
                            if name == "timestep" and not cuda
                            else tolerance
                        )
                        torch.testing.assert_close(
                            actual.grad,
                            expected.grad,
                            **input_tolerance,
                            msg=lambda message: name + "\n" + message,
                        )
            active = []
            nonzero_gradients = {
                "actual_nonzero_parameters": 0,
                "reference_nonzero_parameters": 0,
                "actual_nonzero_refiner_parameters": 0,
                "reference_nonzero_refiner_parameters": 0,
            }
            for name, expected in reference_parameters.items():
                actual = actual_parameters[name]
                assert (actual.grad is None) == (expected.grad is None), name
                if expected.grad is None:
                    continue
                if cp > 1:
                    dist.all_reduce(actual.grad.to_local(), group=mesh["cp"].get_group())
                full_gradient = actual.grad.full_tensor()
                is_refiner = any(
                    stack in name
                    for stack in ("noise_refiner.", "context_refiner.", "token_refiner.")
                )
                for owner, gradient in (("actual", full_gradient), ("reference", expected.grad)):
                    assert torch.isfinite(gradient).all(), (owner, name)
                    nonzero = int(bool(torch.count_nonzero(gradient)))
                    nonzero_gradients[f"{owner}_nonzero_parameters"] += nonzero
                    nonzero_gradients[f"{owner}_nonzero_refiner_parameters"] += nonzero * is_refiner
                if not torch.allclose(full_gradient, expected.grad, **tolerance):
                    torch.save(
                        {
                            "name": name,
                            "actual": full_gradient.detach().cpu(),
                            "expected": expected.grad.detach().cpu(),
                        },
                        Path(result_dir, f"gradient-rank-{rank}.pt"),
                    )
                torch.testing.assert_close(
                    full_gradient,
                    expected.grad,
                    **tolerance,
                    msg=lambda message: name + "\n" + message,
                )
                active.append(name)
            assert any("refiner" in name for name in active)
            assert all(count > 0 for count in nonzero_gradients.values()), nonzero_gradients
            optimizer.step()
            reference_optimizer.step()
            for name, expected in reference_parameters.items():
                actual = actual_parameters[name]
                torch.testing.assert_close(
                    actual.full_tensor(),
                    expected,
                    **tolerance,
                    msg=lambda message: name + "\n" + message,
                )
                for key, value in reference_optimizer.state[expected].items():
                    observed = optimizer.state[actual][key]
                    observed = observed.full_tensor() if isinstance(observed, DTensor) else observed
                    torch.testing.assert_close(
                        observed,
                        value,
                        **tolerance,
                        msg=lambda message: name + key + "\n" + message,
                    )
            result["updates"].append(
                {"step": step, "active_parameters": len(active), **nonzero_gradients}
            )
        expected_shapes = (
            ({16, 32} if options["sp"] else {32, 64})
            if family == "z-image"
            else ({2, 3} if options["sp"] else {4, 5})
        )
        assert set(shapes) == expected_shapes, shapes
        result["refiner_shapes"] = sorted(set(shapes))
        canonical = {
            name: value.full_tensor() if isinstance(value, DTensor) else value
            for name, value in module.state_dict().items()
        }
        assert canonical.keys() == reference.state_dict().keys()
        for name, value in reference.state_dict().items():
            torch.testing.assert_close(canonical[name], value, **tolerance, msg=name)
        restored = copy.deepcopy(reference)
        restored.load_state_dict(canonical, strict=True)
        values, _ = _inputs(family, 7, device, dtype)
        with torch.no_grad():
            actual_outputs, expected_outputs = restored(**values), reference(**values)
        if family == "z-image":
            actual_outputs, expected_outputs = actual_outputs[0], expected_outputs[0]
        for actual, expected in zip(actual_outputs, expected_outputs):
            torch.testing.assert_close(actual, expected, **tolerance)
        result["canonical_state_keys"] = len(canonical)
        if family != "z-image":
            result["refiner_placements"] = {
                name: {
                    "global_shape": list(value.shape),
                    "local_shape": list(value.to_local().shape),
                    "placements": [str(x) for x in value.placements],
                }
                for name, value in actual_parameters.items()
                if "token_refiner.refiner_blocks.0" in name
            }
        result["status"] = "pass"
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result))
        dist.destroy_process_group()


@pytest.mark.parametrize("family", ["z-image", "t2va"])
@pytest.mark.parametrize("mode", ["full", "lora"])
@pytest.mark.parametrize("checkpoint", ["none", "native"])
def test_cpu_native_refiner_sp_updates(tmp_path, family, mode, checkpoint):
    rows = run_gloo_ranks(
        _worker, tmp_path, 2, dict(family=family, mode=mode, checkpoint=checkpoint, sp=True)
    )
    assert all(row.get("status") == "pass" for row in rows), rows


@pytest.mark.parametrize("family", ["z-image", "t2va"])
def test_cpu_native_refiner_tp_control(tmp_path, family):
    rows = run_gloo_ranks(
        _worker, tmp_path, 2, dict(family=family, mode="full", checkpoint="none", sp=False)
    )
    assert all(row.get("status") == "pass" for row in rows), rows


@pytest.mark.parametrize("family,mode", [("t2va", "full"), ("fl2va", "lora"), ("ref2va", "full")])
def test_cpu_h3_refiner_cp_checkpoint_composition(tmp_path, family, mode):
    rows = run_gloo_ranks(
        _worker, tmp_path, 4, dict(family=family, mode=mode, checkpoint="in-forward", sp=True, cp=2)
    )
    assert all(row.get("status") == "pass" for row in rows), rows


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
@pytest.mark.parametrize("family,mode", [("z-image", "full"), ("t2va", "lora")])
def test_cuda_native_refiner_sp_updates(tmp_path, family, mode):
    rows = run_gloo_ranks(
        _worker,
        tmp_path,
        2,
        dict(
            family=family, mode=mode, checkpoint="native", sp=True, cuda=True, bounded_padding=True
        ),
    )
    assert all(row.get("status") == "pass" for row in rows), rows
