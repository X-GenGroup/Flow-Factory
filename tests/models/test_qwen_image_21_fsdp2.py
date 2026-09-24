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

"""Qwen-Image 2.1 native prefix KV cache under FSDP2 parameter sharding."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from diffusers import QwenImage21Transformer2DModel
from flow_factory.models.qwen_image_21 import qwen_image_21 as qwen_image_21_module
from flow_factory.models.qwen_image_21.qwen_image_21 import QwenImage21Adapter

WORLD_SIZE = 2
NUM_LAYERS = 2


def _tiny_transformer() -> QwenImage21Transformer2DModel:
    transformer = QwenImage21Transformer2DModel(
        patch_size=1,
        in_channels=8,
        out_channels=8,
        num_layers=NUM_LAYERS,
        attention_head_dim=16,
        num_attention_heads=2,
        context_in_dim=16,
        mlp_ratio=2,
        axes_dims_rope=(4, 6, 6),
    )
    # Mirror LoRA training: only block parameters train, so the first block's prefill
    # inputs do not require gradients and FSDP2 cannot trigger its post-backward from them.
    for name, parameter in transformer.named_parameters():
        parameter.requires_grad_(name.startswith("transformer_blocks."))
    return transformer


def _adapter(transformer: torch.nn.Module) -> QwenImage21Adapter:
    adapter = object.__new__(QwenImage21Adapter)
    adapter.component_runtime = SimpleNamespace(get_component=lambda name: transformer)
    adapter._warned_cfg_without_negative = False
    adapter._warned_negative_without_cfg = False
    return adapter


def _prediction_kwargs() -> Dict[str, Any]:
    return dict(
        t=torch.tensor([500.0]),
        latents=torch.randn(1, 4, 8),
        prompt_embeds=torch.randn(1, 3, 16),
        prompt_embeds_mask=torch.ones(1, 3, dtype=torch.long),
        image_pad_mask=torch.tensor([[False, True, False]]),
        img_shapes=[(1, 2, 2), (1, 2, 2)],
        condition_image_latents=torch.randn(1, 4, 8),
        negative_prompt_embeds=torch.randn(1, 2, 16),
        negative_prompt_embeds_mask=torch.ones(1, 2, dtype=torch.long),
        negative_image_pad_mask=torch.tensor([[True, False]]),
        guidance_scale=2.0,
        attention_kwargs=None,
        use_kv_cache=True,
    )


def _prepare_like_accelerate_fsdp2(transformer: torch.nn.Module) -> None:
    """Apply Accelerate's FSDP2 activation-checkpoint and transformer-block wrap layout."""
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    mesh = init_device_mesh("cpu", (WORLD_SIZE,))
    for block in transformer.transformer_blocks:
        for child_name, child in list(block.named_children()):
            block.register_module(child_name, checkpoint_wrapper(child, preserve_rng_state=False))
        fully_shard(block, mesh=mesh, reshard_after_forward=True)
    fully_shard(transformer, mesh=mesh)


def _fsdp2_worker(rank: int, store_path: str, result_dir: str, bridge: bool) -> None:
    dist.init_process_group(
        "gloo", init_method=f"file://{store_path}", rank=rank, world_size=WORLD_SIZE
    )
    torch.manual_seed(0)
    transformer = _tiny_transformer()
    reference = copy.deepcopy(transformer)
    kwargs = _prediction_kwargs()

    _adapter(reference)._predict_velocity_one(**kwargs).square().mean().backward()
    expected = {
        name: parameter.grad.detach().clone()
        for name, parameter in reference.named_parameters()
        if parameter.requires_grad
    }

    registrations = []
    register = qwen_image_21_module._register_fsdp2_prefix_cache_pre_backward
    if bridge:

        def counting_register(module: torch.nn.Module, cache: Any) -> int:
            count = register(module, cache)
            registrations.append(count)
            return count

        qwen_image_21_module._register_fsdp2_prefix_cache_pre_backward = counting_register
    else:
        qwen_image_21_module._register_fsdp2_prefix_cache_pre_backward = lambda *_: 0

    _prepare_like_accelerate_fsdp2(transformer)
    result: Dict[str, Any] = {"registrations": registrations}
    try:
        _adapter(transformer)._predict_velocity_one(**kwargs).square().mean().backward()
    except RuntimeError as exc:
        result["error"] = str(exc)
    else:
        max_abs_diff = 0.0
        missing = []
        for name, parameter in transformer.named_parameters():
            canonical = name.replace("._checkpoint_wrapped_module", "")
            if canonical not in expected:
                continue
            if parameter.grad is None:
                missing.append(canonical)
                continue
            actual = parameter.grad.full_tensor()
            max_abs_diff = max(max_abs_diff, (actual - expected[canonical]).abs().max().item())
        result.update(
            max_abs_diff=max_abs_diff,
            missing=missing,
            block0_to_k_grad_norm=expected["transformer_blocks.0.attn.to_k.weight"].norm().item(),
        )
    finally:
        qwen_image_21_module._register_fsdp2_prefix_cache_pre_backward = register
    Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result), encoding="utf-8")
    dist.destroy_process_group()


def _run_two_ranks(tmp_path: Path, *, bridge: bool) -> list[Dict[str, Any]]:
    context = mp.get_context("spawn")
    store_path = str(tmp_path / "gloo-store")
    processes = [
        context.Process(target=_fsdp2_worker, args=(rank, store_path, str(tmp_path), bridge))
        for rank in range(WORLD_SIZE)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=120)
    hanging = [process for process in processes if process.is_alive()]
    for process in hanging:
        process.terminate()
        process.join(timeout=5)
    assert not hanging, "two-rank Qwen-Image 2.1 FSDP2 KV-cache run exceeded 120 seconds"
    assert [process.exitcode for process in processes] == [0] * WORLD_SIZE
    return [
        json.loads(Path(tmp_path, f"rank-{rank}.json").read_text(encoding="utf-8"))
        for rank in range(WORLD_SIZE)
    ]


def test_prefix_cache_bridge_is_a_noop_without_fsdp2() -> None:
    torch.manual_seed(0)
    transformer = _tiny_transformer()
    kwargs = _prediction_kwargs()
    cache = _adapter(transformer)._prefill_kv_cache_one(
        latents=kwargs["latents"],
        prompt_embeds=kwargs["prompt_embeds"],
        prompt_embeds_mask=None,
        image_pad_mask=kwargs["image_pad_mask"],
        img_shapes=kwargs["img_shapes"],
        condition_image_latents=kwargs["condition_image_latents"],
        attention_kwargs=None,
        context_name="cond",
    )

    assert qwen_image_21_module._register_fsdp2_prefix_cache_pre_backward(transformer, cache) == 0
    for layer_cache in cache.layer_caches:
        assert layer_cache.k.grad_fn is not None and layer_cache.v.grad_fn is not None


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="PyTorch gloo backend unavailable",
)
def test_two_rank_fsdp2_native_kv_cache_matches_unsharded_gradients(tmp_path: Path) -> None:
    results = _run_two_ranks(tmp_path, bridge=True)

    for result in results:
        assert "error" not in result, result.get("error")
        assert result["missing"] == []
        # Two denoising caches (positive and CFG negative) register one FSDP2 unit per block.
        assert result["registrations"] == [NUM_LAYERS, NUM_LAYERS]
        assert result["block0_to_k_grad_norm"] > 0
        assert result["max_abs_diff"] <= 1e-5


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="PyTorch gloo backend unavailable",
)
def test_two_rank_fsdp2_native_kv_cache_without_bridge_mixes_dtensor(tmp_path: Path) -> None:
    results = _run_two_ranks(tmp_path, bridge=False)

    for result in results:
        assert "mixed torch.Tensor and DTensor" in result.get("error", "")
