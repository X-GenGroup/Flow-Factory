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

"""SenseNova under tensor and sequence parallelism, behind an odd-length prefix.

The no-grad prefix passes fill the KV cache through the understanding branch at
the prompt's own length, and the CFG-unconditional template is 9 tokens -- odd
at tp=2 -- so the overlay must leave them whole and shard only the generation
branch that the denoising step trains. Each prefix kind is checked for the cache
it builds, then the denoising pass behind it for its output and every parameter
gradient against the unsharded model, and for its tokens actually being split.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.distributed as dist
from conftest import requires_gloo, run_gloo_ranks

WORLD_SIZE = 2
#: ``<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n<img>`` under the
#: SenseNova-U1.5 tokenizer, which every guided T2I run prefills.
PREFIX_LENGTH = 9
HIDDEN_SIZE = 32
#: Pixels per image token: ``patch_size / downsample_ratio`` in the config below.
PATCH_MERGE = 32


def _build() -> torch.nn.Module:
    from flow_factory.models.sensenova.modeling.neo_unify.configuration_neo_chat import (
        NEOChatConfig,
    )
    from flow_factory.models.sensenova.modeling.neo_unify.modeling_neo_chat import NEOChatModel
    from flow_factory.models.sensenova.pipeline import SenseNovaDenoiser

    config = NEOChatConfig(
        vision_config={
            "architectures": ["NEOVisionModel"],
            "patch_size": 16,
            "hidden_size": 16,
            "llm_hidden_size": HIDDEN_SIZE,
            "downsample_ratio": 0.5,
            "max_position_embeddings_vision": 128,
            "num_channels": 3,
            "rope_theta_vision": 10000.0,
        },
        llm_config={
            "architectures": ["Qwen3ForCausalLM"],
            "hidden_size": HIDDEN_SIZE,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "vocab_size": 64,
            "max_position_embeddings": 256,
            "rope_theta": 10000.0,
            "rope_theta_hw": 10000.0,
            "max_position_embeddings_hw": 128,
            "attention_bias": False,
            "attention_dropout": 0.0,
            "rms_norm_eps": 1e-6,
            "pad_token_id": 0,
            "eos_token_id": 1,
            "tie_word_embeddings": False,
        },
        downsample_ratio=0.5,
        template="neo1_0",
        patch_size=16,
        fm_head_layers=2,
        fm_head_dim=16,
        fm_head_mlp_ratio=1,
        use_pixel_head=True,
        t_eps=0.05,
        add_noise_scale_embedding=True,
        noise_scale=1.0,
        noise_scale_mode="resolution",
        noise_scale_base_image_seq_len=64,
        noise_scale_max_value=16.0,
        concat_time_token_num=0,
        time_schedule="standard",
        time_shift_type="exponential",
        base_shift=0.5,
        max_shift=1.15,
        base_image_seq_len=64,
        max_image_seq_len=4096,
    )
    torch.manual_seed(0)
    model = NEOChatModel(config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0, 0.2)
    return SenseNovaDenoiser(model).double()


def _prefill(denoiser: torch.nn.Module, kind: str):
    """Run the real no-grad prefix pass of ``kind`` on a ``PREFIX_LENGTH`` prompt."""
    from flow_factory.models.sensenova.modeling.neo_unify.modeling_qwen3 import (
        create_block_causal_mask,
    )

    generator = torch.Generator().manual_seed(PREFIX_LENGTH)
    positions = torch.arange(PREFIX_LENGTH)
    indexes = torch.stack([positions, torch.zeros_like(positions), torch.zeros_like(positions)])
    mask = {"full_attention": create_block_causal_mask(indexes[0])}
    with torch.no_grad():
        if kind == "t2i":
            input_ids = torch.randint(0, 64, (1, PREFIX_LENGTH), generator=generator)
            return denoiser.model._t2i_prefix_forward(input_ids, indexes, mask)
        embeds = torch.randn(1, PREFIX_LENGTH, HIDDEN_SIZE, generator=generator).double()
        return denoiser.model._it2i_prefix_forward(embeds, indexes, mask)


def _denoise(denoiser: torch.nn.Module, cache: Any, side: int) -> torch.Tensor:
    """Run the denoising pass behind ``cache``, as ``SenseNovaAdapter.forward`` does."""
    from flow_factory.models.sensenova.modeling.neo_unify.modeling_neo_chat import (
        prepare_flash_kv_cache,
    )

    grid = side // PATCH_MERGE
    prepare_flash_kv_cache(cache, current_len=grid * grid, batch_size=1)
    generator = torch.Generator().manual_seed(1)
    return denoiser(
        latents=torch.randn(1, 3, side, side, generator=generator).double(),
        timestep=torch.tensor(0.3, dtype=torch.float64),
        past_key_values=cache,
        indexes_image=denoiser.model._build_t2i_image_indexes(
            grid, grid, PREFIX_LENGTH, device="cpu"
        ),
        attention_mask={"full_attention": None},
        image_size=(side, side),
        noise_scale=1.0,
    )


def _count_tokens(denoiser: torch.nn.Module) -> Dict[str, List[int]]:
    """Record how many tokens each branch's first norm sees, per layer and call."""
    counts: Dict[str, List[int]] = {"understanding": [], "generation": []}
    for layer in denoiser.model.language_model.model.layers:
        for branch, norm in (
            ("understanding", layer.input_layernorm),
            ("generation", layer.input_layernorm_mot_gen),
        ):
            norm.register_forward_pre_hook(
                lambda _module, inputs, _branch=branch: counts[_branch].append(inputs[0].shape[1])
            )
    return counts


def _relative(received: torch.Tensor, expected: torch.Tensor) -> float:
    return ((received - expected).abs().max() / expected.abs().max()).item()


def _apply_sequence_parallel(denoiser: torch.nn.Module) -> None:
    from torch.distributed.device_mesh import init_device_mesh

    from flow_factory.models.sensenova.sensenova import SenseNovaAdapter
    from flow_factory.models.tensor_parallel import apply_tensor_parallel, resolve_parallel_plan

    mesh = init_device_mesh("cpu", (WORLD_SIZE,), mesh_dim_names=("tp",))
    apply_tensor_parallel(
        denoiser, mesh, resolve_parallel_plan(SenseNovaAdapter, sequence_parallel=True)
    )


def _parity_worker(rank: int, store_path: str, result_dir: str, options: Dict[str, Any]) -> None:
    store = dist.FileStore(store_path, WORLD_SIZE)
    dist.init_process_group("gloo", store=store, rank=rank, world_size=WORLD_SIZE)
    result: Dict[str, Any] = {"rank": rank}
    try:
        from torch.distributed.tensor import DTensor

        reference = _build()
        sharded = copy.deepcopy(reference)
        _apply_sequence_parallel(sharded)
        counts = _count_tokens(sharded)
        for kind in ("t2i", "it2i"):
            outcome: Dict[str, Any] = {}
            values = {}
            for label, denoiser in (("expected", reference), ("received", sharded)):
                cache, hidden = _prefill(denoiser, kind)
                velocity = _denoise(denoiser, cache, side=2 * PATCH_MERGE)
                velocity.pow(2).sum().backward()
                values[label] = {
                    "prefix hidden states": hidden,
                    "velocity": velocity.detach(),
                    **{f"keys {index}": layer.keys for index, layer in enumerate(cache.layers)},
                    **{
                        f"grad {name}": (
                            p.grad.full_tensor() if isinstance(p.grad, DTensor) else p.grad
                        )
                        for name, p in denoiser.named_parameters()
                        if p.grad is not None
                    },
                }
                denoiser.zero_grad(set_to_none=True)
            expected, received = values["expected"], values["received"]
            assert expected.keys() == received.keys(), sorted(expected.keys() ^ received.keys())
            heads = received["keys 0"].shape[1]
            for name in [name for name in expected if name.startswith("keys ")]:
                # Tensor parallelism splits the cache by kv head, never by token.
                expected[name] = expected[name][:, rank * heads : (rank + 1) * heads]
            relative = {
                name: _relative(received[name], expected[name])
                for name in expected
                if expected[name].abs().max() > 0
            }
            worst = max(relative, key=relative.get)
            outcome["relative"] = relative[worst]
            outcome["worst"] = worst
            outcome["gradients"] = sum(name.startswith("grad ") for name in relative)
            result[kind] = outcome
        result["tokens"] = counts
    except Exception as error:  # reported to the parent, which asserts on it
        result["error"] = f"{type(error).__name__}: {error}"
    Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result), encoding="utf-8")
    dist.destroy_process_group()


@requires_gloo
def test_an_odd_prefix_runs_whole_and_the_trained_pass_matches_unsharded(tmp_path: Path) -> None:
    results = run_gloo_ranks(_parity_worker, tmp_path, WORLD_SIZE, {})

    image_tokens = 2 * 2
    for result in results:
        assert "error" not in result, result["error"]
        for kind in ("t2i", "it2i"):
            outcome = result[kind]
            assert outcome["relative"] < 1e-9, (kind, outcome)
            # 13 generation-branch tensors per layer, the exit norm, and the 16 of
            # the patch embedder, the timestep and noise-scale embedders and the
            # pixel head.
            assert outcome["gradients"] == 43, (kind, outcome)
        # Two layers, two prefix kinds: the prompt always enters whole, the
        # image tokens always enter split.
        assert result["tokens"]["understanding"] == [PREFIX_LENGTH] * 4, result["tokens"]
        assert result["tokens"]["generation"] == [image_tokens // WORLD_SIZE] * 4, result["tokens"]


def _indivisible_grid_worker(
    rank: int, store_path: str, result_dir: str, options: Dict[str, Any]
) -> None:
    store = dist.FileStore(store_path, WORLD_SIZE)
    dist.init_process_group("gloo", store=store, rank=rank, world_size=WORLD_SIZE)
    result: Dict[str, Any] = {"rank": rank}
    try:
        sharded = _build()
        _apply_sequence_parallel(sharded)
        cache, _ = _prefill(sharded, "t2i")
        try:
            _denoise(sharded, cache, side=3 * PATCH_MERGE)
        except ValueError as error:
            result["refusal"] = str(error)
    except Exception as error:  # reported to the parent, which asserts on it
        result["error"] = f"{type(error).__name__}: {error}"
    Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result), encoding="utf-8")
    dist.destroy_process_group()


@requires_gloo
def test_an_indivisible_image_grid_is_refused_on_the_generation_call(tmp_path: Path) -> None:
    """The prompt never meets a scatter, but the image tokens do.

    A 3x3 grid splits 5/4 at tp=2, which would make the ranks infer different
    global lengths and deadlock at the next gather. ``apply_tensor_parallel``'s
    guard does not see this scatter, so the overlay has to refuse it itself --
    on the generation call, after an odd prefix has gone through.
    """
    results = run_gloo_ranks(_indivisible_grid_worker, tmp_path, WORLD_SIZE, {})

    for result in results:
        assert "error" not in result, result["error"]
        refusal = result.get("refusal")
        assert refusal is not None, result
        assert "9 generated-image tokens" in refusal, refusal
        assert "divisible by tp_size" in refusal, refusal
        assert "parallelism.sequence_parallel: false" in refusal, refusal
