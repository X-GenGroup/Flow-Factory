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

"""The adapters' region-entry plans match their unsharded models, gradient by gradient.

Wan, Qwen-Image, LTX-2 and SenseNova have parity tests of their own; these
cover the other plans that enter attention and feed-forward through
``ColumnParallelRegion``/``SequenceParallelRegion``. A region whose entry does
not reach every consumer of an input leaves the loss bit-identical and only the
upstream gradients short, so every parameter's gradient is compared -- with
LoRA too, where the adapter arm's gradient has to come out partial and be
reduced like the base arm's.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import pytest
import torch
import torch.distributed as dist
from conftest import requires_gloo, run_gloo_ranks
from tp_qualification import assert_optimizer_step_parity

WORLD_SIZE = 2


def _randn(generator: torch.Generator, *shape: int) -> torch.Tensor:
    return torch.randn(*shape, generator=generator, dtype=torch.float64)


def _perturbed(model: torch.nn.Module) -> torch.nn.Module:
    # Zero-initialized output layers would leave half the gradients zero.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0, 0.2)
    return model.double()


def _sd3_5() -> Tuple[torch.nn.Module, Callable[[torch.nn.Module], Any], type]:
    from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel

    from flow_factory.models.stable_diffusion.sd3_5 import SD3_5Adapter

    torch.manual_seed(0)
    model = _perturbed(
        SD3Transformer2DModel(
            sample_size=8,
            patch_size=2,
            in_channels=4,
            num_layers=2,
            attention_head_dim=8,
            num_attention_heads=4,
            joint_attention_dim=16,
            caption_projection_dim=32,
            pooled_projection_dim=8,
            out_channels=4,
            pos_embed_max_size=8,
            dual_attention_layers=(0,),
            qk_norm="rms_norm",
        )
    )

    def forward(module: torch.nn.Module):
        generator = torch.Generator().manual_seed(11)
        return module(
            hidden_states=_randn(generator, 1, 4, 8, 8),
            encoder_hidden_states=_randn(generator, 1, 5, 16),
            pooled_projections=_randn(generator, 1, 8),
            timestep=torch.tensor([500.0], dtype=torch.float64),
            return_dict=False,
        )[0]

    return model, forward, SD3_5Adapter


def _qwen_image_21() -> Tuple[torch.nn.Module, Callable[[torch.nn.Module], Any], type]:
    from diffusers.models.transformers.transformer_qwenimage21 import (
        QwenImage21Transformer2DModel,
    )

    from flow_factory.models.qwen_image_21.qwen_image_21 import QwenImage21Adapter

    torch.manual_seed(0)
    model = _perturbed(
        QwenImage21Transformer2DModel(
            patch_size=1,
            in_channels=8,
            out_channels=8,
            num_layers=2,
            attention_head_dim=16,
            num_attention_heads=4,
            context_in_dim=16,
            axes_dims_rope=(4, 6, 6),
        )
    )
    text, image = 4, 16

    def forward(module: torch.nn.Module):
        generator = torch.Generator().manual_seed(11)
        # Text tokens followed by image slots, each slot standing for 4 latents.
        img_mask = torch.zeros(1, text + image // 4, dtype=torch.bool)
        img_mask[:, text:] = True
        return module(
            hidden_states=_randn(generator, 1, image, 8),
            encoder_hidden_states=_randn(generator, 1, text, 16),
            timestep=torch.tensor([500.0], dtype=torch.float64),
            img_shapes=[[(1, 4, 4)]],
            img_mask=img_mask,
            return_dict=False,
        )[0]

    return model, forward, QwenImage21Adapter


def _z_image() -> Tuple[torch.nn.Module, Callable[[torch.nn.Module], Any], type]:
    from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel

    from flow_factory.models.z_image.z_image import ZImageAdapter

    torch.manual_seed(0)
    model = _perturbed(
        ZImageTransformer2DModel(
            all_patch_size=(2,),
            all_f_patch_size=(1,),
            in_channels=4,
            dim=64,
            n_layers=2,
            n_refiner_layers=1,
            n_heads=2,
            n_kv_heads=2,
            qk_norm=True,
            cap_feat_dim=32,
            axes_dims=[8, 12, 12],
            axes_lens=[64, 32, 32],
        )
    )

    def forward(module: torch.nn.Module):
        generator = torch.Generator().manual_seed(11)
        latents = [_randn(generator, 4, 1, 8, 8) for _ in range(2)]
        captions = [_randn(generator, 6, 32) for _ in range(2)]
        timesteps = torch.full((2,), 0.37, dtype=torch.float64)
        outputs = module(
            latents, timesteps, captions, return_dict=False, patch_size=2, f_patch_size=1
        )[0]
        return torch.stack(list(outputs))

    return model, forward, ZImageAdapter


def _grid(tokens: int, axes: int, offset: float = 0.0) -> torch.Tensor:
    side = int(tokens**0.5)
    ids = torch.zeros(tokens, axes, dtype=torch.float64)
    ids[:, axes - 2] = (torch.arange(tokens) // side).double() + offset
    ids[:, axes - 1] = (torch.arange(tokens) % side).double() + offset
    return ids


def _flux1() -> Tuple[torch.nn.Module, Callable[[torch.nn.Module], Any], type]:
    from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel

    from flow_factory.models.flux.flux1 import Flux1Adapter

    torch.manual_seed(0)
    model = _perturbed(
        FluxTransformer2DModel(
            in_channels=16,
            num_layers=1,
            num_single_layers=1,
            attention_head_dim=8,
            num_attention_heads=2,
            joint_attention_dim=16,
            pooled_projection_dim=8,
            guidance_embeds=True,
            axes_dims_rope=(2, 2, 4),
        )
    )

    def forward(module: torch.nn.Module):
        generator = torch.Generator().manual_seed(11)
        return module(
            hidden_states=_randn(generator, 1, 16, 16),
            encoder_hidden_states=_randn(generator, 1, 4, 16),
            pooled_projections=_randn(generator, 1, 8),
            timestep=torch.tensor([0.5], dtype=torch.float64),
            img_ids=_grid(16, 3),
            txt_ids=torch.zeros(4, 3, dtype=torch.float64),
            guidance=torch.tensor([3.5], dtype=torch.float64),
            return_dict=False,
        )[0]

    return model, forward, Flux1Adapter


def _flux3() -> Tuple[torch.nn.Module, Callable[[torch.nn.Module], Any], type]:
    import diffusers
    from flow_factory.models.flux.flux3 import Flux3Adapter

    torch.manual_seed(0)
    model = _perturbed(
        diffusers.Flux3Transformer2DModel(
            in_channels=16,
            cond_in_channels=16,
            num_layers=1,
            num_single_layers=1,
            attention_head_dim=8,
            num_attention_heads=2,
            joint_attention_dim=16,
            mlp_ratio=3.0,
            flow_map=True,
            axes_dims_rope=(2, 2, 2, 2),
        )
    )

    def forward(module: torch.nn.Module):
        generator = torch.Generator().manual_seed(11)
        return module(
            hidden_states=_randn(generator, 1, 16, 16),
            encoder_hidden_states=_randn(generator, 1, 4, 16),
            timestep=torch.tensor([0.5], dtype=torch.float64),
            img_ids=_grid(16, 4),
            txt_ids=torch.zeros(4, 4, dtype=torch.float64),
            image_cond_hidden_states=_randn(generator, 1, 4, 16),
            image_cond_ids=_grid(4, 4, offset=100.0),
            target_timestep=torch.tensor([0.4], dtype=torch.float64),
            return_dict=False,
        )[0]

    return model, forward, Flux3Adapter


class _BagelLayers(torch.nn.Module):
    """Bagel's two feed-forward branches under the paths its plan names."""

    def __init__(self) -> None:
        super().__init__()
        from flow_factory.models.bagel.modeling.qwen2.configuration_qwen2 import Qwen2Config
        from flow_factory.models.bagel.modeling.qwen2.modeling_qwen2 import Qwen2MLP

        config = Qwen2Config(hidden_size=16, intermediate_size=32, hidden_act="silu")
        layer = torch.nn.Module()
        layer.mlp = Qwen2MLP(config)
        layer.mlp_moe_gen = Qwen2MLP(config)
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([layer])

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        layer = self.model.layers[0]
        return layer.mlp(tokens) + layer.mlp_moe_gen(hidden_state=tokens.flip(1))


def _bagel() -> Tuple[torch.nn.Module, Callable[[torch.nn.Module], Any], type]:
    from flow_factory.models.bagel.bagel import BagelAdapter

    torch.manual_seed(0)
    model = _perturbed(_BagelLayers())

    def forward(module: torch.nn.Module):
        generator = torch.Generator().manual_seed(11)
        return module(_randn(generator, 1, 8, 16))

    return model, forward, BagelAdapter


_ARCHITECTURES: Dict[str, Callable[[], Tuple[torch.nn.Module, Callable, type]]] = {
    "sd3_5": _sd3_5,
    "qwen_image_21": _qwen_image_21,
    "z_image": _z_image,
    "flux1": _flux1,
    "flux3": _flux3,
    "bagel": _bagel,
}


def _with_lora(model: torch.nn.Module, adapter: type) -> torch.nn.Module:
    from peft import LoraConfig, get_peft_model

    targets = adapter.default_target_modules.fget(object.__new__(adapter))
    torch.manual_seed(1)
    # Random `lora_B` as well as `lora_A`, so no `lora_A` gradient is zero.
    config = LoraConfig(r=4, lora_alpha=4, target_modules=targets, init_lora_weights=False)
    return get_peft_model(model, config).double()


def _gradients(
    module: torch.nn.Module, forward: Callable, counts: Dict[str, int]
) -> Dict[str, torch.Tensor]:
    """Backpropagate a sum of squares, recording the backward's collectives in ``counts``."""
    from torch.distributed.tensor import DTensor
    from torch.distributed.tensor.debug import CommDebugMode

    output = forward(module)
    with CommDebugMode() as mode:
        output.pow(2).sum().backward()
    counts.update({str(op).split(".")[-1]: n for op, n in mode.get_comm_counts().items() if n})
    values = {"output": output.detach()}
    for name, parameter in module.named_parameters():
        if parameter.grad is None:
            continue
        grad = parameter.grad
        values[name] = (grad.full_tensor() if isinstance(grad, DTensor) else grad).detach()
    return values


def _parity_worker(rank: int, store_path: str, result_dir: str, options: Dict[str, Any]) -> None:
    store = dist.FileStore(store_path, WORLD_SIZE)
    dist.init_process_group("gloo", store=store, rank=rank, world_size=WORLD_SIZE)
    result: Dict[str, Any] = {"rank": rank}
    try:
        from torch.distributed.device_mesh import init_device_mesh

        from flow_factory.models.tensor_parallel import (
            apply_tensor_parallel,
            peel_peft_wrapper,
            resolve_parallel_plan,
        )

        mesh = init_device_mesh("cpu", (WORLD_SIZE,), mesh_dim_names=("tp",))
        for sequence_parallel, lora in options["cases"]:
            reference, forward, adapter = _ARCHITECTURES[options["architecture"]]()
            if lora:
                reference = _with_lora(reference, adapter)
            sharded = copy.deepcopy(reference)
            apply_tensor_parallel(
                peel_peft_wrapper(sharded),
                mesh,
                resolve_parallel_plan(adapter, sequence_parallel=sequence_parallel),
            )
            counts: Dict[str, int] = {}
            expected = _gradients(reference, forward, {})
            received = _gradients(sharded, forward, counts)
            update = assert_optimizer_step_parity(reference, sharded, rtol=1e-8, atol=1e-10)
            missing = sorted(set(expected) ^ set(received))
            relative = {
                name: ((received[name] - expected[name]).abs().max() / expected[name].abs().max())
                .clamp_max(1e3)
                .item()
                for name in expected
                if name in received and expected[name].abs().max() > 0
            }
            worst = max(relative, key=relative.get)
            result[f"sp={sequence_parallel}/lora={lora}"] = {
                "max_relative_diff": relative[worst],
                "worst": worst,
                "gradients": len(expected) - 1,
                "missing": missing,
                "counts": counts,
                "update": update,
            }
    except Exception as error:  # reported to the parent, which asserts on it
        result["error"] = f"{type(error).__name__}: {error}"
    Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result), encoding="utf-8")
    dist.destroy_process_group()


def _flux3_available() -> bool:
    import diffusers

    return hasattr(diffusers, "Flux3Transformer2DModel")


def _flash_attn_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("flash_attn") is not None


# (architecture, [(sequence_parallel, lora)], tolerance). Built directly rather
# than through `from_pretrained`, these models stay in float64 throughout, so
# the tolerance is accumulation order; a missing or doubled reduction is off by
# order one.
_CASES: List[Tuple[str, List[Tuple[bool, bool]], float]] = [
    ("sd3_5", [(False, False), (True, False), (False, True), (True, True)], 1e-10),
    ("qwen_image_21", [(False, False), (False, True)], 1e-10),
    ("z_image", [(False, False), (True, False)], 1e-10),
    ("flux1", [(False, False), (False, True)], 1e-10),
    ("flux3", [(False, False), (False, True)], 1e-10),
    ("bagel", [(False, False)], 1e-10),
]


#: SD3.5's backward collectives over its two blocks -- the first with dual
#: attention and a context feed-forward, the second ``context_pre_only``. Under
#: tensor parallelism each input stream that needs a gradient costs one
#: all-reduce: 5 + 3, where one per projection was 11 + 7. Under sequence
#: parallelism the five gathered image streams reduce-scatter, the three text
#: streams and the three modulation outputs all-reduce, and the five
#: out-projections and the patch scatter all-gather. Full training also gathers
#: the input derivatives of five row-sharded AdaLN linears (two per block plus
#: the final norm). Default LoRA freezes those linears and their conditioning
#: embeddings, so they do not contribute backward gathers.
_EXPECTED_COUNTS = {
    ("sd3_5", False, False): {"all_reduce": 8, "all_gather_into_tensor": 5},
    ("sd3_5", True, False): {
        "reduce_scatter_tensor": 5,
        "all_reduce": 6,
        "all_gather_into_tensor": 11,
    },
    ("sd3_5", False, True): {"all_reduce": 5},
    ("sd3_5", True, True): {
        "reduce_scatter_tensor": 3,
        "all_reduce": 2,
        "all_gather_into_tensor": 6,
    },
}


@requires_gloo
@pytest.mark.parametrize("architecture, cases, tolerance", _CASES, ids=[case[0] for case in _CASES])
def test_the_region_entry_plans_match_the_unsharded_models(
    tmp_path: Path, architecture: str, cases: List[Tuple[bool, bool]], tolerance: float
) -> None:
    if architecture == "flux3" and not _flux3_available():
        pytest.skip("this diffusers has no Flux3Transformer2DModel")
    if architecture == "bagel" and not _flash_attn_available():
        pytest.skip("Bagel's vendored attention imports flash_attn at module scope")
    results = run_gloo_ranks(
        _parity_worker, tmp_path, WORLD_SIZE, {"architecture": architecture, "cases": cases}
    )
    for result in results:
        assert "error" not in result, result["error"]
        for sequence_parallel, lora in cases:
            entry = result[f"sp={sequence_parallel}/lora={lora}"]
            assert not entry["missing"], entry
            assert entry["max_relative_diff"] < tolerance, (architecture, entry)
            expected = _EXPECTED_COUNTS.get((architecture, sequence_parallel, lora))
            if expected is not None:
                assert entry["counts"] == expected, (architecture, sequence_parallel, lora)
