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

"""Native Kontext components and real unequal-length reference-image operands."""

import sys
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import torch
from flux_sft_fixture import build_flux_sft_fixture, flux_transformer
from PIL import Image

from diffusers import FluxKontextPipeline, FluxPipeline
from flow_factory.models.flux.flux1_kontext import Flux1KontextAdapter
from flow_factory.models.runtime.classic import ClassicPipelineRuntime
from flow_factory.scheduler import FlowMatchEulerDiscreteSDEScheduler


def make_adapter(pipeline):
    adapter = object.__new__(Flux1KontextAdapter)
    adapter.pipeline = pipeline
    adapter.component_runtime = ClassicPipelineRuntime(pipeline)
    adapter._components = adapter.component_runtime.override_components
    adapter.accelerator = SimpleNamespace(device=torch.device("cpu"))
    adapter._has_warned_multi_image = False
    return adapter


def build_fixture(root):
    root = Path(root)
    build_flux_sft_fixture(root)
    base = FluxPipeline.from_pretrained(root / "model", torch_dtype=torch.float32)
    pipeline = FluxKontextPipeline(**base.components)
    pipeline.save_pretrained(root / "model")


def install_lora(module):
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(1)
    targets = Flux1KontextAdapter.default_target_modules.fget(object.__new__(Flux1KontextAdapter))
    return get_peft_model(
        module, LoraConfig(r=2, lora_alpha=2, target_modules=targets, init_lora_weights=False)
    )


@lru_cache(maxsize=4)
def encoded_conditions(root, replica):
    """Encode native prompt/VAE conditions once per logical DP sample in each worker."""
    pipeline = FluxKontextPipeline.from_pretrained(Path(root) / "model", torch_dtype=torch.float32)
    adapter = make_adapter(pipeline)
    with torch.no_grad():
        prompt = adapter.encode_prompt("a red image" if replica else "a blue image")
        image = adapter.encode_image(
            Image.open(Path(root) / "data" / f"{replica}.png"), condition_image_size=(32, 64)
        )
    return prompt, image


def model_inputs(root, replica, step, device):
    prompt, image = encoded_conditions(str(root), replica)
    generator = torch.Generator().manual_seed(100 + 10 * replica + step)
    ids = FluxPipeline._prepare_latent_image_ids(1, 4, 4, "cpu", torch.float32)
    return {
        "latents": torch.randn(1, 16, 16, generator=generator).to(device),
        "image_latents": image["image_latents"].to(device),
        "latent_ids": torch.cat([ids, image["image_ids"][0]], dim=0).to(device),
        "prompt_embeds": prompt["prompt_embeds"].to(device),
        "pooled_prompt_embeds": prompt["pooled_prompt_embeds"].to(device),
        "t": torch.tensor([600.0 - 100 * step], device=device),
        "t_next": torch.tensor([500.0 - 100 * step], device=device),
    }


def reference_forward(model, inputs):
    """Independent native concatenation/position conditioning and target-only projection."""
    target = inputs["latents"]
    text = inputs["prompt_embeds"]
    return model(
        hidden_states=torch.cat([target, inputs["image_latents"]], dim=1),
        encoder_hidden_states=text,
        pooled_projections=inputs["pooled_prompt_embeds"],
        timestep=inputs["t"] / 1000,
        guidance=torch.full((1,), 3.5, device=target.device, dtype=target.dtype),
        txt_ids=torch.zeros(text.shape[1], 3, device=target.device, dtype=target.dtype),
        img_ids=inputs["latent_ids"],
        return_dict=False,
    )[0][:, : target.shape[1]]


def scheduler():
    return FlowMatchEulerDiscreteSDEScheduler(dynamics_type="ODE")


if __name__ == "__main__":
    build_fixture(Path(sys.argv[1]))
