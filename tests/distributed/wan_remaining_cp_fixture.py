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

"""Native remaining Wan layouts with published heads, channels and context lengths.

Layers, head width, FFN and text/CLIP widths are reduced. The 24/40 head geometry,
48/48 or 36/16 latent channels, 512 text tokens, 257/514 image positions and
(1, 2, 2) patches are retained. Twelve generated tokens put the first-frame
boundary inside a CP shard and a TP/SP shard rather than at their boundaries.
"""

from types import SimpleNamespace

import torch

from diffusers import WanTransformer3DModel
from flow_factory.models.wan.wan2_i2v import Wan2_I2V_Adapter
from flow_factory.models.wan.wan2_t2v import Wan2_T2V_Adapter

VARIANTS = ("t2v_dual", "t2v_expanded", "i2v_dual", "i2v_expanded", "flf2v")


def adapter_class(variant):
    return Wan2_T2V_Adapter if variant.startswith("t2v") else Wan2_I2V_Adapter


def pipeline_config(variant):
    return SimpleNamespace(
        expand_timesteps=variant.endswith("expanded"),
        boundary_ratio=(
            (0.9 if variant == "i2v_dual" else 0.875) if variant.endswith("dual") else None
        ),
    )


def build_transformer(variant, seed=29):
    torch.manual_seed(seed)
    expanded = variant.endswith("expanded")
    channels = 48 if expanded else 16
    heads = 24 if expanded else 40
    model = WanTransformer3DModel(
        patch_size=(1, 2, 2),
        num_attention_heads=heads,
        attention_head_dim=8,
        in_channels=36 if variant in ("i2v_dual", "flf2v") else channels,
        out_channels=channels,
        text_dim=16,
        image_dim=16 if variant == "flf2v" else None,
        added_kv_proj_dim=heads * 8 if variant == "flf2v" else None,
        pos_embed_seq_len=514 if variant == "flf2v" else None,
        freq_dim=16,
        ffn_dim=32,
        num_layers=2,
        rope_max_seq_len=32,
    )

    if variant == "flf2v":
        with torch.no_grad():
            model.condition_embedder.image_embedder.pos_embed.normal_(0, 0.2)
    return model


def install_lora(model, variant):
    from peft import LoraConfig, get_peft_model

    cls = adapter_class(variant)
    targets = cls.default_target_modules.fget(object.__new__(cls))
    torch.manual_seed(31)
    return get_peft_model(
        model, LoraConfig(r=2, lora_alpha=2, target_modules=targets, init_lora_weights=False)
    )


def forward_inputs(variant, replica=0, step=0, device="cpu", timestep=600.0):
    generator = torch.Generator().manual_seed(113 + replica * 17 + step)
    expanded = variant.endswith("expanded")
    channels = 48 if expanded else 16
    args = {
        "latents": torch.randn(1, channels, 3, 4, 4, generator=generator).to(device),
        "t": torch.tensor([timestep], device=device),
        "prompt_embeds": torch.randn(1, 512, 16, generator=generator).to(device),
        "guidance_scale": 2.0,
        "guidance_scale_2": 3.0,
        "compute_log_prob": False,
        "return_kwargs": ("velocity",),
    }
    args["negative_prompt_embeds"] = -args["prompt_embeds"]
    if not variant.startswith("t2v"):
        shape = (1, channels, 1, 4, 4) if expanded else (1, 20, 3, 4, 4)
        args["latent_condition"] = torch.randn(shape, generator=generator).to(device)
        args["image_embeds"] = (
            torch.randn(2, 257, 16, generator=generator).to(device) if variant == "flf2v" else None
        )
        if expanded:
            mask = torch.ones(1, 1, 3, 4, 4, device=device)
            mask[:, :, 0] = 0
            args["first_frame_mask"] = mask
    return args


def native_inputs(variant, args):
    """Independent construction of the native operands, outside adapter forward."""
    latents = args["latents"]
    timestep = args["t"]
    if variant == "i2v_expanded":
        mask = args["first_frame_mask"]
        hidden = torch.where(mask.bool(), latents, args["latent_condition"].expand_as(latents))
        time = mask[:, 0, :, ::2, ::2].reshape(1, -1) * timestep[:, None]
    elif variant == "t2v_expanded":
        hidden = latents
        time = timestep[:, None].expand(1, 12)
    elif variant in ("i2v_dual", "flf2v"):
        hidden = torch.cat((latents, args["latent_condition"]), dim=1)
        time = timestep
    else:
        hidden, time = latents, timestep
    result = {
        "hidden_states": hidden,
        "timestep": time,
        "encoder_hidden_states": args["prompt_embeds"],
        "return_dict": False,
    }
    if variant == "flf2v":
        result["encoder_hidden_states_image"] = args["image_embeds"]
    return result


def native_velocity(module, variant, args):
    values = native_inputs(variant, args)
    cond = module(**values)[0]
    uncond = module(**{**values, "encoder_hidden_states": args["negative_prompt_embeds"]})[0]
    scale = (
        args["guidance_scale_2"]
        if variant.endswith("dual")
        and args["t"].item() < pipeline_config(variant).boundary_ratio * 1000
        else args["guidance_scale"]
    )
    return uncond + scale * (cond - uncond)


def build_fixture(root, variant):
    """Save native text/CLIP/VAE components and distinct ordered reference images."""
    from pathlib import Path

    import numpy as np
    from PIL import Image
    from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel
    from wan_sft_fixture import build_wan_sft_fixture

    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline, WanPipeline

    root = Path(root)
    build_wan_sft_fixture(root, 4)
    base = WanPipeline.from_pretrained(root / "model", torch_dtype=torch.float32)
    output_channels = 48 if variant.endswith("expanded") else 16
    torch.manual_seed(37)
    components = {
        "tokenizer": base.tokenizer,
        "text_encoder": base.text_encoder,
        "vae": AutoencoderKLWan(
            base_dim=4,
            z_dim=output_channels,
            dim_mult=[1, 1, 1, 1],
            num_res_blocks=1,
            temperal_downsample=[False, True, True],
            latents_mean=[0.0] * output_channels,
            latents_std=[1.0] * output_channels,
            is_residual=variant.endswith("expanded"),
            in_channels=12 if variant.endswith("expanded") else 3,
            out_channels=12 if variant.endswith("expanded") else 3,
            patch_size=2 if variant.endswith("expanded") else None,
            scale_factor_spatial=16 if variant.endswith("expanded") else 8,
        ),
        "transformer": build_transformer(variant),
        "transformer_2": build_transformer(variant, seed=71) if variant.endswith("dual") else None,
        "scheduler": base.scheduler,
        **vars(pipeline_config(variant)),
    }
    if variant.startswith("t2v"):
        pipeline = WanPipeline(**components)
    else:
        clip = variant == "flf2v"
        pipeline = WanImageToVideoPipeline(
            **components,
            image_encoder=(
                CLIPVisionModel(
                    CLIPVisionConfig(
                        hidden_size=16,
                        intermediate_size=32,
                        num_hidden_layers=2,
                        num_attention_heads=2,
                        image_size=32,
                        patch_size=2,
                    )
                )
                if clip
                else None
            ),
            image_processor=(
                CLIPImageProcessor(
                    size={"shortest_edge": 32}, crop_size={"height": 32, "width": 32}
                )
                if clip
                else None
            ),
        )
    pipeline.save_pretrained(root / "model")
    for index in range(16):
        for slot in ("first", "last"):
            rng = np.random.default_rng(100 + 2 * index + int(slot == "last"))
            pixels = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(root / "data" / f"{index}-{slot}.png")


if __name__ == "__main__":
    import sys

    build_fixture(sys.argv[1], sys.argv[2])
