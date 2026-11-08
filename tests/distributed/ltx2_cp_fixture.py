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

"""Native LTX fixtures retain 32 attention heads and both 128-channel streams."""

from pathlib import Path

import torch


def build_transformer(ltx23=False):
    from diffusers.models.transformers.transformer_ltx2 import LTX2VideoTransformer3DModel

    torch.manual_seed(29)
    return LTX2VideoTransformer3DModel(
        in_channels=128,
        out_channels=128,
        num_attention_heads=32,
        attention_head_dim=16,
        cross_attention_dim=512,
        audio_in_channels=128,
        audio_out_channels=128,
        audio_num_attention_heads=32,
        audio_attention_head_dim=8,
        audio_cross_attention_dim=256,
        num_layers=2,
        caption_channels=60,
        rope_type="split",
        gated_attn=ltx23,
        audio_gated_attn=ltx23,
        cross_attn_mod=ltx23,
        audio_cross_attn_mod=ltx23,
        perturbed_attn=ltx23,
        use_prompt_embeddings=not ltx23,
    )


def build_pipeline(root: Path, *, ltx23: bool, image_conditioned: bool):
    from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors
    from diffusers.pipelines.ltx2.vocoder import LTX2Vocoder
    from transformers import (
        Gemma3Config,
        Gemma3ForConditionalGeneration,
        Gemma3TextConfig,
        GemmaTokenizer,
        SiglipVisionConfig,
    )

    from diffusers import (
        AutoencoderKLLTX2Audio,
        AutoencoderKLLTX2Video,
        FlowMatchEulerDiscreteScheduler,
        LTX2ImageToVideoPipeline,
        LTX2Pipeline,
    )

    root.mkdir(parents=True, exist_ok=True)
    # Native Transformers5 Gemma uses its BPE vocab argument, not a SentencePiece path.
    tokens = [
        "<pad>",
        "<eos>",
        "<bos>",
        "<unk>",
        "<mask>",
        "▁",
        *"abcdefghijklmnopqrstuvwxyz0123456789",
    ]
    tokenizer = GemmaTokenizer(
        vocab={token: index for index, token in enumerate(tokens)}, merges=[]
    )
    text_config = Gemma3TextConfig(
        vocab_size=len(tokenizer),
        hidden_size=60,
        intermediate_size=120,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        sliding_window=64,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        layer_types=["sliding_attention", "full_attention"],
    )
    vision_config = SiglipVisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        image_size=16,
        patch_size=8,
    )
    text_encoder = Gemma3ForConditionalGeneration(
        Gemma3Config(
            text_config=text_config.to_dict(),
            vision_config=vision_config.to_dict(),
            mm_tokens_per_image=4,
        )
    )
    connectors = LTX2TextConnectors(
        caption_channels=60,
        text_proj_in_factor=3,
        video_connector_num_attention_heads=32 if ltx23 else 30,
        video_connector_attention_head_dim=16 if ltx23 else 2,
        audio_connector_num_attention_heads=32 if ltx23 else 30,
        audio_connector_attention_head_dim=8 if ltx23 else 2,
        video_connector_num_layers=2,
        audio_connector_num_layers=2,
        video_connector_num_learnable_registers=128,
        audio_connector_num_learnable_registers=128,
        video_gated_attn=ltx23,
        audio_gated_attn=ltx23,
        rope_type="split",
        per_modality_projections=ltx23,
        video_hidden_dim=512,
        audio_hidden_dim=256,
        proj_bias=ltx23,
    )
    vae = AutoencoderKLLTX2Video(
        latent_channels=128,
        block_out_channels=(16, 32, 32, 32),
        decoder_block_out_channels=(16, 32, 64),
        layers_per_block=(1, 1, 1, 1, 1),
        decoder_layers_per_block=(1, 1, 1, 1),
        decoder_causal=False,
        patch_size=4,
        spatial_compression_ratio=32,
        temporal_compression_ratio=8,
    )
    audio_vae = AutoencoderKLLTX2Audio(
        # Native audio statistics have base_channels entries and normalize packed
        # 8-channel x 16-mel latents, so this structural width must stay 128.
        base_channels=128,
        ch_mult=(1, 2, 4),
        num_res_blocks=1,
        latent_channels=8,
        mel_bins=64,
        sample_rate=16000,
        mel_hop_length=160,
    )
    vocoder = LTX2Vocoder(
        in_channels=128,
        hidden_channels=32,
        out_channels=2,
        resnet_kernel_sizes=[3],
        resnet_dilations=[[1, 3, 5]],
        output_sampling_rate=24000,
    )
    with torch.no_grad():
        for model in (vae, audio_vae):
            model.latents_mean.zero_()
            model.latents_std.fill_(1)
    cls = LTX2ImageToVideoPipeline if image_conditioned else LTX2Pipeline
    pipeline = cls(
        scheduler=FlowMatchEulerDiscreteScheduler(),
        vae=vae,
        audio_vae=audio_vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        connectors=connectors,
        transformer=build_transformer(ltx23),
        vocoder=vocoder,
    )
    return pipeline


def model_inputs(ltx23, per_token, replica=0, step=0, device="cpu"):
    generator = torch.Generator().manual_seed(31 + 17 * replica + step)
    mask = torch.tensor([[1, 1, 0, 1, 0, 0]])
    timestep = torch.tensor([[0.0] * 4 + [500.0] * 8]) if per_token else torch.tensor([500.0])
    inputs = dict(
        hidden_states=torch.randn(1, 12, 128, generator=generator),
        audio_hidden_states=torch.randn(1, 8, 128, generator=generator),
        encoder_hidden_states=torch.randn(1, 6, 512 if ltx23 else 60, generator=generator),
        audio_encoder_hidden_states=torch.randn(1, 6, 256 if ltx23 else 60, generator=generator),
        timestep=timestep,
        audio_timestep=torch.tensor([500.0]),
        sigma=torch.tensor([500.0]),
        audio_sigma=torch.tensor([500.0]),
        encoder_attention_mask=mask,
        audio_encoder_attention_mask=mask,
        num_frames=3,
        height=2,
        width=2,
        audio_num_frames=8,
        fps=24.0,
        use_cross_timestep=ltx23,
        return_dict=False,
    )

    return {
        name: value.to(device) if torch.is_tensor(value) else value
        for name, value in inputs.items()
    }


def adapter_class(per_token):
    from flow_factory.models.ltx2.ltx2_i2av import LTX2_I2AV_Adapter
    from flow_factory.models.ltx2.ltx2_t2av import LTX2_T2AV_Adapter

    return LTX2_I2AV_Adapter if per_token else LTX2_T2AV_Adapter


def install_lora(module, per_token):
    from peft import LoraConfig, get_peft_model

    cls = adapter_class(per_token)
    targets = cls.default_target_modules.fget(object.__new__(cls))
    return get_peft_model(
        module, LoraConfig(r=2, lora_alpha=2, target_modules=targets, init_lora_weights=False)
    )
