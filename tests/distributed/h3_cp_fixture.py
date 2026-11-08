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

"""Native H3 geometry and distinct packed condition/target row operands."""

import torch

from diffusers import MiniMaxH3Transformer3DModel


def build_transformer():
    torch.manual_seed(59)
    return MiniMaxH3Transformer3DModel(
        num_attention_heads=56,
        attention_head_dim=128,
        hidden_size=64,
        num_layers=2,
        num_refiner_layers=2,
        ffn_dim=128,
        in_channels=24,
        audio_in_channels=32,
        patch_size=(1, 2, 2),
        text_dim=64,
        freq_dim=16,
        time_embed_hidden_dim=64,
        time_embed_dim=32,
        rope_freq_dim=16,
    )


def model_inputs(workflow="t2va", replica=0, step=0, odd=False):
    condition_video, condition_audio = {"t2va": (0, 0), "fl2va": (4, 0), "ref2va": (6, 2)}[workflow]
    video_rows, audio_rows = 12 + condition_video, 8 + condition_audio
    text_rows = 5 if odd else 4
    total = video_rows + audio_rows + text_rows
    text_indices = torch.arange(text_rows)
    # Native packing keeps conditions before target audio, then target video.
    prefix_end = text_rows + condition_video + condition_audio
    video_indices = torch.cat(
        (
            torch.arange(text_rows, text_rows + condition_video),
            torch.arange(prefix_end + 8, total),
        )
    )
    audio_indices = torch.cat(
        (
            torch.arange(text_rows + condition_video, prefix_end),
            torch.arange(prefix_end, prefix_end + 8),
        )
    )
    tags = torch.ones(total, dtype=torch.long)
    if workflow != "t2va":
        tags[text_indices[:2]] = 0
    tags[video_indices] = 0
    tags[audio_indices] = 2
    times = torch.tensor([0.35, 0.65, 0.999, 1.0])
    indices = torch.ones(total, dtype=torch.long)
    indices[video_indices] = 0
    indices[audio_indices] = 1
    indices[video_indices[:condition_video]] = 2
    indices[audio_indices[:condition_audio]] = 3
    positions = torch.stack(
        (torch.arange(total).float() / 4, torch.arange(total) % 2, torch.arange(total) % 3), dim=1
    )
    generator = torch.Generator().manual_seed(61 + replica * 13 + step)
    return dict(
        hidden_states=torch.randn(1, video_rows, 96, generator=generator),
        audio_hidden_states=torch.randn(1, audio_rows, 32, generator=generator),
        encoder_hidden_states=torch.randn(1, text_rows, 64, generator=generator),
        timestep=times,
        timestep_indices=indices,
        token_tags=tags,
        position_ids=positions,
        video_indices=video_indices,
        audio_indices=audio_indices,
        text_indices=text_indices,
        return_dict=False,
    )


def adapter_class(workflow):
    from flow_factory.models.minimax_h3.adapters import (
        MiniMaxH3FL2VAAdapter,
        MiniMaxH3Ref2VAAdapter,
        MiniMaxH3T2VAAdapter,
    )

    return {
        "t2va": MiniMaxH3T2VAAdapter,
        "fl2va": MiniMaxH3FL2VAAdapter,
        "ref2va": MiniMaxH3Ref2VAAdapter,
    }[workflow]


def install_lora(module):
    from peft import LoraConfig, get_peft_model

    return get_peft_model(
        module,
        LoraConfig(
            r=2,
            lora_alpha=2,
            target_modules=["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"],
            init_lora_weights=False,
        ),
    )


def build_conditioner():
    """Native VL hidden_states[50], with reduced width and real image/video processors."""
    from tokenizers.pre_tokenizers import ByteLevel
    from transformers import (
        Qwen2TokenizerFast,
        Qwen2VLImageProcessorFast,
        Qwen3VLConfig,
        Qwen3VLForConditionalGeneration,
        Qwen3VLProcessor,
        Qwen3VLVideoProcessor,
    )

    specials = [
        "<pad>",
        "<unk>",
        "<|im_start|>",
        "<|im_end|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|image_pad|>",
        "<|video_pad|>",
    ]
    vocab = {word: index for index, word in enumerate(specials + sorted(ByteLevel.alphabet()))}
    tokenizer = Qwen2TokenizerFast(
        vocab=vocab,
        merges=[],
        pad_token="<pad>",
        unk_token="<unk>",
        eos_token="<|im_end|>",
        additional_special_tokens=specials[2:],
    )
    template = "{% for message in messages %}<|im_start|>{{ message['role'] }}\n{% for item in message['content'] %}{{ item['text'] }}{% endfor %}<|im_end|>\n{% endfor %}"
    image_processor = Qwen2VLImageProcessorFast(
        patch_size=16,
        temporal_patch_size=2,
        merge_size=2,
        size={"shortest_edge": 1024, "longest_edge": 2048},
        image_mean=[0.5] * 3,
        image_std=[0.5] * 3,
    )
    processor = Qwen3VLProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        video_processor=Qwen3VLVideoProcessor(
            patch_size=16,
            temporal_patch_size=2,
            merge_size=2,
            size={"shortest_edge": 1024, "longest_edge": 2048},
            image_mean=[0.5] * 3,
            image_std=[0.5] * 3,
        ),
        chat_template=template,
    )
    text_encoder = Qwen3VLForConditionalGeneration(
        Qwen3VLConfig(
            image_token_id=vocab["<|image_pad|>"],
            video_token_id=vocab["<|video_pad|>"],
            vision_start_token_id=vocab["<|vision_start|>"],
            vision_end_token_id=vocab["<|vision_end|>"],
            text_config={
                "vocab_size": len(vocab),
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 51,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 16,
                "max_position_embeddings": 1024,
                "pad_token_id": 0,
                "bos_token_id": 2,
                "eos_token_id": 3,
                "rope_parameters": {
                    "rope_type": "default",
                    "rope_theta": 10000.0,
                    "mrope_section": [2, 3, 3],
                    "mrope_interleaved": True,
                },
            },
            vision_config={
                "depth": 1,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_heads": 4,
                "out_hidden_size": 64,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2,
                "num_position_embeddings": 16,
                "deepstack_visual_indexes": [],
            },
        )
    )
    return text_encoder, tokenizer, processor


def build_pipeline(workflow):
    from diffusers import (
        AutoencoderKLMiniMaxH3,
        AutoencoderKLMiniMaxH3Audio,
        MiniMaxH3ModularPipeline,
        MiniMaxH3Scheduler,
    )

    text_encoder, tokenizer, processor = build_conditioner()
    vae = AutoencoderKLMiniMaxH3(
        block_out_channels=(8, 16, 16, 32, 32, 32),
        layers_per_block=1,
        norm_num_groups=8,
        decoder_num_layers=2,
        decoder_num_attention_heads=32,
        decoder_attention_head_dim=16,
        latent_channels=24,
    )
    audio_vae = AutoencoderKLMiniMaxH3Audio(
        encoder_dim=4,
        latent_dim=64,
        latent_channels=32,
        decoder_dim=256,
        num_attention_heads=8,
        latents_mean=[0.0] * 32,
        latents_std=[1.0] * 32,
    )
    pipeline = MiniMaxH3ModularPipeline(workflow=workflow)
    pipeline.update_components(
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        processor=processor,
        vae=vae,
        audio_vae=audio_vae,
        scheduler=MiniMaxH3Scheduler(shift=12.0),
        audio_scheduler=MiniMaxH3Scheduler(shift=3.0),
        **{adapter_class(workflow).transformer_component_name: build_transformer()},
    )
    pipeline.register_to_config(
        canvas_short_edge=32, canvas_max_pixels=32 * 32, reference_image_short_edge=32
    )
    return pipeline


def workflow_inputs(workflow, prompt="a red video"):
    """Real native ordered reference media at released frame/audio time scales."""
    import numpy as np
    from PIL import Image

    from flow_factory.models.minimax_h3.dependency import require_minimax_h3_support

    symbols = require_minimax_h3_support()
    values = {"prompt": prompt, "height": 32, "width": 32, "num_frames": 124}
    if workflow == "fl2va":
        values.update(
            image=Image.new("RGB", (32, 32), "red"),
            last_image=Image.new("RGB", (32, 32), "blue"),
        )
    if workflow == "ref2va":
        frames = np.zeros((24, 32, 32, 3), dtype=np.uint8)
        frames[:12, :, :, 0], frames[12:, :, :, 2] = 180, 160
        clock = torch.arange(32000)
        wave = torch.stack((torch.sin(clock * 0.02), torch.cos(clock * 0.03))) * 0.1
        values["references"] = [
            symbols.ImageReference(Image.new("RGB", (32, 32), "red")),
            symbols.AudioReference(wave, 32000),
            symbols.VideoReference(frames, 24.0),
        ]
    return values
