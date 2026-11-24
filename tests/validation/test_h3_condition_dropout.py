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

"""H3 SFT caption omission with the released tokenizer and reduced native networks."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen3VLConfig,
    Qwen3VLForConditionalGeneration,
)

from diffusers import MiniMaxH3Transformer3DModel
from flow_factory.models.minimax_h3.adapters import (
    MiniMaxH3FL2VAAdapter,
    MiniMaxH3Ref2VAAdapter,
    MiniMaxH3T2VAAdapter,
)
from flow_factory.models.minimax_h3.blocks import run_h3_blocks
from flow_factory.models.minimax_h3.dependency import require_minimax_h3_support
from flow_factory.models.minimax_h3.layout import build_row_timesteps
from flow_factory.models.sft_dropout import record_draw


@pytest.fixture(scope="module")
def conditioner():
    checkpoint = os.environ.get("FF_VALIDATION_H3_CHECKPOINT")
    if not checkpoint:
        pytest.skip("set FF_VALIDATION_H3_CHECKPOINT to a local MiniMax-H3 snapshot")
    if not torch.cuda.is_available():
        pytest.skip("H3 conditioning validation requires one GPU")
    torch.manual_seed(31)
    path = Path(checkpoint)
    cfg = Qwen3VLConfig.from_pretrained(path / "text_encoder", local_files_only=True)
    cfg.text_config.update(
        dict(
            hidden_size=24,
            head_dim=8,
            intermediate_size=48,
            num_attention_heads=3,
            num_key_value_heads=1,
            num_hidden_layers=2,
        )
    )
    cfg.text_config.rope_parameters["mrope_section"] = [2, 1, 1]
    cfg.vision_config.update(
        dict(
            depth=1,
            hidden_size=24,
            intermediate_size=48,
            num_heads=3,
            out_hidden_size=24,
            deepstack_visual_indexes=[],
        )
    )
    device = torch.device("cuda")
    return SimpleNamespace(
        _execution_device=device,
        processor=AutoProcessor.from_pretrained(path / "processor", local_files_only=True),
        tokenizer=AutoTokenizer.from_pretrained(path / "tokenizer", local_files_only=True),
        text_encoder=Qwen3VLForConditionalGeneration(cfg).to(device, torch.bfloat16).eval(),
        text_tag=1,
        video_tag=0,
        text_encoder_layer=1,
        fps=24,
    )


@pytest.mark.parametrize(
    "adapter", [MiniMaxH3T2VAAdapter, MiniMaxH3FL2VAAdapter, MiniMaxH3Ref2VAAdapter]
)
def test_caption_omission_preserves_h3_reference_and_target_rows(conditioner, adapter):
    symbols = require_minimax_h3_support()
    image = Image.new("RGB", (32, 32), "red")
    workflow = adapter.workflow
    geometry = dict(
        num_latent_frames=2,
        latent_height=2,
        latent_width=2,
        num_audio_latents=4,
        patch_size=(1, 2, 2),
        audio_channels=2,
        audio_tag=2,
        video_tag=0,
    )
    if workflow == "ref2va":
        references = [
            symbols.ImageReference(image=image),
            symbols.VideoReference(
                frames=np.zeros((49, 32, 32, 3), dtype=np.uint8),
                fps=24,
                audio=torch.ones(1, 3200),
                sample_rate=16000,
            ),
        ]
        encoder = symbols.Ref2VATextEncoderStep()
        raw = {"normalized_references": references}
        layout_builder = symbols.RefPrepareLayoutStep.build_ref2va_packed_sequence
        geometry.update(
            references=references,
            condition_latents=[torch.ones(1, 24, 1, 2, 2), torch.ones(1, 24, 2, 2, 2)],
            audio_condition_latents=[torch.ones(2, 32)],
        )
        expected_prefix = (3, 2)
    else:
        layout_builder = symbols.PrepareLayoutStep.build_packed_sequence
        if workflow == "fl2va":
            encoder = symbols.FL2VATextEncoderStep()
            raw = {"keyframes": [image, Image.new("RGB", (32, 32), "blue")]}
            geometry["keyframe_anchors"] = ("first", "last")
            expected_prefix = (2, 0)
        else:
            encoder, raw, expected_prefix = symbols.TextEncoderStep(), {}, (0, 0)

    caption = "move the left hand"
    policy = adapter.sft_condition_dropout_policy
    assert policy.sample(1.0, SimpleNamespace(prompt=caption), record_draw(42, 0, workflow)) == {
        "text"
    }

    def encode(prompt):
        return run_h3_blocks(
            conditioner,
            [encoder],
            dict(prompt=prompt, **raw),
            requested_outputs=("text_token_tags", "prompt_embeds"),
            workflow=workflow,
        )

    positive, dropped = encode(caption), encode(policy.empty_prompt)
    tags, embeddings = dropped["text_token_tags"], dropped["prompt_embeds"]
    assert tags.numel() == embeddings.shape[1]
    assert 0 < tags.numel() < positive["text_token_tags"].numel()
    assert embeddings.isfinite().all()
    if workflow == "t2va":
        assert conditioner.tokenizer("", add_special_tokens=False).input_ids == []
        assert conditioner.tokenizer(policy.empty_prompt, add_special_tokens=False).input_ids == [
            conditioner.tokenizer.eos_token_id
        ]
        assert tags.tolist() == [1]
    else:
        # The causal Qwen prefix contains every reference label, timestamp and visual token.
        torch.testing.assert_close(positive["text_token_tags"][: len(tags)], tags)
        torch.testing.assert_close(
            positive["prompt_embeds"][:, : len(tags)], embeddings, rtol=0.016, atol=0.0005
        )

    full = layout_builder(positive["text_token_tags"], **geometry)
    positions, row_tags, video_rows, audio_rows, text_rows, video_prefix, audio_prefix = (
        layout_builder(tags, **geometry)
    )
    assert (video_prefix, audio_prefix) == full[-2:] == expected_prefix
    assert (len(video_rows), len(audio_rows)) == (len(full[2]), len(full[3]))
    layout = dict(
        video_indices=video_rows,
        audio_indices=audio_rows,
        text_indices=text_rows,
        num_condition_video_rows=video_prefix,
        num_condition_audio_rows=audio_prefix,
    )
    times, indices = build_row_timesteps(layout, 0.4, 0.6, 0.999)
    device = embeddings.device
    model = MiniMaxH3Transformer3DModel(
        num_attention_heads=1,
        attention_head_dim=12,
        hidden_size=12,
        num_layers=1,
        num_refiner_layers=1,
        ffn_dim=24,
        in_channels=24,
        audio_in_channels=32,
        patch_size=(1, 2, 2),
        text_dim=24,
        freq_dim=4,
        time_embed_hidden_dim=12,
        time_embed_dim=12,
        rope_freq_dim=2,
    ).to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        video, audio = model(
            hidden_states=torch.randn(1, len(video_rows), 96, device=device),
            audio_hidden_states=torch.randn(1, len(audio_rows), 32, device=device),
            encoder_hidden_states=embeddings,
            timestep=times.to(device),
            timestep_indices=indices.to(device),
            token_tags=row_tags.to(device),
            position_ids=positions.to(device),
            video_indices=video_rows.to(device),
            audio_indices=audio_rows.to(device),
            text_indices=text_rows.to(device),
            return_dict=False,
        )
    loss = (
        video[:, video_prefix:].float().square().mean()
        + audio[:, audio_prefix:].float().square().mean()
    )
    loss.backward()
    gradient = model.context_embedder.weight.grad.float().norm()
    assert loss.isfinite() and gradient.isfinite() and gradient > 0
    print(
        json.dumps(
            dict(
                workflow=workflow,
                positive_tokens=len(full[4]),
                retained_tokens=len(text_rows),
                video_prefix_rows=video_prefix,
                audio_prefix_rows=audio_prefix,
                target_video_rows=len(video_rows) - video_prefix,
                target_audio_rows=len(audio_rows) - audio_prefix,
                loss=loss.item(),
                context_projection_grad_norm=gradient.item(),
            )
        )
    )
