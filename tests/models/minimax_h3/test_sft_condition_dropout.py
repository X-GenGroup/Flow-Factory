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

"""Exercise the caption-free T2VA presentation through H3's native conditioner."""

from types import SimpleNamespace

import torch
from transformers import Qwen2Tokenizer, Qwen3VLConfig, Qwen3VLForConditionalGeneration

from flow_factory.models.minimax_h3.adapters import MiniMaxH3T2VAAdapter
from flow_factory.models.minimax_h3.blocks import run_h3_blocks
from flow_factory.models.minimax_h3.dependency import require_minimax_h3_support
from flow_factory.models.minimax_h3.workflow import preprocess_h3_workflow
from flow_factory.models.sft_dropout import record_draw


def test_t2va_caption_dropout_encodes_one_nonlexical_row(monkeypatch) -> None:
    policy = MiniMaxH3T2VAAdapter.sft_condition_dropout_policy
    tokenizer = Qwen2Tokenizer(vocab={"<|endoftext|>": 0, "<|im_end|>": 1}, eos_token="<|im_end|>")
    assert tokenizer("", add_special_tokens=False).input_ids == []
    assert tokenizer(policy.empty_prompt, add_special_tokens=False).input_ids == [
        tokenizer.eos_token_id
    ]

    encoder = Qwen3VLForConditionalGeneration(
        Qwen3VLConfig(
            text_config=dict(
                vocab_size=8,
                hidden_size=24,
                head_dim=8,
                intermediate_size=48,
                num_attention_heads=3,
                num_key_value_heads=1,
                num_hidden_layers=2,
                rope_parameters={"rope_type": "default", "mrope_section": [2, 1, 1]},
            ),
            vision_config=dict(
                depth=1,
                hidden_size=24,
                intermediate_size=48,
                num_heads=3,
                out_hidden_size=24,
                deepstack_visual_indexes=[],
            ),
        )
    ).eval()
    symbols = require_minimax_h3_support()
    components = SimpleNamespace(
        tokenizer=tokenizer,
        text_encoder=encoder,
        text_encoder_layer=1,
        text_tag=1,
        _execution_device=torch.device("cpu"),
        processor=SimpleNamespace(
            create_mm_token_type_ids=lambda ids: [[0] * len(row) for row in ids]
        ),
    )

    def run_text_encoder(pipeline, values, *, workflow):
        return run_h3_blocks(
            pipeline,
            [symbols.TextEncoderStep()],
            values,
            workflow=workflow,
            requested_outputs=("prompt_embeds", "text_token_tags"),
        )

    monkeypatch.setattr(
        "flow_factory.models.minimax_h3.workflow.encode_h3_workflow_inputs", run_text_encoder
    )
    adapter = SimpleNamespace(
        workflow="t2va", pipeline=components, sft_condition_dropout_policy=policy
    )
    assert not policy.sample(0.0, SimpleNamespace(prompt=""), record_draw(42, 0, "already-empty"))
    original = preprocess_h3_workflow(adapter, prompt=[""], height=32, width=32, num_frames=124)
    dropped = preprocess_h3_workflow(
        adapter, prompt=[policy.empty_prompt], height=32, width=32, num_frames=124
    )
    assert dropped["prompt_embeds"].shape == (1, 1, 24)
    assert dropped["prompt_embeds"].isfinite().all()
    assert dropped["text_token_tags"][0].tolist() == [1]
    torch.testing.assert_close(original["prompt_embeds"], dropped["prompt_embeds"], atol=0, rtol=0)
