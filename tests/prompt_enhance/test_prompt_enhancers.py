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

"""Fake-backed wiring tests for the built-in prompt enhancers."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import diffusers
import pytest
import torch
import transformers
from diffusers.pipelines.ltx2 import utils as ltx2_utils
from PIL import Image

from flow_factory.prompt_enhance import PromptEnhanceArguments
from flow_factory.prompt_enhance.enhancers import openai_compatible
from flow_factory.prompt_enhance.enhancers.flux2 import Flux2PromptEnhancer
from flow_factory.prompt_enhance.enhancers.flux3 import Flux3PromptEnhancer
from flow_factory.prompt_enhance.enhancers.ltx2 import LTX2PromptEnhancer
from flow_factory.prompt_enhance.enhancers.openai_compatible import OpenAICompatiblePromptEnhancer

DEVICE = torch.device("cpu")


class _Movable:
    def __init__(self) -> None:
        self.devices: List[torch.device] = []

    def to(self, device: torch.device) -> "_Movable":
        self.devices.append(device)
        return self


def _args(**values: Any) -> PromptEnhanceArguments:
    return PromptEnhanceArguments.from_dict(
        {"input_dir": "in", "output_dir": "out", "model_name_or_path": "repo", **values}
    )


def test_fakes_match_the_pinned_upstream_apis() -> None:
    """The fakes in this file stand in for these upstream entry points and components."""
    upsample = inspect.signature(diffusers.Flux2Pipeline.upsample_prompt).parameters
    assert {"prompt", "images", "temperature", "device"} <= set(upsample)
    enhance = inspect.signature(diffusers.LTX2Pipeline.enhance_prompt).parameters
    assert {
        "prompt",
        "system_prompt",
        "max_new_tokens",
        "seed",
        "generation_kwargs",
        "device",
        "image",
    } <= set(enhance)
    ltx2_components = inspect.signature(diffusers.LTX2Pipeline.__init__).parameters
    assert {"text_encoder", "tokenizer", "processor", "prompt_enhancer"} <= set(ltx2_components)
    assert hasattr(transformers, "Qwen3_5ForConditionalGeneration")


def _image(color: int) -> Image.Image:
    return Image.new("RGB", (4, 4), (color, color, color))


def _patch_pipeline_loading(
    monkeypatch: pytest.MonkeyPatch,
    pipeline_class: Any,
    config: Dict[str, Any],
    pipeline: Any,
) -> Dict[str, Any]:
    calls: Dict[str, Any] = {}

    def from_pretrained(path: str, **kwargs: Any) -> Any:
        calls["path"] = path
        calls["kwargs"] = kwargs
        return pipeline

    monkeypatch.setattr(pipeline_class, "load_config", lambda path: config)
    monkeypatch.setattr(pipeline_class, "from_pretrained", from_pretrained)
    return calls


def test_flux2_loads_only_text_components_and_splits_t2i_from_i2i(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upsample_calls = []

    def upsample_prompt(prompt, images, temperature, device):
        upsample_calls.append((list(prompt), images, temperature, device))
        return [f"up:{text}" for text in prompt]

    pipeline = SimpleNamespace(text_encoder=_Movable(), upsample_prompt=upsample_prompt)
    load_calls = _patch_pipeline_loading(
        monkeypatch,
        diffusers.Flux2Pipeline,
        {
            "_class_name": "Flux2Pipeline",
            "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
            "text_encoder": ["transformers", "Mistral3ForConditionalGeneration"],
            "tokenizer": ["transformers", "PixtralProcessor"],
            "transformer": ["diffusers", "Flux2Transformer2DModel"],
            "vae": ["diffusers", "AutoencoderKLFlux2"],
        },
        pipeline,
    )
    enhancer = Flux2PromptEnhancer(_args(enhancer="flux2", generation={"temperature": 0.3}), DEVICE)
    enhancer.load()

    assert load_calls["kwargs"] == {
        "dtype": torch.bfloat16,
        "scheduler": None,
        "transformer": None,
        "vae": None,
    }
    assert pipeline.text_encoder.devices == [DEVICE]

    reference = _image(1)
    outputs = enhancer.enhance(["a", "b", "c"], [[], [reference], []], seeds=[3, 4, 5])
    assert outputs == ["up:a", "up:b", "up:c"]
    assert upsample_calls == [
        (["a", "c"], None, 0.3, DEVICE),
        (["b"], [[reference]], 0.3, DEVICE),
    ]

    with pytest.raises(ValueError, match="system_prompt must be null"):
        Flux2PromptEnhancer(_args(enhancer="flux2", system_prompt="custom"), DEVICE).load()


def _ltx2_pipeline(prompt_enhancer: Any) -> SimpleNamespace:
    pipeline = SimpleNamespace(
        processor=object(),
        prompt_enhancer=prompt_enhancer,
        text_encoder=_Movable(),
        enhance_calls=[],
    )

    def enhance_prompt(**kwargs):
        pipeline.enhance_calls.append(kwargs)
        return [f"enh:{kwargs['prompt']}"]

    pipeline.enhance_prompt = enhance_prompt
    return pipeline


_LTX2_CONFIG = {
    "_class_name": "LTX2Pipeline",
    "audio_vae": ["diffusers", "AutoencoderKLLTX2Audio"],
    "connectors": ["ltx2", "LTX2TextConnectors"],
    "processor": ["transformers", "Gemma3Processor"],
    "prompt_enhancer": [None, None],
    "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    "text_encoder": ["transformers", "Gemma3ForConditionalGeneration"],
    "tokenizer": ["transformers", "GemmaTokenizerFast"],
    "transformer": ["diffusers", "LTX2VideoTransformer3DModel"],
    "vae": ["diffusers", "AutoencoderKLLTX2Video"],
    "vocoder": ["ltx2", "LTX2Vocoder"],
}


def test_ltx2_uses_upstream_default_system_prompts_per_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _ltx2_pipeline(prompt_enhancer=None)
    load_calls = _patch_pipeline_loading(
        monkeypatch, diffusers.LTX2Pipeline, _LTX2_CONFIG, pipeline
    )
    enhancer = LTX2PromptEnhancer(
        _args(enhancer="ltx2", generation={"max_new_tokens": 64, "temperature": 0.5}),
        DEVICE,
    )
    enhancer.load()

    skipped = {name for name, value in load_calls["kwargs"].items() if value is None}
    assert skipped == {"audio_vae", "connectors", "scheduler", "transformer", "vae", "vocoder"}
    assert pipeline.text_encoder.devices == [DEVICE]

    first, second = _image(1), _image(2)
    outputs = enhancer.enhance(["a", "b"], [[], [first, second]], seeds=[7, 9])
    assert outputs == ["enh:a", "enh:b"]
    text_call, image_call = pipeline.enhance_calls
    assert text_call["system_prompt"] == ltx2_utils.T2V_DEFAULT_SYSTEM_PROMPT
    assert text_call["image"] is None
    assert image_call["system_prompt"] == ltx2_utils.I2V_DEFAULT_SYSTEM_PROMPT
    assert image_call["image"] is first
    assert [text_call["seed"], image_call["seed"]] == [7, 9]
    for call in (text_call, image_call):
        assert call["max_new_tokens"] == 64
        assert call["generation_kwargs"] == {"temperature": 0.5}
        assert call["device"] == DEVICE


def test_ltx2_dedicated_enhancer_selects_ltx25_prompts(monkeypatch: pytest.MonkeyPatch) -> None:
    dedicated = _Movable()
    pipeline = _ltx2_pipeline(prompt_enhancer=dedicated)
    _patch_pipeline_loading(monkeypatch, diffusers.LTX2Pipeline, _LTX2_CONFIG, pipeline)
    enhancer = LTX2PromptEnhancer(_args(enhancer="ltx2"), DEVICE)
    enhancer.load()

    assert dedicated.devices == [DEVICE]
    assert pipeline.text_encoder.devices == []
    assert enhancer.system_prompt_for(False) == ltx2_utils.LTX2_5_T2V_DEFAULT_SYSTEM_PROMPT
    assert enhancer.system_prompt_for(True) == ltx2_utils.LTX2_5_I2V_DEFAULT_SYSTEM_PROMPT
    enhancer.enhance(["a"], [[]], seeds=[1])
    assert pipeline.enhance_calls[0]["generation_kwargs"] is None

    pipeline.processor = None
    with pytest.raises(ValueError, match="'processor' component"):
        enhancer.load()


class _Flux3Inputs(dict):
    def to(self, device: torch.device) -> "_Flux3Inputs":
        return self


class _Flux3Processor:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return _Flux3Inputs(input_ids=torch.tensor([[1, 2, 3]]))

    @staticmethod
    def decode(tokens: torch.Tensor, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is True
        return " " + " ".join(str(value) for value in tokens.tolist()) + "\n"


class _Flux3Model(_Movable):
    def __init__(self) -> None:
        super().__init__()
        self.generate_calls: List[Dict[str, Any]] = []

    def eval(self) -> "_Flux3Model":
        return self

    def generate(self, **kwargs: Any) -> torch.Tensor:
        self.generate_calls.append({**kwargs, "rng_seed": torch.initial_seed()})
        return torch.cat([kwargs["input_ids"], torch.tensor([[10, 11]])], dim=1)


def test_flux3_follows_the_upsampler_recipe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "system_prompts.json").write_text(
        json.dumps({"t2i": "T2I SYSTEM", "i2i": "I2I SYSTEM"}), encoding="utf-8"
    )
    processor, model = _Flux3Processor(), _Flux3Model()
    loads = {}
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", lambda path: processor)

    def load_model(path, dtype):
        loads["model"] = (path, dtype)
        return model

    monkeypatch.setattr(transformers.Qwen3_5ForConditionalGeneration, "from_pretrained", load_model)
    enhancer = Flux3PromptEnhancer(
        _args(
            enhancer="flux3",
            model_name_or_path=str(tmp_path),
            generation={"do_sample": True, "temperature": 0.7},
        ),
        DEVICE,
    )
    enhancer.load()
    assert loads["model"] == (str(tmp_path), torch.bfloat16)
    assert model.devices == [DEVICE]

    reference = _image(3)
    outputs = enhancer.enhance(["a fox", "move it"], [[], [reference]], seeds=[5, 8])
    assert outputs == ["10 11", "10 11"]
    assert [call["rng_seed"] for call in model.generate_calls] == [5, 8]
    t2i_call, i2i_call = processor.calls
    assert t2i_call["messages"] == [
        {"role": "system", "content": "T2I SYSTEM"},
        {"role": "user", "content": "a fox"},
    ]
    assert i2i_call["messages"] == [
        {"role": "system", "content": "I2I SYSTEM"},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": reference},
                {
                    "type": "text",
                    "text": (
                        "Reference images follow in order (the first image, the second image, ...).\n"
                        "User request: move it\nOutput the caption."
                    ),
                },
            ],
        },
    ]
    assert t2i_call["kwargs"] == {
        "add_generation_prompt": True,
        "tokenize": True,
        "return_dict": True,
        "return_tensors": "pt",
        "enable_thinking": False,
    }
    generation = {
        key: value
        for key, value in model.generate_calls[0].items()
        if key not in {"input_ids", "rng_seed"}
    }
    assert generation == {"max_new_tokens": 3584, "do_sample": True, "temperature": 0.7}
    assert set(enhancer.describe()["system_prompts_sha256"]) == {"i2i", "t2i"}


def test_openai_compatible_posts_chat_completions(monkeypatch: pytest.MonkeyPatch) -> None:
    posts = []

    class _Response:
        def __init__(self, prompt: str) -> None:
            self.prompt = prompt

        def raise_for_status(self) -> None:
            return None

        def json(self) -> Dict[str, Any]:
            return {"choices": [{"message": {"content": f"  enhanced {self.prompt}  "}}]}

    def post(url, json, headers, timeout):
        posts.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        user_content = json["messages"][1]["content"]
        prompt = user_content if isinstance(user_content, str) else user_content[0]["text"]
        return _Response(prompt)

    monkeypatch.setattr(openai_compatible.requests, "post", post)
    monkeypatch.setenv("PE_TEST_KEY", "secret")
    enhancer = OpenAICompatiblePromptEnhancer(
        _args(
            enhancer="openai-compatible",
            model_name_or_path="served-model",
            system_prompt="Rewrite the prompt.",
            generation={"temperature": 0.9, "max_new_tokens": 128},
            extra_kwargs={
                "base_url": "http://localhost:8000/v1/",
                "api_key_env": "PE_TEST_KEY",
                "max_concurrency": 1,
            },
        ),
        DEVICE,
    )
    enhancer.load()

    outputs = enhancer.enhance(["a", "b"], [[], [_image(4)]], seeds=[10, 13])
    assert outputs == ["enhanced a", "enhanced b"]
    assert [post["url"] for post in posts] == ["http://localhost:8000/v1/chat/completions"] * 2
    assert [post["json"]["seed"] for post in posts] == [10, 13]
    first, second = (post["json"] for post in posts)
    assert first["model"] == "served-model"
    assert first["messages"] == [
        {"role": "system", "content": "Rewrite the prompt."},
        {"role": "user", "content": "a"},
    ]
    assert first["temperature"] == 0.9
    assert first["max_tokens"] == 128
    image_part = second["messages"][1]["content"][1]
    assert image_part["type"] == "image_url"
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")
    assert posts[0]["headers"]["Authorization"] == "Bearer secret"

    with pytest.raises(ValueError, match="requires system_prompt"):
        OpenAICompatiblePromptEnhancer(
            _args(enhancer="openai-compatible", extra_kwargs={"base_url": "http://x"}),
            DEVICE,
        ).load()
