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

"""Compare SFT caption dropout to native text, CFG and multimodal conditioning."""

from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from flow_factory.data_utils.schema import MediaAsset, NormalizedModelInput
from flow_factory.models.runtime.classic import ClassicPipelineRuntime


@pytest.fixture(params=("cpu", "cuda"))
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    return torch.device(request.param)


def _adapter(adapter_class, pipeline, device):
    adapter = object.__new__(adapter_class)
    adapter.pipeline = pipeline.to(device)
    adapter.component_runtime = ClassicPipelineRuntime(adapter.pipeline)
    adapter.accelerator = SimpleNamespace(device=device)
    return adapter


def _dropped_caption(adapter, *, references=()):
    model_input = NormalizedModelInput(
        prompt="a red image",
        negative_prompt="blue",
        media=tuple(
            MediaAsset(type="image", path=f"reference-{index}.png")
            for index in range(len(references))
        ),
    )
    policy = adapter.sft_condition_dropout_policy
    assert policy.sample(1.0, model_input, lambda _: 0.0) == {"text"}
    return [policy.empty_prompt]


@torch.no_grad()
def test_sd35_empty_caption_reproduces_native_sequence_and_pooled_embeddings(device):
    from sd35_cp_fixture import build_pipeline

    from flow_factory.models.stable_diffusion.sd3_5 import SD3_5Adapter

    pipe = build_pipeline("medium")
    adapter = _adapter(SD3_5Adapter, pipe, device)
    dropped = adapter.preprocess_func(prompt=_dropped_caption(adapter), guidance_scale=1)
    _, native_empty, _, native_pooled = pipe.encode_prompt(
        "a red image",
        "a red image",
        "a red image",
        device=device,
        negative_prompt="",
        do_classifier_free_guidance=True,
    )
    torch.testing.assert_close(dropped["prompt_embeds"], native_empty)
    torch.testing.assert_close(dropped["pooled_prompt_embeds"], native_pooled)


@torch.no_grad()
def test_klein_empty_caption_reproduces_native_unmixed_velocity_with_ordered_references(device):
    from flux2_cp_fixture import build_pipeline, pipeline_adapter

    pipe = build_pipeline("klein4b").to(device)
    adapter = pipeline_adapter(pipe, "klein4b")
    adapter.accelerator.device = device
    refs = [Image.new("RGB", (64, 64), color) for color in ("red", "blue")]
    dropped = adapter.preprocess_func(
        prompt=_dropped_caption(adapter, references=refs),
        images=[refs],
        guidance_scale=1.0,
        max_sequence_length=8,
        condition_image_size=64,
    )
    empty_text, empty_ids = pipe.encode_prompt("", device=device, max_sequence_length=8)
    torch.testing.assert_close(dropped["prompt_embeds"], empty_text)
    torch.testing.assert_close(dropped["text_ids"], empty_ids)
    reference, reference_ids = pipe.prepare_image_latents(
        [pipe.image_processor.preprocess(img, height=64, width=64) for img in refs],
        batch_size=1,
        generator=None,
        device=device,
        dtype=torch.float32,
    )
    torch.testing.assert_close(dropped["image_latents"][0], reference[0])
    torch.testing.assert_close(dropped["image_latent_ids"][0], reference_ids[0])
    assert torch.unique_consecutive(reference_ids[0, :, 0]).numel() == len(refs)
    grid = torch.randn((1, 128, 4, 4), generator=torch.Generator().manual_seed(9)).to(device)
    target = pipe._pack_latents(grid)
    target_ids = pipe._prepare_latent_ids(grid).to(device)
    t = adapter.scheduler.timesteps[1:2].to(device)
    expected = pipe.transformer(
        hidden_states=torch.cat([target, reference], dim=1),
        img_ids=torch.cat([target_ids, reference_ids], dim=1),
        encoder_hidden_states=empty_text,
        txt_ids=empty_ids,
        timestep=t / 1000,
        guidance=None,
        return_dict=False,
    )[0][:, : target.shape[1]]
    actual = adapter.forward(
        t=t,
        t_next=adapter.scheduler.timesteps[2:3].to(device),
        noise_level=0.0,
        latents=target,
        latent_ids=target_ids,
        prompt_embeds=dropped["prompt_embeds"],
        text_ids=dropped["text_ids"],
        image_latents=dropped["image_latents"],
        image_latent_ids=dropped["image_latent_ids"],
        compute_log_prob=False,
        return_kwargs=["velocity"],
        **adapter.offline_training_forward_overrides,
    )
    torch.testing.assert_close(actual.velocity, expected)


@torch.no_grad()
@pytest.mark.parametrize("model", ("qwen-image-edit-plus", "qwen-image-2.1"))
def test_qwen_empty_caption_preserves_reference_dependent_vlm_context(model):
    from flow_factory.models.registry import get_model_adapter_class

    if model == "qwen-image-edit-plus":
        from qwen_edit_cp_fixture import build_pipeline
    else:
        from qwen21_cp_fixture import build_pipeline

    device = torch.device("cpu")
    adapter = _adapter(get_model_adapter_class(model), build_pipeline(), device)
    refs = [Image.new("RGB", (64, 64), color) for color in ("red", "blue")]
    cfg = {"guidance_scale": 2.0} if model == "qwen-image-2.1" else {}
    full = adapter.preprocess_func(
        ["a red image"],
        images=[refs],
        negative_prompt=[""],
        **cfg,
        condition_image_size=64,
    )
    dropped = adapter.preprocess_func(
        _dropped_caption(adapter, references=refs),
        images=[refs],
        negative_prompt=["blue"],
        condition_image_size=64,
    )
    reversed_refs = adapter.preprocess_func(
        [""],
        images=[list(reversed(refs))],
        condition_image_size=64,
    )
    fields = ["prompt_ids", "prompt_embeds", "prompt_embeds_mask"]
    if model == "qwen-image-2.1":
        fields.append("image_pad_mask")
    for key in fields:
        torch.testing.assert_close(dropped[key][0], full[f"negative_{key}"][0])
    assert dropped["prompt_embeds"][0].numel()
    assert not torch.allclose(dropped["prompt_embeds"][0], reversed_refs["prompt_embeds"][0])


def test_caption_policy_rejects_native_and_thresholds_one_record_event():
    from flow_factory.models.sft_dropout import record_draw, resolve_sft_dropout
    from flow_factory.models.stable_diffusion.sd3_5 import SD3_5Adapter

    assert resolve_sft_dropout(SD3_5Adapter, "local/tiny", 0.0) is None
    policy = resolve_sft_dropout(SD3_5Adapter, "local/tiny", 0.5)
    with pytest.raises(ValueError, match="local/tiny.*numeric"):
        resolve_sft_dropout(SD3_5Adapter, "local/tiny", "native")
    caption = NormalizedModelInput("a red image", "blue", ())
    assert policy.sample(0.5, caption, lambda event: 0.499) == {"text"}
    assert not policy.sample(0.5, caption, lambda event: 0.5)
    assert not policy.sample(1, NormalizedModelInput("", "blue", ()), lambda _: 0.0)
    original = record_draw(42, 0, "record-1")
    assert original("caption") == record_draw(42, 0, "record-1")("caption")
    assert (
        len({original("caption"), original("other"), record_draw(42, 1, "record-1")("caption")})
        == 3
    )
