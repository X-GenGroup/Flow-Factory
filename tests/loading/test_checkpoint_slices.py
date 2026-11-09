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

"""Checkpoint-backed construction reproduces the native Diffusers loader exactly."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from flow_factory.loading.checkpoint_slices import (
    CheckpointParameter,
    checkpoint_reader,
    construct_from_checkpoint,
)


def _sd3() -> nn.Module:
    from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel

    torch.manual_seed(0)
    model = SD3Transformer2DModel(
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
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0, 0.2)
    return model


def _wan() -> nn.Module:
    from diffusers import WanTransformer3DModel

    torch.manual_seed(0)
    model = WanTransformer3DModel(
        patch_size=(1, 2, 2),
        num_attention_heads=2,
        attention_head_dim=8,
        in_channels=4,
        out_channels=4,
        text_dim=16,
        freq_dim=16,
        ffn_dim=32,
        num_layers=2,
        cross_attn_norm=True,
        rope_max_seq_len=32,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0, 0.2)
    return model


def _fixture(module: str, function: str, *args):
    def build() -> nn.Module:
        import importlib
        import sys

        distributed = str(Path(__file__).resolve().parents[1] / "distributed")
        if distributed not in sys.path:
            sys.path.insert(0, distributed)
        model = getattr(importlib.import_module(module), function)(*args)
        if isinstance(model, tuple):
            model = model[0]
        return model.float()

    return build


_MODELS = {
    "sd3": _sd3,
    "wan": _wan,
    "wan_i2v": _fixture("wan_i2v_cp_fixture", "build_transformer"),
    "flux1": lambda: _fixture("test_region_entry_parity", "_flux1")(),
    "qwen_image_21": lambda: _fixture("test_region_entry_parity", "_qwen_image_21")(),
    "z_image": lambda: _fixture("test_region_entry_parity", "_z_image")(),
    "qwen_image": _fixture("qwen_cp_fixture", "build_transformer"),
    "flux2_klein4b": _fixture("flux2_cp_fixture", "build_transformer", "klein4b"),
    "ltx2": _fixture("ltx2_cp_fixture", "build_transformer"),
}


def _save(tmp_path: Path, model: str, dtype: torch.dtype, shard: bool = False) -> Path:
    folder = tmp_path / f"{model}-{str(dtype).split('.')[-1]}-{shard}"
    _MODELS[model]().to(dtype).save_pretrained(
        folder, safe_serialization=True, max_shard_size="20KB" if shard else "10GB"
    )
    return folder


def _assert_matches_native(native: nn.Module, built: nn.Module) -> None:
    native_parameters = dict(native.named_parameters())
    built_parameters = dict(built.named_parameters())
    assert native_parameters.keys() == built_parameters.keys()
    for name, parameter in built_parameters.items():
        reader = checkpoint_reader(parameter)
        assert reader is not None, name
        expected = native_parameters[name]
        assert parameter.dtype == expected.dtype == reader.load_dtype, name
        assert parameter.requires_grad == expected.requires_grad, name
        value = reader.read((0,) * expected.dim(), tuple(expected.shape))
        assert torch.equal(value, expected.detach()), name
    native_buffers = dict(native.named_buffers())
    built_buffers = dict(built.named_buffers())
    assert native_buffers.keys() == built_buffers.keys()
    for name, buffer in built_buffers.items():
        assert buffer.dtype == native_buffers[name].dtype, name
        assert torch.equal(buffer, native_buffers[name]), name
    assert built.training == native.training is False


@pytest.mark.parametrize("model", sorted(_MODELS))
@pytest.mark.parametrize("stored", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize(
    "dtype", [None, torch.float32, torch.bfloat16], ids=["none", "fp32", "bf16"]
)
@pytest.mark.parametrize("low_cpu_mem_usage", [True, False], ids=["meta", "eager"])
def test_construction_matches_the_native_loader(
    tmp_path: Path, model: str, stored, dtype, low_cpu_mem_usage: bool
) -> None:
    folder = _save(tmp_path, model, stored)
    component_cls = type(_MODELS[model]())
    if not low_cpu_mem_usage and component_cls._keep_in_fp32_modules:
        pytest.skip("Diffusers requires low_cpu_mem_usage for _keep_in_fp32_modules")
    native = component_cls.from_pretrained(
        folder, torch_dtype=dtype, low_cpu_mem_usage=low_cpu_mem_usage
    )
    built, reason = construct_from_checkpoint(
        component_cls, folder, dtype, low_cpu_mem_usage=low_cpu_mem_usage
    )

    assert reason is None
    _assert_matches_native(native, built)


def test_sharded_checkpoints_read_across_files(tmp_path: Path) -> None:
    folder = _save(tmp_path, "sd3", torch.float32, shard=True)
    assert (folder / "diffusion_pytorch_model.safetensors.index.json").is_file()
    component_cls = type(_sd3())
    native = component_cls.from_pretrained(folder)
    built, reason = construct_from_checkpoint(component_cls, folder, None)

    assert reason is None
    _assert_matches_native(native, built)


def test_checkpoint_parameters_hold_no_weight_storage_and_survive_copies(tmp_path: Path) -> None:
    folder = _save(tmp_path, "sd3", torch.float32)
    built, _ = construct_from_checkpoint(type(_sd3()), folder, torch.float32)
    copied = copy.deepcopy(built)

    for module in (built, copied):
        for parameter in module.parameters():
            assert isinstance(parameter, CheckpointParameter)
            assert parameter.untyped_storage().nbytes() == parameter.element_size()
    first = next(built.parameters())
    assert checkpoint_reader(next(copied.parameters())) == checkpoint_reader(first)


def test_checkpoints_the_native_loader_must_transform_fall_back(tmp_path: Path) -> None:
    component_cls = type(_sd3())
    bin_folder = tmp_path / "bin"
    _sd3().save_pretrained(bin_folder, safe_serialization=False)
    assert construct_from_checkpoint(component_cls, bin_folder, None)[0] is None

    folder = _save(tmp_path, "sd3", torch.float32)
    config = component_cls.load_config(folder)
    config["num_layers"] = 3
    (folder / "config.json").write_text(json.dumps(config))
    module, reason = construct_from_checkpoint(component_cls, folder, None)
    assert module is None and "checkpoint lacks parameters" in reason


def test_reading_a_checkpoint_backed_value_before_materialization_raises(tmp_path: Path) -> None:
    folder = _save(tmp_path, "sd3", torch.float32)
    built, _ = construct_from_checkpoint(type(_sd3()), folder, torch.float32)
    parameter = built.proj_out.weight

    assert parameter.shape == built.proj_out.weight.shape and parameter.dim() == 2
    parameter.requires_grad_(False)
    parameter.data = parameter.data.to(torch.bfloat16)
    assert isinstance(parameter, CheckpointParameter)
    with pytest.raises(RuntimeError, match="'proj_out.weight' was read through"):
        parameter.sum()
    with pytest.raises(RuntimeError, match="before sharded materialization"):
        built.proj_out(torch.ones(1, parameter.shape[1], dtype=torch.bfloat16))


def _sd35_adapter(target_materialization: str, resume_type=None):
    from types import SimpleNamespace

    from flow_factory.hparams import ParallelismArguments
    from flow_factory.models.stable_diffusion.sd3_5 import SD3_5Adapter

    adapter = object.__new__(SD3_5Adapter)
    adapter.config = SimpleNamespace(
        parallelism_args=ParallelismArguments(target_materialization=target_materialization)
    )
    adapter.model_args = SimpleNamespace(
        resume_path="/resume" if resume_type else None,
        resume_type=resume_type,
        finetune_type="lora",
        target_components="transformer",
    )
    adapter._component_load_dtype_manifest = SD3_5Adapter.component_load_dtype_defaults
    adapter._component_load_dtype_overrides = None
    return adapter


def test_the_adapter_loader_keeps_only_targets_in_their_checkpoint(tmp_path: Path) -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "distributed"))
    from sd35_cp_fixture import build_fixture

    from diffusers import StableDiffusion3Pipeline

    build_fixture(tmp_path, "medium")
    path = str(tmp_path / "model")
    native = _sd35_adapter("replicated")._load_diffusers_pipeline(
        StableDiffusion3Pipeline, path, low_cpu_mem_usage=False
    )
    backed = _sd35_adapter("sharded")._load_diffusers_pipeline(
        StableDiffusion3Pipeline, path, low_cpu_mem_usage=False
    )
    resumed = _sd35_adapter("sharded", resume_type="full")._load_diffusers_pipeline(
        StableDiffusion3Pipeline, path, low_cpu_mem_usage=False
    )

    _assert_matches_native(native.transformer, backed.transformer)
    assert not any(isinstance(p, CheckpointParameter) for p in backed.vae.parameters())
    assert not any(isinstance(p, CheckpointParameter) for p in native.transformer.parameters())
    assert not any(isinstance(p, CheckpointParameter) for p in resumed.transformer.parameters())
