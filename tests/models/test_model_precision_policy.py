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

import pytest
import torch

from diffusers import DiffusionPipeline
from flow_factory.models.abc import BaseAdapter
from flow_factory.models.precision import (
    cast_module_role_dtypes,
    component_dtype_mapping,
)


class _AdapterWeights(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lora_A = torch.nn.Linear(2, 2, bias=False)


class _ProtectedDiffusersModel(torch.nn.Module):
    _keep_in_fp32_modules = ["proj_in"]

    def __init__(self) -> None:
        super().__init__()
        self.proj_in = torch.nn.Linear(2, 2)
        self.proj_in.adapter = _AdapterWeights()
        self.body = torch.nn.Linear(2, 2)


_ProtectedDiffusersModel.__module__ = "diffusers.test_precision"


class _PipelineFake(DiffusionPipeline):
    load_call = None
    _optional_components = ["image_encoder"]

    @classmethod
    def load_config(cls, pretrained_model_name_or_path: str, **kwargs):
        return {
            "transformer": ["diffusers", "Transformer"],
            "transformer_2": ["diffusers", "Transformer"],
            "text_encoder": ["transformers", "TextEncoder"],
            "scheduler": ["diffusers", "Scheduler"],
        }

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        cls.load_call = (pretrained_model_name_or_path, kwargs)
        return object()


def test_load_policy_overlay_and_group_resolution() -> None:
    assert component_dtype_mapping(
        user_policy={"transformer_2": None},
        manifest_policy={"transformers": torch.bfloat16, "vae": torch.float32},
        component_names=["transformer", "transformer_2", "vae", "text_encoder"],
        transformer_names=["transformer", "transformer_2"],
        text_encoder_names=["text_encoder"],
    ) == {
        "transformer": torch.bfloat16,
        "vae": torch.float32,
    }


def test_user_default_null_disables_concrete_manifest_defaults() -> None:
    assert (
        component_dtype_mapping(
            user_policy={"default": None},
            manifest_policy={"transformer": torch.float32},
            component_names=["transformer", "vae"],
            transformer_names=["transformer"],
            text_encoder_names=[],
        )
        == {}
    )


def test_present_optional_manifest_selector_resolves_its_dtype() -> None:
    assert component_dtype_mapping(
        user_policy=None,
        manifest_policy={"image_encoder": torch.float32},
        component_names=["transformer", "image_encoder"],
        transformer_names=["transformer"],
        text_encoder_names=[],
        manifest_declared_names=["transformer", "image_encoder"],
    ) == {"image_encoder": torch.float32}


def test_transformers_group_includes_reference_transformer() -> None:
    assert component_dtype_mapping(
        user_policy={"transformers": torch.bfloat16},
        manifest_policy=None,
        component_names=["transformer_ref", "vae"],
        transformer_names=["transformer_ref"],
        text_encoder_names=[],
    ) == {"transformer_ref": torch.bfloat16}


def test_eager_pipeline_loader_expands_role_selectors() -> None:
    adapter = type(
        "AdapterStub",
        (),
        {
            "_component_load_dtype_manifest": {
                "transformers": torch.bfloat16,
                "image_encoder": torch.float32,
            },
            "_component_load_dtype_overrides": None,
            "_resolve_component_load_dtype_mapping": (
                BaseAdapter._resolve_component_load_dtype_mapping
            ),
            "_checkpoint_backed_targets": BaseAdapter._checkpoint_backed_targets,
            "_checkpoint_backed_request": lambda self: (),
        },
    )()

    result = BaseAdapter._load_diffusers_pipeline(
        adapter,
        _PipelineFake,
        "model",
        low_cpu_mem_usage=False,
    )

    assert type(result) is object
    assert _PipelineFake.load_call == (
        "model",
        {
            "low_cpu_mem_usage": False,
            "dtype": {
                "default": None,
                "transformer": torch.bfloat16,
                "transformer_2": torch.bfloat16,
            },
        },
    )


def test_user_policy_cannot_select_an_absent_optional_component() -> None:
    adapter = type(
        "AdapterStub",
        (),
        {
            "_component_load_dtype_manifest": None,
            "_component_load_dtype_overrides": {"image_encoder": torch.float32},
        },
    )()

    with pytest.raises(ValueError, match=r"unknown=.*image_encoder"):
        BaseAdapter._load_diffusers_pipeline(adapter, _PipelineFake, "model")


def test_unknown_load_policy_selector_fails_with_runtime_context() -> None:
    with pytest.raises(ValueError, match=r"unknown=.*missing.*declared=.*transformer"):
        component_dtype_mapping(
            user_policy={"missing": torch.bfloat16},
            manifest_policy=None,
            component_names=["transformer"],
            transformer_names=["transformer"],
            text_encoder_names=[],
        )


def test_unknown_manifest_selector_remains_strict_with_optional_declarations() -> None:
    with pytest.raises(ValueError, match=r"unknown=.*missing"):
        component_dtype_mapping(
            user_policy=None,
            manifest_policy={"missing": torch.bfloat16},
            component_names=["transformer"],
            transformer_names=["transformer"],
            text_encoder_names=[],
            manifest_declared_names=["transformer", "image_encoder"],
        )


def test_role_cast_preserves_diffusers_fp32_islands_but_not_lora_dtype() -> None:
    model = _ProtectedDiffusersModel()
    model.requires_grad_(False)
    model.body.weight.requires_grad_(True)
    model.proj_in.adapter.lora_A.weight.requires_grad_(True)

    result = cast_module_role_dtypes(
        model,
        component_name="transformer",
        trainable_dtype=torch.bfloat16,
        frozen_dtype=torch.bfloat16,
        is_adapter_parameter=lambda name: "lora_" in name,
    )

    assert model.proj_in.weight.dtype is torch.float32
    assert model.proj_in.bias.dtype is torch.float32
    assert model.proj_in.adapter.lora_A.weight.dtype is torch.bfloat16
    assert model.body.weight.dtype is torch.bfloat16
    assert result.protected == 2


def test_full_finetune_keeps_protected_base_parameters_in_fp32() -> None:
    model = _ProtectedDiffusersModel()

    cast_module_role_dtypes(
        model,
        component_name="transformer",
        trainable_dtype=torch.bfloat16,
        frozen_dtype=None,
    )

    assert model.proj_in.weight.dtype is torch.float32
    assert model.proj_in.bias.dtype is torch.float32
    assert model.body.weight.dtype is torch.bfloat16


def test_quantized_component_rejects_storage_dtype_mutation() -> None:
    model = torch.nn.Linear(2, 2)
    model.is_quantized = True

    with pytest.raises(
        ValueError,
        match=r"quantized component='transformer'.*expected existing dtype.*requested",
    ):
        cast_module_role_dtypes(
            model,
            component_name="transformer",
            trainable_dtype=torch.bfloat16,
            frozen_dtype=None,
        )


def test_quantized_component_allows_adapter_parameter_cast() -> None:
    model = _ProtectedDiffusersModel()
    model.is_quantized = True
    model.requires_grad_(False)
    model.proj_in.adapter.lora_A.weight.requires_grad_(True)

    cast_module_role_dtypes(
        model,
        component_name="transformer",
        trainable_dtype=torch.bfloat16,
        frozen_dtype=None,
        is_adapter_parameter=lambda name: "lora_" in name,
    )

    assert model.proj_in.weight.dtype is torch.float32
    assert model.proj_in.adapter.lora_A.weight.dtype is torch.bfloat16


def test_quantized_component_preserves_floating_point_buffers() -> None:
    model = torch.nn.Module()
    model.is_quantized = True
    model.register_buffer("scale", torch.ones(1, dtype=torch.float32))

    cast_module_role_dtypes(
        model,
        component_name="transformer",
        trainable_dtype=torch.bfloat16,
        frozen_dtype=torch.bfloat16,
    )

    assert model.scale.dtype is torch.float32
