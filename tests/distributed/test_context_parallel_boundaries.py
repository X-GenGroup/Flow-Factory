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

"""Model-owned CP declarations preserve serialization and preparation ordering."""

import json
from dataclasses import asdict
from enum import Enum
from types import SimpleNamespace

import pytest
import torch
from accelerate.utils import DistributedType
from peft import LoraConfig, get_peft_model
from test_model_preparation import PreparationAdapter, TinyLinear

from flow_factory.acceleration.attention_backend import AttentionBackendAccelerator
from flow_factory.hparams.parallelism_args import ParallelismArguments
from flow_factory.loading.coordinator import ModelLoadCoordinator
from flow_factory.models.abc import BaseAdapter
from flow_factory.models.context_parallel import ContextParallelBoundary, ContextParallelCapability
from flow_factory.models.qwen_image.qwen_image import QwenImageAdapter
from flow_factory.models.qwen_image.qwen_image_edit_plus import QwenImageEditPlusAdapter
from flow_factory.models.registry import get_model_adapter_class, list_registered_models
from flow_factory.models.wan.wan2_i2v import Wan2_I2V_Adapter
from flow_factory.models.wan.wan2_t2v import Wan2_T2V_Adapter
from flow_factory.trainers.common.model_preparation import ModelPreparation
from flow_factory.trainers.common.runtime_identity import _schema_digest


@pytest.mark.parametrize(
    "model_type,layout,tp,precision",
    [
        ("wan2_t2v", "wan_dense_video", True, "cross_attention_fp32_v1"),
        ("flux1", "flux_joint_image_text", True, None),
        ("qwen-image", "qwen_joint_image_text", True, None),
        ("flux1-kontext", "flux_reference_image_text", True, None),
        ("qwen-image-edit-plus", "qwen_edit_ordered_reference_image_text", True, None),
    ],
)
def test_model_owned_layout_serializes_as_its_declared_string_identity(
    model_type, layout, tp, precision
):
    capability = get_model_adapter_class(model_type).context_parallel_capability
    string_layout = {
        "layout": layout,
        "transport": "ulysses",
        "output_gradient": "local_slice",
        "version": 1,
        "tensor_parallel": tp,
        "required_precision_policy": precision,
    }
    assert json.dumps(asdict(capability), sort_keys=True) == json.dumps(
        string_layout, sort_keys=True
    )
    assert _schema_digest(asdict(capability)) == _schema_digest(string_layout)
    assert type(capability.layout).__module__.startswith("flow_factory.models.")
    assert type(capability.layout).__module__ != "flow_factory.models.context_parallel"


@pytest.mark.parametrize("layout", ["wan_dense_video", "unknown", None, 1])
def test_layout_requires_a_model_owned_enum(layout):
    with pytest.raises(ValueError, match="model-owned ContextParallelLayout member"):
        ContextParallelCapability(layout=layout)


def test_an_unrelated_string_enum_does_not_declare_a_layout():
    class Unrelated(str, Enum):
        WAN = "wan_dense_video"

    with pytest.raises(ValueError, match="model-owned ContextParallelLayout member"):
        ContextParallelCapability(layout=Unrelated.WAN)


# Bagel's dependency-stubbed import contract is included separately in this CPU case.
@pytest.mark.parametrize(
    "model_type", [name for name in list_registered_models() if name != "bagel"]
)
def test_registered_models_explicitly_own_any_sequence_boundary_override(model_type):
    adapter = get_model_adapter_class(model_type)
    if adapter in {Wan2_T2V_Adapter, Wan2_I2V_Adapter, QwenImageAdapter, QwenImageEditPlusAdapter}:
        assert adapter.compose_context_parallel_sequence_boundaries.__func__ is not (
            BaseAdapter.compose_context_parallel_sequence_boundaries.__func__
        )
    else:
        assert adapter.compose_context_parallel_sequence_boundaries.__func__ is (
            BaseAdapter.compose_context_parallel_sequence_boundaries.__func__
        )
        with pytest.raises(ValueError, match=r"does not declare TP/SP\+CP sequence boundaries"):
            adapter.compose_context_parallel_sequence_boundaries(torch.nn.Identity())


@pytest.mark.parametrize("mode,physical_roots", [("full", 2), ("lora", 1)])
@pytest.mark.parametrize("tp_size,sp", [(1, None), (2, False), (2, True)])
def test_preparation_dispatches_composition_once_per_physical_root(
    monkeypatch, mode, physical_roots, tp_size, sp
):
    events = []

    class CPLinear(TinyLinear):
        def set_attention_backend(self, backend):
            assert backend == "native"
            events.append((id(self), "attention"))

        def enable_parallelism(self, config):
            assert (id(self), "attention") in events
            self._parallel_config = SimpleNamespace(context_parallel_config=config)
            events.append((id(self), "native_cp"))

    module = CPLinear()
    if mode == "lora":
        module = get_peft_model(module, LoraConfig(r=2, lora_alpha=2, target_modules=["proj"]))
    accelerator = SimpleNamespace(
        num_processes=4 * tp_size,
        process_index=0,
        is_main_process=True,
        distributed_type=DistributedType.NO,
        state=SimpleNamespace(distributed_type=DistributedType.NO),
        parallelism_config=SimpleNamespace(
            dp_replicate_size=1, dp_shard_size=2, cp_size=2, tp_size=tp_size
        ),
    )
    adapter = PreparationAdapter(module, accelerator, mode)
    adapter.context_parallel_capability = Wan2_T2V_Adapter.context_parallel_capability
    adapter.validate_context_parallel_module = lambda module, degree, tp: None
    adapter.declare_component_variants(("policy", "critic"))
    owned = []

    def compose(root):
        assert events[-1] == (id(root), "native_cp")
        events.append((id(root), "compose"))

        def callback(module, args, kwargs):
            return args, kwargs

        boundary = ContextParallelBoundary(
            root, "input", callback, root.register_forward_pre_hook(callback, with_kwargs=True)
        )
        owned.append(boundary)
        return (boundary,)

    adapter.compose_context_parallel_sequence_boundaries = compose
    monkeypatch.setattr(
        "flow_factory.trainers.common.model_preparation.context_parallel_mesh", lambda *_: object()
    )
    owner = ModelPreparation(
        ModelLoadCoordinator(adapter, accelerator),
        ParallelismArguments(cp_size=2, tp_size=tp_size, dp_shard_size=2, sequence_parallel=sp),
    )
    owner._apply_context_parallel(AttentionBackendAccelerator(backend="native"))
    assert len(owner._context_parallel.modules) == physical_roots
    expected = physical_roots if tp_size > 1 and sp else 0
    assert len(owned) == expected
    assert owner._context_parallel.boundaries == tuple(owned)
    assert sum(event == "native_cp" for _, event in events) == physical_roots
    assert sum(event == "compose" for _, event in events) == expected
