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

"""Flux recipe for the shared fixed-operand CP numerical controls."""

from context_parallel_model_control import (
    clip_prepared_model,
    compare_model_controls,
)
from context_parallel_model_control import qualify_model_control as _qualify_model_control
from flux_sft_fixture import flux_inputs as model_inputs
from flux_sft_fixture import flux_lora as install_lora
from flux_sft_fixture import flux_transformer as build_transformer

wrap_classes = ("FluxTransformerBlock", "FluxSingleTransformerBlock")


def adapter_class():
    """Resolve the real adapter only in workers that prepare the model."""
    from flow_factory.models.flux.flux1 import Flux1Adapter

    return Flux1Adapter


def serialize_inputs(inputs):
    """Retain every fixed model operand for the cross-layout comparison."""
    return {name: value.cpu() for name, value in inputs.items()}


def qualify_model_control(tmp_path, root, options):
    """Keep the Flux lifecycle fixture's numerical-control entry point."""
    return _qualify_model_control(
        tmp_path, root, {**options, "fixture_module": "flux_cp_model_control"}
    )
