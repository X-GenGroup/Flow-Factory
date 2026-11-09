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

import inspect
from copy import deepcopy
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest
import torch
import torch.nn as nn
from accelerate import DistributedType
from diffusers.modular_pipelines.modular_pipeline import ModularPipeline

from flow_factory.models.abc import BaseAdapter
from flow_factory.models.model_bundle import ModelBundle, RoutedComponentProxy
from flow_factory.models.runtime import (
    ClassicPipelineRuntime,
    ModularPipelineRuntime,
    PseudoPipelineRuntime,
)
from flow_factory.trainers.abc import BaseTrainer


class TrackingModule(nn.Module):
    """Small module that records requested device moves."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.moves: List[str] = []

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Scale an input by the tracked parameter."""
        return value * self.weight

    def to(self, *args: Any, **kwargs: Any) -> "TrackingModule":
        """Record a device move without requiring accelerator hardware."""
        device = kwargs.get("device", args[0] if args else None)
        self.moves.append(str(device))
        return self


class DtypeTrackingModule(TrackingModule):
    """Tracking module that applies explicit dtype casts without moving devices."""

    def to(self, *args: Any, **kwargs: Any) -> "DtypeTrackingModule":
        """Apply a requested dtype and record only concrete device moves."""
        dtype = kwargs.get("dtype")
        if dtype is not None:
            self.weight.data = self.weight.data.to(dtype=dtype)
        device = kwargs.get("device", args[0] if args else None)
        if device is not None and not isinstance(device, torch.dtype):
            self.moves.append(str(device))
        return self


class SchedulerFake:
    """Small scheduler-like object used by adapter construction tests."""

    def step(self) -> None:
        """Provide the scheduler step surface."""

    def eval(self) -> None:
        """Provide evaluation mode compatibility."""

    def train(self, mode: bool = True) -> None:
        """Provide training mode compatibility."""

    def rollout(self, mode: bool = True) -> None:
        """Provide rollout mode compatibility."""

    def set_seed(self, seed: int) -> None:
        """Provide trajectory seed compatibility."""


class ClassicPipelineFake:
    """Eager pipeline-like container used by classic runtime tests."""

    def __init__(self) -> None:
        self.text_encoder = TrackingModule()
        self.text_encoder_2 = TrackingModule()
        self.transformer = TrackingModule()
        self.vae = TrackingModule()
        self.scheduler = SchedulerFake()

    @property
    def components(self) -> Dict[str, Any]:
        """Expose canonical eager components."""
        components = {
            "text_encoder": self.text_encoder,
            "text_encoder_2": self.text_encoder_2,
            "transformer": self.transformer,
            "vae": self.vae,
            "scheduler": self.scheduler,
        }
        if hasattr(self, "optional_component"):
            components["optional_component"] = self.optional_component
        return components


class OptionalTransformerPipelineFake(ClassicPipelineFake):
    """Classic pipeline with a legal absent secondary transformer."""

    def __init__(self) -> None:
        super().__init__()
        self.transformer_2 = None

    @property
    def components(self) -> Dict[str, Any]:
        """Expose the absent secondary transformer declaration."""
        return {**super().components, "transformer_2": self.transformer_2}


class CountingClassicPipelineFake(ClassicPipelineFake):
    """Classic pipeline that counts expensive component-map reconstruction."""

    def __init__(self) -> None:
        super().__init__()
        self.component_map_reads = 0

    @property
    def components(self) -> Dict[str, Any]:
        """Count component map access."""
        self.component_map_reads += 1
        return super().components


class BagelContainerFake(nn.Module):
    """Small parent module with a nested transformer alias."""

    def __init__(self) -> None:
        super().__init__()
        self.language_model = TrackingModule()
        self.moves: List[str] = []

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Route a value through the nested language model."""
        return self.language_model(value)

    def to(self, *args: Any, **kwargs: Any) -> "BagelContainerFake":
        """Record parent device movement."""
        device = kwargs.get("device", args[0] if args else None)
        self.moves.append(str(device))
        return self


class ModularPipelineFake:
    """Match pinned ModularPipeline's public spec and materialized-value APIs."""

    def __init__(self, unavailable: List[str] | None = None) -> None:
        self.pretrained_specs = {
            "text_encoder": "text encoder spec",
            "transformer": "transformer spec",
            "vae": "vae spec",
        }
        self.config_specs: Dict[str, Any] = {}
        self.unavailable = set(unavailable or [])
        self.load_calls: List[List[str]] = []

    @property
    def component_names(self) -> List[str]:
        """Expose only names currently present in ``components``."""
        return list(self.components)

    @property
    def pretrained_component_names(self) -> List[str]:
        """Expose every declared lazy from-pretrained component name."""
        return list(self.pretrained_specs)

    @property
    def config_component_names(self) -> List[str]:
        """Expose declared from-config names like pinned ModularPipeline."""
        return list(self.config_specs)

    @property
    def components(self) -> Dict[str, Any]:
        """Expose only materialized values, never the complete lazy spec table."""
        names = [*self.pretrained_component_names, *self.config_component_names]
        return {
            name: getattr(self, name) for name in names if getattr(self, name, None) is not None
        }

    def get_component_spec(self, name: str) -> Any:
        """Return a defensive copy like pinned ModularPipeline."""
        return deepcopy({**self.pretrained_specs, **self.config_specs}[name])

    def load_components(self, names: List[str]) -> None:
        """Materialize every available requested component."""
        self.load_calls.append(list(names))
        for name in names:
            if name not in self.unavailable:
                setattr(self, name, TrackingModule())


class DtypeModularPipelineFake(ModularPipelineFake):
    """Lazy modular pipeline materializing dtype-aware modules."""

    def __init__(self) -> None:
        super().__init__()
        self.config_specs["scheduler"] = "scheduler spec"

    def load_components(self, names: List[str]) -> None:
        """Materialize scheduler and dtype-aware model components."""
        self.load_calls.append(list(names))
        for name in names:
            if name in self.unavailable:
                continue
            component = SchedulerFake() if name == "scheduler" else DtypeTrackingModule()
            setattr(self, name, component)


def test_pinned_modular_pipeline_public_spec_api_shape() -> None:
    pipeline = object.__new__(ModularPipeline)
    pretrained_spec = SimpleNamespace(
        default_creation_method="from_pretrained",
        values=[],
    )
    config_spec = SimpleNamespace(default_creation_method="from_config", values=[])
    pipeline._component_specs = {
        "transformer": pretrained_spec,
        "processor": config_spec,
    }
    pipeline.processor = SimpleNamespace()

    assert pipeline.component_names == ["processor"]
    assert pipeline.pretrained_component_names == ["transformer"]
    assert pipeline.config_component_names == ["processor"]
    assert pipeline.components == {"processor": pipeline.processor}
    copied_spec = pipeline.get_component_spec("transformer")
    assert copied_spec is not pretrained_spec
    copied_spec.values.append("mutated")
    assert pretrained_spec.values == []
    assert "names" in inspect.signature(ModularPipeline.load_components).parameters


def test_classic_runtime_prefers_prepared_component_over_canonical() -> None:
    pipeline = ClassicPipelineFake()
    runtime = ClassicPipelineRuntime(pipeline)
    prepared = TrackingModule()

    runtime.set_prepared_component("transformer", prepared)

    assert runtime.get_component("transformer") is prepared
    assert runtime.get_canonical_component("transformer") is pipeline.transformer
    assert runtime.component_names == [
        "text_encoder",
        "text_encoder_2",
        "transformer",
        "vae",
    ]


def test_pseudo_runtime_uses_explicit_components_and_expands_groups() -> None:
    pipeline = SimpleNamespace()
    components = {
        "text_encoder": TrackingModule(),
        "text_encoder_2": TrackingModule(),
        "transformer": TrackingModule(),
        "transformer_2": TrackingModule(),
        "vae": TrackingModule(),
    }
    runtime = PseudoPipelineRuntime(pipeline, components)

    assert runtime.get_component("vae") is components["vae"]
    assert runtime.resolve_component_names(["text_encoders", "transformers", "vae"]) == [
        "text_encoder",
        "text_encoder_2",
        "transformer",
        "transformer_2",
        "vae",
    ]


def test_pseudo_alias_is_addressable_but_excluded_from_device_enumeration() -> None:
    bagel = BagelContainerFake()
    vae = TrackingModule()
    runtime = PseudoPipelineRuntime(
        SimpleNamespace(),
        {"bagel": bagel, "vae": vae},
        aliases={"transformer": bagel.language_model},
    )

    assert runtime.get_component("transformer") is bagel.language_model
    assert runtime.resolve_component_names() == ["bagel", "vae"]
    assert runtime.resolve_component_names("transformers") == ["transformer"]

    runtime.load_components(["bagel", "transformer", "vae"], device="stage-device")
    runtime.unload_components(["bagel", "transformer", "vae"])

    assert bagel.moves == ["stage-device", "cpu"]
    assert bagel.language_model.moves == []
    assert vae.moves == ["stage-device", "cpu"]


def test_pseudo_runtime_rejects_alias_name_collision() -> None:
    module = TrackingModule()

    with pytest.raises(ValueError, match=r"aliases.*duplicate.*transformer"):
        PseudoPipelineRuntime(
            SimpleNamespace(),
            {"transformer": module},
            aliases={"transformer": module},
        )


def test_pseudo_runtime_rejects_non_module_alias() -> None:
    with pytest.raises(TypeError, match=r"torch\.nn\.Module.*transformer.*str"):
        PseudoPipelineRuntime(
            SimpleNamespace(),
            {"bagel": TrackingModule()},
            aliases={"transformer": "not a module"},
        )


def test_modular_runtime_materializes_only_selected_names() -> None:
    pipeline = ModularPipelineFake()
    runtime = ModularPipelineRuntime(pipeline)

    assert pipeline.components == {}
    assert runtime.declared_component_names == ["text_encoder", "transformer", "vae"]

    runtime.materialize_components(["vae"])

    assert pipeline.load_calls == [["vae"]]
    assert runtime.get_component("vae") is pipeline.vae
    assert not hasattr(pipeline, "transformer")


def test_modular_runtime_uses_pinned_public_component_spec_api_shape() -> None:
    pipeline = ModularPipelineFake()
    pipeline.config_specs["scheduler"] = "scheduler config spec"
    pipeline.transformer = TrackingModule()
    runtime = ModularPipelineRuntime(pipeline)

    assert pipeline.component_names == ["transformer"]
    assert pipeline.pretrained_component_names == [
        "text_encoder",
        "transformer",
        "vae",
    ]
    assert pipeline.config_component_names == ["scheduler"]
    assert pipeline.components == {"transformer": pipeline.transformer}
    assert runtime.canonical_components == {
        "text_encoder": "text encoder spec",
        "transformer": "transformer spec",
        "vae": "vae spec",
        "scheduler": "scheduler config spec",
    }
    assert runtime.declared_component_names == [
        "scheduler",
        "text_encoder",
        "transformer",
        "vae",
    ]
    assert runtime.get_component("transformer") is pipeline.transformer


def test_modular_runtime_reports_materialization_failure_context() -> None:
    pipeline = ModularPipelineFake(unavailable=["transformer"])
    runtime = ModularPipelineRuntime(pipeline)

    with pytest.raises(
        RuntimeError,
        match=r"expected.*transformer.*received.*text_encoder.*vae",
    ):
        runtime.materialize_components(["transformer"])


def test_modular_materialize_none_preserves_all_declared_specs_lazily() -> None:
    pipeline = ModularPipelineFake()
    pipeline.pretrained_specs.update(
        {
            "scheduler": "scheduler spec",
            "tokenizer": "tokenizer spec",
            "processor": "processor spec",
        }
    )
    runtime = ModularPipelineRuntime(pipeline)

    runtime.materialize_components()

    assert pipeline.load_calls == []
    assert runtime.materialized_component_names == []
    assert not hasattr(pipeline, "scheduler")
    assert not hasattr(pipeline, "tokenizer")
    assert not hasattr(pipeline, "processor")


def test_modular_none_enumerates_only_materialized_modules_without_loading_specs() -> None:
    pipeline = ModularPipelineFake()
    pipeline.pretrained_specs.update(
        {
            "scheduler": "scheduler spec",
            "tokenizer": "tokenizer spec",
            "processor": "processor spec",
        }
    )
    pipeline.transformer = TrackingModule()
    runtime = ModularPipelineRuntime(pipeline)

    assert runtime.declared_component_names == [
        "processor",
        "scheduler",
        "text_encoder",
        "tokenizer",
        "transformer",
        "vae",
    ]
    assert runtime.resolve_component_names() == ["transformer"]

    runtime.load_components(device="stage-device")
    runtime.unload_components()

    assert pipeline.load_calls == []
    assert pipeline.transformer.moves == ["stage-device", "cpu"]


def test_classic_optional_none_is_addressable_and_skipped_by_stage_lifecycle() -> None:
    pipeline = ClassicPipelineFake()
    pipeline.optional_component = None
    pipeline.transformer_aux = TrackingModule()
    runtime = ClassicPipelineRuntime(pipeline)

    assert runtime.get_component("optional_component") is None
    assert runtime.resolve_component_names() == [
        "text_encoder",
        "text_encoder_2",
        "transformer",
        "vae",
    ]
    assert "transformer_aux" not in runtime.resolve_component_names()
    assert runtime.resolve_component_names("transformers") == [
        "transformer",
        "transformer_aux",
    ]

    runtime.load_components(["optional_component"], device="stage-device")
    runtime.unload_components(["optional_component"])

    assert pipeline.transformer_aux.moves == []


def test_runtime_rejects_unknown_unload_and_missing_device() -> None:
    runtime = ClassicPipelineRuntime(ClassicPipelineFake())

    with pytest.raises(ValueError, match=r"unknown.*missing.*received"):
        runtime.unload_components(["missing"])
    with pytest.raises(ValueError, match=r"device.*None"):
        runtime.load_components(["vae"], device=None)


def test_runtime_uses_unambiguous_private_device_lifecycle_name() -> None:
    runtime = ClassicPipelineRuntime(ClassicPipelineFake())

    assert runtime._owns_device_lifecycle("vae")
    assert not hasattr(runtime, "_should_manage_device")


def test_classic_materialized_lookup_avoids_component_map_reconstruction() -> None:
    pipeline = CountingClassicPipelineFake()
    runtime = ClassicPipelineRuntime(pipeline)

    assert runtime.get_component("transformer") is pipeline.transformer
    assert runtime.get_canonical_component("vae") is pipeline.vae
    assert pipeline.component_map_reads == 0


def test_stage_load_and_unload_move_only_non_prepared_modules() -> None:
    pipeline = ClassicPipelineFake()
    runtime = ClassicPipelineRuntime(pipeline)
    prepared = TrackingModule()
    runtime.set_prepared_component("transformer", prepared)

    runtime.load_components(["transformer", "vae"], device="stage-device")
    runtime.unload_components(["transformer", "vae"])

    assert prepared.moves == []
    assert pipeline.transformer.moves == []
    assert pipeline.vae.moves == ["stage-device", "cpu"]


def test_prepared_modular_component_never_loads_or_moves_canonical_component() -> None:
    pipeline = ModularPipelineFake()
    runtime = ModularPipelineRuntime(pipeline)
    prepared = TrackingModule()
    runtime.set_prepared_component("transformer", prepared)

    runtime.load_components(["transformer"], device="stage-device")
    runtime.unload_components(["transformer"])

    assert pipeline.load_calls == []
    assert prepared.moves == []
    assert not hasattr(pipeline, "transformer")


def test_component_override_vocabulary_preserves_prepared_compatibility_aliases() -> None:
    pipeline = ClassicPipelineFake()
    runtime = ClassicPipelineRuntime(pipeline)
    replacement = TrackingModule()

    runtime.set_component_override("transformer", replacement)

    assert runtime.get_component("transformer") is replacement
    assert runtime.has_component_override("transformer")
    assert runtime.is_prepared("transformer")
    assert runtime.override_components is runtime.prepared_components

    runtime.load_components(["transformer"], device="stage-device")
    runtime.unload_components(["transformer"])

    assert replacement.moves == []
    assert pipeline.transformer.moves == []


class AcceleratorFake:
    """Minimal accelerator surface used during adapter construction."""

    device = torch.device("cpu")
    distributed_type = DistributedType.NO
    is_fsdp2 = False
    is_main_process = True
    mixed_precision = "no"

    def unwrap_model(self, module: nn.Module) -> nn.Module:
        """Return an unwrapped module unchanged."""
        return module


class FSDP2AcceleratorFake(AcceleratorFake):
    """Expose the FSDP2 mixed-precision policy boundary."""

    distributed_type = DistributedType.FSDP
    is_fsdp2 = True
    mixed_precision = "bf16"


class ExistingStyleAdapterFake(BaseAdapter):
    """Adapter implementing only the existing four abstract methods."""

    def load_pipeline(self) -> ClassicPipelineFake:
        """Return a small eager pipeline."""
        return ClassicPipelineFake()

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Return fake latents unchanged."""
        return latents

    def inference(self, **kwargs: Any) -> List[Any]:
        """Return no generated samples."""
        return []

    def forward(self, **kwargs: Any) -> Any:
        """Return no fake model output."""
        return None


class OptionalTransformerAdapterFake(ExistingStyleAdapterFake):
    """Existing-style adapter with an absent optional transformer."""

    def load_pipeline(self) -> OptionalTransformerPipelineFake:
        """Return a classic pipeline with ``transformer_2=None``."""
        return OptionalTransformerPipelineFake()

    @property
    def transformer_2(self) -> Any:
        """Expose the optional secondary transformer like a real adapter."""
        return self.get_component("transformer_2")


class ModularAdapterFake(ExistingStyleAdapterFake):
    """Adapter fake that selects a lazy modular runtime."""

    def load_pipeline(self) -> ModularPipelineFake:
        """Return a lazy pipeline with a declared scheduler spec."""
        pipeline = ModularPipelineFake()
        pipeline.config_specs["scheduler"] = "scheduler spec"
        return pipeline

    def build_component_runtime(self) -> ModularPipelineRuntime:
        """Build the lazy runtime used by this adapter."""
        return ModularPipelineRuntime(self.load_pipeline())


class SecondaryTransformerPipelineFake(ModularPipelineFake):
    """Lazy pipeline declaring a transformer the adapter never exposes as an attribute."""

    def __init__(self) -> None:
        super().__init__()
        self.pretrained_specs["transformer_ref"] = "reference transformer spec"
        self.config_specs["scheduler"] = "scheduler spec"

    def load_components(self, names: List[str]) -> None:
        """Materialize the scheduler and every requested transformer."""
        self.load_calls.append(list(names))
        for name in names:
            if name in self.unavailable:
                continue
            setattr(self, name, SchedulerFake() if name == "scheduler" else TrackingModule())


class SecondaryTransformerAdapterFake(ExistingStyleAdapterFake):
    """Lazy adapter whose second transformer exists only in the runtime declaration."""

    def load_pipeline(self) -> SecondaryTransformerPipelineFake:
        """Return a lazy pipeline declaring a reference transformer."""
        return SecondaryTransformerPipelineFake()

    def build_component_runtime(self) -> ModularPipelineRuntime:
        """Build the lazy runtime used by this adapter."""
        return ModularPipelineRuntime(self.load_pipeline())


class DtypeModularAdapterFake(ModularAdapterFake):
    """Lazy adapter used to verify post-init component dtype policy."""

    def load_pipeline(self) -> DtypeModularPipelineFake:
        """Return a dtype-aware lazy pipeline."""
        return DtypeModularPipelineFake()

    def _freeze_components(self) -> None:
        """Freeze only the materialized target, matching the H3 lazy adapter."""
        self._freeze_component("transformer", trainable_modules="all")


def _adapter_config(
    frozen_parameters_dtype: Any = None,
) -> SimpleNamespace:
    model_args = SimpleNamespace(
        resume_path=None,
        resume_type=None,
        finetune_type="full",
        target_components=["transformer"],
        target_modules="all",
        trainable_parameters_dtype=torch.float32,
        frozen_parameters_dtype=frozen_parameters_dtype,
    )
    training_args = SimpleNamespace(
        enable_gradient_checkpointing=False,
        latent_storage_dtype=None,
    )
    return SimpleNamespace(
        model_args=model_args,
        training_args=training_args,
        eval_args=SimpleNamespace(),
        scheduler_args=SimpleNamespace(),
        mixed_precision="no",
    )


def test_base_adapter_default_runtime_preserves_existing_subclass_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "flow_factory.models.abc._load_scheduler",
        lambda pipeline_scheduler, scheduler_args: pipeline_scheduler,
    )

    adapter = ExistingStyleAdapterFake(_adapter_config(), AcceleratorFake())

    assert isinstance(adapter.component_runtime, ClassicPipelineRuntime)
    assert adapter.pipeline is adapter.component_runtime.pipeline
    assert adapter._components is adapter.component_runtime.prepared_components
    assert adapter.get_component("transformer") is adapter.pipeline.transformer
    assert adapter.scheduler_group.names == ("latent",)
    assert adapter.scheduler_group.primary is adapter.scheduler


def test_gradient_checkpointing_rejects_unknown_target_component(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "flow_factory.models.abc._load_scheduler",
        lambda pipeline_scheduler, scheduler_args: pipeline_scheduler,
    )
    adapter = ExistingStyleAdapterFake(_adapter_config(), AcceleratorFake())
    adapter.model_args.target_components = ["missing"]

    with pytest.raises(ValueError, match="missing.*transformer"):
        adapter.enable_gradient_checkpointing()


def test_base_adapter_constructs_with_optional_transformer_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "flow_factory.models.abc._load_scheduler",
        lambda pipeline_scheduler, scheduler_args: pipeline_scheduler,
    )

    adapter = OptionalTransformerAdapterFake(_adapter_config(), AcceleratorFake())

    assert adapter.pipeline.transformer_2 is None
    assert adapter.transformer_names == ["transformer"]


def test_base_adapter_materializes_declared_modular_scheduler_before_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = SchedulerFake()

    def load_components(self: ModularPipelineFake, names: List[str]) -> None:
        self.load_calls.append(list(names))
        for name in names:
            setattr(self, name, scheduler if name == "scheduler" else TrackingModule())

    monkeypatch.setattr(ModularPipelineFake, "load_components", load_components)
    monkeypatch.setattr(
        "flow_factory.models.abc._load_scheduler",
        lambda pipeline_scheduler, scheduler_args: pipeline_scheduler,
    )

    adapter = ModularAdapterFake(_adapter_config(), AcceleratorFake())

    assert adapter.scheduler is scheduler
    assert adapter.pipeline.load_calls[0] == ["scheduler"]


def test_stage_materialization_applies_explicit_frozen_dtype_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "flow_factory.models.abc._load_scheduler",
        lambda pipeline_scheduler, scheduler_args: pipeline_scheduler,
    )
    adapter = DtypeModularAdapterFake(
        _adapter_config(frozen_parameters_dtype=torch.float16),
        AcceleratorFake(),
    )

    assert not hasattr(adapter.pipeline, "text_encoder")

    adapter.on_load_components(["text_encoder"], device=torch.device("cpu"))

    assert adapter.pipeline.text_encoder.weight.dtype == torch.float16
    assert not adapter.pipeline.text_encoder.weight.requires_grad
    assert not adapter.pipeline.text_encoder.training


def test_component_frozen_dtype_policy_applies_to_lazy_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "flow_factory.models.abc._load_scheduler",
        lambda pipeline_scheduler, scheduler_args: pipeline_scheduler,
    )
    adapter = DtypeModularAdapterFake(
        _adapter_config(
            frozen_parameters_dtype={
                "default": torch.bfloat16,
                "text_encoder": torch.float16,
                "vae": torch.float32,
            }
        ),
        AcceleratorFake(),
    )

    adapter.on_load_components(["text_encoder", "vae"], device=torch.device("cpu"))

    assert adapter.pipeline.text_encoder.weight.dtype == torch.float16
    assert adapter.pipeline.vae.weight.dtype == torch.float32
    assert not adapter.pipeline.text_encoder.weight.requires_grad
    assert not adapter.pipeline.vae.weight.requires_grad


def test_concrete_frozen_dtype_overrides_group_and_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "flow_factory.models.abc._load_scheduler",
        lambda pipeline_scheduler, scheduler_args: pipeline_scheduler,
    )
    adapter = DtypeModularAdapterFake(
        _adapter_config(
            frozen_parameters_dtype={
                "default": torch.float32,
                "transformers": torch.float16,
                "transformer": torch.bfloat16,
            }
        ),
        AcceleratorFake(),
    )

    assert adapter._frozen_dtype_for_component("transformer") == torch.bfloat16
    assert adapter._frozen_dtype_for_component("text_encoder") == torch.float32
    assert adapter._frozen_dtype_for_component("vae") == torch.float32


def test_component_frozen_dtype_policy_rejects_unknown_selector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "flow_factory.models.abc._load_scheduler",
        lambda pipeline_scheduler, scheduler_args: pipeline_scheduler,
    )

    with pytest.raises(ValueError, match=r"missing.*transformer.*vae"):
        DtypeModularAdapterFake(
            _adapter_config(frozen_parameters_dtype={"missing": torch.float16}),
            AcceleratorFake(),
        )


def test_fsdp2_keeps_target_original_dtype_uniform_and_applies_frozen_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "flow_factory.models.abc._load_scheduler",
        lambda pipeline_scheduler, scheduler_args: pipeline_scheduler,
    )
    adapter = DtypeModularAdapterFake(
        _adapter_config(
            frozen_parameters_dtype={
                "default": torch.float16,
                "vae": torch.float32,
            }
        ),
        FSDP2AcceleratorFake(),
    )

    adapter.on_load_components(["text_encoder", "vae"], device=torch.device("cpu"))

    assert adapter.pipeline.transformer.weight.dtype == torch.float32
    assert adapter.pipeline.text_encoder.weight.dtype == torch.float16
    assert adapter.pipeline.vae.weight.dtype == torch.float32


def test_bundle_proxy_installation_resolves_through_adapter_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "flow_factory.models.abc._load_scheduler",
        lambda pipeline_scheduler, scheduler_args: pipeline_scheduler,
    )
    adapter = ExistingStyleAdapterFake(_adapter_config(), AcceleratorFake())
    canonical = adapter.get_component("transformer")
    bundle = ModelBundle({"transformer": canonical})
    proxy = RoutedComponentProxy(bundle, "transformer", canonical)

    adapter.set_component("transformer", proxy)

    assert adapter.get_component("transformer") is proxy
    assert adapter.get_component_unwrapped("transformer") is canonical
    assert torch.equal(
        adapter.get_component("transformer")(torch.tensor([2.0])), torch.tensor([2.0])
    )


class LifecycleAdapterFake:
    """Adapter-like fake that records public lifecycle override calls."""

    preprocessing_modules = ["text_encoders", "vae"]
    inference_modules = ["transformer", "vae"]

    def __init__(self) -> None:
        self.calls: List[Any] = []
        self.preprocess_func = lambda **kwargs: kwargs
        self.fsdp_cpu_efficient_loading = False

    def on_load_components(self, components: Any, device: Any) -> None:
        """Record public load routing."""
        self.calls.append(("load", components, device))

    def off_load_components(self, components: Any) -> None:
        """Record public unload routing."""
        self.calls.append(("unload", components))

    def _resolve_component_names(self, components: Any = None) -> List[str]:
        """Resolve the groups needed by the trainer regression test."""
        self.calls.append(("resolve", components))
        return ["transformer", "vae"]


class TrainerAcceleratorFake:
    """Minimal trainer accelerator fake."""

    device = torch.device("cpu")
    distributed_type = DistributedType.NO
    num_processes = 1
    process_index = 0

    def wait_for_everyone(self) -> None:
        """Record no-op synchronization."""


class LoadCoordinatorFake:
    def __init__(self, adapter: LifecycleAdapterFake, *, record_finalize: bool = False):
        self.adapter = adapter
        self.record_finalize = record_finalize

    def load_components(self, components: Any, *, device: torch.device) -> None:
        self.adapter.on_load_components(components=components, device=device)
        if self.record_finalize:
            self.adapter.calls.append(("synchronize", components))


def _train_loader_missing_its_cache(**kwargs: Any) -> Any:
    """Request the preprocessing modules as ``_create_or_load_dataset`` does on a miss."""
    from flow_factory.data_utils.loader import _PREPROCESSING_MODULE_LOADER

    load = _PREPROCESSING_MODULE_LOADER.get()
    load()
    load()
    return "train-loader", {}


def test_trainer_preprocessing_routes_through_adapter_public_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = LifecycleAdapterFake()
    trainer = SimpleNamespace(
        adapter=adapter,
        accelerator=TrainerAcceleratorFake(),
        config=SimpleNamespace(data_args=SimpleNamespace(eval_datasets=[])),
        load_coordinator=LoadCoordinatorFake(adapter),
    )
    monkeypatch.setattr(
        "flow_factory.trainers.abc.get_train_dataloader", _train_loader_missing_its_cache
    )
    monkeypatch.setattr("flow_factory.trainers.abc.get_eval_dataloaders", lambda **kwargs: {})

    BaseTrainer._init_dataloader(trainer)

    assert adapter.calls == [
        ("load", adapter.preprocessing_modules, trainer.accelerator.device),
        ("unload", adapter.preprocessing_modules),
    ]


def test_trainer_skips_preprocessing_modules_when_every_cache_hits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = LifecycleAdapterFake()
    trainer = SimpleNamespace(
        adapter=adapter,
        accelerator=TrainerAcceleratorFake(),
        config=SimpleNamespace(data_args=SimpleNamespace(eval_datasets=[])),
        load_coordinator=LoadCoordinatorFake(adapter),
    )
    monkeypatch.setattr(
        "flow_factory.trainers.abc.get_train_dataloader",
        lambda **kwargs: ("train-loader", {}),
    )
    monkeypatch.setattr("flow_factory.trainers.abc.get_eval_dataloaders", lambda **kwargs: {})

    assert BaseTrainer._init_dataloader(trainer) == ("train-loader", {})
    assert adapter.calls == []


def test_trainer_finalizes_lazy_frozen_components_after_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = LifecycleAdapterFake()
    trainer = SimpleNamespace(
        adapter=adapter,
        accelerator=TrainerAcceleratorFake(),
        config=SimpleNamespace(data_args=SimpleNamespace(eval_datasets=[])),
        load_coordinator=LoadCoordinatorFake(adapter, record_finalize=True),
    )
    monkeypatch.setattr(
        "flow_factory.trainers.abc.get_train_dataloader", _train_loader_missing_its_cache
    )
    monkeypatch.setattr("flow_factory.trainers.abc.get_eval_dataloaders", lambda **kwargs: {})

    BaseTrainer._init_dataloader(trainer)

    assert adapter.calls == [
        ("load", adapter.preprocessing_modules, trainer.accelerator.device),
        ("synchronize", adapter.preprocessing_modules),
        ("unload", adapter.preprocessing_modules),
    ]


def test_trainer_inference_load_routes_through_adapter_public_lifecycle() -> None:
    adapter = LifecycleAdapterFake()
    trainer = SimpleNamespace(
        adapter=adapter,
        accelerator=TrainerAcceleratorFake(),
        config=SimpleNamespace(data_args=SimpleNamespace(enable_preprocess=True)),
        load_coordinator=LoadCoordinatorFake(adapter),
    )

    BaseTrainer._load_inference_components(trainer, trainable_module_names=[])

    assert adapter.calls == [
        ("resolve", adapter.inference_modules),
        ("load", ["transformer", "vae"], trainer.accelerator.device),
    ]


def test_trainer_finalizes_lazy_inference_components_after_materialization() -> None:
    adapter = LifecycleAdapterFake()
    trainer = SimpleNamespace(
        adapter=adapter,
        accelerator=TrainerAcceleratorFake(),
        config=SimpleNamespace(data_args=SimpleNamespace(enable_preprocess=True)),
        load_coordinator=LoadCoordinatorFake(adapter, record_finalize=True),
    )

    BaseTrainer._load_inference_components(trainer, trainable_module_names=[])

    assert adapter.calls == [
        ("resolve", adapter.inference_modules),
        ("load", ["transformer", "vae"], trainer.accelerator.device),
        ("synchronize", ["transformer", "vae"]),
    ]


def test_declared_component_owns_parameters_without_being_an_adapter_attribute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lazy target component must reach the optimizer even with no matching attribute."""
    monkeypatch.setattr(
        "flow_factory.models.abc._load_scheduler",
        lambda pipeline_scheduler, scheduler_args: pipeline_scheduler,
    )
    config = _adapter_config()
    config.model_args.target_components = ["transformer_ref"]

    adapter = SecondaryTransformerAdapterFake(config, AcceleratorFake())

    assert not hasattr(adapter, "transformer_ref")
    assert adapter.has_component("transformer_ref")
    assert len(adapter.get_trainable_parameters()) == 1


def test_require_component_names_the_declared_component_the_runtime_cannot_provide(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A declared-but-unset optional component fails with its name, not an AttributeError."""
    monkeypatch.setattr(
        "flow_factory.models.abc._load_scheduler",
        lambda pipeline_scheduler, scheduler_args: pipeline_scheduler,
    )
    adapter = OptionalTransformerAdapterFake(_adapter_config(), AcceleratorFake())

    assert adapter.has_component("transformer_2")
    assert adapter.get_component("transformer_2") is None
    with pytest.raises(ValueError, match="transformer_2"):
        adapter._require_component("transformer_2")


def test_component_override_rejects_an_undeclared_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An override under an undeclared name is unreachable, so it is refused."""
    monkeypatch.setattr(
        "flow_factory.models.abc._load_scheduler",
        lambda pipeline_scheduler, scheduler_args: pipeline_scheduler,
    )
    adapter = ExistingStyleAdapterFake(_adapter_config(), AcceleratorFake())

    with pytest.raises(ValueError, match="transformer_ref"):
        adapter.set_component("transformer_ref", TrackingModule())


def test_reduction_wrappers_reject_subclass_overrides() -> None:
    """The validated reduction wrappers own their contract; only the hooks are overridable."""
    with pytest.raises(TypeError, match="reduce_latent_values"):

        class OverridingReductionAdapterFake(ExistingStyleAdapterFake):
            def reduce_latent_values(self, values: Any, **kwargs: Any) -> Any:
                """Bypass the validated wrapper."""
                return values

    with pytest.raises(TypeError, match="reduce_component_latent_values"):

        class OverridingComponentReductionAdapterFake(ExistingStyleAdapterFake):
            def reduce_component_latent_values(self, values: Any, **kwargs: Any) -> Any:
                """Bypass the validated per-component wrapper."""
                return values
