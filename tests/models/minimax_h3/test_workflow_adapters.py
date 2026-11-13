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

from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace
from typing import Any, ClassVar, Dict, List

import pytest
import torch
import torch.nn as nn
from accelerate import DistributedType

from flow_factory.models.abc import BaseAdapter
from flow_factory.models.minimax_h3._chunking import (
    H3_MAX_ATTENTION_NORM_TOKENS,
    H3_MAX_FEED_FORWARD_TOKENS,
)
from flow_factory.models.minimax_h3.adapters import (
    MiniMaxH3FL2VAAdapter,
    MiniMaxH3Ref2VAAdapter,
    MiniMaxH3T2VAAdapter,
)
from flow_factory.models.minimax_h3.workflow import load_h3_workflow_pipeline
from flow_factory.models.runtime import ModularPipelineRuntime
from flow_factory.samples import ComponentTimes, LatentState
from flow_factory.scheduler import MiniMaxH3SDEScheduler


class UpstreamSchedulerFake:
    """Represent the lazy upstream scheduler replaced by Flow-Factory."""


def test_only_ref2va_requests_fsdp2_default_stream_unshard() -> None:
    assert MiniMaxH3Ref2VAAdapter.fsdp2_use_default_stream_unshard
    assert MiniMaxH3Ref2VAAdapter.fsdp2_use_in_forward_activation_checkpointing
    assert MiniMaxH3Ref2VAAdapter.fsdp2_disable_backward_prefetch
    assert MiniMaxH3Ref2VAAdapter.fsdp2_additional_wrap_module_names == (
        "_ChunkedFeedForward",
        "MiniMaxH3AdaLayerNormModulation",
        "MiniMaxH3Attention",
    )
    assert not MiniMaxH3T2VAAdapter.fsdp2_use_default_stream_unshard
    assert not MiniMaxH3T2VAAdapter.fsdp2_use_in_forward_activation_checkpointing
    assert not MiniMaxH3T2VAAdapter.fsdp2_disable_backward_prefetch
    assert not MiniMaxH3T2VAAdapter.fsdp2_additional_wrap_module_names
    assert not MiniMaxH3FL2VAAdapter.fsdp2_use_default_stream_unshard
    assert not MiniMaxH3FL2VAAdapter.fsdp2_use_in_forward_activation_checkpointing
    assert not MiniMaxH3FL2VAAdapter.fsdp2_disable_backward_prefetch
    assert not MiniMaxH3FL2VAAdapter.fsdp2_additional_wrap_module_names


class SwiGLU(nn.Module):
    """Match the upstream activation class name without adding parameters."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value


class FeedForwardFake(nn.Module):
    """Expose the parameter-free upstream ``ff.net`` module structure."""

    def __init__(self) -> None:
        super().__init__()
        output = nn.Linear(1, 1, bias=False)
        output.weight = None
        self.net = nn.ModuleList([SwiGLU(), nn.Dropout(0.0), output])

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            value = module(value)
        return value


class TransformerBlockFake(nn.Module):
    """Expose one feed-forward child without adding trainable parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.attn = nn.Module()
        self.attn.norm_q = nn.RMSNorm(1, elementwise_affine=False)
        self.attn.norm_k = nn.RMSNorm(1, elementwise_affine=False)
        self.ff = FeedForwardFake()


class TransformerFake(nn.Module):
    """Provide one parameter for BaseAdapter freeze and precision setup."""

    loaded_paths: ClassVar[List[str]] = []

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.token_refiner = nn.Module()
        self.token_refiner.refiner_blocks = nn.ModuleList([TransformerBlockFake()])
        self.transformer_blocks = nn.ModuleList([TransformerBlockFake()])
        self.gradient_checkpointing_calls = 0

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Scale an input by the trainable parameter."""
        return value * self.weight

    def enable_gradient_checkpointing(self) -> None:
        """Record activation of the transformer capability."""
        self.gradient_checkpointing_calls += 1

    @classmethod
    def from_pretrained(cls, path: str) -> "TransformerFake":
        """Build a replacement module while recording its checkpoint path."""
        cls.loaded_paths.append(path)
        replacement = cls()
        replacement.weight.data.fill_(7.0)
        return replacement


class PeftModelEquivalentFake(TransformerFake):
    """Provide the adapter-disable context used by LoRA reference evaluation."""

    def __init__(self) -> None:
        super().__init__()
        self.disable_adapter_events: List[str] = []

    @contextmanager
    def disable_adapter(self) -> Any:
        """Record adapter-disable context entry and exit."""
        self.disable_adapter_events.append("enter")
        try:
            yield
        finally:
            self.disable_adapter_events.append("exit")


class WorkflowPipelineFake:
    """Match released ModularPipeline spec and materialized-value separation."""

    calls: ClassVar[List[tuple[str, str]]] = []
    component_overrides: ClassVar[Dict[str, Any]] = {}

    def __init__(self, workflow: str) -> None:
        transformer_name = "transformer_ref" if workflow == "ref2va" else "transformer"
        self.workflow = workflow
        self.pretrained_specs = {
            "scheduler": "scheduler spec",
            "text_encoder": "text encoder spec",
            "tokenizer": "tokenizer spec",
            "vae": "video vae spec",
            "audio_vae": "audio vae spec",
            transformer_name: "transformer spec",
            "unrelated": "must stay lazy",
            **self.component_overrides,
        }
        self._component_specs = {
            name: SimpleNamespace(
                pretrained_model_name_or_path="MiniMaxAI/MiniMax-H3",
                revision="main",
            )
            for name in self.pretrained_specs
        }
        self.config_specs = {
            "processor": "processor spec",
            "image_processor": "image processor spec",
            "video_processor": "video processor spec",
        }
        for name in self.config_specs:
            setattr(self, name, SimpleNamespace())
        self.load_calls: List[List[str]] = []
        self.load_kwargs: List[Dict[str, Any]] = []

    @property
    def component_names(self) -> List[str]:
        return list(self.components)

    @property
    def pretrained_component_names(self) -> List[str]:
        return list(self.pretrained_specs)

    @property
    def config_component_names(self) -> List[str]:
        return list(self.config_specs)

    @property
    def components(self) -> Dict[str, Any]:
        names = [*self.pretrained_component_names, *self.config_component_names]
        return {
            name: getattr(self, name) for name in names if getattr(self, name, None) is not None
        }

    def get_component_spec(self, name: str) -> Any:
        return deepcopy({**self.pretrained_specs, **self.config_specs}[name])

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, *, workflow: str) -> "WorkflowPipelineFake":
        cls.calls.append((model_name_or_path, workflow))
        return cls(workflow)

    def load_components(self, names: List[str], **kwargs: Any) -> None:
        self.load_calls.append(list(names))
        self.load_kwargs.append(dict(kwargs))
        for name in names:
            if name == "scheduler":
                setattr(self, name, UpstreamSchedulerFake())
            elif name in ("transformer", "transformer_ref"):
                setattr(self, name, TransformerFake())
            else:
                setattr(self, name, SimpleNamespace())


class AcceleratorFake:
    """Provide the BaseAdapter construction surface without distributed setup."""

    device = torch.device("cpu")
    distributed_type = DistributedType.NO
    is_fsdp2 = False
    is_main_process = True
    mixed_precision = "no"

    def unwrap_model(self, module: nn.Module) -> nn.Module:
        return module

    def wait_for_everyone(self) -> None:
        """Provide the checkpoint synchronization surface."""


def _config(target_components: List[str]) -> SimpleNamespace:
    return SimpleNamespace(
        parallelism_args=SimpleNamespace(
            effective_context_parallel_size=1, materializes_targets_sharded=False
        ),
        model_args=SimpleNamespace(
            model_name_or_path="MiniMaxAI/MiniMax-H3",
            resume_path=None,
            resume_type=None,
            finetune_type="full",
            target_components=target_components,
            target_modules="all",
            trainable_parameters_dtype=torch.float32,
            frozen_parameters_dtype=None,
        ),
        training_args=SimpleNamespace(
            enable_gradient_checkpointing=False,
            latent_storage_dtype=None,
            num_inference_steps=4,
        ),
        eval_args=SimpleNamespace(),
        scheduler_args=SimpleNamespace(
            dynamics_type="Flow-SDE",
            noise_level=0.7,
            sde_steps=[0, 2],
            num_sde_steps=1,
            replay_steps="auto",
            seed=17,
        ),
        mixed_precision="no",
    )


@pytest.fixture(autouse=True)
def _fake_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    WorkflowPipelineFake.calls.clear()
    WorkflowPipelineFake.component_overrides = {}
    TransformerFake.loaded_paths.clear()
    monkeypatch.setattr(
        "flow_factory.models.minimax_h3.workflow.require_minimax_h3_support",
        lambda: SimpleNamespace(
            ModularPipeline=WorkflowPipelineFake,
        ),
    )


def test_workflow_loader_preserves_hub_source() -> None:
    model_name_or_path = "MiniMaxAI/MiniMax-H3"
    pipeline = load_h3_workflow_pipeline(model_name_or_path, workflow="t2va")

    assert isinstance(pipeline, WorkflowPipelineFake)
    assert WorkflowPipelineFake.calls == [(model_name_or_path, "t2va")]
    assert all(
        spec.pretrained_model_name_or_path == model_name_or_path and spec.revision == "main"
        for spec in pipeline._component_specs.values()
    )


def test_workflow_loader_rebinds_component_sources_for_local_snapshot(tmp_path) -> None:
    pipeline = load_h3_workflow_pipeline(str(tmp_path), workflow="t2va")

    assert isinstance(pipeline, WorkflowPipelineFake)
    assert WorkflowPipelineFake.calls == [(str(tmp_path), "t2va")]
    assert all(
        spec.pretrained_model_name_or_path == str(tmp_path) and spec.revision is None
        for spec in pipeline._component_specs.values()
    )


@pytest.mark.parametrize(
    ("adapter_class", "workflow", "transformer_name", "preprocessing_names"),
    [
        (
            MiniMaxH3T2VAAdapter,
            "t2va",
            "transformer",
            ["text_encoder", "tokenizer", "processor"],
        ),
        (
            MiniMaxH3FL2VAAdapter,
            "fl2va",
            "transformer",
            ["image_processor", "text_encoder", "tokenizer", "processor", "vae"],
        ),
        (
            MiniMaxH3Ref2VAAdapter,
            "ref2va",
            "transformer_ref",
            [
                "image_processor",
                "text_encoder",
                "tokenizer",
                "processor",
                "vae",
                "audio_vae",
            ],
        ),
    ],
)
def test_workflow_adapter_loads_pruned_runtime_and_exact_setup_components(
    adapter_class: type[BaseAdapter],
    workflow: str,
    transformer_name: str,
    preprocessing_names: List[str],
) -> None:
    adapter = adapter_class(_config([transformer_name]), AcceleratorFake())

    assert BaseAdapter in adapter_class.__bases__
    assert not [
        base
        for base in adapter_class.__mro__[1:]
        if issubclass(base, BaseAdapter) and base is not BaseAdapter
    ]
    assert WorkflowPipelineFake.calls == [("MiniMaxAI/MiniMax-H3", workflow)]
    assert isinstance(adapter.component_runtime, ModularPipelineRuntime)
    assert adapter.pipeline.component_names == list(adapter.pipeline.components)
    assert transformer_name in adapter.pipeline.pretrained_component_names
    assert "unrelated" in adapter.pipeline.pretrained_component_names
    assert set(adapter.pipeline.components) == {
        transformer_name,
        "scheduler",
        "processor",
        "image_processor",
        "video_processor",
    }
    assert adapter.pipeline.load_calls == [[transformer_name], ["scheduler"]]
    assert adapter.pipeline.load_kwargs == [
        {"dtype": torch.bfloat16},
        {"dtype": torch.bfloat16},
    ]
    assert len(adapter.scheduler.timesteps) == 4
    assert len(adapter.audio_scheduler.timesteps) == 4
    assert not hasattr(adapter.pipeline, "unrelated")
    assert transformer_name in adapter.component_runtime.materialized_component_names
    transformer = adapter.get_component(transformer_name)
    assert H3_MAX_FEED_FORWARD_TOKENS == 1024
    assert H3_MAX_ATTENTION_NORM_TOKENS == 1024
    configured_blocks = [
        *transformer.token_refiner.refiner_blocks,
        *transformer.transformer_blocks,
    ]
    assert all(
        getattr(block.ff, "max_tokens", None) == H3_MAX_FEED_FORWARD_TOKENS
        for block in configured_blocks
    )
    assert all(
        getattr(block.attn.norm_q, "max_tokens", None) == H3_MAX_ATTENTION_NORM_TOKENS
        and getattr(block.attn.norm_k, "max_tokens", None) == H3_MAX_ATTENTION_NORM_TOKENS
        for block in configured_blocks
    )
    opposite = "transformer_ref" if transformer_name == "transformer" else "transformer"
    assert opposite not in adapter.component_runtime.declared_component_names

    adapter.on_load_components(adapter.preprocessing_modules, device="cpu")

    assert adapter.preprocessing_modules == preprocessing_names
    expected_lazy_names = [
        name for name in preprocessing_names if name not in adapter.pipeline.config_component_names
    ]
    assert adapter.pipeline.load_calls[-1] == expected_lazy_names
    assert adapter.pipeline.load_kwargs[-1] == {"dtype": torch.bfloat16}
    assert not hasattr(adapter.pipeline, "unrelated")


def test_workflow_adapter_can_disable_default_load_dtype_manifest() -> None:
    config = _config(["transformer"])
    config.model_args.component_load_dtypes = {"default": None}

    adapter = MiniMaxH3T2VAAdapter(config, AcceleratorFake())

    assert adapter.pipeline.load_kwargs == [{}, {}]


_SHARED_H3_ADAPTER_METHODS = (
    "load_pipeline",
    "build_component_runtime",
    "load_scheduler",
    "build_scheduler_group",
    "_init_target_module_map",
    "_freeze_components",
    "preprocess_func",
    "build_training_component_times",
    "add_forward_process_noise",
    "apply_forward_process_noise",
    "decode_latents",
    "empty_decoded_media",
    "inference",
    "_forward_state",
    "forward",
)


@pytest.mark.parametrize("method_name", _SHARED_H3_ADAPTER_METHODS)
def test_workflow_adapters_share_one_implementation_per_method(method_name: str) -> None:
    implementations = {
        getattr(adapter_class, method_name)
        for adapter_class in (
            MiniMaxH3T2VAAdapter,
            MiniMaxH3FL2VAAdapter,
            MiniMaxH3Ref2VAAdapter,
        )
    }

    assert len(implementations) == 1


@pytest.mark.parametrize(
    "adapter_class",
    (MiniMaxH3T2VAAdapter, MiniMaxH3FL2VAAdapter, MiniMaxH3Ref2VAAdapter),
)
def test_empty_decoded_media_keeps_h3_return_structure(adapter_class) -> None:
    adapter = object.__new__(adapter_class)
    assert adapter.empty_decoded_media(2) == (
        [None, None],
        [None, None],
        None,
    )


@pytest.mark.parametrize(
    ("adapter_class", "transformer_name"),
    [
        (MiniMaxH3T2VAAdapter, "transformer"),
        (MiniMaxH3FL2VAAdapter, "transformer"),
        (MiniMaxH3Ref2VAAdapter, "transformer_ref"),
    ],
)
def test_forward_state_routes_through_the_public_forward(
    monkeypatch: pytest.MonkeyPatch,
    adapter_class: type[BaseAdapter],
    transformer_name: str,
) -> None:
    adapter = adapter_class(_config([transformer_name]), AcceleratorFake())
    state = LatentState({"video": torch.ones(1, 1, 96), "audio": torch.ones(1, 1, 32)})
    times = ComponentTimes(
        timestep={"video": torch.tensor([1000.0]), "audio": torch.tensor([1000.0])},
        next_timestep={"video": torch.tensor([0.0]), "audio": torch.tensor([0.0])},
    )
    observed: List[Dict[str, Any]] = []
    monkeypatch.setattr(
        type(adapter),
        "forward",
        lambda self, **kwargs: observed.append(kwargs) or "step-output",
    )

    result = adapter._forward_state(
        batch=SimpleNamespace(),
        state=state,
        times=times,
        next_state=None,
        compute_log_prob=True,
        return_fields=("next_latents",),
        noise_level=0.7,
        forward_kwargs={
            "prompt_embeds": torch.zeros(1, 2, 4),
            "condition_prefixes": {
                "video": torch.zeros(1, 0, 96),
                "audio": torch.zeros(1, 0, 32),
            },
            "layout": {
                "video_indices": torch.arange(1),
                "audio_indices": torch.arange(1, 2),
                "text_indices": torch.arange(2, 4),
                "num_condition_video_rows": 0,
                "num_condition_audio_rows": 0,
            },
        },
    )

    assert result == "step-output"
    assert len(observed) == 1
    assert observed[0]["state"] is state
    assert observed[0]["times"] is times
    assert observed[0]["compute_log_prob"] is True
    assert observed[0]["return_fields"] == ("next_latents",)
    assert observed[0]["noise_level"] == 0.7
    assert "batch" not in observed[0]


def test_ref2va_optimizer_uses_canonical_runtime_component() -> None:
    adapter = MiniMaxH3Ref2VAAdapter(_config(["transformer_ref"]), AcceleratorFake())
    transformer_ref = adapter.get_component("transformer_ref")

    assert adapter.trainable_component_names == ["transformer_ref"]
    assert adapter.get_trainable_parameters() == [transformer_ref.weight]
    assert not adapter.component_runtime.has_component_override("transformer")

    optimizer = torch.optim.AdamW(adapter.get_trainable_parameters())

    assert optimizer.param_groups[0]["params"] == [transformer_ref.weight]


def test_ref2va_gradient_checkpointing_uses_canonical_runtime_component() -> None:
    config = _config(["transformer_ref"])
    config.training_args.enable_gradient_checkpointing = True

    adapter = MiniMaxH3Ref2VAAdapter(config, AcceleratorFake())
    transformer_ref = adapter.get_component("transformer_ref")

    assert transformer_ref.gradient_checkpointing_calls == 1
    assert not adapter.component_runtime.has_component_override("transformer")


def test_ref2va_lora_reference_context_disables_canonical_runtime_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = MiniMaxH3Ref2VAAdapter(_config(["transformer_ref"]), AcceleratorFake())
    peft_model = PeftModelEquivalentFake()
    adapter.set_component("transformer_ref", peft_model)
    adapter.model_args.finetune_type = "lora"
    monkeypatch.setattr("flow_factory.models.abc.PeftModel", PeftModelEquivalentFake)

    with adapter.use_ref_parameters():
        assert peft_model.disable_adapter_events == ["enter"]

    assert peft_model.disable_adapter_events == ["enter", "exit"]
    assert adapter.get_component("transformer_ref") is peft_model


def test_ref2va_parameter_collection_and_snapshots_preserve_canonical_prefix() -> None:
    config = _config(["transformer_ref"])
    config.training_args.requires_ref_model = True
    config.training_args.ref_param_device = "cpu"
    adapter = MiniMaxH3Ref2VAAdapter(config, AcceleratorFake())
    transformer_ref = adapter.get_component("transformer_ref")
    component_name = adapter.trainable_component_names[0]
    named_trainable_parameters = {
        f"{component_name}.{parameter_name}": parameter
        for parameter_name, parameter in transformer_ref.named_parameters()
        if parameter.requires_grad
    }

    assert list(named_trainable_parameters) == ["transformer_ref.weight"]
    assert named_trainable_parameters["transformer_ref.weight"] is transformer_ref.weight

    adapter.add_named_parameters("snapshot")
    adapter._init_ref_parameters()
    assert adapter.get_named_parameters_info("snapshot")["target_components"] == ["transformer_ref"]

    transformer_ref.weight.data.fill_(2.0)
    with adapter.use_named_parameters("snapshot"):
        assert transformer_ref.weight.item() == pytest.approx(1.0)
    assert transformer_ref.weight.item() == pytest.approx(2.0)

    with adapter.use_ref_parameters():
        assert transformer_ref.weight.item() == pytest.approx(1.0)
    assert transformer_ref.weight.item() == pytest.approx(2.0)


def test_ref2va_full_checkpoint_load_installs_canonical_runtime_override(tmp_path: Any) -> None:
    adapter = MiniMaxH3Ref2VAAdapter(_config(["transformer_ref"]), AcceleratorFake())
    original = adapter.get_component("transformer_ref")

    adapter._load_full_model(str(tmp_path))

    replacement = adapter.get_component("transformer_ref")
    assert TransformerFake.loaded_paths == [str(tmp_path)]
    # Current infra loads into the prepared canonical module in place so routing,
    # optimizer identities, and FSDP registration cannot be detached by replacement.
    assert replacement is original
    assert replacement.weight.item() == pytest.approx(7.0)
    assert adapter.component_runtime.override_components == {}
    assert "transformer" not in adapter.component_runtime.declared_component_names


def test_ref2va_model_only_save_routes_canonical_component_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    adapter = MiniMaxH3Ref2VAAdapter(_config(["transformer_ref"]), AcceleratorFake())
    save_directory = tmp_path / "checkpoint"
    save_calls: List[tuple[str, nn.Module, str]] = []

    def record_full_model_save(
        component: nn.Module,
        component_path: str,
        **kwargs: Any,
    ) -> None:
        save_calls.append(("transformer_ref", component, component_path))

    monkeypatch.setattr(adapter, "_save_full_model", record_full_model_save)

    adapter.save_checkpoint(
        str(save_directory),
        dtype=torch.float32,
        save_ema=False,
        model_only=True,
    )

    assert save_calls == [
        ("transformer_ref", adapter.get_component("transformer_ref"), str(save_directory))
    ]


def test_ref2va_unknown_lifecycle_component_fails_with_runtime_context() -> None:
    adapter = MiniMaxH3Ref2VAAdapter(_config(["transformer_ref"]), AcceleratorFake())
    adapter.model_args.target_components = ["missing"]

    with pytest.raises(
        ValueError,
        match=r"unknown components.*missing.*received=.*transformer_ref",
    ):
        adapter.enable_gradient_checkpointing()


@pytest.mark.parametrize(
    ("adapter_class", "workflow", "expected_target", "received_targets"),
    [
        (MiniMaxH3T2VAAdapter, "t2va", "transformer", ["transformer_ref"]),
        (
            MiniMaxH3FL2VAAdapter,
            "fl2va",
            "transformer",
            ["transformer", "audio_vae"],
        ),
        (
            MiniMaxH3Ref2VAAdapter,
            "ref2va",
            "transformer_ref",
            ["transformer", "transformer_ref"],
        ),
    ],
)
def test_target_partition_fails_before_checkpoint_or_lora_setup(
    monkeypatch: pytest.MonkeyPatch,
    adapter_class: type[BaseAdapter],
    workflow: str,
    expected_target: str,
    received_targets: List[str],
) -> None:
    checkpoint_calls: List[Any] = []
    lora_calls: List[Any] = []
    monkeypatch.setattr(
        BaseAdapter,
        "load_checkpoint",
        lambda *args, **kwargs: checkpoint_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        BaseAdapter,
        "apply_lora",
        lambda *args, **kwargs: lora_calls.append((args, kwargs)),
    )
    config = _config(received_targets)
    config.model_args.resume_path = "checkpoint"
    config.model_args.resume_type = "lora"
    config.model_args.finetune_type = "lora"

    with pytest.raises(
        ValueError,
        match=rf"workflow='{workflow}'.*target_components.*\['{expected_target}'\].*{received_targets!r}",
    ):
        adapter_class(config, AcceleratorFake())

    assert WorkflowPipelineFake.calls == []
    assert checkpoint_calls == []
    assert lora_calls == []


@pytest.mark.parametrize(
    ("adapter_class", "transformer_name"),
    [
        (MiniMaxH3T2VAAdapter, "transformer"),
        (MiniMaxH3FL2VAAdapter, "transformer"),
        (MiniMaxH3Ref2VAAdapter, "transformer_ref"),
    ],
)
def test_adapter_builds_fresh_ordered_video_audio_schedulers(
    adapter_class: type[BaseAdapter], transformer_name: str
) -> None:
    adapter = adapter_class(_config([transformer_name]), AcceleratorFake())

    assert isinstance(adapter.scheduler, MiniMaxH3SDEScheduler)
    assert isinstance(adapter.audio_scheduler, MiniMaxH3SDEScheduler)
    assert adapter.scheduler is not adapter.audio_scheduler
    assert adapter.scheduler.shift == 12.0
    assert adapter.audio_scheduler.shift == 3.0
    assert adapter.scheduler_group.names == ("video", "audio")
    assert adapter.scheduler_group.primary_name == "video"
    assert adapter.scheduler_group.primary is adapter.scheduler
    for scheduler in (adapter.scheduler, adapter.audio_scheduler):
        assert scheduler.noise_level == 0.7
        assert scheduler.seed == 17
        assert scheduler.dynamics_type == "Flow-SDE"
