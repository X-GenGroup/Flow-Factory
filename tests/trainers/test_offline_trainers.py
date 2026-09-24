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

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager, nullcontext
from types import MappingProxyType, SimpleNamespace
from typing import Any, Iterator, Mapping

import pytest
import torch

from flow_factory.contracts import OFFLINE_EXECUTION_CONTRACT, MediaType
from flow_factory.data_utils.offline_dataset import (
    DecodedMedia,
    DemonstrationOutputBatch,
    OfflineBatch,
    PreferenceOutputBatch,
)
from flow_factory.data_utils.schema import NormalizedModelInput
from flow_factory.hparams import Arguments
from flow_factory.models.condition_state import PreparedConditionState
from flow_factory.models.output_state import (
    EncodedOutputState,
    GeometrySignature,
    MediaGeometrySignature,
)
from flow_factory.samples import ComponentTimes, LatentState, MultiModalStepOutput, NoisedState
from flow_factory.trainers.abc import BaseTrainer
from flow_factory.trainers.execution import TrainingProgress
from flow_factory.trainers.offline import offline_dpo as offline_dpo_module
from flow_factory.trainers.offline import sft as sft_module
from flow_factory.trainers.offline.offline_dpo import OfflineDPOTrainer
from flow_factory.trainers.offline.sft import SFTTrainer
from flow_factory.trainers.registry import get_trainer_class


class _TrainingArgs(dict):
    """Small mapping/attribute hybrid used by shared forward helpers."""

    def __init__(self) -> None:
        super().__init__(
            guidance_scale=8.0,
            guidance_scale_2=7.0,
            image_guidance_scale=6.0,
        )
        self.weighting_scheme = "uniform"
        self.num_train_timesteps = 2
        self.timestep_range = (0.0, 0.99)
        self.time_shift = 1.0
        self.logit_mean = 0.0
        self.logit_std = 1.0
        self.beta = 2.0


class _Accelerator:
    """Expose only the accumulation surface required by offline trainers."""

    def __init__(self, sync_schedule: list[bool]) -> None:
        self.device = torch.device("cpu")
        self._sync_schedule = iter(sync_schedule)
        self.sync_gradients = False
        self.accumulate_roots: list[Any] = []
        self.backward_losses: list[torch.Tensor] = []
        self.prepare_calls = 0

    @contextmanager
    def accumulate(self, root: Any) -> Iterator[None]:
        self.accumulate_roots.append(root)
        self.sync_gradients = next(self._sync_schedule)
        yield

    def backward(self, loss: torch.Tensor) -> None:
        self.backward_losses.append(loss.detach().clone())
        loss.backward()

    def prepare(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        self.prepare_calls += 1
        raise AssertionError("offline loader must remain owned by DistributedSampler")


class _Adapter:
    """Fake one-component codec and flow model with a reference scope."""

    trajectory_component_order = ("latent",)
    offline_training_forward_overrides = MappingProxyType(
        {
            "guidance_scale": 2.75,
            "guidance_scale_2": 1.25,
            "image_guidance_scale": 1.5,
        }
    )

    def __init__(self) -> None:
        self.policy_weight = torch.nn.Parameter(torch.tensor(0.7))
        self.preprocess_func = object()
        self.pipeline_io_contract = object()
        self.train_calls = 0
        self.encode_calls: list[str] = []
        self.prepare_calls = 0
        self.prepared_condition_ids: list[int] = []
        self.forward_events: list[tuple[float, bool, bool]] = []
        self.forward_override_events: list[tuple[float, bool, dict[str, float]]] = []
        self.drawn_noise: list[LatentState] = []
        self.reused_noise: list[LatentState] = []
        self.ref_scope_enters = 0
        self._ref_active = False

    def train(self, mode: bool = True) -> None:
        assert mode is True
        self.train_calls += 1

    def prepare_condition_state(
        self,
        condition: Mapping[str, Any],
        generator: torch.Generator | None = None,
    ) -> PreparedConditionState:
        del generator
        self.prepare_calls += 1
        return PreparedConditionState.identity(condition)

    def encode_output_state(
        self,
        media_batch: tuple[tuple[DecodedMedia, ...], ...],
        condition: Mapping[str, Any] | PreparedConditionState,
        generator: torch.Generator | None = None,
    ) -> EncodedOutputState:
        self.prepared_condition_ids.append(id(condition))
        del generator
        arm = str(media_batch[0][0].payload)
        self.encode_calls.append(arm)
        arm_value = 1.0 if arm == "rejected" else 0.0
        batch_size = len(media_batch)
        signature = GeometrySignature(
            media=(
                MediaGeometrySignature(
                    type=MediaType.IMAGE,
                    height=8,
                    width=8,
                ),
            )
        )
        return EncodedOutputState(
            clean_state=LatentState({"latent": torch.full((batch_size, 2), arm_value)}),
            forward_context={
                "arm_token": torch.full((batch_size, 1), arm_value),
            },
            decode_context={},
            geometry_signatures=tuple(signature for _ in range(batch_size)),
        )

    def build_training_component_times(
        self,
        primary_timesteps: torch.Tensor,
        *,
        batch: Mapping[str, Any],
    ) -> ComponentTimes:
        assert batch["arm_token"].shape[0] == primary_timesteps.shape[0]
        sigma = primary_timesteps.float() / 1000.0
        return ComponentTimes(
            timestep={"latent": primary_timesteps},
            next_timestep={"latent": torch.zeros_like(primary_timesteps)},
            sigma={"latent": sigma},
            next_sigma={"latent": torch.zeros_like(sigma)},
        )

    def add_forward_process_noise(
        self,
        clean_state: LatentState,
        times: ComponentTimes,
        *,
        generator: torch.Generator | None = None,
    ) -> NoisedState:
        del generator
        noise = LatentState(
            {
                "latent": torch.full_like(
                    clean_state.components["latent"],
                    float(len(self.drawn_noise) + 1),
                )
            }
        )
        self.drawn_noise.append(noise)
        return self.apply_forward_process_noise(clean_state, times, noise)

    def apply_forward_process_noise(
        self,
        clean_state: LatentState,
        times: ComponentTimes,
        noise: LatentState,
    ) -> NoisedState:
        if any(noise is item for item in self.drawn_noise):
            self.reused_noise.append(noise)
        clean = clean_state.components["latent"]
        noise_tensor = noise.components["latent"]
        sigma = times.sigma["latent"].reshape(clean.shape[0], 1)
        return NoisedState(
            state=LatentState({"latent": clean * (1.0 - sigma) + noise_tensor * sigma}),
            target_velocity=LatentState({"latent": noise_tensor - clean}),
            noise=noise,
        )

    def forward_state(
        self,
        *,
        batch: Mapping[str, Any],
        state: LatentState,
        times: ComponentTimes,
        **kwargs: Any,
    ) -> MultiModalStepOutput:
        arm = batch["arm_token"]
        resolved_kwargs = {**batch, **kwargs}
        coordinate = times.timestep["latent"].float().reshape(arm.shape[0], 1) / 1000.0
        self.forward_events.append(
            (float(arm[0].item()), self._ref_active, torch.is_grad_enabled())
        )
        self.forward_override_events.append(
            (
                float(arm[0].item()),
                self._ref_active,
                {
                    key: resolved_kwargs[key]
                    for key in (
                        "guidance_scale",
                        "guidance_scale_2",
                        "image_guidance_scale",
                    )
                },
            )
        )
        if self._ref_active:
            velocity = 0.2 * coordinate - 0.3 * arm
        else:
            velocity = self.policy_weight * (1.0 + coordinate) + 0.4 * arm
        return MultiModalStepOutput(
            velocity=LatentState({"latent": velocity.expand_as(state.components["latent"])})
        )

    @contextmanager
    def use_ref_parameters(self) -> Iterator[None]:
        assert not self._ref_active
        assert not torch.is_grad_enabled()
        self.ref_scope_enters += 1
        self._ref_active = True
        try:
            yield
        finally:
            self._ref_active = False

    @staticmethod
    def reduce_latent_values(
        values: Mapping[str, torch.Tensor],
        *,
        state: LatentState,
    ) -> torch.Tensor:
        del state
        return values["latent"].flatten(1).mean(dim=1)

    @staticmethod
    def reduce_flow_matching_objective_values(
        values: Mapping[str, torch.Tensor],
        *,
        state: LatentState,
    ) -> torch.Tensor:
        return _Adapter.reduce_latent_values(values, state=state)


def _media(arm: str, batch_size: int = 2) -> tuple[tuple[DecodedMedia, ...], ...]:
    return tuple(
        (
            DecodedMedia(
                type="image",
                path=f"{arm}-{index}.png",
                payload=arm,
            ),
        )
        for index in range(batch_size)
    )


def _batch(supervision_type: str, batch_size: int = 2) -> OfflineBatch:
    if supervision_type == "demonstration":
        output: DemonstrationOutputBatch | PreferenceOutputBatch = DemonstrationOutputBatch(
            target_media=_media("target", batch_size)
        )
    else:
        output = PreferenceOutputBatch(
            chosen_media=_media("chosen", batch_size),
            rejected_media=_media("rejected", batch_size),
        )
    return OfflineBatch(
        condition={
            "prompt_embeds": torch.ones(batch_size, 2),
            "guidance_scale": 5.0,
            "guidance_scale_2": 4.0,
            "image_guidance_scale": 3.0,
        },
        condition_ids=tuple(f"condition-{index}" for index in range(batch_size)),
        record_ids=tuple(f"record-{index}" for index in range(batch_size)),
        sources=tuple("source" for _ in range(batch_size)),
        source_ids=torch.zeros(batch_size, dtype=torch.long),
        model_inputs=tuple(
            NormalizedModelInput(prompt="prompt", negative_prompt=None, media=())
            for _ in range(batch_size)
        ),
        supervision_type=supervision_type,
        output=output,
        metadata_json=tuple("{}" for _ in range(batch_size)),
    )


def _trainer(
    trainer_type: type[SFTTrainer] | type[OfflineDPOTrainer],
    sync_schedule: list[bool],
) -> tuple[SFTTrainer | OfflineDPOTrainer, _Adapter, list[dict[str, list[torch.Tensor]]]]:
    trainer = object.__new__(trainer_type)
    trainer.accelerator = _Accelerator(sync_schedule)
    trainer.adapter = _Adapter()
    trainer.training_args = _TrainingArgs()
    trainer.model_bundle = object()
    trainer.autocast = nullcontext
    trainer.progress = TrainingProgress()
    optimizer_windows: list[dict[str, list[torch.Tensor]]] = []

    def apply_optimizer_step(
        loss_info: dict[str, list[torch.Tensor]],
    ) -> dict[str, list[torch.Tensor]]:
        optimizer_windows.append({name: list(values) for name, values in loss_info.items()})
        trainer.step += 1
        return defaultdict(list)

    trainer._apply_optimizer_step = apply_optimizer_step
    return trainer, trainer.adapter, optimizer_windows


def test_offline_trainers_are_dataset_driven_and_registered() -> None:
    assert SFTTrainer.__bases__ == (BaseTrainer,)
    assert OfflineDPOTrainer.__bases__ == (BaseTrainer,)
    assert SFTTrainer.execution_contract is OFFLINE_EXECUTION_CONTRACT
    assert OfflineDPOTrainer.execution_contract is OFFLINE_EXECUTION_CONTRACT
    assert SFTTrainer.paradigm == OfflineDPOTrainer.paradigm == "decoupled"
    assert "sample" not in SFTTrainer.__dict__
    assert "sample" not in OfflineDPOTrainer.__dict__
    assert get_trainer_class("sft") is SFTTrainer
    assert get_trainer_class("offline-dpo") is OfflineDPOTrainer


@pytest.mark.parametrize(
    ("trainer_type", "module", "supervision_type"),
    [
        (SFTTrainer, sft_module, "demonstration"),
        (OfflineDPOTrainer, offline_dpo_module, "preference"),
    ],
)
def test_offline_trainers_build_unprepared_distributed_loaders(
    monkeypatch: pytest.MonkeyPatch,
    trainer_type: type[SFTTrainer] | type[OfflineDPOTrainer],
    module: Any,
    supervision_type: str,
) -> None:
    trainer = object.__new__(trainer_type)
    trainer.config = object()
    trainer.accelerator = _Accelerator([])
    trainer.adapter = SimpleNamespace(
        preprocess_func=object(),
        pipeline_io_contract=object(),
        effective_pipeline_io_contract=object(),
    )
    sentinel = object()
    received: dict[str, Any] = {}

    def fake_builder(**kwargs: Any) -> Any:
        received.update(kwargs)
        return sentinel

    monkeypatch.setattr(module, "build_offline_train_dataloader", fake_builder)

    loader, source_loaders = trainer._build_train_dataloader()

    assert loader is sentinel
    assert source_loaders == {}
    assert received == {
        "config": trainer.config,
        "accelerator": trainer.accelerator,
        "preprocess_func": trainer.adapter.preprocess_func,
        "supervision_type": supervision_type,
        "pipeline_io_contract": trainer.adapter.effective_pipeline_io_contract,
    }
    assert trainer.accelerator.prepare_calls == 0


@pytest.mark.parametrize(
    ("trainer_type", "trainer_name"),
    [(SFTTrainer, "sft"), (OfflineDPOTrainer, "offline-dpo")],
)
def test_offline_trainers_do_not_construct_runtime_feedback_pipeline(
    trainer_type: type[SFTTrainer] | type[OfflineDPOTrainer],
    trainer_name: str,
) -> None:
    """The dataset sampler's final ``auto`` value is not a reward-group layout."""
    trainer = object.__new__(trainer_type)
    trainer.config = Arguments.from_dict(
        {
            "data": {
                "datasets": [
                    {
                        "name": "offline",
                        "dataset_dir": "unused",
                        "train": {"weight": 1},
                    }
                ]
            },
            "scheduler": {"dynamics_type": "ODE"},
            "train": {"trainer_type": trainer_name},
        }
    )
    trainer.training_args = trainer.config.training_args
    trainer.accelerator = SimpleNamespace()
    trainer.adapter = SimpleNamespace(tokenizer=None)
    trainer.log_args = SimpleNamespace(verbose=False)
    trainer.load_coordinator = SimpleNamespace(load_scope=lambda role: nullcontext())

    training_models, eval_models = BaseTrainer._init_reward_model(trainer)

    assert trainer.config.data_args.sampler_type == "auto"
    assert training_models == eval_models == {}
    assert trainer.reward_processor is None
    assert trainer.reward_buffer is None
    assert trainer.advantage_processor is None
    assert trainer.group_coordinator is None


def test_sft_reencodes_targets_and_preserves_optimizer_cadence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer, adapter, optimizer_windows = _trainer(SFTTrainer, [False, True])
    monkeypatch.setattr(
        sft_module,
        "sample_offline_timesteps",
        lambda *args, **kwargs: torch.tensor([[250.0, 250.0], [750.0, 750.0]]),
    )

    trainer.optimize_batch(_batch("demonstration"))
    assert trainer.step == 0
    trainer.optimize_batch(_batch("demonstration"))

    assert trainer.step == 1
    assert adapter.train_calls == 2
    assert adapter.prepare_calls == 2
    assert adapter.encode_calls == ["target", "target"]
    assert len(trainer.accelerator.backward_losses) == 2
    assert len(trainer.accelerator.accumulate_roots) == 2
    assert len(optimizer_windows) == 1
    assert len(optimizer_windows[0]["loss"]) == 2
    assert len(optimizer_windows[0]["flow_matching_loss"]) == 2


def test_sft_adapter_overrides_win_over_batch_and_sampling_guidance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer, adapter, _ = _trainer(SFTTrainer, [True])
    monkeypatch.setattr(
        sft_module,
        "sample_offline_timesteps",
        lambda *args, **kwargs: torch.tensor([[500.0, 500.0]]),
    )

    trainer.optimize_batch(_batch("demonstration"))

    assert trainer.training_args["guidance_scale"] == 8.0
    assert adapter.forward_override_events == [
        (
            0.0,
            False,
            {
                "guidance_scale": 2.75,
                "guidance_scale_2": 1.25,
                "image_guidance_scale": 1.5,
            },
        )
    ]


def test_offline_dpo_shares_schedule_noise_and_reference_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer, adapter, optimizer_windows = _trainer(OfflineDPOTrainer, [True])
    monkeypatch.setattr(
        offline_dpo_module,
        "sample_offline_timesteps",
        lambda *args, **kwargs: torch.tensor([[250.0, 250.0], [750.0, 750.0]]),
    )

    trainer.optimize_batch(_batch("preference"))

    assert trainer.step == 1
    assert adapter.encode_calls == ["chosen", "rejected"]
    assert adapter.prepare_calls == 1
    assert adapter.prepared_condition_ids[0] == adapter.prepared_condition_ids[1]
    assert len(adapter.drawn_noise) == 2
    assert len(adapter.reused_noise) == 4
    assert adapter.reused_noise[0] is adapter.drawn_noise[0]
    assert adapter.reused_noise[1] is adapter.drawn_noise[0]
    assert adapter.reused_noise[2] is adapter.drawn_noise[1]
    assert adapter.reused_noise[3] is adapter.drawn_noise[1]
    assert adapter.ref_scope_enters == 2
    assert len(optimizer_windows) == 1
    assert len(optimizer_windows[0]) == 8
    reference_events = [event for event in adapter.forward_events if event[1]]
    policy_events = [event for event in adapter.forward_events if not event[1]]
    assert len(reference_events) == len(policy_events) == 4
    assert all(not grad_enabled for _, _, grad_enabled in reference_events)
    assert all(grad_enabled for _, _, grad_enabled in policy_events)


def test_offline_dpo_uses_one_adapter_override_mapping_for_policy_and_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer, adapter, _ = _trainer(OfflineDPOTrainer, [True])
    monkeypatch.setattr(
        offline_dpo_module,
        "sample_offline_timesteps",
        lambda *args, **kwargs: torch.tensor([[500.0, 500.0]]),
    )

    trainer.optimize_batch(_batch("preference"))

    assert trainer.training_args["guidance_scale"] == 8.0
    expected = {
        "guidance_scale": 2.75,
        "guidance_scale_2": 1.25,
        "image_guidance_scale": 1.5,
    }
    assert adapter.forward_override_events == [
        (0.0, False, expected),
        (1.0, False, expected),
        (0.0, True, expected),
        (1.0, True, expected),
    ]


def test_offline_trainers_reject_the_other_supervision_branch() -> None:
    sft, sft_adapter, _ = _trainer(SFTTrainer, [])
    dpo, dpo_adapter, _ = _trainer(OfflineDPOTrainer, [])

    with pytest.raises(ValueError, match="supervision_type='demonstration'"):
        sft.optimize_batch(_batch("preference"))
    with pytest.raises(ValueError, match="supervision_type='preference'"):
        dpo.optimize_batch(_batch("demonstration"))
    with pytest.raises(TypeError, match="exact OfflineBatch"):
        dpo.optimize_batch(object())

    assert sft_adapter.train_calls == 0
    assert dpo_adapter.train_calls == 0
