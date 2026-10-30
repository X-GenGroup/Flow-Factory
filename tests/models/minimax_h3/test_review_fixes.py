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

from types import SimpleNamespace
from typing import Any, Dict, List

import pytest
import torch

from flow_factory.models.minimax_h3 import _common as h3_common
from flow_factory.models.minimax_h3 import workflow
from flow_factory.models.minimax_h3.adapters import (
    MiniMaxH3FL2VAAdapter,
    MiniMaxH3Ref2VAAdapter,
    MiniMaxH3T2VAAdapter,
)
from flow_factory.models.minimax_h3.denoise import forward_h3_state
from flow_factory.samples import (
    ComponentTimes,
    LatentState,
    MiniMaxH3T2VASample,
    MultiModalStepOutput,
    ReplayStep,
    StackedSampleBatch,
)
from flow_factory.scheduler import MiniMaxH3SDEScheduler
from flow_factory.trainers.distillation.opd.trainer import DiffusionOPDTrainer
from flow_factory.trainers.forward_process import forward_velocity_state
from flow_factory.trainers.rl.awm import AWMTrainer
from flow_factory.trainers.rl.crd import CRDTrainer
from flow_factory.trainers.rl.dgpo import DGPOTrainer
from flow_factory.trainers.rl.dpo import DPOTrainer
from flow_factory.trainers.rl.dppo import DPPOTrainer
from flow_factory.trainers.rl.grpo import GRPOGuardTrainer, GRPOTrainer
from flow_factory.trainers.rl.nft import DiffusionNFTTrainer


def _adapter(adapter_class: type, transformer: Any = None) -> Any:
    adapter = object.__new__(adapter_class)
    adapter.pipeline = SimpleNamespace()
    adapter.accelerator = SimpleNamespace(device=torch.device("cpu"))
    adapter.scheduler = SimpleNamespace(shift=12.0, noise_level=0.7)
    adapter.audio_scheduler = SimpleNamespace(shift=3.0, noise_level=0.7)
    adapter.training_args = SimpleNamespace(latent_storage_dtype=None)
    resolved_transformer = transformer or torch.nn.Linear(1, 1)
    adapter.get_component = lambda name: resolved_transformer
    adapter.loaded: List[Any] = []
    adapter.on_load_components = lambda components, device=None: adapter.loaded.append(
        (components, device)
    )
    return adapter


def _state(batch_size: int = 1, value: float = 0.0) -> LatentState:
    return LatentState(
        {
            "video": torch.full((batch_size, 2, 96), value),
            "audio": torch.full((batch_size, 3, 32), value),
        }
    )


def test_media_free_video_batch_unwraps_to_single_sample_none() -> None:
    assert workflow._decoded_video_sample([None]) is None


def _times(batch_size: int = 1) -> ComponentTimes:
    return ComponentTimes(
        timestep={name: torch.full((batch_size,), 500.0) for name in ("video", "audio")},
        next_timestep={name: torch.zeros(batch_size) for name in ("video", "audio")},
        sigma={name: torch.full((batch_size,), 0.5) for name in ("video", "audio")},
        next_sigma={name: torch.zeros(batch_size) for name in ("video", "audio")},
    )


@pytest.mark.parametrize(
    ("indices", "expected"),
    [
        ("all", ([0, 1, 2], [0, 1])),
        (None, ([], [])),
        ([-1], ([2], [])),
        ([0, -1], ([0, 2], [0])),
    ],
)
def test_sparse_indices_normalize_once_against_state_positions(indices: Any, expected: Any) -> None:
    assert workflow._resolve_trajectory_positions(indices, 2) == expected


@pytest.mark.parametrize(
    "indices",
    [
        "none",
        (0, 1),
        [True],
        [1.0],
        [-4],
        [3],
        [0, 0],
        [-1, 2],
    ],
)
def test_sparse_indices_reject_invalid_type_range_and_duplicates(indices: Any) -> None:
    with pytest.raises((TypeError, ValueError), match="trajectory_indices"):
        workflow._resolve_trajectory_positions(indices, 2)


@pytest.mark.parametrize(
    ("adapter_class", "extra"),
    [
        (MiniMaxH3T2VAAdapter, {"images": [["frame"]]}),
        (MiniMaxH3T2VAAdapter, {"references": [[{"type": "image"}]]}),
        (MiniMaxH3FL2VAAdapter, {"videos": [["video"]], "images": [["frame"]]}),
        (MiniMaxH3FL2VAAdapter, {"images": [[], []]}),
        (MiniMaxH3Ref2VAAdapter, {"images": [["frame"]], "references": [[{"type": "image"}]]}),
        (
            MiniMaxH3Ref2VAAdapter,
            {"audios": [[torch.zeros(1)]], "references": [[{"type": "image"}]]},
        ),
    ],
)
def test_preprocess_rejects_cross_workflow_inputs(
    monkeypatch, adapter_class: type, extra: Dict[str, Any]
) -> None:
    monkeypatch.setattr(
        workflow,
        "encode_h3_workflow_inputs",
        lambda *args, **kwargs: {"prompt_embeds": torch.zeros(1, 2, 4)},
    )
    adapter = _adapter(adapter_class)
    with pytest.raises(ValueError, match=f"workflow='{adapter.workflow}'"):
        adapter.preprocess_func(prompt=["describe"], height=64, width=96, num_frames=5, **extra)


@pytest.mark.parametrize(
    "field,value", [("negative_prompt", ["avoid this"]), ("guidance_scale", 2.0)]
)
@pytest.mark.parametrize("method", ["preprocess_func", "inference"])
def test_public_execution_rejects_non_neutral_guidance(
    monkeypatch, method: str, field: str, value: Any
) -> None:
    monkeypatch.setattr(
        workflow,
        "encode_h3_workflow_inputs",
        lambda *args, **kwargs: {"prompt_embeds": torch.zeros(1, 2, 4)},
    )
    adapter = _adapter(MiniMaxH3T2VAAdapter)
    arguments = {"prompt": ["describe"], field: value}
    if method == "preprocess_func":
        arguments.update(height=64, width=96, num_frames=5)
    with pytest.raises(ValueError, match=f"{field}"):
        getattr(adapter, method)(**arguments)


def test_public_preprocess_accepts_neutral_guidance_inputs(monkeypatch) -> None:
    monkeypatch.setattr(
        workflow,
        "encode_h3_workflow_inputs",
        lambda *args, **kwargs: {"prompt_embeds": torch.zeros(1, 2, 4)},
    )
    adapter = _adapter(MiniMaxH3T2VAAdapter)

    result = adapter.preprocess_func(
        prompt=["describe"],
        negative_prompt=None,
        guidance_scale=1.0,
        height=64,
        width=96,
        num_frames=5,
    )

    assert result["prompt_embeds"].shape == (1, 2, 4)


class _ReachedH3Blocks(RuntimeError):
    pass


@pytest.mark.parametrize(
    "trainer_class",
    [
        GRPOTrainer,
        GRPOGuardTrainer,
        DPPOTrainer,
        DPOTrainer,
        DGPOTrainer,
        DiffusionNFTTrainer,
        AWMTrainer,
        CRDTrainer,
        DiffusionOPDTrainer,
    ],
)
def test_base_trainer_sample_batch_accepts_default_h3_neutral_guidance(
    monkeypatch, trainer_class: type
) -> None:
    def reached_blocks(*args, **kwargs):
        raise _ReachedH3Blocks

    monkeypatch.setattr(workflow, "prepare_h3_rollout_state", reached_blocks)
    adapter = _adapter(MiniMaxH3T2VAAdapter)
    adapter.scheduler_group = {"video": SimpleNamespace(step=lambda *args, **kwargs: None)}
    trainer = object.__new__(trainer_class)
    trainer.adapter = adapter
    trainer.epoch = 0
    trainer.training_args = {"guidance_scale": 1.0, "seed": 0}

    with pytest.raises(_ReachedH3Blocks):
        trainer.sample_batch(
            {
                "prompt": ["describe"],
                "negative_prompt": None,
                "prompt_embeds": torch.zeros(1, 2, 4),
                "layout": {},
                "geometry": {},
            }
        )


@pytest.mark.parametrize(
    "images",
    [
        [[]],
        [["first", "middle", "last"]],
        [["first"], ["second"]],
    ],
)
@pytest.mark.parametrize("field", ["images", "condition_images"])
def test_fl2va_inference_rejects_invalid_condition_cardinality_before_blocks(
    monkeypatch, field: str, images: List[List[Any]]
) -> None:
    block_calls: List[Any] = []

    def reached_blocks(*args, **kwargs):
        block_calls.append((args, kwargs))
        raise _ReachedH3Blocks

    monkeypatch.setattr(workflow, "prepare_h3_rollout_state", reached_blocks)
    adapter = _adapter(MiniMaxH3FL2VAAdapter)

    with pytest.raises(ValueError, match=r"workflow='fl2va'.*(B=1|one or two)"):
        adapter.inference(
            prompt=["describe"],
            prompt_embeds=torch.zeros(1, 2, 4),
            layout={},
            geometry={},
            **{field: images},
        )

    assert block_calls == []


@pytest.mark.parametrize("images", [[["first"]], [["first", "last"]]])
@pytest.mark.parametrize("field", ["images", "condition_images"])
def test_fl2va_inference_preserves_valid_order_before_blocks(
    monkeypatch, field: str, images: List[List[Any]]
) -> None:
    captured: List[Any] = []

    def reached_blocks(pipeline, values, **kwargs):
        captured.append(values[field])
        raise _ReachedH3Blocks

    monkeypatch.setattr(workflow, "prepare_h3_rollout_state", reached_blocks)
    adapter = _adapter(MiniMaxH3FL2VAAdapter)

    with pytest.raises(_ReachedH3Blocks):
        adapter.inference(
            prompt=["describe"],
            prompt_embeds=torch.zeros(1, 2, 4),
            layout={},
            geometry={},
            **{field: images},
        )

    assert captured == [images]


def test_pinned_preprocessing_runs_under_no_grad(monkeypatch) -> None:
    grad_modes: List[bool] = []
    monkeypatch.setattr(
        workflow,
        "encode_h3_workflow_inputs",
        lambda *args, **kwargs: grad_modes.append(torch.is_grad_enabled())
        or {"prompt_embeds": torch.zeros(1, 2, 4)},
    )
    adapter = _adapter(MiniMaxH3T2VAAdapter)

    adapter.preprocess_func(prompt=["describe"], height=64, width=96, num_frames=5)

    assert grad_modes == [False]


def test_layout_normalization_uses_field_specific_shapes() -> None:
    position_ids = torch.arange(21, dtype=torch.float32).reshape(1, 7, 3) / 3
    layout = workflow._normalize_layout(
        {
            "layout": {
                "position_ids": position_ids,
                "token_tags": torch.arange(7).unsqueeze(0),
                "video_indices": torch.arange(2).unsqueeze(0),
                "num_condition_video_rows": [1],
            }
        }
    )

    assert layout["position_ids"].shape == (7, 3)
    assert layout["position_ids"].dtype == torch.float64
    assert torch.equal(layout["position_ids"], position_ids[0].to(torch.float64))
    assert layout["token_tags"].shape == (7,)
    assert layout["video_indices"].shape == (2,)
    assert layout["num_condition_video_rows"] == 1


def test_layout_normalization_rejects_non_floating_position_ids() -> None:
    with pytest.raises(ValueError, match="expected dtype float32 or float64, received torch.int64"):
        workflow._normalize_layout({"position_ids": torch.zeros(7, 3, dtype=torch.int64)})


def test_decode_materializes_exact_frozen_components(monkeypatch) -> None:
    monkeypatch.setattr(
        workflow,
        "decode_h3_targets",
        lambda *args, **kwargs: ("video", torch.zeros(1, 2, 8), 32000),
    )
    adapter = _adapter(MiniMaxH3T2VAAdapter)

    adapter.decode_latents(
        _state(),
        geometry={
            "num_latent_frames": 1,
            "latent_height": 2,
            "latent_width": 2,
            "num_audio_latents": 3,
        },
    )

    assert adapter.loaded == [(["vae", "video_processor", "audio_vae"], None)]


@pytest.mark.parametrize("guidance_scale", [0.0, 2.0])
def test_internal_forward_rejects_non_neutral_guidance(guidance_scale: float) -> None:
    adapter = _adapter(MiniMaxH3T2VAAdapter)
    with pytest.raises(ValueError, match=r"guidance_scale.*1"):
        adapter.forward(
            state=_state(),
            times=_times(),
            condition_prefixes={
                "video": torch.zeros(1, 1, 96),
                "audio": torch.zeros(1, 1, 32),
            },
            prompt_embeds=torch.zeros(1, 2, 4),
            layout={},
            guidance_scale=guidance_scale,
        )


def test_internal_forward_accepts_opd_neutral_guidance(monkeypatch) -> None:
    expected = SimpleNamespace()
    monkeypatch.setattr(workflow, "forward_h3_state", lambda *args, **kwargs: expected)
    adapter = _adapter(MiniMaxH3T2VAAdapter)

    result = adapter.forward(
        state=_state(),
        times=_times(),
        condition_prefixes={
            "video": torch.zeros(1, 1, 96),
            "audio": torch.zeros(1, 1, 32),
        },
        prompt_embeds=torch.zeros(1, 2, 4),
        layout={},
        guidance_scale=1.0,
    )

    assert result is expected


@pytest.mark.parametrize("entry", ["preprocess", "inference", "decode", "forward"])
def test_all_h3_execution_boundaries_reject_batched_inputs(monkeypatch, entry: str) -> None:
    adapter = _adapter(MiniMaxH3T2VAAdapter)
    with pytest.raises(ValueError, match="B=1"):
        if entry == "preprocess":
            adapter.preprocess_func(prompt=["a", "b"], height=64, width=96, num_frames=5)
        elif entry == "inference":
            adapter.inference(prompt=["a", "b"])
        elif entry == "decode":
            adapter.decode_latents(_state(2), geometry={})
        else:
            adapter.forward(
                state=_state(2),
                times=_times(2),
                condition_prefixes={
                    "video": torch.zeros(2, 1, 96),
                    "audio": torch.zeros(2, 1, 32),
                },
                prompt_embeds=torch.zeros(2, 2, 4),
                layout={},
            )


def test_forward_core_propagates_return_fields_and_attention(monkeypatch) -> None:
    captured: Dict[str, Any] = {}
    monkeypatch.setattr(
        "flow_factory.models.minimax_h3.denoise.run_h3_joint_transformer",
        lambda *args, **kwargs: captured.update(attention=kwargs["attention_kwargs"]) or _state(),
    )
    monkeypatch.setattr(
        "flow_factory.models.minimax_h3.denoise.step_h3_components",
        lambda *args, **kwargs: captured.update(step=kwargs) or SimpleNamespace(),
    )

    forward_h3_state(
        SimpleNamespace(),
        _state(),
        {"video": torch.zeros(1, 1, 96), "audio": torch.zeros(1, 1, 32)},
        torch.zeros(1, 2, 4),
        _times(),
        {},
        SimpleNamespace(),
        SimpleNamespace(),
        attention_kwargs={"scale": 0.5},
        return_kwargs=("next_latents", "velocity"),
    )

    assert captured["attention"] == {"scale": 0.5}
    assert captured["step"]["return_kwargs"] == ("next_latents", "velocity")


def test_real_h3_schedulers_dispatch_video_then_audio_with_prepared_gradients() -> None:
    calls: List[float] = []

    class TrackingScheduler(MiniMaxH3SDEScheduler):
        def step(self, *args, **kwargs):
            calls.append(self.shift)
            return super().step(*args, **kwargs)

    class Transformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.25))

        def forward(self, hidden_states, audio_hidden_states, **kwargs):
            return hidden_states * self.weight, audio_hidden_states * self.weight

    transformer = Transformer()
    video_scheduler = TrackingScheduler(shift=12.0, dynamics_type="Flow-SDE")
    audio_scheduler = TrackingScheduler(shift=3.0, dynamics_type="Flow-SDE")
    video_scheduler.set_timesteps(2)
    audio_scheduler.set_timesteps(2)
    output = forward_h3_state(
        transformer,
        _state(value=1.0),
        {"video": torch.zeros(1, 0, 96), "audio": torch.zeros(1, 0, 32)},
        torch.zeros(1, 2, 4),
        _times(),
        {
            "video_indices": torch.arange(2),
            "audio_indices": torch.arange(2, 5),
            "text_indices": torch.arange(5, 7),
            "num_condition_video_rows": 0,
            "num_condition_audio_rows": 0,
        },
        video_scheduler,
        audio_scheduler,
        generator=torch.Generator().manual_seed(11),
        noise_level=0.7,
        return_kwargs=(
            "next_latents",
            "next_latents_mean",
            "std_dev_t",
            "dt",
            "log_prob",
            "velocity",
        ),
    )

    assert calls == [12.0, 3.0]
    assert output.next_state is not None
    assert output.next_state_mean is not None
    assert output.component_log_probs is not None
    assert output.velocity is not None
    output.next_state.components["video"].sum().backward()
    assert transformer.weight.grad is not None

    replay_generator = torch.Generator().manual_seed(23)
    replay_generator_state = replay_generator.get_state()
    replay = forward_h3_state(
        transformer,
        _state(value=1.0),
        {"video": torch.zeros(1, 0, 96), "audio": torch.zeros(1, 0, 32)},
        torch.zeros(1, 2, 4),
        _times(),
        {
            "video_indices": torch.arange(2),
            "audio_indices": torch.arange(2, 5),
            "text_indices": torch.arange(5, 7),
            "num_condition_video_rows": 0,
            "num_condition_audio_rows": 0,
        },
        video_scheduler,
        audio_scheduler,
        next_state=output.next_state,
        generator=replay_generator,
        noise_level=0.7,
        return_kwargs=("next_latents", "log_prob"),
    )
    assert torch.equal(replay_generator.get_state(), replay_generator_state)
    assert replay.next_state is not None


def test_fl2va_rollout_preserves_ordered_condition_images(monkeypatch) -> None:
    first = torch.zeros(3, 2, 2)
    last = torch.ones(3, 2, 2)
    sample = workflow._build_h3_sample(
        MiniMaxH3FL2VAAdapter,
        prompt="describe",
        prompt_embeds=torch.zeros(2, 4),
        video=torch.zeros(2, 3, 2, 2),
        audio=torch.zeros(2, 8),
        sample_rate=32000,
        trajectory=None,
        condition_images=[first, last],
        extra_kwargs={},
    )
    reversed_sample = workflow._build_h3_sample(
        MiniMaxH3FL2VAAdapter,
        prompt="describe",
        prompt_embeds=torch.zeros(2, 4),
        video=torch.zeros(2, 3, 2, 2),
        audio=torch.zeros(2, 8),
        sample_rate=32000,
        trajectory=None,
        condition_images=[last, first],
        extra_kwargs={},
    )

    assert torch.equal(sample.condition_images[0], first)
    assert torch.equal(sample.condition_images[1], last)
    assert sample.unique_id != reversed_sample.unique_id


def test_existing_t2va_forward_test_uses_matching_sample_class() -> None:
    sample = MiniMaxH3T2VASample(prompt="describe")
    assert type(sample).__name__ == "MiniMaxH3T2VASample"


@pytest.mark.parametrize(
    "output",
    [
        SimpleNamespace(
            log_prob=None, component_log_probs={"video": torch.ones(1), "audio": torch.ones(1)}
        ),
        SimpleNamespace(log_prob=torch.ones(1), component_log_probs=None),
        SimpleNamespace(
            log_prob=torch.ones(1),
            component_log_probs={"audio": torch.ones(1), "video": torch.ones(1)},
        ),
        SimpleNamespace(
            log_prob=torch.ones(1), component_log_probs={"video": None, "audio": torch.ones(1)}
        ),
    ],
)
def test_rollout_collectors_reject_missing_or_malformed_log_tensors(output: Any) -> None:
    with pytest.raises((TypeError, ValueError), match="log_prob"):
        workflow._validate_rollout_log_output(output, "t2va", 0)


def test_rollout_dense_sparse_and_disabled_collection_preserve_rng_state(
    monkeypatch,
) -> None:
    schedules = {
        "video": (torch.tensor([1000.0, 0.0]), torch.tensor([1.0, 0.0])),
        "audio": (torch.tensor([1000.0, 0.0]), torch.tensor([1.0, 0.0])),
    }

    def prepare(*args, generator, **kwargs):
        video = torch.randn((1, 2, 96), generator=generator)
        audio = torch.randn((1, 3, 32), generator=generator)
        return LatentState({"video": video, "audio": audio}), {
            "video": torch.zeros(1, 0, 96),
            "audio": torch.zeros(1, 0, 32),
        }

    monkeypatch.setattr(workflow, "prepare_h3_rollout_state", prepare)
    monkeypatch.setattr(
        workflow,
        "build_h3_schedule_plan",
        lambda *args, **kwargs: SimpleNamespace(schedules=schedules),
    )
    monkeypatch.setattr(
        workflow,
        "decode_h3_targets",
        lambda pipeline, state, *args, **kwargs: (
            torch.full((1, 2, 3, 2, 2), state.components["video"].mean()),
            torch.zeros(1, 2, 8),
            32000,
        ),
    )
    adapter = _adapter(MiniMaxH3T2VAAdapter)

    def forward(**kwargs):
        if kwargs.get("generator") is not None:
            raise AssertionError(
                "inference generator must not reach H3 denoising steps, "
                f"received {kwargs.get('generator')!r}"
            )
        draw = torch.rand(())
        next_state = LatentState(
            {component: values + draw for component, values in kwargs["state"].components.items()}
        )
        return SimpleNamespace(
            next_state=next_state,
            next_state_mean=next_state,
            velocity=next_state,
            log_prob=torch.ones(1),
            component_log_probs={"video": torch.ones(1), "audio": torch.ones(1)},
        )

    adapter.forward = forward
    arguments = {
        "prompt": ["describe"],
        "prompt_embeds": torch.zeros(1, 2, 4),
        "layout": {},
        "geometry": {},
        "num_inference_steps": 1,
        "compute_log_prob": True,
    }
    samples = []
    for indices in ("all", [-1], None):
        torch.manual_seed(0)
        samples.append(
            adapter.inference(
                **arguments,
                trajectory_indices=indices,
                generator=torch.Generator().manual_seed(17),
            )[0]
        )

    assert torch.equal(samples[0].video, samples[1].video)
    assert torch.equal(samples[1].video, samples[2].video)
    assert samples[0].trajectory is not None
    assert samples[1].trajectory.components["video"].state_index_map.tolist() == [-1, 0]
    assert samples[2].trajectory is None


def _training_batch() -> StackedSampleBatch:
    return StackedSampleBatch(
        [
            MiniMaxH3T2VASample(
                prompt="describe",
                prompt_embeds=torch.zeros(2, 4),
                extra_kwargs={
                    "condition_prefixes": {
                        "video": torch.zeros(1, 96),
                        "audio": torch.zeros(1, 32),
                    },
                    "layout": {},
                },
            )
        ]
    )


def _patch_h3_forward(monkeypatch) -> None:
    def forward(*args, **kwargs):
        state = args[1]
        if kwargs.get("velocity_only"):
            return state
        return MultiModalStepOutput(
            next_state=state,
            next_state_mean=state,
            std_dev_t={"video": torch.ones(1), "audio": torch.ones(1)},
            dt={"video": torch.ones(1), "audio": torch.ones(1)},
            log_prob=torch.ones(1),
            component_log_probs={"video": torch.ones(1), "audio": torch.ones(1)},
            velocity=state,
        )

    monkeypatch.setattr(workflow, "forward_h3_state", forward)


@pytest.mark.parametrize("trainer_class", [GRPOTrainer, GRPOGuardTrainer, DPPOTrainer])
def test_coupled_trainers_use_real_h3_forward_bridge(monkeypatch, trainer_class: type) -> None:
    _patch_h3_forward(monkeypatch)
    adapter = _adapter(MiniMaxH3T2VAAdapter)
    trainer = object.__new__(trainer_class)
    trainer.adapter = adapter
    trainer.training_args = {}
    replay = ReplayStep(state=_state(), times=_times(), next_state=_state(1))

    output = trainer._replay_forward(
        _training_batch(),
        replay,
        ("next_latents", "next_latents_mean", "std_dev_t", "dt", "log_prob", "velocity"),
    )

    assert output.velocity.component_names == ("video", "audio")
    assert output.next_state_mean is not None


@pytest.mark.parametrize(
    "trainer_class",
    [DPOTrainer, DGPOTrainer, DiffusionNFTTrainer, AWMTrainer, CRDTrainer],
)
def test_decoupled_trainers_use_real_h3_forward_bridge(monkeypatch, trainer_class: type) -> None:
    _patch_h3_forward(monkeypatch)
    adapter = _adapter(MiniMaxH3T2VAAdapter)
    trainer = object.__new__(trainer_class)
    trainer.adapter = adapter
    trainer.training_args = {}

    velocity = forward_velocity_state(
        trainer,
        _training_batch(),
        _state(),
        _times(),
        source=trainer_class.__name__,
    )

    assert velocity.component_names == ("video", "audio")


@pytest.mark.parametrize("guidance_scale", [1.0, 2.0])
def test_opd_uses_real_h3_bridge_and_requires_neutral_guidance(
    monkeypatch, guidance_scale: float
) -> None:
    _patch_h3_forward(monkeypatch)
    adapter = _adapter(MiniMaxH3T2VAAdapter)
    replay = ReplayStep(state=_state(), times=_times(), next_state=_state(1))
    adapter.get_replay_step = lambda batch, index: replay

    class TrainingArgs(dict):
        def __getattr__(self, name):
            return self[name]

    trainer = object.__new__(DiffusionOPDTrainer)
    trainer.adapter = adapter
    trainer.training_args = TrainingArgs(loss_target="v", guidance_scale=guidance_scale)
    trainer._student_noise_level = 0.0

    if guidance_scale != 1.0:
        with pytest.raises(ValueError, match=r"guidance_scale.*1"):
            trainer._forward_step(_training_batch(), 0, guidance_scale, "student")
        return

    _, target, output = trainer._forward_step(_training_batch(), 0, guidance_scale, "student")
    assert target.component_names == ("video", "audio")
    assert output.velocity is not None


@pytest.mark.parametrize(
    "adapter_class",
    [MiniMaxH3T2VAAdapter, MiniMaxH3FL2VAAdapter, MiniMaxH3Ref2VAAdapter],
)
def test_h3_adapters_apply_and_draw_forward_process_noise(
    adapter_class: type,
) -> None:
    adapter = _adapter(adapter_class)
    clean = _state(value=2.0)
    noise = _state(value=4.0)
    times = _times()

    applied = adapter.apply_forward_process_noise(clean, times, noise)
    drawn = adapter.add_forward_process_noise(
        clean, times, generator=torch.Generator().manual_seed(7)
    )

    assert torch.allclose(applied.state.components["video"], torch.full((1, 2, 96), 3.0))
    assert torch.allclose(applied.target_velocity.components["audio"], torch.full((1, 3, 32), -2.0))
    assert drawn.noise.component_names == ("video", "audio")
    assert torch.allclose(
        drawn.target_velocity.components["video"],
        clean.components["video"] - drawn.noise.components["video"],
    )


def test_h3_draw_delegates_to_deterministic_apply(monkeypatch) -> None:
    calls: List[Any] = []
    expected = SimpleNamespace()
    monkeypatch.setattr(
        h3_common,
        "apply_forward_process_noise",
        lambda clean, times, noise: calls.append((clean, times, noise)) or expected,
        raising=False,
    )

    result = h3_common.draw_forward_process_noise(
        _state(), _times(), generator=torch.Generator().manual_seed(3)
    )

    assert result is expected
    assert calls[0][2].component_names == ("video", "audio")


def test_h3_draw_rejects_batched_state_before_consuming_rng() -> None:
    generator = torch.Generator().manual_seed(5)
    initial_state = generator.get_state()

    with pytest.raises(ValueError, match="B=1"):
        h3_common.draw_forward_process_noise(_state(2), _times(2), generator=generator)

    assert torch.equal(generator.get_state(), initial_state)


@pytest.mark.parametrize(
    ("clean", "times", "noise", "match"),
    [
        (_state(2), _times(2), _state(2), "B=1"),
        (
            _state(),
            _times(),
            LatentState(
                {
                    "audio": torch.zeros(1, 3, 32),
                    "video": torch.zeros(1, 2, 96),
                }
            ),
            "component order",
        ),
        (
            _state(),
            _times(),
            LatentState(
                {
                    "video": torch.zeros(1, 2, 96, dtype=torch.float64),
                    "audio": torch.zeros(1, 3, 32, dtype=torch.float64),
                }
            ),
            "dtype",
        ),
    ],
)
def test_h3_apply_forward_process_noise_validates_contract(
    clean: LatentState, times: ComponentTimes, noise: LatentState, match: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        h3_common.apply_forward_process_noise(clean, times, noise)
