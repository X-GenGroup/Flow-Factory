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

import copy
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from flow_factory.models.minimax_h3 import workflow
from flow_factory.models.minimax_h3._common import build_component_step_output
from flow_factory.models.minimax_h3.adapters import MiniMaxH3T2VAAdapter
from flow_factory.samples import LatentState, StructuredTrajectory
from flow_factory.scheduler import MiniMaxH3SDEScheduler, SchedulerGroup
from flow_factory.scheduler.abc import SDESchedulerOutput
from flow_factory.trainers.distillation.opd.trainer import DiffusionOPDTrainer
from flow_factory.trainers.rl.awm import AWMTrainer
from flow_factory.trainers.rl.crd import CRDTrainer
from flow_factory.trainers.rl.dgpo import DGPOTrainer
from flow_factory.trainers.rl.dpo import DPOTrainer
from flow_factory.trainers.rl.dppo import DPPOTrainer
from flow_factory.trainers.rl.grpo import GRPOGuardTrainer, GRPOTrainer
from flow_factory.trainers.rl.nft import DiffusionNFTTrainer
from flow_factory.utils.group_coordinator import GroupCoordinator


class TinyH3Transformer(torch.nn.Module):
    """Stand in only for unavailable H3 weights."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.2))
        self.keyframe_noise_aug = 0.999

    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return hidden_states * self.weight, audio_hidden_states * self.weight


class Args(dict):
    """Attribute-accessible tiny argument mapping."""

    def __getattr__(self, name: str) -> Any:
        return self[name]


class SingleProcessAccelerator:
    """Single-process optimizer plumbing; model execution remains production H3."""

    def __init__(self, parameter: torch.nn.Parameter) -> None:
        self.device = torch.device("cpu")
        self.num_processes = 1
        self.process_index = 0
        self.is_main_process = True
        self.is_local_main_process = True
        self.sync_gradients = True
        self.parameter = parameter
        self.backward_calls = 0
        self.observed_grad = False

    @contextmanager
    def accumulate(self, model: torch.nn.Module):
        yield

    def backward(self, loss: torch.Tensor) -> None:
        self.backward_calls += 1
        loss.backward()
        self.observed_grad = self.parameter.grad is not None

    def clip_grad_norm_(self, parameters: Any, max_norm: float) -> torch.Tensor:
        return torch.nn.utils.clip_grad_norm_(list(parameters), max_norm)

    def reduce(self, value: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        return value

    def gather(self, value: torch.Tensor) -> torch.Tensor:
        return value

    def unwrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        return model


class GlooAccelerator(SingleProcessAccelerator):
    """Minimal accelerator surface backed by real two-rank gloo collectives."""

    def __init__(self, parameter: torch.nn.Parameter, rank: int, world_size: int) -> None:
        super().__init__(parameter)
        self.num_processes = world_size
        self.process_index = rank
        self.is_main_process = rank == 0
        self.is_local_main_process = rank == 0
        self.reduce_calls = 0
        self.gather_calls = 0

    def reduce(self, value: Any, reduction: str = "mean") -> Any:
        if isinstance(value, dict):
            return {key: self.reduce(item, reduction=reduction) for key, item in value.items()}
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                "expected gloo reduction tensor or dict of tensors, received "
                f"{type(value).__name__}"
            )
        self.reduce_calls += 1
        reduced = value.clone()
        dist.all_reduce(reduced)
        if reduction == "mean":
            reduced /= self.num_processes
        elif reduction != "sum":
            raise ValueError(f"expected gloo reduction 'mean' or 'sum', received {reduction!r}")
        return reduced

    def gather(self, value: torch.Tensor) -> torch.Tensor:
        self.gather_calls += 1
        gathered = [torch.empty_like(value) for _ in range(self.num_processes)]
        dist.all_gather(gathered, value)
        return torch.cat(gathered, dim=0)


def _target_state(value: float = 1.0) -> LatentState:
    return LatentState(
        {
            "video": torch.full((1, 2, 96), value),
            "audio": torch.full((1, 3, 32), value),
        }
    )


def test_logprob_only_reference_state_uses_component_dof_weighted_oracle() -> None:
    video_log_prob = torch.tensor([2.0])
    audio_log_prob = torch.tensor([-1.0])
    reference_state = _target_state()
    video_dof = reference_state.components["video"][0].numel()
    audio_dof = reference_state.components["audio"][0].numel()

    output = build_component_step_output(
        SDESchedulerOutput(log_prob=video_log_prob),
        SDESchedulerOutput(log_prob=audio_log_prob),
        reference_state=reference_state,
    )

    expected = (video_log_prob * video_dof + audio_log_prob * audio_dof) / (video_dof + audio_dof)
    incorrect_simple_mean = (video_log_prob + audio_log_prob) / 2
    torch.testing.assert_close(output.log_prob, expected)
    assert not torch.equal(output.log_prob, incorrect_simple_mean)


def test_no_weight_t2va_harness_rejects_non_transformer_component_name() -> None:
    adapter = _production_adapter()

    assert isinstance(adapter.get_component("transformer"), TinyH3Transformer)
    with pytest.raises(
        ValueError,
        match="T2VA.*expected component name 'transformer'.*'transformer_ref'",
    ):
        adapter.get_component("transformer_ref")


def _production_adapter() -> MiniMaxH3T2VAAdapter:
    adapter = object.__new__(MiniMaxH3T2VAAdapter)
    adapter.pipeline = SimpleNamespace()
    adapter.accelerator = SimpleNamespace(
        device=torch.device("cpu"),
        unwrap_model=lambda model: model,
    )
    adapter.component_runtime = SimpleNamespace(
        declared_component_names=("transformer",),
    )
    adapter.scheduler = MiniMaxH3SDEScheduler(
        shift=12.0,
        dynamics_type="Flow-SDE",
        sde_steps=[0, 1],
        num_sde_steps=2,
    )
    adapter.audio_scheduler = MiniMaxH3SDEScheduler(
        shift=3.0,
        dynamics_type="Flow-SDE",
        sde_steps=[0, 1],
        num_sde_steps=2,
    )
    adapter.scheduler_group = SchedulerGroup(
        {"video": adapter.scheduler, "audio": adapter.audio_scheduler},
        primary_name="video",
    )
    transformer = TinyH3Transformer()

    def get_component(name: str) -> TinyH3Transformer:
        if name != "transformer":
            raise ValueError(
                "MiniMax H3 T2VA no-weight harness expected component name "
                f"'transformer', received {name!r}"
            )
        return transformer

    adapter.get_component = get_component
    adapter.on_load_components = lambda names, device=None: None
    adapter.target_module_map = {"transformer": ["weight"]}
    adapter.model_args = SimpleNamespace(
        finetune_type="full",
        target_components=["transformer"],
    )
    adapter.training_args = SimpleNamespace(
        requires_ref_model=False,
        ref_param_device="cpu",
    )
    adapter._ref_ema = None
    adapter._named_parameters = {}
    return adapter


def _run_production_inference(
    monkeypatch: pytest.MonkeyPatch | None,
    *,
    trajectory_indices: Any,
    adapter: MiniMaxH3T2VAAdapter | None = None,
    callback_fields: tuple[str, ...] = (),
) -> Any:
    prepare = lambda *args, **kwargs: (
        _target_state(),
        {
            "video": torch.zeros(1, 0, 96),
            "audio": torch.zeros(1, 0, 32),
        },
    )
    decode = lambda *args, **kwargs: (
        torch.zeros(1, 2, 3, 2, 2),
        torch.zeros(1, 2, 8),
        32000,
    )
    if monkeypatch is None:
        workflow.prepare_h3_rollout_state = prepare
        workflow.decode_h3_targets = decode
    else:
        monkeypatch.setattr(workflow, "prepare_h3_rollout_state", prepare)
        monkeypatch.setattr(workflow, "decode_h3_targets", decode)
    adapter = _production_adapter() if adapter is None else adapter
    with torch.no_grad():
        sample = adapter.inference(
            prompt=["describe"],
            negative_prompt=None,
            guidance_scale=1.0,
            prompt_embeds=torch.zeros(1, 2, 4),
            layout={
                "video_indices": torch.arange(2),
                "audio_indices": torch.arange(2, 5),
                "text_indices": torch.arange(5, 7),
                "num_condition_video_rows": 0,
                "num_condition_audio_rows": 0,
            },
            geometry={},
            num_inference_steps=2,
            trajectory_indices=trajectory_indices,
            compute_log_prob=True,
            extra_call_back_kwargs=callback_fields,
            generator=torch.Generator().manual_seed(13),
        )[0]
    # Calling `inference` directly skips `_attach_batch_provenance`, which is
    # where a production rollout picks up the sampler-assigned identity. Training
    # draws are seeded from it, so stand in for that step here.
    sample.assign_sampling_identity(group_id=0, group_member_id=0, sample_id=0)
    return sample


def test_no_weight_production_inference_terminal_only_stores_one_state(
    monkeypatch,
) -> None:
    sample = _run_production_inference(monkeypatch, trajectory_indices=[-1])

    trajectory = sample.trajectory
    assert trajectory is not None
    assert trajectory.components["video"].state_index_map.tolist() == [-1, -1, 0]
    assert trajectory.components["audio"].state_index_map.tolist() == [-1, -1, 0]
    assert trajectory.components["video"].states.shape[0] == 1
    assert trajectory.components["audio"].states.shape[0] == 1
    assert trajectory.log_probs is None
    assert trajectory.log_prob_index_map is None
    assert trajectory.component_log_probs is None


def test_no_weight_production_inference_stores_states_in_latent_storage_dtype(
    monkeypatch,
) -> None:
    adapter = _production_adapter()
    adapter.training_args.latent_storage_dtype = "bf16"

    sample = _run_production_inference(
        monkeypatch,
        trajectory_indices="all",
        adapter=adapter,
        callback_fields=("velocity",),
    )

    trajectory = sample.trajectory
    assert trajectory is not None
    for component in ("video", "audio"):
        assert trajectory.components[component].states.dtype == torch.bfloat16


def test_no_weight_production_inference_replays_the_consumed_state_dtype(
    monkeypatch,
) -> None:
    adapter = _production_adapter()
    adapter.training_args.latent_storage_dtype = "bf16"
    consumed_dtypes: list[torch.dtype] = []
    original_forward = type(adapter).forward

    def recording_forward(self, **kwargs: Any) -> Any:
        consumed_dtypes.append(kwargs["state"].components["video"].dtype)
        return original_forward(self, **kwargs)

    monkeypatch.setattr(type(adapter), "forward", recording_forward)

    sample = _run_production_inference(
        monkeypatch,
        trajectory_indices="all",
        adapter=adapter,
    )

    stored = sample.trajectory.components["video"].states
    assert consumed_dtypes == [torch.bfloat16, torch.bfloat16]
    assert stored.dtype == torch.bfloat16


def test_no_weight_production_inference_none_skips_trajectory_construction(
    monkeypatch,
) -> None:
    def forbidden_trajectory_construction(*args, **kwargs):
        raise AssertionError("trajectory_indices=None must not construct an all--1 map")

    monkeypatch.setattr(
        workflow,
        "build_structured_trajectories",
        forbidden_trajectory_construction,
    )

    sample = _run_production_inference(monkeypatch, trajectory_indices=None)

    assert sample.trajectory is None
    assert sample.timesteps is None
    assert sample.all_latents is None
    assert sample.latent_index_map is None
    assert sample.log_probs is None
    assert sample.log_prob_index_map is None


def _trainer_args(**overrides: Any) -> Args:
    values = {
        "per_device_batch_size": 1,
        "num_inner_epochs": 1,
        "shuffle_samples": False,
        "seed": 0,
        "max_grad_norm": 1.0,
        "adv_clip_range": (-5.0, 5.0),
        "clip_range": (-0.2, 0.2),
        "kl_beta": 0.0,
        "kl_type": "v-based",
        "offload_samples_to_cpu": False,
        # NFT and AWM read this to decide whether a group shares one noise
        # tensor; it has no implicit default, so the stub has to carry it.
        "use_shared_noise": False,
    }
    values.update(overrides)
    return Args(values)


def _production_trainer(
    trainer_class: type,
    adapter: MiniMaxH3T2VAAdapter,
    accelerator: SingleProcessAccelerator | None = None,
    **argument_overrides: Any,
) -> tuple[Any, SingleProcessAccelerator]:
    transformer = adapter.get_component("transformer")
    accelerator = (
        SingleProcessAccelerator(transformer.weight) if accelerator is None else accelerator
    )
    adapter.accelerator = accelerator
    trainer = object.__new__(trainer_class)
    trainer.adapter = adapter
    trainer.training_args = _trainer_args(**argument_overrides)
    trainer.accelerator = accelerator
    # Group collectives are scoped in data-parallel replicas; a real coordinator
    # over this single-process accelerator delegates straight back to it.
    trainer.group_coordinator = GroupCoordinator(accelerator, sampler_type="group_distributed")
    trainer.model_bundle = transformer
    trainer.optimizer = torch.optim.SGD(transformer.parameters(), lr=1e-3)
    trainer.autocast = nullcontext
    trainer.epoch = 0
    trainer.step = 0
    trainer.log_args = SimpleNamespace(verbose=False)
    trainer.logger = None
    return trainer, accelerator


def _assert_production_h3_sample(
    adapter: MiniMaxH3T2VAAdapter,
    sample: Any,
) -> None:
    assert type(adapter) is MiniMaxH3T2VAAdapter
    assert isinstance(adapter.scheduler, MiniMaxH3SDEScheduler)
    assert isinstance(adapter.audio_scheduler, MiniMaxH3SDEScheduler)
    assert isinstance(adapter.scheduler_group, SchedulerGroup)
    assert isinstance(sample.trajectory, StructuredTrajectory)


@pytest.mark.parametrize(
    "trainer_class",
    [GRPOTrainer, GRPOGuardTrainer, DPPOTrainer],
)
def test_no_weight_coupled_optimize_uses_production_h3_path(
    monkeypatch,
    trainer_class: type,
) -> None:
    adapter = _production_adapter()
    callback_fields: tuple[str, ...] = ()
    argument_overrides: dict[str, Any] = {}
    if trainer_class is GRPOGuardTrainer:
        callback_fields = ("next_latents_mean",)
    elif trainer_class is DPPOTrainer:
        callback_fields = ("velocity",)
        argument_overrides.update(
            kl_mask_type="v-based",
            kl_guidance_scale=1.0,
            kl_mask_threshold=1e6,
        )
    sample = _run_production_inference(
        monkeypatch,
        trajectory_indices="all",
        adapter=adapter,
        callback_fields=callback_fields,
    )
    sample.extra_kwargs["advantage"] = torch.tensor(1.0)
    _assert_production_h3_sample(adapter, sample)
    trainer, accelerator = _production_trainer(
        trainer_class,
        adapter,
        **argument_overrides,
    )

    result = trainer.optimize([sample])

    assert result is None
    assert accelerator.backward_calls == 2
    assert accelerator.observed_grad


def test_no_weight_coupled_optimize_replays_reduced_precision_storage(
    monkeypatch,
) -> None:
    adapter = _production_adapter()
    adapter.training_args.latent_storage_dtype = "bf16"
    sample = _run_production_inference(
        monkeypatch,
        trajectory_indices="all",
        adapter=adapter,
    )
    sample.extra_kwargs["advantage"] = torch.tensor(1.0)
    trainer, accelerator = _production_trainer(GRPOTrainer, adapter)

    result = trainer.optimize([sample])

    assert result is None
    assert accelerator.backward_calls == 2
    assert accelerator.observed_grad


@pytest.mark.parametrize("trainer_class", [DiffusionNFTTrainer, AWMTrainer])
def test_no_weight_matching_optimize_uses_production_h3_noising_and_forward(
    monkeypatch,
    trainer_class: type,
) -> None:
    adapter = _production_adapter()
    sample = _run_production_inference(
        monkeypatch,
        trajectory_indices=[-1],
        adapter=adapter,
    )
    sample.extra_kwargs["advantage"] = torch.tensor(1.0)
    _assert_production_h3_sample(adapter, sample)
    trainer, accelerator = _production_trainer(
        trainer_class,
        adapter,
        num_train_timesteps=1,
    )
    trainer.time_sampling_strategy = "discrete"
    trainer.time_shift = 1.0
    trainer.num_train_timesteps = 1
    trainer.timestep_range = (0.0, 1.0)
    trainer.off_policy = False
    trainer.kl_type = "v-based"
    if trainer_class is DiffusionNFTTrainer:
        trainer.nft_beta = 1.0
    else:
        trainer.weighting = "Uniform"
        trainer.ghuber_power = 0.25
        trainer.kl_beta = 0.0
        trainer.ema_kl_beta = 0.0

    result = trainer.optimize([sample])

    assert result is None
    assert accelerator.backward_calls == 1
    assert accelerator.observed_grad


def test_no_weight_dpo_optimize_reuses_h3_component_noise(
    monkeypatch,
) -> None:
    adapter = _production_adapter()
    chosen = _run_production_inference(
        monkeypatch,
        trajectory_indices=[-1],
        adapter=adapter,
    )
    rejected = copy.deepcopy(chosen)
    rejected.prompt_embeds = rejected.prompt_embeds + 10.0
    chosen.extra_kwargs["advantage"] = torch.tensor(1.0)
    rejected.extra_kwargs["advantage"] = torch.tensor(-1.0)
    _assert_production_h3_sample(adapter, chosen)
    trainer, accelerator = _production_trainer(
        DPOTrainer,
        adapter,
        beta=1.0,
        timestep_range=(0.0, 1.0),
        weighting_scheme="uniform",
        time_shift=1.0,
    )
    trainer.num_train_timesteps = 1
    trainer.advantage_processor = SimpleNamespace(group_on_same_rank=True)
    adapter.training_args = SimpleNamespace(
        requires_ref_model=True,
        ref_param_device="cpu",
    )
    adapter._init_ref_parameters()
    observed_prompt_markers = []
    original_forward_state = adapter.forward_state

    def recording_forward_state(**kwargs: Any) -> Any:
        observed_prompt_markers.append(float(kwargs["batch"].prompt_embeds.flatten()[0]))
        return original_forward_state(**kwargs)

    adapter.forward_state = recording_forward_state

    result = trainer.optimize([chosen, rejected])

    assert result is None
    assert accelerator.backward_calls == 1
    assert accelerator.observed_grad
    assert observed_prompt_markers == [
        float(chosen.prompt_embeds.flatten()[0]),
        float(rejected.prompt_embeds.flatten()[0]),
        float(chosen.prompt_embeds.flatten()[0]),
        float(rejected.prompt_embeds.flatten()[0]),
    ]


def test_no_weight_dgpo_optimize_uses_h3_group_noising_and_forward(
    monkeypatch,
) -> None:
    adapter = _production_adapter()
    positive = _run_production_inference(
        monkeypatch,
        trajectory_indices=[-1],
        adapter=adapter,
    )
    positive.extra_kwargs["advantage"] = torch.tensor(1.0)
    _assert_production_h3_sample(adapter, positive)
    trainer, accelerator = _production_trainer(
        DGPOTrainer,
        adapter,
        per_device_batch_size=1,
        group_size=1,
    )
    trainer.dpo_beta = 1.0
    trainer.use_shared_noise = True
    trainer.clip_dsm = False
    trainer.clip_kl = False
    trainer.use_ema_ref = False
    trainer.kl_cfg = 1.0
    trainer.time_sampling_strategy = "discrete"
    trainer.time_shift = 1.0
    trainer.num_train_timesteps = 1
    trainer.timestep_range = (0.0, 1.0)
    trainer.kl_beta = 0.0
    trainer.kl_type = "v-based"
    trainer._requires_ema_ref = False

    result = trainer.optimize([positive])

    assert result is None
    assert accelerator.backward_calls == 1
    assert accelerator.observed_grad


def test_no_weight_crd_optimize_replays_h3_noise_without_redraw(
    monkeypatch,
) -> None:
    adapter = _production_adapter()
    sample = _run_production_inference(
        monkeypatch,
        trajectory_indices=[-1],
        adapter=adapter,
    )
    sample.extra_kwargs["advantage"] = torch.tensor(1.0)
    _assert_production_h3_sample(adapter, sample)
    trainer, accelerator = _production_trainer(CRDTrainer, adapter)
    trainer.crd_beta = 1.0
    trainer.crd_loss_type = "mse"
    trainer.use_old_for_loss = False
    trainer.adaptive_logp = False
    trainer.weight_temp = -1.0
    trainer.kl_beta = 0.0
    trainer.kl_cfg = 1.0
    trainer.reward_adaptive_kl = False
    trainer.time_sampling_strategy = "discrete"
    trainer.time_shift = 1.0
    trainer.num_train_timesteps = 1
    trainer.timestep_range = (0.0, 1.0)
    trainer.kl_type = "v-based"

    result = trainer.optimize([sample])

    assert result is None
    assert accelerator.backward_calls == 1
    assert accelerator.observed_grad


def test_no_weight_opd_optimize_uses_neutral_h3_structured_replay(
    monkeypatch,
) -> None:
    adapter = _production_adapter()
    sample = _run_production_inference(
        monkeypatch,
        trajectory_indices="all",
        adapter=adapter,
    )
    _assert_production_h3_sample(adapter, sample)
    trainer, accelerator = _production_trainer(
        DiffusionOPDTrainer,
        adapter,
        num_inference_steps=2,
        timestep_range=(0.0, 0.5),
        guidance_scale=1.0,
        loss_target="v",
        self_normalize=False,
    )
    adapter.add_named_parameters("teacher")
    with torch.no_grad():
        adapter.get_component("transformer").weight.add_(0.1)
    trainer._teacher_names = ["teacher"]
    trainer._teacher_gs = [1.0]
    trainer._source_to_teacher = {"dataset": 0}
    trainer._available_sources = {"dataset"}
    trainer._teacher_target_store_device = torch.device("cpu")
    trainer._is_sde = True
    trainer._student_noise_level = float(adapter.scheduler.noise_level)

    result = trainer.optimize([sample])

    assert result is None
    assert accelerator.backward_calls == 1
    assert accelerator.observed_grad


def _gloo_structured_optimize_worker(
    rank: int,
    world_size: int,
    store_path: str,
    result_dir: str,
) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{store_path}",
        rank=rank,
        world_size=world_size,
    )
    dgpo_adapter = _production_adapter()
    dgpo_sample = _run_production_inference(
        None,
        trajectory_indices=[-1],
        adapter=dgpo_adapter,
    )
    dgpo_sample.extra_kwargs["advantage"] = torch.tensor(1.0 if rank == 0 else -1.0)
    dgpo_accelerator = GlooAccelerator(
        dgpo_adapter.get_component("transformer").weight,
        rank,
        world_size,
    )
    dgpo, _ = _production_trainer(
        DGPOTrainer,
        dgpo_adapter,
        accelerator=dgpo_accelerator,
        per_device_batch_size=1,
        group_size=2,
    )
    dgpo.dpo_beta = 1.0
    dgpo.use_shared_noise = True
    dgpo.clip_dsm = False
    dgpo.clip_kl = False
    dgpo.use_ema_ref = False
    dgpo.kl_cfg = 1.0
    dgpo.time_sampling_strategy = "discrete"
    dgpo.time_shift = 1.0
    dgpo.num_train_timesteps = 1
    dgpo.timestep_range = (0.0, 1.0)
    dgpo.kl_beta = 0.0
    dgpo.kl_type = "v-based"
    dgpo._requires_ema_ref = False
    dgpo.optimize([dgpo_sample])

    crd_adapter = _production_adapter()
    crd_sample = _run_production_inference(
        None,
        trajectory_indices=[-1],
        adapter=crd_adapter,
    )
    crd_sample.extra_kwargs["advantage"] = torch.tensor(1.0 if rank == 0 else -1.0)
    crd_accelerator = GlooAccelerator(
        crd_adapter.get_component("transformer").weight,
        rank,
        world_size,
    )
    crd, _ = _production_trainer(
        CRDTrainer,
        crd_adapter,
        accelerator=crd_accelerator,
    )
    crd.crd_beta = 1.0
    crd.crd_loss_type = "mse"
    crd.use_old_for_loss = False
    crd.adaptive_logp = False
    crd.weight_temp = -1.0
    crd.kl_beta = 0.0
    crd.kl_cfg = 1.0
    crd.reward_adaptive_kl = False
    crd.time_sampling_strategy = "discrete"
    crd.time_shift = 1.0
    crd.num_train_timesteps = 1
    crd.timestep_range = (0.0, 1.0)
    crd.kl_type = "v-based"
    crd.optimize([crd_sample])

    Path(result_dir, f"rank-{rank}.txt").write_text(
        f"{dgpo_accelerator.reduce_calls},{crd_accelerator.gather_calls}",
        encoding="utf-8",
    )
    dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="PyTorch gloo backend unavailable",
)
def test_no_weight_two_rank_gloo_runs_structured_dgpo_and_crd(
    tmp_path,
) -> None:
    world_size = 2
    context = mp.get_context("spawn")
    store_path = str(tmp_path / "gloo-store")
    processes = [
        context.Process(
            target=_gloo_structured_optimize_worker,
            args=(rank, world_size, store_path, str(tmp_path)),
        )
        for rank in range(world_size)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=45)
    hanging = [process for process in processes if process.is_alive()]
    for process in hanging:
        process.terminate()
        process.join(timeout=5)
    assert not hanging, "two-rank H3 gloo optimize smoke exceeded 45 seconds"
    assert [process.exitcode for process in processes] == [0, 0]
    for rank in range(world_size):
        dgpo_reduce_calls, crd_gather_calls = (
            Path(tmp_path, f"rank-{rank}.txt").read_text(encoding="utf-8").split(",")
        )
        assert int(dgpo_reduce_calls) >= 1
        assert int(crd_gather_calls) == 1
