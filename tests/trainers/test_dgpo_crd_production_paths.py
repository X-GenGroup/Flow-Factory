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

"""Production-path coverage for the migrated DGPO and CRD trainers.

The parity suite in ``test_dgpo_crd_parity.py`` pins each helper in isolation.
This module drives the real ``optimize()`` entry points end to end with a tiny
trainable adapter, so gradient ownership, parameter-swap scopes, backward /
optimizer cadence, logging, distributed reduction and the pre-migration
numerical oracle are all exercised through the code path production runs.
"""

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pytest
import torch

from diffusers.utils.torch_utils import randn_tensor
from flow_factory.models.abc import BaseAdapter
from flow_factory.samples import BaseSample, ComponentTimes, LatentState
from flow_factory.scheduler import SDESchedulerOutput
from flow_factory.trainers.common.group_noise import (
    STREAM_FORWARD_NOISE,
    identity_timestep_generator,
    independent_batch_generator,
    sample_noise_ids,
)
from flow_factory.trainers.rl.crd import CRDTrainer, _CRDStep
from flow_factory.trainers.rl.dgpo import (
    _SEED_TAG_SHARED_NOISE,
    _SEED_TAG_SHARED_TIMESTEPS,
    DGPOTrainer,
)
from flow_factory.utils.base import create_generator, to_broadcast_tensor
from flow_factory.utils.noise_schedule import TimeSampler, flow_match_sigma

# Per-scope additive bias, so a parameter swap is observable in the output and a
# forgotten swap cannot be mistaken for a correct one.
_SCOPE_BIAS: Dict[str, float] = {
    "policy": 0.0,
    "ref": 0.25,
    "ema_ref": -0.5,
    CRDTrainer._OLD_PARAMS_NAME: 0.125,
    CRDTrainer._SAMPLING_PARAMS_NAME: 0.375,
}


class TrainingArgsFake(dict):
    """Mapping/attribute hybrid mirroring ``ArgABC`` unpacking behaviour."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as error:
            raise AttributeError(key) from error


class SchedulerFake:
    """Scheduler stub recording the seeds the trainer dispatches."""

    def __init__(self) -> None:
        self.noise_level = 0.7
        self.train_timesteps = torch.tensor([0, 1])
        self.seeds: List[int] = []

    def step(self) -> None:
        """Provide scheduler compatibility."""

    def set_seed(self, seed: int) -> None:
        """Record the dispatched seed."""
        self.seeds.append(seed)


class TrainableAdapterFake(BaseAdapter):
    """One trainable scalar plus a per-scope bias, wired into ``velocity``.

    ``velocity = latents * weight + bias(scope)`` keeps the forward
    differentiable w.r.t. a single parameter, which is what the gradient and
    oracle assertions compare, while the scope bias makes every parameter swap
    visible in the returned tensor.
    """

    def load_pipeline(self) -> Any:
        """Return an unused pipeline fake."""
        raise NotImplementedError

    def decode_latents(self, latents: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Return latents unchanged."""
        return latents

    def inference(self, **kwargs: Any) -> List[BaseSample]:
        """Return no samples."""
        return []

    def forward(self, **kwargs: Any) -> SDESchedulerOutput:
        """Record the call and return the scope-dependent velocity."""
        latents = kwargs["latents"]
        velocity = self.velocity(latents, self.active_scope)
        self.forward_calls.append(
            {
                "scope": self.active_scope,
                "latents": latents.detach().clone(),
                "t": kwargs["t"].detach().clone(),
                "guidance_scale": kwargs.get("guidance_scale"),
                "velocity_requires_grad": velocity.requires_grad,
            }
        )
        self.events.append(f"forward:{self.active_scope}")
        return SDESchedulerOutput(velocity=velocity)

    def velocity(self, latents: torch.Tensor, scope: str) -> torch.Tensor:
        """Model output for a scope; reused verbatim by the legacy oracles."""
        return latents * self.weight + _SCOPE_BIAS[scope]

    def get_trainable_parameters(self) -> List[torch.Tensor]:
        """Expose the single trainable scalar."""
        return [self.weight]

    def get_named_parameters(self, name: str) -> List[torch.Tensor]:
        """Expose a named snapshot's storage for in-place blending."""
        return self.named_snapshots[name]

    def rollout(self, *args: Any, **kwargs: Any) -> None:
        """Record the rollout-mode switch."""
        self.events.append("mode:rollout")

    def train(self, mode: bool = True) -> None:
        """Record the train-mode switch."""
        self.events.append("mode:train" if mode else "mode:eval")

    def ema_step(self, step: int) -> None:
        """Record the sampling-EMA advance."""
        self.events.append(f"ema_step:{step}")

    @contextmanager
    def _scope(self, name: str) -> Iterator[None]:
        previous = self.active_scope
        self.active_scope = name
        self.events.append(f"enter:{name}")
        try:
            yield
        finally:
            self.active_scope = previous
            self.events.append(f"exit:{name}")

    @contextmanager
    def use_ref_parameters(self) -> Iterator[None]:
        """Record the reference-parameter scope."""
        with self._scope("ref"):
            yield

    @contextmanager
    def use_ema_parameters(self) -> Iterator[None]:
        """Record the sampling-EMA scope."""
        with self._scope("ema"):
            yield

    @contextmanager
    def use_named_parameters(self, name: str) -> Iterator[None]:
        """Record a named-snapshot scope."""
        with self._scope(name):
            yield


def _adapter(weight: float = 0.7) -> TrainableAdapterFake:
    adapter = object.__new__(TrainableAdapterFake)
    adapter.pipeline = SimpleNamespace(scheduler=SchedulerFake())
    adapter.scheduler_group = adapter.build_scheduler_group()
    adapter.weight = torch.nn.Parameter(torch.tensor(weight))
    adapter.named_snapshots = {
        "ema_ref": [torch.tensor(0.1)],
        CRDTrainer._OLD_PARAMS_NAME: [torch.tensor(0.2)],
        CRDTrainer._SAMPLING_PARAMS_NAME: [torch.tensor(0.3)],
    }
    adapter.active_scope = "policy"
    adapter.forward_calls = []
    adapter.events = []
    return adapter


class OptimizerFake:
    """Optimizer recording its call order into the shared event log."""

    def __init__(self, events: List[str]) -> None:
        self.events = events

    def step(self) -> None:
        """Record the optimizer step."""
        self.events.append("optimizer:step")

    def zero_grad(self) -> None:
        """Record the gradient reset."""
        self.events.append("optimizer:zero_grad")


class AcceleratorFake:
    """Single-process accelerator recording backward / clip / reduce calls."""

    def __init__(self, events: List[str], num_processes: int = 1) -> None:
        self.events = events
        self.device = torch.device("cpu")
        self.num_processes = num_processes
        self.sync_gradients = True
        self.is_main_process = True
        self.is_local_main_process = False
        self.losses: List[torch.Tensor] = []
        self.reduced: List[Any] = []

    @contextmanager
    def accumulate(self, model: Any) -> Iterator[None]:
        """Record the accumulation window."""
        self.events.append("accumulate:enter")
        try:
            yield
        finally:
            self.events.append("accumulate:exit")

    def backward(self, loss: torch.Tensor) -> None:
        """Record the loss and run the real backward."""
        self.events.append("backward")
        self.losses.append(loss.detach().clone())
        loss.backward()

    def clip_grad_norm_(self, parameters: Any, max_norm: float) -> torch.Tensor:
        """Record the clip call without touching the gradients."""
        self.events.append("clip_grad_norm")
        return torch.tensor(1.5)

    def reduce(self, tensor: Any, reduction: str = "sum") -> Any:
        """Return the local value; a no-op in single-process contexts."""
        self.reduced.append((reduction, tensor))
        return tensor

    def gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """Return the local tensor; a no-op in single-process contexts."""
        return tensor


class PeerSumAcceleratorFake(AcceleratorFake):
    """Two-rank accelerator adding a fixed peer contribution to sum reductions.

    Only the group-sum collective in ``_compute_group_dgpo_loss`` uses
    ``reduction="sum"``; the metric reduction inside ``_finalize_step`` uses
    ``"mean"`` over a dict and is left alone.
    """

    def __init__(self, events: List[str], peer_sums: torch.Tensor) -> None:
        super().__init__(events, num_processes=2)
        self.peer_sums = peer_sums
        self.group_sum_inputs: List[torch.Tensor] = []
        self.group_sum_outputs: List[torch.Tensor] = []

    def reduce(self, tensor: Any, reduction: str = "sum") -> Any:
        """Sum the rank-local group contributions with a recorded peer rank."""
        if reduction == "sum" and isinstance(tensor, torch.Tensor):
            self.group_sum_inputs.append(tensor.detach().clone())
            summed = tensor + self.peer_sums
            self.group_sum_outputs.append(summed.detach().clone())
            return summed
        return super().reduce(tensor, reduction)


class LoggerFake:
    """Logging backend recording every payload the trainer emits."""

    def __init__(self, events: List[str]) -> None:
        self.events = events
        self.payloads: List[Tuple[int, Dict[str, Any]]] = []

    def log_data(self, data: Dict[str, Any], step: int) -> None:
        """Record one logged payload."""
        self.events.append("log")
        self.payloads.append((step, dict(data)))


def _samples(
    unique_ids: List[int],
    values: List[float],
    advantages: Optional[List[float]] = None,
) -> List[BaseSample]:
    """Terminal-only rollout samples whose single stored latent is the terminal.

    ``all_latents`` holds exactly one column so the pre-migration ``[:, -1]``
    slice and the migrated ``latent_index_map`` read resolve to the same tensor
    — the precondition for comparing against a legacy oracle at all.
    """
    advantages = advantages if advantages is not None else values
    return [
        BaseSample(
            timesteps=torch.tensor([1000.0, 0.0]),
            all_latents=torch.tensor([[value, value + 1.0, value + 2.0]]),
            latent_index_map=torch.tensor([-1, -1, 0]),
            prompt_embeds=torch.tensor([value]),
            extra_kwargs={"advantage": torch.tensor(advantage)},
            _unique_id=unique_id,
            # Training draws are seeded from the sampler-assigned identity, which
            # production supplies through `_attach_batch_provenance`. Keyed to
            # `unique_id` so distinct samples stay distinct, as in the parity test.
            sampling_sample_id=unique_id,
        )
        for unique_id, value, advantage in zip(unique_ids, values, advantages)
    ]


# ============================ DGPO production path ============================


def _dgpo_trainer(
    adapter: TrainableAdapterFake,
    *,
    accelerator: Optional[AcceleratorFake] = None,
    events: Optional[List[str]] = None,
    use_shared_noise: bool = True,
    clip_dsm: bool = False,
    clip_kl: bool = False,
    use_ema_ref: bool = False,
    kl_beta: float = 0.0,
    kl_cfg: float = 1.0,
    dpo_beta: float = 4.0,
    group_size: int = 2,
    num_train_timesteps: int = 1,
    per_device_batch_size: int = 2,
    num_inner_epochs: int = 1,
) -> DGPOTrainer:
    """Assemble a DGPO trainer whose ``optimize()`` runs end to end."""
    events = events if events is not None else adapter.events
    trainer = object.__new__(DGPOTrainer)
    trainer.adapter = adapter
    trainer.training_args = TrainingArgsFake(
        seed=7,
        group_size=group_size,
        clip_range=(-0.2, 0.2),
        adv_clip_range=(-5.0, 5.0),
        max_grad_norm=1.0,
        per_device_batch_size=per_device_batch_size,
        num_inner_epochs=num_inner_epochs,
    )
    trainer.autocast = nullcontext
    trainer.accelerator = accelerator if accelerator is not None else AcceleratorFake(events)
    trainer.optimizer = OptimizerFake(events)
    # `_apply_optimizer_step` hands the prepared root's parameters to the clip helper.
    trainer.model_bundle = torch.nn.Module()
    trainer.logger = LoggerFake(events)
    trainer.log_args = SimpleNamespace(verbose=False, save_freq=0, save_dir=None, run_name="run")
    trainer.epoch = 3
    trainer.step = 0
    trainer.dpo_beta = dpo_beta
    trainer.use_shared_noise = use_shared_noise
    trainer.clip_dsm = clip_dsm
    trainer.clip_kl = clip_kl
    trainer.use_ema_ref = use_ema_ref
    trainer.kl_beta = kl_beta
    trainer.kl_cfg = kl_cfg
    trainer.kl_type = "v-based"
    trainer.off_policy = False
    trainer.switch_ema_ref = 0
    trainer.ema_ref_max_decay = 0.9
    trainer.ema_ref_ramp_rate = 0.01
    trainer.num_train_timesteps = num_train_timesteps
    trainer.time_sampling_strategy = "uniform"
    trainer.time_shift = 1.0
    trainer.timestep_range = (0.0, 1.0)
    trainer._requires_ema_ref = clip_dsm or clip_kl or use_ema_ref
    return trainer


def _dgpo_shared_timesteps(trainer: DGPOTrainer, inner_epoch: int = 0) -> torch.Tensor:
    """Reproduce ``_sample_shared_timesteps`` outside the trainer."""
    generator = create_generator(
        trainer.training_args.seed,
        trainer.epoch,
        inner_epoch,
        _SEED_TAG_SHARED_TIMESTEPS,
    )
    return TimeSampler.uniform(
        batch_size=1,
        num_timesteps=trainer.num_train_timesteps,
        timestep_range=trainer.timestep_range,
        time_shift=trainer.time_shift,
        device=torch.device("cpu"),
        generator=generator,
    ).squeeze(-1)


def _legacy_shared_noise(
    trainer: DGPOTrainer,
    clean: torch.Tensor,
    unique_ids: List[int],
    inner_epoch: int,
    timestep_index: int = 0,
) -> torch.Tensor:
    """Reproduce the per-timestep group noise seed, including ``timestep_index``."""
    cache: Dict[int, torch.Tensor] = {}
    rows: List[torch.Tensor] = []
    for unique_id in unique_ids:
        noise = cache.get(unique_id)
        if noise is None:
            generator = create_generator(
                trainer.training_args.seed,
                trainer.epoch,
                inner_epoch,
                int(unique_id),
                _SEED_TAG_SHARED_NOISE,
                timestep_index,
                device=clean.device,
            )
            noise = randn_tensor(
                clean.shape[1:], generator=generator, device=clean.device, dtype=clean.dtype
            )
            cache[unique_id] = noise
        rows.append(noise)
    return torch.stack(rows, dim=0)


def _legacy_dgpo_step(
    trainer: DGPOTrainer,
    adapter: TrainableAdapterFake,
    weight: torch.nn.Parameter,
    samples: List[BaseSample],
    *,
    peer_sums: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Pre-migration DGPO loss for one micro-batch and one timestep.

    Written against raw tensors and the legacy reductions
    (``.reshape(B, -1).mean(1)``) so it is an independent oracle rather than a
    second call into the migrated helpers.
    """
    batch = BaseSample.stack(samples)
    clean = batch["all_latents"][:, -1]
    unique_ids = [int(sample.unique_id) for sample in samples]

    shared_timesteps = _dgpo_shared_timesteps(trainer)
    t_flat = shared_timesteps.unsqueeze(1).expand(-1, clean.shape[0])[0]
    sigma = to_broadcast_tensor(flow_match_sigma(t_flat), clean)
    noise = _legacy_shared_noise(trainer, clean, unique_ids, inner_epoch=0)
    noised = (1 - sigma) * clean + sigma * noise
    target_v = noise - clean

    def velocity(scope: str) -> torch.Tensor:
        return noised * weight + _SCOPE_BIAS[scope]

    def dsm(prediction: torch.Tensor) -> torch.Tensor:
        return (target_v - prediction).square().reshape(clean.shape[0], -1).mean(dim=1)

    adv_clip_range = trainer.training_args.adv_clip_range
    adv = torch.clamp(batch["advantage"], adv_clip_range[0], adv_clip_range[1])

    model_v = velocity("policy")
    with torch.no_grad():
        old_v = velocity("ema_ref") if trainer._requires_ema_ref else None
        ref_v = velocity("ref") if (trainer.enable_kl_loss or not trainer.use_ema_ref) else None
    ref_dgpo_v = old_v if trainer.use_ema_ref else ref_v

    dsm_loss = dsm(model_v)
    should_clip: Optional[torch.Tensor] = None
    if (trainer.clip_dsm or trainer.clip_kl) and old_v is not None:
        clip_range = trainer.training_args.clip_range
        ratio = torch.exp(-dsm_loss.detach() + dsm(old_v))
        should_clip = torch.where(adv > 0, ratio > 1.0 + clip_range[1], ratio < 1.0 + clip_range[0])
        if trainer.clip_dsm:
            dsm_loss = torch.where(should_clip, dsm_loss.detach(), dsm_loss)

    with torch.no_grad():
        ref_dsm = dsm(ref_dgpo_v)
    per_sample = (
        adv * trainer.dpo_beta * (dsm_loss.detach() - ref_dsm) / trainer.training_args.group_size
    )
    local_uids = torch.as_tensor(unique_ids, dtype=torch.int64)
    _, inverse = torch.unique(local_uids, return_inverse=True)
    local_sums = torch.zeros(int(inverse.max().item()) + 1, dtype=per_sample.dtype)
    local_sums.scatter_add_(0, inverse, per_sample)
    global_sums = local_sums.detach() if peer_sums is None else local_sums.detach() + peer_sums
    group_weights = torch.sigmoid(global_sums)[inverse].detach()
    dgpo_loss = (group_weights * adv * dsm_loss).mean()

    loss = dgpo_loss
    if trainer.enable_kl_loss and ref_v is not None:
        kl_div = (model_v - ref_v).square().reshape(clean.shape[0], -1).mean(dim=1)
        if trainer.clip_kl and should_clip is not None:
            kl_div = torch.where(should_clip, kl_div.detach(), kl_div)
        loss = loss + trainer.kl_beta * kl_div.mean()
    return {"loss": loss, "dgpo_loss": dgpo_loss, "group_sums": global_sums}


def test_dgpo_optimize_runs_the_real_loop_with_correct_gradient_ownership() -> None:
    adapter = _adapter()
    trainer = _dgpo_trainer(adapter, clip_dsm=True, kl_beta=0.05, kl_cfg=2.0)
    samples = _samples([4, 4], [1.0, 2.0])

    trainer.optimize(samples)

    scopes = [call["scope"] for call in adapter.forward_calls]
    assert scopes == ["ema_ref", "policy", "ref"]
    # Only the current-policy forward may own the graph.
    assert [call["velocity_requires_grad"] for call in adapter.forward_calls] == [
        False,
        True,
        False,
    ]
    assert adapter.weight.grad is not None
    assert len(trainer.accelerator.losses) == 1
    assert trainer.accelerator.losses[0].requires_grad is False
    # The old / reference forwards ran under their own swap and CFG override.
    assert adapter.forward_calls[0]["guidance_scale"] == 1.0
    assert adapter.forward_calls[1]["guidance_scale"] == 1.0
    assert adapter.forward_calls[2]["guidance_scale"] == 2.0


def test_dgpo_optimize_keeps_the_mode_scope_and_optimizer_call_order() -> None:
    adapter = _adapter()
    events: List[str] = []
    accelerator = AcceleratorFake(events)
    adapter.events = events
    trainer = _dgpo_trainer(adapter, accelerator=accelerator, events=events, use_ema_ref=True)

    trainer.optimize(_samples([4, 4], [1.0, 2.0]))

    assert events == [
        "mode:rollout",
        "mode:train",
        "accumulate:enter",
        "enter:ema_ref",
        "forward:ema_ref",
        "exit:ema_ref",
        "forward:policy",
        "backward",
        "clip_grad_norm",
        "optimizer:step",
        "optimizer:zero_grad",
        "log",
        "accumulate:exit",
    ]
    assert trainer.step == 1


def test_dgpo_optimize_logs_the_expected_metric_keys() -> None:
    adapter = _adapter()
    trainer = _dgpo_trainer(adapter, clip_dsm=True, kl_beta=0.05)

    trainer.optimize(_samples([4, 4], [1.0, 2.0]))

    step, payload = trainer.logger.payloads[0]
    assert step == 0
    assert set(payload) == {
        "train/clip_ratio",
        "train/dsm_loss",
        "train/kl_div",
        "train/kl_loss",
        "train/dgpo_loss",
        "train/loss",
        "train/grad_norm",
    }


def test_dgpo_optimize_advances_the_ema_ref_snapshot_once_per_step() -> None:
    adapter = _adapter()
    trainer = _dgpo_trainer(adapter, use_ema_ref=True)
    trainer.step = 20
    before = adapter.named_snapshots["ema_ref"][0].clone()

    trainer.optimize(_samples([4, 4], [1.0, 2.0]))

    decay = min(trainer.ema_ref_max_decay, trainer.ema_ref_ramp_rate * 20)
    expected = before * decay + adapter.weight.detach() * (1.0 - decay)
    assert torch.equal(adapter.named_snapshots["ema_ref"][0], expected)


def test_dgpo_optimize_consumes_no_global_rng_under_shared_noise() -> None:
    adapter = _adapter()
    trainer = _dgpo_trainer(adapter, use_shared_noise=True)
    torch.manual_seed(1234)
    before = torch.get_rng_state()

    trainer.optimize(_samples([4, 4], [1.0, 2.0]))

    assert torch.equal(torch.get_rng_state(), before)


def test_dgpo_optimize_matches_the_legacy_single_step_loss_and_gradient() -> None:
    samples = _samples([4, 4], [1.0, 2.0])

    adapter = _adapter()
    trainer = _dgpo_trainer(adapter, clip_dsm=True, clip_kl=True, kl_beta=0.05)
    trainer.optimize(samples)

    oracle_adapter = _adapter()
    oracle_trainer = _dgpo_trainer(oracle_adapter, clip_dsm=True, clip_kl=True, kl_beta=0.05)
    legacy = _legacy_dgpo_step(oracle_trainer, oracle_adapter, oracle_adapter.weight, samples)
    legacy["loss"].backward()

    assert torch.equal(trainer.accelerator.losses[0], legacy["loss"].detach())
    assert torch.equal(adapter.weight.grad, oracle_adapter.weight.grad)


def test_dgpo_optimize_matches_the_legacy_optimizer_and_draw_cadence() -> None:
    samples = _samples([4, 4], [1.0, 2.0])
    adapter = _adapter()
    events: List[str] = []
    adapter.events = events
    trainer = _dgpo_trainer(adapter, accelerator=AcceleratorFake(events), events=events)

    torch.manual_seed(4321)
    before = torch.get_rng_state()
    trainer.optimize(samples)

    # The legacy step drew shared noise and shared timesteps from explicit
    # generators too, so a faithful migration touches the global stream zero times.
    assert torch.equal(torch.get_rng_state(), before)
    assert [event for event in events if event.startswith("optimizer")] == [
        "optimizer:step",
        "optimizer:zero_grad",
    ]
    assert events.index("backward") < events.index("clip_grad_norm")
    assert events.index("clip_grad_norm") < events.index("optimizer:step")


def test_dgpo_group_reduction_sums_rank_local_contributions_before_the_sigmoid() -> None:
    samples = _samples([4, 4], [1.0, 2.0])
    peer_sums = torch.tensor([0.75])

    adapter = _adapter()
    events: List[str] = []
    adapter.events = events
    accelerator = PeerSumAcceleratorFake(events, peer_sums)
    trainer = _dgpo_trainer(adapter, accelerator=accelerator, events=events)
    trainer.optimize(samples)

    oracle_adapter = _adapter()
    oracle_trainer = _dgpo_trainer(oracle_adapter)
    legacy = _legacy_dgpo_step(
        oracle_trainer, oracle_adapter, oracle_adapter.weight, samples, peer_sums=peer_sums
    )

    assert len(accelerator.group_sum_inputs) == 1
    assert torch.equal(
        accelerator.group_sum_outputs[0], accelerator.group_sum_inputs[0] + peer_sums
    )
    assert torch.equal(accelerator.group_sum_outputs[0], legacy["group_sums"])
    assert torch.equal(trainer.accelerator.losses[0], legacy["loss"].detach())


def test_dgpo_shared_noise_is_row_order_invariant_per_uid_and_component() -> None:
    """Ranks holding the same group in a different row order must agree bit-wise."""
    adapter = _adapter()
    trainer = _dgpo_trainer(adapter)
    clean_state = LatentState({"latent": torch.zeros(3, 4)})

    forward_order = trainer._shared_group_noise(
        clean_state, _samples([8, 9, 8], [1.0, 2.0, 3.0]), inner_epoch=1, timestep_index=0
    ).components["latent"]
    shuffled_order = trainer._shared_group_noise(
        clean_state, _samples([9, 8, 8], [3.0, 1.0, 2.0]), inner_epoch=1, timestep_index=0
    ).components["latent"]

    assert torch.equal(forward_order[0], shuffled_order[1])
    assert torch.equal(forward_order[2], shuffled_order[2])
    assert torch.equal(forward_order[1], shuffled_order[0])
    assert not torch.equal(forward_order[0], forward_order[1])


def test_dgpo_shared_noise_is_row_order_invariant_across_components() -> None:
    """Row order must not leak into any component's draw, not just the primary."""
    adapter = _adapter()
    adapter.trajectory_component_order = ("video", "audio")
    trainer = _dgpo_trainer(adapter)
    clean_state = LatentState({"video": torch.zeros(2, 3), "audio": torch.zeros(2, 5)})

    forward_order = trainer._shared_group_noise(
        clean_state, _samples([2, 6], [1.0, 2.0]), inner_epoch=0, timestep_index=0
    )
    reversed_order = trainer._shared_group_noise(
        clean_state, _samples([6, 2], [2.0, 1.0]), inner_epoch=0, timestep_index=0
    )

    for name in ("video", "audio"):
        assert torch.equal(forward_order.components[name][0], reversed_order.components[name][1])
        assert torch.equal(forward_order.components[name][1], reversed_order.components[name][0])


def test_dgpo_missing_old_velocity_raises_with_the_active_flags() -> None:
    """``use_ema_ref`` without an EMA ref is a config bug, not an ``AssertionError``."""
    adapter = _adapter()
    trainer = _dgpo_trainer(adapter, use_ema_ref=True)
    trainer._requires_ema_ref = False

    with pytest.raises(
        ValueError,
        match=r"DGPOTrainer.*old policy velocity.*use_ema_ref=True.*_requires_ema_ref=False",
    ):
        trainer.optimize(_samples([4, 4], [1.0, 2.0]))


def test_dgpo_missing_reference_velocity_raises_with_the_active_flags() -> None:
    """The reference selection is unreachable-``None`` today, so guard it directly."""
    trainer = _dgpo_trainer(_adapter(), use_ema_ref=False, kl_beta=0.0)

    with pytest.raises(
        ValueError,
        match=r"DGPOTrainer.*reference velocity.*use_ema_ref=False.*kl_beta=",
    ):
        trainer._select_dgpo_reference(LatentState({"latent": torch.zeros(2, 3)}), None)


def test_dgpo_reference_selection_follows_the_use_ema_ref_flag() -> None:
    old_v = LatentState({"latent": torch.zeros(2, 3)})
    ref_v = LatentState({"latent": torch.ones(2, 3)})

    ema_trainer = _dgpo_trainer(_adapter(), use_ema_ref=True)
    plain_trainer = _dgpo_trainer(_adapter(), use_ema_ref=False)

    assert ema_trainer._select_dgpo_reference(old_v, ref_v) is old_v
    assert plain_trainer._select_dgpo_reference(old_v, ref_v) is ref_v


def test_dgpo_shared_noise_error_context_names_the_epoch_and_timestep() -> None:
    adapter = _adapter()
    trainer = _dgpo_trainer(adapter)
    clean_state = LatentState({"latent": torch.zeros(2, 4)})
    trainer._draw_group_component_noise = lambda **kwargs: torch.zeros(9)

    with pytest.raises(
        ValueError,
        match=r"inner_epoch=5.*timestep_index=2.*component order \('latent',\)",
    ):
        trainer._shared_group_noise(clean_state, _samples([1, 1], [1.0, 2.0]), 5, timestep_index=2)


def test_dgpo_build_noised_inputs_threads_the_timestep_index_into_errors() -> None:
    adapter = _adapter()
    trainer = _dgpo_trainer(adapter, num_train_timesteps=2)
    samples = _samples([1, 1], [1.0, 2.0])
    prepped = trainer._prep_training_batch(
        {
            "batch": BaseSample.stack(samples),
            "group_info": {
                "local_group_indices": torch.zeros(2, dtype=torch.int64),
                "num_groups": 1,
            },
            "timesteps": torch.tensor([[700.0, 700.0], [300.0, 300.0]]),
            "samples_slice": samples,
            "inner_epoch": 0,
        }
    )
    trainer._draw_group_component_noise = lambda **kwargs: torch.zeros(9)

    with pytest.raises(ValueError, match=r"timestep_index=1"):
        trainer._build_noised_inputs(prepped, 1)


# ============================ CRD production path ============================


def _crd_trainer(
    adapter: TrainableAdapterFake,
    *,
    events: Optional[List[str]] = None,
    accelerator: Optional[AcceleratorFake] = None,
    adaptive_logp: bool = False,
    use_old_for_loss: bool = True,
    weight_temp: float = -1.0,
    crd_beta: float = 0.5,
    crd_loss_type: str = "mse",
    kl_beta: float = 0.01,
    kl_cfg: float = 1.0,
    reward_adaptive_kl: bool = False,
    num_train_timesteps: int = 1,
    per_device_batch_size: int = 2,
    num_inner_epochs: int = 1,
) -> CRDTrainer:
    """Assemble a CRD trainer whose ``optimize()`` runs end to end."""
    events = events if events is not None else adapter.events
    trainer = object.__new__(CRDTrainer)
    trainer.adapter = adapter
    trainer.training_args = TrainingArgsFake(
        seed=11,
        adv_clip_range=(-5.0, 5.0),
        max_grad_norm=1.0,
        per_device_batch_size=per_device_batch_size,
        num_inner_epochs=num_inner_epochs,
        offload_samples_to_cpu=False,
        guidance_scale=3.0,
    )
    trainer.autocast = nullcontext
    trainer.accelerator = accelerator if accelerator is not None else AcceleratorFake(events)
    trainer.optimizer = OptimizerFake(events)
    # `_apply_optimizer_step` hands the prepared root's parameters to the clip helper.
    trainer.model_bundle = torch.nn.Module()
    trainer.logger = LoggerFake(events)
    trainer.log_args = SimpleNamespace(verbose=False, save_freq=0, save_dir=None, run_name="run")
    trainer.epoch = 2
    trainer.step = 0
    trainer.adaptive_logp = adaptive_logp
    trainer.use_old_for_loss = use_old_for_loss
    trainer.weight_temp = weight_temp
    trainer.crd_beta = crd_beta
    trainer.crd_loss_type = crd_loss_type
    trainer.kl_beta = kl_beta
    trainer.kl_cfg = kl_cfg
    trainer.kl_type = "v-based"
    trainer.reward_adaptive_kl = reward_adaptive_kl
    trainer.num_train_timesteps = num_train_timesteps
    trainer.time_sampling_strategy = "uniform"
    trainer.time_shift = 1.0
    trainer.timestep_range = (0.0, 1.0)
    trainer.old_model_decay = 0
    trainer.sampling_model_decay = 1
    return trainer


def _legacy_crd_step(
    trainer: CRDTrainer,
    weight: torch.nn.Parameter,
    samples: List[BaseSample],
) -> Dict[str, torch.Tensor]:
    """Pre-migration CRD loss for one micro-batch and one timestep.

    Draws through the same identity-keyed generators the trainer uses, so the
    comparison stays a check of CRD's *math* against an independent reference
    rather than a check of which RNG it happens to consume. Before the noise
    seeding migration both sides drew from the global RNG and the caller seeded
    it; the draws are now keyed by `(seed, epoch, sample identity)`, so seeding
    the global RNG no longer makes the two agree.
    """
    batch = BaseSample.stack(samples)
    clean = batch["all_latents"][:, -1]
    batch_size = clean.shape[0]

    identity = sample_noise_ids(samples)
    all_timesteps = TimeSampler.uniform(
        batch_size=batch_size,
        num_timesteps=trainer.num_train_timesteps,
        timestep_range=trainer.timestep_range,
        time_shift=trainer.time_shift,
        device=torch.device("cpu"),
        generator=identity_timestep_generator(trainer, identity, 0),
    )
    t_flat = all_timesteps[0]
    sigma = to_broadcast_tensor(flow_match_sigma(t_flat), clean)
    noise = randn_tensor(
        clean.shape,
        device=clean.device,
        dtype=clean.dtype,
        generator=independent_batch_generator(trainer, identity, 0, STREAM_FORWARD_NOISE),
    )
    noised = (1 - sigma) * clean + sigma * noise
    v_target = noise - clean

    old_scope = CRDTrainer._OLD_PARAMS_NAME if trainer.use_old_for_loss else "ref"
    with torch.no_grad():
        old_v = noised * weight + _SCOPE_BIAS[old_scope]
    forward_pred = noised * weight + _SCOPE_BIAS["policy"]
    with torch.no_grad():
        ref_pred = noised * weight + _SCOPE_BIAS["ref"]

    if trainer.adaptive_logp:
        with torch.no_grad():
            weight_theta = (
                torch.abs(forward_pred.double() - v_target.double())
                .mean(dim=tuple(range(1, forward_pred.ndim)), keepdim=True)
                .clip(min=1e-5)
            )
            weight_old = (
                torch.abs(old_v.double() - v_target.double())
                .mean(dim=tuple(range(1, old_v.ndim)), keepdim=True)
                .clip(min=1e-5)
            )
        r_theta = -(
            (forward_pred - v_target) ** 2 / weight_theta - (old_v - v_target) ** 2 / weight_old
        )
    else:
        r_theta = -((forward_pred - v_target) ** 2 - (old_v - v_target) ** 2)
    r_theta_local = r_theta.mean(dim=tuple(range(1, r_theta.ndim)))
    r_theta_gathered = r_theta_local.detach()

    adv_clip_range = trainer.training_args.adv_clip_range
    adv_clipped = torch.clamp(batch["advantage"], adv_clip_range[0], adv_clip_range[1])
    adv_cur_rank = torch.clamp((adv_clipped / max(adv_clip_range)) / 2.0 + 0.5, 0, 1)
    adv_cur = adv_cur_rank.detach()

    weight_temp = torch.inf if trainer.weight_temp < 0 else trainer.weight_temp
    softmax_p = torch.softmax(adv_cur / weight_temp, dim=0)
    adv_avg = (adv_cur * softmax_p).sum(dim=0, keepdim=True)
    reward_avg = (r_theta_gathered * softmax_p).sum(dim=0, keepdim=True)
    centered_adv = adv_cur_rank - adv_avg
    centered_reward = r_theta_local - reward_avg.detach()
    ori_policy_loss = ((trainer.crd_beta * centered_reward - centered_adv) ** 2).mean()
    policy_loss = (ori_policy_loss * adv_clip_range[1] / max(trainer.crd_beta, 1e-8)).mean()

    kl_div = ((forward_pred - ref_pred) ** 2).mean(dim=tuple(range(1, forward_pred.ndim)))
    if trainer.reward_adaptive_kl:
        min_coef = 1e-4 / max(trainer.kl_beta, 1e-8)
        kl_loss = trainer.kl_beta * torch.mean((min_coef + adv_cur_rank * (1 - min_coef)) * kl_div)
    else:
        kl_loss = trainer.kl_beta * kl_div.mean()
    return {"loss": policy_loss + kl_loss, "r_theta": r_theta_local}


def test_crd_optimize_runs_pass_one_before_pass_two_with_the_right_scopes() -> None:
    adapter = _adapter()
    events: List[str] = []
    adapter.events = events
    trainer = _crd_trainer(adapter, events=events, accelerator=AcceleratorFake(events))

    trainer.optimize(_samples([4, 4], [1.0, 2.0]))

    assert events == [
        "mode:rollout",
        f"enter:{CRDTrainer._OLD_PARAMS_NAME}",
        f"forward:{CRDTrainer._OLD_PARAMS_NAME}",
        f"exit:{CRDTrainer._OLD_PARAMS_NAME}",
        "mode:train",
        "accumulate:enter",
        "forward:policy",
        "enter:ref",
        "forward:ref",
        "exit:ref",
        "backward",
        "clip_grad_norm",
        "optimizer:step",
        "optimizer:zero_grad",
        "log",
        "accumulate:exit",
    ]


def test_crd_optimize_finishes_every_pass_one_batch_before_any_pass_two_forward() -> None:
    """The two-pass design is global: no policy forward may precede the last snapshot one."""
    adapter = _adapter()
    events: List[str] = []
    adapter.events = events
    trainer = _crd_trainer(
        adapter,
        events=events,
        accelerator=AcceleratorFake(events),
        num_train_timesteps=2,
        per_device_batch_size=2,
    )
    samples = _samples([4, 4, 9, 9], [1.0, 2.0, 3.0, 4.0])

    trainer.optimize(samples)

    old_scope = CRDTrainer._OLD_PARAMS_NAME
    steps_per_run = 4  # 2 micro-batches x 2 timesteps
    pass_one: List[str] = []
    for _ in range(steps_per_run):
        pass_one += [f"enter:{old_scope}", f"forward:{old_scope}", f"exit:{old_scope}"]
    pass_two: List[str] = []
    for _ in range(steps_per_run):
        pass_two += [
            "accumulate:enter",
            "forward:policy",
            "enter:ref",
            "forward:ref",
            "exit:ref",
            "backward",
            "clip_grad_norm",
            "optimizer:step",
            "optimizer:zero_grad",
            "log",
            "accumulate:exit",
        ]
    assert events == ["mode:rollout"] + pass_one + ["mode:train"] + pass_two

    train_switch = events.index("mode:train")
    assert (
        max(index for index, event in enumerate(events) if event == f"forward:{old_scope}")
        < train_switch
    )
    assert (
        min(
            index
            for index, event in enumerate(events)
            if event in ("forward:policy", "forward:ref")
        )
        > train_switch
    )


def test_crd_optimize_repeats_the_optimizer_cadence_once_per_batch_timestep() -> None:
    adapter = _adapter()
    events: List[str] = []
    adapter.events = events
    trainer = _crd_trainer(
        adapter,
        events=events,
        accelerator=AcceleratorFake(events),
        num_train_timesteps=2,
        per_device_batch_size=2,
    )

    trainer.optimize(_samples([4, 4, 9, 9], [1.0, 2.0, 3.0, 4.0]))

    for event in ("backward", "clip_grad_norm", "optimizer:step", "optimizer:zero_grad", "log"):
        assert events.count(event) == 4, event
    assert len(trainer.accelerator.losses) == 4
    assert [step for step, _ in trainer.logger.payloads] == [0, 1, 2, 3]
    assert trainer.step == 4
    # Each micro-batch keeps its own pass-1 timesteps; the two batches carry
    # different conditioning, so the recorded policy inputs must differ.
    policy_latents = [
        call["latents"] for call in adapter.forward_calls if call["scope"] == "policy"
    ]
    assert len(policy_latents) == 4
    assert not torch.equal(policy_latents[0], policy_latents[2])


def test_crd_optimize_gives_the_graph_to_the_current_forward_only() -> None:
    adapter = _adapter()
    trainer = _crd_trainer(adapter)

    trainer.optimize(_samples([4, 4], [1.0, 2.0]))

    graph_owners = {call["scope"]: call["velocity_requires_grad"] for call in adapter.forward_calls}
    assert graph_owners == {
        CRDTrainer._OLD_PARAMS_NAME: False,
        "policy": True,
        "ref": False,
    }
    assert adapter.weight.grad is not None
    assert trainer.accelerator.losses[0].requires_grad is False


def test_crd_optimize_routes_the_reference_guidance_scale_from_training_args() -> None:
    adapter = _adapter()
    trainer = _crd_trainer(adapter, kl_cfg=1.0)

    trainer.optimize(_samples([4, 4], [1.0, 2.0]))

    scopes = {call["scope"]: call["guidance_scale"] for call in adapter.forward_calls}
    assert scopes["ref"] == 3.0

    cfg_adapter = _adapter()
    cfg_trainer = _crd_trainer(cfg_adapter, kl_cfg=4.5)
    cfg_trainer.optimize(_samples([4, 4], [1.0, 2.0]))
    cfg_scopes = {call["scope"]: call["guidance_scale"] for call in cfg_adapter.forward_calls}
    assert cfg_scopes["ref"] == 4.5


def test_crd_optimize_leaves_the_process_rng_untouched() -> None:
    """Pass 1 draws timesteps then one noise per step; pass 2 draws nothing.

    Those draws are keyed by `(seed, epoch, sample identity)` rather than taken
    from the process RNG, which is what lets two ranks holding the same sample
    draw the same tensor. This used to assert the exact legacy *order* of global
    RNG consumption; the stronger invariant now is that there is none, so a
    reordering elsewhere in the process cannot perturb CRD's draws.
    """
    adapter = _adapter()
    trainer = _crd_trainer(adapter, num_train_timesteps=2)
    samples = _samples([4, 4], [1.0, 2.0])

    torch.manual_seed(555)
    before = torch.get_rng_state().clone()
    trainer.optimize(samples)

    assert torch.equal(torch.get_rng_state(), before)

    # The same run under a different global seed must reproduce the loss, which
    # is the property the identity keying buys.
    other_adapter = _adapter()
    other_trainer = _crd_trainer(other_adapter, num_train_timesteps=2)
    torch.manual_seed(31337)
    other_trainer.optimize(_samples([4, 4], [1.0, 2.0]))

    assert torch.equal(trainer.accelerator.losses[0], other_trainer.accelerator.losses[0])
    old_scope_calls = [
        call for call in adapter.forward_calls if call["scope"] == CRDTrainer._OLD_PARAMS_NAME
    ]
    assert len(old_scope_calls) == 2
    # One timestep row per training step, drawn from the identity-keyed stream.
    expected_timesteps = TimeSampler.uniform(
        batch_size=2,
        num_timesteps=2,
        timestep_range=trainer.timestep_range,
        time_shift=trainer.time_shift,
        device=torch.device("cpu"),
        generator=identity_timestep_generator(trainer, sample_noise_ids(samples), 0),
    )
    assert torch.equal(old_scope_calls[0]["t"], expected_timesteps[0])
    assert torch.equal(old_scope_calls[1]["t"], expected_timesteps[1])


def test_crd_optimize_matches_the_legacy_single_step_loss_and_gradient() -> None:
    samples = _samples([4, 4], [1.0, 2.0])

    adapter = _adapter()
    trainer = _crd_trainer(adapter)
    torch.manual_seed(2026)
    trainer.optimize(samples)

    oracle_adapter = _adapter()
    oracle_trainer = _crd_trainer(oracle_adapter)
    torch.manual_seed(2026)
    legacy = _legacy_crd_step(oracle_trainer, oracle_adapter.weight, samples)
    legacy["loss"].backward()

    assert torch.equal(trainer.accelerator.losses[0], legacy["loss"].detach())
    assert torch.equal(adapter.weight.grad, oracle_adapter.weight.grad)


@pytest.mark.parametrize("adaptive_logp", [False, True])
def test_crd_optimize_matches_the_legacy_adaptive_and_plain_reward(
    adaptive_logp: bool,
) -> None:
    samples = _samples([4, 4], [1.0, 2.0])

    adapter = _adapter()
    trainer = _crd_trainer(adapter, adaptive_logp=adaptive_logp, reward_adaptive_kl=True)
    torch.manual_seed(31)
    trainer.optimize(samples)

    oracle_adapter = _adapter()
    oracle_trainer = _crd_trainer(
        oracle_adapter, adaptive_logp=adaptive_logp, reward_adaptive_kl=True
    )
    torch.manual_seed(31)
    legacy = _legacy_crd_step(oracle_trainer, oracle_adapter.weight, samples)
    legacy["loss"].backward()

    assert torch.equal(trainer.accelerator.losses[0], legacy["loss"].detach())
    assert torch.equal(adapter.weight.grad, oracle_adapter.weight.grad)


def test_crd_optimize_logs_the_expected_metric_keys() -> None:
    adapter = _adapter()
    trainer = _crd_trainer(adapter)

    trainer.optimize(_samples([4, 4], [1.0, 2.0]))

    step, payload = trainer.logger.payloads[0]
    assert step == 0
    assert set(payload) == {
        "train/policy_loss",
        "train/unweighted_policy_loss",
        "train/kl_div",
        "train/kl_loss",
        "train/r_theta_mean",
        "train/loss",
        "train/old_kl_div",
        "train/old_deviate",
        "train/grad_norm",
    }


def test_crd_start_updates_both_snapshots_after_optimize() -> None:
    adapter = _adapter()
    trainer = _crd_trainer(adapter)
    iterations = iter([True, False])
    trainer.should_continue_training = lambda: next(iterations)
    trainer.eval_args = SimpleNamespace(eval_freq=0)
    trainer.sample = lambda: []
    trainer.prepare_feedback = lambda samples: None
    trainer.optimize = lambda samples: None
    order: List[str] = []
    trainer._update_old_model = lambda: order.append("old")
    trainer._update_sampling_model = lambda: order.append("sampling")

    trainer.start()

    assert order == ["old", "sampling"]
    assert f"ema_step:{2}" in adapter.events
    assert adapter.scheduler_group.primary.seeds == [13]


def _with_snapshot_storage(adapter: TrainableAdapterFake) -> List[str]:
    """Give the adapter the named-snapshot internals ``_blend_named_params`` walks."""
    copied: List[str] = []
    adapter._named_parameters = {
        name: SimpleNamespace(
            target_components=("transformer",),
            ema_wrapper=SimpleNamespace(ema_parameters=storage),
        )
        for name, storage in adapter.named_snapshots.items()
    }
    adapter._get_component_parameters = lambda components: [adapter.weight]
    adapter.update_named_parameters = copied.append
    return copied


def test_crd_update_old_model_blends_the_snapshot_with_the_scheduled_decay() -> None:
    adapter = _adapter()
    _with_snapshot_storage(adapter)
    trainer = _crd_trainer(adapter)
    trainer.old_model_decay = "0-0.25-0.0-0.25"
    trainer.step = 6
    before = adapter.named_snapshots[CRDTrainer._OLD_PARAMS_NAME][0].clone()

    trainer._update_old_model()

    expected = before * 0.25 + adapter.weight.detach() * 0.75
    assert torch.equal(adapter.named_snapshots[CRDTrainer._OLD_PARAMS_NAME][0], expected)
    assert trainer.logger.payloads[-1] == (6, {"train/old_model_decay": 0.25})


def test_crd_update_sampling_model_full_copies_when_the_decay_is_zero() -> None:
    adapter = _adapter()
    copied = _with_snapshot_storage(adapter)
    trainer = _crd_trainer(adapter)
    trainer.sampling_model_decay = "0-0.0-0.0-0.0"

    trainer._update_sampling_model()

    assert copied == [CRDTrainer._SAMPLING_PARAMS_NAME]


def test_crd_pass_two_rejects_a_noise_state_that_does_not_match_the_clean_state() -> None:
    """A pass-1/pass-2 geometry drift must name the offending timestep."""
    adapter = _adapter()
    trainer = _crd_trainer(adapter)
    batch = BaseSample.stack(_samples([4, 4], [1.0, 2.0]))
    clean_state = adapter.get_terminal_state(batch)
    times = adapter.build_training_component_times(torch.tensor([700.0, 700.0]), batch=batch)
    step = _CRDStep(
        times=times,
        noise=LatentState({"latent": torch.zeros(2, 9)}),
        old_velocity=LatentState({"latent": torch.zeros(2, 3)}),
    )

    with pytest.raises(
        ValueError,
        match=r"CRDTrainer.*timestep_index=1.*component 'latent'.*\(2, 3\).*\(2, 9\)",
    ):
        trainer._rebuild_noised_state(clean_state, step, timestep_index=1)


def _crd_rebuild_inputs(
    trainer: CRDTrainer, adapter: TrainableAdapterFake
) -> Tuple[LatentState, ComponentTimes]:
    """Terminal clean state plus component times for a direct pass-2 rebuild."""
    batch = BaseSample.stack(_samples([4, 4], [1.0, 2.0]))
    clean_state = adapter.get_terminal_state(batch)
    times = adapter.build_training_component_times(torch.tensor([700.0, 700.0]), batch=batch)
    return clean_state, times


def test_crd_pass_two_rejects_a_non_latent_state_noise_before_component_lookup() -> None:
    adapter = _adapter()
    trainer = _crd_trainer(adapter)
    clean_state, times = _crd_rebuild_inputs(trainer, adapter)
    step = _CRDStep(
        times=times,
        noise={"latent": torch.zeros(2, 3)},
        old_velocity=LatentState({"latent": torch.zeros(2, 3)}),
    )

    with pytest.raises(
        TypeError,
        match=r"expected LatentState for .*timestep_index=1.* on CRDTrainer, received dict",
    ):
        trainer._rebuild_noised_state(clean_state, step, timestep_index=1)


@pytest.mark.parametrize(
    "components",
    [
        {"other": torch.zeros(2, 3)},
        {"latent": torch.zeros(2, 3), "extra": torch.zeros(2, 3)},
    ],
)
def test_crd_pass_two_rejects_a_mismatched_noise_component_order(
    components: Dict[str, torch.Tensor],
) -> None:
    """Missing or extra noise components must report the order, not raise KeyError."""
    adapter = _adapter()
    trainer = _crd_trainer(adapter)
    clean_state, times = _crd_rebuild_inputs(trainer, adapter)
    step = _CRDStep(
        times=times,
        noise=LatentState(components),
        old_velocity=LatentState({"latent": torch.zeros(2, 3)}),
    )

    with pytest.raises(
        ValueError,
        match=(
            r"expected .*timestep_index=1.* on CRDTrainer in component order "
            r"\('latent',\), received "
        ),
    ):
        trainer._rebuild_noised_state(clean_state, step, timestep_index=1)


def test_crd_pass_two_rejects_a_mismatched_clean_state_component_order() -> None:
    adapter = _adapter()
    trainer = _crd_trainer(adapter)
    _, times = _crd_rebuild_inputs(trainer, adapter)
    step = _CRDStep(
        times=times,
        noise=LatentState({"latent": torch.zeros(2, 3)}),
        old_velocity=LatentState({"latent": torch.zeros(2, 3)}),
    )

    with pytest.raises(
        ValueError,
        match=(
            r"expected .*clean state.*timestep_index=1.* on CRDTrainer in component order "
            r"\('latent',\), received \('other',\)"
        ),
    ):
        trainer._rebuild_noised_state(
            LatentState({"other": torch.zeros(2, 3)}), step, timestep_index=1
        )
