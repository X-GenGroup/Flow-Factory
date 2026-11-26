"""Regression coverage for the iTDM objective, update schedule and recipe geometry."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from flow_factory.hparams import Arguments
from flow_factory.trainers.distillation import itdm
from flow_factory.trainers.registry import get_trainer_class


@pytest.mark.parametrize("reward", ["pickscore", "ocr", "geneval"])
def test_itdm_config(reward, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "8")
    root = Path(__file__).resolve().parents[2]
    config = Arguments.load_from_yaml(root / f"examples/itdm/lora/sd3_5/{reward}.yaml")
    args = config.training_args
    assert get_trainer_class(args.trainer_type) is itdm.ITDMTrainer
    assert args.global_std == (reward != "geneval")
    assert args.dsm_with_default is False
    assert args.stack_base_lora_variants == ()
    assert args.advantage_aggregation == "gdpo"
    assert (args.unique_sample_num_per_epoch, args.group_size) == (24, 24)
    assert args.gradient_step_per_epoch == 1
    assert args.num_batches_per_epoch == 9
    assert config.scheduler_args.solver == "dpmpp_2m"
    assert (args.num_train_timesteps, args.max_timestep_index) == (4, 8)
    assert (args.num_inference_steps, config.eval_args.num_inference_steps) == (10, 40)
    assert args.gradient_accumulation_steps == 36
    assert args.execution_contract == itdm.ITDMTrainer.execution_contract


def test_signed_velocity_revision_and_detach():
    policy = torch.tensor([0.1, 0.2, 0.3, 0.4]).reshape(-1, 1, 1, 1).requires_grad_()
    residual = torch.ones_like(policy, requires_grad=True)
    loss = itdm.revised_velocity_loss(
        policy, residual, torch.zeros_like(policy), torch.tensor([-7.0, 0.0, 2.5, 7.0]), 5.0
    )
    weights = torch.tensor([-1.0, 0.0, 0.5, 1.0])
    torch.testing.assert_close(loss, weights.square().mean())
    loss.backward()
    torch.testing.assert_close(policy.grad.flatten(), 2 * weights / 4)
    assert residual.grad is None


@pytest.mark.parametrize(
    "epoch,expected", [(0, ["fake"] * 2), (1, ["fake"] * 2 + ["generator"] * 2)]
)
@pytest.mark.parametrize("shuffle", [False, True])
def test_ttur_reuses_time_windows(monkeypatch, epoch, expected, shuffle):
    args = SimpleNamespace(
        per_device_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_timesteps=2,
        max_timestep_index=3,
        ttur=2,
        num_inner_epochs=1,
        seed=24,
        shuffle_samples=shuffle,
    )
    phases = [SimpleNamespace(role_name=r) for r in ("fake", "generator")]
    trainer = SimpleNamespace(
        training_args=args,
        epoch=epoch,
        adapter=SimpleNamespace(
            train=lambda: None, requires_preserved_replay_batch_composition=False
        ),
        _role_update_plan=lambda: SimpleNamespace(phases=phases),
        optimization_roles={p.role_name: SimpleNamespace(step=0) for p in phases},
        _loss=lambda *args: None,
    )
    trainer._order_samples_for_optimize = itdm.BaseTrainer._order_samples_for_optimize.__get__(
        trainer
    )
    orders = []
    calls = []

    def record_phase(_, role, work, loss):
        orders.append(list(loss.args[0]))
        calls.append((role, work))

    monkeypatch.setattr(itdm, "run_role_phase", record_phase)
    monkeypatch.setattr(itdm, "log_distillation_metrics", lambda _: None)
    itdm.ITDMTrainer.optimize(trainer, list(range(8)))
    assert sorted(orders[0]) == list(range(8))
    assert (orders[0] != list(range(8))) == shuffle
    assert all(order == orders[0] for order in orders)
    assert [role for role, _ in calls] == expected
    if epoch == 1:
        assert [w for _, w in calls[:2]] == [w for _, w in calls[2:]]
