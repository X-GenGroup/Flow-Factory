from pathlib import Path

import pytest
import torch

from diffusers import DPMSolverMultistepScheduler, FlowMatchEulerDiscreteScheduler
from flow_factory.hparams import Arguments
from flow_factory.hparams.scheduler_args import SchedulerArguments
from flow_factory.scheduler.flow_match_euler_discrete import set_scheduler_timesteps
from flow_factory.scheduler.loader import load_scheduler


@pytest.mark.parametrize("steps", [10, 40])
def test_dpmpp_rollout_and_velocity_query(steps):
    args = SchedulerArguments(dynamics_type="ODE", solver="dpmpp_2m")
    scheduler = load_scheduler(FlowMatchEulerDiscreteScheduler(), args)
    ref = DPMSolverMultistepScheduler(
        use_flow_sigmas=True,
        flow_shift=3.0,
        prediction_type="flow_prediction",
        solver_order=2,
        solver_type="midpoint",
        algorithm_type="dpmsolver++",
    )
    for _ in range(2):
        ref.set_timesteps(steps)
        actual_times = set_scheduler_timesteps(scheduler, steps, seq_len=1024, device="cpu")
        torch.testing.assert_close(actual_times, ref.timesteps)
        torch.testing.assert_close(scheduler.sigmas, ref.sigmas)
        x = expected = torch.randn(2, 4, 8, 8)
        for t in actual_times:
            v = torch.randn_like(x, requires_grad=True)
            prediction = scheduler.step(
                v, actual_times[-1:].expand(2), x, return_kwargs=["velocity"]
            )
            prediction.velocity.sum().backward()
            torch.testing.assert_close(v.grad, torch.ones_like(v))
            with torch.no_grad():
                expected = ref.step(v, t, expected).prev_sample
                x = scheduler.step(v, t, x).next_latents
            torch.testing.assert_close(x, expected, rtol=0, atol=0)
    assert args.to_dict()["solver"] == "dpmpp_2m"
    assert "solver" not in SchedulerArguments().to_dict()
    with pytest.raises(ValueError, match="requires"):
        SchedulerArguments(solver="dpmpp_2m", dynamics_type="CPS")


@pytest.mark.parametrize("solver", ["pipeline", "dpmpp_2m"])
def test_pickscore_recipe_solver_selection(solver, tmp_path, monkeypatch):
    import yaml

    monkeypatch.setenv("WORLD_SIZE", "8")
    path = Path(__file__).resolve().parents[2] / "examples/itdm/lora/sd3_5/pickscore.yaml"
    recipe = yaml.safe_load(path.read_text())
    recipe["log"]["save_dir"] = str(tmp_path)
    recipe["scheduler"]["solver"] = solver
    args = Arguments.from_dict(recipe)
    assert args.scheduler_args.solver == solver
    assert args.scheduler_args.to_dict().get("solver", "pipeline") == solver
    if solver == "dpmpp_2m":
        recipe = yaml.safe_load(path.read_text())
        recipe["log"]["save_dir"] = str(tmp_path)
        recipe["model"]["model_type"] = "flux1"
        recipe["scheduler"]["solver"] = solver
        with pytest.raises(ValueError, match="SD3.5 iTDM"):
            Arguments.from_dict(recipe)
