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

"""Native audio/video ART and complete-group streaming integration."""

import json
import os
import traceback
from datetime import timedelta
from pathlib import Path
from threading import Event, Lock

import pytest
import torch
import torch.distributed as dist
from conftest import run_gloo_ranks
from wan_online_reward import MemberReward


class PendingMemberReward(MemberReward):
    """Hold later group rewards until a real policy backward has begun."""

    began_backward = Event()
    lock = Lock()
    calls = 0

    def __call__(self, sampling_sample_id, sampling_group_member_id):
        with self.lock:
            position = type(self).calls
            type(self).calls += len(sampling_sample_id)
        if position >= 2 and not self.began_backward.wait(timeout=60):
            raise RuntimeError("ART did not begin optimization while later rewards were pending")
        return super().__call__(sampling_sample_id, sampling_group_member_id)


def _worker(rank, store_path, result_dir, options):
    os.environ.update(
        RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2", ACCELERATE_USE_FSDP="false"
    )
    torch.cuda.set_device(rank)
    result = {"rank": rank}
    trainer = None
    try:
        dist.init_process_group(
            "nccl",
            store=dist.FileStore(store_path, 2),
            rank=rank,
            world_size=2,
            timeout=timedelta(seconds=180),
        )
        from importlib import import_module

        fixture = import_module(
            "ltx2_coupled_fixture" if options["family"] == "ltx2" else "h3_coupled_fixture"
        )
        make_config = fixture.make_config

        from flow_factory.hparams import Arguments
        from flow_factory.trainers.loader import load_trainer

        root = Path(options["root"])
        original = make_config(
            root,
            Path(result_dir) / "saves",
            {
                "training_mode": "full",
                "algorithm": "grpo",
                "dp_size": 2,
                "cp_size": 1,
                "tp_size": 1,
            },
            "ltx20_t2av" if options["family"] == "ltx2" else "t2va",
        )
        c = original.to_dict()
        c["mixed_precision"] = "no"
        c["model"]["trainable_parameters_dtype"] = "float32"
        c["train"] = c.pop("training")
        c["rewards"] = [r.to_dict() for r in original.reward_args]
        if options["overlap"]:
            c["rewards"][0].update(
                reward_model="test_art_components.PendingMemberReward", batch_size=1, num_workers=8
            )
        c["optimizers"] = [o.to_dict() for o in original.optimizer_args]
        c.pop("reward", None)
        c.pop("optimizer", None)
        c.pop("eval_reward", None)
        from dataclasses import fields

        for f in fields(original.training_args):
            if not f.init:
                c["train"].pop(f.name, None)
        # The source recipe's reference KL settings are GRPO's; ART keeps its own defaults.
        c["train"].pop("kl_type", None)
        c["train"].pop("kl_beta", None)
        c["train"].update(
            trainer_type="art",
            ema_decay=0.0,
            gradient_accumulation_steps="auto",
            gradient_step_per_epoch=2 if options["family"] == "ltx2" else 3,
            reward_optimization_overlap=options["overlap"],
            reward_optimization_overlap_mode=options["mode"] if options["overlap"] else "ordered",
            max_epochs=2,
            unique_sample_num_per_epoch=8,
        )
        c["parallelism"].update(dp_shard_size=1, dp_replicate_size=2)
        c["scheduler"].update(dynamics_type="ODE", replay_steps=[0, 1])
        c["log"]["save_freq"] = 0
        trainer = load_trainer(Arguments.from_dict(c))
        from contextlib import nullcontext

        trainer.autocast = nullcontext  # Explicit FP32 numerical control; native default is BF16.
        updates = []
        traces = {
            "initial": {
                n: p.detach().cpu().clone()
                for n, p in trainer.model_bundle.named_parameters()
                if p.requires_grad
            }
        }
        original_sample = trainer.sample

        def sample():
            samples = original_sample()
            traces[f"sample-{trainer.cycle_index}"] = [
                {n: c.states.detach().cpu().clone() for n, c in s.trajectory.components.items()}
                for s in samples
            ]
            return samples

        trainer.sample = sample
        from test_model_preparation import runtime_observation

        result["runtime"] = runtime_observation(trainer.accelerator)
        logged = []
        original_log = trainer.log_data

        def log(data, **kwargs):
            logged.append(data)
            return original_log(data, **kwargs)

        trainer.log_data = log
        original_backward = trainer._backward_full_output_loss

        def backward(loss):
            PendingMemberReward.began_backward.set()
            return original_backward(loss)

        trainer._backward_full_output_loss = backward
        original_step = trainer._apply_optimizer_step

        def step(metrics):
            assert torch.isfinite(torch.stack(metrics["loss"])).all()
            gradients = [p.grad for p in trainer.model_bundle.parameters() if p.grad is not None]
            assert gradients and all(torch.isfinite(g).all() for g in gradients)
            assert any(torch.count_nonzero(g) for g in gradients)
            traces[f"grad-{len(updates)}"] = {
                n: p.grad.detach().cpu().clone()
                for n, p in trainer.model_bundle.named_parameters()
                if p.grad is not None
            }
            updates.append({k: float(torch.stack(v).mean()) for k, v in metrics.items()})
            return original_step(metrics)

        trainer._apply_optimizer_step = step
        original_forward = trainer.adapter.forward_state
        seen = set()

        def forward(**kwargs):
            out = original_forward(**kwargs)
            seen.update(out.velocity.components)
            return out

        trainer.adapter.forward_state = forward
        from online_optimizer_observations import observe_optimizer_steps

        observations = {
            "initial_parameters": traces["initial"],
            "updates": [],
            "backward_losses": [],
        }
        with observe_optimizer_steps(trainer, observations, cp_size=1, is_saving=lambda: False):
            trainer.start()
        torch.save(observations, Path(result_dir) / f"observations-{rank}.pt")
        assert len(updates) == 2 * trainer.training_args.gradient_step_per_epoch and seen == {
            "video",
            "audio",
        }
        assert all(
            "art_keep_fraction/audio" in u and "art_keep_fraction/video" in u for u in updates
        )
        if options["overlap"]:
            assert any(
                float(
                    row.get(
                        "timing/reward_overlap/optimization_started_while_rewards_pending_seconds",
                        0,
                    )
                )
                > 0
                for row in logged
            )
            assert any(float(row.get("train/reward_overlap/tile_count", 0)) > 2 for row in logged)
        result.update(updates=updates, components=sorted(seen), step=trainer.step)
        torch.save(traces, Path(result_dir) / f"trace-{rank}.pt")
        state = {
            n: p.detach().cpu()
            for n, p in trainer.model_bundle.named_parameters()
            if p.requires_grad
        }
        torch.save(state, Path(result_dir) / f"state-{rank}.pt")
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result))
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize("family", ["ltx2", "h3"])
def test_cuda_art_joint_av_ordered_stream(tmp_path, family):
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    from importlib import import_module

    fixture = import_module("ltx2_coupled_fixture" if family == "ltx2" else "h3_coupled_fixture")
    build_fixture = fixture.build_fixture

    root = tmp_path / "fixture"
    build_fixture(root, 2, "ltx20_t2av" if family == "ltx2" else "t2va")
    (root / "tiny").mkdir()
    (root / "tiny" / "train.jsonl").write_text(
        "".join(json.dumps({"prompt": f"a red bird {i}"}) + "\n" for i in range(16))
    )
    outputs = []
    for mode in ("drained", "ordered", "ready"):
        overlap = mode != "drained"
        out = tmp_path / mode
        records = run_gloo_ranks(
            _worker,
            out,
            2,
            {"root": str(root), "overlap": overlap, "family": family, "mode": mode},
            timeout=360,
        )
        assert all("error" not in r for r in records), records
        outputs.append(torch.load(out / "state-0.pt", weights_only=True))
    from test_context_parallel_training import _check_adam_trajectory

    traces = []
    for mode in ("drained", "ordered", "ready"):
        records = torch.load(tmp_path / mode / "observations-0.pt", weights_only=False)
        _check_adam_trajectory(records)
        traces.append(torch.load(tmp_path / mode / "trace-0.pt", weights_only=True))
    # Same initial parameters and first rollout; later reductions need numerical
    # rather than bitwise equality. Check gradients, not only near-base weights.
    for name, value in traces[0]["initial"].items():
        torch.testing.assert_close(value, traces[1]["initial"][name], rtol=0, atol=0)
    for left, right in zip(traces[0]["sample-0"], traces[1]["sample-0"]):
        for name, value in left.items():
            torch.testing.assert_close(value, right[name], rtol=0, atol=0)
    for step in range(4 if family == "ltx2" else 6):
        left = traces[0][f"grad-{step}"]
        right = traces[1][f"grad-{step}"]
        for name, value in left.items():
            torch.testing.assert_close(value, right[name], rtol=1e-5, atol=1e-7)
        a = torch.cat([v.flatten() for v in left.values()])
        b = torch.cat([right[n].flatten() for n in left])
        assert (a - b).norm() / a.norm().clamp_min(1e-12) < 1e-5
    for name, value in outputs[0].items():
        torch.testing.assert_close(value, outputs[1][name], rtol=1e-5, atol=1e-7)
