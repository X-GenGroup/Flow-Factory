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

"""Native CP generation, feedback and real role updates."""

import copy
import hashlib
import json
import os
import subprocess
import sys
import traceback
from collections import Counter
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from conftest import run_gloo_ranks
from dmd2_objective_reference import objective_negative_controls, objective_reference
from online_observations import cpu as local_cpu_tree
from online_observations import observe_dmd2_objectives
from online_update_reference import compare_common_state, compare_feedback_updates
from test_context_parallel_state import _assert_ema_trajectory, _cpu_tree, _state
from test_context_parallel_training import _compare, _compare_bf16_updates
from test_model_preparation import full_value, initialize_cuda, runtime_observation
from wan_online_reward import MemberReward


def _config(root, result_dir, options):
    from flow_factory.hparams import Arguments

    algorithm = options["algorithm"]
    roles = ("generator", "fake") if algorithm == "dmd2" else ("default",)
    train = {
        "trainer_type": algorithm,
        "max_epochs": options.get("epochs", 2),
        "per_device_batch_size": 1,
        "resolution": [32, 32],
        "num_frames": 5,
        "frame_rate": 4.0,
        "num_inference_steps": 3,
        "guidance_scale": options.get("guidance_scale", 1.0),
        "group_size": 1 if algorithm == "dmd2" else 2,
        "unique_sample_num_per_epoch": (4 if algorithm == "dmd2" else options.get("dp_size", 2)),
        "gradient_step_per_epoch": 1,
        "gradient_accumulation_steps": 2 if algorithm == "dmd2" else "auto",
        "ema_decay": 0.9,
        "ema_update_interval": 1,
        "ema_decay_schedule": "constant",
        "enable_gradient_checkpointing": True,
        "seed": 17,
        "shuffle_samples": False,
        "latent_storage_dtype": "float32",
    }
    if algorithm == "dmd2":
        train.update(real_guidance_scale=1.0, replay_rtol=0.0, replay_atol=0.0)
    else:
        train.update(kl_beta=0.01, kl_type="v-based", global_std=False, advantage_aggregation="sum")
    return Arguments.from_dict(
        {
            "mixed_precision": "bf16",
            "parallelism": {
                "dp_shard_size": options.get("dp_size", 2),
                "cp_size": options["cp_size"],
                "tp_size": options["tp_size"],
                "sequence_parallel": options["tp_size"] > 1,
            },
            "model": {
                "model_type": "wan2_t2v",
                "model_name_or_path": str(root / "model"),
                "target_components": ["transformer"],
                "finetune_type": options["mode"],
                "target_modules": "all" if options["mode"] == "full" else "default",
                "component_load_dtypes": "float32",
                "lora_rank": 2,
                "lora_alpha": 2,
                "compute_precision_policy": "cross_attention_fp32_v1",
                **(
                    {"resume_type": "state", "resume_path": options["resume"]}
                    if options.get("resume")
                    else {}
                ),
            },
            "data": {
                "datasets": [
                    {"name": "tiny", "dataset_dir": str(root / "prompts"), "train": {"weight": 1}}
                ],
                "cache_dir": str(root / "online-cache"),
                "dataloader_num_workers": 0,
                "preprocessing_batch_size": 1,
                "sampler_type": options.get("sampler_type", "auto"),
                "sampler_subgroup_size": options.get("subgroup_size"),
                "enable_preprocess": True,
            },
            "train": train,
            "log": {
                "save_model_only": False,
                "save_freq": 1,
                "save_dir": str(Path(result_dir) / "saves"),
                "run_name": "wan-online",
                "logging_backend": "none",
                "verbose": False,
            },
            "eval": {"eval_freq": 0},
            "scheduler": {
                "dynamics_type": "ODE" if algorithm == "dmd2" else "Flow-SDE",
                "sde_steps": [0, 1],
                "num_sde_steps": 2,
                "seed": 17,
            },
            "rewards": (
                []
                if algorithm == "dmd2"
                else [
                    {
                        "name": "member",
                        "reward_model": "wan_online_reward.MemberReward",
                        "device": "cpu",
                        "batch_size": 2,
                        "async_reward": True,
                    }
                ]
            ),
            "acceleration": {
                "shared": [{"name": "attention_backend", "params": {"backend": "native"}}]
            },
            "optimizers": [
                {
                    "name": role,
                    "learning_rate": 1e-3,
                    "eps": 1e-4,
                    "max_grad_norm": 1.0,
                    "update_frequency": 2 if role == "fake" else 1,
                }
                for role in roles
            ],
        }
    )


def _worker(rank, store_path, result_dir, options):
    os.environ.update(
        FSDP_VERSION="2",
        FSDP_AUTO_WRAP_POLICY="TRANSFORMER_BASED_WRAP",
        FSDP_TRANSFORMER_CLS_TO_WRAP="WanTransformerBlock",
        FSDP_CPU_RAM_EFFICIENT_LOADING="false",
        FSDP_RESHARD_AFTER_FORWARD="true",
        FSDP_ACTIVATION_CHECKPOINTING="true",
    )
    model_degree = options["cp_size"] * options["tp_size"]
    initialize_cuda(rank, store_path, world_size=options.get("dp_size", 2) * model_degree)
    result = {"rank": rank, **options}
    try:
        from flow_factory.trainers.loader import load_trainer

        if options["algorithm"] == "dmd2":
            from flow_factory.trainers.distillation.dmd2 import DMD2Trainer

            initialize_snapshots = DMD2Trainer._initialize_snapshots
            after_cycle = DMD2Trainer._after_acquisition_cycle

            def initialize_named_snapshot(self):
                initialize_snapshots(self)
                self.adapter.component_variant_registry.add_snapshot("fake", "slow_fake")

            def update_named_snapshot(self):
                after_cycle(self)
                self.adapter.component_variant_registry.update_snapshot("slow_fake", decay=0.8)

            # Exercise the sanctioned snapshot lifecycle on a real trainer. The
            # same declaration exists before exact restore on every worker leg.
            DMD2Trainer._initialize_snapshots = initialize_named_snapshot
            DMD2Trainer._after_acquisition_cycle = update_named_snapshot
        trainer = load_trainer(_config(Path(options["fixture"]), result_dir, options))
        result["runtime"] = runtime_observation(trainer.accelerator)
        records = {
            "leg_kind": (
                "common_state_shadow" if options.get("common_state_reference") else "normal"
            ),
            "initial_parameters": _cpu_tree(dict(trainer.model_bundle.named_parameters())),
            "updates": [],
            "samples": [],
            "feedback": [],
            "backward_losses": [],
            "replay_log_probs": [],
            "replay_velocities": [],
            "replay_means": [],
            "replay_inputs": [],
            "metrics": [],
            "reference_calls": 0,
        }
        loaded = _state(trainer)
        parameters = dict(trainer.model_bundle.named_parameters())
        common_state = None
        if options.get("common_state_reference"):
            reference_rank = (
                rank // model_degree * options["tp_size"] * options["reference_cp_size"]
            )
            reference_path = (
                Path(options["common_state_reference"]) / f"observations-{reference_rank}.pt"
            )
            common_state = torch.load(reference_path, weights_only=True)["records"]
            records["state_reference"] = {
                "cp_size": options["reference_cp_size"],
                "rank": reference_rank,
                "sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
            }
            _compare(records["initial_parameters"], common_state["initial_parameters"], exact=True)
        names = {id(p): name for name, p in parameters.items()}
        snapshot_names = (
            {
                f"{record.component_name}.{record.parameter_name}": names[id(record.parameter)]
                for record in trainer.adapter.component_variant_registry.parameter_records("fake")
            }
            if options["algorithm"] == "dmd2"
            else {}
        )
        role_names = {
            role: {names[id(p)] for p in spec.parameters}
            for role, spec in trainer.optimization_roles.items()
        }
        sample_batch = trainer.sample_batch
        feedback = trainer.prepare_feedback
        backward = trainer.accelerator.backward
        clip = trainer.accelerator.clip_grad_norm_
        optimizer_step = trainer.optimizer.step
        pending = {}
        saving = False

        @wraps(sample_batch)
        def observed_samples(*args, **kwargs):
            if options["algorithm"] == "grpo":
                # Observe the native rollout without a second forward or a
                # different collective path. These callbacks are test-only.
                kwargs["extra_call_back_kwargs"] = ["velocity", "next_latents_mean"]
            samples = sample_batch(*args, **kwargs)
            records["samples"].append(
                [
                    {
                        "source": s.source_id,
                        "group": s.sampling_group_id,
                        "member": s.sampling_group_member_id,
                        "sample": s.sampling_sample_id,
                    }
                    for s in samples
                ]
            )
            return samples

        def observed_feedback(samples):
            returned = feedback(samples)
            if options["algorithm"] == "dmd2":
                assert all("advantage" not in sample.extra_kwargs for sample in samples)
                return returned
            records["feedback"].append(
                [
                    {"sample": s.sampling_sample_id, "advantage": _cpu_tree(s["advantage"])}
                    for s in samples
                ]
            )
            return returned

        def observed_backward(loss, **kwargs):
            records["backward_losses"].append(loss.detach().cpu() / options["cp_size"])
            return backward(loss, **kwargs)

        def observed_clip(*args, **kwargs):
            role = trainer.role_optimization._active_role_name or trainer._primary_role()
            gradients = {
                name: full_value(p.grad).cpu().clone()
                for name, p in parameters.items()
                if p.grad is not None
            }
            assert set(gradients) == role_names[role]
            assert any(torch.count_nonzero(g) for g in gradients.values())
            pending.update(
                role=role,
                gradients=gradients,
                before=_cpu_tree(parameters),
                loss=torch.stack(records["backward_losses"]),
            )
            records["backward_losses"].clear()
            return clip(*args, **kwargs)

        def observed_step(*args, **kwargs):
            if saving:
                return optimizer_step(*args, **kwargs)
            optimizer = trainer.optimizer.optimizer
            role = pending["role"]
            spec = trainer.optimization_roles[role]
            assert len(spec.optimizer_group_ids) == 1
            group = optimizer.param_groups[spec.optimizer_group_ids[0]]
            settings = {key: group[key] for key in ("lr", "betas", "eps", "weight_decay")}
            settings["max_grad_norm"] = spec.config.max_grad_norm
            clipped = {
                name: full_value(parameters[name].grad).cpu().clone()
                for name in pending["gradients"]
            }
            old_states = {
                name: _cpu_tree(optimizer.state.get(p, {}))
                for name, p in parameters.items()
                if name not in role_names[role]
            }
            returned = optimizer_step(*args, **kwargs)
            after = _cpu_tree(parameters)
            for name, p in parameters.items():
                if name not in role_names[role]:
                    _compare(after[name], pending["before"][name], exact=True)
                    _compare(_cpu_tree(optimizer.state.get(p, {})), old_states[name], exact=True)
            records["updates"].append(
                {
                    "role": role,
                    "before": pending["before"],
                    "parameters": after,
                    "gradients": pending["gradients"],
                    "clipped_gradients": clipped,
                    "optimizer": settings,
                    "loss": pending["loss"],
                    "deltas": {
                        name: after[name] - pending["before"][name]
                        for name in sorted(role_names[role])
                    },
                    "moments": {
                        name: _cpu_tree(optimizer.state[parameters[name]])
                        for name in sorted(role_names[role])
                    },
                }
            )
            if common_state is not None:
                from torch.distributed.tensor import DTensor, distribute_tensor

                index = len(records["updates"]) - 1
                reference_update = common_state["updates"][index]
                assert reference_update["role"] == role
                expected_before = (
                    common_state["initial_parameters"]
                    if index == 0
                    else common_state["updates"][index - 1]["parameters"]
                )
                _compare(pending["before"], expected_before, exact=True)
                # Diagnostic intervention: discard this leg's updated weights,
                # so the next forward starts at the reference trajectory's state.
                # It proves local gradients, not uninterrupted optimization.
                trainer.adapter.component_variant_registry.prepare_parameter_write()
                with torch.no_grad():
                    for name, parameter in parameters.items():
                        value = reference_update["parameters"][name].to(parameter.device)
                        if isinstance(parameter, DTensor):
                            value = distribute_tensor(
                                value,
                                parameter.device_mesh,
                                parameter.placements,
                                src_data_rank=None,
                            )
                        parameter.copy_(value)
                _compare(_cpu_tree(parameters), reference_update["parameters"], exact=True)
            return returned

        reference = trainer.adapter.use_ref_parameters

        @contextmanager
        def observed_reference(*args, **kwargs):
            records["reference_calls"] += 1
            with reference(*args, **kwargs):
                yield

        trainer.adapter.use_ref_parameters = observed_reference
        if options["algorithm"] == "grpo":
            replay_forward = trainer._replay_forward

            def observed_replay(batch, replay, return_fields):
                records["replay_inputs"].append(
                    local_cpu_tree(
                        {
                            "replay": replay,
                            "conditioning": {
                                key: batch.get(key)
                                for key in (
                                    "prompt_embeds",
                                    "negative_prompt_embeds",
                                    "guidance_scale",
                                )
                            },
                        }
                    )
                )
                fields = tuple(dict.fromkeys((*return_fields, "velocity", "next_latents_mean")))
                output = replay_forward(batch, replay, fields)
                # One optimizer step per fresh acquisition: every replay here is
                # still on-policy, including the second cycle and resumed cycle.
                torch.testing.assert_close(output.log_prob, replay.log_prob, rtol=0, atol=0)
                positions = (
                    (batch["timesteps"] == replay.times.timestep["latent"].reshape(-1, 1))
                    .all(dim=0)
                    .nonzero()
                    .flatten()
                )
                assert positions.numel() == 1
                step_index = positions.item()
                for field, replayed, record in (
                    ("velocity", output.velocity, "replay_velocities"),
                    ("next_latents_mean", output.next_state_mean, "replay_means"),
                ):
                    stored = trainer.adapter.get_replay_callback(batch, step_index, field)
                    assert replayed.component_names == stored.component_names
                    for component in stored.component_names:
                        torch.testing.assert_close(
                            replayed.components[component],
                            stored.components[component],
                            rtol=0,
                            atol=0,
                        )
                    records[record].append(_cpu_tree(dict(replayed.components)))
                records["replay_log_probs"].append(_cpu_tree(output.log_prob))
                return output

            trainer._replay_forward = observed_replay
        log_data = trainer.log_data

        def observed_metrics(data, **kwargs):
            selected = {
                key: _cpu_tree(value)
                for key, value in data.items()
                if key.startswith(("train/reward_", "train/adv_", "train/ratio_"))
            }
            if selected:
                records["metrics"].append(selected)
            return log_data(data, **kwargs)

        trainer.log_data = observed_metrics
        save = trainer._save_exact_training_state
        saved = {}

        def observed_save(path):
            nonlocal saving
            saved[trainer.cycle_index] = _state(trainer)
            saving = True
            try:
                return save(path)
            finally:
                saving = False

        trainer._save_exact_training_state = observed_save
        trainer.sample_batch = observed_samples
        trainer.prepare_feedback = observed_feedback
        trainer.accelerator.backward = observed_backward
        trainer.accelerator.clip_grad_norm_ = observed_clip
        trainer.optimizer.step = observed_step
        with observe_dmd2_objectives(trainer) as objectives:
            trainer.start()
        records["objectives"] = objectives
        for event in objectives:
            check = objective_reference(event, options["cp_size"])
            assert check["latent_pass"] and check["root_pass"]
            assert all(objective_negative_controls(event, options["cp_size"]).values())
        final = _state(trainer)
        assert trainer.step == options.get("epochs", 2)
        assert trainer.progress.rollout_iteration == options.get("epochs", 2)
        assert trainer.progress.data_epoch == 0
        cycles = options.get("epochs", 2) - (1 if options.get("resume") else 0)
        assert records["reference_calls"] > 0
        if options["mode"] == "full":
            _compare(
                final["runtime"]["children"]["adapter_reference"],
                loaded["runtime"]["children"]["adapter_reference"],
                exact=True,
            )
        if options["algorithm"] == "dmd2":
            assert not trainer.reward_models and not MemberReward.scored and not records["feedback"]
            assert Counter(u["role"] for u in records["updates"]) == {
                "generator": cycles,
                "fake": cycles * 2,
            }
        else:
            expected_scored = [
                sample["sample"]
                for cycle in records["feedback"]
                for index, sample in enumerate(cycle)
                if index % model_degree == rank % model_degree
            ]
            assert Counter(MemberReward.scored) == Counter(expected_scored)
            assert len(records["feedback"]) == cycles
            assert len(records["replay_log_probs"]) == 4 * cycles
        assert not records["backward_losses"]
        torch.save(
            {
                "records": records,
                "loaded": loaded,
                "final": final,
                "saved": saved,
                "snapshot_names": snapshot_names,
                "rewards": MemberReward.scored,
            },
            Path(result_dir, f"observations-{rank}.pt"),
        )
        result.update(
            status="pass",
            optimizer_steps=trainer.step,
            roles=list(role_names),
            reward_calls=len(MemberReward.scored),
            elementwise_comparison=True,
            nonzero_gradients=True,
            execution_capability=f"wan-{options['algorithm']}-cp-v1",
        )
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result, indent=2))
        dist.destroy_process_group()


def _qualify_online(
    tmp_path,
    trainer_type,
    mode,
    tp_size,
    guidance_scale=1.0,
    *,
    dp_size=2,
    sampler_type="auto",
    subgroup_size=None,
):
    algorithm = trainer_type
    world = 2 * dp_size * tp_size
    available = int(os.environ.get("FF_TEST_CUDA_RANKS", torch.cuda.device_count()))
    if available < world:
        pytest.skip(f"native CP online qualification requires {world} CUDA ranks")
    root = tmp_path.resolve() / "fixture"
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("wan_sft_fixture.py")),
            str(root),
            str(2 * tp_size),
        ],
        check=True,
    )
    prompts = root / "prompts"
    prompts.mkdir()
    (prompts / "train.jsonl").write_text(
        "".join(
            json.dumps({"prompt": "a red video" if i % 2 else "a blue video"}) + "\n"
            for i in range(8)
        )
    )
    groups = {}
    for degree in (1, 2):
        directory = tmp_path / f"cp{degree}"
        rows = run_gloo_ranks(
            _worker,
            directory,
            dp_size * degree * tp_size,
            {
                "fixture": str(root),
                "dp_size": dp_size,
                "sampler_type": sampler_type,
                "subgroup_size": subgroup_size,
                "cp_size": degree,
                "tp_size": tp_size,
                "mode": mode,
                "algorithm": algorithm,
                "guidance_scale": guidance_scale,
            },
        )
        for row in rows:
            assert "error" not in row, row.get("error")
            assert row["status"] == "pass"
        groups[degree] = [
            torch.load(directory / f"observations-{rank}.pt", weights_only=True)
            for rank in range(dp_size * degree * tp_size)
        ]
    common_state = []
    shadows = {}
    feedback_sensitive = algorithm == "dmd2" or guidance_scale > 1.0
    for degree, reference_degree in ((2, 1), (1, 2)) if feedback_sensitive else ():
        directory = tmp_path / f"common-state-cp{degree}-from-cp{reference_degree}"
        shadow_world = dp_size * degree * tp_size
        rows = run_gloo_ranks(
            _worker,
            directory,
            shadow_world,
            {
                "fixture": str(root),
                "dp_size": dp_size,
                "sampler_type": sampler_type,
                "subgroup_size": subgroup_size,
                "cp_size": degree,
                "tp_size": tp_size,
                "mode": mode,
                "algorithm": algorithm,
                "guidance_scale": guidance_scale,
                "common_state_reference": str(tmp_path / f"cp{reference_degree}"),
                "reference_cp_size": reference_degree,
            },
        )
        for row in rows:
            assert "error" not in row, row.get("error")
            assert row["status"] == "pass"
        shadows[degree] = []
        for rank in range(shadow_world):
            shadow = torch.load(directory / f"observations-{rank}.pt", weights_only=True)
            shadows[degree].append(shadow["records"])
            expected = groups[reference_degree][
                rank // (degree * tp_size) * reference_degree * tp_size
            ]
            common_state.append(
                {
                    "cp_size": degree,
                    "reference_cp_size": reference_degree,
                    "rank": rank,
                    "updates": compare_common_state(shadow["records"], expected["records"]),
                }
            )
    residuals = []
    for rank, actual in enumerate(groups[2]):
        expected = groups[1][rank // (2 * tp_size) * tp_size]
        left, right = actual["records"], expected["records"]
        _compare(left["samples"], right["samples"], exact=True)
        _compare(left["feedback"], right["feedback"], exact=True)
        _compare(left["metrics"], right["metrics"], exact=True)
        _compare(left["initial_parameters"], right["initial_parameters"], exact=True)
        for role in {u["role"] for u in left["updates"]}:
            a = {
                "initial_parameters": left["initial_parameters"],
                "updates": [u for u in left["updates"] if u["role"] == role],
            }
            b = {
                "initial_parameters": right["initial_parameters"],
                "updates": [u for u in right["updates"] if u["role"] == role],
            }
            metrics = (
                compare_feedback_updates(
                    a,
                    b,
                    algorithm="dmd2" if algorithm == "dmd2" else "grpo_cfg",
                    same_state_reference=(
                        {
                            "updates": [
                                u
                                for u in shadows[1][rank // (2 * tp_size) * tp_size]["updates"]
                                if u["role"] == role
                            ]
                        }
                        if algorithm == "dmd2"
                        else None
                    ),
                )
                if feedback_sensitive
                else _compare_bf16_updates(a, b)
            )
            residuals.append({"rank": rank, "role": role, "updates": metrics})
    for degree in (1, 2):
        rewards = [sample for rank in groups[degree] for sample in rank["rewards"]]
        assert len(rewards) == (4 * dp_size if algorithm == "grpo" else 0)
    # Restore from the actual drained checkpoint produced halfway through the
    # continuous CP run. No separate model/optimizer format is used by this probe.
    resume_dir = tmp_path / "resumed"
    rows = run_gloo_ranks(
        _worker,
        resume_dir,
        world,
        {
            "fixture": str(root),
            "dp_size": dp_size,
            "sampler_type": sampler_type,
            "subgroup_size": subgroup_size,
            "cp_size": 2,
            "tp_size": tp_size,
            "mode": mode,
            "algorithm": algorithm,
            "guidance_scale": guidance_scale,
            "resume": str(tmp_path / "cp2/saves/wan-online/checkpoints/checkpoint-1"),
        },
    )
    for row in rows:
        assert "error" not in row, row.get("error")
    continuation = []
    for rank, continuous in enumerate(groups[2]):
        resumed = torch.load(resume_dir / f"observations-{rank}.pt", weights_only=True)
        _compare(resumed["loaded"], continuous["saved"][1], exact=True)
        expected = continuous["records"]
        joined = copy.deepcopy(expected)
        for key in (
            "samples",
            "feedback",
            "replay_log_probs",
            "replay_velocities",
            "replay_means",
            "replay_inputs",
            "metrics",
            "updates",
        ):
            cutoff = len(expected[key]) // 2
            joined[key] = expected[key][:cutoff] + resumed["records"][key]
            if key != "updates":
                _compare(joined[key], expected[key], exact=True)
        for role in sorted({u["role"] for u in expected["updates"]}):
            a = {
                "initial_parameters": joined["initial_parameters"],
                "updates": [u for u in joined["updates"] if u["role"] == role],
            }
            b = {
                "initial_parameters": expected["initial_parameters"],
                "updates": [u for u in expected["updates"] if u["role"] == role],
            }
            continuation.append(
                {"rank": rank, "role": role, "updates": _compare_bf16_updates(a, b)}
            )
        # Online EMA runs once per complete acquisition, after every role phase.
        cycle_updates = joined["updates"][2::3] if algorithm == "dmd2" else joined["updates"]
        _assert_ema_trajectory(continuous["loaded"], resumed["final"], cycle_updates)
        _compare(resumed["final"]["runtime"], continuous["final"]["runtime"])
        if algorithm == "dmd2":
            _assert_named_snapshot(continuous, resumed, joined["updates"])
    print(
        json.dumps(
            {
                "algorithm": algorithm,
                "mode": mode,
                "tp_size": tp_size,
                "dp_size": dp_size,
                "sampler_type": sampler_type,
                "subgroup_size": subgroup_size,
                "subgroup_count": dp_size // subgroup_size if subgroup_size is not None else None,
                "guidance_scale": guidance_scale,
                "exact_loaded_state": True,
                "common_state_gradients": common_state,
                "residuals": residuals,
                "continuation": continuation,
            }
        )
    )


@pytest.mark.parametrize("trainer_type", ["grpo", "dmd2"])
@pytest.mark.parametrize("mode", ["full", "lora"])
@pytest.mark.parametrize("tp_size", [1, 2])
def test_cuda_wan_online_cp_equivalence(tmp_path, trainer_type, mode, tp_size):
    _qualify_online(tmp_path, trainer_type, mode, tp_size)


@pytest.mark.parametrize("tp_size,mode", [(1, "full"), (2, "lora")])
def test_cuda_wan_grpo_cfg_cp_equivalence(tmp_path, tp_size, mode):
    _qualify_online(tmp_path, "grpo", mode, tp_size, guidance_scale=3.0)


def _assert_named_snapshot(initial_run, resumed, updates):
    """Verify the changed fake-role snapshot against independent FP64 recurrence."""

    def snapshot(state):
        return state["runtime"]["children"]["multirole"]["variant_snapshots"]

    start = snapshot(initial_run["loaded"])
    end = snapshot(resumed["final"])
    assert start["update_counts"] == {"slow_fake": 0}
    assert end["update_counts"] == {"slow_fake": 2}
    before = start["snapshots"]["slow_fake"]["parameters"]
    after = end["snapshots"]["slow_fake"]["parameters"]
    gamma = 4 * torch.finfo(torch.float32).eps
    for key, name in initial_run["snapshot_names"].items():
        expected = before[key].double()
        error = torch.zeros_like(expected)
        for update in updates[2::3]:
            live = update["parameters"][name].double()
            error = 0.8 * error + gamma * (0.8 * expected.abs() + 0.2 * live.abs())
            expected = 0.8 * expected + 0.2 * live
        assert torch.all((after[key].double() - expected).abs() <= error + 1e-30), key
    assert any(not torch.equal(before[key], after[key]) for key in before)
