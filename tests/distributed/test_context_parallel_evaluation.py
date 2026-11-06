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

"""Native evaluation, independent EMA forwards and subsequent on-policy updates."""

import copy
import json
import os
import subprocess
import sys
import traceback
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from conftest import run_gloo_ranks
from test_context_parallel_online import _config as online_config
from test_context_parallel_state import _cpu_tree
from test_context_parallel_training import _bf16_tensors, _compare
from test_context_parallel_training import _config as sft_config
from test_model_preparation import full_value, initialize_cuda, runtime_observation
from wan_eval_reward import PromptReward


def _config(root, result_dir, options):
    from flow_factory.hparams import DataArguments, MultiRewardArguments
    from flow_factory.hparams.training_args._base import EvaluationArguments

    factory = sft_config if options["algorithm"] == "sft" else online_config
    config = factory(
        root,
        result_dir,
        {**options, "checkpointing": True, "ema_decay": 0.9, "ema_update_interval": 1, "epochs": 2},
    )
    data = config.data_args.to_dict()
    data["datasets"].extend(
        [
            {
                "name": "odd",
                "dataset_dir": str(root / "odd"),
                "eval": {"guidance_scale": 3.0, "num_inference_steps": 2},
            },
            {
                "name": "single",
                "dataset_dir": str(root / "single"),
                "eval": {"resolution": [32, 48]},
            },
        ]
    )
    return replace(
        config,
        data_args=DataArguments.from_dict(data),
        eval_args=EvaluationArguments(
            eval_freq=1,
            resolution=[32, 32],
            num_frames=5,
            frame_rate=4.0,
            num_inference_steps=3,
            guidance_scale=1.0,
            seed=0,
            per_device_batch_size=1,
        ),
        eval_reward_args=MultiRewardArguments.from_dict(
            [
                {
                    "name": "score",
                    "reward_model": "wan_eval_reward.PromptReward",
                    "device": "cpu",
                    "batch_size": 2,
                    "async_reward": True,
                }
            ]
        ),
        log_args=replace(config.log_args, save_model_only=True, save_freq=0),
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
    world = 2 * options["cp_size"] * options["tp_size"]
    initialize_cuda(rank, store_path, world_size=world)
    result = {"rank": rank, **options}
    try:
        from flow_factory.models.tensor_parallel import peel_peft_wrapper
        from flow_factory.trainers.common.model_preparation import ModelPreparation
        from flow_factory.trainers.loader import load_trainer

        native = []
        parallelize = ModelPreparation.parallelize

        def capture_native(owner, *args, **kwargs):
            registry = owner.adapter.component_variant_registry
            route = registry.resolve_route(registry.base_variant, "transformer")
            native.append(copy.deepcopy(registry.bundle_members()[route]))
            return parallelize(owner, *args, **kwargs)

        ModelPreparation.parallelize = capture_native
        try:
            trainer = load_trainer(_config(Path(options["fixture"]), result_dir, options))
        finally:
            ModelPreparation.parallelize = parallelize
        result["runtime"] = runtime_observation(trainer.accelerator)
        registry = trainer.adapter.component_variant_registry
        route = registry.resolve_route(registry.base_variant, "transformer")
        module = registry.bundle_members()[route]
        reference = native[0].to(device=trainer.accelerator.device, dtype=torch.bfloat16).eval()
        peel_peft_wrapper(reference).set_attention_backend("native")
        active = False
        evaluations, metrics, replay_count, reference_calls = [], [], 0, 0
        initial = _cpu_tree(dict(trainer.model_bundle.named_parameters()))
        sample_batch, evaluate = trainer.sample_batch, trainer.evaluate

        def checked_forward(module, args, kwargs, output):
            nonlocal reference_calls
            if active:
                expected = reference(*args, **kwargs)
                left = output[0] if isinstance(output, tuple) else output.sample
                right = expected[0] if isinstance(expected, tuple) else expected.sample
                _bf16_tensors(
                    {"velocity": left.detach().float()},
                    {"velocity": right.detach().float()},
                    atol=2 / 128,
                )
                reference_calls += 1
            return output

        module.register_forward_hook(checked_forward, with_kwargs=True)

        def observed_samples(batch, **kwargs):
            if active:
                source = batch["source"][0] if "source" in batch else batch["__source__"][0]
                if source == "odd":
                    assert kwargs["guidance_scale"] == 3.0 and kwargs["num_inference_steps"] == 2
                elif source == "single":
                    assert (
                        kwargs["height"] == 32
                        and kwargs["width"] == 48
                        and kwargs["num_inference_steps"] == 3
                    )
                else:
                    raise AssertionError(source)
                assert kwargs["seed"] == 0
                expected_ema = trainer.adapter.ema_wrapper.ema_parameters
                _compare(
                    _cpu_tree(trainer.adapter._ema_tracked_parameters()),
                    _cpu_tree(expected_ema),
                    exact=True,
                )
            elif options["algorithm"] == "grpo":
                kwargs["extra_call_back_kwargs"] = ["velocity", "next_latents_mean"]
            return sample_batch(batch, **kwargs)

        trainer.sample_batch = observed_samples

        def observed_evaluate():
            nonlocal active
            before = _cpu_tree(dict(trainer.model_bundle.named_parameters()))
            # Build the independent model from the authoritative original shards,
            # then replace its tracked values directly from the EMA payload.
            state = _cpu_tree(module.state_dict())
            names = {id(p): name for name, p in module.named_parameters()}
            for parameter, ema in zip(
                trainer.adapter._ema_tracked_parameters(),
                trainer.adapter.ema_wrapper.ema_parameters,
                strict=True,
            ):
                key = names[id(parameter)].replace("_checkpoint_wrapped_module.", "")
                state[key] = full_value(ema).cpu().clone()
            reference.load_state_dict(state, strict=True)
            active = True
            try:
                evaluate()
            finally:
                active = False
            _compare(_cpu_tree(dict(trainer.model_bundle.named_parameters())), before, exact=True)
            assert all(not b.all_samples for b in trainer.eval_dataset_reward_buffers.values())
            evaluations.append(trainer.step)

        trainer.evaluate = observed_evaluate
        log_data = trainer.log_data

        def observed_log(data, **kwargs):
            if any(k.startswith("eval/") for k in data):
                metrics.append({k: v for k, v in data.items() if "reward_" in k})
            return log_data(data, **kwargs)

        trainer.log_data = observed_log
        if options["algorithm"] == "grpo":
            replay_forward = trainer._replay_forward

            def checked_replay(batch, replay, return_fields):
                nonlocal replay_count
                output = replay_forward(
                    batch,
                    replay,
                    tuple(dict.fromkeys((*return_fields, "velocity", "next_latents_mean"))),
                )
                torch.testing.assert_close(output.log_prob, replay.log_prob, rtol=0, atol=0)
                positions = (
                    (batch["timesteps"] == replay.times.timestep["latent"].reshape(-1, 1))
                    .all(dim=0)
                    .nonzero()
                    .flatten()
                )
                assert positions.numel() == 1
                for field, actual in (
                    ("velocity", output.velocity),
                    ("next_latents_mean", output.next_state_mean),
                ):
                    expected = trainer.adapter.get_replay_callback(batch, positions.item(), field)
                    for name in expected.component_names:
                        torch.testing.assert_close(
                            actual.components[name], expected.components[name], rtol=0, atol=0
                        )
                replay_count += 1
                return output

            trainer._replay_forward = checked_replay
        optimizer_step = trainer.optimizer.step
        gradient_checks = []

        def checked_step(*args, **kwargs):
            gradients = [
                full_value(p.grad) for p in trainer.model_bundle.parameters() if p.grad is not None
            ]
            assert gradients and all(torch.isfinite(g).all() for g in gradients)
            assert any(torch.count_nonzero(g) for g in gradients)
            gradient_checks.append(len(gradients))
            return optimizer_step(*args, **kwargs)

        trainer.optimizer.step = checked_step
        trainer.start()
        final = _cpu_tree(dict(trainer.model_bundle.named_parameters()))
        assert any(not torch.equal(v, final[k]) for k, v in initial.items())
        assert len(evaluations) == 2 and reference_calls > 0
        assert evaluations[1] > evaluations[0]
        assert replay_count == (8 if options["algorithm"] == "grpo" else 0)
        scored = [None] * world
        dist.all_gather_object(scored, PromptReward.scored)
        expected = Counter({("odd", f"a red video {i}"): 2 for i in range(3)})
        expected[("single", "a blue video 0")] = 2
        assert Counter(tuple(row) for rank_rows in scored for row in rank_rows) == expected
        if rank == 0:
            assert len(metrics) == 4
            for row in metrics:
                source = next(iter(row)).split("/")[1]
                assert row[f"eval/{source}/reward_score_count"] == (3 if source == "odd" else 1)
                assert row[f"eval/{source}/reward_score_mean"] == (1.0 if source == "odd" else 0.0)
                assert row[f"eval/{source}/reward_score_min"] == 0.0
                assert row[f"eval/{source}/reward_score_max"] == (2.0 if source == "odd" else 0.0)
                expected_std = (2.0 / 3.0) ** 0.5 if source == "odd" else 0.0
                assert row[f"eval/{source}/reward_score_std"] == pytest.approx(
                    expected_std, rel=1e-12
                )
        result.update(
            status="pass",
            evaluations=evaluations,
            independent_ema_forwards=reference_calls,
            exact_restoration=True,
            exactly_once_rewards=True,
            replay_checks=replay_count,
            optimizer_steps=trainer.step,
            gradient_checks=len(gradient_checks),
            elementwise_comparison=True,
            nonzero_gradients=True,
            varying_rewards=True,
            execution_capability="wan-cp-evaluation-v1",
        )
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result, indent=2))
        dist.destroy_process_group()


def _qualify_evaluation(tmp_path, trainer_type, mode, tp_size, cp_size):
    world = 2 * cp_size * tp_size
    if torch.cuda.device_count() < world:
        pytest.skip(f"native evaluation qualification requires {world} CUDA ranks")
    root = tmp_path / "fixture"
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("wan_sft_fixture.py")),
            str(root),
            str(2 * tp_size),
        ],
        check=True,
    )
    for name, count, color in (("prompts", 8, "red"), ("odd", 3, "red"), ("single", 1, "blue")):
        folder = root / name
        folder.mkdir()
        split = "train" if name == "prompts" else "test"
        (folder / f"{split}.jsonl").write_text(
            "".join(json.dumps({"prompt": f"a {color} video {i}"}) + "\n" for i in range(count))
        )
    rows = run_gloo_ranks(
        _worker,
        tmp_path / "ranks",
        world,
        {
            "algorithm": trainer_type,
            "mode": mode,
            "tp_size": tp_size,
            "cp_size": cp_size,
            "fixture": str(root),
        },
    )
    assert all(row.get("status") == "pass" for row in rows), rows
    print(json.dumps({"probe": "cp_evaluation", "ranks": rows}))


@pytest.mark.parametrize("trainer_type", ["sft", "grpo", "dmd2"])
@pytest.mark.parametrize("mode", ["full", "lora"])
@pytest.mark.parametrize("tp_size", [1, 2])
def test_cuda_cp_evaluation_preserves_ema_and_training(tmp_path, trainer_type, mode, tp_size):
    _qualify_evaluation(tmp_path, trainer_type, mode, tp_size, cp_size=2)


@pytest.mark.parametrize("trainer_type,mode,tp_size", [("sft", "full", 1), ("grpo", "lora", 2)])
def test_cuda_cp_off_evaluation_regression(tmp_path, trainer_type, mode, tp_size):
    _qualify_evaluation(tmp_path, trainer_type, mode, tp_size, cp_size=1)
