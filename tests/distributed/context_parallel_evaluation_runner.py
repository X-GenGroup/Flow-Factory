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
import math
import os
import statistics
import traceback
from collections import Counter
from importlib import import_module
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from conftest import run_gloo_ranks
from test_context_parallel_state import _cpu_tree, _state
from test_context_parallel_training import _bf16_tensors, _compare
from test_model_preparation import full_value, initialize_cuda, runtime_observation
from wan_eval_reward import PromptReward


def _assert_statistics(row, source, prompts):
    # The oracle reads only original dataset rows, independent of loader padding
    # and the distributed summary implementation.
    values = [float(prompt.rsplit(" ", 1)[-1]) for prompt in prompts]
    prefix = f"eval/{source}/reward_score_"
    assert row[prefix + "count"] == len(values)
    expected = (
        {
            "mean": statistics.fmean(values),
            "std": statistics.pstdev(values),
            "min": min(values),
            "max": max(values),
        }
        if values
        else dict.fromkeys(("mean", "std", "min", "max"), float("nan"))
    )
    for name, value in expected.items():
        actual = row[prefix + name]
        if math.isnan(value):
            assert math.isnan(actual)
        else:
            assert actual == pytest.approx(value, rel=1e-12, abs=0)


def _config(root, result_dir, options):
    return import_module(options.get("case_module", "wan_evaluation_fixture")).make_config(
        root, result_dir, options
    )


def _worker(rank, store_path, result_dir, options):
    case = import_module(options.get("case_module", "wan_evaluation_fixture"))
    os.environ.update(
        FSDP_VERSION="2",
        FSDP_AUTO_WRAP_POLICY="TRANSFORMER_BASED_WRAP",
        FSDP_TRANSFORMER_CLS_TO_WRAP=",".join(case.wrap_classes),
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
        loaded = _state(trainer) if options.get("state_resume") else None
        initial_cycle = trainer.cycle_index
        cycles = options.get("epochs", 2) - initial_cycle
        prompts = case._evaluation_prompts(Path(options["fixture"]))
        model_degree = options["cp_size"] * options["tp_size"]
        data_rank = rank // model_degree
        registry = trainer.adapter.component_variant_registry
        route = registry.resolve_route(registry.base_variant, "transformer")
        module = registry.bundle_members()[route]
        reference = native[0].to(device=trainer.accelerator.device, dtype=torch.bfloat16).eval()
        peel_peft_wrapper(reference).set_attention_backend("native")
        active = False
        evaluations, metrics, replay_count, reference_calls = [], [], 0, 0
        inputs, media, updates, saved = [], {name: [] for name in prompts}, [], {}
        failed_evaluations = 0
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
                elif source == "divisible":
                    assert kwargs["guidance_scale"] == 1.0 and kwargs["num_inference_steps"] == 3
                else:
                    raise AssertionError(source)
                inputs.append((source, list(batch["prompt"])))
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
        evaluate_dataset = trainer._evaluate_dataset

        def observed_dataset(source, loader):
            data = evaluate_dataset(source, loader)
            samples = data[f"eval/{source}/samples"]
            actual = [sample.prompt for sample in samples]
            # Batch size one and DP2: each real row belongs to its alternating
            # logical replica, including replicas that have no real media.
            assert actual == prompts[source][data_rank::2]
            assert all(sample.source == source for sample in samples)
            media[source].append(actual)
            _assert_statistics(data, source, prompts[source])
            return data

        trainer._evaluate_dataset = observed_dataset

        def observed_evaluate():
            nonlocal active, failed_evaluations
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
            # Inject once after a real update, when EMA differs from live weights.
            # All ranks finish the same async reward collectives before raising.
            if trainer.cycle_index == 1:
                tracked = _cpu_tree(trainer.adapter._ema_tracked_parameters())
                ema = _cpu_tree(trainer.adapter.ema_wrapper.ema_parameters)
                assert any(not torch.equal(a, b) for a, b in zip(tracked, ema, strict=True))
                buffer = trainer.eval_dataset_reward_buffers["odd"]
                finalize = buffer.finalize

                def fail_after_rewards(*args, **kwargs):
                    finalize(*args, **kwargs)
                    raise RuntimeError("injected native evaluation failure")

                buffer.finalize = fail_after_rewards
                active = True
                try:
                    with pytest.raises(RuntimeError, match="injected native evaluation failure"):
                        evaluate()
                finally:
                    active = False
                    buffer.finalize = finalize
                _compare(
                    _cpu_tree(dict(trainer.model_bundle.named_parameters())), before, exact=True
                )
                assert all(not b.all_samples for b in trainer.eval_dataset_reward_buffers.values())
                failed_evaluations += 1
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
            returned = optimizer_step(*args, **kwargs)
            if options.get("state_resume"):
                updates.append({"gradients": _cpu_tree(gradients), "state": _state(trainer)})
            return returned

        trainer.optimizer.step = checked_step
        save = trainer._save_exact_training_state

        def observed_save(path):
            saved[trainer.cycle_index] = _state(trainer)
            return save(path)

        if options.get("state_resume"):
            trainer._save_exact_training_state = observed_save
        trainer.start()
        final = _cpu_tree(dict(trainer.model_bundle.named_parameters()))
        assert any(not torch.equal(v, final[k]) for k, v in initial.items())
        assert len(evaluations) == cycles and reference_calls > 0
        assert all(after > before for before, after in zip(evaluations, evaluations[1:]))
        assert failed_evaluations == int(initial_cycle <= 1 < options.get("epochs", 2))
        assert all(len(rows) == cycles for rows in media.values())
        assert replay_count == (8 if options["algorithm"] == "grpo" else 0)
        scored, peer_inputs = [None] * world, [None] * world
        dist.all_gather_object(scored, PromptReward.scored)
        dist.all_gather_object(peer_inputs, inputs)
        for start in range(0, world, model_degree):
            assert all(
                peer_inputs[r] == peer_inputs[start] for r in range(start, start + model_degree)
            )
        expected = Counter(
            {(source, prompt): cycles for source, rows in prompts.items() for prompt in rows}
        )
        expected.update({("odd", prompt): failed_evaluations for prompt in prompts["odd"]})
        assert Counter(tuple(row) for rank_rows in scored for row in rank_rows) == expected
        for worker_rank, rank_rows in enumerate(scored):
            data_rank, peer_rank = divmod(worker_rank, model_degree)
            owned_prompts = {
                source: rows[data_rank::2][peer_rank::model_degree]
                for source, rows in prompts.items()
            }
            expected_local = Counter(
                {
                    (source, prompt): cycles
                    for source, rows in owned_prompts.items()
                    for prompt in rows
                }
            )
            expected_local.update(
                {("odd", prompt): failed_evaluations for prompt in owned_prompts["odd"]}
            )
            assert Counter(tuple(row) for row in rank_rows) == expected_local
        if rank == 0:
            assert len(metrics) == len(prompts) * cycles
            for row in metrics:
                source = next(iter(row)).split("/")[1]
                _assert_statistics(row, source, prompts[source])
        if options.get("state_resume"):
            torch.save(
                {"loaded": loaded, "saved": saved, "final": _state(trainer), "updates": updates},
                Path(result_dir, f"observations-{rank}.pt"),
            )
        result.update(
            status="pass",
            evaluations=evaluations,
            independent_ema_forwards=reference_calls,
            exact_restoration=True,
            exactly_once_rewards=True,
            datasets={name: len(rows) for name, rows in prompts.items()},
            unpadded_media=True,
            model_peer_inputs=True,
            pointwise_reward_ownership=True,
            exception_cleanup=failed_evaluations,
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


def _qualify_evaluation(
    tmp_path, trainer_type, mode, tp_size, cp_size, *, case_module="wan_evaluation_fixture"
):
    world = 2 * cp_size * tp_size
    if torch.cuda.device_count() < world:
        pytest.skip(f"native evaluation qualification requires {world} CUDA ranks")
    root = tmp_path / "fixture"
    case = import_module(case_module)
    case.build_fixture(root, tp_size)
    state_resume = trainer_type == "sft" and cp_size == 2
    options = {
        "algorithm": trainer_type,
        "mode": mode,
        "tp_size": tp_size,
        "cp_size": cp_size,
        "fixture": str(root),
        "case_module": case_module,
        "state_resume": state_resume,
        "epochs": 3 if state_resume else 2,
    }
    rows = run_gloo_ranks(
        _worker,
        tmp_path / "ranks",
        world,
        options,
    )
    assert all(row.get("status") == "pass" for row in rows), rows
    resumed_rows = []
    if state_resume:
        # Epoch 1 evaluates/restores EMA, epoch 2 trains and saves, then a new
        # process group restores checkpoint 2 and repeats the final epoch.
        resumed_rows = run_gloo_ranks(
            _worker,
            tmp_path / "resumed",
            world,
            {**options, "resume": str(tmp_path / "ranks/saves/wan-sft/checkpoints/checkpoint-2")},
        )
        assert all(row.get("status") == "pass" for row in resumed_rows), resumed_rows
        for rank in range(world):
            continuous = torch.load(tmp_path / f"ranks/observations-{rank}.pt", weights_only=True)
            resumed = torch.load(tmp_path / f"resumed/observations-{rank}.pt", weights_only=True)
            _compare(resumed["loaded"], continuous["saved"][2], exact=True)
            assert len(continuous["updates"]) == 6 and len(resumed["updates"]) == 2
            _compare(resumed["updates"], continuous["updates"][4:], exact=True)
            _compare(resumed["final"], continuous["final"], exact=True)
    print(
        json.dumps(
            {
                "probe": "cp_evaluation",
                "ranks": rows,
                "resumed_ranks": resumed_rows,
                "exact_evaluation_resume": state_resume,
            }
        )
    )
