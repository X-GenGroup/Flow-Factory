"""One GPU-plan launch.

The process loads the real trainer, rolls one microbatch, and writes the
noise tensors the PR compares. A second launch is a second process. Feat and
main are selected by ``PYTHONPATH``, not by a branch inside this file.
"""

from __future__ import annotations

import argparse
import functools
import importlib
import inspect
from pathlib import Path

import torch
from cells import require_cell
from keys import (
    BRANCH,
    CELL,
    CRD_NOISE,
    CRD_PASS2_RANDN,
    CRD_TIMES,
    DPO_NOISE,
    DPO_TIMES,
    EVAL_GENERATOR_SEED,
    EVAL_SEED,
    EVAL_X_T,
    FRESH_NOISE,
    GROUP_IDS,
    H3_RUNS,
    INIT_SAME_NOISE,
    LAUNCH,
    MEMBER_IDS,
    NFT_NOISE,
    NFT_TIMES,
    NUM_TRAIN_TIMESTEPS,
    PAIRS,
    PLAN_ORDER,
    QUERY,
    RANK,
    RECORD_IDS,
    REPLAY_LOG_PROB,
    ROW_ORDERS,
    SAMPLE_IDS,
    SCHEDULER_SEED,
    SDE_AFTER,
    SDE_BEFORE,
    SDE_NOISE,
    SDE_STEP_NOISE,
    SHARED_NOISE,
    TERM_NOISE,
    TIME_MATRIX,
    TRAIN_SEED,
    USE_SHARED_NOISE,
    WORLD,
    X_T,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one GPU-plan cell and dump tensors.")
    parser.add_argument("--cell", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--branch", required=True, choices=("feat", "main"))
    parser.add_argument("--launch", required=True, type=int)
    return parser.parse_args()


def call_with_supported(fn, positional: tuple, keywords: dict):
    """Call ``fn`` with the keyword arguments its signature accepts."""

    params = inspect.signature(fn).parameters
    accepts_var = any(item.kind == inspect.Parameter.VAR_KEYWORD for item in params.values())
    accepted = {key: value for key, value in keywords.items() if key in params or accepts_var}
    return fn(*positional, **accepted)


def extract_components(state) -> dict[str, torch.Tensor]:
    """Return float32 CPU components from a latent or noised state."""

    if hasattr(state, "noise") and not hasattr(state, "components"):
        return extract_components(state.noise)
    components = getattr(state, "components", None)
    if not isinstance(components, dict) or not components:
        raise TypeError(f"expected a component state, received {type(state).__name__}: {state!r}")
    extracted = {}
    for name, value in components.items():
        if not torch.is_tensor(value):
            raise TypeError(f"expected tensor component {name!r}, received {type(value).__name__}")
        extracted[str(name)] = value.detach().float().cpu()
    return extracted


def initial_trajectory_state(sample) -> dict[str, torch.Tensor]:
    """Return each H3 component's stored initial latent."""

    trajectory = getattr(sample, "trajectory", None)
    components = getattr(trajectory, "components", None)
    if not isinstance(components, dict) or not components:
        raise RuntimeError(
            "expected an H3 sample.trajectory with component states, "
            f"received {type(trajectory).__name__}"
        )
    extracted = {}
    for name, component in components.items():
        index_map = component.state_index_map
        stored = int(index_map[0].item())
        states = component.states
        if stored < 0 or stored >= states.shape[0]:
            raise RuntimeError(
                f"component {name} rollout position 0 maps to stored index {stored}, "
                f"states shape {tuple(states.shape)}, index_map={index_map.tolist()}"
            )
        extracted[str(name)] = states[stored].detach().float().cpu()
    return extracted


def initial_latent(sample):
    """Return the stored x_T for one sample.

    Image adapters store one ``all_latents`` tensor. MiniMax H3 stores a
    structured trajectory with separate video and audio states.
    """

    latents = getattr(sample, "all_latents", None)
    if not torch.is_tensor(latents) or latents.shape[0] < 1:
        if getattr(sample, "trajectory", None) is not None:
            return initial_trajectory_state(sample)
        raise RuntimeError(
            "expected sample.all_latents to be a non-empty trajectory tensor, "
            f"received {None if latents is None else tuple(getattr(latents, 'shape', None))}"
        )
    index_map = getattr(sample, "latent_index_map", None)
    if torch.is_tensor(index_map):
        matches = (index_map == 0).nonzero(as_tuple=False)
        if matches.numel() != 1:
            raise RuntimeError(
                "expected one stored trajectory index 0, "
                f"received latent_index_map={index_map.tolist()}"
            )
        return latents[int(matches[0])].detach().float().cpu()
    return latents[0].detach().float().cpu()


def xt_by_id(samples) -> dict[str, torch.Tensor]:
    keyed = {}
    for index, sample in enumerate(samples):
        sample_id = getattr(sample, "sampling_sample_id", None)
        if sample_id is None:
            raise RuntimeError(f"sample[{index}] is missing sampling_sample_id")
        keyed[str(int(sample_id))] = initial_latent(sample)
    if not keyed:
        raise RuntimeError("rollout produced no samples")
    return keyed


def next_batch(trainer):
    if getattr(trainer, "_gpu_plan_iter", None) is None:
        loader = trainer.dataloader
        if loader is None:
            raise RuntimeError(f"{type(trainer).__name__} has no training dataloader")
        if hasattr(loader, "set_epoch"):
            loader.set_epoch(0)
        trainer._gpu_plan_iter = iter(loader)
    try:
        return next(trainer._gpu_plan_iter)
    except StopIteration as exc:
        raise RuntimeError("the gpu-plan dataloader ran out of batches") from exc


def plan_order(trainer) -> list[int]:
    sampler = getattr(trainer.dataloader, "batch_sampler", None)
    if sampler is None or not hasattr(sampler, "_build_plan"):
        raise TypeError(
            "expected the train batch sampler to expose _build_plan, "
            f"received {type(sampler).__name__}"
        )
    plan = sampler._build_plan(epoch=0)
    rank = int(trainer.accelerator.process_index)
    order = [int(assignment) for batch in plan.batches_for_rank(rank) for assignment in batch]
    if not order:
        raise RuntimeError(f"rank {rank} sampling plan is empty")
    return order


def snapshot_schedulers(adapter) -> dict:
    group = adapter.scheduler_group
    by_name = {}
    for name in group.names:
        scheduler = group[name]
        by_name[name] = {
            "steps": scheduler.current_sde_steps.detach().cpu().to(torch.int64),
            "seed": int(scheduler.seed),
        }
    return {"primary": group.primary_name, "by_name": by_name}


def record_cycle(trainer, payload: dict) -> None:
    payload[SDE_BEFORE] = snapshot_schedulers(trainer.adapter)
    trainer.acquisition_driver.prepare_cycle(
        trainer,
        trainer.progress,
        seed=int(trainer.training_args.seed),
    )
    payload[SDE_AFTER] = snapshot_schedulers(trainer.adapter)


def split_noise(drawn: list[torch.Tensor], sample_ids: list[str]) -> dict[str, torch.Tensor]:
    if len(drawn) == 1 and drawn[0].shape[:1] == (len(sample_ids),):
        return {sample_ids[index]: drawn[0][index].clone() for index in range(len(sample_ids))}
    if len(drawn) == len(sample_ids):
        return {sample_ids[index]: drawn[index].clone() for index in range(len(sample_ids))}
    shapes = [tuple(tensor.shape) for tensor in drawn]
    raise ValueError(f"cannot align {len(drawn)} randn draws {shapes} with sample ids {sample_ids}")


def install_step_recorder(adapter, *, replay: bool):
    """Record every Gaussian drawn inside ``scheduler.step``."""

    recorded: dict[str, list] = {}
    originals = {}
    replay_log: list[tuple[torch.Tensor, torch.Tensor]] = []
    for name in adapter.scheduler_group.names:
        scheduler = adapter.scheduler_group[name]
        if id(scheduler) in originals:
            recorded[name] = originals[id(scheduler)][2]
            continue
        bucket: list = []
        original = scheduler.step

        def make_step(original_step, destination):
            state = {"replayed": False}

            @functools.wraps(original_step)
            def step(*args, **kwargs):
                drawn = []
                real_randn = torch.randn

                def capturing_randn(*randn_args, **randn_kwargs):
                    out = real_randn(*randn_args, **randn_kwargs)
                    drawn.append(out.detach().float().cpu())
                    return out

                torch.randn = capturing_randn
                try:
                    result = original_step(*args, **kwargs)
                finally:
                    torch.randn = real_randn
                if drawn:
                    destination.append(drawn)
                log_prob = getattr(result, "log_prob", None)
                next_latents = getattr(result, "next_latents", None)
                if (
                    replay
                    and drawn
                    and not state["replayed"]
                    and kwargs.get("next_latents") is None
                    and torch.is_tensor(log_prob)
                    and torch.is_tensor(next_latents)
                ):
                    state["replayed"] = True
                    replay_kwargs = dict(kwargs)
                    replay_kwargs["next_latents"] = next_latents
                    replayed = original_step(*args, **replay_kwargs)
                    replayed_log_prob = getattr(replayed, "log_prob", None)
                    if not torch.is_tensor(replayed_log_prob):
                        raise RuntimeError(
                            "scheduler replay with next_latents did not return log_prob"
                        )
                    replay_log.append(
                        (
                            log_prob.detach().float().cpu(),
                            replayed_log_prob.detach().float().cpu(),
                        )
                    )
                return result

            return step

        scheduler.step = make_step(original, bucket)
        recorded[name] = bucket
        originals[id(scheduler)] = (scheduler, original, bucket)

    def restore() -> None:
        for scheduler, original, _bucket in originals.values():
            scheduler.step = original

    return recorded, replay_log, restore


def snapshot_noise(recorded, sample_ids: list[str]) -> dict:
    parsed = {}
    cleared = set()
    for name, steps in recorded.items():
        parsed[name] = [split_noise(drawn, sample_ids) for drawn in steps]
        if id(steps) not in cleared:
            steps.clear()
            cleared.add(id(steps))
    if not any(parsed.values()):
        raise RuntimeError("scheduler.step drew no Gaussian noise")
    return parsed


def batch_sample_ids(batch) -> list[str]:
    from flow_factory.data_utils.sampling_plan import SAMPLING_SAMPLE_ID_COLUMN

    values = batch.get(SAMPLING_SAMPLE_ID_COLUMN)
    if not isinstance(values, (list, tuple)) or not values:
        raise RuntimeError(f"batch is missing {SAMPLING_SAMPLE_ID_COLUMN}; received {values!r}")
    return [str(int(value)) for value in values]


def identity_lists(samples) -> tuple[list[str], list[int], list[int]]:
    sample_ids = []
    group_ids = []
    member_ids = []
    for index, sample in enumerate(samples):
        sample_id = getattr(sample, "sampling_sample_id", None)
        group_id = getattr(sample, "sampling_group_id", None)
        member_id = getattr(sample, "sampling_group_member_id", None)
        if sample_id is None or group_id is None or member_id is None:
            raise RuntimeError(
                f"sample[{index}] is missing planned identity: "
                f"sample={sample_id}, group={group_id}, member={member_id}"
            )
        sample_ids.append(str(int(sample_id)))
        group_ids.append(int(group_id))
        member_ids.append(int(member_id))
    return sample_ids, group_ids, member_ids


def capture_rollout(trainer, payload, *, microbatches: int, compute_log_prob: bool, replay: bool):
    record_cycle(trainer, payload)
    recorded, replay_log, restore = install_step_recorder(trainer.adapter, replay=replay)
    samples = []
    try:
        for index in range(microbatches):
            batch = next_batch(trainer)
            sample_ids = batch_sample_ids(batch)
            rolled = trainer.sample_batch(
                batch,
                reward_buffer=None,
                compute_log_prob=compute_log_prob,
            )
            noise = snapshot_noise(recorded, sample_ids)
            if index == 0:
                payload[SAMPLE_IDS] = sample_ids
                payload[X_T] = xt_by_id(rolled)
                payload[SDE_NOISE] = noise
            samples.extend(rolled)
    finally:
        restore()
    sample_ids, group_ids, member_ids = identity_lists(samples)
    payload[GROUP_IDS] = group_ids
    payload[MEMBER_IDS] = member_ids
    if replay:
        if not replay_log:
            raise RuntimeError("DGPO rollout did not replay a stored next_latents log_prob")
        payload[REPLAY_LOG_PROB] = [replay_log[0][0], replay_log[0][1]]
    return samples


def reverse_rows(batch) -> dict:
    prompts = batch.get("prompt")
    if not isinstance(prompts, (list, tuple)) or len(prompts) < 2:
        raise ValueError(f"row swap requires at least 2 prompt rows, received {prompts!r}")
    count = len(prompts)
    return {key: _reverse_value(key, value, count) for key, value in batch.items()}


def _reverse_value(key, value, count: int):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        if len(value) != count:
            raise ValueError(f"field {key} has length {len(value)}, expected {count}")
        return list(reversed(value))
    if torch.is_tensor(value):
        if value.shape[:1] != (count,):
            raise ValueError(
                f"field {key} shape {tuple(value.shape)} does not start with batch {count}"
            )
        return torch.flip(value, [0])
    if isinstance(value, dict):
        return {name: _reverse_value(f"{key}.{name}", item, count) for name, item in value.items()}
    raise TypeError(f"field {key} has unsupported type {type(value).__name__}")


def capture_row_swap(trainer, payload) -> None:
    record_cycle(trainer, payload)
    batch = next_batch(trainer)
    recorded, _replay_log, restore = install_step_recorder(trainer.adapter, replay=False)
    orders = []
    try:
        for label, rows in (("forward", batch), ("reversed", reverse_rows(batch))):
            sample_ids = batch_sample_ids(rows)
            samples = trainer.sample_batch(rows, reward_buffer=None, compute_log_prob=False)
            orders.append(
                {
                    "label": label,
                    "sample_ids": sample_ids,
                    "x_t": xt_by_id(samples),
                    "sde_noise": snapshot_noise(recorded, sample_ids),
                }
            )
    finally:
        restore()
    payload[ROW_ORDERS] = orders


def capture_h3(trainer, payload) -> None:
    record_cycle(trainer, payload)
    batch = next_batch(trainer)
    sample_ids = batch_sample_ids(batch)
    recorded, _replay_log, restore = install_step_recorder(trainer.adapter, replay=False)
    from accelerate.utils import set_seed

    runs = []
    try:
        for generator_seed in (1, 2):
            set_seed(int(trainer.training_args.seed), device_specific=True)
            generator = torch.Generator(device="cpu")
            generator.manual_seed(generator_seed)
            samples = trainer.sample_batch(
                batch,
                reward_buffer=None,
                compute_log_prob=False,
                generator=generator,
            )
            runs.append(
                {
                    "generator_seed": generator_seed,
                    "x_t": xt_by_id(samples),
                    "sde_noise": snapshot_noise(recorded, sample_ids),
                }
            )
    finally:
        restore()
    payload[H3_RUNS] = runs
    payload[SAMPLE_IDS] = sample_ids


def stacked_batch(trainer, samples):
    batch = next(trainer._iter_prefetched_batches(samples, len(samples)))
    if len(getattr(batch, "samples", ())) != len(samples):
        raise RuntimeError(
            f"stacked batch has {len(getattr(batch, 'samples', ()))} samples, "
            f"expected {len(samples)}"
        )
    return batch


def capture_dgpo(trainer, samples, payload) -> None:
    batch = stacked_batch(trainer, samples)
    clean = trainer.adapter.get_terminal_state(batch)
    noises = []
    for timestep_index in (0, 1):
        noise = call_with_supported(
            trainer._shared_group_noise,
            (clean, batch.samples, 0),
            {"timestep_index": timestep_index},
        )
        noises.append(extract_components(noise))
    payload[SHARED_NOISE] = noises


def capture_nft(trainer, samples, payload) -> None:
    batch = stacked_batch(trainer, samples)
    clean = trainer.adapter.get_terminal_state(batch)
    recorded_times = []
    original = trainer._sample_timesteps

    def wrapped(*args, **kwargs):
        out = original(*args, **kwargs)
        if not torch.is_tensor(out):
            raise TypeError(f"NFT timesteps must be a tensor, received {type(out).__name__}")
        recorded_times.append(out.detach().float().cpu())
        return out

    trainer._sample_timesteps = wrapped
    noises = []
    try:
        for batch_index in (0, 1):
            before = len(recorded_times)
            steps = call_with_supported(
                trainer._precompute_sampling_policy_steps,
                (batch, clean, len(samples)),
                {"inner_epoch": 0, "batch_index": batch_index},
            )
            if len(recorded_times) != before + 1:
                raise RuntimeError(
                    "NFT precompute should sample timesteps once per batch index, "
                    f"batch_index={batch_index} added {len(recorded_times) - before}"
                )
            noises.append([extract_components(step.noised) for step in steps])
    finally:
        trainer._sample_timesteps = original
    payload[NFT_TIMES] = recorded_times
    payload[NFT_NOISE] = noises


def capture_crd(trainer, samples, payload) -> None:
    batch = stacked_batch(trainer, samples)
    recorded_times = []
    original = trainer._sample_timesteps

    def wrapped(*args, **kwargs):
        out = original(*args, **kwargs)
        recorded_times.append(out.detach().float().cpu())
        return out

    trainer._sample_timesteps = wrapped
    try:
        prepared = call_with_supported(
            trainer._precompute_old_velocities,
            (batch,),
            {"inner_epoch": 0},
        )
    finally:
        trainer._sample_timesteps = original
    if len(recorded_times) != 1:
        raise RuntimeError(f"expected one CRD timestep draw, received {len(recorded_times)}")
    payload[CRD_TIMES] = recorded_times[0]
    payload[CRD_NOISE] = [extract_components(step.noise) for step in prepared.steps]
    clean = trainer.adapter.get_terminal_state(batch)
    count = {"n": 0}
    real_randn = torch.randn

    def counting_randn(*args, **kwargs):
        count["n"] += 1
        return real_randn(*args, **kwargs)

    torch.randn = counting_randn
    try:
        for index, step in enumerate(prepared.steps):
            call_with_supported(
                trainer._rebuild_noised_state,
                (clean, step),
                {"timestep_index": index},
            )
    finally:
        torch.randn = real_randn
    payload[CRD_PASS2_RANDN] = int(count["n"])


def _one_record_id(batch) -> str:
    record_ids = getattr(batch, "record_ids", None)
    if not isinstance(record_ids, (list, tuple)) or len(record_ids) != 1:
        raise RuntimeError(f"expected one offline record id, received {record_ids!r}")
    return str(record_ids[0])


def capture_sft(trainer, payload) -> None:
    module = importlib.import_module("flow_factory.trainers.offline.sft")
    current = {"times": None, "noise": []}
    original_times = module.sample_offline_timesteps
    original_noise = module.build_noised_output_state

    def wrap_times(*args, **kwargs):
        out = original_times(*args, **kwargs)
        current["times"] = out.detach().float().cpu()
        return out

    def wrap_noise(*args, **kwargs):
        times, noised = original_noise(*args, **kwargs)
        current["noise"].append(extract_components(noised))
        return times, noised

    module.sample_offline_timesteps = wrap_times
    module.build_noised_output_state = wrap_noise
    matrices = []
    noises = []
    record_ids = []
    try:
        for _index in range(2):
            current["times"] = None
            current["noise"] = []
            batch = next_batch(trainer)
            record_ids.append(_one_record_id(batch))
            trainer.optimize_batch(batch)
            if not torch.is_tensor(current["times"]) or len(current["noise"]) != 2:
                raise RuntimeError(
                    "SFT microstep should record one time matrix and 2 noise terms, "
                    f"received times={type(current['times']).__name__} "
                    f"noise_terms={len(current['noise'])}"
                )
            matrices.append(current["times"])
            noises.append(list(current["noise"]))
    finally:
        module.sample_offline_timesteps = original_times
        module.build_noised_output_state = original_noise
    payload[TIME_MATRIX] = torch.cat(matrices, dim=1)
    payload[TERM_NOISE] = noises
    payload[RECORD_IDS] = record_ids


def capture_offline_dpo(trainer, payload) -> None:
    module = importlib.import_module("flow_factory.trainers.offline.offline_dpo")
    current = {"times": None, "noise": []}
    original_times = module.sample_offline_timesteps
    original_noise = module.build_noised_output_state

    def wrap_times(*args, **kwargs):
        out = original_times(*args, **kwargs)
        current["times"] = out.detach().float().cpu()
        return out

    def wrap_noise(*args, **kwargs):
        times, noised = original_noise(*args, **kwargs)
        current["noise"].append(extract_components(noised))
        return times, noised

    module.sample_offline_timesteps = wrap_times
    module.build_noised_output_state = wrap_noise
    pairs = []
    try:
        for _index in range(2):
            current["times"] = None
            current["noise"] = []
            batch = next_batch(trainer)
            record_id = _one_record_id(batch)
            trainer.optimize_batch(batch)
            if not torch.is_tensor(current["times"]) or len(current["noise"]) != 4:
                raise RuntimeError(
                    "offline DPO should record 2 terms x chosen/rejected, "
                    f"received {len(current['noise'])} noise calls"
                )
            pairs.append(
                {
                    "record_id": record_id,
                    "times": current["times"],
                    "terms": [
                        {"chosen": current["noise"][0], "rejected": current["noise"][1]},
                        {"chosen": current["noise"][2], "rejected": current["noise"][3]},
                    ],
                }
            )
    finally:
        module.sample_offline_timesteps = original_times
        module.build_noised_output_state = original_noise
    payload[PAIRS] = pairs


def capture_online_dpo(trainer, payload) -> None:
    record_cycle(trainer, payload)
    samples = []
    for _index in range(2):
        batch = next_batch(trainer)
        samples.extend(trainer.sample_batch(batch, reward_buffer=None, compute_log_prob=False))
    _sample_ids, group_ids, member_ids = identity_lists(samples)
    payload[SAMPLE_IDS] = _sample_ids
    payload[GROUP_IDS] = group_ids
    payload[MEMBER_IDS] = member_ids
    for sample, member_id in zip(samples, member_ids):
        sample.extra_kwargs["advantage"] = 1.0 if member_id == 0 else -1.0
    times = []
    noises = []
    original_times = trainer._sample_timesteps
    original_add = trainer.adapter.add_forward_process_noise
    original_apply = trainer.adapter.apply_forward_process_noise

    def wrap_times(*args, **kwargs):
        out = original_times(*args, **kwargs)
        times.append(out.detach().float().cpu())
        return out

    nesting = {"add": 0}

    def wrap_add(*args, **kwargs):
        # add_forward_process_noise applies the draw through apply_forward_process_noise.
        # Only the outer chosen/rejected calls belong in the pair record.
        nesting["add"] += 1
        try:
            out = original_add(*args, **kwargs)
        finally:
            nesting["add"] -= 1
        noises.append(("chosen", extract_components(out)))
        return out

    def wrap_apply(*args, **kwargs):
        out = original_apply(*args, **kwargs)
        if nesting["add"] == 0:
            noises.append(("rejected", extract_components(out)))
        return out

    trainer._sample_timesteps = wrap_times
    trainer.adapter.add_forward_process_noise = wrap_add
    trainer.adapter.apply_forward_process_noise = wrap_apply
    try:
        trainer.optimize(samples)
    finally:
        trainer._sample_timesteps = original_times
        trainer.adapter.add_forward_process_noise = original_add
        trainer.adapter.apply_forward_process_noise = original_apply
    if len(times) != 1:
        raise RuntimeError(f"expected one online-DPO timestep draw, received {len(times)}")
    if len(noises) != 4 or [name for name, _noise in noises] != [
        "chosen",
        "rejected",
        "chosen",
        "rejected",
    ]:
        raise RuntimeError(
            "expected chosen/rejected noise for 2 terms, "
            f"received {[name for name, _noise in noises]}"
        )
    payload[DPO_TIMES] = times[0]
    payload[DPO_NOISE] = [
        {"chosen": noises[0][1], "rejected": noises[1][1]},
        {"chosen": noises[2][1], "rejected": noises[3][1]},
    ]


def _reset_draw_counts(trainer) -> None:
    if hasattr(trainer, "_independent_draw_counts"):
        trainer._independent_draw_counts = {}


def _fresh_noise_calls(trainer, unit) -> list[list[torch.Tensor]]:
    matching = importlib.import_module("flow_factory.trainers.distillation.distribution_matching")
    try:
        group = importlib.import_module("flow_factory.trainers.common.group_noise")
    except ImportError:
        group = None

    def one_call() -> list[torch.Tensor]:
        with torch.no_grad():
            replay_batch, clean, model_noise = trainer._replay_generator_prediction(unit)
            times = trainer._sample_score_query_times(unit, replay_batch)
            generator = None
            if group is not None and hasattr(group, "sequenced_draw_generator"):
                generator = group.sequenced_draw_generator(
                    trainer,
                    group.STREAM_CONDITIONAL_FRESH,
                    group.sample_noise_ids(unit.samples),
                    unit.boundary_index,
                )
            drawn = []
            real_randn = torch.randn
            real_randn_like = torch.randn_like

            def capturing_randn(*args, **kwargs):
                out = real_randn(*args, **kwargs)
                drawn.append(out.detach().float().cpu())
                return out

            def capturing_randn_like(*args, **kwargs):
                # main draws fresh noise with randn_like; that path does not call torch.randn.
                out = real_randn_like(*args, **kwargs)
                drawn.append(out.detach().float().cpu())
                return out

            torch.randn = capturing_randn
            torch.randn_like = capturing_randn_like
            try:
                call_with_supported(
                    matching.tdm_conditional_renoise,
                    (trainer.adapter, clean, model_noise),
                    {
                        "mid_times": unit.mid_times,
                        "target_times": times,
                        "importance_clip": float(trainer.training_args.tdm_importance_clip),
                        "generator": generator,
                    },
                )
            finally:
                torch.randn = real_randn
                torch.randn_like = real_randn_like
        if not drawn:
            raise RuntimeError("tdm_conditional_renoise drew no Gaussian")
        return drawn

    _reset_draw_counts(trainer)
    first = one_call()
    second = one_call()
    _reset_draw_counts(trainer)
    restarted = one_call()
    return [first, second, restarted]


def capture_tdm(trainer, payload) -> None:
    record_cycle(trainer, payload)
    samples = trainer.sample()
    expected = int(trainer.training_args.per_device_batch_size)
    if len(samples) != expected:
        raise RuntimeError(f"TDM sample() returned {len(samples)} samples, expected {expected}")
    units = trainer._build_boundary_units(samples)
    if len(units) < 2:
        raise RuntimeError(f"expected at least 2 TDM boundaries, received {len(units)}")
    unit = units[0]
    _reset_draw_counts(trainer)
    query_one = trainer._sample_perturbation_times(unit).detach().float().cpu()
    query_two = trainer._sample_perturbation_times(unit).detach().float().cpu()
    _reset_draw_counts(trainer)
    query_restart = trainer._sample_perturbation_times(unit).detach().float().cpu()
    trainer.training_args.tdm_query_max_sigma = 0.3
    rebuilt = trainer._build_boundary_units(samples)
    _reset_draw_counts(trainer)
    query_sigma = trainer._sample_perturbation_times(rebuilt[0]).detach().float().cpu()
    payload[QUERY] = [query_one, query_two, query_restart, query_sigma]
    payload[FRESH_NOISE] = _fresh_noise_calls(trainer, unit)


def capture_eval(trainer, payload) -> None:
    if not trainer.eval_dataloaders:
        raise RuntimeError("eval capture requires a configured eval dataloader")
    from flow_factory.utils.base import create_generator_by_prompt

    name = next(iter(trainer.eval_dataloaders))
    batch = next(iter(trainer.eval_dataloaders[name]))
    dataset = trainer._eval_dataset_configs[name]
    batch = trainer._augment_batch_with_source(batch, name, dataset.source_id)
    used_seed = int(trainer.training_args.seed)
    generator = create_generator_by_prompt(batch["prompt"], used_seed)
    eval_kwargs = {}
    if dataset.eval:
        eval_kwargs = dict(dataset.eval.get_merged_eval_kwargs(trainer.eval_args))
    samples = trainer.sample_batch(
        batch,
        reward_buffer=None,
        compute_log_prob=False,
        generator=generator,
        **eval_kwargs,
    )
    if len(samples) != 1:
        raise RuntimeError(f"expected one eval sample, received {len(samples)}")
    payload[EVAL_X_T] = initial_latent(samples[0])
    payload[EVAL_GENERATOR_SEED] = used_seed


def fill_meta(trainer, payload, args, cell) -> None:
    training_args = trainer.training_args
    num_train_timesteps = getattr(trainer, "num_train_timesteps", None)
    if num_train_timesteps is None:
        num_train_timesteps = getattr(training_args, "num_train_timesteps", 0)
    payload.update(
        {
            CELL: cell.name,
            BRANCH: args.branch,
            LAUNCH: int(args.launch),
            RANK: int(trainer.accelerator.process_index),
            WORLD: int(trainer.accelerator.num_processes),
            TRAIN_SEED: int(training_args.seed),
            SCHEDULER_SEED: int(trainer.config.scheduler_args.seed),
            EVAL_SEED: int(trainer.eval_args.seed) if trainer.eval_args.seed is not None else None,
            INIT_SAME_NOISE: bool(getattr(training_args, "init_same_noise", False)),
            SDE_STEP_NOISE: "sample",
            USE_SHARED_NOISE: getattr(training_args, "use_shared_noise", None),
            NUM_TRAIN_TIMESTEPS: int(num_train_timesteps or 0),
        }
    )


def run_cell(trainer, cell, payload) -> None:
    samples = None
    if "plan" in cell.captures:
        payload[PLAN_ORDER] = plan_order(trainer)
    if "rollout" in cell.captures:
        samples = capture_rollout(
            trainer,
            payload,
            microbatches=cell.microbatches,
            compute_log_prob="dgpo" in cell.captures,
            replay="dgpo" in cell.captures,
        )
    if "eval" in cell.captures:
        capture_eval(trainer, payload)
    if "dgpo" in cell.captures:
        capture_dgpo(trainer, samples, payload)
    if "nft" in cell.captures:
        capture_nft(trainer, samples, payload)
    if "crd" in cell.captures:
        capture_crd(trainer, samples, payload)
    if "row_swap" in cell.captures:
        capture_row_swap(trainer, payload)
    if "h3" in cell.captures:
        capture_h3(trainer, payload)
    if "sft" in cell.captures:
        capture_sft(trainer, payload)
    if "offline_dpo" in cell.captures:
        capture_offline_dpo(trainer, payload)
    if "online_dpo" in cell.captures:
        capture_online_dpo(trainer, payload)
    if "tdm" in cell.captures:
        capture_tdm(trainer, payload)


def main() -> None:
    args = parse_args()
    cell = require_cell(args.cell)
    from flow_factory.hparams import Arguments
    from flow_factory.trainers.loader import load_trainer

    config_path = Path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"missing gpu-plan config {config_path}")
    trainer = load_trainer(Arguments.load_from_yaml(str(config_path)))
    world = int(trainer.accelerator.num_processes)
    if world != cell.world_size:
        raise RuntimeError(
            f"cell {cell.name} requires world_size={cell.world_size}, received {world}"
        )
    payload = {}
    fill_meta(trainer, payload, args, cell)
    run_cell(trainer, cell, payload)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rank = int(trainer.accelerator.process_index)
    destination = out / f"rank{rank}.pt"
    torch.save(payload, destination)
    if rank == 0:
        print(f"gpu-plan wrote {destination}", flush=True)
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
