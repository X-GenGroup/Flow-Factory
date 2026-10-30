"""Compare GPU-plan dumps against the PR checks.

``torch.equal`` is the reproducibility relation. Diversity uses mean pairwise
L2 and unique coordinate counts. Transformer outputs, images, and rewards are
not compared.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from cells import SECTION_DIVERSITY, SECTION_ISOLATION, SECTION_REPRO, require_cell
from keys import (
    CRD_NOISE,
    CRD_PASS2_RANDN,
    CRD_TIMES,
    DPO_NOISE,
    DPO_TIMES,
    EVAL_GENERATOR_SEED,
    EVAL_SEED,
    EVAL_X_T,
    FRESH_NOISE,
    H3_RUNS,
    NFT_NOISE,
    NFT_TIMES,
    NUM_TRAIN_TIMESTEPS,
    PAIRS,
    PLAN_ORDER,
    QUERY,
    REPLAY_LOG_PROB,
    ROW_ORDERS,
    SDE_AFTER,
    SDE_BEFORE,
    SDE_NOISE,
    SHARED_NOISE,
    TERM_NOISE,
    TIME_MATRIX,
    TRAIN_SEED,
    X_T,
)


@dataclass(frozen=True)
class Result:
    name: str
    ok: bool
    detail: str


def mean_pairwise_l2(tensors: list[torch.Tensor]) -> float:
    if len(tensors) < 2:
        raise ValueError(f"pairwise L2 needs at least 2 tensors, received {len(tensors)}")
    flat = [tensor.reshape(-1).float() for tensor in tensors]
    width = flat[0].numel()
    for tensor in flat[1:]:
        if tensor.numel() != width:
            raise ValueError(f"pairwise L2 shape mismatch: {width} vs {tensor.numel()}")
    distances = [
        torch.linalg.vector_norm(flat[left] - flat[right]).item()
        for left in range(len(flat))
        for right in range(left + 1, len(flat))
    ]
    return sum(distances) / len(distances)


def states_equal(left: dict, right: dict) -> bool:
    if set(left) != set(right):
        return False
    return all(torch.equal(left[name], right[name]) for name in left)


def only_scheduler_steps(payload: dict) -> list:
    noise = payload.get(SDE_NOISE)
    if not isinstance(noise, dict) or len(noise) != 1:
        raise ValueError(f"expected one scheduler noise record, received {type(noise).__name__}")
    return next(iter(noise.values()))


def only_value(mapping: dict) -> torch.Tensor:
    if len(mapping) != 1:
        raise ValueError(f"expected one sample row, received {sorted(mapping)}")
    return next(iter(mapping.values()))


def rank_values(ranks: list[dict], key: str) -> list[torch.Tensor]:
    return [only_value(rank[key]) for rank in ranks]


def rank_sde(ranks: list[dict], step_index: int) -> list[torch.Tensor]:
    values = []
    for rank in ranks:
        steps = only_scheduler_steps(rank)
        if step_index >= len(steps):
            raise ValueError(f"rank has {len(steps)} SDE draws, need index {step_index}")
        values.append(only_value(steps[step_index]))
    return values


def primary_sde(payload: dict, when: str) -> tuple[torch.Tensor, int]:
    snap = payload[when]
    primary = snap["primary"]
    entry = snap["by_name"][primary]
    return entry["steps"], int(entry["seed"])


def result(name: str, ok: bool, detail: str) -> Result:
    return Result(name, ok, detail)


def missing(name: str) -> Result:
    return Result(name, False, "缺少 dump，这一格还没有跑完")


def load_launch(root: Path, cell: str, branch: str, launch: int) -> list[dict]:
    directory = root / cell / branch / f"launch{launch}"
    paths = sorted(directory.glob("rank*.pt"))
    if not paths:
        return []
    return [torch.load(path, map_location="cpu", weights_only=False) for path in paths]


def _same_launch(name: str, left: dict, right: dict, key: str) -> Result:
    if key not in left or key not in right:
        return missing(f"{name} {key}")
    return result(name, torch.equal(left[key], right[key]), f"{key} 两次启动逐元素相同")


def check_repro_pair(cell: str, first: list[dict], second: list[dict], world: int) -> list[Result]:
    label = f"{cell} 复现"
    if len(first) != world or len(second) != world:
        return [missing(f"{label} ranks {len(first)}/{len(second)}")]
    checks = []
    for rank, (left, right) in enumerate(zip(first, second)):
        if X_T in left:
            checks.append(
                result(
                    f"{label} rank{rank} x_T",
                    left[X_T] == right[X_T] or _nested_equal(left[X_T], right[X_T]),
                    "x_T 与第二次启动相同",
                )
            )
        if SDE_NOISE in left:
            checks.append(
                result(
                    f"{label} rank{rank} sde",
                    _nested_equal(left[SDE_NOISE], right[SDE_NOISE]),
                    "逐步 SDE 噪声与第二次启动相同",
                )
            )
    return checks


def _nested_equal(left, right) -> bool:
    if torch.is_tensor(left) or torch.is_tensor(right):
        return torch.is_tensor(left) and torch.is_tensor(right) and torch.equal(left, right)
    if isinstance(left, dict) or isinstance(right, dict):
        if not isinstance(left, dict) or not isinstance(right, dict) or set(left) != set(right):
            return False
        return all(_nested_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
            return False
        return len(left) == len(right) and all(_nested_equal(a, b) for a, b in zip(left, right))
    return left == right


def check_grpo_defaults(ranks: list[dict], branch: str) -> list[Result]:
    label = f"grpo_defaults {branch}"
    if len(ranks) != 2:
        return [missing(label)]
    xt = mean_pairwise_l2(rank_values(ranks, X_T))
    sde = mean_pairwise_l2(rank_sde(ranks, 0))
    return [
        result(f"{label} x_T 距离", xt > 0, f"组内 x_T 平均 L2={xt}"),
        result(f"{label} SDE 距离", sde > 0, f"组内 SDE 平均 L2={sde}"),
    ]


def check_grpo_explicit(ranks: list[dict]) -> list[Result]:
    if len(ranks) != 2:
        return [missing("grpo_explicit")]
    xt = rank_values(ranks, X_T)
    step0 = rank_sde(ranks, 0)
    try:
        step1 = rank_sde(ranks, 1)
    except ValueError as exc:
        return [result("grpo_explicit SDE 步数", False, str(exc))]
    return [
        result(
            "grpo_explicit 组内 x_T",
            torch.equal(xt[0], xt[1]),
            "init_same_noise 下两名成员 x_T 相同",
        ),
        result(
            "grpo_explicit 成员 SDE",
            not torch.equal(step0[0], step0[1]),
            "step 0 的 SDE 噪声在两名成员之间不同",
        ),
        result(
            "grpo_explicit 时间步 SDE",
            not torch.equal(step0[0], step1[0]),
            "同一样本的 step 0 和 step 1 不同",
        ),
    ]


def check_row_swap(ranks: list[dict]) -> list[Result]:
    if len(ranks) != 1 or ROW_ORDERS not in ranks[0]:
        return [missing("grpo_rows")]
    orders = ranks[0][ROW_ORDERS]
    if len(orders) != 2:
        return [result("grpo_rows", False, f"需要正反两种行序，收到 {len(orders)}")]
    forward, reversed_order = orders
    same = _nested_equal(forward["sde_noise"], reversed_order["sde_noise"])
    ids = list(forward["sample_ids"])
    first_steps = only_scheduler_steps({"sde_noise": forward["sde_noise"]})
    member_noise = [first_steps[0][ids[0]], first_steps[0][ids[1]]]
    return [
        result("grpo_rows 行交换", same, "交换行后噪声仍留在原来的 sampling_sample_id 上"),
        result(
            "grpo_rows 成员不同",
            not torch.equal(member_noise[0], member_noise[1]),
            "两个 sample id 的 SDE 噪声不同",
        ),
    ]


def check_dgpo_repro(ranks: list[dict]) -> list[Result]:
    if len(ranks) != 2 or SHARED_NOISE not in ranks[0]:
        return [missing("dgpo")]
    left, right = ranks
    t0_match = states_equal(left[SHARED_NOISE][0], right[SHARED_NOISE][0])
    t1_match = states_equal(left[SHARED_NOISE][1], right[SHARED_NOISE][1])
    t_differ = not states_equal(left[SHARED_NOISE][0], left[SHARED_NOISE][1])
    replay = left[REPLAY_LOG_PROB]
    return [
        result("dgpo t0 跨 rank", t0_match, "timestep 0 的共享噪声跨 rank 相同"),
        result("dgpo t1 跨 rank", t1_match, "timestep 1 的共享噪声跨 rank 相同"),
        result("dgpo 跨时间步", t_differ, "timestep 1 与 timestep 0 不同"),
        result(
            "dgpo replay log_prob",
            torch.equal(replay[0], replay[1]),
            "用存下的 next_latents 重放，log_prob 不变",
        ),
    ]


def check_dgpo_diversity(feat: list[dict], main: list[dict]) -> list[Result]:
    checks = []
    for branch, ranks, should_differ in (("feat", feat, True), ("main", main, False)):
        if len(ranks) != 2 or SHARED_NOISE not in ranks[0]:
            checks.append(missing(f"dgpo {branch}"))
            continue
        differ = not states_equal(ranks[0][SHARED_NOISE][0], ranks[0][SHARED_NOISE][1])
        checks.append(
            result(
                f"dgpo {branch} 跨时间步",
                differ is should_differ,
                "feat 必须跨时间步重抽；main 的共享噪声不随时间步变化",
            )
        )
        checks.append(
            result(
                f"dgpo {branch} 组内",
                states_equal(ranks[0][SHARED_NOISE][0], ranks[1][SHARED_NOISE][0]),
                "同一时间步组内仍然相同",
            )
        )
    return checks


def check_nft(ranks: list[dict], *, shared: bool) -> list[Result]:
    label = "nft_shared" if shared else "nft_off"
    if len(ranks) != 2 or NFT_TIMES not in ranks[0]:
        return [missing(label)]
    times = ranks[0][NFT_TIMES]
    width = int(ranks[0][NUM_TRAIN_TIMESTEPS])
    unique = int(torch.unique(times[0]).numel())
    noise_match = states_equal(ranks[0][NFT_NOISE][0][0], ranks[1][NFT_NOISE][0][0])
    checks = [
        result(f"{label} 坐标宽度", unique == width, f"unique={unique}, stratified width={width}"),
        result(
            f"{label} batch index",
            not torch.equal(times[0], times[1]),
            "batch index 1 的时间坐标与 index 0 不同",
        ),
        result(
            f"{label} 跨 rank 时间",
            torch.equal(ranks[0][NFT_TIMES][0], ranks[1][NFT_TIMES][0]),
            "同一 batch index 的时间坐标跨 rank 相同",
        ),
        result(
            f"{label} 噪声共享",
            noise_match if shared else not noise_match,
            "use_shared_noise 打开时组内噪声相同，关闭时逐样本不同",
        ),
        result(
            f"{label} 噪声时间步",
            not states_equal(ranks[0][NFT_NOISE][0][0], ranks[0][NFT_NOISE][0][1]),
            "同一条样本的两个训练时间步噪声不同",
        ),
        result(
            f"{label} rollout x_T",
            mean_pairwise_l2(rank_values(ranks, X_T)) > 0,
            "两条 rollout x_T 不同",
        ),
    ]
    return checks


def check_h3(ranks: list[dict], *, require_sde_stable: bool) -> list[Result]:
    if not ranks or H3_RUNS not in ranks[0] or len(ranks[0][H3_RUNS]) != 2:
        return [missing("h3")]
    first, second = ranks[0][H3_RUNS]
    xt_changed = not _nested_equal(first["x_t"], second["x_t"])
    sde_stable = _nested_equal(first["sde_noise"], second["sde_noise"])
    names = list(first["sde_noise"])
    component_differ = False
    if len(names) >= 2 and first["sde_noise"][names[0]] and first["sde_noise"][names[1]]:
        left = only_value(first["sde_noise"][names[0]][0])
        right = only_value(first["sde_noise"][names[1]][0])
        component_differ = not torch.equal(left, right)
    checks = [
        result("h3 generator 只改 x_T", xt_changed, "更换 request generator 后 x_T 改变"),
        result(
            "h3 视频与音频",
            component_differ,
            f"scheduler 分量 {names} 的第一步噪声不同",
        ),
    ]
    if require_sde_stable:
        checks.append(
            result(
                "h3 SDE 不吃 request generator",
                sde_stable,
                "重设进程种子后 SDE 噪声不随 generator 改变",
            )
        )
    else:
        checks.append(
            result(
                "h3 main SDE 记录",
                True,
                f"main 上更换 generator 后 SDE {'不变' if sde_stable else '改变'}",
            )
        )
    return checks


def _matrix_not_constant(matrix: torch.Tensor) -> bool:
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
        return False
    rows = all(torch.equal(matrix[0], matrix[row]) for row in range(1, matrix.shape[0]))
    cols = all(torch.equal(matrix[:, 0], matrix[:, col]) for col in range(1, matrix.shape[1]))
    return not rows and not cols


def check_sft(ranks: list[dict]) -> list[Result]:
    if len(ranks) != 1 or TIME_MATRIX not in ranks[0]:
        return [missing("sft")]
    payload = ranks[0]
    matrix = payload[TIME_MATRIX]
    term0 = [record[0] for record in payload[TERM_NOISE]]
    return [
        result(
            "sft 时间矩阵", _matrix_not_constant(matrix), f"time matrix shape={tuple(matrix.shape)}"
        ),
        result(
            "sft 两条记录",
            not states_equal(term0[0], term0[1]),
            "record A 与 record B 在 term 0 的噪声不同",
        ),
        result(
            "sft 两个 term",
            not states_equal(payload[TERM_NOISE][0][0], payload[TERM_NOISE][0][1]),
            "term 1 与 term 0 不同",
        ),
    ]


def check_offline_dpo(ranks: list[dict]) -> list[Result]:
    if len(ranks) != 1 or PAIRS not in ranks[0] or len(ranks[0][PAIRS]) != 2:
        return [missing("offline_dpo")]
    pairs = ranks[0][PAIRS]
    shared = all(
        states_equal(term["chosen"], term["rejected"]) for pair in pairs for term in pair["terms"]
    )
    terms_differ = not states_equal(pairs[0]["terms"][0]["chosen"], pairs[0]["terms"][1]["chosen"])
    records_differ = not states_equal(
        pairs[0]["terms"][0]["chosen"], pairs[1]["terms"][0]["chosen"]
    )
    return [
        result("offline_dpo 配对共享", shared, "chosen 与 rejected 在每个 term 上同一张噪声"),
        result("offline_dpo 两个 term", terms_differ, "两个 term 的噪声不同"),
        result("offline_dpo record_id", records_differ, "另一个 record_id 画出不同的噪声"),
    ]


def check_online_dpo(ranks: list[dict]) -> list[Result]:
    if len(ranks) != 2 or DPO_NOISE not in ranks[0]:
        return [missing("online_dpo")]
    shared = all(states_equal(term["chosen"], term["rejected"]) for term in ranks[0][DPO_NOISE])
    terms_differ = not states_equal(
        ranks[0][DPO_NOISE][0]["chosen"], ranks[0][DPO_NOISE][1]["chosen"]
    )
    ranks_differ = not states_equal(
        ranks[0][DPO_NOISE][0]["chosen"], ranks[1][DPO_NOISE][0]["chosen"]
    )
    times = ranks[0][DPO_TIMES]
    return [
        result("online_dpo 配对共享", shared, "pair 内共享噪声"),
        result("online_dpo 时间步", terms_differ, "两个训练时间步的噪声不同"),
        result("online_dpo 另一个 rank", ranks_differ, "另一个 rank 的 sample id 画出不同的噪声"),
        result(
            "online_dpo 坐标",
            int(torch.unique(times).numel()) > 1,
            "一次抽样里的训练坐标不全相等",
        ),
    ]


def check_crd(ranks: list[dict]) -> list[Result]:
    if len(ranks) != 2 or CRD_NOISE not in ranks[0]:
        return [missing("crd")]
    differ_rank = not states_equal(ranks[0][CRD_NOISE][0], ranks[1][CRD_NOISE][0])
    differ_time = not states_equal(ranks[0][CRD_NOISE][0], ranks[0][CRD_NOISE][1])
    return [
        result("crd 时间步", differ_time, "timestep 1 与 timestep 0 不同"),
        result("crd sample id", differ_rank, "不同 sampling_sample_id 的噪声不同"),
        result(
            "crd pass 2",
            int(ranks[0][CRD_PASS2_RANDN]) == 0 and int(ranks[1][CRD_PASS2_RANDN]) == 0,
            "pass 2 复用已存噪声，不再 randn",
        ),
    ]


def check_tdm(ranks: list[dict], *, feat: bool) -> list[Result]:
    if not ranks or QUERY not in ranks[0] or len(ranks[0][QUERY]) != 4:
        return [missing("tdm")]
    query_one, query_two, query_restart, query_sigma = ranks[0][QUERY]
    fresh = ranks[0][FRESH_NOISE]
    checks = [
        result(
            "tdm 重复抽样", not torch.equal(query_one, query_two), "同一 boundary 的两次 query 不同"
        ),
        result(
            "tdm fresh 重复",
            not _nested_equal(fresh[0], fresh[1]),
            "下一次 fresh noise 是新张量",
        ),
    ]
    if query_one.numel() >= 2:
        checks.append(
            result(
                "tdm batch 内 query",
                not bool(torch.all(query_one.reshape(-1) == query_one.reshape(-1)[0]).item()),
                "batch 内的 query 值不全相等",
            )
        )
    if len(fresh[0]) >= 2:
        checks.append(
            result(
                "tdm 分量 fresh",
                not torch.equal(fresh[0][0], fresh[0][1]),
                "video 与 audio 的 fresh noise 不同",
            )
        )
    if feat:
        checks.append(
            result("tdm 重置", torch.equal(query_one, query_restart), "重置计数器后 query 序列复现")
        )
        checks.append(
            result(
                "tdm max_sigma",
                torch.equal(query_one, query_sigma),
                "trajectory interval 下 tdm_query_max_sigma 不改变这次抽样",
            )
        )
        checks.append(
            result(
                "tdm fresh 重置", _nested_equal(fresh[0], fresh[2]), "重置后 fresh noise 序列复现"
            )
        )
    return checks


def check_seed_scheduler(baseline: list[dict], variant: list[dict]) -> list[Result]:
    if not baseline or not variant:
        return [missing("seed_scheduler")]
    before_steps, before_seed = primary_sde(baseline[0], SDE_BEFORE)
    variant_steps, variant_seed = primary_sde(variant[0], SDE_BEFORE)
    after_steps, after_seed = primary_sde(baseline[0], SDE_AFTER)
    variant_after_steps, variant_after_seed = primary_sde(variant[0], SDE_AFTER)
    train_seed = int(baseline[0][TRAIN_SEED])
    return [
        result(
            "scheduler.seed 生效前",
            before_seed == int(baseline[0]["scheduler_seed"])
            and variant_seed == int(variant[0]["scheduler_seed"])
            and before_seed != variant_seed,
            f"prepare_cycle 前 scheduler.seed 分别为 {before_seed} 和 {variant_seed}",
        ),
        result(
            "scheduler.seed 生效后",
            after_seed == train_seed and variant_after_seed == train_seed,
            f"prepare_cycle 后运行时种子是 train.seed + rollout_iteration = {after_seed}",
        ),
        result(
            "scheduler 子集记录",
            torch.equal(after_steps, variant_after_steps),
            (
                "前后子集"
                f"{'不同' if not torch.equal(before_steps, variant_steps) else '碰巧相同'}"
                "；生效后的子集相同"
            ),
        ),
    ]


def check_seed_eval(baseline: list[dict], variant: list[dict]) -> list[Result]:
    if not baseline or not variant or EVAL_X_T not in baseline[0] or EVAL_X_T not in variant[0]:
        return [missing("seed_eval")]
    same_xt = torch.equal(baseline[0][EVAL_X_T], variant[0][EVAL_X_T])
    uses_train = int(baseline[0][EVAL_GENERATOR_SEED]) == int(baseline[0][TRAIN_SEED]) and int(
        variant[0][EVAL_GENERATOR_SEED]
    ) == int(variant[0][TRAIN_SEED])
    changed_eval = int(baseline[0][EVAL_SEED]) != int(variant[0][EVAL_SEED])
    return [
        result(
            "eval.seed 缺口",
            same_xt and uses_train and changed_eval,
            "只改 eval.seed 时 eval x_T 不变；当前 generator 用的是 train.seed",
        )
    ]


def _launch_equal(
    cell: str, first: list[dict], second: list[dict], keys: tuple[str, ...]
) -> list[Result]:
    if not first or not second or len(first) != len(second):
        return [missing(f"{cell} 第二次启动")]
    checks = []
    for rank, (left, right) in enumerate(zip(first, second)):
        for key in keys:
            checks.append(
                result(
                    f"{cell} rank{rank} {key}",
                    key in left and key in right and _nested_equal(left[key], right[key]),
                    f"{key} 与第二次启动相同",
                )
            )
    return checks


def positive_distance_survives(name: str, feat: float, main: float) -> Result:
    ok = main <= 0 or feat > 0
    return result(name, ok, f"main L2={main}, feat L2={feat}")


def _selected(name: str, included: set[str] | None) -> bool:
    return included is None or name in included


def evaluate(root: Path, section: str, included: set[str] | None = None) -> list[Result]:
    """Evaluate one PR section from dumps under ``root``.

    ``included`` limits the report to cells this invocation launched. Missing
    dumps outside that set are not failures.
    """

    if section == SECTION_REPRO:
        return _evaluate_repro(root, included)
    if section == SECTION_DIVERSITY:
        return _evaluate_diversity(root, included)
    if section == SECTION_ISOLATION:
        return _evaluate_isolation(root, included)
    raise ValueError(f"unknown section {section!r}")


def _evaluate_repro(root: Path, included: set[str] | None) -> list[Result]:
    checks = []
    specs = {
        "grpo_defaults": ("x_t", "sde_noise"),
        "grpo_explicit": ("x_t", "sde_noise"),
        "grpo_rows": ("row_orders",),
        "dgpo": ("shared_noise", "replay_log_prob", "x_t", "sde_noise"),
        "nft_off": ("nft_times", "nft_noise", "x_t"),
        "nft_shared": ("nft_times", "nft_noise", "x_t"),
        "h3": ("h3_runs",),
        "sft": ("time_matrix", "term_noise"),
        "offline_dpo": ("pairs",),
        "online_dpo": ("dpo_times", "dpo_noise"),
        "crd": ("crd_times", "crd_noise", "crd_pass2_randn"),
        "tdm": ("query", "fresh_noise"),
        "tdm_h3": ("query", "fresh_noise"),
    }
    for name, keys in specs.items():
        if not _selected(name, included):
            continue
        cell = require_cell(name)
        first = load_launch(root, name, "feat", 0)
        second = load_launch(root, name, "feat", 1)
        checks.extend(_launch_equal(name, first, second, keys))
        if name == "grpo_explicit" and first:
            checks.extend(check_grpo_explicit(first))
        elif name == "grpo_rows" and first:
            checks.extend(check_row_swap(first))
        elif name == "dgpo" and first:
            checks.extend(check_dgpo_repro(first))
        elif name == "nft_off" and first:
            checks.extend(check_nft(first, shared=False))
        elif name == "nft_shared" and first:
            checks.extend(check_nft(first, shared=True))
        elif name == "h3" and first:
            checks.extend(check_h3(first, require_sde_stable=True))
        elif name == "sft" and first:
            checks.extend(check_sft(first))
        elif name == "offline_dpo" and first:
            checks.extend(check_offline_dpo(first))
        elif name == "online_dpo" and first:
            checks.extend(check_online_dpo(first))
        elif name == "crd" and first:
            checks.extend(check_crd(first))
        elif name in {"tdm", "tdm_h3"} and first:
            checks.extend(check_tdm(first, feat=True))
        del cell
    return checks


def _evaluate_diversity(root: Path, included: set[str] | None) -> list[Result]:
    checks = []
    for name in ("grpo_defaults",):
        if not _selected(name, included):
            continue
        feat = load_launch(root, name, "feat", 0)
        main = load_launch(root, name, "main", 0)
        if not feat or not main:
            checks.append(missing(f"{name} diversity"))
            continue
        feat_stats = check_grpo_defaults(feat, "feat")
        main_stats = check_grpo_defaults(main, "main")
        checks.extend(feat_stats)
        checks.extend(main_stats)
        checks.append(
            positive_distance_survives(
                "grpo_defaults x_T 没有被压扁",
                mean_pairwise_l2(rank_values(feat, X_T)),
                mean_pairwise_l2(rank_values(main, X_T)),
            )
        )
    if _selected("grpo_explicit", included):
        feat_explicit = load_launch(root, "grpo_explicit", "feat", 0)
        main_explicit = load_launch(root, "grpo_explicit", "main", 0)
        if feat_explicit:
            checks.extend(check_grpo_explicit(feat_explicit))
        else:
            checks.append(missing("grpo_explicit diversity"))
        if main_explicit:
            distance = mean_pairwise_l2(rank_values(main_explicit, X_T))
            checks.append(
                result("main 默认 x_T", distance > 0, f"main 没有 init_same_noise，L2={distance}")
            )
    if _selected("dgpo", included):
        checks.extend(
            check_dgpo_diversity(
                load_launch(root, "dgpo", "feat", 0),
                load_launch(root, "dgpo", "main", 0),
            )
        )
    for name, shared in (("nft_off", False), ("nft_shared", True)):
        if not _selected(name, included):
            continue
        feat = load_launch(root, name, "feat", 0)
        main = load_launch(root, name, "main", 0)
        if not feat or not main:
            checks.append(missing(f"{name} diversity"))
            continue
        checks.extend(check_nft(feat, shared=shared))
        feat_unique = int(torch.unique(feat[0][NFT_TIMES][0]).numel())
        main_unique = int(torch.unique(main[0][NFT_TIMES][0]).numel())
        checks.append(
            result(
                f"{name} unique 不少于 main",
                feat_unique >= main_unique,
                f"feat unique={feat_unique}, main unique={main_unique}",
            )
        )
    if _selected("h3", included):
        feat_h3 = load_launch(root, "h3", "feat", 0)
        main_h3 = load_launch(root, "h3", "main", 0)
        if feat_h3:
            checks.extend(check_h3(feat_h3, require_sde_stable=True))
        else:
            checks.append(missing("h3 feat diversity"))
        if main_h3:
            checks.extend(check_h3(main_h3, require_sde_stable=False))
        else:
            checks.append(missing("h3 main diversity"))
    for name, checker in (
        ("sft", check_sft),
        ("offline_dpo", check_offline_dpo),
        ("online_dpo", check_online_dpo),
        ("crd", check_crd),
    ):
        if not _selected(name, included):
            continue
        feat = load_launch(root, name, "feat", 0)
        main = load_launch(root, name, "main", 0)
        if not feat or not main:
            checks.append(missing(f"{name} diversity"))
            continue
        checks.extend(checker(feat))
        if name == "sft":
            checks.append(
                result(
                    "sft unique 不少于 main",
                    int(torch.unique(feat[0][TIME_MATRIX]).numel())
                    >= int(torch.unique(main[0][TIME_MATRIX]).numel()),
                    "feat 的时间矩阵 unique-count 不低于 main",
                )
            )
        if name in {"online_dpo", "crd"}:
            feat_unique = int(
                torch.unique(feat[0][DPO_TIMES if name == "online_dpo" else CRD_TIMES]).numel()
            )
            main_unique = int(
                torch.unique(main[0][DPO_TIMES if name == "online_dpo" else CRD_TIMES]).numel()
            )
            checks.append(
                result(
                    f"{name} 坐标宽度",
                    feat_unique >= main_unique,
                    f"feat unique={feat_unique}, main unique={main_unique}",
                )
            )
    for name in ("tdm", "tdm_h3"):
        if not _selected(name, included):
            continue
        feat = load_launch(root, name, "feat", 0)
        main = load_launch(root, name, "main", 0)
        if not feat or not main:
            checks.append(missing(f"{name} diversity"))
            continue
        checks.extend(check_tdm(feat, feat=True))
        checks.extend(check_tdm(main, feat=False))
    return checks


def _evaluate_isolation(root: Path, included: set[str] | None) -> list[Result]:
    checks = []
    if _selected("seed_train_explicit", included):
        explicit_base = load_launch(root, "grpo_explicit", "feat", 0)
        explicit_var = load_launch(root, "seed_train_explicit", "feat", 0)
        if not explicit_base or not explicit_var:
            checks.append(missing("seed_train_explicit"))
        else:
            checks.extend(
                [
                    result(
                        "train.seed x_T",
                        not _nested_equal(explicit_base[0][X_T], explicit_var[0][X_T]),
                        "只改 train.seed 后显式 generator 的 x_T 改变",
                    ),
                    result(
                        "train.seed 采样顺序",
                        explicit_base[0][PLAN_ORDER] != explicit_var[0][PLAN_ORDER],
                        "采样计划的数据集顺序改变",
                    ),
                ]
            )
    if _selected("seed_train_nft", included):
        nft_base = load_launch(root, "nft_shared", "feat", 0)
        nft_var = load_launch(root, "seed_train_nft", "feat", 0)
        if not nft_base or not nft_var:
            checks.append(missing("seed_train_nft"))
        else:
            checks.append(
                result(
                    "train.seed NFT 坐标",
                    not torch.equal(nft_base[0][NFT_TIMES][0], nft_var[0][NFT_TIMES][0]),
                    "NFT/AWM 的训练坐标改变",
                )
            )
    if _selected("seed_scheduler", included):
        checks.extend(
            check_seed_scheduler(
                load_launch(root, "grpo_defaults", "feat", 0),
                load_launch(root, "seed_scheduler", "feat", 0),
            )
        )
    if _selected("seed_eval", included):
        checks.extend(
            check_seed_eval(
                load_launch(root, "grpo_defaults", "feat", 0),
                load_launch(root, "seed_eval", "feat", 0),
            )
        )
    return checks


def format_results(results: list[Result]) -> str:
    lines = []
    for item in results:
        status = "通过" if item.ok else "失败"
        lines.append(f"{status}  {item.name}  {item.detail}")
    passed = sum(item.ok for item in results)
    lines.append(f"合计 {passed}/{len(results)}")
    return "\n".join(lines)


def self_test() -> None:
    """Check the predicates on CPU tensors. This does not launch a model."""

    image = torch.arange(4, dtype=torch.float32)
    other = image + 1
    shared = {"latent": image.clone()}
    shared_other = {"latent": other.clone()}
    ranks = [
        {
            X_T: {"0": image},
            SDE_NOISE: {"scheduler": [{"0": image}, {"0": other}]},
            SHARED_NOISE: [shared, shared_other],
            REPLAY_LOG_PROB: [image, image.clone()],
        },
        {
            X_T: {"1": other},
            SDE_NOISE: {"scheduler": [{"1": other}, {"1": image}]},
            SHARED_NOISE: [shared, shared_other],
            REPLAY_LOG_PROB: [image, image.clone()],
        },
    ]
    defaults = check_grpo_defaults(ranks, "feat")
    if not all(item.ok for item in defaults):
        raise AssertionError(defaults)
    explicit = [
        {X_T: {"0": image}, SDE_NOISE: {"scheduler": [{"0": image}, {"0": other}]}},
        {X_T: {"1": image.clone()}, SDE_NOISE: {"scheduler": [{"1": other}, {"1": image}]}},
    ]
    if not all(item.ok for item in check_grpo_explicit(explicit)):
        raise AssertionError(check_grpo_explicit(explicit))
    collapsed = [
        {X_T: {"0": image}, SDE_NOISE: {"scheduler": [{"0": image}, {"0": image}]}},
        {X_T: {"1": image.clone()}, SDE_NOISE: {"scheduler": [{"1": image.clone()}, {"1": other}]}},
    ]
    if any(item.ok and "时间步" in item.name for item in check_grpo_explicit(collapsed)):
        raise AssertionError("identical SDE steps must fail the step check")
    matrix = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    sft_rank = [
        {
            TIME_MATRIX: matrix,
            TERM_NOISE: [[shared, shared_other], [shared_other, shared]],
        }
    ]
    if not all(item.ok for item in check_sft(sft_rank)):
        raise AssertionError(check_sft(sft_rank))
    query = torch.tensor([0.2, 0.8])
    fresh = [[image, other], [other, image], [image.clone(), other.clone()]]
    tdm_rank = [{QUERY: [query, query + 1, query.clone(), query.clone()], FRESH_NOISE: fresh}]
    if not all(item.ok for item in check_tdm(tdm_rank, feat=True)):
        raise AssertionError(check_tdm(tdm_rank, feat=True))
    print("gpu-plan compare self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser(description="比较 GPU plan 的 dump。")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--section", choices=(SECTION_REPRO, SECTION_DIVERSITY, SECTION_ISOLATION))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.root is None or args.section is None:
        raise SystemExit("需要 --root 和 --section，或者 --self-test")
    results = evaluate(args.root, args.section)
    print(format_results(results))
    if not results or not all(item.ok for item in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
