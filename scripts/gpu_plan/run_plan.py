"""Launch the PR #267 GPU plan.

默认只打印格子，不占 GPU。当前机器上如果已有训练，脚本会拒绝启动。
SD3.5 和 MiniMax-H3 都在第一次真正启动时从 Hugging Face 拉取。

示例：
  python scripts/gpu_plan/run_plan.py --check-configs
  python scripts/gpu_plan/run_plan.py --section reproducibility --cell grpo_defaults
  python scripts/gpu_plan/run_plan.py --section all
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import yaml
from cells import (
    CELLS,
    SECTION_DIVERSITY,
    SECTION_ISOLATION,
    SECTION_REPRO,
    Cell,
    Seeds,
    render_config,
    require_cell,
)

REPO = Path("/root/Flow-Factory")
MAIN_WORKTREE = Path("/root/Flow-Factory-main")
FF_PYTHON = Path("/opt/conda/envs/ff/bin/python")
PROBE = REPO / "scripts/gpu_plan/probe.py"
DEFAULT_ROOT = REPO / ".scratch" / "gpu_plan"
SECTIONS = (SECTION_REPRO, SECTION_ISOLATION, SECTION_DIVERSITY, "sd3", "h3", "all")


@dataclass(frozen=True)
class Job:
    """One process launch. ``name`` is the output directory, ``cell`` selects the probe."""

    name: str
    cell: str
    branch: str
    launch: int
    seeds: Seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按 PR #267 的 GPU test plan 启动独立进程。默认不启动。",
    )
    parser.add_argument("--list", action="store_true", help="打印格子后退出")
    parser.add_argument("--check-configs", action="store_true", help="只校验 YAML，不加载模型")
    parser.add_argument("--section", choices=SECTIONS)
    parser.add_argument("--cell", help="只跑这一个格子，或隔离变体的目录名")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--devices", default=os.environ.get("FF_GPU_PLAN_DEVICES", "0,1"))
    parser.add_argument("--port", type=int, default=29621)
    parser.add_argument("--fresh", action="store_true", help="丢掉已有 dump，重新启动")
    parser.add_argument(
        "--force",
        action="store_true",
        help="目标 GPU 仍有显存占用时也启动。不要在当前训练还在跑时使用。",
    )
    return parser.parse_args()


def reexec_into_ff() -> None:
    if Path(sys.executable).resolve() == FF_PYTHON.resolve():
        return
    if not FF_PYTHON.is_file():
        raise FileNotFoundError(f"ff python is missing: {FF_PYTHON}")
    os.execv(str(FF_PYTHON), [str(FF_PYTHON), *sys.argv])


def print_list() -> None:
    print("PR #267 GPU plan 格子。每一格是独立进程，不是同进程里再抽一次噪声。")
    print("复现：feat 启动两次，torch.equal。")
    print("隔离：只改 train.seed / scheduler.seed / eval.seed 中的一个。")
    print("多样性：feat 和 origin/main 各启动一次。")
    print("H3 格子会拉取 MiniMaxAI/MiniMax-H3，并使用 124 帧。")
    for cell in CELLS:
        sections = ",".join(cell.sections)
        print(f"  {cell.name:16}  world={cell.world_size}  {sections:28}  {cell.checklist}")
    print("  seed_train_explicit  只把 grpo_explicit 的 train.seed 改成 43")
    print("  seed_train_nft       只把 nft_shared 的 train.seed 改成 43")
    print("  seed_scheduler       只把 grpo_defaults 的 scheduler.seed 改成 8")
    print("  seed_eval            只把 grpo_defaults 的 eval.seed 改成 12")


def _write_yaml(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(document, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def check_configs() -> None:
    """Load every cell's YAML with the world size it will be launched under."""

    from flow_factory.data_utils.sampling_plan import build_sampling_plan
    from flow_factory.hparams import Arguments

    os.chdir(REPO)

    def plan_order(seed: int) -> list[int]:
        plan = build_sampling_plan(
            layout_name="global_batch",
            dataset_size=8,
            per_device_batch_size=1,
            group_size=2,
            unique_group_count=8,
            num_replicas=2,
            seed=seed,
            epoch=0,
        )
        return [int(item) for batch in plan.batches_for_rank(0) for item in batch]

    baseline = plan_order(42)
    if baseline == plan_order(43):
        raise RuntimeError(
            "train.seed 42 and 43 build the same sampling plan; change the isolation seed"
        )

    with tempfile.TemporaryDirectory(prefix="gpu-plan-config-") as tmp:
        root = Path(tmp)
        for cell in CELLS:
            path = root / f"{cell.name}.yaml"
            _write_yaml(path, render_config(cell, Seeds(), f"check-{cell.name}"))
            os.environ["WORLD_SIZE"] = str(cell.world_size)
            Arguments.load_from_yaml(str(path))
            print(f"配置通过  {cell.name}  world={cell.world_size}")
    print("采样计划在 train.seed 42 和 43 下不同。")


def jobs_for(section: str, cell_name: str | None) -> list[Job]:
    selected = []
    cells = CELLS
    if section == "sd3":
        cells = tuple(cell for cell in CELLS if cell.kind not in {"h3", "tdm_h3"})
        sections = (SECTION_REPRO, SECTION_ISOLATION, SECTION_DIVERSITY)
    elif section == "h3":
        cells = tuple(cell for cell in CELLS if cell.kind in {"h3", "tdm_h3"})
        sections = (SECTION_REPRO, SECTION_DIVERSITY)
    elif section == "all":
        sections = (SECTION_REPRO, SECTION_ISOLATION, SECTION_DIVERSITY)
    else:
        sections = (section,)

    def wanted(name: str, cell: str) -> bool:
        return cell_name is None or cell_name in {name, cell}

    for item in sections:
        if item == SECTION_REPRO:
            for cell in cells:
                if SECTION_REPRO not in cell.sections or not wanted(cell.name, cell.name):
                    continue
                for launch in (0, 1):
                    selected.append(Job(cell.name, cell.name, "feat", launch, Seeds()))
        elif item == SECTION_DIVERSITY:
            for cell in cells:
                if SECTION_DIVERSITY not in cell.sections or not wanted(cell.name, cell.name):
                    continue
                selected.append(Job(cell.name, cell.name, "feat", 0, Seeds()))
                selected.append(Job(cell.name, cell.name, "main", 0, Seeds()))
        elif item == SECTION_ISOLATION:
            variants = (
                Job("seed_train_explicit", "grpo_explicit", "feat", 0, Seeds(train=43)),
                Job("seed_train_nft", "nft_shared", "feat", 0, Seeds(train=43)),
                Job("seed_scheduler", "grpo_defaults", "feat", 0, Seeds(scheduler=8)),
                Job("seed_eval", "grpo_defaults", "feat", 0, Seeds(eval=12)),
            )
            selected.extend(job for job in variants if wanted(job.name, job.cell))
            baselines = (
                Job("grpo_explicit", "grpo_explicit", "feat", 0, Seeds()),
                Job("nft_shared", "nft_shared", "feat", 0, Seeds()),
                Job("grpo_defaults", "grpo_defaults", "feat", 0, Seeds()),
            )
            for baseline in baselines:
                if wanted(baseline.name, baseline.cell) or any(
                    wanted(job.name, job.cell) and job.cell == baseline.cell for job in variants
                ):
                    selected.append(baseline)
    # A cell that appears in two sections must still launch once per branch/launch/seed.
    unique = []
    seen = set()
    for job in selected:
        key = (job.name, job.branch, job.launch, job.seeds)
        if key in seen:
            continue
        seen.add(key)
        unique.append(job)
    return unique


def ensure_main_worktree() -> Path:
    if (MAIN_WORKTREE / "src" / "flow_factory").is_dir():
        return MAIN_WORKTREE
    if MAIN_WORKTREE.exists():
        raise RuntimeError(f"{MAIN_WORKTREE} exists but is not a Flow-Factory checkout")
    subprocess.run(
        ["git", "worktree", "add", str(MAIN_WORKTREE), "origin/main"],
        cwd=REPO,
        check=True,
    )
    return MAIN_WORKTREE


def worktree_for(branch: str) -> Path:
    if branch == "feat":
        return REPO
    if branch == "main":
        return ensure_main_worktree()
    raise ValueError(f"unknown branch {branch!r}")


def device_list(spec: str, world: int) -> list[str]:
    devices = [item.strip() for item in spec.split(",") if item.strip()]
    if len(devices) < world:
        raise RuntimeError(f"cell needs {world} devices from {spec!r}, received {devices}")
    return devices[:world]


def assert_gpus_idle(devices: list[str], force: bool) -> None:
    if force:
        return
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    used = {}
    for line in completed.stdout.splitlines():
        index, memory = [item.strip() for item in line.split(",")]
        used[index] = int(memory)
    busy = [f"GPU {index} 已用 {used[index]} MiB" for index in devices if used.get(index, 0) > 2048]
    if busy:
        raise RuntimeError(
            "目标 GPU 不是空闲的，gpu plan 没有启动："
            + "；".join(busy)
            + "。先停掉当前训练。只有确认可以抢占时才加 --force。"
        )


def launch_complete(path: Path, world: int) -> bool:
    return all((path / f"rank{rank}.pt").is_file() for rank in range(world))


def launch_job(job: Job, root: Path, devices: str, port: int, fresh: bool, force: bool) -> None:
    cell = require_cell(job.cell)
    launch_dir = root / job.name / job.branch / f"launch{job.launch}"
    if fresh and launch_dir.exists():
        shutil.rmtree(launch_dir)
    if launch_complete(launch_dir, cell.world_size):
        print(f"已有 dump，跳过  {job.name} {job.branch} launch{job.launch}")
        return
    chosen = device_list(devices, cell.world_size)
    assert_gpus_idle(chosen, force)
    launch_dir.mkdir(parents=True, exist_ok=True)
    config_path = launch_dir / "config.yaml"
    _write_yaml(
        config_path,
        render_config(cell, job.seeds, f"gpu-plan-{job.name}-{job.branch}-{job.launch}"),
    )
    worktree = worktree_for(job.branch)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(worktree / "src")
    env["CUDA_VISIBLE_DEVICES"] = ",".join(chosen)
    env["PYTHONUNBUFFERED"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["WORLD_SIZE"] = str(cell.world_size)
    if cell.accelerate_config:
        command = [
            str(FF_PYTHON),
            "-m",
            "accelerate.commands.launch",
            "--config_file",
            str(REPO / cell.accelerate_config),
            "--num_processes",
            str(cell.world_size),
            "--main_process_port",
            str(port),
            "--mixed_precision",
            "bf16",
            str(PROBE),
        ]
    else:
        command = [
            str(FF_PYTHON),
            "-m",
            "torch.distributed.run",
            "--nproc_per_node",
            str(cell.world_size),
            "--master_port",
            str(port),
            str(PROBE),
        ]
    command += [
        "--cell",
        cell.name,
        "--config",
        str(config_path),
        "--out",
        str(launch_dir),
        "--branch",
        job.branch,
        "--launch",
        str(job.launch),
    ]
    print(
        f"启动  {job.name} {job.branch} launch{job.launch}  "
        f"devices={env['CUDA_VISIBLE_DEVICES']}  port={port}",
        flush=True,
    )
    subprocess.run(command, cwd=worktree, env=env, check=True)
    if not launch_complete(launch_dir, cell.world_size):
        raise RuntimeError(f"launch finished without rank dumps in {launch_dir}")


def sections_to_report(section: str) -> tuple[str, ...]:
    if section == "all":
        return (SECTION_REPRO, SECTION_ISOLATION, SECTION_DIVERSITY)
    if section == "sd3":
        return (SECTION_REPRO, SECTION_ISOLATION, SECTION_DIVERSITY)
    if section == "h3":
        return (SECTION_REPRO, SECTION_DIVERSITY)
    return (section,)


def main() -> None:
    args = parse_args()
    if args.list or (args.section is None and not args.check_configs):
        print_list()
        if args.section is None and not args.check_configs:
            print("没有 --section，未启动 GPU。")
            return
    if args.check_configs:
        reexec_into_ff()
        check_configs()
        if args.section is None:
            return
    if args.section is None:
        return
    reexec_into_ff()
    planned = jobs_for(args.section, args.cell)
    if not planned:
        raise RuntimeError(f"section {args.section!r} cell {args.cell!r} 没有要启动的格子")
    port = args.port
    for job in planned:
        launch_job(job, args.root, args.devices, port, args.fresh, args.force)
        port += 1
    from compare import evaluate, format_results

    included = {job.name for job in planned}
    for section in sections_to_report(args.section):
        results = evaluate(args.root, section, included)
        print(f"\n{section}")
        if not results:
            print("这一节没有选中的格子")
            continue
        print(format_results(results))
        if not all(item.ok for item in results):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
