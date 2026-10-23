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

import argparse
import logging
import os
import signal
import subprocess

# src/flow_factory/cli.py
import sys

import torch
import yaml

from .utils.env_utils import ENV_VAR_MAPPINGS, env_lookup

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"
)
logger = logging.getLogger(__name__)


def resolve_multinode_env() -> dict:
    """Detect multi-node configuration from environment variables.

    Returns:
        Only the successfully resolved launch fields; empty when not in a multi-node environment
        (``master_ip`` or ``num_machines`` unset, or ``num_machines <= 1``).
    """
    master_ip = env_lookup("master_ip")
    num_machines_str = env_lookup("num_machines")

    # Consider it a multi-node environment only when both master_ip and num_machines > 1 are present
    if not master_ip or not num_machines_str:
        return {}

    num_machines = int(num_machines_str)
    if num_machines <= 1:
        return {}

    env_config = {
        "main_process_ip": master_ip,
        "num_machines": num_machines,
    }

    # Optional: machine_rank
    rank_str = env_lookup("machine_rank")
    if rank_str is not None:
        env_config["machine_rank"] = int(rank_str)

    # Optional: master_port
    port_str = env_lookup("master_port")
    if port_str is not None:
        env_config["main_process_port"] = int(port_str)

    # Optional: gpus_per_node -> used to compute num_processes
    gpus_str = env_lookup("gpus_per_node")
    if gpus_str is not None:
        env_config["num_processes"] = num_machines * int(gpus_str)

    return env_config


def get_gpu_count():
    """Detect available accelerator count (CUDA or Ascend NPU).

    torch.cuda.device_count() returns 0 on NPU pods unless torch_npu is
    imported (the launcher process does not import it itself), which made
    ff-train silently fall back to single-process mode on multi-NPU nodes.
    """
    # Prefer the platform-provided count (QZ exports GPUS_PER_NODE in the job
    # payload); runtime probing can fail on nodes with a broken NPU driver and
    # silently degrade multi-NPU jobs to single-process direct launch.
    _env_count = os.environ.get("GPUS_PER_NODE", "")
    if _env_count.isdigit() and int(_env_count) > 0:
        return int(_env_count)
    try:
        if torch.cuda.is_available():
            return torch.cuda.device_count()
    except Exception:
        pass
    try:
        import torch_npu  # noqa: F401

        if torch.npu.is_available():
            return torch.npu.device_count()
    except Exception:
        pass
    return 0


def parse_args():
    """Parse command line arguments with optional multi-node overrides."""
    parser = argparse.ArgumentParser(description="Flow-Factory Launcher")
    parser.add_argument("config", type=str, help="Path to YAML config")

    # Multi-node override arguments (CLI > ENV > YAML)
    parser.add_argument(
        "--num_processes",
        type=int,
        default=None,
        help="Total number of processes (overrides YAML and env)",
    )
    parser.add_argument("--num_machines", type=int, default=None, help="Number of nodes")
    parser.add_argument("--machine_rank", type=int, default=None, help="Rank of the current node")
    parser.add_argument(
        "--main_process_ip", type=str, default=None, help="IP address of the master node"
    )
    parser.add_argument(
        "--main_process_port", type=int, default=None, help="Port of the master node"
    )
    return parser.parse_known_args()


def train_cli():
    """Launch ``flow_factory.train`` on one or more processes."""
    _launch_module("flow_factory.train", "Training")


def prompt_enhance_cli():
    """Launch ``flow_factory.prompt_enhance`` on one or more processes."""
    _launch_module("flow_factory.prompt_enhance", "Prompt enhancement")


def _launch_module(module: str, task_name: str):
    """Run ``python -m module <config>`` directly or through ``accelerate launch``."""
    # 1. Parse known args and keep the rest in 'unknown'
    args, unknown = parse_args()
    config = yaml.safe_load(open(args.config, "r"))
    config_file = config.get("config_file")

    # 2. Three-layer config merging: YAML (baseline) -> ENV (auto-detect) -> CLI (highest priority)
    env_overrides = resolve_multinode_env()
    is_multinode = len(env_overrides) > 0

    if is_multinode:
        logger.info(f"Detected multi-node environment variables: {env_overrides}")

    def _resolve(key: str, default=None):
        """Resolve a config value by priority: CLI args > env_overrides > yaml config > default."""
        cli_val = getattr(args, key, None)
        if cli_val is not None:
            return cli_val
        if key in env_overrides:
            return env_overrides[key]
        if key in config:
            return config[key]
        return default

    gpu_count = get_gpu_count()
    num_machines = _resolve("num_machines", 1)
    machine_rank = _resolve("machine_rank", 0)
    main_process_ip = _resolve("main_process_ip")
    main_process_port = _resolve("main_process_port", 29500)
    mixed_precision = config.get("mixed_precision", "bf16")

    # num_processes: if not explicitly specified, infer from num_machines * gpu_count
    num_procs = _resolve("num_processes")
    if num_procs is None:
        num_procs = max(1, num_machines * gpu_count) if gpu_count > 0 else 1

    if num_procs > gpu_count * max(num_machines, 1) and gpu_count > 0:
        logger.warning(
            f"Requested {num_procs} processes but {num_machines} node(s) x "
            f"{gpu_count} GPU(s)/node = {num_machines * gpu_count} GPUs available."
        )

    # 3. Build the arguments for the launched module
    script_args = [args.config] + unknown

    # Check if RANK was set by an external distributed launcher (torchrun) vs platform injection.
    # QZ platform pre-sets RANK but not LOCAL_RANK; torchrun sets both.
    # For multi-NPU setups with platform-injected RANK, force accelerate launch so each
    # subprocess initializes torch_npu in the correct device context.
    _rank_set = os.environ.get("RANK") is not None
    _local_rank_set = os.environ.get("LOCAL_RANK") is not None
    _platform_injected = _rank_set and not _local_rank_set and gpu_count > 1

    if (_rank_set and not _platform_injected) or num_procs <= 1:
        # Already launched by an external launcher (e.g. torchrun), or single-process mode
        cmd = [sys.executable, "-m", module, *script_args]
        logger.info(f"Direct launch: {' '.join(cmd)}")
    else:
        # Launch via accelerate
        cmd = [
            "accelerate",
            "launch",
            "--num_processes",
            str(num_procs),
            "--num_machines",
            str(num_machines),
            "--machine_rank",
            str(machine_rank),
            "--main_process_port",
            str(main_process_port),
            "--mixed_precision",
            str(mixed_precision),
        ]

        # Multi-node requires main_process_ip
        if num_machines > 1:
            if not main_process_ip:
                logger.error(
                    f"main_process_ip is required for multi-node launches. "
                    f"Provide it via one of:\n"
                    f"  1. Environment variable: {', '.join(ENV_VAR_MAPPINGS['master_ip'])}\n"
                    f"  2. CLI argument: --main_process_ip <ip>\n"
                    f"  3. accelerate config_file with main_process_ip set"
                )
                sys.exit(1)
            cmd.extend(["--main_process_ip", main_process_ip])

        if config_file:
            cmd.extend(["--config_file", config_file])

        cmd.extend(["-m", module, *script_args])

        logger.info("=" * 60)
        logger.info("Flow-Factory Launch Configuration")
        logger.info(f"  Num machines:       {num_machines}")
        logger.info(f"  Machine rank:       {machine_rank}")
        logger.info(f"  Num processes:      {num_procs}")
        logger.info(f"  Master node:        {main_process_ip or 'localhost'}:{main_process_port}")
        logger.info(f"  Mixed precision:    {mixed_precision}")
        logger.info(f"  Accelerate config:  {config_file or 'None (using defaults)'}")
        logger.info("=" * 60)

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        if e.returncode in (-signal.SIGINT, 128 + signal.SIGINT):
            logger.info(f"{task_name} interrupted.")
            sys.exit(128 + signal.SIGINT)
        raise
    except KeyboardInterrupt:
        logger.info(f"{task_name} interrupted.")
        sys.exit(128 + signal.SIGINT)


if __name__ == "__main__":
    train_cli()
