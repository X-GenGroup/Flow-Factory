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

# src/flow_factory/cli.py
import sys
import os
import signal
import subprocess
import argparse
import logging
import torch
import yaml


logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s')
logger = logging.getLogger(__name__)


# ========================================================================
# Multi-node environment variable mapping table.
# Supports multiple cluster scheduler naming conventions (ordered by priority).
# ========================================================================
_ENV_VAR_MAPPINGS = {
    "master_ip":      ["MASTER_IP", "MASTER_ADDR", "CHIEF_IP"],
    "master_port":    ["MASTER_PORT"],
    "machine_rank":   ["MACHINE_RANK", "NODE_RANK", "INDEX"],
    "num_machines":   ["NUM_MACHINES", "NUM_NODES", "HOST_NUM"],
    "gpus_per_node":  ["GPUS_PER_NODE", "HOST_GPU_NUM"],
}


def _env_lookup(key: str) -> str | None:
    """Look up a value from environment variable mapping by priority."""
    for env_name in _ENV_VAR_MAPPINGS.get(key, []):
        val = os.environ.get(env_name)
        if val is not None and val != "":
            return val
    return None


def resolve_multinode_env() -> dict:
    """
    Detect multi-node configuration from environment variables.

    Returns a dict containing only successfully resolved fields.
    Returns an empty dict if not in a multi-node environment (i.e. key variables missing).
    """
    master_ip = _env_lookup("master_ip")
    num_machines_str = _env_lookup("num_machines")

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
    rank_str = _env_lookup("machine_rank")
    if rank_str is not None:
        env_config["machine_rank"] = int(rank_str)

    # Optional: master_port
    port_str = _env_lookup("master_port")
    if port_str is not None:
        env_config["main_process_port"] = int(port_str)

    # Optional: gpus_per_node -> used to compute num_processes
    gpus_str = _env_lookup("gpus_per_node")
    if gpus_str is not None:
        env_config["num_processes"] = num_machines * int(gpus_str)

    return env_config


def get_gpu_count():
    """Detect available GPU count using torch."""
    try:
        return torch.cuda.device_count()
    except (ImportError, RuntimeError):
        return 0


def parse_args():
    """Parse command line arguments with optional multi-node overrides."""
    parser = argparse.ArgumentParser(description="Flow-Factory Launcher")
    parser.add_argument("config", type=str, help="Path to YAML config")

    # Multi-node override arguments (CLI > ENV > YAML)
    parser.add_argument("--num_processes", type=int, default=None,
                        help="Total number of processes (overrides YAML and env)")
    parser.add_argument("--num_machines", type=int, default=None,
                        help="Number of nodes")
    parser.add_argument("--machine_rank", type=int, default=None,
                        help="Rank of the current node")
    parser.add_argument("--main_process_ip", type=str, default=None,
                        help="IP address of the master node")
    parser.add_argument("--main_process_port", type=int, default=None,
                        help="Port of the master node")
    return parser.parse_known_args()


def train_cli():
    # 1. Parse known args and keep the rest in 'unknown'
    args, unknown = parse_args()
    config = yaml.safe_load(open(args.config, 'r'))
    config_file = config.get('config_file')

    # 2. Three-layer config merging: YAML (baseline) -> ENV (auto-detect) -> CLI (highest priority)
    env_overrides = resolve_multinode_env()
    is_multinode = len(env_overrides) > 0

    if is_multinode:
        logger.info(f"Detected multi-node environment variables: {env_overrides}")

    def _resolve(key: str, default=None):
        """Resolve a config value with priority: CLI args > env_overrides > yaml config > default."""
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

    # 3. Build the arguments for the training script
    script_args = [args.config] + unknown

    if os.environ.get("RANK") is not None or num_procs <= 1:
        # Already launched by an external launcher (e.g. torchrun), or single-process mode
        cmd = [sys.executable, "-m", "flow_factory.train", *script_args]
        logger.info(f"Direct launch: {' '.join(cmd)}")
    else:
        # Launch via accelerate
        cmd = [
            "accelerate", "launch",
            "--num_processes", str(num_procs),
            "--num_machines", str(num_machines),
            "--machine_rank", str(machine_rank),
            "--main_process_port", str(main_process_port),
            "--mixed_precision", str(mixed_precision),
        ]

        # Multi-node requires main_process_ip
        if num_machines > 1:
            if not main_process_ip:
                logger.error(
                    f"main_process_ip is required for multi-node training. "
                    f"Provide it via one of:\n"
                    f"  1. Environment variable: {', '.join(_ENV_VAR_MAPPINGS['master_ip'])}\n"
                    f"  2. CLI argument: --main_process_ip <ip>\n"
                    f"  3. accelerate config_file with main_process_ip set"
                )
                sys.exit(1)
            cmd.extend(["--main_process_ip", main_process_ip])

        if config_file:
            cmd.extend(["--config_file", config_file])

        cmd.extend(["-m", "flow_factory.train", *script_args])

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
            logger.info("Training interrupted.")
            sys.exit(128 + signal.SIGINT)
        raise
    except KeyboardInterrupt:
        logger.info("Training interrupted.")
        sys.exit(128 + signal.SIGINT)


if __name__ == "__main__":
    train_cli()