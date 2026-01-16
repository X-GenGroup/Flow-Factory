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
import subprocess
import argparse
import logging
import torch
import yaml


logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s')
logger = logging.getLogger(__name__)


def get_gpu_count():
    """Detect available GPU count using torch."""
    try:
        return torch.cuda.device_count()
    except (ImportError, RuntimeError):
        return 0


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Flow-Factory Launcher")
    parser.add_argument("config", type=str, help="Path to YAML config")
    return parser.parse_known_args()


def train_cli():
    # 1. Parse known args (config, num_processes) and keep the rest in 'unknown'
    args, unknown = parse_args()
    config = yaml.safe_load(open(args.config, 'r'))
    config_file = config['config_file']
    
    # Determine process count
    gpu_count = get_gpu_count()
    num_procs = config["num_processes"] or max(1, gpu_count)
    
    if config["num_processes"] and config["num_processes"] > gpu_count > 0:
        logger.warning(
            f"Requested {config['num_processes']} processes but only {gpu_count} GPUs available."
        )
    
    logger.info(f"Launching with {num_procs} processes on {gpu_count} available GPUs")

    # 2. Build the arguments strictly for the training script
    # We explicitly pass the config file, followed by any unparsed arguments
    script_args = [args.config] + unknown
    
    if os.environ.get("RANK") is not None or num_procs <= 1:
        # Single process direct launch
        cmd = [sys.executable, "-m", "flow_factory.train", *script_args]
        logger.info(f"Direct launch: {' '.join(cmd)}")
    else:
        # Multi-process launch via accelerate
        cmd = [
            "accelerate", "launch",
            "--num_processes", str(num_procs),
            "--main_process_port", str(config["main_process_port"]),
            "--mixed_precision", str(config["mixed_precision"]),
            "--config_file", config_file,
            "-m", "flow_factory.train",
            *script_args
        ]
        logger.info(f"Accelerate launch: {' '.join(cmd[:6])}...")
    
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    train_cli()