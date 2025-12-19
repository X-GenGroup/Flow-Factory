# flow_factory/cli.py
import sys
import os
import subprocess
import argparse
import logging
import torch

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
    parser.add_argument("--num_processes", type=int, default=None, 
                        help="Number of processes (default: auto-detect from GPUs)")
    parser.add_argument("--main_process_port", type=str, default="29500",
                        help="Port for distributed communication")
    return parser.parse_known_args()


def train_cli():
    # 1. Parse known args (config, num_processes) and keep the rest in 'unknown'
    args, unknown = parse_args()
    
    # Determine process count
    gpu_count = get_gpu_count()
    num_procs = args.num_processes or max(1, gpu_count)
    
    if args.num_processes and args.num_processes > gpu_count > 0:
        logger.warning(
            f"Requested {args.num_processes} processes but only {gpu_count} GPUs available."
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
            "--main_process_port", args.main_process_port,
            "-m", "flow_factory.train",
            *script_args
        ]
        logger.info(f"Accelerate launch: {' '.join(cmd[:6])}...")
    
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    train_cli()