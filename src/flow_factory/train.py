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

# src/flow_factory/train.py
import os
import argparse
import logging
from .hparams import Arguments
from .trainers import load_trainer

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s')
logger = logging.getLogger("flow_factory.train")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Flow-Factory Training")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    return parser.parse_known_args()


def main():
    args, unknown = parse_args()
    
    # Load configuration
    config = Arguments.load_from_yaml(args.config)
    
    # Log distributed setup info (only from rank 0)
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    
    if rank == 0:
        logger.info("=" * 100)
        logger.info("Flow-Factory Training Initialized")
        logger.info(f"World Size: {world_size}")
        logger.info("=" * 100)
        logger.info(f"Config: {args.config}")
        logger.info(f"\n{config}")
        logger.info("=" * 100)
    
    # Launch trainer
    trainer = load_trainer(config)
    trainer.start()
    
    if rank == 0:
        logger.info("Training completed successfully")


if __name__ == "__main__":
    main()