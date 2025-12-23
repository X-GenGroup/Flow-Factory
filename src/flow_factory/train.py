# flow_factory/train.py
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