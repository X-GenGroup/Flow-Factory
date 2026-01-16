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

# src/flow_factory/utils/logger_utils.py
import os
import logging
import torch

def get_rank():
    """Get process rank for distributed training."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', 0)))

def setup_logger(name: str = None, level: int = logging.INFO, rank_zero_only: bool = False):
    """
    Setup logger with rank information.
    
    Args:
        name: Logger name
        level: Logging level
        rank_zero_only: If True, only rank 0 will output logs
    """
    rank = get_rank()
    
    # Silence non-zero ranks if requested
    if rank_zero_only and rank != 0:
        logger = logging.getLogger(name)
        logger.setLevel(logging.CRITICAL + 1)  # Effectively disable
        return logger
    
    formatter = logging.Formatter(
        f'[%(asctime)s] [Rank {rank}] [%(levelname)s] [%(name)s]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.propagate = False
    
    return logger