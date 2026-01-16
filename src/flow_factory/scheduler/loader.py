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

# src/flow_factory/scheduler/loader.py
"""
Scheduler Loader
Factory function to instantiate SDE schedulers from pipeline schedulers.
"""
from typing import Union
from diffusers.schedulers.scheduling_utils import SchedulerMixin

from .abc import SDESchedulerMixin
from .registry import get_sde_scheduler_class
from ..hparams import SchedulerArguments
from ..utils.logger_utils import setup_logger

logger = setup_logger(__name__)


def load_scheduler(
    pipeline_scheduler: SchedulerMixin,
    scheduler_args: SchedulerArguments,
) -> SDESchedulerMixin:
    """
    Create an SDE scheduler from a pipeline scheduler and scheduler args.
    
    Merges the original scheduler config with SDE-specific args.
    
    Args:
        pipeline_scheduler: Scheduler from pipeline.from_pretrained()
        scheduler_args: SchedulerArguments with SDE config
    
    Returns:
        Custom SDE scheduler instance
    
    Example:
        >>> pipe = DiffusionPipeline.from_pretrained("...")
        >>> sde_scheduler = load_scheduler(pipe.scheduler, scheduler_args)
    """
    sde_class = get_sde_scheduler_class(pipeline_scheduler)
    
    # Merge base config with SDE args
    base_config = dict(pipeline_scheduler.config)
    base_config.update(scheduler_args.to_dict())
    
    scheduler = sde_class(**base_config)
    logger.info(f"Loaded SDE scheduler: {sde_class.__name__}")
    return scheduler