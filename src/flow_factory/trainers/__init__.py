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

# src/flow_factory/trainers/__init__.py
"""Trainers for online RL, distillation, and offline fine-tuning."""

from .abc import BaseTrainer
from .execution import (
    AcquisitionDriver,
    DatasetAcquisitionDriver,
    GenerationAcquisitionDriver,
    TrainingProgress,
    build_acquisition_driver,
)
from .loader import load_trainer
from .registry import get_trainer_class, list_registered_trainers

__all__ = [
    "BaseTrainer",
    "AcquisitionDriver",
    "DatasetAcquisitionDriver",
    "GenerationAcquisitionDriver",
    "TrainingProgress",
    "build_acquisition_driver",
    "get_trainer_class",
    "list_registered_trainers",
    "load_trainer",
]
