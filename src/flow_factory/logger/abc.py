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

# src/flow_factory/logger/abc.py
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List

from ..hparams import *
from .formatting import LogFormatter, LogImage, LogVideo, LogTable


class Logger(ABC):
    platform: Any

    def __init__(self, config: Arguments):
        self.config = config
        self.clean_up_freq = 10
        self._pending_cleanup: List[Dict] = []
        self._init_platform()

    @abstractmethod
    def _init_platform(self):
        pass

    def log_data(
        self,
        data: Dict[str, Any],
        step: int,
        keys: Optional[str] = None,
    ):
        # 1. Process rules (Mean, Paths, wrappers) into IR
        formatted_dict = LogFormatter.format_dict(data)
        
        # 2. Filter keys if requested
        if keys:
            valid_keys = keys.split(',')
            formatted_dict = {k: v for k, v in formatted_dict.items() if k in valid_keys}

        # 3. Convert IR to Platform Objects
        final_dict = {}
        for k, v in formatted_dict.items():
            converted = self._recursive_convert(v)
            if isinstance(converted, dict):  # for LogTable conversion returning dict, e.g., SwanlabLogger
                final_dict.update(converted)
            else:
                final_dict[k] = converted

        # 4. Actual Logging
        if final_dict:
            self._log_impl(final_dict, step)
            
        # 5. Cleanup temporary files periodically
        if len(self._pending_cleanup) >= self.clean_up_freq:
            first_data = self._pending_cleanup.pop(0)
            self._cleanup_temp_files(first_data)
        self._pending_cleanup.append(formatted_dict)

    def _recursive_convert(
        self, 
        value: Any, 
        height: Optional[int] = None,
        width: Optional[int] = None
    ) -> Any:
        """Recursively convert IR objects to platform objects."""
        if isinstance(value, (list, tuple)):
            return [self._recursive_convert(v, height, width) for v in value if v is not None]
        return self._convert_to_platform(value, height, width)
    
    def _cleanup_temp_files(self, data: Dict):
        for value in data.values():
            if isinstance(value, (LogImage, LogVideo, LogTable)):
                value.cleanup()
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, (LogImage, LogVideo, LogTable)):
                        item.cleanup()

    @abstractmethod
    def _convert_to_platform(
        self, 
        value: Any, 
        height: Optional[int] = None,
        width: Optional[int] = None
    ) -> Any:
        """
        Convert a single IR object to platform-specific object.
        
        Args:
            value: IR object (LogImage, LogVideo, LogTable) or pass-through value.
            height: Optional target height for resize (aspect-ratio preserved if width is None).
            width: Optional target width for resize (aspect-ratio preserved if height is None).
        
        Returns:
            Platform-specific object (e.g., wandb.Image, swanlab.Video).
        """
        pass

    @abstractmethod
    def _log_impl(self, data: Dict, step: int):
        pass