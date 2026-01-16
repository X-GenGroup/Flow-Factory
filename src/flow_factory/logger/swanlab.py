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

# src/flow_factory/logger/swanlab.py
from typing import Any, Dict, Optional
import swanlab
from .abc import Logger
from .formatting import LogImage, LogVideo, LogTable
from ..utils.logger_utils import setup_logger

logger = setup_logger(__name__)


class SwanlabLogger(Logger):
    def _init_platform(self):
        swanlab.init(
            project=self.config.project,
            name=self.config.run_name,
            config=self.config.to_dict()
        )
        self.platform = swanlab

    def _convert_to_platform(
        self, 
        value: Any, 
        height: Optional[int] = None,
        width: Optional[int] = None
    ) -> Any:
        if isinstance(value, LogImage):
            return swanlab.Image(value.get_value(height, width), caption=value.caption)
        
        if isinstance(value, LogVideo):
            return swanlab.Video(value.get_value('gif', height, width), caption=value.caption)
        
        if isinstance(value, LogTable):
            # SwanLab does not support Table natively, convert to dict of column lists
            h = height or value.target_height  # Use specified height or default
            table_dict = {col: [] for col in value.columns}
            for row in value.rows:
                for col, item in zip(value.columns, row):
                    converted = self._convert_to_platform(item, height=h) if item is not None else None
                    table_dict[col].append(converted)
            return table_dict
        
        return value

    def _log_impl(self, data: Dict, step: int):
        self.platform.log(data, step=step)