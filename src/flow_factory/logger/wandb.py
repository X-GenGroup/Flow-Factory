# src/flow_factory/logger/wandb.py
from typing import Any, Dict, Optional
import wandb
from .abc import Logger
from .formatting import LogImage, LogVideo, LogTable


class WandbLogger(Logger):
    def _init_platform(self):
        wandb.init(
            project=self.config.project,
            name=self.config.run_name,
            config=self.config.to_dict()
        )
        self.platform = wandb

    def _convert_to_platform(
        self, 
        value: Any, 
        height: Optional[int] = None,
        width: Optional[int] = None
    ) -> Any:
        if isinstance(value, LogImage):
            return wandb.Image(value.get_value(height, width), caption=value.caption)
        
        if isinstance(value, LogVideo):
            return wandb.Video(value.get_value(height=height, width=width), caption=value.caption)
        
        if isinstance(value, LogTable):
            # For LogTable, all items have the same height for better formatting
            h = height or value.target_height # Use specified height or default
            data = [
                [
                    self._convert_to_platform(item, height=h) if item is not None else None 
                    for item in row
                ]
                for row in value.rows
            ]
            return wandb.Table(columns=value.columns, data=data)
        
        return value

    def _log_impl(self, data: Dict, step: int):
        self.platform.log(data, step=step)