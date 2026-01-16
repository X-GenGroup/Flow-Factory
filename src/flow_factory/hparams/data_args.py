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

# src/flow_factory/hparams/data_args.py
import yaml
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional, Tuple, Union, List, Iterable
from .abc import ArgABC


@dataclass
class DataArguments(ArgABC):
    r"""Arguments pertaining to data input for training and evaluation."""
    dataset_dir: str = field(
        default="data",
        metadata={"help": "Path to the folder containing the datasets."},
    )
    image_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the folder containing conditioning images. Defaults to 'images' subfolder in dataset_dir."},
    )
    video_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the folder containing conditioning videos. Defaults to 'videos' subfolder in dataset_dir."},
    )
    preprocessing_batch_size: int = field(
        default=8,
        metadata={"help": "The batch size for preprocessing the datasets."},
    )
    dataloader_num_workers: int = field(
        default=16,
        metadata={"help": "The number of workers for DataLoader."},
    )
    enable_preprocess: bool = field(
        default=True,
        metadata={"help": "Whether to enable preprocessing of the dataset."},
    )
    force_reprocess: bool = field(
        default=True,
        metadata={"help": "Whether to force reprocessing of the dataset even if cached data exists."},
    )
    max_dataset_size: Optional[int] = field(
        default=None,
        metadata={"help": "If set, limits the maximum number of samples in the dataset."},
    )

    def __post_init__(self):
        self.dataset = self.dataset_dir

    def to_dict(self) -> dict[str, Any]:
        return super().to_dict()

    def __str__(self) -> str:
        """Pretty print configuration as YAML."""
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False, indent=2)
    
    def __repr__(self) -> str:
        """Same as __str__ for consistency."""
        return self.__str__()