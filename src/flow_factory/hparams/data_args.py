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

from dataclasses import dataclass, field
from typing import Any, List, Literal, Optional

# src/flow_factory/hparams/data_args.py
import yaml

from .abc import ArgABC
from .dataset_args import DatasetArguments


@dataclass
class DataArguments(ArgABC):
    r"""Arguments pertaining to data input for training and evaluation."""

    dataset_dir: str = field(
        default="data",
        metadata={"help": "Path to the folder containing the datasets."},
    )
    cache_dir: str = field(
        default="~/.cache/flow_factory/datasets",
        metadata={
            "help": "Directory for caching preprocessed datasets (fingerprinted by content hash)."
        },
    )
    image_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the folder containing conditioning images. Defaults to 'images' subfolder in dataset_dir."
        },
    )
    video_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the folder containing conditioning videos. Defaults to 'videos' subfolder in dataset_dir."
        },
    )
    audio_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the folder containing audio files. Defaults to 'audios' subfolder in dataset_dir."
        },
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
        metadata={
            "help": "Whether to force reprocessing of the dataset even if cached data exists."
        },
    )
    max_dataset_size: Optional[int] = field(
        default=None,
        metadata={"help": "If set, limits the maximum number of samples in the dataset."},
    )
    preprocess_parallelism: Literal["global", "local"] = field(
        default="local",
        metadata={
            "help": (
                "Controls how distributed preprocessing is parallelized. "
                "'global': all processes across all nodes split and merge the dataset (requires a shared filesystem). "
                "'local': each node independently splits the dataset among its local processes and merges locally "
                "(no shared filesystem required across nodes)."
            )
        },
    )
    sampler_type: Literal[
        "auto",
        "distributed_k_repeat",
        "group_contiguous",
        "group_distributed",
        "group_tiled",
    ] = field(
        default="auto",
        metadata={
            "help": (
                "Sampler strategy for K-repeat distributed sampling. "
                "'auto': prefer group_contiguous (minimal communication), "
                "fall back to distributed_k_repeat when geometric constraints "
                "(unique_sample_num % world_size) cannot be satisfied. "
                "'distributed_k_repeat': shuffle K copies globally across ranks "
                "(fewer constraints, extra all-gather communication). "
                "'group_contiguous': keep all K copies of each group on the same rank "
                "(requires unique_sample_num divisible by world_size). "
                "'group_distributed': pack complete groups into each global microbatch "
                "(requires group_size to divide world_size * per_device_batch_size). "
                "'group_tiled': pack complete groups into the smallest global-microbatch "
                "window (supports every positive group/global-batch geometry). "
                "DGPO accepts only 'group_distributed'; 'auto' resolves to it."
            )
        },
    )

    datasets: Optional[List[DatasetArguments]] = field(
        default=None,
        metadata={
            "help": (
                "Unified list of dataset folders. Each entry opts into "
                "training (`train:` sub-block) and/or evaluation (`eval:` sub-block); "
                "shared `image_dir` / `video_dir` / `audio_dir` apply to both splits. "
                "When set, the legacy `dataset_dir` is ignored for any split that has a "
                "matching dataset entry. See `DatasetArguments` for the per-entry schema."
            )
        },
    )

    def __post_init__(self):
        self.dataset = self.dataset_dir

        # Coerce list-of-dict -> list-of-DatasetArguments
        # (`ArgABC.from_dict` does NOT recurse into nested list-of-ArgABC fields).
        # Idempotent: re-running on already-coerced entries is a no-op.
        if self.datasets is not None:
            coerced: List[DatasetArguments] = []
            for item in self.datasets:
                if isinstance(item, DatasetArguments):
                    coerced.append(item)
                elif isinstance(item, dict):
                    coerced.append(DatasetArguments.from_dict(item))
                else:
                    raise TypeError(
                        f"data.datasets entry must be dict or DatasetArguments, "
                        f"got {type(item).__name__}."
                    )
            # Treat empty list like None so downstream `if data_args.datasets:` works.
            self.datasets = coerced or None

    # ------------------------------------------------------------------
    # Per-split convenience accessors (filter `datasets` by participation).
    # Trainers and the data-loader read these instead of branching on the
    # nested `train:` / `eval:` blocks themselves.
    # ------------------------------------------------------------------

    @property
    def training_datasets(self) -> List[DatasetArguments]:
        """Datasets that opt into training (have `train: enabled`)."""
        return [d for d in (self.datasets or []) if d.is_training_source]

    @property
    def eval_datasets(self) -> List[DatasetArguments]:
        """Datasets that opt into evaluation (have `eval: enabled`)."""
        return [d for d in (self.datasets or []) if d.is_eval_source]

    # ------------------------------------------------------------------
    # Source-id registry (populated by Arguments._assign_source_ids;
    # public for trainers, log builders, and reward routing).
    # ------------------------------------------------------------------

    @property
    def source_id_to_name(self) -> List[str]:
        """Stable mapping ``source_id (int) -> source name (str)``.

        Established once in :meth:`Arguments._assign_source_ids` after
        dataset canonicalization; identical on every rank because YAML
        parsing is deterministic and every rank receives the same
        config dict. Returns ``[]`` in legacy single-source mode (no
        ``data.datasets``) — consumers should treat the empty list as
        "no source-id space defined" and fall back to legacy behavior.
        """
        if self.datasets is None:
            return []
        return [d.name for d in self.datasets]

    @property
    def source_name_to_id(self) -> dict[str, int]:
        """Inverse mapping, cached after first access."""
        cached = getattr(self, "_source_name_to_id_cache", None)
        if cached is None:
            cached = {n: i for i, n in enumerate(self.source_id_to_name)}
            object.__setattr__(self, "_source_name_to_id_cache", cached)
        return cached

    def to_dict(self) -> dict[str, Any]:
        return super().to_dict()

    def __str__(self) -> str:
        """Pretty print configuration as YAML."""
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False, indent=2)

    def __repr__(self) -> str:
        """Same as __str__ for consistency."""
        return self.__str__()
