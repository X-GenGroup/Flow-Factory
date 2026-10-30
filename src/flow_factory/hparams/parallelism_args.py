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

from dataclasses import dataclass
from typing import Any, Dict, Optional

# src/flow_factory/hparams/parallelism_args.py
import yaml

from .abc import ArgABC

_SIZE_FIELDS = ("dp_replicate_size", "dp_shard_size", "tp_size")


@dataclass
class ParallelismArguments(ArgABC):
    r"""Arguments for the multidimensional parallelism plan.

    Flow-Factory trains one sample per data-parallel replica, so today the launched
    world *is* the data-parallel dimension. A ``parallelism`` block splits that world
    into independent dimensions, which the trainer factory turns into Accelerate's
    ``ParallelismConfig`` and therefore into a ``DeviceMesh``:

    * ``dp_shard_size`` — ranks that shard parameters, gradients and optimizer state
      (FSDP).
    * ``dp_replicate_size`` — ranks that replicate parameters (HSDP when combined with
      ``dp_shard_size``).
    * ``tp_size`` — ranks that split one sample's transformer weights. Every rank in a
      tensor-parallel group consumes the *same* sample.

    Every field is optional and defaults to ``None``, which leaves that dimension to
    Accelerate's own default of 1. Accelerate requires
    ``dp_replicate_size * dp_shard_size * tp_size`` to equal ``num_processes``, so an
    omitted dimension has to be reconciled with the launched world size.

    Context parallelism is deliberately not exposed. It splits the sequence dimension,
    so a rank would hold a spatial slice of the latent and every adapter's
    ``decode_latents`` would have to gather before the VAE. Accelerate's separate
    ``sp_size`` is the DeepSpeed Ulysses implementation, which builds no mesh and does
    not compose with a tensor-parallel dimension; sequence parallelism in this
    framework is a placement inside the tensor-parallel plan instead.

    Example YAML::

        parallelism:
          dp_shard_size: 4
          tp_size: 2
    """

    dp_replicate_size: Optional[int] = None
    dp_shard_size: Optional[int] = None
    tp_size: Optional[int] = None

    def __post_init__(self) -> None:
        for name in _SIZE_FIELDS:
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(
                    f"parallelism.{name} must be at least 1 when set, received {value}."
                )

    @property
    def is_configured(self) -> bool:
        """True when at least one parallelism dimension was requested explicitly."""
        return any(getattr(self, name) is not None for name in _SIZE_FIELDS)

    def to_dict(self) -> Dict[str, Any]:
        return {name: getattr(self, name) for name in _SIZE_FIELDS}

    def __str__(self) -> str:
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False, indent=2)

    def __repr__(self) -> str:
        return self.__str__()
