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

    Flow-Factory trains one sample per data-parallel replica, so without this
    block the launched world *is* the data-parallel dimension. A ``parallelism``
    block splits that world into independent dimensions, which become a
    ``DeviceMesh``:

    * ``dp_shard_size`` -- ranks that shard parameters, gradients and optimizer
      state (FSDP).
    * ``dp_replicate_size`` -- ranks that replicate parameters (HSDP when
      combined with ``dp_shard_size``).
    * ``tp_size`` -- ranks that split one sample's transformer weights. Every
      rank in a tensor-parallel group consumes the *same* sample, so the
      data-parallel world is ``num_processes / tp_size``, not ``num_processes``.

    Every field is optional and defaults to ``None``, which leaves that
    dimension at 1. The product of the three must equal the launched world size,
    so an omitted dimension still has to be reconciled with it.

    ``sequence_parallel`` is a flag rather than a dimension, because Megatron's
    sequence parallelism reuses the tensor-parallel group: it shards the token
    axis exactly where the tensor-parallel plan left activations replicated, and
    gathers back around attention. An all-reduce is a reduce-scatter followed by
    an all-gather, so it costs no extra communication -- it buys the activation
    memory of the norm/residual/modulation region, which is where these
    architectures spend most of it. It therefore requires ``tp_size > 1``.

    Context parallelism is deliberately not exposed. It splits the sequence for
    *attention* as well, so a rank would hold a spatial slice of the latent and
    every adapter's ``decode_latents`` would have to gather before the VAE.

    Example YAML::

        parallelism:
          dp_shard_size: 4
          tp_size: 2
          sequence_parallel: true
    """

    dp_replicate_size: Optional[int] = None
    dp_shard_size: Optional[int] = None
    tp_size: Optional[int] = None
    sequence_parallel: bool = False

    def __post_init__(self) -> None:
        for name in _SIZE_FIELDS:
            value = getattr(self, name)
            if value is None:
                continue
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(
                    f"parallelism.{name} must be an integer when set, received "
                    f"{value!r} of type {type(value).__name__}."
                )
            if value < 1:
                raise ValueError(
                    f"parallelism.{name} must be at least 1 when set, received {value}."
                )
        if not isinstance(self.sequence_parallel, bool):
            raise ValueError(
                "parallelism.sequence_parallel must be a boolean, received "
                f"{self.sequence_parallel!r} of type {type(self.sequence_parallel).__name__}."
            )
        if self.sequence_parallel and self.effective_tensor_parallel_size == 1:
            raise ValueError(
                "parallelism.sequence_parallel shards the token axis across the "
                "tensor-parallel group, so it requires parallelism.tp_size > 1; "
                f"received tp_size={self.tp_size!r}."
            )

    @property
    def is_configured(self) -> bool:
        """True when at least one parallelism dimension was requested explicitly."""
        return any(getattr(self, name) is not None for name in _SIZE_FIELDS)

    @property
    def effective_tensor_parallel_size(self) -> int:
        """The tensor-parallel degree, with an unset dimension meaning 1."""
        return self.tp_size or 1

    def to_dict(self) -> Dict[str, Any]:
        sizes = {name: getattr(self, name) for name in _SIZE_FIELDS}
        return {**sizes, "sequence_parallel": self.sequence_parallel}

    def __str__(self) -> str:
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False, indent=2)

    def __repr__(self) -> str:
        return self.__str__()
