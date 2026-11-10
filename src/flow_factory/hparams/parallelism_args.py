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

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any, Dict, Optional

# src/flow_factory/hparams/parallelism_args.py
import yaml

from ..contracts.topology import ParallelTopology
from .abc import ArgABC

_SIZE_FIELDS = ("dp_replicate_size", "dp_shard_size", "tp_size", "cp_size")
_TARGET_MATERIALIZATIONS = ("replicated", "sharded")


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
      data-parallel world is ``num_processes / (tp_size * cp_size)``.

    Every size is optional and defaults to ``None``. An omitted ``dp_shard_size``
    uses ``world_size / (dp_replicate_size * tp_size * cp_size)``; other omitted
    dimensions mean one. YAML ``null`` also means omitted. Explicit sizes are
    strict constraints: a nonintegral remainder or mismatched complete product
    raises before model loading instead of overriding the user's dimensions.
    The world size is the global process count exported by the launcher.

    ``sequence_parallel`` is a flag rather than a dimension, because Megatron's
    sequence parallelism reuses the tensor-parallel group: it shards the token
    axis exactly where the tensor-parallel plan left activations replicated, and
    gathers back around attention. Its activation traffic matches tensor
    parallelism alone in both directions: each all-reduce of the plan becomes a
    reduce-scatter and an all-gather, at the out-projection and at the
    attention or feed-forward entry -- on every adapter but MiniMax-H3, whose
    entries still leave one all-reduce per projection in backward. What it adds
    are small all-reduces of the gradients of replicated parameters and
    broadcast conditioning inside the sharded region. What it buys is the
    activation memory of the norm/residual/modulation region, which is where
    these architectures spend most of it. It requires ``tp_size > 1``.

    Sequence parallelism reduces retained activations in model-declared regions.
    Its memory and communication costs depend on the architecture, token geometry
    and checkpointing; parameter coverage is not a throughput guarantee.
    The field is tri-state:

    * unset (``None``) -- the adapter decides, through its
      ``sequence_parallel_by_default``: on for SD3.5 and Wan T2V, whose overlays
      are verified end to end on real weights, off elsewhere.
    * ``True`` -- require it. An adapter without a plan is an error, because the
      run asked for something it cannot have.
    * ``False`` -- do not use it, even where a plan exists.

    ``verify_replicated_inputs`` is a validation switch, off by default. The
    plan assumes every peer of a group enters the sharded component with the
    same inputs; when they differ, the output matches neither peer's unsharded
    result and nothing raises. With the switch on, each forward first compares
    every tensor input bit for bit across the group and raises naming the first
    that differs. It costs four collectives per forward, so it is meant for GPU
    validation runs rather than production.

    ``cp_size`` splits attention tokens across ranks sharing one logical sample.
    FSDP2 with Ulysses uses the model and trainer capability contracts, optionally
    composed with the declared training-aware TP/SP plan. The loaded layout,
    precision and dependency tuple must pass their preflight guards.
    The adapter restores the full output before scheduler or objective execution.
    Logical data parallelism is ``num_processes / (tp_size * cp_size)``.

    ``target_materialization`` selects how FSDP2 target roots reach their
    prepared layout:

    * unset (``None``, default) -- prefer ``sharded``. When the run is outside
      its qualified boundary (not FSDP2, an adapter without the capability,
      or FSDP2 CPU offload), fall back to ``replicated``,
      with a warning under FSDP2.
    * ``replicated`` -- every rank keeps its loaded root while tensor
      parallelism and FSDP2 shard it. The initialization peak holds the
      TP-local root and every new FSDP2 shard.
    * ``sharded`` -- require it; outside the boundary is an error. After
      initial-state synchronization, the loaded values
      are retained as a host-side source and the target parameters move to the
      meta device. Tensor/sequence parallelism and FSDP2 build the final layout
      on meta, then each rank allocates only its local shards and copies its
      own block of the source. The local values equal the replicated path, the
      peak stays near the final sharded state, tensor parallelism stops
      broadcasting loaded weights, and ``fsdp_cpu_ram_efficient_loading``
      composes with tensor parallelism.

    The trainer resolves an unset value before loading weights and records the
    result in this field.

    Example YAML::

        parallelism:
          tp_size: 2
          # dp_shard_size is inferred: 4 on an eight-process launch.
    """

    dp_replicate_size: Optional[int] = None
    dp_shard_size: Optional[int] = None
    tp_size: Optional[int] = None
    cp_size: Optional[int] = None
    sequence_parallel: Optional[bool] = None
    verify_replicated_inputs: bool = False
    target_materialization: Optional[str] = None

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
        if self.sequence_parallel is not None and not isinstance(self.sequence_parallel, bool):
            raise ValueError(
                "parallelism.sequence_parallel must be a boolean when set, received "
                f"{self.sequence_parallel!r} of type {type(self.sequence_parallel).__name__}."
            )
        if self.sequence_parallel and self.effective_tensor_parallel_size == 1:
            raise ValueError(
                "parallelism.sequence_parallel shards the token axis across the "
                "tensor-parallel group, so it requires parallelism.tp_size > 1; "
                f"received tp_size={self.tp_size!r}."
            )
        if not isinstance(self.verify_replicated_inputs, bool):
            raise ValueError(
                "parallelism.verify_replicated_inputs must be a boolean, received "
                f"{self.verify_replicated_inputs!r} of type "
                f"{type(self.verify_replicated_inputs).__name__}."
            )
        if self.verify_replicated_inputs and self.effective_tensor_parallel_size == 1:
            raise ValueError(
                "parallelism.verify_replicated_inputs compares inputs across the "
                "tensor-parallel group, so it requires parallelism.tp_size > 1; "
                f"received tp_size={self.tp_size!r}."
            )

        if (
            self.target_materialization is not None
            and self.target_materialization not in _TARGET_MATERIALIZATIONS
        ):
            raise ValueError(
                "parallelism.target_materialization must be unset or one of "
                f"{_TARGET_MATERIALIZATIONS}, received {self.target_materialization!r}."
            )

    @property
    def materializes_targets_sharded(self) -> bool:
        """True once resolved to materialize FSDP2 target roots after meta preparation."""
        return self.target_materialization == "sharded"

    @classmethod
    def from_dict(cls, args_dict: Mapping[str, Any]) -> "ParallelismArguments":
        """Parse the ``parallelism`` block, refusing any key it does not declare.

        Other blocks keep an unknown key in ``extra_kwargs`` for adapter- or
        algorithm-specific use; nothing reads this block that way, and a
        misspelled flag here -- ``sequence_paralel: false`` -- would otherwise
        change the run's geometry by being ignored.

        Args:
            args_dict: The ``parallelism`` block.

        Returns:
            The parsed arguments.

        Raises:
            TypeError: If the block is not a mapping.
            ValueError: If it carries a key this class does not declare.
        """
        if not isinstance(args_dict, Mapping):
            raise TypeError(
                "expected the parallelism block as a mapping, received "
                f"{type(args_dict).__name__}: {args_dict!r}"
            )
        valid = sorted(field.name for field in fields(cls) if field.name != "extra_kwargs")
        unknown = sorted(set(args_dict) - set(valid), key=str)
        if unknown:
            raise ValueError(f"expected parallelism keys in {valid}, received unknown={unknown}")
        return super().from_dict(dict(args_dict))

    @property
    def is_configured(self) -> bool:
        """True when at least one parallelism dimension was requested explicitly."""
        return any(getattr(self, name) is not None for name in _SIZE_FIELDS)

    @property
    def effective_tensor_parallel_size(self) -> int:
        """The tensor-parallel degree, with an unset dimension meaning 1."""
        return self.tp_size or 1

    @property
    def effective_context_parallel_size(self) -> int:
        """Return the context-parallel degree, with an unset dimension meaning one."""
        return self.cp_size or 1

    def resolve_topology(self, world_size: int) -> ParallelTopology:
        """Resolve the configured dimensions before any backend allocation."""
        return ParallelTopology.resolve(
            world_size,
            dp_replicate_size=self.dp_replicate_size,
            dp_shard_size=self.dp_shard_size,
            tp_size=self.tp_size,
            cp_size=self.cp_size,
        )

    def to_dict(self) -> Dict[str, Any]:
        sizes = {name: getattr(self, name) for name in _SIZE_FIELDS}
        return {
            **sizes,
            "sequence_parallel": self.sequence_parallel,
            "verify_replicated_inputs": self.verify_replicated_inputs,
            "target_materialization": self.target_materialization,
        }

    def __str__(self) -> str:
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False, indent=2)

    def __repr__(self) -> str:
        return self.__str__()
