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
from typing import Any, Dict, List, Union

# src/flow_factory/hparams/acceleration_args.py
import yaml

from .abc import ArgABC


@dataclass
class AccelerationSpec(ArgABC):
    r"""One accelerator entry: an id (or python path) plus its constructor params.

    The ``name`` resolves through the accelerator registry
    (:mod:`flow_factory.acceleration.registry`); ``params`` is forwarded verbatim
    to the accelerator constructor.

    Example YAML::

        - name: attention_backend
          params: { backend: _flash_3_hub }
    """

    name: str = ""
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError(
                "Each acceleration entry must declare a non-empty `name` "
                "(e.g. `attention_backend`, `torch_compile`, `diffusers_cache`)."
            )

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "params": dict(self.params)}


def _parse_specs(
    value: Union[Dict[str, Any], List[Dict[str, Any]], None],
) -> List[AccelerationSpec]:
    """Parse a slot value into an ordered list of :class:`AccelerationSpec`.

    Accepts a list of entries (canonical) or a single entry dict (shorthand for a
    one-element list), mirroring ``MultiRewardArguments.from_dict``. List order is
    the application order.

    Args:
        value: The raw ``shared`` / ``rollout`` value from the config.

    Returns:
        Ordered list of specs (empty when ``value`` is ``None``).

    Raises:
        ValueError: If ``value`` is neither a list nor a dict.
    """
    if value is None:
        return []
    if isinstance(value, dict):
        return [AccelerationSpec.from_dict(value)]
    if isinstance(value, list):
        return [AccelerationSpec.from_dict(entry) for entry in value]
    raise ValueError(
        f"Acceleration slot must be a list of {{name, params}} entries or a single "
        f"entry dict; got {type(value).__name__}."
    )


@dataclass
class AccelerationArguments(ArgABC):
    r"""Arguments for the model-agnostic acceleration plugin layer.

    Two independent slots, each an **ordered list** of accelerator entries (both
    empty by default, so acceleration is opt-in). List order is the
    application order:

    * ``shared`` — persistent ``stage='both'`` accelerators applied to rollout and
      the training forward, in order, via each accelerator's ``setup()`` (e.g.
      ``attention_backend`` then ``torch_compile`` — backend first so the compiled
      graph captures it).
    * ``rollout`` — accelerators applied only during Stage-3 rollout, nested in
      order via each accelerator's ``rollout_context()``. May be lossy (e.g.
      feature caching), in which case the trainer-paradigm validator restricts them
      to decoupled / distillation algorithms.

    Example YAML::

        acceleration:
          shared:
            - name: attention_backend
              params: { backend: _flash_3_hub }
            - name: torch_compile
              params: { mode: auto }  # auto (default) | regional | full
          rollout:
            - name: diffusers_cache
              params: { policy: first_block, threshold: 0.08 }
    """

    shared: List[AccelerationSpec] = field(default_factory=list)
    rollout: List[AccelerationSpec] = field(default_factory=list)

    @classmethod
    def from_dict(cls, args_dict: Dict[str, Any]) -> "AccelerationArguments":
        """Build from a config dict, parsing each slot into an ordered spec list.

        Args:
            args_dict: The ``acceleration`` config block.

        Returns:
            An ``AccelerationArguments`` with parsed ``shared`` / ``rollout`` lists.
        """
        args_dict = dict(args_dict or {})
        return cls(
            shared=_parse_specs(args_dict.pop("shared", None)),
            rollout=_parse_specs(args_dict.pop("rollout", None)),
            extra_kwargs=args_dict,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "shared": [spec.to_dict() for spec in self.shared],
            "rollout": [spec.to_dict() for spec in self.rollout],
        }

    def __str__(self) -> str:
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False, indent=2)

    def __repr__(self) -> str:
        return self.__str__()
