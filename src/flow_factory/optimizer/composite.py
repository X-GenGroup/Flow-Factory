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

"""Present several optimizers to the framework as a single one."""

from typing import Any, Callable, Dict, List, Optional, Sequence

import torch


class CompositeOptimizer(torch.optim.Optimizer):
    """Drive several optimizers as one object.

    The framework prepares exactly one optimizer root, because DeepSpeed builds one
    engine and FSDP2 wraps one root. Some algorithms nevertheless need two update
    rules at once: ``torch.optim.Muon`` rejects any parameter that is not a matrix,
    so a Muon-optimized variant also needs AdamW for its biases, normalization
    parameters and embeddings.

    This holds those children and exposes their parameter groups as one list, so
    every reader (gradient clipping, checkpointing, the role coordinator) sees the
    layout it expects. It deliberately implements nothing beyond delegation.

    Args:
        optimizers: Child optimizers, in the order their groups should appear.

    Raises:
        ValueError: If no child optimizer is supplied.
        TypeError: If a child is not a ``torch.optim.Optimizer``.
    """

    def __init__(self, optimizers: Sequence[torch.optim.Optimizer]) -> None:
        children = list(optimizers)
        if not children:
            raise ValueError("expected at least one child optimizer, received none")
        for index, optimizer in enumerate(children):
            if not isinstance(optimizer, torch.optim.Optimizer):
                raise TypeError(
                    f"expected torch.optim.Optimizer at index {index}, received "
                    f"{type(optimizer).__name__}: {optimizer!r}"
                )
        self.optimizers = children
        # Deliberately skip Optimizer.__init__: this owns no parameters of its own,
        # and re-registering the children's groups here would double-count them.
        self.defaults: Dict[str, Any] = {}
        self.state: Dict[Any, Any] = {}

    @property
    def param_groups(self) -> List[Dict[str, Any]]:
        """Return every child's parameter groups, child order preserved."""
        return [group for optimizer in self.optimizers for group in optimizer.param_groups]

    @param_groups.setter
    def param_groups(self, value: Any) -> None:
        """Reject reassignment, which would silently detach the children."""
        raise AttributeError(
            "CompositeOptimizer.param_groups is a view over its child optimizers and "
            "cannot be reassigned; mutate the child optimizers instead"
        )

    def add_param_group(self, param_group: Dict[str, Any]) -> None:
        """Reject late group addition, which has no unambiguous child to join."""
        raise NotImplementedError(
            "CompositeOptimizer cannot add a parameter group after construction: the "
            "child optimizer that should own it is ambiguous"
        )

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Step every child optimizer.

        Args:
            closure: Optional closure re-evaluating the model, evaluated once.

        Returns:
            The closure's value when supplied.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for optimizer in self.optimizers:
            optimizer.step()
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear gradients on every child optimizer."""
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> Dict[str, Any]:
        """Return one state dict holding every child's state in order."""
        return {"composite": [optimizer.state_dict() for optimizer in self.optimizers]}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restore every child's state from a dict this class produced.

        Args:
            state_dict: Mapping previously returned by :meth:`state_dict`.

        Raises:
            ValueError: If ``state_dict`` has no ``composite`` list or its child count does
                not match this optimizer.
        """
        children_state = state_dict.get("composite")
        if not isinstance(children_state, list):
            raise ValueError(
                "expected a CompositeOptimizer state dict with a 'composite' list, "
                f"received keys {sorted(state_dict)}"
            )
        if len(children_state) != len(self.optimizers):
            raise ValueError(
                f"expected state for {len(self.optimizers)} child optimizers, "
                f"received {len(children_state)}"
            )
        for optimizer, child_state in zip(self.optimizers, children_state):
            optimizer.load_state_dict(child_state)

    def __repr__(self) -> str:
        """Describe the composite by its children."""
        children = ", ".join(type(optimizer).__name__ for optimizer in self.optimizers)
        return f"CompositeOptimizer({children})"
