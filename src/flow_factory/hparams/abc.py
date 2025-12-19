from typing import Any, Literal, Optional, Union
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field


@dataclass
class ArgABC(ABC):
    """Abstract Base Class for Hyper-Parameters configurations."""

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        pass