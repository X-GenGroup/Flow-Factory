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

# src/flow_factory/hparams/abc.py

from dataclasses import dataclass, field, fields, asdict
from typing import Any, Dict
from abc import ABC, abstractmethod


@dataclass(kw_only=True)
class ArgABC(ABC):
    """Abstract Base Class with 'extra_kwargs' support."""

    extra_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, args_dict: Dict[str, Any]):
        """
        Init from dict. Unknown keys are moved to 'extra_kwargs' if that field exists.
        """
        field_names = {f.name for f in fields(cls)}
        
        # 1. Separate known fields from unknown (extra) fields
        init_data = {}
        extras = {}
        
        for k, v in args_dict.items():
            if k in field_names:
                init_data[k] = v
            else:
                extras[k] = v

        # 2. If the class has an 'extra_kwargs' field, inject the leftovers there
        if "extra_kwargs" in field_names:
            # If the config actually had an explicit "extra_kwargs" key, merge it
            if "extra_kwargs" in init_data:
                 # existing ones + parsed ones
                extras.update(init_data["extra_kwargs"])
            
            init_data["extra_kwargs"] = extras
        
        return cls(**init_data)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict, flattening extra_kwargs into the root."""
        d = asdict(self)
        extras = d.pop("extra_kwargs", {})
        # Merge extras back into the main dict for a clean export
        d.update(extras) 
        return d

    # --- Magic methods for ** unpacking ---
    def keys(self):
        """Yields keys from standard fields AND keys inside extra_kwargs."""
        # 1. Yield standard fields (skipping extra_kwargs itself)
        for f in fields(self):
            if f.name == "extra_kwargs":
                continue
            yield f.name
        
        # 2. Yield keys found inside extra_kwargs
        if hasattr(self, "extra_kwargs") and isinstance(self.extra_kwargs, dict):
            yield from self.extra_kwargs.keys()

    def __getitem__(self, key):
        """Looks in attributes first, then falls back to extra_kwargs."""
        # 1. Try to get attribute normally
        if hasattr(self, key) and key != "extra_kwargs":
            return getattr(self, key)
        
        # 2. Try to get from extra_kwargs dictionary
        if hasattr(self, "extra_kwargs") and isinstance(self.extra_kwargs, dict):
            if key in self.extra_kwargs:
                return self.extra_kwargs[key]
        
        raise KeyError(key)