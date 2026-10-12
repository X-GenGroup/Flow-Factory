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

"""Build one optimizer root from per-variant optimizer configurations."""

from typing import Any, Dict, List, Sequence, Tuple

import torch

from ..hparams.optimizer_args import (
    AdamWOptimizerArguments,
    MuonOptimizerArguments,
    OptimizerArguments,
)
from .composite import CompositeOptimizer

# Muon orthogonalizes matrices. Anything else - biases, normalization scales,
# 1D embeddings - is rejected by torch.optim.Muon and goes to the AdamW half.
MUON_MINIMUM_DIMENSIONS = 2


def validate_muon_available() -> None:
    """Require the optional PyTorch Muon implementation.

    Raises:
        ValueError: If this PyTorch build does not expose ``torch.optim.Muon``.
    """
    if not hasattr(torch.optim, "Muon"):
        raise ValueError(
            f"torch.optim.Muon is unavailable in PyTorch {torch.__version__}. Install a "
            "PyTorch build that provides Muon (standard releases include it from 2.9), "
            "or select the adamw optimizer."
        )


def split_muon_parameters(
    parameters: Sequence[torch.nn.Parameter],
) -> Tuple[List[torch.nn.Parameter], List[torch.nn.Parameter]]:
    """Split parameters into the Muon-eligible matrices and the AdamW remainder.

    Args:
        parameters: Parameters owned by one variant.

    Returns:
        The matrix parameters and everything else, in the input order.
    """
    matrices = [p for p in parameters if p.ndim == MUON_MINIMUM_DIMENSIONS]
    remainder = [p for p in parameters if p.ndim != MUON_MINIMUM_DIMENSIONS]
    return matrices, remainder


def _adamw_group(
    config: OptimizerArguments,
    parameters: Sequence[torch.nn.Parameter],
    *,
    betas: Tuple[float, float],
    eps: float,
) -> Dict[str, Any]:
    """Build one AdamW parameter group tagged with the variant it belongs to."""
    return {
        "params": list(parameters),
        "role_name": config.name,
        "lr": config.learning_rate,
        "betas": betas,
        "weight_decay": config.weight_decay,
        "eps": eps,
    }


def build_optimizer(
    configs: Sequence[OptimizerArguments],
    parameters_by_name: Dict[str, Sequence[torch.nn.Parameter]],
) -> torch.optim.Optimizer:
    """Build the single optimizer root covering every configured variant.

    All-AdamW runs, the common case, produce one ``torch.optim.AdamW``. As soon as one
    variant asks for Muon the result is a :class:`CompositeOptimizer`, because
    ``torch.optim.Muon`` accepts only matrices and the remaining parameters still need
    AdamW.

    Args:
        configs: One configuration per trainable variant, in optimizer-group order.
        parameters_by_name: Parameters owned by each configured variant.

    Returns:
        One optimizer whose ``param_groups`` carry a ``role_name`` per group.

    Raises:
        ValueError: If a configuration has no parameters, uses an unknown optimizer type,
            selects Muon without matrix parameters, or the current PyTorch build lacks
            ``torch.optim.Muon``.
    """
    adamw_groups: List[Dict[str, Any]] = []
    muon_groups: List[Dict[str, Any]] = []

    for config in configs:
        parameters = list(parameters_by_name.get(config.name, ()))
        if not parameters:
            raise ValueError(f"expected optimizer {config.name!r} to own parameters, received none")
        if isinstance(config, MuonOptimizerArguments):
            matrices, remainder = split_muon_parameters(parameters)
            if not matrices:
                raise ValueError(
                    f"expected variant {config.name!r} to own at least one matrix parameter "
                    "for Muon, received none; use the adamw optimizer instead"
                )
            muon_groups.append(
                {
                    "params": matrices,
                    "role_name": config.name,
                    "lr": config.learning_rate,
                    "weight_decay": config.weight_decay,
                    "momentum": config.momentum,
                    "nesterov": config.nesterov,
                    "ns_coefficients": config.ns_coefficients,
                    "ns_steps": config.ns_steps,
                    "eps": config.eps,
                    "adjust_lr_fn": config.adjust_lr_fn,
                }
            )
            if remainder:
                adamw_groups.append(
                    _adamw_group(
                        config,
                        remainder,
                        betas=config.fallback_betas,
                        eps=config.fallback_eps,
                    )
                )
            continue
        if isinstance(config, AdamWOptimizerArguments):
            adamw_groups.append(
                _adamw_group(config, parameters, betas=config.betas, eps=config.eps)
            )
            continue
        raise ValueError(
            f"expected a registered optimizer arguments type for {config.name!r}, "
            f"received {type(config).__name__}"
        )

    if not muon_groups:
        return torch.optim.AdamW(adamw_groups)

    validate_muon_available()
    children: List[torch.optim.Optimizer] = [torch.optim.Muon(muon_groups)]
    if adamw_groups:
        children.append(torch.optim.AdamW(adamw_groups))
    return CompositeOptimizer(children)


def uses_muon(configs: Sequence[OptimizerArguments]) -> bool:
    """Return whether any configuration selects Muon.

    Args:
        configs: Optimizer configurations for this run.

    Returns:
        True when at least one variant is optimized with Muon.
    """
    return any(isinstance(config, MuonOptimizerArguments) for config in configs)
