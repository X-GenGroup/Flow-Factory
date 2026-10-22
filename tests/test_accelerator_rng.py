"""Accelerator-agnostic RNG state handling (CUDA, Ascend NPU, XPU, ...)."""

import pytest
import torch

from flow_factory.utils.base import accelerator_rng_module, isolated_rng


def _active_accelerator_module():
    """Return the torch module of the active accelerator, or ``None``."""
    if not torch.accelerator.is_available():
        return None
    return getattr(torch, torch.accelerator.current_accelerator().type, None)


def _accelerator_device():
    """Return a device on the active accelerator that this host can allocate on."""
    device_type = torch.accelerator.current_accelerator().type
    if hasattr(torch.accelerator, "current_device_index"):
        return torch.device(device_type, torch.accelerator.current_device_index())
    return torch.device(device_type)


def test_accelerator_rng_module_follows_active_accelerator():
    module = accelerator_rng_module()
    expected = _active_accelerator_module()
    if expected is None or not hasattr(expected, "get_rng_state_all"):
        assert module is None
    else:
        assert module is expected


def test_isolated_rng_restores_accelerator_rng_state():
    if accelerator_rng_module() is None:
        pytest.skip("no accelerator exposing RNG state APIs on this host")
    device = _accelerator_device()

    torch.manual_seed(2026)
    expected = torch.randn(4, device=device)
    torch.manual_seed(2026)
    with isolated_rng(1234):
        torch.randn(4, device=device)

    assert torch.equal(expected, torch.randn(4, device=device))


def test_isolated_rng_still_isolates_cpu_rng_state():
    torch.manual_seed(2026)
    expected = torch.randn(4)
    torch.manual_seed(2026)
    with isolated_rng(1234):
        torch.randn(4)

    assert torch.equal(expected, torch.randn(4))
