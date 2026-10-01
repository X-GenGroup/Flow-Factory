"""BaseAdapter.device resolves Accelerate's index-less single-process CUDA device."""

from types import SimpleNamespace

import pytest
import torch

from flow_factory.models.abc import BaseAdapter


def _adapter_device(accelerator_device: torch.device) -> torch.device:
    stub = SimpleNamespace(accelerator=SimpleNamespace(device=accelerator_device))
    return BaseAdapter.device.fget(stub)


def test_index_less_cuda_resolves_to_current_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)

    assert _adapter_device(torch.device("cuda")) == torch.device("cuda", 3)


@pytest.mark.parametrize("device", [torch.device("cuda", 1), torch.device("cpu")])
def test_concrete_devices_pass_through(monkeypatch, device):
    def fail():
        raise AssertionError("a concrete device must not query the current CUDA device")

    monkeypatch.setattr(torch.cuda, "current_device", fail)

    assert _adapter_device(device) == device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_tensors_on_single_process_device_match_adapter_device():
    tensor = torch.zeros(1, device=torch.device("cuda"))

    assert tensor.device == _adapter_device(torch.device("cuda"))
