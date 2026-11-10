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

from pathlib import Path

import torch
import torchaudio

from flow_factory.utils.audio import save_audio


def test_audio_codec_roundtrip_and_feature_gradients(tmp_path: Path):
    """Native Flux BF16 gradients must remain valid after Torch runtime setup."""


import contextlib

import pytest
import torch
from flux_sft_fixture import flux_inputs, flux_lora, flux_transformer
from torch.nn.attention import SDPBackend, sdpa_kernel

from flow_factory.trainers.loader import Accelerator


def test_cuda_native_attention_gradients_after_runtime_setup():
    if not torch.cuda.is_available():
        pytest.skip("CUDA attention qualification requires one GPU")
    accelerator = Accelerator(mixed_precision="no")
    device = accelerator.device
    results = []
    for backend in (SDPBackend.FLASH_ATTENTION, None):
        model = flux_lora(flux_transformer()).to(device)
        model.set_attention_backend("native")
        inputs = flux_inputs(0, 0, device)
        context = sdpa_kernel(backend) if backend is not None else contextlib.nullcontext()
        with context:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(**inputs, return_dict=False)[0]
                loss = output.float().square().mean()
            loss.backward()
        gradients = {
            name: param.grad.detach().cpu()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        assert gradients and all(torch.isfinite(value).all() for value in gradients.values())
        results.append({"output": output.detach().cpu(), "gradients": gradients})
    reference, native = results
    # Identical kernels/operands avoid compounding a different BF16 rounding
    # trajectory through the full network. Independent FP64 SDPA math is below.
    torch.testing.assert_close(native["output"], reference["output"], rtol=0, atol=0)
    assert native["gradients"].keys() == reference["gradients"].keys()
    for name in reference["gradients"]:
        torch.testing.assert_close(
            native["gradients"][name], reference["gradients"][name], rtol=0, atol=0, msg=name
        )


def test_cuda_native_sdpa_derivatives_against_fp64():
    if not torch.cuda.is_available():
        pytest.skip("CUDA attention qualification requires one GPU")
    device = Accelerator(mixed_precision="no").device
    generator = torch.Generator(device=device).manual_seed(11)
    # BSHD storage with a BHSD view matches native Diffusers attention inputs.
    operands = [
        torch.randn(1, 24, 4, 8, generator=generator, device=device, dtype=torch.bfloat16)
        .transpose(1, 2)
        .requires_grad_()
        for _ in range(3)
    ]
    upstream = torch.randn(
        1, 24, 4, 8, generator=generator, device=device, dtype=torch.bfloat16
    ).transpose(1, 2)
    expected_inputs = [x.detach().double().requires_grad_() for x in operands]
    with sdpa_kernel(SDPBackend.MATH):
        expected = torch.nn.functional.scaled_dot_product_attention(*expected_inputs)
    expected_gradients = torch.autograd.grad(expected, expected_inputs, upstream.double())
    actual = torch.nn.functional.scaled_dot_product_attention(*operands)
    gradients = torch.autograd.grad(actual, operands, upstream)
    # A fused BF16 operator rounds its output/derivatives. Bound each element by
    # two BF16 quanta at peak magnitude and the complete signal by two quanta.
    # This operator-local oracle does not accumulate rounding through a model.
    for value, reference in zip((actual, *gradients), (expected, *expected_gradients)):
        value = value.double()
        assert torch.isfinite(value).all() and torch.count_nonzero(value)
        quantum = torch.finfo(torch.bfloat16).eps
        torch.testing.assert_close(
            value, reference, rtol=0, atol=2 * quantum * reference.abs().max().item()
        )
        relative_error = (value - reference).norm() / reference.norm()
        assert relative_error <= 2 * quantum, relative_error
        for factor in (0, 0.5, 2):
            assert (factor * value - reference).norm() / reference.norm() > 2 * quantum
