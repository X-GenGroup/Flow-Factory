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

"""Version-bounded Torch process policy, applied once a worker device is known."""

import warnings

import torch
from packaging.version import Version


def configure_torch_runtime(device: torch.device) -> None:
    """Avoid the measured Torch 2.11 Hopper cuDNN SDPA backward regression.

    This public Torch switch is process-wide and persists for the worker lifetime,
    including rollout, checkpoint recomputation and Diffusers CP backward replay.
    It leaves cuDNN convolutions and the other SDPA backends unchanged. An explicit
    user ``sdpa_kernel(CUDNN_ATTENTION)`` context can still override Torch's switch;
    that configuration is outside the qualified native-attention path.

    Remove this guard only with a passing output/gradient witness for the affected
    short-sequence layouts, not merely a newer cuDNN version or finite loss.
    """
    if Version(torch.__version__).release[:2] != (2, 11) or device.type != "cuda":
        return
    if torch.cuda.get_device_capability(device)[0] != 9:
        return
    if not torch.backends.cuda.cudnn_sdp_enabled():
        return
    torch.backends.cuda.enable_cudnn_sdp(False)
    warnings.warn(
        f"Torch {torch.__version__} on Hopper: disabling cuDNN SDPA for this worker "
        f"(cuDNN={torch.backends.cudnn.version()}). Its backward produced incorrect "
        "gradients in the native Flux BF16 control. Torch may still select Flash, "
        "Efficient or Math SDPA; this policy applies to both inference and training. "
        "See docs/development/validation/torch211.md.",
        RuntimeWarning,
        stacklevel=2,
    )
