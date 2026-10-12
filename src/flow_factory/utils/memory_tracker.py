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

# src/flow_factory/utils/memory_tracker.py

import gc
import sys
import threading
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, TextIO, Union

import numpy as np
import torch
from accelerate import Accelerator


class ModelMemoryTracker:
    """Memory tracker for model parameters and buffers."""

    def __init__(self, accelerator, log_file: Optional[Union[str, TextIO]] = None):
        self.accelerator = accelerator
        self.model_stats = {}
        self.log_file = log_file

    def _print(self, *args, **kwargs):
        """Print to ``log_file`` (path or stream) when set, otherwise to stdout."""
        if self.log_file:
            if isinstance(self.log_file, str):
                with open(self.log_file, "a", encoding="utf-8") as f:
                    print(*args, file=f, **kwargs)
                    f.flush()
            else:
                print(*args, file=self.log_file, **kwargs)
                self.log_file.flush()
        else:
            print(*args, **kwargs)

    def register_model(self, model, model_name: str):
        """Register ``model`` and record its parameter, buffer, and dtype statistics."""
        if not self.accelerator.is_local_main_process:
            return

        param_size = 0
        buffer_size = 0
        trainable_params = 0
        total_params = 0

        dtype_breakdown = defaultdict(int)

        for name, param in model.named_parameters():
            param_memory = param.numel() * param.element_size()
            param_size += param_memory
            total_params += param.numel()
            dtype_breakdown[str(param.dtype)] += param.numel()

            if param.requires_grad:
                trainable_params += param.numel()

        for buffer in model.buffers():
            buffer_size += buffer.numel() * buffer.element_size()

        self.model_stats[model_name] = {
            "param_size_gb": param_size / 1024**3,
            "buffer_size_gb": buffer_size / 1024**3,
            "total_size_gb": (param_size + buffer_size) / 1024**3,
            "trainable_params": trainable_params,
            "total_params": total_params,
            "trainable_ratio": trainable_params / (total_params + 1e-8) * 100,
            "dtype_breakdown": dict(dtype_breakdown),
        }

    def print_stats(self, model_name: str = None):
        """Print memory statistics for ``model_name``, or for all registered models."""
        if not self.accelerator.is_local_main_process:
            return

        models_to_print = [model_name] if model_name else list(self.model_stats.keys())

        for name in models_to_print:
            if name in self.model_stats:
                stats = self.model_stats[name]
                self._print(f"[{name}] Model Memory:")
                self._print(f"  Total: {stats['total_size_gb']:.2f}GB")
                self._print(f"  Params: {stats['param_size_gb']:.2f}GB")
                self._print(f"  Buffers: {stats['buffer_size_gb']:.2f}GB")
                self._print(
                    f"  Trainable: {stats['trainable_params']:,} ({stats['trainable_ratio']:.2f}%)"
                )
                self._print(f"  Dtype breakdown: {stats['dtype_breakdown']}")


class TensorMemoryTracker:
    """Memory tracker for named tensors, with optional accumulated statistics."""

    def __init__(
        self,
        accelerator,
        enable_accumulation: bool = True,
        log_file: Optional[Union[str, TextIO]] = None,
    ):
        self.accelerator = accelerator
        self.enable_accumulation = enable_accumulation
        self.log_file = log_file
        self.tensor_stats = defaultdict(
            lambda: {
                "current_memory_mb": 0,
                "total_memory_mb": 0,
                "count": 0,
                "avg_memory_mb": 0,
                "max_memory_mb": 0,
                "shapes": [],
                "dtypes": set(),
            }
        )
        self.lock = threading.Lock()

    def _print(self, *args, **kwargs):
        """Print to ``log_file`` (path or stream) when set, otherwise to stdout."""
        if self.log_file:
            if isinstance(self.log_file, str):
                with open(self.log_file, "a", encoding="utf-8") as f:
                    print(*args, file=f, **kwargs)
                    f.flush()
            else:
                print(*args, file=self.log_file, **kwargs)
                self.log_file.flush()
        else:
            print(*args, **kwargs)

    def track_tensor(self, tensor: torch.Tensor, name: str, stage: str = ""):
        """Record a tensor's memory under ``name``, prefixed with ``stage`` when given."""
        if not self.accelerator.is_local_main_process:
            return

        if not torch.is_tensor(tensor):
            return

        memory_mb = tensor.numel() * tensor.element_size() / 1024**2
        full_name = f"{stage}_{name}" if stage else name

        with self.lock:
            stats = self.tensor_stats[full_name]
            stats["current_memory_mb"] = memory_mb

            if self.enable_accumulation:
                stats["total_memory_mb"] += memory_mb
                stats["count"] += 1
                stats["avg_memory_mb"] = stats["total_memory_mb"] / stats["count"]
                stats["max_memory_mb"] = max(stats["max_memory_mb"], memory_mb)

            stats["shapes"].append(tuple(tensor.shape))
            stats["dtypes"].add(str(tensor.dtype))

            # Keep the 10 most recent shapes.
            if len(stats["shapes"]) > 10:
                stats["shapes"] = stats["shapes"][-10:]

    def track_tensor_dict(self, tensor_dict: Dict[str, Any], stage: str = ""):
        """Record each tensor in ``tensor_dict``, including tensors in list/tuple values."""
        for name, tensor in tensor_dict.items():
            if torch.is_tensor(tensor):
                self.track_tensor(tensor, name, stage)
            elif isinstance(tensor, (list, tuple)):
                for i, item in enumerate(tensor):
                    if torch.is_tensor(item):
                        self.track_tensor(item, f"{name}_{i}", stage)

    def track_samples(self, samples: List[Dict[str, Any]], stage: str = "samples"):
        """Record the tensor memory of a list of sample dicts, summed per key."""
        if not self.accelerator.is_local_main_process:
            return

        # Sum memory per key across samples.
        key_stats = defaultdict(
            lambda: {"total_memory_mb": 0, "count": 0, "shapes": [], "dtypes": set()}
        )

        for i, sample in enumerate(samples):
            for key, value in sample.items():
                if torch.is_tensor(value):
                    memory_mb = value.numel() * value.element_size() / 1024**2
                    key_stats[key]["total_memory_mb"] += memory_mb
                    key_stats[key]["count"] += 1
                    key_stats[key]["shapes"].append(tuple(value.shape))
                    key_stats[key]["dtypes"].add(str(value.dtype))

        # Update the accumulated statistics.
        with self.lock:
            for key, stats in key_stats.items():
                full_name = f"{stage}_{key}"
                self.tensor_stats[full_name]["current_memory_mb"] = stats["total_memory_mb"]

                if self.enable_accumulation:
                    self.tensor_stats[full_name]["total_memory_mb"] += stats["total_memory_mb"]
                    self.tensor_stats[full_name]["count"] += stats["count"]
                    if self.tensor_stats[full_name]["count"] > 0:
                        self.tensor_stats[full_name]["avg_memory_mb"] = (
                            self.tensor_stats[full_name]["total_memory_mb"]
                            / self.tensor_stats[full_name]["count"]
                        )
                    self.tensor_stats[full_name]["max_memory_mb"] = max(
                        self.tensor_stats[full_name]["max_memory_mb"], stats["total_memory_mb"]
                    )

                self.tensor_stats[full_name]["shapes"].extend(stats["shapes"])
                self.tensor_stats[full_name]["dtypes"].update(stats["dtypes"])

                # Keep the 20 most recent shapes.
                if len(self.tensor_stats[full_name]["shapes"]) > 20:
                    self.tensor_stats[full_name]["shapes"] = self.tensor_stats[full_name]["shapes"][
                        -20:
                    ]

    def print_stats(self, stage: str = None, top_k: int = None):
        """Print tensor memory statistics, largest current memory first."""
        if not self.accelerator.is_local_main_process:
            return

        # Filter by stage prefix.
        items_to_print = []
        for name, stats in self.tensor_stats.items():
            if stage is None or name.startswith(stage):
                items_to_print.append((name, stats))

        # Sort by current memory, largest first.
        items_to_print.sort(key=lambda x: x[1]["current_memory_mb"], reverse=True)

        if top_k:
            items_to_print = items_to_print[:top_k]

        if items_to_print:
            self._print(f"\n[Tensor Memory Stats{' - ' + stage if stage else ''}]:")
            total_current = sum(stats["current_memory_mb"] for _, stats in items_to_print)
            self._print(f"  Total Current Memory: {total_current:.2f}MB")

            for name, stats in items_to_print:
                self._print(f"  {name}:")
                self._print(f"    Current: {stats['current_memory_mb']:.2f}MB")
                if self.enable_accumulation and stats["count"] > 0:
                    self._print(
                        f"    Avg: {stats['avg_memory_mb']:.2f}MB, Max: {stats['max_memory_mb']:.2f}MB, Count: {stats['count']}"
                    )
                self._print(f"    Recent shapes: {list(set(stats['shapes'][-5:]))}")
                self._print(f"    Dtypes: {list(stats['dtypes'])}")

    def clear_stats(self, stage: str = None):
        """Clear the statistics whose names start with ``stage``, or all statistics."""
        with self.lock:
            if stage:
                keys_to_remove = [k for k in self.tensor_stats.keys() if k.startswith(stage)]
                for key in keys_to_remove:
                    del self.tensor_stats[key]
            else:
                self.tensor_stats.clear()


class OptimizerMemoryTracker:
    """Memory tracker for optimizer gradients and states."""

    def __init__(self, accelerator, log_file: Optional[Union[str, TextIO]] = None):
        self.accelerator = accelerator
        self.optimizer_stats = {}
        self.log_file = log_file

    def _print(self, *args, **kwargs):
        """Print to ``log_file`` (path or stream) when set, otherwise to stdout."""
        if self.log_file:
            if isinstance(self.log_file, str):
                with open(self.log_file, "a", encoding="utf-8") as f:
                    print(*args, file=f, **kwargs)
                    f.flush()
            else:
                print(*args, file=self.log_file, **kwargs)
                self.log_file.flush()
        else:
            print(*args, **kwargs)

    def track_optimizer(self, optimizer, name: str = "optimizer"):
        """Record the gradient and state memory of ``optimizer`` under ``name``."""
        if not self.accelerator.is_local_main_process:
            return

        grad_memory = 0
        state_memory = 0
        param_count = 0

        for group in optimizer.param_groups:
            for param in group["params"]:
                param_count += 1

                # Gradient memory
                if param.grad is not None:
                    grad_memory += param.grad.numel() * param.grad.element_size()

                # Optimizer state memory
                if param in optimizer.state:
                    state = optimizer.state[param]
                    for key, value in state.items():
                        if torch.is_tensor(value):
                            state_memory += value.numel() * value.element_size()

        self.optimizer_stats[name] = {
            "grad_memory_gb": grad_memory / 1024**3,
            "state_memory_gb": state_memory / 1024**3,
            "total_memory_gb": (grad_memory + state_memory) / 1024**3,
            "param_count": param_count,
        }

    def print_stats(self, name: str = None):
        """Print memory statistics for optimizer ``name``, or for all tracked optimizers."""
        if not self.accelerator.is_local_main_process:
            return

        optimizers_to_print = [name] if name else list(self.optimizer_stats.keys())

        for opt_name in optimizers_to_print:
            if opt_name in self.optimizer_stats:
                stats = self.optimizer_stats[opt_name]
                self._print(f"[{opt_name}] Optimizer Memory:")
                self._print(f"  Total: {stats['total_memory_gb']:.2f}GB")
                self._print(f"  Gradients: {stats['grad_memory_gb']:.2f}GB")
                self._print(f"  States: {stats['state_memory_gb']:.2f}GB")
                self._print(f"  Param Count: {stats['param_count']:,}")


class GPUMemoryTracker:
    """Tracker of overall CUDA memory usage across named stage snapshots."""

    def __init__(self, accelerator, log_file: Optional[Union[str, TextIO]] = None):
        self.accelerator = accelerator
        self.memory_history = []
        self.baseline_memory = None
        self.last_snapshot = None
        self.log_file = log_file

    def _print(self, *args, **kwargs):
        """Print to ``log_file`` (path or stream) when set, otherwise to stdout."""
        if self.log_file:
            if isinstance(self.log_file, str):
                with open(self.log_file, "a", encoding="utf-8") as f:
                    print(*args, file=f, **kwargs)
                    f.flush()
            else:
                print(*args, file=self.log_file, **kwargs)
                self.log_file.flush()
        else:
            print(*args, **kwargs)

    def snapshot(self, stage_name: str):
        """Empty the CUDA cache and record a snapshot of the current memory usage.

        Returns:
            ``(snapshot, increase)``, where ``increase`` is the allocated GB relative to the
            previous snapshot (or to zero for the first one); ``(None, 0)`` on processes that
            are not the local main process.
        """
        if not self.accelerator.is_local_main_process:
            return None, 0

        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3

        snapshot = {
            "stage": stage_name,
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "timestamp": len(self.memory_history),
        }

        self.memory_history.append(snapshot)

        if self.baseline_memory is None:
            self.baseline_memory = allocated

        increase = snapshot["allocated_gb"] - (
            self.last_snapshot["allocated_gb"] if self.last_snapshot else 0
        )
        self.last_snapshot = snapshot

        return snapshot, increase

    def print_current(self, stage_name: str):
        """Take a snapshot and print the current CUDA memory usage."""
        snapshot, increase = self.snapshot(stage_name)
        if snapshot and self.accelerator.is_local_main_process:
            increase_to_base_line = (
                snapshot["allocated_gb"] - self.baseline_memory if self.baseline_memory else 0
            )
            self._print(
                f"[{stage_name}] GPU Memory Usage:"
                f"    Allocated: {snapshot['allocated_gb']:.2f}GB, "
                f"    Reserved: {snapshot['reserved_gb']:.2f}GB, "
                f"    Increase: {increase:+.2f}GB"
                f"    Increase to Baseline: {increase_to_base_line:+.2f}GB"
            )

    def print_summary(self):
        """Print peak and baseline memory and the largest per-stage memory increases."""
        if not self.accelerator.is_local_main_process or not self.memory_history:
            return

        self._print("\n=== GPU Memory Summary ===")
        allocated_gbs = np.array([s["allocated_gb"] for s in self.memory_history])
        max_allocated = np.max(allocated_gbs)
        max_reserved = np.max([s["reserved_gb"] for s in self.memory_history])

        self._print(f"Peak Allocated: {max_allocated:.2f}GB")
        self._print(f"Peak Reserved: {max_reserved:.2f}GB")
        self._print(f"Baseline Memory: {self.baseline_memory:.2f}GB")

        if len(self.memory_history) < 2:
            self._print("=========================\n")
            return

        # Report the top-k stages with the largest memory increase.
        top_k = 3
        if len(self.memory_history) < top_k:
            top_k = len(self.memory_history)

        total_increase = allocated_gbs[-1] - allocated_gbs[0]
        self._print(f"Total Memory Increase: {total_increase:+.2f}GB")

        # Memory increase of each stage over the previous snapshot.
        increases = np.diff(allocated_gbs)
        stages = [s["stage"] for s in self.memory_history[1:]]

        # Indices of the top_k largest increases, largest first.
        top_indices = np.argsort(increases)[::-1][:top_k]

        self._print("Top Memory Increases:")
        for i, idx in enumerate(top_indices, 1):
            if idx < len(stages) and increases[idx] > 0:
                self._print(f"  #{i}: {increases[idx]:.2f}GB at stage '{stages[idx]}'")

        self._print("=========================\n")

    def cleanup(self):
        """Run garbage collection and empty the CUDA cache on the local main process."""
        if self.accelerator.is_local_main_process:
            gc.collect()
            torch.cuda.empty_cache()


class MemoryProfiler:
    """Combined profiler over the model, tensor, optimizer, and CUDA memory trackers.

    Output goes to stdout, or to ``log_file``: a path opened in append mode, or an open stream.

    Example:
        >>> profiler = MemoryProfiler(accelerator, log_file="/path/to/memory_log.txt")
        >>> profiler.set_log_file("/path/to/other_log.txt")
        >>> with redirect_memory_logs(profiler, "/path/to/temp_log.txt"):
        ...     profiler.print_full_report()
    """

    def __init__(
        self,
        accelerator,
        enable_tensor_accumulation: bool = True,
        log_file: Optional[Union[str, TextIO]] = None,
    ):
        self.accelerator = accelerator
        self.log_file = log_file
        self.model_tracker = ModelMemoryTracker(accelerator, log_file)
        self.tensor_tracker = TensorMemoryTracker(accelerator, enable_tensor_accumulation, log_file)
        self.optimizer_tracker = OptimizerMemoryTracker(accelerator, log_file)
        self.gpu_tracker = GPUMemoryTracker(accelerator, log_file)

    def _print(self, *args, **kwargs):
        """Print to ``log_file`` (path or stream) when set, otherwise to stdout."""
        if self.log_file:
            if isinstance(self.log_file, str):
                with open(self.log_file, "a", encoding="utf-8") as f:
                    print(*args, file=f, **kwargs)
                    f.flush()
            else:
                print(*args, file=self.log_file, **kwargs)
                self.log_file.flush()
        else:
            print(*args, **kwargs)

    def register_model(self, model, model_name: str):
        """Register a model with the model tracker."""
        self.model_tracker.register_model(model, model_name)

    def track_optimizer(self, optimizer, name: str = "optimizer"):
        """Record optimizer memory with the optimizer tracker."""
        self.optimizer_tracker.track_optimizer(optimizer, name)

    def track_tensors(self, tensor_dict: Dict[str, Any], stage: str = ""):
        """Record the tensors in ``tensor_dict`` with the tensor tracker."""
        self.tensor_tracker.track_tensor_dict(tensor_dict, stage)

    def track_samples(self, samples: List[Dict[str, Any]], stage: str = "samples"):
        """Record the per-key tensor memory of ``samples`` with the tensor tracker."""
        self.tensor_tracker.track_samples(samples, stage)

    def snapshot(self, stage_name: str):
        """Take a CUDA memory snapshot and print it."""
        self.gpu_tracker.print_current(stage_name)

    def print_full_report(self, stage: str = None):
        """Print the model, optimizer, tensor, and CUDA memory reports."""
        if not self.accelerator.is_local_main_process:
            return

        self._print(f"\n{'='*50}")
        self._print(f"Memory Report{' - ' + stage if stage else ''}")
        self._print(f"{'='*50}")

        self.model_tracker.print_stats()
        self.optimizer_tracker.print_stats()
        self.tensor_tracker.print_stats(stage, top_k=15)
        self.gpu_tracker.print_summary()

    def cleanup_and_snapshot(self, stage_name: str):
        """Clean up memory, then take a snapshot named ``{stage_name}_after_cleanup``."""
        self.gpu_tracker.cleanup()
        self.snapshot(f"{stage_name}_after_cleanup")

    def set_log_file(self, log_file: Optional[Union[str, TextIO]]):
        """Set the log destination of the profiler and all of its trackers."""
        self.log_file = log_file
        self.model_tracker.log_file = log_file
        self.tensor_tracker.log_file = log_file
        self.optimizer_tracker.log_file = log_file
        self.gpu_tracker.log_file = log_file


@contextmanager
def redirect_memory_logs(profiler: MemoryProfiler, log_file: Union[str, TextIO]):
    """Temporarily redirect ``profiler`` output to ``log_file``, restoring it on exit."""
    original_log_file = profiler.log_file
    try:
        profiler.set_log_file(log_file)
        yield
    finally:
        profiler.set_log_file(original_log_file)
