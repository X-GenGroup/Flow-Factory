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
from collections import defaultdict
from typing import Dict, List, Optional, Union, Any, TextIO
import threading
import torch
from accelerate import Accelerator
import sys
from contextlib import contextmanager
import numpy as np

class ModelMemoryTracker:
    """追踪模型参数的显存使用情况"""
    
    def __init__(self, accelerator, log_file: Optional[Union[str, TextIO]] = None):
        self.accelerator = accelerator
        self.model_stats = {}
        self.log_file = log_file
        
    def _print(self, *args, **kwargs):
        """内部打印方法，支持文件重定向"""
        if self.log_file:
            if isinstance(self.log_file, str):
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    print(*args, file=f, **kwargs)
                    f.flush()
            else:
                print(*args, file=self.log_file, **kwargs)
                self.log_file.flush()
        else:
            print(*args, **kwargs)
        
    def register_model(self, model, model_name: str):
        """注册模型并记录其内存使用"""
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
            'param_size_gb': param_size / 1024**3,
            'buffer_size_gb': buffer_size / 1024**3,
            'total_size_gb': (param_size + buffer_size) / 1024**3,
            'trainable_params': trainable_params,
            'total_params': total_params,
            'trainable_ratio': trainable_params / (total_params + 1e-8) * 100,
            'dtype_breakdown': dict(dtype_breakdown)
        }
        
    def print_stats(self, model_name: str = None):
        """打印模型内存统计"""
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
                self._print(f"  Trainable: {stats['trainable_params']:,} ({stats['trainable_ratio']:.2f}%)")
                self._print(f"  Dtype breakdown: {stats['dtype_breakdown']}")

class TensorMemoryTracker:
    """追踪自定义张量的显存使用情况，支持累加统计"""
    
    def __init__(self, accelerator, enable_accumulation: bool = True, log_file: Optional[Union[str, TextIO]] = None):
        self.accelerator = accelerator
        self.enable_accumulation = enable_accumulation
        self.log_file = log_file
        self.tensor_stats = defaultdict(lambda: {
            'current_memory_mb': 0,
            'total_memory_mb': 0,
            'count': 0,
            'avg_memory_mb': 0,
            'max_memory_mb': 0,
            'shapes': [],
            'dtypes': set()
        })
        self.lock = threading.Lock()
        
    def _print(self, *args, **kwargs):
        """内部打印方法，支持文件重定向"""
        if self.log_file:
            if isinstance(self.log_file, str):
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    print(*args, file=f, **kwargs)
                    f.flush()
            else:
                print(*args, file=self.log_file, **kwargs)
                self.log_file.flush()
        else:
            print(*args, **kwargs)
        
    def track_tensor(self, tensor: torch.Tensor, name: str, stage: str = ""):
        """记录单个张量"""
        if not self.accelerator.is_local_main_process:
            return
            
        if not torch.is_tensor(tensor):
            return
            
        memory_mb = tensor.numel() * tensor.element_size() / 1024**2
        full_name = f"{stage}_{name}" if stage else name
        
        with self.lock:
            stats = self.tensor_stats[full_name]
            stats['current_memory_mb'] = memory_mb
            
            if self.enable_accumulation:
                stats['total_memory_mb'] += memory_mb
                stats['count'] += 1
                stats['avg_memory_mb'] = stats['total_memory_mb'] / stats['count']
                stats['max_memory_mb'] = max(stats['max_memory_mb'], memory_mb)
                
            stats['shapes'].append(tuple(tensor.shape))
            stats['dtypes'].add(str(tensor.dtype))
            
            # 保持最近的10个shape记录
            if len(stats['shapes']) > 10:
                stats['shapes'] = stats['shapes'][-10:]
    
    def track_tensor_dict(self, tensor_dict: Dict[str, Any], stage: str = ""):
        """记录张量字典"""
        for name, tensor in tensor_dict.items():
            if torch.is_tensor(tensor):
                self.track_tensor(tensor, name, stage)
            elif isinstance(tensor, (list, tuple)):
                for i, item in enumerate(tensor):
                    if torch.is_tensor(item):
                        self.track_tensor(item, f"{name}_{i}", stage)
    
    def track_samples(self, samples: List[Dict[str, Any]], stage: str = "samples"):
        """专门用于追踪samples列表中的张量, 按key累加"""
        if not self.accelerator.is_local_main_process:
            return
            
        # 按key分组统计
        key_stats = defaultdict(lambda: {'total_memory_mb': 0, 'count': 0, 'shapes': [], 'dtypes': set()})
        
        for i, sample in enumerate(samples):
            for key, value in sample.items():
                if torch.is_tensor(value):
                    memory_mb = value.numel() * value.element_size() / 1024**2
                    key_stats[key]['total_memory_mb'] += memory_mb
                    key_stats[key]['count'] += 1
                    key_stats[key]['shapes'].append(tuple(value.shape))
                    key_stats[key]['dtypes'].add(str(value.dtype))
        
        # 更新累计统计
        with self.lock:
            for key, stats in key_stats.items():
                full_name = f"{stage}_{key}"
                self.tensor_stats[full_name]['current_memory_mb'] = stats['total_memory_mb']
                
                if self.enable_accumulation:
                    self.tensor_stats[full_name]['total_memory_mb'] += stats['total_memory_mb']
                    self.tensor_stats[full_name]['count'] += stats['count']
                    if self.tensor_stats[full_name]['count'] > 0:
                        self.tensor_stats[full_name]['avg_memory_mb'] = (
                            self.tensor_stats[full_name]['total_memory_mb'] / 
                            self.tensor_stats[full_name]['count']
                        )
                    self.tensor_stats[full_name]['max_memory_mb'] = max(
                        self.tensor_stats[full_name]['max_memory_mb'], 
                        stats['total_memory_mb']
                    )
                
                self.tensor_stats[full_name]['shapes'].extend(stats['shapes'])
                self.tensor_stats[full_name]['dtypes'].update(stats['dtypes'])
                
                # 保持最近的20个shape记录
                if len(self.tensor_stats[full_name]['shapes']) > 20:
                    self.tensor_stats[full_name]['shapes'] = self.tensor_stats[full_name]['shapes'][-20:]
    
    def print_stats(self, stage: str = None, top_k: int = None):
        """打印张量内存统计"""
        if not self.accelerator.is_local_main_process:
            return
            
        # 过滤指定stage
        items_to_print = []
        for name, stats in self.tensor_stats.items():
            if stage is None or name.startswith(stage):
                items_to_print.append((name, stats))
        
        # 按当前内存使用量排序
        items_to_print.sort(key=lambda x: x[1]['current_memory_mb'], reverse=True)
        
        if top_k:
            items_to_print = items_to_print[:top_k]
            
        if items_to_print:
            self._print(f"\n[Tensor Memory Stats{' - ' + stage if stage else ''}]:")
            total_current = sum(stats['current_memory_mb'] for _, stats in items_to_print)
            self._print(f"  Total Current Memory: {total_current:.2f}MB")
            
            for name, stats in items_to_print:
                self._print(f"  {name}:")
                self._print(f"    Current: {stats['current_memory_mb']:.2f}MB")
                if self.enable_accumulation and stats['count'] > 0:
                    self._print(f"    Avg: {stats['avg_memory_mb']:.2f}MB, Max: {stats['max_memory_mb']:.2f}MB, Count: {stats['count']}")
                self._print(f"    Recent shapes: {list(set(stats['shapes'][-5:]))}")
                self._print(f"    Dtypes: {list(stats['dtypes'])}")
    
    def clear_stats(self, stage: str = None):
        """清除统计数据"""
        with self.lock:
            if stage:
                keys_to_remove = [k for k in self.tensor_stats.keys() if k.startswith(stage)]
                for key in keys_to_remove:
                    del self.tensor_stats[key]
            else:
                self.tensor_stats.clear()

class OptimizerMemoryTracker:
    """追踪优化器状态的显存使用"""
    
    def __init__(self, accelerator, log_file: Optional[Union[str, TextIO]] = None):
        self.accelerator = accelerator
        self.optimizer_stats = {}
        self.log_file = log_file
        
    def _print(self, *args, **kwargs):
        """内部打印方法，支持文件重定向"""
        if self.log_file:
            if isinstance(self.log_file, str):
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    print(*args, file=f, **kwargs)
                    f.flush()
            else:
                print(*args, file=self.log_file, **kwargs)
                self.log_file.flush()
        else:
            print(*args, **kwargs)
        
    def track_optimizer(self, optimizer, name: str = "optimizer"):
        """记录优化器内存使用"""
        if not self.accelerator.is_local_main_process:
            return
            
        grad_memory = 0
        state_memory = 0
        param_count = 0
        
        for group in optimizer.param_groups:
            for param in group['params']:
                param_count += 1
                
                # 计算梯度内存
                if param.grad is not None:
                    grad_memory += param.grad.numel() * param.grad.element_size()
                
                # 计算优化器状态内存
                if param in optimizer.state:
                    state = optimizer.state[param]
                    for key, value in state.items():
                        if torch.is_tensor(value):
                            state_memory += value.numel() * value.element_size()
        
        self.optimizer_stats[name] = {
            'grad_memory_gb': grad_memory / 1024**3,
            'state_memory_gb': state_memory / 1024**3,
            'total_memory_gb': (grad_memory + state_memory) / 1024**3,
            'param_count': param_count
        }
    
    def print_stats(self, name: str = None):
        """打印优化器内存统计"""
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
    """GPU显存总体使用情况追踪"""
    
    def __init__(self, accelerator, log_file: Optional[Union[str, TextIO]] = None):
        self.accelerator = accelerator
        self.memory_history = []
        self.baseline_memory = None
        self.last_snapshot = None
        self.log_file = log_file
        
    def _print(self, *args, **kwargs):
        """内部打印方法，支持文件重定向"""
        if self.log_file:
            if isinstance(self.log_file, str):
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    print(*args, file=f, **kwargs)
                    f.flush()
            else:
                print(*args, file=self.log_file, **kwargs)
                self.log_file.flush()
        else:
            print(*args, **kwargs)
        
    def snapshot(self, stage_name: str):
        """记录当前显存使用快照"""
        if not self.accelerator.is_local_main_process:
            return None, 0
            
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        
        snapshot = {
            'stage': stage_name,
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'timestamp': len(self.memory_history)
        }
        
        self.memory_history.append(snapshot)
        
        if self.baseline_memory is None:
            self.baseline_memory = allocated

        increase = snapshot['allocated_gb'] - (self.last_snapshot['allocated_gb'] if self.last_snapshot else 0)
        self.last_snapshot = snapshot

        return snapshot, increase

    def print_current(self, stage_name: str):
        """打印当前显存使用"""
        snapshot, increase = self.snapshot(stage_name)
        if snapshot and self.accelerator.is_local_main_process:
            # increase = snapshot['allocated_gb'] - self.baseline_memory if self.baseline_memory else 0
            increase_to_base_line = snapshot['allocated_gb'] - self.baseline_memory if self.baseline_memory else 0
            self._print(f"[{stage_name}] GPU Memory Usage:"
                       f"    Allocated: {snapshot['allocated_gb']:.2f}GB, "
                       f"    Reserved: {snapshot['reserved_gb']:.2f}GB, "
                       f"    Increase: {increase:+.2f}GB"
                       f"    Increase to Baseline: {increase_to_base_line:+.2f}GB"
                       )
    
    def print_summary(self):
        """打印内存使用总结"""
        if not self.accelerator.is_local_main_process or not self.memory_history:
            return
            
        self._print("\n=== GPU Memory Summary ===")
        allocated_gbs = np.array([s['allocated_gb'] for s in self.memory_history])
        max_allocated = np.max(allocated_gbs)
        max_reserved = np.max([s['reserved_gb'] for s in self.memory_history])
        
        self._print(f"Peak Allocated: {max_allocated:.2f}GB")
        self._print(f"Peak Reserved: {max_reserved:.2f}GB")
        self._print(f"Baseline Memory: {self.baseline_memory:.2f}GB")
        
        if len(self.memory_history) < 2:
            self._print("=========================\n")
            return
        
        # 显示top-k最大内存增长的阶段
        top_k = 3
        if len(self.memory_history) < top_k:
            top_k = len(self.memory_history)

        total_increase = allocated_gbs[-1] - allocated_gbs[0]
        self._print(f"Total Memory Increase: {total_increase:+.2f}GB")
        
        # 计算每个阶段的内存增长
        increases = np.diff(allocated_gbs)
        stages = [s['stage'] for s in self.memory_history[1:]]
        
        # 找到top_k最大增长
        top_indices = np.argsort(increases)[::-1][:top_k]  # 降序排列，取前top_k个
        
        self._print("Top Memory Increases:")
        for i, idx in enumerate(top_indices, 1):
            if idx < len(stages) and increases[idx] > 0:
                self._print(f"  #{i}: {increases[idx]:.2f}GB at stage '{stages[idx]}'")

        self._print("=========================\n")
    
    def cleanup(self):
        """执行内存清理"""
        if self.accelerator.is_local_main_process:
            gc.collect()
            torch.cuda.empty_cache()

class MemoryProfiler:
    """综合内存分析器"""
    
    def __init__(self, accelerator, enable_tensor_accumulation: bool = True, log_file: Optional[Union[str, TextIO]] = None):
        self.accelerator = accelerator
        self.log_file = log_file
        self.model_tracker = ModelMemoryTracker(accelerator, log_file)
        self.tensor_tracker = TensorMemoryTracker(accelerator, enable_tensor_accumulation, log_file)
        self.optimizer_tracker = OptimizerMemoryTracker(accelerator, log_file)
        self.gpu_tracker = GPUMemoryTracker(accelerator, log_file)
        
    def _print(self, *args, **kwargs):
        """内部打印方法，支持文件重定向"""
        if self.log_file:
            if isinstance(self.log_file, str):
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    print(*args, file=f, **kwargs)
                    f.flush()
            else:
                print(*args, file=self.log_file, **kwargs)
                self.log_file.flush()
        else:
            print(*args, **kwargs)
        
    def register_model(self, model, model_name: str):
        """注册模型"""
        self.model_tracker.register_model(model, model_name)
        
    def track_optimizer(self, optimizer, name: str = "optimizer"):
        """追踪优化器"""
        self.optimizer_tracker.track_optimizer(optimizer, name)
        
    def track_tensors(self, tensor_dict: Dict[str, Any], stage: str = ""):
        """追踪张量字典"""
        self.tensor_tracker.track_tensor_dict(tensor_dict, stage)
        
    def track_samples(self, samples: List[Dict[str, Any]], stage: str = "samples"):
        """追踪samples数据"""
        self.tensor_tracker.track_samples(samples, stage)
        
    def snapshot(self, stage_name: str):
        """记录当前状态快照"""
        self.gpu_tracker.print_current(stage_name)
        
    def print_full_report(self, stage: str = None):
        """打印完整报告"""
        if not self.accelerator.is_local_main_process:
            return
            
        self._print(f"\n{'='*50}")
        self._print(f"Memory Report{' - ' + stage if stage else ''}")
        self._print(f"{'='*50}")
        
        self.model_tracker.print_stats()
        self.optimizer_tracker.print_stats()
        self.tensor_tracker.print_stats(stage, top_k=15)  # 显示top 15张量
        self.gpu_tracker.print_summary()
        
    def cleanup_and_snapshot(self, stage_name: str):
        """清理内存并记录快照"""
        self.gpu_tracker.cleanup()
        self.snapshot(f"{stage_name}_after_cleanup")
        
    def set_log_file(self, log_file: Optional[Union[str, TextIO]]):
        """设置日志文件"""
        self.log_file = log_file
        self.model_tracker.log_file = log_file
        self.tensor_tracker.log_file = log_file
        self.optimizer_tracker.log_file = log_file
        self.gpu_tracker.log_file = log_file

# 上下文管理器，用于临时重定向输出
@contextmanager
def redirect_memory_logs(profiler: MemoryProfiler, log_file: Union[str, TextIO]):
    """临时重定向内存分析器的输出到指定文件"""
    original_log_file = profiler.log_file
    try:
        profiler.set_log_file(log_file)
        yield
    finally:
        profiler.set_log_file(original_log_file)


# def usage_examples():
#     # 方式1：初始化时指定日志文件
#     profiler = MemoryProfiler(accelerator, log_file="/path/to/memory_log.txt")

#     # 方式2：使用文件对象
#     with open("/path/to/memory_log.txt", "w") as f:
#         profiler = MemoryProfiler(accelerator, log_file=f)
#         profiler.print_full_report()

#     # 方式3：运行时设置
#     profiler = MemoryProfiler(accelerator)
#     profiler.set_log_file("/path/to/memory_log.txt")

#     # 方式4：使用上下文管理器临时重定向
#     with redirect_memory_logs(profiler, "/path/to/temp_log.txt"):
#         profiler.print_full_report()