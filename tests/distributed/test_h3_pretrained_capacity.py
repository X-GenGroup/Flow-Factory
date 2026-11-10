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

"""Root-run released Ref2VA preparation capacity canary; no media-quality claim."""

import hashlib
import importlib.metadata
import json
import os
import resource
import traceback
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from conftest import run_gloo_ranks

MODEL_REPOSITORY = "MiniMaxAI/MiniMax-H3"
MODEL_REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"
DP_SIZE = int(os.environ.get("FF_H3_CAPACITY_DP", "2"))
TP_SIZE = int(os.environ.get("FF_H3_CAPACITY_TP", "4"))
TARGET_MATERIALIZATION = os.environ.get("FF_H3_TARGET_MATERIALIZATION", "replicated")
EFFICIENT_LOADING = os.environ.get("FF_H3_CAPACITY_EFFICIENT_LOADING") == "1"
WORLD_SIZE = DP_SIZE * TP_SIZE


def resolve_checkpoint():
    """Use an explicitly staged snapshot or the pinned ordinary Hugging Face cache."""
    explicit = os.environ.get("FF_H3_MODEL_PATH")
    if explicit:
        return Path(explicit).expanduser().resolve(strict=True)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(MODEL_REPOSITORY, revision=MODEL_REVISION, local_files_only=True))


def checkpoint_identity(root):
    """Hash the actual native root before workers allocate any GPU model state."""
    folder = root / "transformer_ref"
    index = folder / "diffusion_pytorch_model.safetensors.index.json"
    names = {"config.json", index.name, *json.loads(index.read_text())["weight_map"].values()}
    missing = sorted(name for name in names if not (folder / name).is_file())
    if missing:
        raise ValueError(
            f"H3 capacity canary requires every released transformer_ref shard: {missing}"
        )
    identity = {}
    for name in sorted(names):
        digest = hashlib.sha256()
        with (folder / name).open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        identity[name] = {"sha256": digest.hexdigest(), "bytes": (folder / name).stat().st_size}
    return identity


def capacity_config(checkpoint, result_dir):
    from flow_factory.hparams import Arguments

    return Arguments.from_dict(
        {
            "mixed_precision": "bf16",
            "parallelism": {
                "dp_shard_size": DP_SIZE,
                "tp_size": TP_SIZE,
                "cp_size": 1,
                "sequence_parallel": True,
                "target_materialization": TARGET_MATERIALIZATION,
            },
            "model": {
                "model_type": "minimax-h3-ref2va",
                "model_name_or_path": str(checkpoint),
                "target_components": ["transformer_ref"],
                "finetune_type": "lora",
                "target_modules": "default",
                "component_load_dtypes": "float32",
                "lora_rank": 2,
                "lora_alpha": 2,
            },
            "train": {
                "trainer_type": "grpo",
                "resolution": [32, 32],
                "num_frames": 124,
                "frame_rate": 24.0,
                "guidance_scale": 1.0,
                "num_inference_steps": 3,
                "group_size": 2,
                "unique_sample_num_per_epoch": 4,
                "gradient_step_per_epoch": 2,
                "gradient_accumulation_steps": "auto",
                "enable_gradient_checkpointing": True,
                "seed": 17,
            },
            "log": {"save_dir": str(result_dir), "logging_backend": "none"},
            "scheduler": {
                "dynamics_type": "Flow-SDE",
                "sde_steps": [0, 1],
                "num_sde_steps": 2,
                "seed": 17,
            },
        }
    )


def parameter_map(model):
    from flow_factory.models.variants import ComponentVariantRegistry

    return {
        ComponentVariantRegistry._canonical_prepared_parameter_name(name): param
        for name, param in model.named_parameters()
    }


def capacity_inputs(rank):
    """Keep native AV rows identical within each TP replica."""
    from h3_cp_fixture import model_inputs

    replica = rank // TP_SIZE
    values = model_inputs("ref2va", replica=replica, odd=False)
    generator = torch.Generator().manual_seed(100 + replica)
    values["encoder_hidden_states"] = torch.randn(1, 4, 5120, generator=generator)
    return values


def worker(rank, store_path, result_dir, options):
    os.environ.update(
        RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(WORLD_SIZE), ACCELERATE_USE_FSDP="true"
    )
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group(
        "nccl",
        store=dist.FileStore(store_path, WORLD_SIZE),
        rank=rank,
        world_size=WORLD_SIZE,
        timeout=timedelta(seconds=1800),
    )
    result = {
        "rank": rank,
        "claim": "released_ref2va_tp4_lora_preparation_capacity_and_bounded_native_step",
        "target_materialization": TARGET_MATERIALIZATION,
        "cpu_ram_efficient_loading": EFFICIENT_LOADING,
        "dp_rank": rank // TP_SIZE,
        "tp_rank": rank % TP_SIZE,
        "memory": [],
    }
    output = Path(result_dir, f"rank-{rank}.json")

    def record(stage):
        torch.cuda.synchronize()
        result["memory"].append(
            {
                "stage": stage,
                "allocated": torch.cuda.memory_allocated(),
                "reserved": torch.cuda.memory_reserved(),
                "peak_allocated": torch.cuda.max_memory_allocated(),
                "peak_reserved": torch.cuda.max_memory_reserved(),
                "host_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            }
        )
        output.write_text(json.dumps(result, indent=2))
        torch.cuda.reset_peak_memory_stats()

    try:
        from accelerate import FullyShardedDataParallelPlugin
        from accelerate.parallelism_config import ParallelismConfig
        from torch.distributed.tensor import DTensor, Shard

        from flow_factory.loading.backend import configure_backend_loading
        from flow_factory.loading.coordinator import ModelLoadCoordinator
        from flow_factory.models.minimax_h3.adapters import MiniMaxH3Ref2VAAdapter
        from flow_factory.models.tensor_parallel import peel_peft_wrapper
        from flow_factory.trainers.common.model_preparation import ModelPreparation
        from flow_factory.trainers.loader import Accelerator

        checkpoint = Path(options["checkpoint"])
        config = capacity_config(checkpoint, result_dir)
        accelerator = Accelerator(
            mixed_precision="bf16",
            fsdp_plugin=FullyShardedDataParallelPlugin(
                fsdp_version=2,
                auto_wrap_policy="TRANSFORMER_BASED_WRAP",
                transformer_cls_names_to_wrap=["MiniMaxH3TransformerBlock"],
                cpu_ram_efficient_loading=EFFICIENT_LOADING,
                reshard_after_forward=True,
                activation_checkpointing=True,
            ),
            parallelism_config=ParallelismConfig(
                dp_replicate_size=1, dp_shard_size=DP_SIZE, cp_size=1, tp_size=TP_SIZE, sp_size=1
            ),
        )
        configure_backend_loading(accelerator, MiniMaxH3Ref2VAAdapter)
        result["runtime"] = {
            "world_size": dist.get_world_size(),
            "rank": rank,
            "device": accelerator.device.type,
            "device_index": accelerator.device.index,
            "distributed_type": accelerator.distributed_type.name,
            "fsdp_version": accelerator.state.fsdp_plugin.fsdp_version,
            "zero_stage": None,
            "process_group_backend": dist.get_backend(),
            "cp_size": 1,
            "tp_size": TP_SIZE,
            "dependencies": {
                name: importlib.metadata.version(name)
                for name in ("torch", "diffusers", "accelerate", "peft")
            },
        }
        result["checkpoint"] = str(checkpoint)
        result["checkpoint_repository"] = MODEL_REPOSITORY
        result["checkpoint_revision"] = MODEL_REVISION
        result["weight_identity"] = options["weight_identity"]
        record("before_adapter_load")
        torch.manual_seed(123)
        adapter = MiniMaxH3Ref2VAAdapter(config, accelerator)
        module = adapter.get_component("transformer_ref")
        native = peel_peft_wrapper(module)
        assert native.config.num_attention_heads == 56
        assert native.config.attention_head_dim == 128
        assert native.config.num_layers == 50
        assert native.config.hidden_size == 5376
        assert native.config.ffn_dim == 14336
        assert native.config.time_embed_dim == 2688
        assert native.config.text_dim == 5120
        base_parameters = sum(
            parameter.numel()
            for name, parameter in native.named_parameters()
            if "lora_" not in name
        )
        assert base_parameters == 33_122_992_896
        result["base_parameters"] = base_parameters
        result["device_total_bytes"] = torch.cuda.get_device_properties(rank).total_memory
        assert all(
            param.device.type == "cpu" and param.dtype == torch.float32
            for param in module.parameters()
        )
        result["native_config"] = dict(native.config)
        result["materialized_components"] = list(
            adapter.component_runtime.materialized_component_names
        )
        assert "text_encoder" not in result["materialized_components"]
        record("cpu_adapter_loaded")
        coordinator = ModelLoadCoordinator(adapter, accelerator)
        coordinator.bootstrap_targets()
        adapter.declare_component_variants(("policy",))
        # The production backend owns H3 in-forward checkpoint installation.
        adapter.disable_gradient_checkpointing()
        preparation = ModelPreparation(coordinator, config.parallelism_args)
        module.set_attention_backend("native")
        preparation.parallelize()
        assert all(param.dtype == torch.float32 for param in module.parameters())
        local_heads = [block.attn.heads for block in native.transformer_blocks]
        assert local_heads == [56 // TP_SIZE] * 50
        result["local_attention_heads"] = local_heads
        result["original_parameter_dtype"] = "torch.float32"
        record("tp_sp_prepared_before_fsdp")
        registry = adapter.component_variant_registry
        optimizer = torch.optim.AdamW(
            [
                {"params": registry.parameters(name), "role_name": name}
                for name in registry.variant_names
            ],
            lr=0.001,
            eps=1e-4,
        )
        model, optimizer = preparation.prepare(optimizer)
        preparation.install_routes()
        result["fsdp_dim_names"] = list(accelerator.parallelism_config.fsdp_dim_names)
        placements = {}
        for name, param in parameter_map(model).items():
            if not name.endswith(("adaln_proj.linear.weight", "norm_out.linear.weight")):
                continue
            assert isinstance(param, DTensor)
            assert param.device_mesh.mesh_dim_names == ("dp_shard_cp", "tp")
            assert tuple(param.device_mesh.mesh.shape) == (DP_SIZE, TP_SIZE)
            assert param.placements == (Shard(0), Shard(1))
            assert param.numel() == param.to_local().numel() * WORLD_SIZE
            placements[name] = {
                "mesh_names": list(param.device_mesh.mesh_dim_names),
                "mesh_shape": list(param.device_mesh.mesh.shape),
                "placements": [str(value) for value in param.placements],
                "global_shape": list(param.shape),
                "local_shape": list(param.to_local().shape),
            }
        assert len(placements) == 51
        assert all(param.dtype == torch.float32 for param in model.parameters())
        result["modulation_placements"] = placements
        record("fsdp_prepared")
        values = capacity_inputs(rank)
        packed_rows = values["position_ids"].shape[0]
        assert packed_rows == 32 and packed_rows % TP_SIZE == 0
        result["packed_rows"] = packed_rows
        result["sp_local_rows"] = packed_rows // TP_SIZE
        values = {
            key: value.to(accelerator.device) if torch.is_tensor(value) else value
            for key, value in values.items()
        }
        with accelerator.autocast():
            video, audio = adapter.get_component("transformer_ref")(**values)
            loss = (
                video[:, 6:].float().square().mean() * 0.7
                + audio[:, 2:].float().square().mean() * 0.3
            )
        assert torch.isfinite(video).all() and torch.isfinite(audio).all()
        assert torch.isfinite(loss)
        accelerator.backward(loss)
        count = nonzero = 0
        before = {}
        for name, param in parameter_map(model).items():
            if not param.requires_grad:
                assert param.grad is None, name
                continue
            assert param.grad is not None, name
            gradient = param.grad.to_local() if isinstance(param.grad, DTensor) else param.grad
            assert torch.isfinite(gradient).all(), name
            count += 1
            nonzero += int(bool(gradient.count_nonzero()))
            before[name] = (
                (param.to_local() if isinstance(param, DTensor) else param).detach().clone()
            )
        assert count > 0 and nonzero > 0
        optimizer.step()
        updated = 0
        for name, param in parameter_map(model).items():
            if name in before:
                local = param.to_local() if isinstance(param, DTensor) else param
                assert torch.isfinite(local).all(), name
                updated += int(not torch.equal(local, before[name]))
        assert updated > 0
        for state in optimizer.state.values():
            for value in state.values():
                if torch.is_tensor(value):
                    local = value.to_local() if isinstance(value, DTensor) else value
                    assert torch.isfinite(local).all()
        result.update(
            loss=float(loss.detach()),
            gradient_tensors=count,
            nonzero_gradient_tensors=nonzero,
            updated_tensors=updated,
            video_shape=list(video.shape),
            audio_shape=list(audio.shape),
            status="pass",
        )
        record("one_native_step_completed")
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        output.write_text(json.dumps(result, indent=2))
        dist.destroy_process_group()


@pytest.mark.skipif(
    torch.cuda.device_count() < WORLD_SIZE,
    reason=f"H3 DP{DP_SIZE} TP{TP_SIZE} capacity requires {WORLD_SIZE} CUDA devices",
)
@pytest.mark.skipif(
    os.environ.get("FF_RUN_H3_CAPACITY") != "1",
    reason="set FF_RUN_H3_CAPACITY=1 for the released-weight capacity canary",
)
def test_released_h3_ref2va_tp4_lora_preparation_capacity(tmp_path):
    checkpoint = resolve_checkpoint()
    identity = checkpoint_identity(checkpoint)
    rows = run_gloo_ranks(
        worker,
        tmp_path,
        WORLD_SIZE,
        {"checkpoint": str(checkpoint), "weight_identity": identity},
        timeout=1800,
    )
    assert all(row.get("status") == "pass" and "error" not in row for row in rows), rows
