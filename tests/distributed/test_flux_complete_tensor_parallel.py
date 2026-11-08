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

"""One two-GPU SFT scenario for complete Flux1/Kontext weight TP and image SP."""

import copy
import json
import os
import subprocess
import sys
import traceback
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from conftest import run_gloo_ranks
from PIL import Image
from test_flux_sequence_parallel import _assert_added_placements
from test_model_preparation import assert_update_parity, full_value, parameter_map
from torch.distributed.device_mesh import init_device_mesh

from flow_factory.models.flux.flux1 import Flux1Adapter
from flow_factory.models.flux.flux1_kontext import Flux1KontextAdapter
from flow_factory.models.tensor_parallel import apply_tensor_parallel, resolve_parallel_plan


def _fixture(root, family):
    """Encode genuine prompt/reference/target images into fixed supervised operands."""
    from diffusers import FluxKontextPipeline, FluxPipeline
    from flow_factory.data_utils.offline_dataset import load_offline_manifest
    from flow_factory.models.flux._output import prepare_flux1_output_latents
    from flow_factory.models.runtime.classic import ClassicPipelineRuntime

    builder = "flux_sft_fixture.py" if family == "flux1" else "flux_kontext_cp_fixture.py"
    subprocess.run([sys.executable, str(Path(__file__).with_name(builder)), str(root)], check=True)
    manifest = root / "data" / "train.jsonl"
    records = [json.loads(line) for line in manifest.read_text().splitlines()][:3]
    if family == "flux1-kontext":
        for index, record in enumerate(records):
            record["input"]["media"] = [{"type": "image", "path": f"{index + 3}.png"}]
    manifest.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    records = load_offline_manifest(manifest, supervision_type="demonstration")
    pipeline_class = FluxPipeline if family == "flux1" else FluxKontextPipeline
    pipeline = pipeline_class.from_pretrained(root / "model", torch_dtype=torch.float32)
    adapter_class = Flux1Adapter if family == "flux1" else Flux1KontextAdapter
    adapter = object.__new__(adapter_class)
    adapter.pipeline = pipeline
    adapter.component_runtime = ClassicPipelineRuntime(pipeline)
    adapter._components = adapter.component_runtime.override_components
    adapter.accelerator = SimpleNamespace(device=torch.device("cpu"))
    adapter._has_warned_multi_image = False
    operands = []
    with torch.no_grad():
        for step, record in enumerate(records):
            prompt = adapter.encode_prompt(record.model_input.prompt)
            pixels = pipeline.image_processor.preprocess(
                Image.open(record.supervision.target.media[0].path), height=64, width=64
            )
            clean, target_ids = prepare_flux1_output_latents(
                adapter, pixels, torch.Generator().manual_seed(170 + step)
            )
            noise = torch.randn(clean.shape, generator=torch.Generator().manual_seed(270 + step))
            time = 0.2 + step * 0.2
            noised = (1 - time) * clean + time * noise
            image, ids = noised, target_ids
            if family == "flux1-kontext":
                condition = adapter.encode_image(
                    Image.open(record.model_input.media[0].path), condition_image_size=(32, 64)
                )
                image = torch.cat([noised, condition["image_latents"]], dim=1)
                ids = torch.cat([target_ids, condition["image_ids"][0]], dim=0)
            text = prompt["prompt_embeds"]
            operands.append(
                {
                    "inputs": {
                        "hidden_states": image,
                        "encoder_hidden_states": text,
                        "pooled_projections": prompt["pooled_prompt_embeds"],
                        "timestep": torch.tensor([time]),
                        "guidance": torch.tensor([3.5]),
                        "txt_ids": torch.zeros(text.shape[1], 3),
                        "img_ids": ids,
                        "return_dict": False,
                    },
                    "target": noise - clean,
                    "target_tokens": clean.shape[1],
                    "condition_images": len(record.model_input.media),
                }
            )
    torch.save(operands, root / "operands.pt")
    return operands


def _sft_worker(rank, store_path, result_dir, options):
    """Component SFT proof: no trainer, FSDP, evaluation or resume claim."""
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2")
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group(
        "nccl",
        store=dist.FileStore(store_path, 2),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=90),
    )
    result = {"rank": rank, **options}
    try:
        from diffusers import FluxTransformer2DModel

        mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("tp",))
        device = torch.device("cuda", rank)
        fixture = Path(options["fixture"])
        native = FluxTransformer2DModel.from_pretrained(
            fixture / "model" / "transformer", torch_dtype=torch.float32
        ).to(device)
        native.set_attention_backend("native")
        reference = copy.deepcopy(native)
        native.enable_gradient_checkpointing()
        reference.enable_gradient_checkpointing()
        adapter = Flux1Adapter if options["family"] == "flux1" else Flux1KontextAdapter
        apply_tensor_parallel(native, mesh, resolve_parallel_plan(adapter, sequence_parallel=True))
        placements = _assert_added_placements(native)
        optimizer = torch.optim.AdamW(native.parameters(), lr=1e-3, eps=1e-4)
        reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-3, eps=1e-4)
        parameters = parameter_map(native)
        added = {path + ".weight" for path in placements}
        updates, condition_lengths = [], []
        operands = torch.load(fixture / "operands.pt", weights_only=True)
        for row in operands:
            inputs = {
                name: value.to(device) if torch.is_tensor(value) else value
                for name, value in row["inputs"].items()
            }
            target = row["target"].to(device)
            actual = native(**inputs)[0][:, : row["target_tokens"]]
            expected = reference(**inputs)[0][:, : row["target_tokens"]]
            torch.testing.assert_close(actual, expected, rtol=3e-4, atol=2e-6)
            loss = (actual - target).square().mean()
            reference_loss = (expected - target).square().mean()
            torch.testing.assert_close(loss, reference_loss, rtol=3e-4, atol=2e-6)
            loss.backward()
            reference_loss.backward()
            before = {name: full_value(parameters[name]) for name in sorted(added)}
            audit = assert_update_parity(reference, native, reference_optimizer, optimizer)
            changed = [
                name
                for name in sorted(added)
                if not torch.equal(before[name], full_value(parameters[name]))
            ]
            assert set(changed) == added
            audit.update(
                loss=float(loss.detach()),
                newly_sharded_updated_parameters=changed,
                condition_images=row["condition_images"],
            )
            updates.append(audit)
            condition_lengths.append(inputs["hidden_states"].shape[1])
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
        assert len(updates) == 3
        expected_tokens = 16 if options["family"] == "flux1" else 24
        assert condition_lengths == [expected_tokens] * 3, condition_lengths
        canonical = {name: full_value(value) for name, value in native.state_dict().items()}
        checkpoint = Path(result_dir, f"canonical-{rank}.pt")
        torch.save(canonical, checkpoint)
        restored = copy.deepcopy(reference)
        restored.load_state_dict(torch.load(checkpoint, weights_only=True))
        torch.testing.assert_close(restored(**inputs)[0], native(**inputs)[0], rtol=3e-4, atol=2e-6)
        result.update(
            status="pass",
            updates=updates,
            condition_tokens=condition_lengths,
            placements=placements,
            tp_size=mesh.size(),
            tp_rank=mesh.get_local_rank(),
            dp_size=1,
            cp_size=1,
            sequence_parallel=True,
            canonical_reload=True,
            world_size=dist.get_world_size(),
            device="cuda",
            backend=dist.get_backend(),
            execution_scope="component_supervised_flow",
        )
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result, indent=2))
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Flux TP2 SFT requires two CUDA ranks")
def test_cuda_flux_tp2_single_stream_sft(tmp_path):
    """One component scenario, sequential native T2I and conditioned supervised flow."""
    all_results = {}
    for family in ("flux1", "flux1-kontext"):
        root = tmp_path / family
        root.mkdir()
        fixture = root / "fixture"
        _fixture(fixture, family)
        results = run_gloo_ranks(_sft_worker, root, 2, {"family": family, "fixture": str(fixture)})
        assert all(row.get("status") == "pass" and "error" not in row for row in results), results
        all_results[family] = results
    print(json.dumps({"probe": "flux_complete_tp_sft", "families": all_results}))


@pytest.mark.parametrize("family", ["flux1", "flux1-kontext"])
def test_complete_tp_sft_recipe_is_small_and_explicit(tmp_path, family):
    operands = _fixture(tmp_path / "fixture", family)
    assert len(operands) == 3
    for row in operands:
        assert row["target"].shape == (1, 16, 16)
        assert torch.isfinite(row["target"]).all() and torch.count_nonzero(row["target"])
        assert row["condition_images"] == (0 if family == "flux1" else 1)
        assert row["inputs"]["hidden_states"].shape[1] == (16 if family == "flux1" else 24)
        if family == "flux1-kontext":
            assert torch.equal(row["inputs"]["img_ids"][16:, 0], torch.ones(8))
