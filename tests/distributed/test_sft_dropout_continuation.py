"""Record-level SFT dropout across a real Wan DP2 × CP2 checkpoint boundary."""

import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from conftest import run_gloo_ranks
from test_context_parallel_training import _compare, _config
from test_model_preparation import full_value, initialize_cuda, runtime_observation


def _worker(rank, store, output, options):
    os.environ.update(
        FSDP_VERSION="2",
        FSDP_AUTO_WRAP_POLICY="TRANSFORMER_BASED_WRAP",
        FSDP_TRANSFORMER_CLS_TO_WRAP="WanTransformerBlock",
        FSDP_CPU_RAM_EFFICIENT_LOADING="false",
        FSDP_RESHARD_AFTER_FORWARD="true",
        FSDP_ACTIVATION_CHECKPOINTING="false",
    )
    initialize_cuda(rank, store, world_size=4)
    result = {"rank": rank}
    try:
        from flow_factory.trainers.loader import load_trainer

        config = _config(Path(options["fixture"]), output, options)
        config.training_args.condition_dropout = 0.5
        trainer = load_trainer(config)
        result["runtime"] = runtime_observation(trainer.accelerator)
        assert trainer.accelerator.is_fsdp2
        assert config.model_args.model_type == "wan2_t2v"
        assert config.training_args.trainer_type == "sft"
        assert config.parallelism_args.effective_context_parallel_size == 2
        assert trainer.accelerator.mixed_precision == "bf16"
        result["execution_capability"] = "wan-sft-ulysses-bf16-v1"
        batches, updates = [], []
        optimize = trainer.optimize_batch
        step = trainer._apply_optimizer_step

        def observe_batch(batch):
            batches.append(
                [
                    (rid, bool(mask))
                    for rid, mask in zip(
                        batch.record_ids, batch.condition["_sft_mask"], strict=True
                    )
                ]
            )
            return optimize(batch)

        def observe_update(metrics):
            gradients = {
                k: full_value(p.grad).cpu().clone()
                for k, p in trainer.model_bundle.named_parameters()
                if p.requires_grad
            }
            losses = torch.stack(metrics["loss"]).detach().cpu()
            new_metrics = step(metrics)
            weights = {
                k: full_value(p).cpu().clone()
                for k, p in trainer.model_bundle.named_parameters()
                if p.requires_grad
            }
            updates.append((losses, gradients, weights))
            return new_metrics

        trainer.optimize_batch = observe_batch
        trainer._apply_optimizer_step = observe_update
        trainer.start()
        assert trainer.step == 4
        torch.save(updates, Path(output, f"updates-{rank}.pt"))
        result["batches"] = batches
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        Path(output, f"rank-{rank}.json").write_text(json.dumps(result))
        dist.destroy_process_group()


@pytest.mark.skipif(
    torch.cuda.device_count() < 4, reason="Wan SFT continuation requires local DP2 × CP2"
)
def test_cuda_dropout_dp2_cp2_continuation(tmp_path):
    fixture = tmp_path / "fixture"
    subprocess.run(
        [sys.executable, str(Path(__file__).with_name("wan_sft_fixture.py")), str(fixture)],
        check=True,
    )
    options = dict(
        fixture=str(fixture),
        cp_size=2,
        mode="lora",
        checkpointing=False,
        epochs=2,
        save_freq=1,
        save_model_only=False,
    )
    groups = []
    for name in ("continuous", "resumed"):
        if name == "resumed":
            options["resume"] = str(tmp_path / "continuous/saves/wan-sft/checkpoints/checkpoint-1")
        result = run_gloo_ranks(_worker, tmp_path / name, 4, options)
        for row in result:
            assert "error" not in row, row.get("error")
        groups.append([row["batches"] for row in result])
    continuous, resumed = groups
    assert continuous[0] == continuous[1] and continuous[2] == continuous[3]
    assert any(
        left[0][1] != right[0][1] for left, right in zip(continuous[0], continuous[2], strict=True)
    )
    for rank in range(4):
        assert len(resumed[rank]) == 4 and continuous[rank][4:] == resumed[rank]
        continuous_updates = torch.load(
            tmp_path / "continuous" / f"updates-{rank}.pt", weights_only=True
        )
        resumed_updates = torch.load(tmp_path / "resumed" / f"updates-{rank}.pt", weights_only=True)
        _compare(resumed_updates, continuous_updates[2:], exact=True)
