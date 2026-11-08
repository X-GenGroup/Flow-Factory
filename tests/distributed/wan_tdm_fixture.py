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

"""Explicit native Wan TDM trajectory and two-role recipe."""

import json
import subprocess
import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

from tdm_observations import observe_objectives
from test_context_parallel_training import _compare

from flow_factory.hparams import Arguments

wrap_classes = ("WanTransformerBlock",)
checkpoint_run_name = "wan-tdm"
execution_capability = "wan-tdm-cp-v1"


def build_fixture(root, tp_size):
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("wan_sft_fixture.py")),
            str(root),
            str(2 * tp_size),
        ],
        check=True,
    )
    prompts = root / "prompts"
    prompts.mkdir()
    (prompts / "train.jsonl").write_text(
        "".join(
            json.dumps({"prompt": f"a {'red' if i % 2 else 'blue'} video {i}"}) + "\n"
            for i in range(8)
        )
    )
    (prompts / "test.jsonl").write_text(json.dumps({"prompt": "a red video 3"}) + "\n")


def config_dict(root, result_dir, options):
    """Return a complete TDM recipe, reusable for the strictly extended R1 leaf."""
    return {
        "mixed_precision": "bf16",
        "parallelism": {
            "dp_shard_size": options["dp_size"],
            "cp_size": options["cp_size"],
            "tp_size": options["tp_size"],
            "sequence_parallel": options["tp_size"] > 1,
        },
        "model": {
            "model_type": "wan2_t2v",
            "model_name_or_path": str(root / "model"),
            "target_components": ["transformer"],
            "finetune_type": "full",
            "target_modules": "all",
            "component_load_dtypes": "float32",
            "compute_precision_policy": "cross_attention_fp32_v1",
            **(
                {"resume_type": "state", "resume_path": options["resume"]}
                if options.get("resume")
                else {}
            ),
        },
        "data": {
            "datasets": [
                {
                    "name": "tiny",
                    "dataset_dir": str(root / "prompts"),
                    "train": {"weight": 1},
                    "eval": {},
                }
            ],
            "cache_dir": str(root / "tdm-cache"),
            "dataloader_num_workers": 0,
            "preprocessing_batch_size": 1,
            "sampler_type": "rank_local",
            "enable_preprocess": True,
        },
        "train": {
            "trainer_type": "tdm",
            "max_epochs": 2,
            "per_device_batch_size": 1,
            "resolution": [32, 32],
            "num_frames": 5,
            "frame_rate": 4.0,
            "num_inference_steps": 2,
            "guidance_scale": 1.0,
            "real_guidance_scale": 2.0,
            "group_size": 1,
            "unique_sample_num_per_epoch": 4,
            "gradient_step_per_epoch": 1,
            "gradient_accumulation_steps": 4,
            "ema_decay": 0.9,
            "ema_update_interval": 1,
            "ema_decay_schedule": "constant",
            "enable_gradient_checkpointing": True,
            "seed": 17,
            "shuffle_samples": False,
            "latent_storage_dtype": "float32",
            "tdm_query_interval": "reverse",
            "tdm_query_distribution": "source_uniform",
            "tdm_query_max_sigma": 1.0,
            "use_huber": True,
            "huber_c": 0.1,
            "replay_rtol": 0.0,
            "replay_atol": 0.0,
        },
        "scheduler": {"dynamics_type": "ODE", "noise_level": 0.0},
        "log": {
            "save_model_only": False,
            "save_freq": 1,
            "save_dir": str(Path(result_dir) / "saves"),
            "run_name": checkpoint_run_name,
            "logging_backend": "none",
            "verbose": False,
        },
        "eval": {
            "eval_freq": 1,
            "resolution": [32, 32],
            "num_frames": 5,
            "frame_rate": 4.0,
            "num_inference_steps": 2,
            "guidance_scale": 1.0,
            "per_device_batch_size": 1,
            "seed": 0,
        },
        "eval_rewards": [
            {
                "name": "evaluation",
                "reward_model": "wan_eval_reward.PromptReward",
                "device": "cpu",
                "batch_size": 1,
            }
        ],
        "acceleration": {
            "shared": [{"name": "attention_backend", "params": {"backend": "native"}}]
        },
        "optimizers": [
            {
                "name": role,
                "learning_rate": 1e-3 if role == "fake" else 5e-4,
                "eps": 1e-4,
                "max_grad_norm": 1.0,
                "update_frequency": 2 if role == "fake" else 1,
            }
            for role in ("generator", "fake")
        ],
    }


def make_config(root, result_dir, options):
    return Arguments.from_dict(config_dict(root, result_dir, options))


@contextmanager
def observe_training(trainer, records):
    assert not trainer.reward_models and trainer.reward_buffer is None
    feedback, decode = trainer.prepare_feedback, trainer.adapter.decode_latents

    def observed_feedback(samples):
        assert trainer.reward_buffer is None and not trainer.reward_models
        before = [dict(sample.extra_kwargs) for sample in samples]
        returned = feedback(samples)
        assert [dict(sample.extra_kwargs) for sample in samples] == before
        assert all("advantage" not in sample.extra_kwargs for sample in samples)
        return returned

    trainer.prepare_feedback = observed_feedback
    with observe_objectives(trainer, records):
        try:
            yield
        finally:
            trainer.prepare_feedback = feedback
            assert trainer.adapter.decode_latents == decode


def assert_worker(trainer, records, loaded, final, options):
    cycles = 1 if options.get("resume") else 2
    assert Counter(row["role"] for row in records["updates"]) == {
        "fake": 2 * cycles,
        "generator": cycles,
    }
    assert Counter(row["kind"] for row in records["objectives"]) == {
        "fake": 8 * cycles,
        "generator": 4 * cycles,
    }
    assert [len(row["loss"]) for row in records["updates"]] == [4] * (3 * cycles)
    assert {row["boundary"] for row in records["objectives"]} == {1, 2}
    assert len(records["samples"]) == 2 * cycles
    assert trainer.training_args.gradient_accumulation_steps == 4
    assert len(records["score_queries"]) == 8 * cycles
    for reference, fake in zip(
        records["score_queries"][::2], records["score_queries"][1::2], strict=True
    ):
        assert (reference["role"], fake["role"]) == ("reference", "fake")
        _compare(reference["state"], fake["state"], exact=True)
        _compare(reference["times"], fake["times"], exact=True)


def assert_continuation(continuous, resumed, joined, options):
    for key in (
        "samples",
        "replays",
        "queries",
        "score_queries",
        "objectives",
        "evaluations",
        "renoising",
        "projections",
    ):
        _compare(joined[key], continuous["records"][key], exact=True)
