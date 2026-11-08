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

"""Native structured AV generation, joint GRPO, evaluation and drained recovery."""

import math
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

import torch
from context_parallel_overlap_runner import _audit_fresh_replay
from ltx2_coupled_observations import assert_objectives, observe_objectives

from flow_factory.hparams import Arguments

audit_kind = "coupled_replay"
audit_steps = 2
wrap_classes = ("LTX2VideoTransformerBlock",)
reward_media_field = "video"


def build_fixture(root, tp_size, variant):
    from ltx2_cp_fixture import build_pipeline
    from PIL import Image

    assert tp_size == 2
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    build_pipeline(
        root / "build",
        ltx23=variant.startswith("ltx23"),
        image_conditioned=variant.endswith("i2av"),
    ).save_pretrained(root / "model")
    (root / "data").mkdir(exist_ok=True)
    for index in range(16):
        Image.new("RGB", (64, 64), (80 + index * 7, 25 + index * 3, 210 - index * 7)).save(
            root / "data" / f"{index}-first.png"
        )


def dataset_record(root, prompt, index, variant):
    row = {"prompt": prompt}
    if variant.endswith("i2av"):
        row["images"] = [str(Path(root) / "data" / f"{index}-first.png")]
    return row


def make_config(root, result_dir, options, variant):
    mode = options["training_mode"]
    assert options["algorithm"] == "grpo"
    extra = {
        "negative_prompt": "blue",
        "max_sequence_length": 128,
        "audio_guidance_scale": 2.5,
        "stg_scale": 0.1,
        "audio_stg_scale": 0.15,
        "spatio_temporal_guidance_blocks": [0],
        "modality_scale": 1.1,
        "audio_modality_scale": 1.15,
        "use_cross_timestep": variant.startswith("ltx23"),
    }
    return Arguments.from_dict(
        {
            "mixed_precision": "bf16",
            "parallelism": {
                "dp_shard_size": options["dp_size"],
                "cp_size": options["cp_size"],
                "tp_size": options["tp_size"],
                "sequence_parallel": options["tp_size"] > 1,
            },
            "model": {
                "model_type": "ltx2_i2av" if variant.endswith("i2av") else "ltx2_t2av",
                "model_name_or_path": str(root / "model"),
                "target_components": ["transformer"],
                "finetune_type": mode,
                "target_modules": "all" if mode == "full" else "default",
                "component_load_dtypes": "float32",
                "lora_rank": 2,
                "lora_alpha": 2,
            },
            "data": {
                "datasets": [
                    {"name": "tiny", "dataset_dir": str(root / "tiny"), "train": {"weight": 1}}
                ],
                "cache_dir": str(root / "overlap-cache"),
                "dataloader_num_workers": 0,
                "preprocessing_batch_size": 1,
                "sampler_type": "rank_local",
                "enable_preprocess": True,
            },
            "train": {
                "trainer_type": "grpo",
                "max_epochs": 2,
                "per_device_batch_size": 1,
                "resolution": [64, 64],
                "num_frames": 17,
                "frame_rate": 50.0,
                "num_inference_steps": 3,
                "guidance_scale": 2.0,
                "group_size": 2,
                "unique_sample_num_per_epoch": 2 * options["dp_size"],
                "gradient_step_per_epoch": 2,
                "gradient_accumulation_steps": "auto",
                "ema_decay": 0.9,
                "ema_update_interval": 1,
                "ema_decay_schedule": "constant",
                "enable_gradient_checkpointing": True,
                "seed": 17,
                "shuffle_samples": False,
                "latent_storage_dtype": "float32",
                "kl_beta": 0.0,
                "global_std": False,
                "advantage_aggregation": "sum",
                "extra_kwargs": extra,
            },
            "eval": {"eval_freq": 0, **extra},
            "log": {
                "save_model_only": False,
                "save_dir": str(result_dir),
                "logging_backend": "none",
            },
            "scheduler": {
                "dynamics_type": "Flow-SDE",
                "sde_steps": [0, 1],
                "num_sde_steps": 2,
                "seed": 17,
            },
            "rewards": [
                {
                    "name": "score",
                    "reward_model": "wan_online_reward.MemberReward",
                    "device": "cpu",
                    "async_reward": True,
                }
            ],
            "acceleration": {
                "shared": [{"name": "attention_backend", "params": {"backend": "native"}}]
            },
            "optimizers": [
                {"name": "default", "learning_rate": 1e-3, "eps": 1e-4, "max_grad_norm": 1.0}
            ],
        }
    )


def score_media(media):
    assert media.shape == (17, 3, 64, 64) and torch.isfinite(media).all()
    return float(media.double().mean())


def reference_media_score(media):
    values = media.reshape(-1).tolist()
    return math.fsum(values) / len(values)


@contextmanager
def observe_training_terms(trainer, terms):
    original = trainer._replay_forward

    @wraps(original)
    def replay(batch, step, return_fields):
        assert tuple(step.times.timestep) == ("video", "audio")
        torch.testing.assert_close(
            step.times.timestep["video"], step.times.timestep["audio"], rtol=0, atol=0
        )
        if "advantage" in batch:
            sample = batch.samples[0]
            terms.append(
                (
                    (sample.source, int(sample.sampling_sample_id)),
                    float(step.times.timestep["video"].reshape(-1)[0]),
                    trainer.accelerator.gradient_accumulation_steps,
                )
            )
        return original(batch, step, return_fields)

    trainer._replay_forward = replay
    try:
        yield
    finally:
        trainer._replay_forward = original


def audit_samples(trainer, samples, variant):
    for sample in samples:
        assert sample.trajectory.component_names == ("video", "audio")
        assert sample.all_latents is None and sample.timesteps is None
        assert sample.video.shape == (17, 3, 64, 64)
        assert (
            sample.audio.ndim == 2
            and sample.audio.shape[0] == 2
            and torch.isfinite(sample.audio).all()
        )
        assert sample.audio_sample_rate == 24000
        assert sample.connector_attention_mask.shape == (128,)
        assert not torch.equal(
            sample.connector_prompt_embeds, sample.negative_connector_prompt_embeds
        )
        video = sample.trajectory.components["video"]
        audio = sample.trajectory.components["audio"]
        assert video.states.shape[1:] == (12, 128) and audio.states.shape[1:] == (8, 128)
        torch.testing.assert_close(video.timesteps, audio.timesteps, rtol=0, atol=0)
        if variant.endswith("i2av"):
            assert len(sample.condition_images) == 1
            torch.testing.assert_close(
                sample.conditioning_mask,
                torch.tensor([1.0] * 4 + [0.0] * 8, device=sample.conditioning_mask.device),
                rtol=0,
                atol=0,
            )
            assert video.active_mask is not None and torch.count_nonzero(video.active_mask) == 8
            torch.testing.assert_close(
                video.states[:, :4],
                video.states[:1, :4].expand_as(video.states[:, :4]),
                rtol=0,
                atol=0,
            )
    return _audit_fresh_replay(trainer, samples)
