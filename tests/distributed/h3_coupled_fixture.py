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
from h3_coupled_observations import assert_objectives, observe_objectives

from flow_factory.hparams import Arguments

audit_kind = "coupled_replay"
audit_steps = 2
wrap_classes = ("MiniMaxH3TransformerBlock",)
reward_media_field = "video"


def build_fixture(root, tp_size, workflow):
    import wave

    import av
    import numpy as np
    from h3_cp_fixture import build_pipeline, workflow_inputs
    from PIL import Image

    assert tp_size in (1, 2)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    build_pipeline(workflow).save_pretrained(str(root / "model"), overwrite_modular_index=True)
    (root / "data").mkdir(exist_ok=True)
    for index in range(16):
        Image.new("RGB", (32, 32), (80 + index * 7, 25 + index * 3, 210 - index * 7)).save(
            root / "data" / f"{index}-first.png"
        )
        Image.new("RGB", (32, 32), (25 + index * 3, 210 - index * 7, 80 + index * 7)).save(
            root / "data" / f"{index}-last.png"
        )
    if workflow == "ref2va":
        references = workflow_inputs(workflow)["references"]
        waveform = references[1].audio.T.numpy()
        with wave.open(str(root / "data" / "voice.wav"), "wb") as handle:
            handle.setnchannels(2)
            handle.setsampwidth(2)
            handle.setframerate(32000)
            handle.writeframes((waveform * 32767).astype("<i2").tobytes())
        with av.open(str(root / "data" / "motion.mp4"), mode="w") as container:
            stream = container.add_stream("mpeg4", rate=24)
            stream.width = stream.height = 32
            stream.pix_fmt = "yuv420p"
            for pixels in references[2].frames:
                for packet in stream.encode(
                    av.VideoFrame.from_ndarray(np.asarray(pixels), format="rgb24")
                ):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)


def dataset_record(root, prompt, index, workflow):
    media = Path(root) / "data"
    if workflow == "ref2va":
        return {
            "prompt": prompt,
            "references": [
                {"type": "image", "path": str(media / f"{index}-first.png")},
                {"type": "audio", "path": str(media / "voice.wav"), "sample_rate": 32000},
                {"type": "video", "path": str(media / "motion.mp4"), "fps": 24.0},
            ],
        }
    return {
        "prompt": prompt,
        "images": [str(media / f"{index}-first.png"), str(media / f"{index}-last.png")],
    }


def make_config(root, result_dir, options, workflow):
    mode = options["training_mode"]
    assert options["algorithm"] == "grpo"
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
                "model_type": f"minimax-h3-{workflow}",
                "model_name_or_path": str(root / "model"),
                "target_components": ["transformer_ref" if workflow == "ref2va" else "transformer"],
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
                "resolution": [32, 32],
                "num_frames": 124,
                "frame_rate": 24.0,
                "num_inference_steps": 3,
                "guidance_scale": 1.0,
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
            },
            "eval": {"eval_freq": 0},
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
    assert media.shape == (124, 3, 32, 32) and torch.isfinite(media).all()
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
        assert tuple(step.times.sigma) == ("video", "audio")
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


def audit_samples(trainer, samples, workflow):
    from flow_factory.samples.references import parse_reference_manifest

    for sample in samples:
        assert sample.trajectory.component_names == ("video", "audio")
        assert sample.all_latents is None and sample.timesteps is None
        assert sample.video.shape == (124, 3, 32, 32)
        assert sample.audio.shape == (2, 165600) and torch.isfinite(sample.audio).all()
        assert sample.audio_sample_rate == 32000
        video, audio = (sample.trajectory.components[name] for name in ("video", "audio"))
        assert video.states.shape[1:] == (37, 96) and audio.states.shape[1:] == (414, 32)
        assert not torch.equal(video.timesteps, audio.timesteps)
        layout, prefixes = sample.extra_kwargs["layout"], sample.extra_kwargs["condition_prefixes"]
        assert prefixes["video"].shape[0] == layout["num_condition_video_rows"] > 0
        assert prefixes["audio"].shape[0] == layout["num_condition_audio_rows"]
        assert layout["position_ids"].shape[0] == sample.prompt_embeds.shape[0] + 451 + sum(
            value.shape[0] for value in prefixes.values()
        )
        if workflow == "fl2va":
            assert len(sample.condition_images) == 2
            assert prefixes["video"].shape == (2, 96) and prefixes["audio"].shape == (0, 32)
        else:
            refs = parse_reference_manifest(sample.reference_manifest, row_index=0)
            assert [ref["type"] for ref in refs] == ["image", "audio", "video"]
            assert prefixes["video"].shape == (8, 96) and prefixes["audio"].shape == (80, 32)
    return _audit_fresh_replay(trainer, samples)
