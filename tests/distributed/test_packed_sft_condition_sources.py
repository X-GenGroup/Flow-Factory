# Copyright 2026 Jayce-Ping
# SPDX-License-Identifier: Apache-2.0

"""Exercise cached and raw V2 rows through the real packed-prefix SFT training loop."""

import copy
import json
from unittest.mock import patch

import pytest
import torch
from PIL import Image

from flow_factory.hparams import Arguments
from flow_factory.trainers.loader import load_trainer


def _model(family, root, monkeypatch):
    if family == "bagel":
        import bagel_cp_fixture as fixture
        from test_bagel_parallel import _cpu_flash_reference

        pytest.importorskip("flash_attn", reason="BAGEL optional package; CPU uses the SDPA oracle")

        from flow_factory.models.bagel.bagel import BagelAdapter
        from flow_factory.models.bagel.data.transforms import ImageTransform
        from flow_factory.models.bagel.modeling.bagel import qwen2_navit, siglip_navit

        monkeypatch.setattr(qwen2_navit, "flash_attn_varlen_func", _cpu_flash_reference)
        monkeypatch.setattr(siglip_navit, "flash_attn_varlen_func", _cpu_flash_reference)
        fixture.build_tokenizer(root)
        pipeline = fixture.build_pipeline(heads=4, kv_heads=2, head_dim=16)
        init = BagelAdapter._init_tokenizer_and_transforms

        def tiny_transforms(self):
            init(self)
            self.vae_transform = ImageTransform(64, 32, 16)
            self.vit_transform = ImageTransform(56, 28, 14)

        monkeypatch.setattr(BagelAdapter, "_init_tokenizer_and_transforms", tiny_transforms)
        monkeypatch.setattr(BagelAdapter, "load_pipeline", lambda self: copy.deepcopy(pipeline))
    else:
        import sensenova_cp_fixture as fixture

        from flow_factory.models.sensenova.pipeline import SenseNovaPseudoPipeline
        from flow_factory.models.sensenova.sensenova import SenseNovaAdapter

        pipeline = SenseNovaPseudoPipeline(
            fixture.build_denoiser().model, fixture.build_tokenizer(root)
        )
        monkeypatch.setattr(SenseNovaAdapter, "load_pipeline", lambda self: copy.deepcopy(pipeline))


@pytest.mark.parametrize("family", ["bagel", "sensenova"])
def test_packed_sft_raw_and_cached_match(family, tmp_path, monkeypatch):
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    model = tmp_path / "model"
    _model(family, model, monkeypatch)
    data = tmp_path / "data"
    data.mkdir()
    for name, color in (("red", "red"), ("blue", "blue"), ("target", "green")):
        Image.new("RGB", (64, 64), color).save(data / f"{name}.png")

    def media(name):
        return {"type": "image", "path": f"{name}.png"}

    row = {
        "schema_version": 2,
        "input": {"prompt": "combine these", "media": [media("red"), media("blue")]},
        "supervision": {"type": "demonstration", "target": {"media": [media("target")]}},
    }
    (data / "train.jsonl").write_text(json.dumps(row) + "\n")

    observations = []
    for cached in (False, True):
        config = Arguments.from_dict(
            {
                "mixed_precision": "bf16",
                "num_processes": 1,
                "model": {
                    "model_type": family,
                    "model_name_or_path": str(model),
                    "finetune_type": "full",
                    "target_components": ["transformer"],
                    "component_load_dtypes": "float32",
                },
                "data": {
                    "datasets": [
                        {"name": "ordered-edits", "dataset_dir": str(data), "train": {"weight": 1}}
                    ],
                    "enable_preprocess": cached,
                    "dataloader_num_workers": 0,
                    "preprocessing_batch_size": 1,
                    "cache_dir": str(tmp_path / "cache"),
                },
                "train": {
                    "trainer_type": "sft",
                    "max_epochs": 1,
                    "condition_dropout": 1.0,
                    "resolution": [64, 64],
                    "gradient_accumulation_steps": 1,
                    "per_device_batch_size": 1,
                    "num_train_timesteps": 1,
                    "weighting_scheme": "uniform",
                    "enable_gradient_checkpointing": False,
                    "ema_decay": 0,
                    "seed": 17,
                },
                "scheduler": {"dynamics_type": "ODE"},
                "eval": {"eval_freq": 0},
                "log": {
                    "save_dir": str(tmp_path / "runs"),
                    "run_name": f"{family}-{cached}",
                    "logging_backend": "none",
                },
            }
        )
        trainer = load_trainer(config)
        term = {}
        original = trainer.adapter.forward_state

        def capture(*args, **kwargs):
            batch = kwargs["batch"]
            term["mask"] = batch["sft_condition_masks"]
            images = trainer.adapter._normalize_condition_images(batch["condition_images"])
            term["references"] = [image.getpixel((0, 0)) for image in images[0]]
            result = original(*args, **kwargs)
            term["velocity"] = result.velocity.components["latent"].detach().clone()
            return result

        encode = trainer.adapter.encode_output_state

        def target(*args, **kwargs):
            result = encode(*args, **kwargs)
            term["target"] = result.clean_state.components["latent"].detach().clone()
            return result

        prefill_method = "_build_gen_context" if family == "bagel" else "_build_context"
        prefill = getattr(trainer.adapter, prefill_method)
        with (
            patch.object(trainer.adapter, "forward_state", side_effect=capture),
            patch.object(trainer.adapter, "encode_output_state", side_effect=target),
            patch.object(trainer.adapter, prefill_method, wraps=prefill) as native_prefix,
        ):
            batch = next(iter(trainer.dataloader))
            assert ("_sft_raw" in batch.condition) == (not cached)
            assert batch.output.target_media[0][0].payload.getpixel((0, 0)) == (0, 128, 0)
            trainer.optimize_batch(batch)
        (call,) = native_prefix.call_args_list
        if family == "bagel":
            assert call.kwargs["sft_condition_masks"] == term["mask"]
        else:
            assert call.kwargs["sft_mask"] == term["mask"][0]
        assert trainer.progress.optimizer_step == 1 and len(term["mask"]) == 1 and term["mask"][0]
        assert term["references"] == [(255, 0, 0), (0, 0, 255)]
        observations.append(term)
        trainer.accelerator.end_training()
    first, second = observations
    assert first["mask"] == second["mask"]
    torch.testing.assert_close(first["target"], second["target"], atol=0, rtol=0)
    torch.testing.assert_close(first["velocity"], second["velocity"], atol=0, rtol=0)
