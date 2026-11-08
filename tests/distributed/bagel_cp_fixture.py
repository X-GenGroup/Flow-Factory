# Copyright 2026 Jayce-Ping
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Real Bagel MoT/GQA, native VAE/ViT and ordered-image fixture construction."""

from pathlib import Path

import torch


def build_tokenizer(root):
    from tokenizers import ByteLevelBPETokenizer
    from transformers import Qwen2Tokenizer

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    core = ByteLevelBPETokenizer()
    specials = [
        "<pad>",
        "<unk>",
        "<|im_start|>",
        "<|im_end|>",
        "<|vision_start|>",
        "<|vision_end|>",
    ]
    core.train_from_iterator(
        ["a red image", "a blue small image", "red blue green small image"],
        vocab_size=384,
        special_tokens=specials,
    )
    core.save_model(str(root))
    tokenizer = Qwen2Tokenizer(
        vocab=str(root / "vocab.json"),
        merges=str(root / "merges.txt"),
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="<|im_end|>",
        additional_special_tokens=specials[2:],
    )
    tokenizer.save_pretrained(root)
    return tokenizer


def build_language_model(*, heads=28, kv_heads=4, head_dim=128):
    from flow_factory.models.bagel.modeling.bagel.qwen2_navit import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(823)
    return Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=384,
            hidden_size=heads * head_dim,
            intermediate_size=1024,
            num_hidden_layers=2,
            num_attention_heads=heads,
            num_key_value_heads=kv_heads,
            max_position_embeddings=32768,
            rope_theta=1000000.0,
            rms_norm_eps=1e-6,
            qk_norm=True,
            tie_word_embeddings=False,
            layer_module="Qwen2MoTDecoderLayer",
            freeze_und=False,
            pad_token_id=0,
            bos_token_id=2,
            eos_token_id=3,
        )
    )


def install_lora(module):
    from peft import LoraConfig, get_peft_model

    from flow_factory.models.bagel.bagel import BagelAdapter

    torch.manual_seed(37)
    return get_peft_model(
        module,
        LoraConfig(
            r=2,
            lora_alpha=2,
            init_lora_weights=False,
            target_modules=BagelAdapter.default_target_modules.fget(object.__new__(BagelAdapter)),
        ),
    )


def build_pipeline(*, heads=28, kv_heads=4, head_dim=128, full_vae=False):
    from flow_factory.models.bagel.modeling.autoencoder import AutoEncoder, AutoEncoderParams
    from flow_factory.models.bagel.modeling.bagel import (
        Bagel,
        BagelConfig,
        SiglipVisionConfig,
        SiglipVisionModel,
    )
    from flow_factory.models.bagel.pipeline import BagelPseudoPipeline

    language = build_language_model(heads=heads, kv_heads=kv_heads, head_dim=head_dim)
    vision_config = SiglipVisionConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        patch_size=14,
        image_size=980,
        rope=False,
    )
    vision = SiglipVisionModel(vision_config)
    vision.vision_model.embeddings.convert_conv2d_to_linear(vision_config)
    vae_config = AutoEncoderParams(
        resolution=256,
        in_channels=3,
        downsample=8,
        ch=128 if full_vae else 32,
        out_ch=3,
        ch_mult=[1, 2, 4, 4],
        num_res_blocks=2 if full_vae else 1,
        z_channels=16,
        scale_factor=0.3611,
        shift_factor=0.1159,
    )
    config = BagelConfig(
        llm_config=language.config,
        vit_config=vision_config,
        vae_config=vae_config,
        latent_patch_size=2,
        max_latent_size=64,
        vit_max_num_patch_per_side=70,
    )
    model = Bagel(language, vision, config)
    # A pretrained checkpoint's flow head is nonzero; initialize a nondegenerate
    # deterministic fixture rather than retaining Bagel's pretraining zero head.
    torch.nn.init.normal_(model.llm2vae.weight, std=0.01)
    vae = AutoEncoder(vae_config)
    vae.reg.sample = False
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("language_model."))
    vae.requires_grad_(False)
    return BagelPseudoPipeline(model, vae, config=config)


def make_input_adapter(pipeline, tokenizer, device, *, transformer=None):
    from types import SimpleNamespace

    from flow_factory.models.bagel.bagel import BagelAdapter
    from flow_factory.models.bagel.data.data_utils import add_special_tokens
    from flow_factory.models.bagel.data.transforms import ImageTransform

    adapter = object.__new__(BagelAdapter)
    adapter.pipeline = pipeline
    adapter.accelerator = SimpleNamespace(device=torch.device(device))
    adapter.component_runtime = SimpleNamespace(
        get_component=lambda name: (
            transformer
            if name == "transformer" and transformer is not None
            else getattr(pipeline, name)
        )
    )
    adapter._tokenizer, adapter.new_token_ids, _ = add_special_tokens(tokenizer)
    # Native transforms with smaller images for the mathematical model control.
    # The serialized lifecycle uses Bagel's ordinary 512/224 preprocessing limits.
    adapter.vae_transform = ImageTransform(64, 32, 16)
    adapter.vit_transform = ImageTransform(56, 28, 14)
    return adapter


def condition_images(replica=0, *, ragged=True):
    import numpy as np
    from PIL import Image

    images = []
    for sample, count in enumerate((1, 2) if ragged else (2, 2)):
        row = []
        for reference in range(count):
            width = (32 if sample == 0 else 48) if reference == 0 else 64
            pixels = np.random.default_rng(711 + 100 * replica + 10 * sample + reference).integers(
                0, 256, (32, width, 3), dtype=np.uint8
            )
            row.append(Image.fromarray(pixels))
        images.append(row)
    return images


def build_fixture(root):
    """Serialize genuine native loader inputs; the lifecycle patches no model loader."""
    from safetensors.torch import save_file

    root = Path(root)
    model_root = root / "model"
    build_tokenizer(model_root)
    pipeline = build_pipeline(full_vae=True)
    pipeline.bagel.config.llm_config.to_json_file(model_root / "llm_config.json")
    vision_config = pipeline.bagel.config.vit_config.to_dict()
    # Bagel's native loader removes the last pretrained vision block.
    vision_config["num_hidden_layers"] += 1
    import json

    (model_root / "vit_config.json").write_text(json.dumps(vision_config, indent=2))
    save_file(pipeline.bagel.state_dict(), model_root / "ema.safetensors")
    save_file(pipeline.vae.state_dict(), model_root / "ae.safetensors")
    data = root / "data"
    data.mkdir(exist_ok=True)
    for index in range(8):
        for reference, image in enumerate(condition_images(index, ragged=False)[0]):
            image.save(data / f"{index}-{reference}.png")


if __name__ == "__main__":
    import sys

    build_fixture(sys.argv[1])


def model_call_inputs(adapter, replica, step, *, ragged=False):
    """Build native image/text CFG caches and ordinary packed decoder operands."""
    device = adapter.device
    with torch.autocast(device.type, dtype=torch.bfloat16):
        contexts = adapter._build_gen_context(
            ["a red image", "a blue small image"], condition_images(replica, ragged=ragged)
        )
    inputs = adapter._prepare_gen_inputs(
        *contexts,
        [(64, 64), (64, 64)],
        device,
        generator=torch.Generator().manual_seed(991 + replica * 10 + step),
    )
    generation = inputs[0]
    model = adapter.pipeline.bagel
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
        latents = generation["packed_init_noises"]
        hidden = (
            model.vae2llm(latents)
            + model.time_embedder(latents.new_full((latents.shape[0],), 0.5 - 0.1 * step))
            + model.latent_pos_embed(generation["packed_vae_position_ids"])
        )
        packed = hidden.new_zeros(
            (int(generation["packed_seqlens"].sum()), model.hidden_size), dtype=torch.float32
        )
        packed[generation["packed_vae_token_indexes"]] = hidden.float()
    common = dict(
        packed_query_sequence=packed,
        query_lens=generation["packed_seqlens"],
        update_past_key_values=False,
        is_causal=False,
        mode="gen",
        packed_text_ids=generation["packed_text_ids"],
        packed_text_indexes=generation["packed_text_indexes"],
        packed_vae_token_indexes=generation["packed_vae_token_indexes"],
    )
    calls = []
    for index, (context, metadata) in enumerate(zip(contexts, inputs)):
        prefix = "cfg_" if index else ""
        positions = "packed_position_ids"
        query_indexes = "packed_query_indexes" if index else "packed_indexes"
        calls.append(
            dict(
                common,
                packed_query_position_ids=metadata[prefix + positions],
                packed_query_indexes=metadata[prefix + query_indexes],
                key_values_lens=metadata[prefix + "key_values_lens"],
                packed_key_value_indexes=metadata[prefix + "packed_key_value_indexes"],
                past_key_values=context["past_key_values"],
            )
        )
    observation = {
        "prefix_lengths": [context["kv_lens"] for context in contexts],
        "position_offsets": [context["ropes"] for context in contexts],
        "query_lengths": generation["packed_seqlens"].tolist(),
        "reference_counts": [len(images) for images in condition_images(replica, ragged=ragged)],
        "cache_detached": all(
            not tensor.requires_grad
            for context in contexts
            for tensor in context["past_key_values"].key_cache.values()
        ),
    }
    assert observation["cache_detached"]
    return calls, observation


def forward_calls(model, calls):
    """Keep each CFG call in its own forward precision scope."""
    outputs = []
    for inputs in calls:
        with torch.autocast(inputs["packed_query_sequence"].device.type, dtype=torch.bfloat16):
            outputs.append(model(**inputs).packed_query_sequence)
    return torch.stack(outputs)


def qualify_native_dependencies():
    """Run the production CP dependency gate before loading this model probe."""
    from importlib.metadata import version

    from flow_factory.hparams import Arguments
    from flow_factory.models.bagel.bagel import BagelAdapter
    from flow_factory.trainers.common.context_parallel import validate_context_parallel_config
    from flow_factory.trainers.offline.sft import SFTTrainer

    config = Arguments.from_dict(
        {
            "mixed_precision": "bf16",
            "parallelism": {
                "dp_shard_size": 2,
                "cp_size": 2,
                "tp_size": 2,
                "sequence_parallel": True,
            },
            "data": {
                "datasets": [{"name": "fixture", "dataset_dir": "unused", "train": {"weight": 1}}]
            },
            "model": {"model_type": "bagel", "target_components": ["transformer"]},
            "train": {"trainer_type": "sft", "gradient_accumulation_steps": 1, "ema_decay": 0},
            "log": {"save_model_only": True},
            "eval": {"eval_freq": 0},
            "acceleration": {
                "shared": [{"name": "attention_backend", "params": {"backend": "native"}}]
            },
        }
    )
    validate_context_parallel_config(config, BagelAdapter, SFTTrainer)
    return {
        package: version(package) for package in ("torch", "accelerate", "diffusers", "flash-attn")
    }
