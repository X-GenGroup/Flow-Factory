# Copyright 2026 Jayce-Ping
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Native SenseNova U1.0/U1.5 structures for prefix and spatial-head controls."""

import torch


def build_denoiser(*, pixel_head=True, native_heads=False):
    from flow_factory.models.sensenova.modeling.neo_unify.configuration_neo_chat import (
        NEOChatConfig,
    )
    from flow_factory.models.sensenova.modeling.neo_unify.modeling_neo_chat import NEOChatModel
    from flow_factory.models.sensenova.pipeline import SenseNovaDenoiser

    heads, kv_heads, head_dim = (32, 8, 128) if native_heads else (4, 2, 8)
    hidden = heads * head_dim
    config = NEOChatConfig(
        vision_config=dict(
            architectures=["NEOVisionModel"],
            patch_size=16,
            hidden_size=64 if native_heads else 16,
            llm_hidden_size=hidden,
            downsample_ratio=0.5,
            max_position_embeddings_vision=10000,
            num_channels=3,
            rope_theta_vision=10000.0,
        ),
        llm_config=dict(
            architectures=["Qwen3ForCausalLM"],
            hidden_size=hidden,
            intermediate_size=1024 if native_heads else 64,
            num_hidden_layers=2,
            num_attention_heads=heads,
            num_key_value_heads=kv_heads,
            head_dim=head_dim,
            vocab_size=384,
            max_position_embeddings=262144,
            rope_theta=5000000.0,
            rope_theta_hw=10000.0,
            max_position_embeddings_hw=10000,
            attention_bias=False,
            attention_dropout=0.0,
            rms_norm_eps=1e-6,
            pad_token_id=0,
            eos_token_id=1,
            tie_word_embeddings=False,
        ),
        downsample_ratio=0.5,
        template="neo1_0",
        patch_size=16,
        fm_head_layers=2,
        fm_head_dim=1536,
        fm_head_mlp_ratio=1,
        use_pixel_head=pixel_head,
        t_eps=0.05,
        add_noise_scale_embedding=True,
        noise_scale=1.0,
        noise_scale_mode="resolution",
        noise_scale_base_image_seq_len=64,
        noise_scale_max_value=16.0,
        concat_time_token_num=0,
        time_schedule="standard",
        time_shift_type="exponential",
        base_shift=0.5,
        max_shift=1.15,
        base_image_seq_len=64,
        max_image_seq_len=4096,
    )
    torch.manual_seed(812)
    return SenseNovaDenoiser(NEOChatModel(config))


def install_lora(module):
    from peft import LoraConfig, get_peft_model

    from flow_factory.models.sensenova.sensenova import SenseNovaAdapter

    torch.manual_seed(915)
    return get_peft_model(
        module,
        LoraConfig(
            r=2,
            lora_alpha=2,
            init_lora_weights=False,
            target_modules=SenseNovaAdapter.default_target_modules.fget(
                object.__new__(SenseNovaAdapter)
            ),
        ),
    )


def prefix_and_inputs(module, *, side=96, replica=0):
    """Use real native text prefill and all three nonconstant RoPE axes."""
    from flow_factory.models.sensenova.modeling.neo_unify.modeling_neo_chat import (
        prepare_flash_kv_cache,
    )
    from flow_factory.models.sensenova.modeling.neo_unify.modeling_qwen3 import (
        create_block_causal_mask,
    )
    from flow_factory.models.sensenova.pipeline import SenseNovaTextPrefix
    from flow_factory.models.tensor_parallel import peel_peft_wrapper

    raw = peel_peft_wrapper(module)
    device, dtype = raw.device, raw.dtype
    length = 9 + 2 * replica
    time = torch.arange(length, device=device)
    indexes = torch.stack([time, torch.zeros_like(time), torch.zeros_like(time)])
    ids = (torch.arange(length, device=device)[None] + 2 + replica) % 384
    with torch.no_grad():
        result = module(
            prefix=SenseNovaTextPrefix(
                ids, indexes, {"full_attention": create_block_causal_mask(indexes[0]).to(dtype)}
            )
        )
        cache, prefix_hidden = result.past_key_values, result.hidden_states
    grid = side // 32
    prepare_flash_kv_cache(cache, current_len=grid * grid, batch_size=1)
    inputs = dict(
        latents=torch.randn(
            1, 3, side, side, generator=torch.Generator().manual_seed(991 + replica)
        ).to(device=device, dtype=dtype),
        timestep=torch.tensor(0.37, device=device, dtype=dtype),
        past_key_values=cache,
        indexes_image=raw.model._build_t2i_image_indexes(grid, grid, length, device=device),
        attention_mask={"full_attention": None},
        image_size=(side, side),
        noise_scale=1.0,
    )
    return inputs, prefix_hidden


def build_tokenizer(root):
    from pathlib import Path

    from tokenizers import ByteLevelBPETokenizer
    from transformers import PreTrainedTokenizerFast

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    special = [
        "<pad>",
        "<unk>",
        "<|im_start|>",
        "<|im_end|>",
        "<img>",
        "</img>",
        "<IMG_CONTEXT>",
        "<image>",
        "<think>",
        "</think>",
    ]
    core = ByteLevelBPETokenizer()
    core.train_from_iterator(
        ["a red image", "a blue image", "Combine Image-1 and Image-2 into one picture."],
        vocab_size=384,
        special_tokens=special,
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=core._tokenizer,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="<|im_end|>",
        additional_special_tokens=special[2:],
    )
    tokenizer.save_pretrained(root)
    return tokenizer


def condition_images(replica):
    import numpy as np
    from PIL import Image

    return [
        Image.fromarray(
            np.random.default_rng(781 + replica * 10 + index).integers(
                0, 256, (32, width, 3), dtype=np.uint8
            )
        )
        for index, width in enumerate((32, 64))
    ]


def make_input_adapter(module, tokenizer, device, *, accelerator=None, transformer=None):
    from types import SimpleNamespace

    from accelerate.utils import DistributedType

    from flow_factory.models.sensenova.sensenova import SenseNovaAdapter

    adapter = object.__new__(SenseNovaAdapter)
    adapter.accelerator = accelerator or SimpleNamespace(
        device=torch.device(device),
        distributed_type=DistributedType.NO,
        is_fsdp2=False,
        unwrap_model=lambda value: value,
    )
    adapter.pipeline = SimpleNamespace(tokenizer=tokenizer, transformer=module)
    adapter.component_runtime = SimpleNamespace(
        get_component=lambda name: transformer if transformer is not None else module
    )
    return adapter


def conditioned_inputs(adapter, replica, step, *, side=96):
    context = adapter._build_context(
        "Combine the red and blue images.", (side, side), 3.0, 1.5, condition_images(replica)
    )
    common = dict(
        latents=torch.randn(
            1, 3, side, side, generator=torch.Generator().manual_seed(871 + replica * 10 + step)
        ).to(device=adapter.device, dtype=adapter.transformer.dtype),
        timestep=torch.tensor(0.37 - 0.07 * step, device=adapter.device),
        image_size=(side, side),
        noise_scale=1.0,
    )
    calls = []
    observation = {
        "reference_sizes": [image.size for image in condition_images(replica)],
        "prefix_lengths": [],
        "spatial_axes": [],
    }
    for prefix in ("", "img_", "uncond_"):
        cache = context[prefix + "past_key_values"]
        assert cache is not None
        assert all(
            not layer.keys.requires_grad and not layer.values.requires_grad
            for layer in cache.layers
        )
        indexes = context[prefix + "indexes_image"]
        calls.append(
            dict(
                common,
                past_key_values=cache,
                indexes_image=indexes,
                attention_mask=context[prefix + "attention_mask"],
            )
        )
        observation["prefix_lengths"].append(cache.get_seq_length())
        observation["spatial_axes"].append([int(indexes[index].max()) for index in range(3)])
    return calls, observation


def build_fixture(root, *, pixel_head):
    from pathlib import Path

    root = Path(root)
    model_root = root / "model"
    build_tokenizer(model_root)
    module = build_denoiser(pixel_head=pixel_head, native_heads=True)
    module.model.save_pretrained(model_root, safe_serialization=True)
    data = root / "data"
    data.mkdir(exist_ok=True)
    for replica in range(8):
        for index, image in enumerate(condition_images(replica)):
            image.save(data / f"{replica}-{index}.png")


if __name__ == "__main__":
    import sys

    build_fixture(sys.argv[1], pixel_head=sys.argv[2] == "u15")


def inactive_parameter_ids(module):
    """Exact native modules used only by detached prefix construction or text logits."""
    from flow_factory.models.tensor_parallel import peel_peft_wrapper

    raw = peel_peft_wrapper(module)
    model = raw.model
    language = model.language_model
    roots = [model.vision_model, language.lm_head, language.model.embed_tokens, language.model.norm]
    for block in language.model.layers:
        roots.extend([block.mlp, block.input_layernorm, block.post_attention_layernorm])
        roots.extend(
            getattr(block.self_attn, name)
            for name in (
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "q_norm",
                "q_norm_hw",
                "k_norm",
                "k_norm_hw",
            )
        )
    return {id(parameter) for root in roots for parameter in root.parameters()}


def qualify_native_dependencies():
    """Run the production CP dependency gate before loading this model probe."""
    from importlib.metadata import version

    from flow_factory.hparams import Arguments
    from flow_factory.models.sensenova.sensenova import SenseNovaAdapter
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
            "model": {"model_type": "sensenova", "target_components": ["transformer"]},
            "train": {"trainer_type": "sft", "gradient_accumulation_steps": 1, "ema_decay": 0},
            "log": {"save_model_only": True, "save_dir": ".scratch/sensenova-preflight"},
            "eval": {"eval_freq": 0},
            "acceleration": {
                "shared": [{"name": "attention_backend", "params": {"backend": "native"}}]
            },
        }
    )
    validate_context_parallel_config(config, SenseNovaAdapter, SFTTrainer)
    return {
        package: version(package)
        for package in ("torch", "accelerate", "diffusers", "transformers")
    }
