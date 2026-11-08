# Copyright 2026 Jayce-Ping
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Packed Bagel numerical controls; CPU math is not a native FlashAttention GPU receipt."""

import copy
import json
import traceback
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from conftest import requires_gloo, run_gloo_ranks


def _cpu_flash_reference(q, k, v, cu_seqlens_q, cu_seqlens_k, causal=False, **kwargs):
    """Independent per-sample SDPA oracle for the CUDA-only native varlen operation."""
    results = []
    for sample in range(cu_seqlens_q.numel() - 1):
        qs, qe = int(cu_seqlens_q[sample]), int(cu_seqlens_q[sample + 1])
        ks, ke = int(cu_seqlens_k[sample]), int(cu_seqlens_k[sample + 1])
        query = q[qs:qe].transpose(0, 1)[None]
        key = k[ks:ke].repeat_interleave(q.shape[1] // k.shape[1], dim=1).transpose(0, 1)[None]
        value = v[ks:ke].repeat_interleave(q.shape[1] // v.shape[1], dim=1).transpose(0, 1)[None]
        mask = None
        if causal:
            # Native Flash varlen uses bottom-right causal alignment for cached Q/KV.
            mask = torch.arange(ke - ks)[None, :] <= torch.arange(qe - qs)[:, None] + (ke - ks) - (
                qe - qs
            )
        results.append(
            torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=mask)[
                0
            ].transpose(0, 1)
        )
    return torch.cat(results)


def _merge_indexes(query_lengths, prefix_lengths):
    queries, prefixes = [], []
    offset = 0
    for query, prefix in zip(query_lengths, prefix_lengths):
        prefixes.extend(range(offset, offset + prefix))
        queries.extend(range(offset + prefix, offset + prefix + query))
        offset += prefix + query
    return torch.tensor(queries, dtype=torch.long), torch.tensor(prefixes, dtype=torch.long)


@torch.no_grad()
def _prefill(module):
    from flow_factory.models.bagel.modeling.bagel.qwen2_navit import NaiveCache

    generator = torch.Generator().manual_seed(77)
    cache = NaiveCache(2)
    lengths = [0, 0]
    # Native whole-prefix updates include one absent reference: no query from sample0.
    for round_index, queries in enumerate(([3, 5], [4, 6], [0, 4])):
        query_indexes, prefix_indexes = _merge_indexes(queries, lengths)
        total = sum(queries)
        kwargs = dict(
            packed_query_sequence=torch.randn(
                total, module.config.hidden_size, generator=generator
            ),
            query_lens=torch.tensor(queries),
            packed_query_position_ids=torch.cat(
                [torch.arange(prefix, prefix + query) for prefix, query in zip(lengths, queries)]
            ),
            packed_query_indexes=query_indexes,
            past_key_values=cache,
            key_values_lens=torch.tensor(lengths),
            packed_key_value_indexes=prefix_indexes,
            update_past_key_values=True,
            is_causal=round_index == 0,
            mode="und",
            packed_text_ids=torch.randint(1, 32, (total,), generator=generator),
            packed_text_indexes=torch.arange(total),
        )
        cache = module(**kwargs).past_key_values
        lengths = [old + query for old, query in zip(lengths, queries)]
    assert lengths == [7, 15]
    assert all(
        not value.requires_grad
        for value in (*cache.key_cache.values(), *cache.value_cache.values())
    )
    return cache, lengths


def _inputs(cache, lengths, width):
    generator = torch.Generator().manual_seed(78)
    queries = [6, 10]
    query_indexes, prefix_indexes = _merge_indexes(queries, lengths)
    return dict(
        packed_query_sequence=torch.randn(16, width, generator=generator),
        query_lens=torch.tensor(queries),
        packed_query_position_ids=torch.cat(
            [torch.arange(prefix, prefix + query) for prefix, query in zip(lengths, queries)]
        ),
        packed_query_indexes=query_indexes,
        past_key_values=cache,
        key_values_lens=torch.tensor(lengths),
        packed_key_value_indexes=prefix_indexes,
        update_past_key_values=False,
        is_causal=False,
        mode="gen",
        packed_text_ids=torch.tensor([1, 2, 1, 2]),
        packed_text_indexes=torch.tensor([0, 5, 6, 15]),
        packed_vae_token_indexes=torch.tensor([1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14]),
    )


def _cpu_worker(rank, store_path, result_dir, options):
    result = {"rank": rank, **options}
    dist.init_process_group("gloo", init_method=f"file://{store_path}", rank=rank, world_size=4)
    try:
        from diffusers.models._modeling_parallel import ContextParallelConfig
        from peft import LoraConfig, get_peft_model
        from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
        from torch.distributed.tensor import DTensor

        from flow_factory.models.bagel.bagel import BagelAdapter
        from flow_factory.models.bagel.modeling.bagel import qwen2_navit as native
        from flow_factory.models.tensor_parallel import apply_tensor_parallel, peel_peft_wrapper

        # Only the CPU oracle replaces this CUDA-only operation; actual native GPU
        # controls will retain the installed flash_attn_varlen_func unchanged.
        native.flash_attn_varlen_func = _cpu_flash_reference
        torch.manual_seed(19)
        module = native.Qwen2ForCausalLM(
            native.Qwen2Config(
                vocab_size=32,
                hidden_size=224 if options["tp"] == 4 else 64,
                intermediate_size=128,
                num_hidden_layers=2,
                num_attention_heads=28 if options["tp"] == 4 else 4,
                num_key_value_heads=4 if options["tp"] == 4 else 2,
                layer_module="Qwen2MoTDecoderLayer",
                qk_norm=True,
                freeze_und=False,
                pad_token_id=0,
                tie_word_embeddings=False,
            )
        )
        if options["mode"] == "lora":
            targets = BagelAdapter.default_target_modules.fget(object.__new__(BagelAdapter))
            module = get_peft_model(
                module,
                LoraConfig(r=2, lora_alpha=2, target_modules=targets, init_lora_weights=False),
            )
        reference = copy.deepcopy(module)
        whole = init_device_mesh("cpu", (options["cp"], options["tp"]), mesh_dim_names=("cp", "tp"))
        raw = peel_peft_wrapper(module)
        plan = BagelAdapter.tensor_parallel_plan()
        if options["sp"]:
            plan = {**plan, **BagelAdapter.sequence_parallel_plan()}
        apply_tensor_parallel(raw, whole["tp"], plan)
        BagelAdapter.validate_context_parallel_module(raw, options["cp"], options["tp"])
        if options["cp"] > 1 or options["sp"]:
            BagelAdapter.configure_context_parallel_module(raw)
        raw.set_attention_backend("native")
        cpmesh = DeviceMesh(
            "cpu", whole.mesh.T.unsqueeze(1), mesh_dim_names=("tensor", "ring", "ulysses")
        )
        if options["cp"] > 1:
            BagelAdapter.install_context_parallel_module(
                raw, ContextParallelConfig(ring_degree=1, ulysses_degree=options["cp"], mesh=cpmesh)
            )
        result["cases"] = []
        for checkpoint in (False, True):
            if checkpoint:
                module.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
                reference.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            module.zero_grad(set_to_none=True)
            reference.zero_grad(set_to_none=True)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                cache, lengths = _prefill(module)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                reference_cache, reference_lengths = _prefill(reference)
                assert lengths == reference_lengths
                for actual_cache, expected_cache in zip(
                    (cache.key_cache, cache.value_cache),
                    (reference_cache.key_cache, reference_cache.value_cache),
                ):
                    for index in actual_cache:
                        expected_value = expected_cache[index]
                        if options["sp"]:
                            expected_value = expected_value.chunk(options["tp"], dim=1)[
                                rank % options["tp"]
                            ]
                        torch.testing.assert_close(
                            actual_cache[index], expected_value, rtol=0.03, atol=0.03
                        )
                inputs = _inputs(cache, lengths, raw.config.hidden_size)
                refinputs = _inputs(reference_cache, lengths, raw.config.hidden_size)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                expected = reference(**refinputs).packed_query_sequence
            with torch.autocast("cpu", dtype=torch.bfloat16):
                actual = module(**inputs).packed_query_sequence
            torch.testing.assert_close(actual, expected, rtol=0.025, atol=0.02)
            # A squared final RMSNorm output is nearly constant and measures
            # cancellation error rather than a denoising gradient. Use an
            # independent target so every routed branch has a useful oracle.
            target = torch.randn(expected.shape, generator=torch.Generator().manual_seed(93))
            (actual - target).square().mean().backward()
            (expected - target).square().mean().backward()
            worst = 0.0
            active = []
            inactive = []
            for (name, p), (refname, q) in zip(
                module.named_parameters(), reference.named_parameters()
            ):
                assert name == refname and (p.grad is None) == (q.grad is None), (
                    name,
                    p.grad is None,
                    q.grad is None,
                )
                if p.grad is None:
                    if p.requires_grad:
                        inactive.append(name)
                    continue
                gradient = p.grad.full_tensor() if isinstance(p.grad, DTensor) else p.grad
                gradient = gradient.clone()
                dist.all_reduce(gradient, group=whole["cp"].get_group())
                torch.testing.assert_close(gradient, q.grad, rtol=0.05, atol=0.015, msg=name)
                error = float((gradient - q.grad).norm() / q.grad.norm().clamp_min(1e-15))
                worst = max(error, worst)
                assert error < 0.025, (name, error)
                active.append(name)
            state = getattr(raw, "_ff_bagel_parallel_state", None)
            assert state is None or state.call.get() is None
            result["cases"].append(
                {
                    "checkpoint": checkpoint,
                    "active": active,
                    "inactive": inactive,
                    "worst_relative_gradient_error": worst,
                }
            )
        result["status"] = "pass"
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result, indent=2))
        dist.destroy_process_group()


@requires_gloo
@pytest.mark.parametrize("mode", ["full", "lora"])
@pytest.mark.parametrize("sp", [False, True])
@pytest.mark.parametrize("cp,tp", [(2, 2), (1, 4)])
def test_cpu_bagel_packed_native_collectives(tmp_path, mode, sp, cp, tp):
    results = run_gloo_ranks(_cpu_worker, tmp_path, 4, {"mode": mode, "sp": sp, "cp": cp, "tp": tp})
    assert all(row.get("status") == "pass" and "error" not in row for row in results), results


def test_native_bagel_condition_fixture_preserves_order_and_zero_query_round(tmp_path, monkeypatch):
    from bagel_cp_fixture import (
        build_pipeline,
        build_tokenizer,
        forward_calls,
        make_input_adapter,
        model_call_inputs,
    )
    from test_bagel_context_parallel import _native_ragged_control

    from flow_factory.models.bagel.modeling.bagel import qwen2_navit, siglip_navit
    from flow_factory.models.model_bundle import ModelBundle

    # CPU fixture semantics only. CUDA selectors must use both installed Flash kernels.
    monkeypatch.setattr(qwen2_navit, "flash_attn_varlen_func", _cpu_flash_reference)
    monkeypatch.setattr(siglip_navit, "flash_attn_varlen_func", _cpu_flash_reference)
    pipeline = build_pipeline(heads=4, kv_heads=2, head_dim=16)
    tokenizer = build_tokenizer(tmp_path / "tokenizer")
    assert tokenizer.encode("a red image") != tokenizer.encode("a blue small image")
    adapter = make_input_adapter(pipeline, tokenizer, "cpu")
    calls, observed = model_call_inputs(adapter, 0, 0, ragged=True)
    assert observed["reference_counts"] == [1, 2]
    assert observed["query_lengths"] == [18, 18]
    assert observed["prefix_lengths"] == [[18, 42], [12, 36], [6, 6]]
    output = forward_calls(pipeline.transformer, calls)
    assert output.shape == (3, 36, 64) and torch.isfinite(output).all()
    assert not torch.equal(output[0], output[1]) and not torch.equal(output[0], output[2])
    for call in calls:
        assert (
            sum(call["key_values_lens"].tolist()) == call["past_key_values"].key_cache[0].shape[0]
        )
    assert adapter.requires_preserved_replay_batch_composition is True
    original_forward = pipeline.transformer.forward
    generation_packs = []

    def require_original_pack(*args, **kwargs):
        if kwargs.get("mode") == "gen" and not kwargs.get("update_past_key_values", True):
            lengths = kwargs["query_lens"].tolist()
            assert lengths == observed["query_lengths"], "Bagel control must preserve its B2 pack"
            generation_packs.append(lengths)
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(pipeline.transformer, "forward", require_original_pack)
    ragged = _native_ragged_control(ModelBundle({"transformer": pipeline.transformer}), adapter, 0)
    assert generation_packs == [[18, 18]] * 3
    assert ragged["reference_counts"] == [1, 2] and ragged["active_gradients"] > 0
    assert ragged["requires_preserved_replay_batch_composition"] is True
    assert ragged["original_pack_preserved"] and ragged["cache_unchanged"]
    assert ragged["original_pack_forward_calls"] == 3
    assert ragged["finite_output"] and ragged["nonzero_gradients"]


def test_bagel_fixture_position_table_covers_native_ordered_reference_preprocessing():
    from bagel_cp_fixture import build_pipeline, condition_images

    from flow_factory.models.bagel.data.transforms import ImageTransform
    from flow_factory.models.bagel.modeling.bagel.siglip_navit import SiglipVisionEmbeddings

    # Keep the native lifecycle transform; small mathematical-probe images conceal
    # the mismatch between stride-70 positions and the old 16x16 position table.
    pipeline = build_pipeline(heads=4, kv_heads=2, head_dim=16)
    model = pipeline.bagel
    embeddings = model.vit_model.vision_model.embeddings
    transform = ImageTransform(980, 224, 14)
    old_config = copy.deepcopy(embeddings.config)
    old_config.image_size = 224
    old_embeddings = SiglipVisionEmbeddings(old_config)
    old_embeddings.convert_conv2d_to_linear(old_config)
    assert old_embeddings.position_embedding.num_embeddings == 256

    lengths, positions = [0], [0]
    images = condition_images(ragged=False)[0]
    assert [image.size for image in images] == [(32, 32), (64, 32)]
    for image, token_count, max_position in zip(images, (256, 512), (1065, 1081)):
        inputs, lengths, positions = model.prepare_vit_images(
            lengths,
            positions,
            [image],
            transform,
            {"start_of_image": 1, "end_of_image": 2},
        )
        pixels = inputs["packed_vit_tokens"]
        position_ids = inputs["packed_vit_position_ids"]
        assert inputs["vit_token_seqlens"].tolist() == [token_count]
        assert position_ids.min().item() == 0 and position_ids.max().item() == max_position
        with torch.no_grad():
            output = embeddings(pixels, position_ids)
        assert output.shape == (token_count, 64) and torch.isfinite(output).all()
        with pytest.raises(IndexError, match="index out of range"):
            old_embeddings(pixels, position_ids)
    assert embeddings.position_embedding.num_embeddings == model.vit_max_num_patch_per_side**2
    assert lengths == [256 + 512 + 4] and positions == [2]


def test_bagel_lifecycle_config_keeps_native_cfg_without_unsupported_negative_prompt(
    tmp_path, monkeypatch
):
    from bagel_coupled_fixture import make_config

    monkeypatch.setenv("WORLD_SIZE", "8")
    config = make_config(
        tmp_path,
        tmp_path / "result",
        dict(
            algorithm="grpo-guard",
            training_mode="lora",
            dp_size=2,
            tp_size=2,
            cp_size=2,
            sequence_parallel=True,
        ),
    )
    assert config.model_args.model_type == "bagel"
    for arguments in (config.training_args, config.eval_args):
        assert "negative_prompt" not in arguments.extra_kwargs
        assert arguments.extra_kwargs["cfg_text_scale"] == 2.0
        assert arguments.extra_kwargs["cfg_img_scale"] == 1.5
    assert config.parallelism_args.sequence_parallel is True
