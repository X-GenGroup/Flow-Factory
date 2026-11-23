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

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml
from torch.distributed.fsdp import FSDPModule
from torch.utils.checkpoint import checkpoint
from transformers import CLIPConfig, CLIPModel, CLIPTextConfig, CLIPVisionConfig

from diffusers import AutoencoderKL
from flow_factory.contracts.topology import ParallelTopology
from flow_factory.hparams import Arguments
from flow_factory.hparams.component_execution import ComponentExecutionPolicy
from flow_factory.models.auxiliary_fsdp import prepare_reward_gradient_auxiliaries
from flow_factory.models.component_execution import (
    ComponentExecution,
    ComponentExecutionPlan,
    OptimizationGradients,
    validate_component_execution,
)
from flow_factory.rewards.pick_score import PickScoreRewardModel


def _scorer():
    text = CLIPTextConfig(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_hidden_layers=1,
        max_position_embeddings=8,
        eos_token_id=2,
    )
    vision = CLIPVisionConfig(
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_hidden_layers=1,
        image_size=8,
        patch_size=4,
    )
    scorer = object.__new__(PickScoreRewardModel)
    scorer.model = (
        CLIPModel(
            CLIPConfig(text_config=text.to_dict(), vision_config=vision.to_dict(), projection_dim=4)
        )
        .eval()
        .requires_grad_(False)
    )
    scorer.device = torch.device("cpu")
    scorer.processor = _Processor()
    return scorer


class _Processor:
    image_processor = SimpleNamespace(
        do_resize=False,
        do_center_crop=False,
        do_normalize=False,
    )

    def __call__(self, *, text, **kwargs):
        return {"input_ids": torch.tensor([[1, 2]] * len(text))}


def _qualify(rank, store):
    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=2)
    try:
        torch.manual_seed(47)
        decoder = (
            AutoencoderKL(
                block_out_channels=(8,),
                layers_per_block=1,
                latent_channels=2,
                norm_num_groups=2,
                mid_block_add_attention=False,
            )
            .eval()
            .requires_grad_(False)
        )
        scorer = _scorer()
        dense_decoder = copy.deepcopy(decoder)
        dense_scorer = copy.deepcopy(scorer)
        execution = ComponentExecution(
            ComponentExecutionPolicy(placement="fsdp2", shard_group="data"),
            OptimizationGradients.INPUT,
        )
        overrides = {}
        adapter = SimpleNamespace(
            component_execution_plan=ComponentExecutionPlan(
                {"vae": execution}, {("score",): execution}
            ),
            _require_component=lambda name: {"vae": decoder}[name],
            set_component=overrides.__setitem__,
        )
        config = SimpleNamespace(get_identity_key=lambda: ("score",))
        trainer = SimpleNamespace(
            adapter=adapter,
            accelerator=SimpleNamespace(device=torch.device("cpu")),
            reward_models={"score": scorer, "alias": scorer},
            reward_loader=SimpleNamespace(
                get_reward_configs=lambda _: {"score": config, "alias": config}
            ),
        )
        assert prepare_reward_gradient_auxiliaries(trainer) == (
            "model.vae",
            "rewards.score",
            "rewards.alias",
        )
        assert overrides == {"vae": decoder}
        assert isinstance(decoder, FSDPModule)
        assert isinstance(scorer.model.vision_model, FSDPModule)

        policy = torch.nn.Conv2d(2, 2, 1)
        dense_policy = copy.deepcopy(policy)
        optimizers = [
            torch.optim.SGD(item.parameters(), lr=0.03) for item in (dense_policy, policy)
        ]
        torch.manual_seed(59 + rank)
        sample = torch.randn(1, 2, 8, 8)

        def step(policy, vae, reward, optimizer, recompute):
            optimizer.zero_grad()
            latent = policy(sample)
            latent.retain_grad()

            def decode(value):
                return vae.decode(value, return_dict=False)[0].sigmoid()

            image = checkpoint(decode, latent, use_reentrant=False) if recompute else decode(latent)
            scores = reward.score_differentiable(prompt=["prompt"], image=image).rewards
            (-scores.mean()).backward()
            assert latent.grad.norm() > 0
            updates = [p.grad.detach().clone() for p in policy.parameters()]
            optimizer.step()
            return image.detach(), latent.grad.detach(), scores.detach(), *updates

        for recompute in (False, True, False):
            expected = step(dense_policy, dense_decoder, dense_scorer, optimizers[0], recompute)
            result = step(policy, decoder, scorer, optimizers[1], recompute)
            for actual, reference in zip(result, expected, strict=True):
                torch.testing.assert_close(actual, reference, atol=2e-6, rtol=2e-5)
            for actual, reference in zip(
                policy.parameters(), dense_policy.parameters(), strict=True
            ):
                torch.testing.assert_close(actual, reference, atol=2e-6, rtol=2e-5)
        for parameter in (*decoder.parameters(), *scorer.model.parameters()):
            assert not parameter.requires_grad and parameter.grad is None

        # Detaching the image breaks the exact reward-to-policy path used above.
        output = decoder.decode(policy(sample), return_dict=False)[0].detach().sigmoid()
        assert not scorer.score_differentiable(
            prompt=["prompt"], image=output
        ).rewards.requires_grad
    finally:
        dist.destroy_process_group()


def test_native_frozen_vae_and_pickscore_input_gradients(tmp_path):
    mp.spawn(_qualify, args=(str(tmp_path / "store"),), nprocs=2, join=True)


def _config(model=None, rewards=None, eval_dataset=False):
    raw = yaml.safe_load(Path("examples/refl/lora/sd3_5/pickscore.yaml").read_text())
    raw["rewards"][0]["device"] = "cpu"
    if eval_dataset:
        raw["data"]["datasets"][0]["eval"] = {"enabled": True}
    raw["component_execution"] = {"model": model or {}, "rewards": rewards or {}}
    return Arguments.from_dict(raw)


@pytest.mark.parametrize("target", ["vae", "alignment"])
def test_preflight_requires_dp_only_fsdp2(monkeypatch, target):
    policy = {"placement": "fsdp2", "shard_group": "data"}
    section = {"model": {"vae": policy}} if target == "vae" else {"rewards": {"alignment": policy}}
    config = _config(**section)
    monkeypatch.setattr(
        "flow_factory.models.component_execution.parallel_topology",
        lambda _: ParallelTopology(dp_shard_size=2),
    )
    accelerator = SimpleNamespace(is_fsdp2=True, process_index=0, device=torch.device("cpu"))
    validate_component_execution(config, config.component_execution_args, accelerator)
    accelerator.is_fsdp2 = False
    with pytest.raises(ValueError, match="data-only FSDP2"):
        validate_component_execution(config, config.component_execution_args, accelerator)


def test_sharded_reward_cannot_share_eval_identity(monkeypatch):
    config = _config(
        rewards={"alignment": {"placement": "fsdp2", "shard_group": "data"}}, eval_dataset=True
    )
    monkeypatch.setattr(
        "flow_factory.models.component_execution.parallel_topology",
        lambda _: ParallelTopology(dp_shard_size=2),
    )
    accelerator = SimpleNamespace(is_fsdp2=True, process_index=0, device=torch.device("cpu"))
    with pytest.raises(ValueError, match="shares its sharded scorer with evaluation"):
        validate_component_execution(config, config.component_execution_args, accelerator)
