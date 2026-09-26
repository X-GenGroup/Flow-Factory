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

"""Small native transformers exercising complete TDM cycles and real role storage.

Also runnable with two CPU torchrun ranks, or TDM_TEST_CUDA=1 for CUDA/bfloat16.
No pretrained weights, text encoders, datasets, or VAE are needed.
"""

import os
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import InitProcessGroupKwargs
from peft import LoraConfig, get_peft_model

from diffusers import FluxTransformer2DModel, SD3Transformer2DModel
from flow_factory.hparams import TDMTrainingArguments
from flow_factory.models.flux.flux1 import Flux1Adapter, Flux1Sample
from flow_factory.models.model_bundle import ModelBundle, RoutedComponentProxy
from flow_factory.models.stable_diffusion.sd3_5 import SD3_5Adapter, SD3_5Sample
from flow_factory.samples import ComponentTrajectory, StructuredTrajectory
from flow_factory.scheduler import FlowMatchEulerDiscreteSDEScheduler, SchedulerGroup
from flow_factory.trainers.distillation.tdm import TDMTrainer
from flow_factory.trainers.distillation.tdm_time_sampling import capture_generation_shift
from flow_factory.trainers.role_optimization import (
    OptimizationRole,
    RoleOptimizationCoordinator,
    RoleOptimizerConfig,
)


def native_trainer(kind, mode, distribution, query_interval):
    """Prepare native full/PEFT roles through the same bundle, EMA, and optimizer owners."""
    handlers = [DistributedDataParallelKwargs(find_unused_parameters=True)]
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        handlers.append(
            InitProcessGroupKwargs(
                backend="nccl" if os.environ.get("TDM_TEST_CUDA") == "1" else "gloo"
            )
        )
    accelerator = Accelerator(
        cpu=os.environ.get("TDM_TEST_CUDA") != "1",
        gradient_accumulation_steps=8,
        kwargs_handlers=handlers,
    )
    args = TDMTrainingArguments(
        num_inference_steps=4,
        tdm_query_distribution=distribution,
        tdm_query_interval=query_interval,
        per_device_batch_size=1,
        gradient_accumulation_steps=8,
        ttur_fake_updates=2,
        guidance_scale=1.0,
        real_guidance_scale=4.5,
        enable_gradient_checkpointing=True,
        ema_update_interval=1,
        ema_device="cpu",
        latent_storage_dtype="fp32",
        replay_atol=1e-4,
        replay_rtol=1e-4,
    )
    common = dict(
        in_channels=4,
        out_channels=4,
        num_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=16,
        pooled_projection_dim=8,
    )
    if kind == "sd3":
        model = SD3Transformer2DModel(
            **common, sample_size=4, patch_size=2, caption_projection_dim=16, pos_embed_max_size=4
        )
        cls = SD3_5Adapter
    else:
        model = FluxTransformer2DModel(
            **common, num_single_layers=1, axes_dims_rope=(2, 2, 4), guidance_embeds=True
        )
        cls = Flux1Adapter
    model.enable_gradient_checkpointing()
    if mode == "lora":
        model = get_peft_model(
            model, LoraConfig(r=2, lora_alpha=2, target_modules=["to_q", "to_v"])
        )
    model.to(accelerator.device)
    adapter = object.__new__(cls)
    adapter.accelerator = accelerator
    adapter.training_args = args
    adapter.model_args = SimpleNamespace(
        finetune_type=mode,
        target_components=["transformer"],
        trainable_parameters_dtype=torch.float32,
    )
    adapter.target_module_map = {"transformer": "all" if mode == "full" else ["to_q", "to_v"]}
    components = {"transformer": model}
    adapter.pipeline = SimpleNamespace()
    adapter.component_runtime = SimpleNamespace(
        get_component=components.__getitem__,
        get_canonical_component=components.__getitem__,
        set_component_override=components.__setitem__,
        declared_component_names=("transformer",),
    )
    adapter.scheduler = FlowMatchEulerDiscreteSDEScheduler(dynamics_type="ODE", shift=3)
    adapter.scheduler_group = SchedulerGroup({"latent": adapter.scheduler}, primary_name="latent")
    adapter.declare_component_variants(("generator", "fake"))
    registry = adapter.component_variant_registry
    members = registry.bundle_members()
    configs = {
        name: RoleOptimizerConfig(
            role_name=name,
            learning_rate=1e-3,
            adam_betas=(0.0, 0.9),
            adam_weight_decay=0.0,
            adam_epsilon=1e-8,
            max_grad_norm=1.0,
        )
        for name in registry.variant_names
    }
    optimizer = torch.optim.AdamW(
        [
            dict(
                params=registry.parameters(name),
                role_name=name,
                lr=1e-3,
                betas=(0.0, 0.9),
                weight_decay=0.0,
            )
            for name in registry.variant_names
        ]
    )
    bundle, optimizer = accelerator.prepare(ModelBundle(members), optimizer)
    adapter.set_component(
        "transformer", RoutedComponentProxy(bundle, "transformer", registry, members)
    )
    adapter._init_ema()
    adapter._init_ref_parameters()
    trainer = object.__new__(TDMTrainer)
    trainer.adapter, trainer.accelerator = adapter, accelerator
    trainer.training_args, trainer.model_args = args, adapter.model_args
    trainer.config = SimpleNamespace()
    trainer.model_bundle, trainer.optimizer = bundle, optimizer
    trainer.optimization_roles = {
        name: OptimizationRole(configs[name], tuple(optimizer.param_groups[i]["params"]), (i,))
        for i, name in enumerate(registry.variant_names)
    }
    trainer.role_optimization = RoleOptimizationCoordinator(
        accelerator, bundle, optimizer, trainer.optimization_roles
    )
    trainer.autocast = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if accelerator.device.type == "cuda"
        else nullcontext
    )
    trainer.log_args = SimpleNamespace(verbose=False)
    trainer.step = trainer.epoch = 0
    trainer._tdm_generation_provenance = {}
    trainer._initialize_snapshots()
    return trainer


@torch.no_grad()
def rollout(trainer, kind):
    """Store all actual native Euler boundaries, including the terminal latent."""
    adapter = trainer.adapter
    device = trainer.accelerator.device
    adapter.eval()
    batch = dict(
        prompt_embeds=torch.randn(1, 2, 16, device=device),
        pooled_prompt_embeds=torch.randn(1, 8, device=device),
    )
    if kind == "sd3":
        batch.update(
            negative_prompt_embeds=torch.randn(1, 2, 16, device=device),
            negative_pooled_prompt_embeds=torch.randn(1, 8, device=device),
        )
        x = torch.randn(1, 4, 4, 4, device=device)
        sample_cls = SD3_5Sample
    else:
        batch["img_ids"] = torch.zeros(4, 3, device=device)
        x = torch.randn(1, 4, 4, device=device)
        sample_cls = Flux1Sample
    with capture_generation_shift(adapter.scheduler) as shifts:
        adapter.scheduler.set_timesteps(sigmas=[1.0, 0.75, 0.5, 0.25], device=device)
    sigmas = adapter.scheduler.sigmas
    states = [x[0]]
    with adapter.use_component_variant("generator"):
        for start, end in zip(sigmas[:-1], sigmas[1:]):
            with trainer.autocast():
                result = adapter.forward(
                    t=start.reshape(1) * 1000,
                    t_next=end.reshape(1) * 1000,
                    latents=x,
                    guidance_scale=1.0,
                    compute_log_prob=False,
                    return_kwargs=["next_latents_mean"],
                    **batch,
                )
            x = result.next_latents_mean
            states.append(x[0])
    fields = {name: value if name == "img_ids" else value[0] for name, value in batch.items()}
    sample = sample_cls(
        **fields,
        trajectory=StructuredTrajectory(
            components={
                "latent": ComponentTrajectory(
                    states=torch.stack(states),
                    timesteps=sigmas * 1000,
                    sigmas=sigmas,
                    state_index_map=torch.arange(5, device=device),
                )
            }
        ),
    )
    trainer._record_tdm_generation_provenance([sample], shifts[0])
    return sample


@pytest.mark.parametrize("kind", ["sd3", "flux"])
@pytest.mark.parametrize("mode", ["full", "lora"])
@pytest.mark.parametrize(
    "distribution", ["actual_uniform", "conditional_logit_normal", "source_uniform"]
)
@pytest.mark.parametrize("query_interval", ["reverse", "trajectory"])
def test_native_tdm_sampling_complete_cycles(kind, mode, distribution, query_interval):
    torch.manual_seed(73)
    trainer = native_trainer(kind, mode, distribution, query_interval)
    registry = trainer.adapter.component_variant_registry
    initial = {
        name: [p.detach().clone() for p in registry.parameters(name)]
        for name in registry.variant_names
    }
    for iteration in range(2):
        torch.manual_seed(100 + iteration + trainer.accelerator.process_index * 7)
        samples = [[rollout(trainer, kind)] for _ in range(2)]
        trainer.optimize(samples)
        assert trainer.step == iteration + 1
        assert trainer.role_optimization.roles["generator"].step == iteration + 1
        assert trainer.role_optimization.roles["fake"].step == 2 * (iteration + 1)
        trainer.adapter.ema_step(trainer.step)
        for role in trainer.role_optimization.roles.values():
            assert torch.isfinite(torch.tensor(role.last_grad_norm)) and role.last_grad_norm > 0
    for name in registry.variant_names:
        assert any(
            not torch.equal(p, before)
            for p, before in zip(registry.parameters(name), initial[name])
        )
        for p in registry.parameters(name):
            gathered = trainer.accelerator.gather(p.detach().reshape(1, -1))
            torch.testing.assert_close(gathered, gathered[:1].expand_as(gathered))
    assert trainer.adapter.ema_wrapper.state_dict()["num_updates"] == 2
    trainer.accelerator.free_memory()
