# New Model Guidance

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Step-by-Step Implementation](#step-by-step-implementation)
  - [Step 1: Define Sample Dataclass](#step-1-define-sample-dataclass)
  - [Step 2: Create Adapter Class](#step-2-create-adapter-class)
  - [Step 3: Configure Module Properties](#step-3-configure-module-properties)
  - [Step 4: Implement Encoding Methods](#step-4-implement-encoding-methods)
  - [Step 5: Implement `inference()`](#step-5-implement-inference)
  - [Step 6: Implement `forward()`](#step-6-implement-forward)
  - [Step 7: Register the Adapter](#step-7-register-the-adapter)
- [Advanced: Custom `preprocess_func`](#advanced-custom-preprocess_func)
- [Advanced: Pseudo-Pipeline for Non-Diffusers Models](#advanced-pseudo-pipeline-for-non-diffusers-models)
- [Data Format Conventions](#data-format-conventions)
- [Checklist](#checklist)

## Overview

Flow-Factory uses a **model adapter** pattern that wraps [diffusers](https://github.com/huggingface/diffusers) pipelines into a unified interface for RL training. Each adapter maps a diffusers pipeline to a consistent API that the training loop can call without knowing model-specific details.

The relationship is straightforward:

```
diffusers Pipeline               Flow-Factory Adapter
┌────────────────────┐           ┌──────────────────────┐
│ Flux2KleinPipeline │  wraps    │ Flux2KleinAdapter    │
│  ├─ text_encoder   │ ───────►  │  ├─ load_pipeline()  │
│  ├─ vae            │           │  ├─ encode_prompt()  │
│  ├─ transformer    │           │  ├─ encode_image()   │
│  ├─ scheduler      │           │  ├─ inference()      │
│  └─ __call__()     │           │  └─ forward()        │
└────────────────────┘           └──────────────────────┘
```

The adapter's `inference()` method corresponds to the pipeline's `__call__()`, while `forward()` extracts and wraps the single-step denoising logic from inside the pipeline's denoising loop.

> **Reference**: For a concrete example, compare [`Flux2KleinPipeline.__call__()`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux2/pipeline_flux2_klein.py#L609) with [`Flux2KleinAdapter.inference()`](https://github.com/X-GenGroup/Flow-Factory/blob/main/src/flow_factory/models/flux/flux2_klein.py#L374).


## Architecture

`BaseAdapter` (`src/flow_factory/models/abc.py`) provides all distributed training infrastructure out of the box:

| Capability | What `BaseAdapter` Handles |
|---|---|
| **Component management** | Automatic discovery of text encoders, VAEs, and transformers from the pipeline |
| **LoRA / Full fine-tuning** | `apply_lora()` with component-aware target module mapping |
| **Mixed precision** | Inference dtype for frozen components, master dtype for trainable parameters |
| **EMA** | EMA parameter snapshots for off-policy sampling and KL regularization |
| **Reference parameters** | Stored original weights for KL divergence computation |
| **Mode management** | `train()`, `eval()`, `rollout()` mode switching |
| **Checkpoint** | Save/load with LoRA-aware serialization |
| **Gradient checkpointing** | Automatic enablement on transformer components |

Your adapter only needs to implement the model-specific logic: **how to encode inputs, how to run inference, and how to perform a single denoising step**.
There are several PRs for you to reference: [FLUX2-Klein](#9), [Z-Image-Omni](#22).

## Step-by-Step Implementation

### Step 1: Define Sample Dataclass

Create a dataclass that extends `BaseSample` (or a task-specific variant) to carry model-specific fields through the training pipeline.

```python
# src/flow_factory/models/my_model/my_model.py
from dataclasses import dataclass
from typing import ClassVar, Optional
import torch
from flow_factory.samples import T2ISample  # or BaseSample, ImageConditionSample, T2VSample, ...

@dataclass
class MyModelSample(T2ISample):
    """Sample output for MyModel."""
    # Class-level: fields shared across all samples in a batch (not stacked)
    _shared_fields: ClassVar[frozenset[str]] = frozenset({})
    
    # Instance-level: model-specific fields (without batch dimension)
    latent_ids: Optional[torch.Tensor] = None      # e.g., (seq_len, 4)
    text_ids: Optional[torch.Tensor] = None         # e.g., (text_len, 4)
```

**Available base classes:**

| Base Class | Use Case | Extra Fields |
|---|---|---|
| `BaseSample` | Generic | `image`, `video`, `prompt`, `all_latents`, `log_probs`, ... |
| `T2ISample` | Text-to-image | Alias of `BaseSample` |
| `T2VSample` | Text-to-video | Alias of `BaseSample` |
| `ImageConditionSample` | Image-conditioned generation | `condition_images: List[Tensor]` |
| `VideoConditionSample` | Video-conditioned generation | `condition_videos: List[Tensor]` |

> See [`src/flow_factory/samples/samples.py`](src/flow_factory/samples/samples.py) for all available classes.

> **Key**: The `_shared_fields` class variable declares fields that are identical across a batch (e.g., `height`, `width`, `latent_index_map`). During `BaseSample.stack()`, shared fields take the first element instead of stacking.


### Step 2: Create Adapter Class

Subclass `BaseAdapter` and implement `load_pipeline()`:

```python
from flow_factory.models.abc import BaseAdapter
from flow_factory.hparams import Arguments
from accelerate import Accelerator
from diffusers import MyModelPipeline  # Your diffusers pipeline

class MyModelAdapter(BaseAdapter):
    def __init__(self, config: Arguments, accelerator: Accelerator):
        super().__init__(config, accelerator)
        # Type hints for IDE support (pipeline is loaded in super().__init__)
        self.pipeline: MyModelPipeline
    
    def load_pipeline(self) -> MyModelPipeline:
        """Load the diffusers pipeline. Called by BaseAdapter.__init__."""
        return MyModelPipeline.from_pretrained(
            self.model_args.model_name_or_path,
            low_cpu_mem_usage=False,  # Required for FSDP compatibility
        )
```

> See [Advanced: Pseudo-Pipeline for Non-Diffusers Models](#advanced-pseudo-pipeline-for-non-diffusers-models) for custom models.

### Step 3: Configure Module Properties

Override these properties to tell the framework which components to manage at each stage:

```python
class MyModelAdapter(BaseAdapter):
    # ...

    @property
    def default_target_modules(self) -> List[str]:
        """
        Default trainable layers for both LoRA and full fine-tuning.
        Inspect your transformer's named modules to identify attention and FFN layers.
        """
        return [
            "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
            "ff.linear_in", "ff.linear_out",
        ]
    
    @property
    def preprocessing_modules(self) -> List[str]:
        """
        Components needed during offline preprocessing (Stage 1).
        These are loaded onto GPU for encoding, then offloaded to free VRAM.
        Use group name 'text_encoders' to include all detected text encoders.
        """
        return ['text_encoders', 'vae']
    
    @property
    def inference_modules(self) -> List[str]:
        """
        Components that must remain on GPU during the training loop
        (sampling + optimization). Typically the denoising backbone + VAE decoder.
        """
        return ['transformer', 'vae']
```

**Defaults in `BaseAdapter`:**

| Property | Default |
|---|---|
| `default_target_modules` | `['to_q', 'to_k', 'to_v', 'to_out.0']` |
| `preprocessing_modules` | `['text_encoders', 'vae']` |
| `inference_modules` | `['transformer', 'vae']` |

Override only when your model deviates — for example, [WAN-T2V](src/flow_factory/models/wan/wan2_t2v.py) models need `['text_encoders', 'vae', 'image_encoder']` for preprocessing and conditionally include `transformer_2` for inference.

> **Tip**: Use `print(dict(self.pipeline.named_children()))` to discover available component names.


### Step 4: Implement Encoding Methods

Implement the three encoding methods. The default [`preprocess_func` in `BaseAdapter`](https://github.com/X-GenGroup/Flow-Factory/blob/main/src/flow_factory/models/abc.py#L1826) calls them independently:

```python
preprocess_func(prompt, images, videos, **kwargs):
    results = {}
    results.update(self.encode_prompt(prompt, **kwargs))     # if prompt is not None
    results.update(self.encode_image(images, **kwargs))      # if images is not None
    results.update(self.encode_video(videos, **kwargs))      # if videos is not None
    return results
```

#### `encode_prompt`

```python
def encode_prompt(
    self,
    prompt: Union[str, List[str]],  # Batched text prompts
    max_sequence_length: int = 512,
    **kwargs,
) -> Dict[str, Union[List[Any], torch.Tensor]]:
    """
    Encode text prompts into embeddings.
    
    Args:
        prompt: A single string or a batch of strings.
    
    Returns:
        Dict with batched tensors. Must include 'prompt_ids' for tokenizer-based
        reward models. Common keys:
        - 'prompt_ids': (B, seq_len) token IDs
        - 'prompt_embeds': (B, seq_len, D) hidden states
        - 'text_ids': (B, seq_len, 4) position IDs (model-specific)
    """
    prompt = [prompt] if isinstance(prompt, str) else prompt
    # ... encode using self.pipeline.text_encoder / self.tokenizer
    return {'prompt_ids': ..., 'prompt_embeds': ..., ...}
```

#### `encode_image`

```python
def encode_image(
    self,
    images: MultiImageBatch,
    condition_image_size: Union[int, Tuple[int, int]] = (1024, 1024),
    **kwargs,
) -> Dict[str, Union[List[Any], torch.Tensor]]:
    """
    Encode condition images into latent representations.
    
    Args:
        images: Multi-image batch — the canonical format is List[List[Image.Image]],
                where images[i] is a list of condition images for sample i.
    
    Returns:
        Dict with encoded representations. For models with variable-length
        condition sequences, return Lists instead of stacked Tensors:
        - 'condition_images': List[List[Tensor(3, H, W)]] — resized images
        - 'image_latents': List[Tensor(seq_len, C)] or Tensor(B, seq_len, C)
        - 'image_latent_ids': List[Tensor(seq_len, 4)] or Tensor(B, seq_len, 4)
    """
```

> **Important**: The `images` input follows the **multi-image batch** convention: `List[List[Image.Image]]`. Each sample can have zero, one, or multiple condition images. See [Data Format Conventions](#data-format-conventions) for details.

#### `encode_video`

```python
def encode_video(
    self,
    videos: MultiVideoBatch,
    **kwargs,
) -> Optional[Dict[str, Union[List[Any], torch.Tensor]]]:
    """
    Encode condition videos into latent representations.
    Return None if video encoding is not applicable.
    
    Args:
        videos: Multi-video batch — List[List[List[Image.Image]]] or similar.
    """
    return None
```


### Step 5: Implement `inference()`

This is the core generation method, analogous to `diffusers:Pipeline.__call__()`. It runs the full denoising loop and returns `List[BaseSample]`.

**The method must accept both raw inputs and pre-encoded inputs** — raw inputs are used when preprocessing is disabled; pre-encoded inputs come from the cached dataset during normal training.

```python
@torch.no_grad()
def inference(
    self,
    # Raw inputs (used when preprocessing is disabled)
    prompt: Optional[List[str]] = None,
    images: Optional[MultiImageBatch] = None,
    # Pre-encoded inputs (from preprocessing cache)
    prompt_ids: Optional[torch.Tensor] = None,
    prompt_embeds: Optional[torch.Tensor] = None,
    # Generation parameters
    height: int = 1024,
    width: int = 1024,
    num_inference_steps: int = 50,
    guidance_scale: float = 4.0,
    generator: Optional[torch.Generator] = None,
    # RL-specific parameters
    compute_log_prob: bool = True,
    trajectory_indices: TrajectoryIndicesType = 'all',
    extra_call_back_kwargs: List[str] = [],
) -> List[MyModelSample]:
    """
    Full denoising inference loop.
    
    Stages (mirroring Pipeline.__call__):
        1. Encode prompts (skip if pre-encoded)
        2. Encode condition images (skip if pre-encoded)
        3. Prepare initial noise latents
        4. Set up timestep schedule
        5. Denoising loop — call self.forward() at each step
        6. Decode final latents to pixel space
        7. Package results into Sample dataclasses
    """
    device = self.device
    
    # 1. Encode prompt (skip if already encoded)
    if prompt_embeds is None:
        encoded = self.encode_prompt(prompt=prompt, ...)
        prompt_embeds = encoded['prompt_embeds']
        prompt_ids = encoded['prompt_ids']
    
    batch_size = prompt_embeds.shape[0]
    
    # 2. Encode condition images (if applicable)
    # ...
    
    # 3. Prepare initial noise
    latents = randn_tensor(shape, generator=generator, device=device)
    
    # 4. Set timestep schedule
    timesteps, num_inference_steps = set_scheduler_timesteps(
        self.scheduler, num_inference_steps, device
    )
    
    # 5. Denoising loop with trajectory selective collection
    latent_collector = create_trajectory_collector(trajectory_indices, num_inference_steps)
    latent_collector.collect(latents, step_idx=0)
    
    if compute_log_prob:
        log_prob_collector = create_trajectory_collector(trajectory_indices, num_inference_steps)
    
    callback_collector = create_callback_collector(trajectory_indices, num_inference_steps)
    
    for i, t in enumerate(timesteps):
        t_next = timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor(0, device=device)
        noise_level = self.scheduler.get_noise_level_for_timestep(t)
        current_compute_log_prob = compute_log_prob and noise_level > 0
        
        # Single denoising step via forward()
        output = self.forward(
            t=t, t_next=t_next,
            latents=latents,
            prompt_embeds=prompt_embeds,
            compute_log_prob=current_compute_log_prob,
            noise_level=noise_level,
            return_kwargs=['next_latents', 'log_prob', 'noise_pred', ...],
            ...
        )
        
        latents = output.next_latents
        latent_collector.collect(latents, i + 1) # Call at every step. Selective mechanism is handled internally.
        if current_compute_log_prob:
            log_prob_collector.collect(output.log_prob, i)
        callback_collector.collect_step(
            i, output, extra_call_back_kwargs,
            capturable={'noise_level': noise_level}
        )
    
    # 6. Decode latents → images
    images = self.decode_latents(latents, output_type='pt')
    
    # 7. Package into samples (one per batch element, WITHOUT batch dimension)
    all_latents = latent_collector.get_result()
    latent_index_map = latent_collector.get_index_map()
    all_log_probs = log_prob_collector.get_result() if compute_log_prob else None
    log_prob_index_map = log_prob_collector.get_index_map() if compute_log_prob else None
    extra_call_back_res = callback_collector.get_result()
    callback_index_map = callback_collector.get_index_map()
    
    samples = [
        MyModelSample(
            # Denoising Trajectory
            timesteps=timesteps,
            all_latents=torch.stack([lat[b] for lat in all_latents], dim=0),
            log_probs=torch.stack([lp[b] for lp in all_log_probs], dim=0) if all_log_probs else None,
            latent_index_map=latent_index_map,
            log_prob_index_map=log_prob_index_map,
            # Generation
            image=images[b],
            # Generation Parameters
            height=height, width=width,
            # Prompt Info
            prompt=prompt[b],
            prompt_ids=prompt_ids[b],
            prompt_embeds=prompt_embeds[b],
            # Extra kwargs
            extra_kwargs={
                **{k: v[b] for k, v in extra_call_back_res.items()},
                'callback_index_map': callback_index_map,
            },
        )
        for b in range(batch_size)
    ]
    return samples
```

**Key utilities:**

| Utility | Purpose |
|---|---|
| `create_trajectory_collector(indices, T)` | Selectively stores latents/log-probs only at specified timesteps |
| `create_callback_collector(indices, T)` | Captures arbitrary per-step outputs (e.g., `noise_level`, `noise_pred`) |


### Step 6: Implement `forward()`

This method wraps a **single denoising step** — the body of the `for i, t in enumerate(timesteps)` loop from the *diffusers pipeline*. It calls the transformer and the scheduler.

```python
def forward(
    self,
    # Timestep info
    t: torch.Tensor,                # Current timestep (scalar tensor)
    t_next: Optional[torch.Tensor] = None,  # Next timestep
    # Latent state
    latents: torch.Tensor,          # (B, seq_len, C)
    next_latents: Optional[torch.Tensor] = None,  # Target for log-prob
    # Conditioning (all batched)
    prompt_embeds: torch.Tensor,    # (B, text_len, D)
    # ...model-specific condition inputs...
    # Control flags
    guidance_scale: float = 4.0,
    noise_level: Optional[float] = None,
    compute_log_prob: bool = True,
    return_kwargs: List[str] = ['noise_pred', 'next_latents', 'log_prob', ...],
) -> SDESchedulerOutput:
    """
    Single denoising step: transformer forward + scheduler step.
    
    This method corresponds to the body of the denoising loop in 
    Pipeline.__call__(). It is called by both inference() (full generation)
    and the trainer's optimization loop (per-timestep gradient computation).
    
    Returns:
        SDESchedulerOutput with fields gated by `return_kwargs`:
        - next_latents: Denoised latents for the next step
        - noise_pred: Model's velocity/noise prediction
        - log_prob: Log-probability under the SDE formulation
        - next_latents_mean: Deterministic mean (before noise injection)
        - std_dev_t, dt: SDE statistics
    """
    batch_size = latents.shape[0]
    
    # 1. Prepare model input
    #    (e.g., concatenate condition image latents, handle CFG doubling)
    
    # 2. Transformer forward pass
    noise_pred = self.transformer(
        hidden_states=latents,
        timestep=t.expand(batch_size) / 1000,
        encoder_hidden_states=prompt_embeds,
        ...,
        return_dict=False,
    )[0]
    
    # 3. Post-process (e.g., extract target portion, apply CFG)
    #    noise_pred = noise_pred[:, :latents.shape[1]]  # Remove condition tokens
    
    # 4. Scheduler step — this handles SDE dynamics and log-prob computation
    output = self.scheduler.step(
        noise_pred=noise_pred,
        timestep=t,
        latents=latents,
        timestep_next=t_next,
        next_latents=next_latents,
        compute_log_prob=compute_log_prob,
        return_dict=True,
        return_kwargs=return_kwargs,
        noise_level=noise_level,
    )
    return output
```

> **Note**: The `scheduler.step()` call is standardized across all models — it handles SDE noise injection, ODE stepping, and log-probability computation. You only need to implement the transformer-specific logic before it.

> For a detailed walkthrough of how `inference()` and `forward()` fit into the six-stage training pipeline, see the [Workflow Guidance — Stage 3: Trajectory Generation](workflow.md#stage-3-trajectory-generation) and [Stage 6: Policy Optimization](workflow.md#stage-6-policy-optimization).

### Step 7: Register the Adapter

Add your adapter to the registry in `src/flow_factory/models/registry.py`:

```python
_MODEL_ADAPTER_REGISTRY: Dict[str, str] = {
    # ... existing entries ...
    'my-model': 'flow_factory.models.my_model.my_model.MyModelAdapter',
}
```

Now it can be used via config:

```yaml
model:
  model_type: "my-model"
  model_name_or_path: "org/my-model-checkpoint"
```

## Advanced: Custom `preprocess_func`

The default `preprocess_func` calls `encode_prompt`, `encode_image`, and `encode_video` independently. Override it when your model requires **cross-modal preprocessing** — for example, FLUX.2 uses its text encoder to "upsample" (rewrite) prompts based on input images before encoding ([here](https://github.com/X-GenGroup/Flow-Factory/blob/main/src/flow_factory/models/flux/flux2.py#L371)):

```python
# src/flow_factory/models/flux/flux2.py — Flux2Adapter.preprocess_func()
def preprocess_func(
    self,
    prompt: List[str],
    images: Optional[MultiImageBatch] = None,
    caption_upsample_temperature: Optional[float] = None,
    **kwargs,
) -> Dict[str, Union[List[Any], torch.Tensor]]:
    # 1. Normalize images to List[List[Image | None]]
    # ...
    
    # 2. Cross-modal: rewrite prompts using text encoder + images
    if caption_upsample_temperature is not None:
        final_prompts = [
            self.pipeline.upsample_prompt(prompt=p, images=imgs, temperature=caption_upsample_temperature)
            for p, imgs in zip(prompt, images)
        ]
    else:
        final_prompts = prompt
    
    # 3. Encode prompts (with rewritten text)
    batch = self.encode_prompt(prompt=final_prompts, **kwargs)
    
    # 4. Encode images separately
    if has_images:
        batch.update(self.encode_image(images=images, **kwargs))
    
    return batch
```

**When to override `preprocess_func`:**

| Scenario | Override Needed? |
|---|---|
| Standard independent encoding (text + image + video) | No — default works |
| Prompt rewriting that depends on input images | Yes |
| Joint text-image encoding (e.g., interleaved tokens) | Yes |
| Custom normalization or augmentation during preprocessing | Yes |


## Advanced: Pseudo-Pipeline for Non-Diffusers Models

Not all models have a diffusers pipeline. For models like [Bagel](https://github.com/ByteDance-Seed/Bagel) — a unified multimodal foundation model that combines LLM, ViT, and VAE in a single architecture — you can create a **pseudo-pipeline** that mimics the diffusers `Pipeline` interface just enough for `BaseAdapter` to work.

> **Reference implementation**: See the [`bagel` branch](https://github.com/X-GenGroup/Flow-Factory/tree/bagel/src/flow_factory/models/bagel) for the complete working example.

### Why a Pseudo-Pipeline?

`BaseAdapter` accesses model components via `getattr(self.pipeline, name)`. It expects a pipeline object with:

1. **Named component attributes** — e.g., `.transformer`, `.vae`, `.scheduler`
2. **A `from_pretrained()` class method** — for weight loading

A pseudo-pipeline satisfies these requirements without inheriting from `DiffusionPipeline`. It serves as a **flat component container** with stateless utility methods.

### Design Pattern

The key insight from the Bagel implementation is the **separation of concerns**:

```
BagelPseudoPipeline (pipeline.py)         BagelAdapter (bagel.py)
┌────────────────────────────────┐         ┌──────────────────────────────┐
│ Component ownership (flat):    │         │ Training-aware methods:      │
│  .transformer  (Qwen2LLM)      │         │  .forward()                  │
│  .vit          (SiglipVision)  │         │  .inference()                │
│  .vae          (AutoEncoder)   │         │  ._forward_flow()            │
│  .vae2llm      (nn.Linear)     │         │  .forward_cache_update_*()   │
│  .llm2vae      (nn.Linear)     │         │                              │
│  .time_embedder                │         │ self.transformer resolves to │
│  .connector                    │         │ ACCELERATOR-WRAPPED module   │
│  .config                       │         │ for proper gradient flow     │
│                                │         │ in distributed training      │
│ Stateless utilities:           │         │                              │
│  .prepare_text_tokens()        │         │ Encoding methods:            │
│  .prepare_vit_tokens()         │         │  .encode_prompt()            │
│  .prepare_vae_latent()         │         │  .encode_image()             │
│  .from_pretrained()            │         │                              │
└────────────────────────────────┘         └──────────────────────────────┘
```

**Critical rule**: Any method that calls `self.transformer(...)` for a forward pass must live in the **Adapter**, not the pipeline. In the pipeline, `self.transformer` is the raw `nn.Module`; in the Adapter, `self.transformer` resolves to the accelerator-wrapped version via `get_component('transformer')`, which is essential for FSDP/DDP gradient correctness.

### Implementation

#### 1. Create the Pseudo-Pipeline

```python
# src/flow_factory/models/my_model/pipeline.py

class MyModelPseudoPipeline:
    """
    Flat component container — NO NEED to be a DiffusionPipeline subclass.
    Owns all nn.Modules as direct attributes so BaseAdapter can
    access them via getattr(self.pipeline, name).
    """
    
    def __init__(
        self,
        config: MyModelConfig,
        transformer: nn.Module,
        vae: nn.Module,
        # ... other components ...
        scheduler: Optional[Any] = None,
    ):
        # Flat component storage — BaseAdapter discovers these by name
        self.transformer = transformer
        self.vae = vae
        self.scheduler = scheduler
    
    @classmethod
    def from_pretrained(cls, model_path: str, low_cpu_mem_usage=False, **kwargs):
        """
        Load all components from a checkpoint directory.
        """
        # 1. Instantiate components
        config=MyModelConfig(...)
        transformer = MyTransformer(...)
        vae = MyVAE(...)
        
        return cls(config=config, transformer=transformer, vae=vae, ...)
```

**Weight remapping**: If the original model uses a nested structure (e.g., `model.language_model.layers.0.self_attn`), create a `_PREFIX_MAP` to flatten keys to the pipeline layout. For Bagel, it is like:

```python
# Bagel example: nested → flat key remapping
_PREFIX_MAP = {
    "language_model.": "transformer.",
    "vit_model.":      "vit.",
    "vae2llm.":        "vae2llm.",
    "llm2vae.":        "llm2vae.",
}
```

#### 2. Override Adapter Properties for Non-Standard Components

Bagel has no text encoder (the LLM handles text as part of its context). Override the discovery properties:

```python
class BagelAdapter(BaseAdapter):
    
    @property
    def text_encoder_names(self) -> List[str]:
        return []  # LLM handles text — no separate text encoder
    
    @property
    def text_encoders(self) -> List[nn.Module]:
        return []
    
    @property
    def preprocessing_modules(self) -> List[str]:
        # ViT and connector needed for encoding condition images into KV-cache
        return ["vae", "vit", "connector", "vit_pos_embed"]
    
    @property
    def inference_modules(self) -> List[str]:
        # Everything needed during training loop
        return [
            "transformer", "vit", "vae",
            "vae2llm", "llm2vae",
            "time_embedder", "latent_pos_embed",
            "connector", "vit_pos_embed",
        ]
```

#### 3. Implement `inference` and `forward` Functions

The `inference()` and `forward()` methods follow the same patterns described in [Step 5](#step-5-implement-inference) and [Step 6](#step-6-implement-forward).

For a detailed walkthrough of how `inference()` and `forward()` fit into the six-stage training pipeline, see the [Workflow Guidance — Stage 3: Trajectory Generation](workflow.md#stage-3-trajectory-generation) and [Stage 6: Policy Optimization](workflow.md#stage-6-policy-optimization).

### When to Use a Pseudo-Pipeline

| Scenario | Approach |
|---|---|
| Model has a `diffusers` pipeline | Use the `diffusers` pipeline directly (standard path) |
| Model has no `diffusers` pipeline but follows a standard text-encoder + denoiser + VAE architecture | Create a pseudo-pipeline with flat component layout |
| Model has a fundamentally different architecture (e.g., unified MLLM) | Create a pseudo-pipeline and handle custom `inference/forward` calls in the Adapter |


## Data Format Conventions

All encoding methods and `inference()`/`forward()` receive **batched** inputs. Here are the canonical formats:

### Text

| Parameter | Format | Example Shape |
|---|---|---|
| `prompt` | `List[str]` | Length `B` |
| `prompt_ids` | `torch.Tensor` | `(B, seq_len)` |
| `prompt_embeds` | `torch.Tensor` | `(B, seq_len, D)` |

### Images

| Parameter | Format | Description |
|---|---|---|
| `images` | `List[List[Image.Image]]` | **Multi-image batch**: `images[i]` is a list of condition images for sample `i`. Each inner list can have 0, 1, or N images. |
| `condition_images` | `List[List[Tensor(C,H,W)]]` | Resized/preprocessed version of above |
| `image_latents` | `List[Tensor(seq,C)]` or `Tensor(B,seq,C)` | VAE-encoded latents. Use `List` for variable-length sequences, `Tensor` when all samples share the same sequence length. |

> The multi-image batch convention (`List[List[...]]`) is critical for models that support varying numbers of condition images per sample. Always normalize your input to this format in `encode_image()`.

### Videos

| Parameter | Format | Description |
|---|---|---|
| `videos` | `List[List[List[Image.Image]]]` | **Multi-video batch**: `videos[i]` is a list of condition videos, each video is a list of frames. |
| `condition_videos` | `List[List[Tensor(T,C,H,W)]]` | Preprocessed version |

### Sample Fields (no batch dimension)

Fields stored in `BaseSample` are per-sample (no batch dimension):

| Field | Shape | Description |
|---|---|---|
| `all_latents` | `(num_stored, seq_len, C)` | Trajectory latents at selected timesteps |
| `log_probs` | `(num_stored,)` | Per-step log-probabilities |
| `image` | `(C, H, W)` | Generated image tensor |
| `video` | `(T, C, H, W)` | Generated video tensor |


## Checklist

Before submitting a new model adapter, verify:

- [ ] **`load_pipeline()`** — Returns the correct diffusers pipeline with `low_cpu_mem_usage=False`
- [ ] **`default_target_modules`** — Lists attention and FFN layer names matching your transformer architecture
- [ ] **`preprocessing_modules`** — Includes all components needed for encoding (text encoders, VAE, image encoders)
- [ ] **`inference_modules`** — Includes all components needed during the training loop
- [ ] **`encode_prompt()`** — Returns dict with at least `prompt_ids` and `prompt_embeds`
- [ ] **`encode_image()`** — Handles `MultiImageBatch` input format; returns `None` or empty for text-only models
- [ ] **`inference()`** — Accepts both raw and pre-encoded inputs; returns `List[Sample]`
- [ ] **`forward()`** — Single denoising step; ends with `self.scheduler.step()`; returns `SDESchedulerOutput`
- [ ] **Sample dataclass** — All fields without batch dimension; `_shared_fields` correctly set
- [ ] **Registry entry** — Added to `_MODEL_ADAPTER_REGISTRY`
- [ ] **Tested** — Runs at least one epoch of GRPO training without errors