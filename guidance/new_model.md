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
- [Advanced: Offline Output-State Encoding](#advanced-offline-output-state-encoding)
- [Advanced: Custom `preprocess_func`](#advanced-custom-preprocess_func)
- [Advanced: Pseudo-Pipeline for Non-Diffusers Models](#advanced-pseudo-pipeline-for-non-diffusers-models)
- [Data Format Conventions](#data-format-conventions)
- [Checklist](#checklist)

## Overview

Flow-Factory uses a **model adapter** pattern that wraps [diffusers](https://github.com/huggingface/diffusers) pipelines into a unified interface for online and offline fine-tuning. Each adapter maps a diffusers pipeline to a consistent API that the training loop can call without knowing model-specific details.

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

> **Examples**: There are several PRs that adapte new models in this framework: [FLUX2-Klein](https://github.com/X-GenGroup/Flow-Factory/pull/9), [Z-Image-Omni](https://github.com/X-GenGroup/Flow-Factory/pull/22).


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

### Gradient checkpointing contract

`train.enable_gradient_checkpointing` accepts the existing boolean or a selective
policy such as `{mode: fraction, fraction: 0.25}`, `{every_n: 4}`, or
`{layers: [0, 4, 8]}`. Selective policies discover ordered blocks from a
Diffusers model's `_repeated_blocks` declaration. Adapters with multiple forward
stacks should override `_gradient_checkpointing_units()` and return their blocks
in execution order.

Checkpointing has one owner. Before model loading, an FSDP2 full model policy is
normalized to backend activation checkpointing, even when backend checkpointing
was not explicitly enabled, so recomputation stays inside the sharded
mixed-precision boundary. FSDP1 keeps model-level ownership. FSDP2 rejects every
selective train-level policy because model-level `fraction`, `every_n`, or
`layers` boundaries sit outside the FSDP2 input-cast boundary and cannot replay
it safely. Transformers-style components support full checkpointing through
`gradient_checkpointing_enable()`, but must expose the Diffusers callback API to
support selective modes on compatible backends.

Pairwise objectives retain two trainable policy graphs until their joint
backward. An adapter that cannot fit both graphs on replicated-parameter
backends may set `requires_pairwise_policy_activation_offload = True` after the
need is demonstrated with production geometry. Shared pairwise trainers then
store each arm's saved autograd tensors in pinned CPU memory and restore them
for backward. FSDP2 keeps its backend-owned parameter-sharding/checkpointing
path and ignores this opt-in. This policy controls cross-arm activation
storage; it does not replace block-level gradient checkpointing or change the
pairwise objective.

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
| `T2AVSample` | Text-to-audio-video | Alias of `BaseSample` |
| `ImageConditionSample` | Image-conditioned generation | `condition_images`: per-sample `List[Tensor(C,H,W)]` (or `List[PIL.Image]` when the subclass sets `condition_images_as_pil=True`); always `List`, never batched tensor |
| `VideoConditionSample` | Video-conditioned generation | `condition_videos`: per-sample `List[Tensor(T,C,H,W)]` (or `List[List[PIL.Image]]` when `condition_videos_as_pil=True`); always `List`, never batched tensor |
| `I2ISample` | Image-to-image | `ImageConditionSample` subclass |
| `I2VSample` | Image-to-video | `ImageConditionSample` subclass |
| `I2AVSample` | Image-to-audio-video | `ImageConditionSample` subclass |
| `V2VSample` | Video-to-video | `VideoConditionSample` subclass |

> See [`src/flow_factory/samples/samples.py`](../src/flow_factory/samples/samples.py) for all available classes.

> **Key**: The `_shared_fields` class variable declares fields that are identical across a batch (e.g., `height`, `width`, `latent_index_map`). During `BaseSample.stack()`, shared fields take the first element instead of stacking.

> **Type determinism for `gather_samples`**: `ImageConditionSample.__post_init__` and `VideoConditionSample.__post_init__` canonicalize to a deterministic per-sample type across all samples and ranks — `List[Tensor]` by default, or `List[PIL.Image]` / `List[List[PIL.Image]]` when the subclass sets `condition_images_as_pil` / `condition_videos_as_pil` (adapters that persist condition media as PIL via `python_format_columns`, e.g. Bagel and SenseNova). When defining custom sample fields that will be gathered across ranks (via `gather_samples`), ensure each field has a **consistent type** on every sample — mixing `Tensor` on some samples and `List[Tensor]` on others will cause `gather_samples` to fall through to slow pickle-based `gather_object`. Prefer `List[Tensor]` for variable-length sequences.


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

Override only when your model deviates — for example, [WAN-T2V](../src/flow_factory/models/wan/wan2_t2v.py) models need `['text_encoders', 'vae', 'image_encoder']` for preprocessing and conditionally include `transformer_2` for inference.

> **Tip**: Use `print(dict(self.pipeline.named_children()))` to discover available component names.


### Step 4: Implement Encoding Methods

Override the encoders your model consumes. The default `BaseAdapter` implementation of every per-modality encoder is a no-op `pass` that returns `None`; the default [`preprocess_func` in `BaseAdapter`](https://github.com/X-GenGroup/Flow-Factory/blob/main/src/flow_factory/models/abc.py) dispatches to all four encoders and skips any that return `None`:

```python
preprocess_func(prompt, images, videos, audios, **kwargs):
    results = {}
    for inputs, encoder in [
        (prompt, self.encode_prompt),
        (images, self.encode_image),
        (videos, self.encode_video),
        (audios, self.encode_audio),
    ]:
        if inputs is not None:
            encoded = encoder(inputs, **kwargs)
            if encoded is not None:  # skip no-op default
                results.update(encoded)
    return results
```

Text-to-image, text-to-video, and text-to-audio-video adapters usually override only
`encode_prompt` because they have no condition media. Image-conditioned tasks add
`encode_image`; video-conditioned tasks add `encode_video`; audio-conditioned tasks add
`encode_audio`. These functions encode inputs, not supervised outputs—offline targets belong to
the output-state codec. There is no need to add stub `pass` overrides for unused modalities;
`BaseAdapter` already provides them.

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

> **Important**: The `images` input follows the **multi-image batch** convention: `List[List[Image.Image]]`. Each sample can have zero, one, or multiple condition images. See [Data Format Conventions](#data-format-conventions) for details. Adapters that persist a returned image column as PIL (declare it in `python_format_columns`, e.g. Bagel and SenseNova `condition_images`) may keep it as PIL; the dataset stores those columns via the HF Image feature and reads them back as PIL.

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

#### `encode_audio`

```python
def encode_audio(
    self,
    audios: MultiAudioBatch,
    **kwargs,
) -> Optional[Dict[str, Union[List[Any], torch.Tensor]]]:
    """
    Encode condition audio inputs into latent / feature representations.
    Override this when the model consumes audio; otherwise the BaseAdapter
    no-op default returns ``None`` and ``preprocess_func`` skips integration.

    Args:
        audios: Multi-audio batch — ``List[List[Tensor]]`` where ``audios[i]``
                is a list of audio tensors for sample ``i``. Each Tensor is
                loaded by ``flow_factory.utils.audio.load_audio`` (mono shape
                ``(samples,)`` or stereo ``(channels, samples)``, time-domain).
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
    audios: Optional[MultiAudioBatch] = None,  # only declare if the model consumes audio
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
            return_kwargs=['next_latents', 'log_prob', 'velocity', ...],
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
| `create_callback_collector(indices, T)` | Captures arbitrary per-step outputs (e.g., `noise_level`, `velocity`) |


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
    return_kwargs: List[str] = ['velocity', 'next_latents', 'log_prob', ...],
) -> SDESchedulerOutput:
    """
    Single denoising step: transformer forward + scheduler step.
    
    This method corresponds to the body of the denoising loop in 
    Pipeline.__call__(). It is called by both inference() (full generation)
    and the trainer's optimization loop (per-timestep gradient computation).
    
    Returns:
        SDESchedulerOutput with fields gated by `return_kwargs`:
        - next_latents: Denoised latents for the next step
        - velocity: Model's velocity prediction
        - log_prob: Log-probability under the SDE formulation
        - next_latents_mean: Deterministic mean (before noise injection)
        - std_dev_t, dt: SDE statistics
    """
    batch_size = latents.shape[0]
    
    # 1. Prepare model input
    #    (e.g., concatenate condition image latents, handle CFG doubling)
    
    # 2. Transformer forward pass
    velocity = self.transformer(
        hidden_states=latents,
        timestep=t.expand(batch_size) / 1000,
        encoder_hidden_states=prompt_embeds,
        ...,
        return_dict=False,
    )[0]
    
    # 3. Post-process (e.g., extract target portion, apply CFG)
    #    velocity = velocity[:, :latents.shape[1]]  # Remove condition tokens
    
    # 4. Scheduler step — this handles SDE dynamics and log-prob computation
    output = self.scheduler.step(
        velocity=velocity,
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

## Advanced: Offline Output-State Encoding

Online-only adapters can keep the default `build_output_state_codec() -> None`. To support SFT or
offline DPO, an adapter must additionally declare both sides of its pipeline and provide an
on-the-fly output codec:

1. Set a class-level `pipeline_io_contract`. It owns per-type and aggregate input counts,
   order/binding, optional semantic input slots, negative prompt policy, the exact ordered output
   media sequence, rate requirements, geometry source, and batch capability. Explicit V2 slots
   reserve declared arguments; unslotted inputs fill the remaining slots in declaration order, and
   output media must never carry slots. If checkpoints behind one adapter expose narrower behavior, override
   `_resolve_pipeline_io_contract()` and return an immutable instance-specific specialization;
   offline data validation consumes `effective_pipeline_io_contract`.
2. If cached input fields are not already the exact forward condition, override
   `build_condition_state_preparer()` with a declaration-only preparer. Its
   `required_components` lists runtime encoders and `prepare_condition_state()` returns one
   `PreparedConditionState` per batch. Put input-owned model fields in `forward_context` and
   input-owned target-binding fields in `output_context`. The two runtime consumer views may
   intentionally share an input-owned tensor (for example a mask or layout), but each merged
   consumer view must remain collision-free with cached fields and later candidate-owned output
   fields.
3. Override `build_output_state_codec()` with a declaration-only codec. Its
   `required_components` names logical runtime components such as `("vae",)`; construction must
   not load, materialize, move, replace, or cast them.
4. Return an `EncodedOutputState` containing a detached `LatentState`, output-derived forward and
   decode contexts, and one exact geometry signature per sample.
5. Override `_validate_encoded_output_geometry()` so configured, condition-derived, and
   output-derived dimensions cannot drift silently.
6. Declare a complete immutable `offline_training_forward_overrides` mapping whenever the base
   `{"guidance_scale": 1.0}` contract does not describe the adapter. Offline trainers apply this
   mapping after sampling configuration and cached batch conditions, so it owns loss-time model
   conditioning. Conventional CFG branches must all be set to their neutral point (for example,
   both Wan transformer scales or Bagel text/image CFG scales); guidance-distilled models instead
   declare the explicit guidance-embedding value used by their official training recipe. Replace
   the complete mapping so permissive `**kwargs` forwards do not receive unrelated base keys.

The dataset remains responsible only for strict V2 parsing and CPU media decoding. The adapter
owns numerical condition/output semantics. The SFT/offline-DPO trainer first calls
`prepare_condition_state()` once, then calls `encode_output_state()` under `torch.no_grad`; offline
DPO passes the same prepared object to both preference candidates. Target, chosen, and rejected
latents are not preprocessing-cache columns. Declared condition and output components are loaded
through `ModelLoadCoordinator`, never from inside a preparer or codec.

Condition encoding and target encoding should share role-neutral numerical transforms instead of
duplicating VAE math. Extract helpers for pixel preprocessing, posterior extraction, latent
normalization, patchification, IDs, and packing, then make the posterior policy an explicit
argument:

```python
def encode_vae_image(adapter, pixels, *, sample_mode, generator=None):
    posterior = adapter.vae.encode(pixels).latent_dist
    latent = (
        posterior.sample(generator=generator)
        if sample_mode == "sample"
        else posterior.mode()
    )
    return normalize_and_pack(adapter, latent)
```

The helper is role-neutral; the caller is not. Follow the official Diffusers pipeline for each
role. Condition paths commonly use posterior `argmax`/`mode` for stable conditioning, while
training targets use posterior `sample` and forward the caller's generator. Never merge the two
entry points in a way that silently changes this policy. Tests should compare the shared transform
against the pinned Diffusers helper and assert both sample/argmax behavior and generator routing.

An output codec is not merely an `encode_image()` alias. It may need output-specific geometry,
multi-component state order, active masks, rate alignment, or forward context that condition
encoding does not own. If those semantics are not lossless, set a concrete
`output_state_codec_unavailable_reason` so offline selection fails before downloading weights.

Multi-modal objectives may need a reduction different from trajectory likelihoods. Override the
protected `_reduce_flow_matching_objective_values()` hook only for that objective. Do not change
`reduce_latent_values()` merely to implement SFT: online policy gradients, replay log-probability,
and distillation continue to rely on their established trajectory-wide reduction.

SenseNova is an example of an important boundary: its existing condition schema uses grouped
`images` with within-type order. Do not advertise heterogeneous ordered references merely because
several images are accepted. Dataset media and ordered-reference entries use `type` as their sole
discriminator, including at the adapter preprocessing boundary.

## Advanced: Custom `preprocess_func`

The default `preprocess_func` calls `encode_prompt`, `encode_image`, `encode_video` and `encode_audio` independently. Override it when your model requires **cross-modal preprocessing** — for example, FLUX.2 uses its text encoder to "upsample" (rewrite) prompts based on input images before encoding ([here](https://github.com/X-GenGroup/Flow-Factory/blob/main/src/flow_factory/models/flux/flux2.py#L371)):

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

If the override accepts semantic inputs only through `**kwargs`, declare every
output-affecting key in `preprocess_cache_fields`. Set `preprocess_cache_version`
and bump it whenever helper behavior changes without changing the wrapper source.
Adapters that consume the ordered Ref2VA-style `references` column must also set
`supports_ordered_references = True`.

For flow models whose predicted velocity points from noise toward clean data
(`clean - noise`) instead of the default noise-ward direction, set
`flow_velocity_direction = "data"`. Trainers use this declaration when projecting
velocity predictions to `x0`; do not duplicate the sign convention inside a trainer.


## Advanced: Pseudo-Pipeline for Non-Diffusers Models

Not all models have a diffusers pipeline. Unified Transformers models such as
[Bagel](https://github.com/ByteDance-Seed/Bagel) and
[SenseNova-U1](https://github.com/OpenSenseNova/SenseNova-U1) can use a
**pseudo-pipeline** as an explicit component container.

> **Reference implementations**:
> - [`src/flow_factory/models/bagel/`](../src/flow_factory/models/bagel) (`bagel`) exposes the unified Bagel model plus its VAE and tokenizer.
> - [`src/flow_factory/models/sensenova/`](../src/flow_factory/models/sensenova) (`sensenova`) exposes one `SenseNovaDenoiser` component while NEO-Unify owns tokenization, vision encoding, and pixel-space flow matching.

### Why a Pseudo-Pipeline?

`BaseAdapter` resolves model components through `ComponentRuntime`, not through
Python attribute probing. A pseudo-pipeline supplies:

1. **Explicit canonical components** — declared by `PseudoPipelineRuntime`; only components that actually exist are listed
2. **A `from_pretrained()` class method** — for weight loading

A pseudo-pipeline satisfies these requirements without inheriting from
`DiffusionPipeline`. Do not declare absent components as placeholders: SenseNova,
for example, has no standalone Flow-Factory VAE or text encoder.

### Design Pattern

Many non-diffusers models (e.g., Bagel and SenseNova) are a **single composite `nn.Module`** that internally contains sub-modules (LLM, ViT, projectors, etc.). Unlike diffusers pipelines where components are independent top-level objects, these models have a deeply nested structure.

The key design pattern is to store the **full composite model** on the pipeline while creating **aliases** to its key sub-modules that `BaseAdapter` needs to manage (freeze, LoRA, prepare with accelerator):

```
BagelPseudoPipeline (pipeline.py)         BagelAdapter (bagel.py)
┌────────────────────────────────┐         ┌──────────────────────────────┐
│ Component ownership:           │         │ Training-aware methods:      │
│  .bagel       (full Bagel model│         │  .forward()                  │
│                wraps LLM+ViT+  │         │  .inference()                │
│                projectors)     │         │  ._forward_flow()            │
│  .transformer (alias →         │         │  ._build_gen_context()       │
│                bagel.language_ │         │                              │
│                model)          │         │ In the Adapter:              │
│  .vae         (AutoEncoder,    │         │  self.transformer resolves   │
│                separate model) │         │  to ACCELERATOR-WRAPPED LLM  │
│  .scheduler   (None initially) │         │  via get_component()         │
│  ._bagel_config                │         │                              │
│                                │         │ Sub-modules accessed via:    │
│ Loading:                       │         │  self.pipeline.bagel.vae2llm │
│  .from_pretrained()            │         │  self.pipeline.bagel.llm2vae │
│                                │         │  self.pipeline.bagel.*       │
└────────────────────────────────┘         └──────────────────────────────┘
```

**Critical rule**: Any code that calls `self.transformer(...)` for a **gradient-bearing forward pass** must live in the **Adapter**, not the pipeline. In the Adapter, `self.transformer` resolves to the accelerator-wrapped version via `get_component('transformer')`, which is essential for FSDP/DDP gradient correctness. Non-gradient utility calls (e.g., preparing KV caches, encoding condition images) can use `self.pipeline.bagel.*` directly since those run under `@torch.no_grad`.

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

> **Why list both `"bagel"` and `"transformer"`?** The `"transformer"` is an alias pointing into `"bagel"` (they share parameters). `"transformer"` is listed so that `on_load_components` / `off_load_components` can skip it when it's accelerator-managed (prepared components are not manually moved). `"bagel"` is listed to ensure the full model — including sub-modules like ViT, projectors, and embedders that are NOT separate pipeline attributes — is moved to the correct device.

#### 3. Implement `inference` and `forward` Functions

The `inference()` and `forward()` methods follow the same patterns described in [Step 5](#step-5-implement-inference) and [Step 6](#step-6-implement-forward).

For non-diffusers models, the adapter typically accesses sub-modules via `self.pipeline.model.sub_module` for utility operations (e.g., `self.pipeline.bagel.vae2llm`, `self.pipeline.bagel.time_embedder`) while routing the main denoising forward pass through `self.transformer` (the accelerator-wrapped alias).

For a detailed walkthrough of how `inference()` and `forward()` fit into the six-stage training pipeline, see the [Workflow Guidance — Stage 3: Trajectory Generation](workflow.md#stage-3-trajectory-generation) and [Stage 6: Policy Optimization](workflow.md#stage-6-policy-optimization).

### When to Use a Pseudo-Pipeline

| Scenario | Approach |
|---|---|
| Model has a `diffusers` pipeline | Use the `diffusers` pipeline directly (standard path) |
| Model is a single composite `nn.Module` (e.g., unified MLLM with LLM + ViT + VAE) | Create a pseudo-pipeline storing the full model + aliasing the trainable sub-module |
| Model has separate independent components but no diffusers pipeline | Create a pseudo-pipeline with direct component attributes |


## Data Format Conventions

**Critical convention — batch boundary:**

> All inputs to `preprocess_func()`, `encode_image()`, `encode_video()`, `encode_audio()`, `inference()`, and `forward()` carry a **batch dimension**. Tensors have shape `(B, ...)` and condition collections use `List[...]` with length `B`.
>
> `condition_images` at the method level is **model-dependent** — there is no single canonical batch type:
> - Single condition image per sample with uniform shape (e.g. Flux1-Kontext): batched `Tensor(B, C, H, W)`. `condition_images[b]` yields `Tensor(C,H,W)`, which `ImageConditionSample.__post_init__` unbinds to `[Tensor(C,H,W)]`.
> - Multiple condition images per sample, or variable shapes (e.g. Flux2, Qwen-Image-Edit, Bagel, SenseNova): `List[List[Tensor(C,H,W)]]` of length `B`. `condition_images[b]` yields `List[Tensor(C,H,W)]` directly.
>
> The value stored on `sample.condition_images` after `inference()` is per-sample (no batch dimension); its element type is set by `ImageConditionSample.condition_images_as_pil` — `List[Tensor(C,H,W)]` in `[0,1]` by default, or `List[PIL.Image]` when the adapter persists condition_images via the HF Image feature (declares them in `python_format_columns` and sets `condition_images_as_pil=True` on its sample, e.g. Bagel and SenseNova). `condition_videos` follows the same model-dependent pattern.
>
> Fields stored on `BaseSample` (and subclass) instances are **per-sample** — the batch dimension is stripped. `sample.condition_images` is one sample's images (`List[Tensor(C,H,W)]`, or `List[PIL.Image]` when `condition_images_as_pil=True`), not the full batch. This is enforced at construction time when `inference()` slices `condition_images[b]` for each `b` in `range(batch_size)`.

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
| `condition_images` | `List[List[Tensor(C,H,W)]]` in `[0,1]` (or `List[List[PIL.Image]]` for `python_format_columns` adapters, e.g. Bagel and SenseNova) | Resized/preprocessed version of above |
| `image_latents` | `List[Tensor(seq,C)]` or `Tensor(B,seq,C)` | VAE-encoded latents. Use `List` for variable-length sequences, `Tensor` when all samples share the same sequence length. |

> The multi-image batch convention (`List[List[...]]`) is critical for models that support varying numbers of condition images per sample. Always normalize your input to this format in `encode_image()`.

### Videos

| Parameter | Format | Description |
|---|---|---|
| `videos` | `List[List[List[Image.Image]]]` | **Multi-video batch**: `videos[i]` is a list of condition videos, each video is a list of frames. |
| `condition_videos` | `List[List[Tensor(T,C,H,W)]]` (or `List[List[List[PIL.Image]]]` frame-lists when the sample sets `condition_videos_as_pil`) | Preprocessed version |

### Audio

| Parameter | Format | Description |
|---|---|---|
| `audios` | `MultiAudioBatch` (= `List[List[Tensor(samples,)]]` mono or `List[List[Tensor(channels, samples)]]` stereo) | **Multi-audio batch**: `audios[i]` is a list of audio tensors for sample `i`. Tensors are loaded by `flow_factory.utils.audio.load_audio`. Empty samples contribute `[]`. |
| `condition_audios` | `List[List[Tensor]]` | Preprocessed/resampled version stored on `BaseSample` subclasses. |
| `audio_features` | `List[Tensor(seq, D)]` or `Tensor(B, seq, D)` | Encoder output. Use `List` for variable-length sequences, `Tensor` when all samples share the same sequence length. |

> Type aliases live in `flow_factory/utils/audio.py`. `MultiAudioBatch` mirrors `MultiImageBatch` / `MultiVideoBatch`: nested per-sample list with one Tensor per condition audio. Override `encode_audio()` only if your model consumes audio — text/image/video-only adapters inherit `BaseAdapter`'s no-op default.

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
- [ ] **Preprocessing cache contract** — `**kwargs`-hidden semantic fields are listed in `preprocess_cache_fields`; helper-only behavior changes bump `preprocess_cache_version`
- [ ] **Ordered references** — Set `supports_ordered_references = True` only when the adapter consumes the validated heterogeneous `references` array
- [ ] **Velocity direction** — Set `flow_velocity_direction = "data"` when the model predicts `clean - noise`; otherwise keep the default `"noise"`
- [ ] **`encode_prompt()`** — Override only if your model needs text conditioning; returns dict with at least `prompt_ids` and `prompt_embeds` (text/image/video/audio-only models inherit the no-op default)
- [ ] **`encode_image()`** — Override only if your model consumes images; handles `MultiImageBatch` input format (text-only models inherit the no-op default)
- [ ] **`encode_video()`** — Override only if your model consumes videos; handles `MultiVideoBatch` input format
- [ ] **`encode_audio()`** — Override only if your model consumes audio; handles `MultiAudioBatch` input format (text/image/video-only models inherit the no-op default)
- [ ] **`inference()`** — Accepts both raw and pre-encoded inputs; returns `List[Sample]`
- [ ] **`forward()`** — Single denoising step; ends with `self.scheduler.step()`; returns `SDESchedulerOutput`
- [ ] **Pipeline I/O contract** — Declares exact input/output media, rate, geometry, and batch semantics before enabling offline training
- [ ] **Effective checkpoint contract (when needed)** — `_resolve_pipeline_io_contract()` narrows a class-level superset without changing public dataset or algorithm code
- [ ] **Condition-state preparer (when needed)** — Declaration-only logical component requirements; one input realization reused by every candidate/forward in the batch
- [ ] **Output-state codec (when supported)** — Declaration-only logical component requirements; on-the-fly detached target encoding; exact geometry validation
- [ ] **Objective reduction (when specialized)** — Override only the offline flow-matching hook; online trajectory reduction remains unchanged
- [ ] **Offline forward overrides (when supported)** — Complete immutable adapter mapping; sampling controls never define offline loss semantics; every CFG branch is neutralized or every distilled guidance condition is explicitly pinned
- [ ] **Role-neutral encoder math** — Condition/output paths reuse transforms but explicitly preserve official posterior `sample` versus `argmax` policy and generator routing
- [ ] **Explicit offline blocker (when unsupported)** — `output_state_codec_unavailable_reason` names the missing lossless semantic boundary
- [ ] **Sample dataclass** — All fields without batch dimension; `_shared_fields` correctly set; custom field types are consistent (no `Tensor` vs `List[Tensor]` mixing across samples)
- [ ] **Registry entry** — Added to `_MODEL_ADAPTER_REGISTRY`
- [ ] **Tested** — Runs at least one rollout cycle for online support and one complete dataloader epoch for any declared offline support

## Component Runtime and Structured Replay

Choose the runtime explicitly in `build_component_runtime()`:

- `ClassicPipelineRuntime` for eagerly materialized diffusers pipelines.
- `ModularPipelineRuntime` for declared specs whose materialized modules depend on a
  workflow.
- `PseudoPipelineRuntime` for adapters assembling components without a diffusers
  pipeline.

Component membership uses canonical lookup through declared specs, not `hasattr`.
When a logical target is a submodule alias of a larger physical root, declare
`alias_routes` explicitly, for example
`{"transformer": ("bagel", ("language_model",))}`. The load planner then marks
the physical root as target-owned; auxiliary lifecycle code can move frozen
siblings but must leave the prepared target route untouched. `BaseAdapter`
freezes materialized roots before reopening the logical target.

Adapters should leave `supports_fsdp2_cpu_efficient_loading = False` unless their
component source can selectively materialize TARGET state without applying the
rank-zero/meta policy to text encoders, VAEs, or reward models. Modular adapters
that satisfy this contract may opt in.
Keep declared specs distinct from materialized modules; use
`materialize_components(None)` only when the workflow genuinely needs every
declaration. A prepared/replacement override must be installed through the runtime so
`adapter.pipeline`, `adapter.scheduler`, and optimizer identities remain coherent.

Multimodal adapters declare `trajectory_component_order` and one scheduler per
component in `scheduler_group`. Distributed preparation wraps the resulting
`ModelBundle`; adapter access after prepare routes through `RoutedComponentProxy`
rather than replacing registered modules.

MiniMax H3 is the reference for workflow-pruned modular components and separate
video/audio trajectories. See `src/flow_factory/models/minimax_h3/` and the
[MiniMax H3 dataset contracts](datasets.md#minimax-h3-datasets).
