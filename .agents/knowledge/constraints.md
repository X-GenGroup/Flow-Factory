# Hard Constraints

These constraints MUST NOT be violated. Consult this file before making any code changes.

---

## Registry & Loading (1–5)

### 1. Registry Path Accuracy
The three registries (`_TRAINER_REGISTRY`, `_MODEL_ADAPTER_REGISTRY`, `_REWARD_MODEL_REGISTRY`) map string identifiers to **fully qualified Python class paths** for lazy import. If you move, rename, or restructure a class, the corresponding registry entry MUST be updated, or `ImportError` will occur at runtime.

### 2. Registry Identifier Convention
Registry keys are **case-insensitive** (lowered at lookup). Model adapter keys use lowercase with hyphens (e.g., `flux1-kontext`). Trainer keys use lowercase (e.g., `grpo-guard`). Reward keys use lowercase (e.g., `pickscore`). New entries must follow the same convention.

### 3. Dynamic Import Fallback
All three registries support a **direct Python path** fallback (e.g., `my_package.models.CustomAdapter`). If an identifier is not found in the registry, it is treated as a fully qualified import path. Do not break this two-mode resolution logic.

### 4. Decorator Registration
`@register_trainer` and `@register_reward_model` decorators exist for convenience but the canonical entries are the static dicts. If you use the decorator, ensure the static dict is also updated if the class should be discoverable by default.

### 5. Adapter `load_pipeline()` Must Return a DiffusionPipeline
Every `BaseAdapter` subclass's `load_pipeline()` must return a `diffusers.DiffusionPipeline` (or compatible object). The base class's `__init__` immediately accesses `.scheduler` on the returned object.

---

## Training Pipeline (6–10)

### 6. Six-Stage Pipeline Order
The training loop executes: Data Preprocessing → K-Repeat Sampling → Trajectory Generation → Reward Computation → Advantage Computation → Policy Optimization. This order is invariant. Do not reorder or skip stages.

### 7. Coupled vs Decoupled Paradigm
- **Coupled** (GRPO, GRPO-Guard): Training timesteps are coupled with SDE-based sampling. Requires log-probability computation. Must use SDE dynamics (`Flow-SDE`, `Dance-SDE`, `CPS`).
- **Decoupled** (NFT, AWM): Training timesteps are decoupled from sampling. Can use any dynamics including `ODE`.

Mixing paradigms (e.g., using `ODE` dynamics with `GRPO`) will produce incorrect gradients silently.

### 8. Component Offloading Lifecycle
Text encoders and VAEs are loaded for Stage 1 (preprocessing), then offloaded to free VRAM before the training loop. They are reloaded for inference during sampling. Do not assume these components are always on-device.

### 9. Accelerator `prepare()` Scope
Only **trainable modules** and the **optimizer** go through `accelerator.prepare()`. The dataloader uses a custom distributed sampler (`DistributedKRepeatSampler` or `GroupContiguousSampler`) and is NOT prepared via accelerator. Breaking this causes duplicate data or incorrect gradient accumulation.

### 9a. Sampler Geometric Constraints
Both samplers require `M * K ≡ 0 (mod W * B * G)` where M=unique_sample_num, K=group_size, W=world_size, B=per_device_batch_size, G=gradient_step_per_epoch. **GroupContiguousSampler** adds a stricter constraint: `M ≡ 0 (mod W)`. Auto-adjustment uses GCD-based rounding (DistributedKRepeatSampler) or LCM-based rounding (GroupContiguousSampler). See `.agents/knowledge/samplers.md` for full details.

### 10. DeepSpeed ZeRO-3 Is Unsupported
Reward model sharding under ZeRO-3 is broken even with `GatherParameter` context manager (see `trainers/abc.py` line 119–123). Only ZeRO-1 and ZeRO-2 are safe. Document this if users ask.

---

## Base Class Interfaces (11–14)

### 11. BaseTrainer Abstract Contract
`BaseTrainer.__init__` expects `(accelerator, config, adapter)`. Subclasses must implement the `start()`, `optimize()`, and `evaluate()` methods. The `_initialization()` method is called in `__init__` and handles dataloader, optimizer, accelerator preparation, reward model loading, and `AdvantageProcessor` instantiation — do not duplicate this logic.

**Trainer hierarchy**: `GRPOTrainer`, `DiffusionNFTTrainer`, and `AWMTrainer` all extend `BaseTrainer` directly. Only `GRPOGuardTrainer` extends `GRPOTrainer`. All trainers delegate advantage computation to `self.advantage_processor.compute_advantages()`.

### 12. BaseAdapter Abstract Methods
Subclasses of `BaseAdapter` MUST implement these 7 abstract methods:
- `load_pipeline()` → returns a DiffusionPipeline
- `encode_prompt()` → text → embeddings
- `encode_image()` → image → latents
- `encode_video()` → video frames → latents
- `decode_latents()` → latents → pixels
- `inference()` → full multi-step denoising (corresponds to pipeline `__call__`)
- `forward()` → single-step denoising for training loss computation

Note: `preprocess_func()` is a **concrete method** on `BaseAdapter` that calls the abstract encoding methods above. It does NOT need to be overridden unless the model requires non-standard preprocessing.

Breaking any of these signatures breaks the entire training pipeline.

### 13. BaseRewardModel Paradigm Split
- `PointwiseRewardModel.__call__` receives batches of size `batch_size`, returns rewards of shape `(batch_size,)`
- `GroupwiseRewardModel.__call__` receives all samples in a group (size `group_size`), returns rewards of shape `(group_size,)`

The `RewardProcessor` dispatches differently based on the model type. Do not change the calling convention.

### 14. Sample Dataclass Hierarchy
`BaseSample` → `T2ISample`, `ImageConditionSample`, `T2VSample`, etc. The `_shared_fields` class variable determines which fields are NOT stacked across a batch. Incorrect `_shared_fields` causes silent data corruption during collation.

---

## Configuration System (15–17)

### 15. Pydantic Hparams Synchronization
All config dataclasses live in `hparams/`. The top-level `Arguments` aggregates `DataArguments`, `ModelArguments`, `TrainingArguments`, `RewardArguments`, `LogArguments`, etc. Changes to these dataclasses MUST be reflected in:
1. The dataclass definition
2. ALL YAML configs under `examples/` — this includes:
   - **Field renames or removals**: Update every occurrence in all configs (old keys silently fall back to defaults)
   - **New user-facing fields**: Add to ALL example configs with the default value and an `# Options: ...` comment, so users can discover and override them
   - **New algorithm-specific subclass fields**: Add to the corresponding algorithm's example configs
3. Any code that accesses `config.<field_name>`

> **Why add new fields explicitly?** Pydantic dataclasses have defaults, so omitted fields don't break configs. But users copy-paste from examples — if a field isn't shown, they won't know it exists. Treat `examples/` as the user-facing API surface.

### 16. Algorithm-Specific Training Args
`TrainingArguments` has algorithm-specific subclasses (`GRPOTrainingArguments`, `NFTTrainingArguments`, `AWMTrainingArguments`). The correct subclass is resolved by `get_training_args_class()`. Adding a new algorithm requires adding a corresponding subclass and updating the resolver.

### 17. YAML Config Structure
Example configs follow this structure:
```yaml
model:
  model_type: "flux1"        # Must match registry key
  model_path: "..."
train:
  trainer_type: "grpo"       # Must match registry key
  dynamics_type: "Flow-SDE"  # Must be valid dynamics
data:
  dataset: "..."
rewards:
  reward_model: "PickScore"  # Must match registry key
```
Keys must exactly match the Pydantic field names. Typos fail silently with default values.

---

## Distributed Training (18–20)

### 18. All-Rank Synchronization Points
`accelerator.wait_for_everyone()` must be called at critical synchronization points (after preprocessing, before/after evaluation, checkpoint saving). Missing barriers cause deadlocks or race conditions.

### 19. FSDP CPU Efficient Loading
When using FSDP with CPU offloading, frozen components (text encoder, VAE) may be uninitialized on Rank > 0. The `_synchronize_frozen_components()` method handles this. Do not remove or bypass it.

### 20. Mixed Precision Consistency
The adapter sets inference dtype for frozen components and training dtype for trainable parameters in `_mix_precision()`. Autocast context is configured in `BaseTrainer.__init__`. Do not manually cast tensors unless you understand the precision boundary.

---

## Code Quality (21–24)

### 21. Formatting Standards
- **Black** with `line-length=100`, targeting Python 3.10–3.12
- **isort** with `profile="black"`, `line_length=100`
- Comments and docstrings in **English**

### 22. Import Style
- Use relative imports within `flow_factory` package (e.g., `from ..hparams import *`)
- Use absolute imports for external packages
- Follow existing wildcard import patterns for `hparams`

### 23. Type Annotations
All public methods must have type annotations. Use `typing` module types (`List`, `Dict`, `Optional`, `Tuple`, `Union`) for Python 3.10 compatibility.

### 24. License Header
All source files must include the Apache 2.0 license header with `Copyright 2026 Jayce-Ping`.

### 25. Fail-Fast Error Handling
The codebase prefers **raising exceptions with detailed debug information** over silent auto-fallback for unexpected situations. Do not add defensive fallback code that silently recovers from invalid inputs — instead, let the error propagate with a clear message so the caller can diagnose and fix the root cause. Examples of what NOT to do:
- Silently substituting a default range when no valid indices are found
- Auto-swapping `start`/`end` when they are reversed
- Clamping out-of-range values to valid bounds without reporting

Auto-fallback is only acceptable when the user explicitly requests it or when documented as intentional design (e.g., `_standardize_timestep_range` coercing a scalar to a tuple is a deliberate convenience, not error recovery).
