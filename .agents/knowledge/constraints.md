# Hard Constraints

Quick index: **#1-5** Registry | **#6-10** Training Pipeline | **#11-14** Base Classes | **#15-17** Config | **#18-20** Distributed | **#21-27** Code Quality

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
- **Decoupled** (DPO, NFT, AWM): Training timesteps are decoupled from sampling. Can use any dynamics including `ODE`.

Mixing paradigms (e.g., using `ODE` dynamics with `GRPO`) will produce incorrect gradients silently.

### 8. Component Offloading Lifecycle
Text encoders and VAEs are loaded for Stage 1 (preprocessing), then offloaded to free VRAM before the training loop. They are reloaded for inference during sampling. Do not assume these components are always on-device.

### 9. Accelerator `prepare()` Scope
Only **trainable modules** and the **optimizer** go through `accelerator.prepare()`. The dataloader uses a custom distributed sampler (`DistributedKRepeatSampler` or `GroupContiguousSampler`) and is NOT prepared via accelerator. Breaking this causes duplicate data or incorrect gradient accumulation.

### 9a. Sampler Geometric Constraints
Both samplers require `M * K ≡ 0 (mod W * B * G)` where M=unique_sample_num, K=group_size, W=world_size, B=per_device_batch_size, G=gradient_step_per_epoch — **unless** `gradient_accumulation_steps` is set manually, in which case the constraint reduces to `M * K ≡ 0 (mod W * B)`. **GroupContiguousSampler** adds: `M ≡ 0 (mod W)`. See `topics/samplers.md` for full details.

### 10. DeepSpeed ZeRO-3 Is Unsupported
Reward model sharding under ZeRO-3 is broken even with `GatherParameter` context manager (see `trainers/abc.py` line 119–123). Only ZeRO-1 and ZeRO-2 are safe. Document this if users ask.

---

## Base Class Interfaces (11–14)

### 11. BaseTrainer Abstract Contract
`BaseTrainer.__init__` expects `(accelerator, config, adapter)`. Subclasses must implement `start()`, `prepare_feedback()`, `optimize()`, and `evaluate()`. The `_initialization()` method handles dataloader, optimizer, accelerator preparation, reward model loading, and `AdvantageProcessor` instantiation — do not duplicate this logic.

**Per-epoch hook order**: `sample()` (Stages 2–3) → `prepare_feedback()` (Stages 4–5) → `optimize()` (Stage 6). `DPOTrainer` forms chosen/rejected pairs at the **start** of `optimize()` (not in `prepare_feedback()`).

**Trainer hierarchy**: `GRPOTrainer`, `DPOTrainer`, `DiffusionNFTTrainer`, `AWMTrainer` all extend `BaseTrainer` directly. Only `GRPOGuardTrainer` extends `GRPOTrainer`. All trainers delegate advantage computation to `self.advantage_processor.compute_advantages()`.

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
All config dataclasses live in `hparams/`. The top-level `Arguments` aggregates `DataArguments`, `ModelArguments`, `TrainingArguments`, `RewardArguments`, `LogArguments`, etc. Field changes MUST be reflected in:
1. The dataclass definition
2. ALL YAML configs under `examples/` (renames/removals: search-replace; new user-facing fields: add with defaults and `# Options:` comments)
3. Any code that accesses `config.<field_name>`

### 16. Algorithm-Specific Training Args
`TrainingArguments` has algorithm-specific subclasses (`GRPOTrainingArguments`, `DPOTrainingArguments`, `NFTTrainingArguments`, `AWMTrainingArguments`). The correct subclass is resolved by `get_training_args_class()`. Adding a new algorithm requires adding a corresponding subclass and updating the resolver.

### 17. YAML Config Structure
Config keys must exactly match Pydantic field names. Typos fail silently with default values. See `examples/` for canonical config templates; structure defined in `hparams/args.py`.

---

## Distributed Training (18–20)

### 18. All-Rank Synchronization Points
`accelerator.wait_for_everyone()` must be called at critical synchronization points (after preprocessing, before/after evaluation, checkpoint saving). Missing barriers cause deadlocks or race conditions.

### 19. FSDP CPU Efficient Loading
When using FSDP with CPU offloading, frozen components (text encoder, VAE) may be uninitialized on Rank > 0. The `_synchronize_frozen_components()` method handles this. Do not remove or bypass it.

### 20. Mixed Precision Consistency
The adapter sets inference dtype for frozen components and training dtype for trainable parameters in `_mix_precision()`. Autocast context is configured in `BaseTrainer.__init__`. Do not manually cast tensors unless you understand the precision boundary. Details: `topics/dtype_precision.md`.

---

## Code Quality (21–27)

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

### 25. Logger Message Style
Logger messages referencing config parameters MUST use user-facing field names (not shorthand like `M`, `K`, `W`), show concrete values in parentheses (e.g., `unique_sample_num_per_epoch(32)`), and structure multi-constraint messages with numbered lines.

### 26. Fail-Fast Error Handling
Raise exceptions with detailed debug information over silent auto-fallback. Do not add defensive fallback code that silently recovers from invalid inputs. Auto-fallback is only acceptable when documented as intentional design. Details: `.cursor/rules/no-defensive-except.mdc`.

### 27. Docstring Style
All public functions and methods must have Google-style docstrings in English: imperative one-liner summary, `Args:`, `Returns:`, optional `Note:`. Private helpers (`_func`) may use a one-liner docstring if the behavior is obvious.
