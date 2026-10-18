<!-- The parent SKILL.md selects the risk profile. Apply only the sections needed by that profile. -->

# New Training Algorithm Integration

## Context Routing

Read `../../../../guidance/algorithms.md`, `../../../../guidance/workflow.md`, and
`../../../knowledge/topics/train_inference_consistency.md` when the selected profile requires integration detail. Also read
`../../../knowledge/topics/component_variants.md` for more than one live trainable copy and
`../../../../guidance/datasets.md` for dataset acquisition.

## 1. Classify the Algorithm

Decide these independent properties before writing code:

- **Acquisition**: generated collection or finite dataset (`generation` / `dataset`).
- **Feedback**: runtime reward/advantage or none (`runtime_reward` / `none`).
- **Paradigm**: `coupled`, `decoupled`, or `distillation`.
- **Dynamics**: coupled objectives require SDE transition densities; decoupled/distillation
  objectives are solver-agnostic unless their own math narrows the choice.
- **Supervision**: prompts, demonstrations, preference pairs, or a new typed model-neutral record.
- **State**: legacy single-component or structured multi-component trajectory.
- **Ownership**: one live policy, several live trainable roles, or temporal reference/EMA snapshots.

Use a predefined immutable contract when it matches:

| Composition | Constant | Examples |
|---|---|---|
| `generation + runtime_reward` | `ONLINE_EXECUTION_CONTRACT` | GRPO, online DPO |
| `generation + none` | `ONLINE_NO_FEEDBACK_EXECUTION_CONTRACT` | DiffusionOPD, DMD2, TDM |
| `dataset + none` | `OFFLINE_EXECUTION_CONTRACT` | SFT, offline DPO |

Do not infer these axes from batch fields or expose them as user-configurable fields. A genuinely new
composition requires an explicit execution-contract/driver design, not a trainer-local branch.

Study the closest direct-`BaseTrainer` implementation: GRPO for coupled replay, NFT/AWM for
decoupled generation, DiffusionOPD or DMD2/TDM for reward-free generation, and SFT/offline DPO for
finite data. New trainers default to direct `BaseTrainer` inheritance; the existing sanctioned
strict extensions are GRPO-Guard/DPPO from GRPO and TDM-R1 from TDM.

## 2. Add Algorithm-Specific Arguments

Create `src/flow_factory/hparams/training_args/my_algo.py`. The arguments class and trainer class
must declare the same class-level contract; it is not serialized or user-overridable.

```python
from dataclasses import dataclass, field
from typing import ClassVar, Literal

from ...contracts.execution import OFFLINE_EXECUTION_CONTRACT, ExecutionContract
from ._offline import OfflineFlowMatchingTrainingArguments


@dataclass
class MyAlgoTrainingArguments(OfflineFlowMatchingTrainingArguments):
    """Configure MyAlgo over finite dataset acquisitions."""

    execution_contract: ClassVar[ExecutionContract] = OFFLINE_EXECUTION_CONTRACT
    trainer_type: Literal["my-algo"] = field(default="my-algo")
    my_specific_param: float = field(
        default=0.1,
        metadata={"help": "Describe the objective parameter."},
    )

    def __post_init__(self) -> None:
        """Validate the fixed trainer identity and objective parameters."""
        super().__post_init__()
        if self.trainer_type != "my-algo":
            raise ValueError("MyAlgoTrainingArguments requires trainer_type='my-algo'")
```

Dataset algorithms inherit `OfflineFlowMatchingTrainingArguments` so finite-loader cadence,
explicit accumulation, and offline flow-matching fields remain shared. Generated algorithms
inherit `TrainingArguments` and use the corresponding online contract constant. If an
optimize/reference branch uses stronger CFG than rollout, override
`get_preprocess_guidance_scale()` so negative conditions are encoded. Keep algorithm-owned
objective validation in this class.

Register and re-export the class in:

- `hparams/training_args/_registry.py`
- `hparams/training_args/__init__.py`
- `hparams/__init__.py`

## 3. Implement the Trainer Hook, Not the Loop

`BaseTrainer.start()` owns seeding, acquisition dispatch, periodic save/eval boundaries, progress,
EMA cadence, and `_after_acquisition_cycle()`. Do not override it or manually advance progress.
Place the implementation under `src/flow_factory/trainers/<category>/my_algo.py` so the relative
imports below match the existing `rl`, `distillation`, and `offline` package depth.

### Generated acquisition

```python
from typing import ClassVar, List, Literal

from ...contracts import ONLINE_EXECUTION_CONTRACT, ExecutionContract
from ...samples import BaseSample
from ..abc import BaseTrainer


class MyOnlineTrainer(BaseTrainer):
    """Optimize MyAlgo from generated examples and runtime feedback."""

    paradigm: ClassVar[Literal["decoupled"]] = "decoupled"
    execution_contract: ClassVar[ExecutionContract] = ONLINE_EXECUTION_CONTRACT

    def sample(self) -> List[BaseSample]:
        """Generate the state required by the objective."""
        return self.generate_samples(
            reward_buffer=self.reward_buffer,
            compute_log_prob=False,
            trajectory_indices=[-1],
        )

    def optimize(self, samples: List[BaseSample]) -> None:
        """Apply the algorithm objective to one generated collection."""
        ...
```

Prefer `generate_samples()` because it owns rollout mode, source/metadata propagation, reward
buffering, acceleration context, and loader iteration. Override acquisition internals only when the
algorithm truly needs a different collection shape and preserve those contracts.

Generated no-feedback trainers select `ONLINE_NO_FEEDBACK_EXECUTION_CONTRACT`, pass no training
reward buffer, and do not simulate absent feedback with reward no-ops. Override `_run_training_step`
only for a different grouping of sample/optimize work, while retaining the outer shared loop.

### Finite dataset acquisition

```python
from typing import Any, ClassVar, Dict, Literal, Tuple

from torch.utils.data import DataLoader

from ...contracts import OFFLINE_EXECUTION_CONTRACT, ExecutionContract
from ...data_utils.offline_train_data import build_offline_train_dataloader
from ..abc import BaseTrainer


class MyOfflineTrainer(BaseTrainer):
    """Optimize MyAlgo over one complete finite loader per data epoch."""

    paradigm: ClassVar[Literal["decoupled"]] = "decoupled"
    execution_contract: ClassVar[ExecutionContract] = OFFLINE_EXECUTION_CONTRACT

    def _build_train_dataloader(self) -> Tuple[DataLoader, Dict[str, DataLoader]]:
        """Build the official-distributed-sampler loader for the typed schema."""
        return build_offline_train_dataloader(...), {}

    def optimize_batch(self, batch: Any) -> None:
        """Apply one gradient-accumulation microstep from a dataset batch."""
        ...
```

Reuse the demonstration/preference loader when its schema matches; otherwise extend the typed data
layer first. The driver calls `set_epoch(data_epoch)` and advances only after clean exhaustion.
Offline optimization must:

- cache input conditions only and encode outputs on demand;
- call `adapter.prepare_condition_state()` once per batch;
- use `adapter.encode_output_state()` and adapter-owned offline forward overrides;
- preserve shared schedule/noise/reference scope for preference arms;
- use explicit positive gradient accumulation with no partial-window flush.

Evaluation remains generation-based.

## 4. Register the Trainer

Add the canonical lazy path to `_TRAINER_REGISTRY` in `trainers/registry.py`. Registry keys are
lowercase. A decorator is optional but never replaces the static entry. Verify trainer and argument
registry keys together and preserve direct Python-path fallback.

## 5. Add Multi-Role Training Only When Required

One live policy uses the default base variant and one optimizer entry. If several trainable copies
must coexist:

1. Declare ordered role names through `TrainingArguments.required_trainable_roles`; the first owns
   canonical base routes.
2. Return a `RoleUpdatePlan` when roles have different cadence.
3. Let `BaseTrainer._declare_model_variants()` materialize variants before prepare. Algorithm names
   remain in the trainer layer, not `models/`.
4. Run forwards under `adapter.use_component_variant(role)` and updates through
   `RoleOptimizationCoordinator` or existing role-runtime helpers.
5. Use temporal ref/named/EMA snapshots for frozen or time-shifted weights, never a live variant.
6. Add one top-level `optimizers:` entry per role. Muon may contribute two parameter groups for one
   role, so store group tuples rather than assuming one-to-one ownership.
7. Preserve one prepared `ModelBundle` and one optimizer root. Muon requires a PyTorch build with
   `torch.optim.Muon` and is supported on DDP/FSDP2, not DeepSpeed/FSDP1. Multi-role DeepSpeed is
   ZeRO-1/2 only. Multi-role FSDP2 requires `use_orig_params=True`; the framework rebinds the
   registry after prepare, and optimizer references must point to the prepared root's
   DTensor-backed parameters.
8. Declare checkpoint runtime children before exact resume. Resumable saves include all training
   roles, role counters, optimizer ownership, and variant snapshots.

Use DMD2/TDM/TDM-R1 and `../../../knowledge/topics/component_variants.md` as references.

## 6. Configuration and Documentation

Create `examples/{algorithm}/{finetune_type}/{model_type}/default.yaml`:

- runtime rewards only for `runtime_reward`; evaluation rewards are independent;
- dataset acquisition uses unit source weights, `sampler_type: auto`, `max_epochs`, and explicit
  accumulation;
- generation acquisition uses valid grouped-sampler geometry;
- coupled algorithms use SDE dynamics;
- `optimizers:` has one named entry per trainable role, including `max_grad_norm` and
  `optimizer: adamw|muon`;
- document the objective and selection in `guidance/algorithms.md` and workflow/data changes in the
  matching guides.

## 7. Verification

- Trainer/argument registries resolve and their contracts match before heavyweight loading.
- Hook validation selects `optimize` or `optimize_batch` correctly.
- Run at least two complete acquisition cycles; for dataset mode, test clean exhaustion and a failed
  batch that must not advance `data_epoch`.
- Verify numerical objective invariants and legacy/structured state when applicable.
- Verify model-only save/load and exact-state resume when state semantics change.
- Test at least two adapters when the objective is model-neutral; dataset algorithms require an
  offline-capable adapter.
- Cover DDP and an affected sharded backend. Multi-role changes cover DDP/ZeRO-2/FSDP2; Muon adds
  positive DDP/FSDP2 and early negative availability/DeepSpeed/FSDP1 cases.
- Run `/ff-review` before commit.

## Common Failures

- Missing or mismatched argument/trainer `execution_contract` or `paradigm`.
- Overriding `start()`, manually advancing progress, or duplicating `_initialization()`.
- Inferring online/offline behavior from a batch or using `optimize()` for finite data.
- Calling adapter inference directly while losing source, reward, or acceleration invariants.
- Adding trainer-to-trainer inheritance without matching a sanctioned strict extension and
  documenting why direct `BaseTrainer` plus shared helpers is insufficient.
- Reimplementing advantage gather/scatter.
- Caching target/chosen/rejected state or redrawing candidate-specific input conditions.
- Using rollout CFG semantics for offline flow matching.
- Treating a frozen reference as a variant, preparing roles separately, or assuming one group per
  role.
- Omitting training roles/runtime children from resumable checkpoints.
