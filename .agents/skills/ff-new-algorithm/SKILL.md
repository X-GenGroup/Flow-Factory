---
name: ff-new-algorithm
description: "Complete workflow for adding a new RL training algorithm. Covers paradigm selection, TrainingArguments subclass, trainer implementation, registry, example config, and verification. Trigger: 'add algorithm', 'new trainer', 'new training method', 'implement algorithm'."
---

# New RL Algorithm Integration

> **Authoritative reference**: `guidance/algorithms.md`

## Prerequisites

Determine your algorithm's characteristics:
- **Paradigm**: Coupled (needs log-probabilities, must use SDE) or Decoupled (solver-agnostic, can use ODE)?
- **Dynamics**: Which SDE/ODE formulation? (`Flow-SDE`, `Dance-SDE`, `CPS`, `ODE`)
- **Advantage**: How are advantages computed from rewards? (Most algorithms can delegate to `AdvantageProcessor`)
- **Loss**: What is the policy optimization objective?

## Phase 1: Design

1. **Study existing implementations**:
   - Coupled example: `trainers/grpo.py` (GRPO)
   - Decoupled example: `trainers/nft.py` (DiffusionNFT) or `trainers/awm.py` (AWM)
2. **Identify what's shared vs unique**:
   - Shared: Data loading, reward computation, `AdvantageProcessor`, adapter interface, checkpoint logic
   - Unique: `start()` method, loss function, algorithm-specific hyperparameters
   - Note: All trainers (GRPO, NFT, AWM) extend `BaseTrainer` directly. Only `GRPOGuardTrainer` extends `GRPOTrainer`.

## Phase 2: Configuration

### Step 1 — Define Algorithm-Specific Arguments

Add a new `TrainingArguments` subclass in `src/flow_factory/hparams/training_args.py`:

```python
class MyAlgoTrainingArguments(TrainingArguments):
    """Training arguments specific to MyAlgo."""
    my_specific_param: float = 0.1
    another_param: int = 10
```

### Step 2 — Register in Argument Resolver

Update `get_training_args_class()` in `hparams/training_args.py`:

```python
def get_training_args_class(trainer_type: str):
    mapping = {
        'grpo': GRPOTrainingArguments,
        'nft': NFTTrainingArguments,
        'awm': AWMTrainingArguments,
        'my_algo': MyAlgoTrainingArguments,  # Add this
    }
    return mapping.get(trainer_type, TrainingArguments)
```

## Phase 3: Trainer Implementation

### Step 3 — Create Trainer Class

```python
# src/flow_factory/trainers/my_algo.py
from .abc import BaseTrainer

class MyAlgoTrainer(BaseTrainer):
    """My custom RL algorithm trainer."""

    def start(self):
        """Main training loop — implements the 6-stage pipeline."""
        # Stage 1: Data & rewards initialized in BaseTrainer.__init__
        while self.should_continue_training():
            # Checkpoint & evaluation (standard pattern)
            if self.log_args.save_freq > 0 and self.epoch % self.log_args.save_freq == 0:
                self.save_checkpoint(save_dir, epoch=self.epoch)
            if self.eval_args.eval_freq > 0 and self.epoch % self.eval_args.eval_freq == 0:
                self.evaluate()

            # Stage 2+3: Sampling & trajectory generation
            samples = self.sample()

            # Stage 4+5+6: Reward, advantage, optimization
            self.optimize(samples)

            self.adapter.ema_step(step=self.epoch)
            self.epoch += 1

    def evaluate(self):
        """Evaluation loop — reuse pattern from GRPO/NFT."""
        pass

    def sample(self):
        """Stages 2-3: K-repeat sampling + trajectory generation."""
        # Use self.adapter.inference() for trajectory generation
        pass

    def compute_advantages(self, samples, rewards, store_to_samples=True, aggregation_func=None):
        """Delegates to AdvantageProcessor — standard pattern for all trainers."""
        aggregation_func = aggregation_func or self.training_args.advantage_aggregation
        return self.advantage_processor.compute_advantages(
            samples=samples,
            rewards=rewards,
            store_to_samples=store_to_samples,
            aggregation_func=aggregation_func,
            step=self.step,
        )

    def optimize(self, samples):
        """Stages 4-6: Reward → advantage → policy update."""
        # Stage 4: Reward computation
        rewards = self.reward_buffer.finalize(store_to_samples=True, split='all')

        # Stage 5: Advantage computation (delegates to AdvantageProcessor)
        advantages = self.compute_advantages(samples, rewards)

        # Stage 6: Policy optimization
        # Use self.adapter.forward() for single-step denoising
        # Compute loss, backprop, step
        pass
```

> **Note**: `AdvantageProcessor` is automatically instantiated in `BaseTrainer._init_reward_model()`.
> All trainers should delegate advantage computation via `self.advantage_processor.compute_advantages()`
> rather than implementing their own gather/scatter logic.

### Step 4 — Register in Trainer Registry

Add to `_TRAINER_REGISTRY` in `src/flow_factory/trainers/registry.py`:

```python
'my_algo': 'flow_factory.trainers.my_algo.MyAlgoTrainer',
```

## Phase 4: Configuration & Examples

Create example config `examples/my_algo/lora/flux1.yaml`:

```yaml
model:
  model_type: "flux1"
  model_path: "black-forest-labs/FLUX.1-dev"
  finetune_type: "lora"
  target_components: ["transformer"]

train:
  trainer_type: "my_algo"
  dynamics_type: "ODE"          # Or appropriate dynamics
  my_specific_param: 0.1
  learning_rate: 1e-6
  group_size: 4
  num_inference_steps: 28

data:
  dataset: "path/to/dataset"

rewards:
  reward_model: "PickScore"
  batch_size: 16
```

## Phase 5: Verification

- [ ] `MyAlgoTrainingArguments` correctly parsed from YAML
- [ ] `get_training_args_class('my_algo')` returns correct subclass
- [ ] `get_trainer_class('my_algo')` loads `MyAlgoTrainer`
- [ ] Training runs end-to-end for ≥2 epochs without errors
- [ ] Loss values are numerically reasonable (not NaN, decreasing)
- [ ] Rewards improve over training
- [ ] Checkpoint save/load works correctly
- [ ] Works with at least two different model adapters
- [ ] Coupled algorithms only use SDE dynamics
- [ ] Decoupled algorithms work with both SDE and ODE dynamics

## Common Pitfalls

1. **Not subclassing `TrainingArguments`** — algorithm-specific params won't be parsed from YAML
2. **Forgetting `get_training_args_class` update** — falls back to base `TrainingArguments`, losing custom params
3. **Using ODE with coupled paradigm** — no log-probabilities available, silent incorrect gradients
4. **Not calling `self.should_continue_training()`** — infinite loop if `max_epochs` is set
5. **Duplicating `_initialization()` logic** — already called in `BaseTrainer.__init__`; don't re-prepare modules
6. **Reimplementing advantage gather/scatter** — use `self.advantage_processor.compute_advantages()` instead; it handles both sampler topologies automatically
7. **Extending `GRPOTrainer` unnecessarily** — unless your algorithm extends GRPO's PPO-clipped loss, extend `BaseTrainer` directly (as NFT and AWM do)
