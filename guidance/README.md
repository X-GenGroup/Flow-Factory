# Flow-Factory user guide

Start with a model and a training objective, then choose how to use your GPUs. Flow-Factory
checks that the model, algorithm and resource layout can work together.

| Guide | Use it to |
|---|---|
| [Feature overview](overview.md) | Choose models, paradigms, algorithms, tuning and backend responsibilities |
| [Installation](installation.md) · [Loading and resume](configuration/loading-and-resume.md) | Choose a compatible environment and migrate checkpoints |
| [Training workflow](training/workflow.md) | Follow data preparation, generation, feedback and optimization |
| [Algorithms](training/algorithms.md) | Choose online learning, offline training or distillation |
| [Datasets](training/datasets.md) · [Rewards](training/rewards.md) | Configure training inputs and feedback |
| [Parallel quickstart](parallelism/quickstart.md) | Generate and launch complete 4/4/8-GPU SD3.5 SFT configurations |
| [Parallel training](parallelism/README.md) | Understand DP, TP, SP and CP, then choose a GPU layout |
| [Model coverage](parallelism/models.md) · [Algorithm coverage](parallelism/algorithms.md) | Check supported regions and composition limits |
| [Precision](configuration/compute-precision.md) · [Acceleration](configuration/acceleration.md) | Set compute precision and optional accelerators |
| [Logging](configuration/logging.md) | Choose a backend and bound media encoding off the training thread |
| [Async rewards](training/workflow/async_reward.md) · [Reward overlap](training/workflow/reward_optimization_overlap.md) | Run feedback alongside useful training work |
| [Prompt enhancement](training/prompt-enhance.md) | Prepare enhanced prompts before training |

Start from a nearby [example](../examples/README.md) and use the guides to understand its choices.
Implementation contracts and detailed numerical evidence live in the
[developer documentation](../docs/README.md).

For reward-gradient policy training, see [ReFL](training/refl.md).
