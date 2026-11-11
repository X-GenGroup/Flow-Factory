# Choose a training workflow

Flow-Factory separates the model's computation from the algorithm's objective and the backend's
parameter placement. Start with the data and feedback you have, choose a model and algorithm,
then select a GPU layout. Adding parallelism does not change the intended learning objective.

## Choose the training paradigm

| Your inputs | Training path | Algorithms to explore |
|---|---|---|
| Target images, videos or audio-video demonstrations | Offline supervised matching; no training rollout or reward service | SFT |
| Chosen/rejected target pairs | Offline preference optimization with a reference | Offline DPO |
| Conditions and a scoring function | Generate samples, collect feedback, optimize | GRPO, GRPO-Guard, DPPO, NFT, AWM, DGPO, online DPO, CRD |
| Teacher predictions or trajectories | Generate training inputs and distill without a training reward service | DiffusionOPD, DMD2, TDM |
| Teacher/trajectory training plus reward feedback | Reward-guided multi-role training | TDM-R1 |

Distillation and online acquisition are not opposites: a distillation algorithm can generate its
own inputs. Evaluation rewards are also separate from training rewards. The
[algorithm guide](training/algorithms.md) explains objectives and tuning modes; the
[execution coverage](parallelism/algorithms.md) explains their state and parallel requirements.

Full tuning and LoRA choose which parameters update. Teacher/reference/EMA state and simultaneous
trainable roles describe different responsibilities. For example, DMD2's fake and generator
roles are not independent DP replicas. Use the existing algorithm recipe to establish those
roles rather than constructing extra models in a launcher.

## Choose a model and data contract

The [README model catalog](../README.md#-supported-models) lists checkpoints and registry keys.
Image and editing models include SD3.5, FLUX, Qwen Image, Z-Image, Bagel and SenseNova; Wan covers
video, and LTX-2/MiniMax H3 cover joint audio-video workflows. One family may have several adapters
or conditioning variants, so a family count is not an adapter count.

Use [V2 demonstrations/preferences](training/datasets.md) for offline training and the declared
conditioning schema for generated samples. Ordered reference images, cached prefixes, video
frames and audio streams are model inputs with their own ordering and geometry. They cannot be
replaced with a generic text prompt. Check the [model layout](parallelism/models.md) before
changing resolution, reference count, batch size or token lengths.

## Choose a training backend

| Backend | What it distributes | Good starting point | Current boundary |
|---|---|---|---|
| DDP | Samples; gradients synchronize while model and optimizer state are replicated | Model fits on each GPU and ordinary data parallelism is enough | Not the current TP/CP training backend |
| DeepSpeed ZeRO-2 | Samples plus optimizer/gradient state | Reduce optimizer memory while retaining replicated parameters | Requires the DeepSpeed extra; not the current TP/CP path |
| FSDP2 | Parameters, gradients and optimizer state across data peers | Sharded model state, or a declared TP/SP/CP composition | Requires the selected model/layout and backend configuration |

Keep three settings distinct: **FSDP2** is a training backend, **Ulysses** is the admitted CP
transport, and **native** is the attention backend used by current CP recipes. An optional Flash
attention backend is not an EP implementation and is not automatically admitted by CP.

Use the [parallel quickstart](parallelism/quickstart.md) for FSDP2 layouts, or choose a nearby
[existing recipe](../examples/README.md) for DDP/ZeRO-2. Memory saved by sharding can come with
communication overhead; choose based on the measured bottleneck rather than assuming more
parallel axes always improve throughput.

## Add feedback overlap and recovery deliberately

The [tile-stream overlap guide](training/workflow/reward_optimization_overlap.md) explains how
asynchronous CPU rewards overlap optimization for admitted algorithms and sampler layouts.
This is functional overlap; it does not promise that every reward service's latency is fully
hidden. Ready scheduling may change accumulation order without changing the objective contract.

[Training-state resume](configuration/loading-and-resume.md) restores optimizer and runtime
state at the admitted safe boundary and topology. Loading weights to start another run is a
different operation. EP expert placement and EP checkpoint semantics remain
[planned work](../roadmap/milestones/v0.6-v0.7.md).

## Read support and evidence together

A capability declaration describes accepted structure; a recipe demonstrates a concrete
configuration; a numerical or lifecycle test establishes its stated guarantee. These are not
interchangeable. Use the [model](parallelism/models.md) and [algorithm](parallelism/algorithms.md)
tables together, and retain the version/source identity of linked evidence.

The [installation guide](installation.md) distinguishes the published v0.5.0 environment from
latest-main compatibility work. The [development guide](../docs/README.md) is for contributors;
ordinary use should not require reading agent handoffs or reconstructing a PR stack.
