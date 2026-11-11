# Start parallel training on one node

This walkthrough uses SD3.5 Medium, LoRA SFT and the two checked-in demonstration records. It
isolates the distributed layout from reward services and generates three complete training YAMLs.
The fixture is a functional starting point, not a meaningful fine-tuning dataset or a memory
benchmark. Obtain access to the model weights before launching and replace the fixture with your
own [V2 demonstrations](../training/datasets.md) for a real run.

Run from the repository root after [installation](../installation.md). You need four local CUDA
GPUs for either TP/SP or CP, and eight for their composition. These are **FSDP2 trainer** layouts:
a two-GPU TP component test does not establish a two-GPU FSDP2+TP training recipe.

## Generate complete configurations

The checked-in [SD3.5 SFT recipe](../../examples/sft/lora/sd3_5/default.yaml) supplies the model,
dataset, objective, optimizer and logging settings. This script copies it and explicitly sets
each distributed layout. It does not modify the checked-in recipe or start training.

```bash
python - <<'PYCONFIG'
from copy import deepcopy
from pathlib import Path
import yaml

base = yaml.safe_load(Path("examples/sft/lora/sd3_5/default.yaml").read_text())
output = Path(".scratch/parallel-quickstart")
output.mkdir(parents=True, exist_ok=True)
layouts = {
    "tp-sp": (4, 2, 1, True),
    "cp": (4, 1, 2, False),
    "tp-sp-cp": (8, 2, 2, True),
}
for name, (ranks, tp, cp, sp) in layouts.items():
    config = deepcopy(base)
    config.update(config_file="config/accelerate_configs/fsdp2.yaml",
                  num_processes=ranks, mixed_precision="bf16")
    config["parallelism"] = dict(dp_replicate_size=1, dp_shard_size=2,
                                 tp_size=tp, cp_size=cp, sequence_parallel=sp)
    config["model"]["target_components"] = ["transformer"]
    config["train"].update(per_device_batch_size=1, gradient_accumulation_steps=1)
    config["acceleration"] = {
        "shared": [{"name": "attention_backend", "params": {"backend": "native"}}]
    }
    config["log"]["run_name"] = "sd35-sft-" + name
    path = output / (name + ".yaml")
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    print(path)
PYCONFIG
```

Each generated YAML is standalone: inspect and edit it before use. Put training options in the
YAML; do not assume arbitrary nested CLI flags override it. `ff-train` supports launcher flags
such as `--num_processes`, but the rank count must still agree with the YAML's parallel geometry.

## Launch one layout

Run one command at a time from the repository root, in an ordinary single-node shell without
multi-node launcher environment overrides:

```bash
# Four GPUs: two data replicas, each using TP2 with SP.
CUDA_VISIBLE_DEVICES=0,1,2,3 ff-train .scratch/parallel-quickstart/tp-sp.yaml

# Four GPUs: two data replicas, each using CP2; TP/SP are off.
CUDA_VISIBLE_DEVICES=0,1,2,3 ff-train .scratch/parallel-quickstart/cp.yaml

# Eight GPUs: two data replicas, each using TP2 x CP2 with SP.
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ff-train .scratch/parallel-quickstart/tp-sp-cp.yaml
```

The launcher uses [FSDP2](../../config/accelerate_configs/fsdp2.yaml), BF16 and backend-owned
activation checkpointing. That launcher disables CPU-RAM-efficient loading, as CP currently
requires. Leave `train.enable_gradient_checkpointing: false` from the base recipe so a second
model-owned checkpointing policy does not compete with the backend.

All three layouts have two logical data replicas. With one example per device and one accumulation
step, an optimizer update consumes two logical examples, not four or eight. The tiny fixture
therefore supplies one global microbatch per epoch. TP/CP peers cooperate on the same examples.
Model downloads and preprocessing still need their normal resources; this guide does not assert
that arbitrary resolutions or variants fit a particular GPU capacity.

## Adapt the recipe

Use the [configuration guide](configuration.md) before replacing the model or algorithm. In
particular, current CP requires native attention, BF16, AdamW, batch size one and FSDP2; Wan also
requires `model.compute_precision_policy: cross_attention_fp32_v1`. Ordered image references,
head divisibility and token geometry remain adapter-specific. SP requires TP and is explicitly
selected in these recipes.

Use [algorithm coverage](algorithms.md) for online rewards, roles and overlap, and
[model coverage](models.md) for full/LoRA and activation-region restrictions. Offline SFT does
not exercise online reward overlap. Changing the algorithm means selecting its complete recipe
and data/reward requirements, not only replacing `trainer_type` in this SFT YAML.

## Evidence and troubleshooting

The recipes assemble existing configuration contracts. They do not claim three new real-weight
GPU executions: numerical and lifecycle coverage remains in the linked
[model references](models.md), [v0.5 qualification](../../docs/release/v0.5-validation.md) and
[Torch 2.10–2.14 qualification](../../docs/development/validation/torch210-214.md).

| Symptom | Check first |
|---|---|
| World-size mismatch | Launched processes versus `dp_replicate_size * dp_shard_size * cp_size * tp_size` |
| CP backend/precision rejection | FSDP2 launcher, BF16, native attention, AdamW and the model's precision policy |
| Head/token divisibility failure | Exact model variant, text/image/video/audio lengths and declared padding rules |
| Out of memory | Frozen encoders/VAE preprocessing versus denoiser load versus training activations; see [loading](../configuration/loading-and-resume.md) |
| Different resumed run | Weights-only restart versus exact state restore, topology and ordered/ready scheduling |
