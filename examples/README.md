# Examples

Training configs for all supported algorithm–model combinations.

## Directory Structure

```
examples/{algorithm}/{finetune_type}/{model_type}/{variant}.yaml
```

| Level | Description | Examples |
|-------|-------------|---------|
| `algorithm` | Training algorithm | `grpo`, `nft`, `awm`, `dgpo`, `dpo` |
| `finetune_type` | Parameter-efficient or full | `lora`, `full` |
| `model_type` | Model family (underscore-separated) | `flux1`, `sd3_5`, `wan21`, `ltx2` |
| `variant` | Config variant | `default.yaml`, `nocfg.yaml`, `t2v.yaml` |

**Naming rules**:
- Model directory names use underscores matching the config's `model_type` field (e.g., `sd3-5` → `sd3_5`, `flux1-kontext` → `flux1_kontext`).
- `default.yaml` is the baseline config for a model. Use descriptive names for variants (`nocfg.yaml`, `rational_rewards_t2i.yaml`, `t2v.yaml`, `i2v.yaml`).

**Quick start**:
```bash
ff-train examples/grpo/lora/flux1/default.yaml
```

## Contributing

We welcome community contributions! Here's what you can contribute and how:

### Verified Training Configs

If you've tested a model–algorithm combination and confirmed reward improvement, submit a PR with:
- The config YAML following the directory structure above
- A brief note in the PR description about hardware used and observed reward trend

### Custom Reward Models

New reward models are welcome — add the implementation under `src/flow_factory/rewards/` and include an example config that uses it. Please ensure your reward model's dependencies are compatible with the existing environment (check `pyproject.toml`).

### New Model Adapters

See the [New Model Guide](../guidance/new_model.md) for how to add a new diffusion/flow-matching model. Include at least one example config with your PR.

### Guidelines

- Configs should be self-contained and runnable with `ff-train`
- Include comments for non-obvious parameter choices
- If your config requires a specific dataset, document how to obtain it
- Test on at least one hardware configuration before submitting
