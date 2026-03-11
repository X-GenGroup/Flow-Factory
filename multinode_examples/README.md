
# Multi-Node Training with Flow-Factory

This directory contains examples and utilities for running Flow-Factory training across multiple nodes.

## How It Works

`ff-train` (the CLI entry point) automatically detects multi-node cluster environments by reading
well-known environment variables. There is **no need** to maintain separate configs for single-node
vs. multi-node — the same YAML config works for both.

### Three-Layer Config Merging

Launch parameters are resolved with the following priority (highest to lowest):

| Priority | Source | Example |
|----------|--------|---------|
| 1 (highest) | CLI arguments | `--num_machines 4 --main_process_ip 10.0.0.1` |
| 2 | Environment variables | `MASTER_IP`, `NUM_MACHINES`, `MACHINE_RANK`, etc. |
| 3 (lowest) | YAML config | `num_processes: 8`, `main_process_port: 29500` |

### Supported Environment Variables

The following environment variables are auto-detected (listed by priority within each group):

| Parameter | Env Var (Priority 1) | Env Var (Priority 2) | Env Var (Priority 3) |
|-----------|----------------------|----------------------|----------------------|
| Master IP | `MASTER_IP` | `MASTER_ADDR` | `CHIEF_IP` |
| Master Port | `MASTER_PORT` | — | — |
| Node Rank | `MACHINE_RANK` | `NODE_RANK` | `INDEX` |
| Num Nodes | `NUM_MACHINES` | `NUM_NODES` | `HOST_NUM` |
| GPUs per Node | `GPUS_PER_NODE` | `HOST_GPU_NUM` | — |

Multi-node mode is activated when **both** a master IP and `num_nodes > 1` are detected.

## Quick Start

### Option 1: Direct `ff-train` (Recommended)

When the cluster scheduler injects the required environment variables, simply run:

```bash
ff-train multinode_examples/train.yaml
```

`ff-train` will automatically:
1. Detect multi-node env vars (`MASTER_IP`, `NUM_MACHINES`, etc.)
2. Compute `num_processes = num_nodes * gpus_per_node`
3. Build and execute the appropriate `accelerate launch` command

### Option 2: Via `launch_multinode.sh`

A convenience wrapper that adds logging:

```bash
bash multinode_examples/launch_multinode.sh multinode_examples/train.yaml
```

### Option 3: CLI Overrides

Override any launch parameter directly from the command line:

```bash
ff-train multinode_examples/train.yaml \
  --num_machines 4 \
  --machine_rank 0 \
  --main_process_ip 10.0.0.1 \
  --main_process_port 29500 \
  --num_processes 32
```

## Files in This Directory

| File | Description |
|------|-------------|
| `train.yaml` | Example training config for multi-node (same format as single-node configs) |
| `fsdp2_wan.yaml` | Example accelerate config with FSDP2 HYBRID_SHARD for multi-node Wan training |
| `launch_multinode.sh` | Convenience shell wrapper around `ff-train` |

## Example: 4-Node FSDP2 Training

1. **`train.yaml`** — Point `config_file` to the multi-node accelerate config:
   ```yaml
   config_file: multinode_examples/fsdp2_wan.yaml
   num_processes: 32  # 4 nodes x 8 GPUs (used as fallback if env vars are absent)
   ```

2. **`fsdp2_wan.yaml`** — Configure FSDP2 with HYBRID_SHARD topology:
   ```yaml
   fsdp_config:
     fsdp_sharding_strategy: HYBRID_SHARD
     fsdp_device_mesh_shape: [4, 8]  # [num_nodes, gpus_per_node]
   ```

3. **Launch on each node** (scheduler handles env vars):
   ```bash
   ff-train multinode_examples/train.yaml
   ```

## Notes

- The YAML `num_processes` field acts as a **fallback** only. When env vars are detected, the actual
  process count is computed dynamically as `num_nodes * gpus_per_node`.
- If your cluster scheduler uses non-standard env var names, map them manually before launching:
  ```bash
  export MASTER_IP=${MY_CUSTOM_MASTER_IP}
  export NUM_MACHINES=${MY_CUSTOM_NODE_COUNT}
  export MACHINE_RANK=${MY_CUSTOM_RANK}
  ff-train your_config.yaml
  ```
- `accelerate config_file` (e.g., `fsdp2_wan.yaml`) still controls FSDP strategy, mixed precision,
  and other accelerate-specific settings. The multi-node launch parameters (`num_machines`,
  `machine_rank`, `main_process_ip`, etc.) are injected as CLI args to `accelerate launch`,
  which override whatever is written in the accelerate config file.
