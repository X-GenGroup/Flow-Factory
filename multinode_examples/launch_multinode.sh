#!/bin/bash
# multinode_examples/launch_multinode.sh
#
# Multi-node launch script for Flow-Factory.
# Now simply delegates to `ff-train`, which auto-detects multi-node env vars.
#
# Required environment variables (injected by cluster scheduler):
#   MASTER_ADDR / CHIEF_IP / MASTER_IP   - Master node IP
#   MASTER_PORT                          - Master node port
#   NODE_RANK / INDEX / MACHINE_RANK     - Current node rank
#   NUM_NODES / HOST_NUM / NUM_MACHINES  - Total number of nodes
#   HOST_GPU_NUM / GPUS_PER_NODE         - GPUs per node
#
# Usage:
#   bash launch_multinode.sh <train_config.yaml> [extra_args...]
#
# Example:
#   bash launch_multinode.sh multinode_examples/train.yaml
#
# If your cluster uses non-standard variable names, map them before calling:
#   export MASTER_ADDR=${MY_CUSTOM_MASTER_IP}
#   export NUM_NODES=${MY_CUSTOM_NODE_COUNT}

set -euo pipefail

TRAIN_CONFIG=${1:?"Usage: bash launch_multinode.sh <train_config.yaml> [extra_args...]"}
shift  # Remove first arg so "$@" contains only extra args

echo "=== Flow-Factory Multi-Node Launch ==="
echo "Master:         ${MASTER_ADDR:-${CHIEF_IP:-unknown}}:${MASTER_PORT:-unknown}"
echo "Num nodes:      ${NUM_NODES:-${HOST_NUM:-unknown}}"
echo "GPUs per node:  ${HOST_GPU_NUM:-unknown}"
echo "Node rank:      ${NODE_RANK:-${INDEX:-unknown}}"
echo "Train config:   ${TRAIN_CONFIG}"
echo ""

# ff-train auto-detects multi-node env vars and builds the accelerate launch command
ff-train "${TRAIN_CONFIG}" "$@" 2>&1 | tee "train_node_${NODE_RANK:-${INDEX:-0}}.log"
