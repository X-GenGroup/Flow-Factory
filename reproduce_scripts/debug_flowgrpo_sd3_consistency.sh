#!/bin/bash
# =============================================================================
# Debug script for GRPO train-inference consistency analysis (SD3.5)
# Flow-Factory Version
#
# This script runs a minimal GRPO training loop with tensor dumping enabled
# to diagnose ratio drift (new_log_prob != old_log_prob on on-policy step).
#
# When FLOW_FACTORY_DEBUG_OUTPUT_DIR is set (done automatically here):
#   - All ranks dump per-step SDE tensors during sampling
#   - Training-side shuffle is DISABLED automatically for exact correspondence
#   - All ranks dump per-step SDE tensors during training (inner_epoch=0 only)
#
# Debug tensors are saved with per-rank, per-batch organization:
#   ${DEBUG_OUTPUT_DIR}/sampling/rank_RRR/batch_BBB/step_SSS/  -- sampling tensors
#   ${DEBUG_OUTPUT_DIR}/training/rank_RRR/batch_BBB/step_SSS/  -- training tensors
#
# After running, use scripts/analyze_debug_tensors.py to compare:
#   python scripts/analyze_debug_tensors.py ${DEBUG_OUTPUT_DIR}
#   python scripts/analyze_debug_tensors.py ${DEBUG_OUTPUT_DIR} --rank 0 --batch 0 --verbose
#
# Usage:
#   bash reproduce_scripts/debug_flowgrpo_sd3_consistency.sh
#   # With custom number of GPUs:
#   NUM_GPUS=4 bash reproduce_scripts/debug_flowgrpo_sd3_consistency.sh
#   # With custom debug output directory:
#   DEBUG_OUTPUT_DIR=/tmp/debug_out bash reproduce_scripts/debug_flowgrpo_sd3_consistency.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEBUG_OUTPUT_DIR=${DEBUG_OUTPUT_DIR:-"${REPO_ROOT}/debug_output"}
CONFIG_FILE="${REPO_ROOT}/examples/grpo/lora/sd3_5_debug.yaml"
NUM_GPUS=${NUM_GPUS:-8}

export FLOW_FACTORY_DEBUG_OUTPUT_DIR="${DEBUG_OUTPUT_DIR}"
export FLOW_FACTORY_DEBUG_MAX_EPOCHS="1"

echo "============================================"
echo " Flow-Factory GRPO Train-Inference"
echo " Consistency Debug (SD3.5)"
echo "============================================"
echo " DEBUG_OUTPUT_DIR: ${DEBUG_OUTPUT_DIR}"
echo " CONFIG_FILE:      ${CONFIG_FILE}"
echo " NUM_GPUS:         ${NUM_GPUS}"
echo ""
echo " Debug features enabled:"
echo "   - All ranks save tensors (rank_RRR/batch_BBB/step_SSS)"
echo "   - Training shuffle DISABLED for exact correspondence"
echo "   - Only 1 epoch, 1 inner_epoch (pure on-policy)"
echo "============================================"

rm -rf "${DEBUG_OUTPUT_DIR}"
mkdir -p "${DEBUG_OUTPUT_DIR}"

source /root/flow-factory/bin/activate
ff-train "${CONFIG_FILE}"

echo ""
echo "============================================"
echo " Debug tensors saved to: ${DEBUG_OUTPUT_DIR}"
echo ""
echo " Directory structure:"
echo "   ${DEBUG_OUTPUT_DIR}/"
echo "   ├── sampling/rank_000/batch_000/step_000/..."
echo "   └── training/rank_000/batch_000/step_000/..."
echo ""
echo " Run analysis (all ranks, all batches):"
echo "   python scripts/analyze_debug_tensors.py ${DEBUG_OUTPUT_DIR}"
echo ""
echo " Run analysis (specific rank and batch):"
echo "   python scripts/analyze_debug_tensors.py ${DEBUG_OUTPUT_DIR} --rank 0 --batch 0 -v"
echo "============================================"
