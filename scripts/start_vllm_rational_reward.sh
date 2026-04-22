#!/usr/bin/env bash
# Start vLLM OpenAI-compatible server for Rational Rewards judge weights on Hugging Face:
#   - T2I weights: TIGER-Lab/RationalRewards-8B-T2I  → served OpenAI model id: RationalRewards-8B-T2I
#   - Edit weights: TIGER-Lab/RationalRewards-8B-Edit → served OpenAI model id: RationalRewards-8B-Edit
# Training YAML: api_base_url=http://<host>:<port>/v1 and vlm_model must equal --served-model-name.
#
# Usage (2 GPUs; data-parallel-size defaults to len(CUDA_VISIBLE_DEVICES)):
#   export CUDA_VISIBLE_DEVICES=0,1
#   export MODEL_PATH="TIGER-Lab/RationalRewards-8B-T2I"
#   ./scripts/start_vllm_rational_reward.sh --max-model-len 8192
#
# Edit judge:
#   export MODEL_PATH="TIGER-Lab/RationalRewards-8B-Edit"
#   ./scripts/start_vllm_rational_reward.sh --max-model-len 8192
#
# Optional environment variables:
#   VLLM_BIN          vLLM entrypoint (default: vllm from PATH)
#   PORT              listen port (default: 8000)
#   HOST              bind address (default: 0.0.0.0)
#   SERVED_MODEL_NAME OpenAI "model" id (default: RationalRewards-8B-T2I or RationalRewards-8B-Edit
#                       inferred from MODEL_PATH; override if you use a custom --served-model-name, must match "vlm" key in YAML config)
#   TENSOR_PARALLEL_SIZE  (default: 1)
#   DATA_PARALLEL_SIZE    If unset: number of entries in CUDA_VISIBLE_DEVICES (comma-separated),
#                           or 1 if CUDA_VISIBLE_DEVICES is unset. Set explicitly to override.
#   GPU_MEMORY_UTILIZATION (default: 0.9)
#   Any extra arguments are forwarded to `vllm serve` (e.g. --max-model-len 8192).

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-TIGER-Lab/RationalRewards-8B-T2I}"

if [[ -n "${SERVED_MODEL_NAME:-}" ]]; then
  :
elif [[ "${MODEL_PATH}" == *"RationalRewards-8B-Edit"* ]]; then
  SERVED_MODEL_NAME="RationalRewards-8B-Edit"
else
  SERVED_MODEL_NAME="RationalRewards-8B-T2I"
fi

VLLM_BIN="${VLLM_BIN:-vllm}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"

case "${DATA_PARALLEL_SIZE-unset}" in
  unset)
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
      DATA_PARALLEL_SIZE="$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')"
    else
      DATA_PARALLEL_SIZE=1
    fi
    ;;
esac

exec "${VLLM_BIN}" serve "${MODEL_PATH}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --data-parallel-size "${DATA_PARALLEL_SIZE}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  "$@"
