#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PRENORM="${PRENORM:-1}"
export LAYER_SCALE_INIT="${LAYER_SCALE_INIT:-0.0}"
export HEAD_DIM="${HEAD_DIM:-16}"
export ASPECT_RATIO="${ASPECT_RATIO:-40}"
export DEPTH="${DEPTH:-8}"
export H_CYCLES="${H_CYCLES:-1}"
export L_CYCLES="${L_CYCLES:-1}"
export TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-32768}"
export DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-2}"
export FORWARD_DTYPE="${FORWARD_DTYPE:-bfloat16}"
export AR_NUM_GPUS="${AR_NUM_GPUS:-4}"
export AR_MAX_RUNTIME_MIN="${AR_MAX_RUNTIME_MIN:-10}"

exec "${ROOT_DIR}/run_external_controller.sh" "$@"

