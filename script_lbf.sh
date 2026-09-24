#!/usr/bin/env bash
set -euo pipefail

# Runs the Level-Based Foraging (LBF) benchmark for THIEF and baseline models.
# Output directories and logs are routed to results/lbf.

SEEDS="${LBF_SEEDS:-0-4}"
TIMESTEPS="${LBF_TIMESTEPS:-400000}"
JOBS="${LBF_JOBS:-1}"
DEVICE="${LBF_DEVICE:-}"
OUT_ROOT="${LBF_OUT_ROOT:-results/lbf}"

DEVICE_ARG=""
if [ -n "$DEVICE" ]; then
  DEVICE_ARG="--device $DEVICE"
fi

echo "Running LBF benchmark (seeds: ${SEEDS}, timesteps: ${TIMESTEPS}, jobs: ${JOBS}, out: ${OUT_ROOT})"

# shellcheck disable=SC2086
PYTHONPATH=src/py uv run python src/py/train_lbf.py \
    --algos thief,roma,rode,mappo,hmappo,coma \
    --envs all \
    --seeds "${SEEDS}" \
    --timesteps "${TIMESTEPS}" \
    --eval-episodes 100 \
    --jobs "${JOBS}" \
    --out-root "${OUT_ROOT}" \
    ${DEVICE_ARG} \
    "$@"
