#!/usr/bin/env bash
set -euo pipefail

# Benchmark pipeline runner for HEIST algorithms across seeds and curriculum stages.

PYTHONPATH=src/py uv run python src/py/curriculum.py \
    --algos roma,rode \
    --concurrent-algos 2 \
    --seeds 0-9 \
    --rust \
    --eval \
    --eval-episodes 1000