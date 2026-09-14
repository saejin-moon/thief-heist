#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Full MARL Benchmark Suite Pipeline:
# - All 7 algorithms in priority order (thief, ecoop, mappo, hmappo, coop, marc, coma)
# - Dynamic concurrency: 2 algorithms simultaneously with automatic queue popping
# - 10 seeds (0-9 inclusive) across 5 curriculum stages
# - Native Rust environment acceleration
# - Full parallel evaluation suite (1,000 episodes per seed/stage)
# - THIEF component micro-ablation suite (7 variants)
# - LLM-ready master benchmark summary report (Markdown & JSON)
# =============================================================================

PYTHONPATH=src/py uv run python src/py/curriculum.py \
    --algos all \
    --concurrent-algos 2 \
    --seeds 0-9 \
    --rust \
    --eval \
    --eval-episodes 1000 \
    --ablations "$@"