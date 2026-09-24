#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# LBF Benchmark Suite: THIEF vs ROMA, RODE, MAPPO, H-MAPPO, COMA
#
# Runs the canonical Level-Based Foraging benchmark for all algorithms that
# have a meaningful LBF adaptation. ECOOP, COOP, and MARC are deliberately
# excluded — they rely on HEIST-specific role structure and cannot be cleanly
# adapted to the homogeneous-agent LBF setting.
#
# Paper references:
#   THIEF:  this paper
#   ROMA:   Wang et al., ICML 2020 "ROMA: Multi-Agent RL with Emergent Roles"
#   RODE:   Wang et al., ICLR 2021 "RODE: Learning Roles to Decompose MARL"
#   MAPPO:  Yu et al., NeurIPS 2022 "The Surprising Effectiveness of PPO in CMARL"
#   H-MAPPO: Vezhnevets et al., ICML 2017 "FeUdal Networks for Hierarchical RL"
#   COMA:   Foerster et al., AAAI 2018 "Counterfactual Multi-Agent Policy Gradients"
#   LBF env: Papoudakis et al., NeurIPS D&B 2021
#
# Outputs:
#   results/lbf/<env>/<algo>/seed_<s>/results.json
#   results/lbf/lbf_benchmark_summary.{json,md}
#   paper/tables/lbf_benchmark.tex
#   paper/tables/lbf_ablation.tex
#
# Environment variable overrides:
#   LBF_SEEDS      (default: 0-4)
#   LBF_TIMESTEPS  (default: 400000)
#   LBF_JOBS       (default: 1 — set >1 for parallel CPU seeds)
#   LBF_DEVICE     (default: auto)
#   LBF_OUT_ROOT   (default: results/lbf)
# =============================================================================

SEEDS="${LBF_SEEDS:-0-4}"
TIMESTEPS="${LBF_TIMESTEPS:-400000}"
JOBS="${LBF_JOBS:-1}"
DEVICE="${LBF_DEVICE:-}"
OUT_ROOT="${LBF_OUT_ROOT:-results/lbf}"

DEVICE_ARG=""
if [ -n "$DEVICE" ]; then
  DEVICE_ARG="--device $DEVICE"
fi

echo "=================================================================="
echo " LBF Benchmark: thief, roma, rode, mappo, hmappo, coma"
echo " Seeds: ${SEEDS} | Timesteps: ${TIMESTEPS} | Jobs: ${JOBS}"
echo " Output: ${OUT_ROOT}"
echo "=================================================================="

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
