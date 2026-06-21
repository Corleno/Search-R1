#!/usr/bin/env bash
set -euo pipefail

# Plot student and teacher eval test scores from replay eval metrics.
#
# Usage (from repo root):
#   bash scripts/experiments/rui_meng/eval_replay_sdpo_v02_noclip/plot_eval_test_scores.sh
#
# Optional environment overrides:
#   EVAL_DIR   — default: res/eval_replays/exp_sdpo_searchr1_0620
#   OUTPUT_DIR — default: res/eval_replay_analysis/exp_sdpo_searchr1_0620
#   STEPS      — comma-separated steps (default: 1,11,...,191)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${PROJECT_ROOT}"

EVAL_DIR="${EVAL_DIR:-res/eval_replays/exp_sdpo_searchr1_0620}"
OUTPUT_DIR="${OUTPUT_DIR:-res/eval_replay_analysis/exp_sdpo_searchr1_0620}"
STEPS="${STEPS:-1,11,21,31,41,51,61,71,81,91,101,111,121,131,141,151,161,171,181,191}"

python3 "${SCRIPT_DIR}/plot_eval_test_scores.py" \
  --eval-dir "${EVAL_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --steps "${STEPS}"

echo "Eval test score plot complete. Outputs in ${OUTPUT_DIR}"
