#!/usr/bin/env bash
set -euo pipefail

# Plot training replay pass rates for mopd and fs_mopd variants on one figure.
#
# Usage (from repo root):
#   bash scripts/experiments/rui_meng/plot_reward_trend_compare_mopd_variates.sh
#
# Prerequisites:
#   Run analyze_reward_trend_nq_sdpo_qwen25_3b_em_replay.sh for mopd and fs_mopd first
#   so reward_summary.json exists.
#
# Optional environment overrides:
#   OUTPUT_DIR      — default: res/train_replay_analysis/compare/mopd-fs_mopd
#   EXPERIMENTS     — comma-separated LABEL=ANALYSIS_DIR entries
#   ROLLING_WINDOW  — rolling mean window when SHOW_ROLLING=1 (default: 10)
#   SHOW_ROLLING    — set to 1 to overlay rolling mean lines
#   PLOT_X_MIN      — x-axis minimum (default: 0)
#   PLOT_X_MAX      — x-axis maximum (default: 200)
#   STEP_STRIDE     — plot every Nth step (default: 5)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

OUTPUT_DIR="${OUTPUT_DIR:-res/train_replay_analysis/compare/mopd-fs_mopd}"
ROLLING_WINDOW="${ROLLING_WINDOW:-10}"
PLOT_X_MIN="${PLOT_X_MIN:-0}"
PLOT_X_MAX="${PLOT_X_MAX:-200}"
STEP_STRIDE="${STEP_STRIDE:-5}"
EXPERIMENTS="${EXPERIMENTS:-mopd=res/train_replay_analysis/sdpo_v02_noclip_replay_ref_qwen2.5-3b-qwen2.5-7b-instruct,fs_mopd=res/train_replay_analysis/sdpo_v02_noclip_replay_ref_qwen2.5-3b-qwen2.5-7b-instruct-empty-reprompt}"

ARGS=(
  --output-dir "${OUTPUT_DIR}"
  --experiments "${EXPERIMENTS}"
  --rolling-window "${ROLLING_WINDOW}"
  --plot-x-min "${PLOT_X_MIN}"
  --plot-x-max "${PLOT_X_MAX}"
  --step-stride "${STEP_STRIDE}"
)

if [[ -n "${SHOW_ROLLING:-}" ]]; then
  ARGS+=(--show-rolling)
fi
if [[ -n "${NO_PLOT_XLIM:-}" ]]; then
  ARGS+=(--no-plot-xlim)
fi

python3 "${SCRIPT_DIR}/plot_reward_trend_compare.py" "${ARGS[@]}"

echo "MOPD-variant comparison plot complete. Outputs in ${OUTPUT_DIR}"
