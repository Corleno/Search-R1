#!/usr/bin/env bash
set -euo pipefail

# Analyze reward trend and distribution from training replay JSONL files.
#
# Usage (from repo root):
#   bash scripts/experiments/rui_meng/eval_replay_sdpo_v02_noclip/analyze_reward_trend.sh
#
# Optional environment overrides:
#   REPLAY_DIR      — default: res/train_replays/exp_sdpo_searchr1_0620
#   OUTPUT_DIR      — default: res/train_replay_analysis/exp_sdpo_searchr1_0620
#   STEP_MIN        — minimum step (inclusive)
#   STEP_MAX        — maximum step (inclusive)
#   STEP_STRIDE     — analyze every Nth step (default: 1)
#   ROLLING_WINDOW  — rolling mean window for trend plot (default: 10)
#   SNAPSHOT_STEPS  — comma-separated steps for snapshot bars (default: 1,50,100,150,200)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${PROJECT_ROOT}"

REPLAY_DIR="${REPLAY_DIR:-res/train_replays/exp_sdpo_searchr1_0620}"
OUTPUT_DIR="${OUTPUT_DIR:-res/train_replay_analysis/exp_sdpo_searchr1_0620}"
STEP_STRIDE="${STEP_STRIDE:-1}"
ROLLING_WINDOW="${ROLLING_WINDOW:-10}"
SNAPSHOT_STEPS="${SNAPSHOT_STEPS:-1,50,100,150,200}"

ARGS=(
  --replay-dir "${REPLAY_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --step-stride "${STEP_STRIDE}"
  --rolling-window "${ROLLING_WINDOW}"
  --snapshot-steps "${SNAPSHOT_STEPS}"
)

if [[ -n "${STEP_MIN:-}" ]]; then
  ARGS+=(--step-min "${STEP_MIN}")
fi
if [[ -n "${STEP_MAX:-}" ]]; then
  ARGS+=(--step-max "${STEP_MAX}")
fi

python3 "${SCRIPT_DIR}/analyze_reward_trend.py" "${ARGS[@]}"

echo "Reward analysis complete. Outputs in ${OUTPUT_DIR}"
