#!/usr/bin/env bash
set -euo pipefail

# Analyze training collapse signals from replay JSONL files.
#
# Usage (from repo root):
#   bash scripts/experiments/rui_meng/analyze_collapse_nq_sdpo_qwen25_3b_em_replay.sh
#
# Defaults to nq_sdpo-qwen2.5-3b-em-noclip-replay.
#
# Optional environment overrides:
#   REPLAY_DIR         — override replay directory
#   OUTPUT_DIR         — override output directory
#   STEP_MIN / STEP_MAX / STEP_STRIDE
#   SAMPLE_SIZE        — records subsampled per step for pairwise traj sim (default: 256)
#   PAIR_SAMPLE_SIZE   — max random pairs per step (default: 2000)
#   SEED               — RNG seed (default: 0)
#   PLOT_X_MIN / PLOT_X_MAX
#   NO_PLOT_XLIM=1     — autoscale x-axis

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SHARED_SCRIPT="${SCRIPT_DIR}/analyze_collapse.py"
cd "${PROJECT_ROOT}"

EXPERIMENT_NAME="nq_sdpo-qwen2.5-3b-em-noclip-replay"
REPLAY_DIR="${REPLAY_DIR:-res/train_replays/${EXPERIMENT_NAME}}"
OUTPUT_DIR="${OUTPUT_DIR:-res/train_replay_analysis/${EXPERIMENT_NAME}}"
STEP_STRIDE="${STEP_STRIDE:-1}"
SAMPLE_SIZE="${SAMPLE_SIZE:-256}"
PAIR_SAMPLE_SIZE="${PAIR_SAMPLE_SIZE:-2000}"
SEED="${SEED:-0}"
PLOT_X_MIN="${PLOT_X_MIN:-0}"
PLOT_X_MAX="${PLOT_X_MAX:-200}"

ARGS=(
  --replay-dir "${REPLAY_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --step-stride "${STEP_STRIDE}"
  --sample-size "${SAMPLE_SIZE}"
  --pair-sample-size "${PAIR_SAMPLE_SIZE}"
  --seed "${SEED}"
  --plot-x-min "${PLOT_X_MIN}"
  --plot-x-max "${PLOT_X_MAX}"
)

if [[ -n "${STEP_MIN:-}" ]]; then
  ARGS+=(--step-min "${STEP_MIN}")
fi
if [[ -n "${STEP_MAX:-}" ]]; then
  ARGS+=(--step-max "${STEP_MAX}")
fi
if [[ -n "${NO_PLOT_XLIM:-}" ]]; then
  ARGS+=(--no-plot-xlim)
fi

PYTHON_BIN="${PYTHON:-python3}"
"${PYTHON_BIN}" "${SHARED_SCRIPT}" "${ARGS[@]}"

echo "Collapse analysis complete. Outputs in ${OUTPUT_DIR}"
