#!/usr/bin/env bash
set -euo pipefail

# Wrapper — defaults VARIANT to ppo (ppoclip). See parent script for all options.
#
# Usage (from repo root):
#   bash scripts/experiments/rui_meng/eval_replay_nq_sdpo_qwen25_3b_em_ppoclip_replay/analyze_reward_trend.sh [VARIANT]
#
# VARIANT: ppo (default) | base | ref | ema

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_SCRIPT="${SCRIPT_DIR}/../analyze_reward_trend_nq_sdpo_qwen25_3b_em_replay.sh"

if [[ $# -eq 0 ]]; then
  exec bash "${SHARED_SCRIPT}" ppo
fi
exec bash "${SHARED_SCRIPT}" "$@"
