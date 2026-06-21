#!/usr/bin/env bash
set -euo pipefail

# Usage: bash eval_step_n.sh <step>
# Example: bash eval_step_n.sh 0011
step=${1:-0011}
step_num=$((10#${step}))

# if step is less than 10, use the Qwen/Qwen2.5-3B for BASE_MODEL, otherwise use the checkpoint of the maximum step of the checkpoint (devidable by 10) before the current step
if [ "${step_num}" -le 10 ]; then
    BASE_MODEL=Qwen/Qwen2.5-3B
else
    BASE_MODEL=verl_checkpoints/exp_sdpo_searchr1_0620/actor/global_step_$((${step_num} - ${step_num} % 10))
fi

# Eval baseline (pre-update) rollouts: training saves replays from global_steps=1 as step_0001.jsonl.
REPLAY_PATH=res/train_replays/exp_sdpo_searchr1_0620/step_${step}.jsonl \
EVAL_PROMPT_MODE=student \
BASE_MODEL=${BASE_MODEL} \
EVAL_REPLAY_REQUIRE_DISTILLATION_MASK=true \
EVAL_REPLAY_OUTPUT_PATH=res/eval_replays/exp_sdpo_searchr1_0620/step_${step}_$EVAL_PROMPT_MODE \
bash scripts/experiments/rui_meng/eval_replay.sh

# Eval for teacher mode
REPLAY_PATH=res/train_replays/exp_sdpo_searchr1_0620/step_${step}.jsonl \
EVAL_PROMPT_MODE=teacher \
BASE_MODEL=${BASE_MODEL} \
EVAL_REPLAY_REQUIRE_DISTILLATION_MASK=true \
EVAL_REPLAY_OUTPUT_PATH=res/eval_replays/exp_sdpo_searchr1_0620/step_${step}_$EVAL_PROMPT_MODE \
bash scripts/experiments/rui_meng/eval_replay.sh

echo "Completed evaluation for step ${step}"