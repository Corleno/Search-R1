#!/usr/bin/env bash
set -euo pipefail

# Replay-based eval: re-run multi-turn search from train/val replay JSONL.
# Switch starting prompt via EVAL_PROMPT_MODE=student|teacher.
#
# Student mode works on train_replay and val_replay.
# Teacher mode requires train_replay (records must include teacher_prompt).
#
# Usage:
#   REPLAY_PATH=res/train_replays/test_0619.jsonl \
#   EVAL_PROMPT_MODE=student \
#   BASE_MODEL=Qwen/Qwen2.5-3B \
#   bash scripts/experiments/rui_meng/eval_replay.sh
#
#   EVAL_PROMPT_MODE=teacher REPLAY_PATH=res/train_replays/test_0619.jsonl bash ...
#
#   REPLAY_PATH=val_replays/_grpo_v02_test_64.jsonl EVAL_PROMPT_MODE=student bash ...
#
# Optional:
#   EVAL_REPLAY_LIMIT=8          # subset for smoke tests
#   EVAL_REPLAY_DEDUPE_KEY=none  # evaluate every JSONL line (per-agent)
#   EVAL_REPLAY_REQUIRE_DISTILLATION_MASK=true  # train_replay only: self_distillation_mask > 0
#   EVAL_REPLAY_OUTPUT_PATH=...  # export re-run trajectories
#   EVAL_LOGGER=console          # skip wandb
#   RETRIEVER_URL, CUDA_VISIBLE_DEVICES, VAL_BATCH_SIZE

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

REPLAY_PATH="${REPLAY_PATH:-}"
if [[ -z "${REPLAY_PATH}" ]]; then
  echo "REPLAY_PATH is required (path to train_replay.jsonl or val_replay.jsonl)"
  exit 1
fi
if [[ ! -f "${REPLAY_PATH}" ]]; then
  echo "Replay file not found: ${REPLAY_PATH}"
  exit 1
fi

EVAL_PROMPT_MODE="${EVAL_PROMPT_MODE:-student}"
if [[ "${EVAL_PROMPT_MODE}" != "student" && "${EVAL_PROMPT_MODE}" != "teacher" ]]; then
  echo "EVAL_PROMPT_MODE must be student or teacher, got: ${EVAL_PROMPT_MODE}"
  exit 1
fi

BASE_MODEL="${BASE_MODEL:-PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-grpo-v0.2}"
WAND_PROJECT="${WAND_PROJECT:-Search-R1}"
REPLAY_BASENAME="$(basename "${REPLAY_PATH}" .jsonl)"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-eval-replay-${REPLAY_BASENAME}-${EVAL_PROMPT_MODE}}"
EVAL_REPLAY_DIR="${EVAL_REPLAY_DIR:-eval_replays/${EXPERIMENT_NAME}}"
EVAL_REPLAY_OUTPUT_PATH="${EVAL_REPLAY_OUTPUT_PATH:-${EVAL_REPLAY_DIR}/eval_replay_${EVAL_PROMPT_MODE}_$(date +%Y%m%d_%H%M%S).jsonl}"

VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-256}"
RETRIEVER_URL="${RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}"
EVAL_LOGGER="${EVAL_LOGGER:-wandb}"
LOGGER_ARG="['${EVAL_LOGGER//,/','}']"

echo "eval_replay:"
echo "  REPLAY_PATH=${REPLAY_PATH}"
echo "  EVAL_PROMPT_MODE=${EVAL_PROMPT_MODE}"
echo "  BASE_MODEL=${BASE_MODEL}"
echo "  EVAL_REPLAY_OUTPUT_PATH=${EVAL_REPLAY_OUTPUT_PATH}"

if [[ "${EVAL_LOGGER}" != "console" ]]; then
  if [ -z "${WANDB_API_KEY_SEARCH_R1:-}" ]; then
    echo "WANDB_API_KEY_SEARCH_R1 is not set (or set EVAL_LOGGER=console)"
    exit 1
  fi
  unset WANDB_BASE_URL
  unset WANDB_ENTITY
  export WANDB_API_KEY="${WANDB_API_KEY_SEARCH_R1}"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
export NCCL_NET_PLUGIN=dummy_name

cd "${PROJECT_ROOT}"

EXTRA_ARGS=()
if [[ -n "${EVAL_REPLAY_LIMIT:-}" ]]; then
  EXTRA_ARGS+=("trainer.eval_replay_limit=${EVAL_REPLAY_LIMIT}")
fi
if [[ -n "${EVAL_REPLAY_DEDUPE_KEY:-}" ]]; then
  EXTRA_ARGS+=("trainer.eval_replay_dedupe_key=${EVAL_REPLAY_DEDUPE_KEY}")
fi
if [[ -n "${EVAL_REPLAY_DEDUPE_STRATEGY:-}" ]]; then
  EXTRA_ARGS+=("trainer.eval_replay_dedupe_strategy=${EVAL_REPLAY_DEDUPE_STRATEGY}")
fi
if [[ "${EVAL_REPLAY_REQUIRE_DISTILLATION_MASK:-false}" == "true" ]]; then
  EXTRA_ARGS+=("trainer.eval_replay_require_distillation_mask=true")
fi

# Dummy parquet paths (not used when eval_from_replay + val_only).
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data/nq_hotpotqa_train}"
TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/train.parquet}"
VAL_FILE="${VAL_FILE:-${DATA_DIR}/test.parquet}"

LOG_NAME="${EXPERIMENT_NAME}-$(date +%Y%m%d_%H%M%S).log"

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=512 \
    data.val_batch_size="${VAL_BATCH_SIZE}" \
    data.max_prompt_length=4096 \
    data.max_response_length=500 \
    data.max_start_length=2048 \
    data.max_obs_length=500 \
    data.return_raw_chat=true \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path="${BASE_MODEL}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size=64 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=128 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n_agent=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=128 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.no_think_rl=false \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.actor.state_masking=true \
    trainer.logger="${LOGGER_ARG}" \
    +trainer.val_only=true \
    trainer.eval_from_replay=true \
    trainer.eval_replay_path="${REPLAY_PATH}" \
    trainer.eval_prompt_mode="${EVAL_PROMPT_MODE}" \
    trainer.save_eval_replay=true \
    trainer.eval_replay_output_path="${EVAL_REPLAY_OUTPUT_PATH}" \
    +trainer.val_before_train=true \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.project_name="${WAND_PROJECT}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.total_epochs=1 \
    trainer.total_training_steps=1 \
    trainer.default_local_dir="verl_checkpoints/${EXPERIMENT_NAME}" \
    do_search=true \
    max_turns=4 \
    retriever.url="${RETRIEVER_URL}" \
    retriever.topk=3 \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "${LOG_NAME}"
