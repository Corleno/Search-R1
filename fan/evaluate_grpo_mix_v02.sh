#!/usr/bin/env bash
set -euo pipefail

# Standalone eval-only run aligned with fan/train_grpo_mix_v02.sh (v0.2 GRPO: main_ppo + adv_estimator=grpo).
# Differs from scripts/nq_hotpotqa/evaluate.sh, which uses algorithm.adv_estimator=gae + a critic.
#
# Requires: retrieval server up (default http://127.0.0.1:8001/retrieve), same data layout as training.
#
# Usage:
#   BASE_MODEL=Qwen/Qwen2.5-3B bash fan/evaluate_grpo_mix_v02.sh
#   BASE_MODEL=verl_checkpoints/<exp>/actor/global_step_100 bash fan/evaluate_grpo_mix_v02.sh
#
# Optional env:
#   VAL_DATA_NUM=256     # subsample val set (must be >= VAL_BATCH_SIZE for drop_last to keep >=1 batch)
#   VAL_BATCH_SIZE=256   # lower if you use a small VAL_DATA_NUM
#   TRAIN_FILE=...       # still loaded (trainer asserts train dataloader non-empty); default = training train.parquet
#   EVAL_LOGGER='console' or 'wandb' or 'console,wandb'

PROJECT_ROOT="/nfshome/fayang/workspace/Search-R1"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data/mix_train_nq_hotpotqa_test7}"
TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/train.parquet}"
VAL_FILE="${VAL_FILE:-${DATA_DIR}/test.parquet}"

if [[ ! -f "${TRAIN_FILE}" ]]; then
  echo "Missing train parquet (needed for dataloader init): ${TRAIN_FILE}"
  exit 1
fi
if [[ ! -f "${VAL_FILE}" ]]; then
  echo "Missing val parquet: ${VAL_FILE}"
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

WAND_PROJECT="${WAND_PROJECT:-Search-R1}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-mix_train_nq_hotpotqa_test7-search-r1-grpo-qwen2.5-3b-em-eval}"
RETRIEVER_URL="${RETRIEVER_URL:-http://127.0.0.1:8001/retrieve}"

VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-256}"
VAL_DATA_NUM="${VAL_DATA_NUM:-}" # empty => full val set

# Hydra list for logger; default console only (no W&B login required).
EVAL_LOGGER="${EVAL_LOGGER:-console}"
# Pass as a single-quoted Hydra list: [a,b] — avoid spaces inside brackets for this shell.
LOGGER_ARG="['${EVAL_LOGGER//,/','}']"

export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"

cd "${PROJECT_ROOT}"

VAL_NUM_ARG=()
if [[ -n "${VAL_DATA_NUM}" ]]; then
  VAL_NUM_ARG=(data.val_data_num="${VAL_DATA_NUM}")
fi

LOG_NAME="${EXPERIMENT_NAME}-$(date +%Y%m%d_%H%M%S).log"

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_data_num=null \
    "${VAL_NUM_ARG[@]}" \
    data.train_batch_size=512 \
    data.val_batch_size="${VAL_BATCH_SIZE}" \
    data.max_prompt_length=4096 \
    data.max_response_length=500 \
    data.max_start_length=2048 \
    data.max_obs_length=500 \
    data.shuffle_train_dataloader=True \
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
    actor_rollout_ref.ref.log_prob_micro_batch_size=128 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.no_think_rl=false \
    actor_rollout_ref.rollout.n_agent=1 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.actor.state_masking=true \
    trainer.logger="${LOGGER_ARG}" \
    +trainer.val_only=true \
    +trainer.val_before_train=true \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.project_name="${WAND_PROJECT}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.total_epochs=15 \
    trainer.total_training_steps=1005 \
    trainer.default_local_dir="verl_checkpoints/${EXPERIMENT_NAME}" \
    max_turns=4 \
    retriever.url="${RETRIEVER_URL}" \
    retriever.topk=3 \
    2>&1 | tee "${LOG_NAME}"
