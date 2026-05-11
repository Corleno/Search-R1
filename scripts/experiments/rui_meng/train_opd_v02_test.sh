#!/usr/bin/env bash
set -euo pipefail

# Personal OPD (on-policy distillation) script, parallel to train_grpo_v02.sh but with OPD teacher RM
# and token_reward_direct-style advantages (see README_OPD.md). Search enabled like train_grpo_v02.sh.
# Default data dir matches train_grpo_v02.sh; override DATA_DIR for other parquet trees.
#
# Usage:
#   From repo root (recommended):
#     TEACHER_MODEL=Qwen/Qwen2.5-7B-Instruct ./scripts/experiments/rui_meng/train_opd_v02_test.sh
#   Or from anywhere:
#     bash /path/to/Search-R1/scripts/experiments/rui_meng/train_opd_v02_test.sh
#
# Prerequisites:
#   - Cwd resolves to repo root (script cd's there).
#   - Parquet train/test under DATA_DIR (default /data/nq_hotpotqa_train).
#   - Retrieval HTTP server on retriever.url (default http://127.0.0.1:8000/retrieve; override RETRIEVER_URL).
#   - TEACHER_MODEL: causal LM for OPD (HF id or checkpoint); defaults to BASE_MODEL if unset (dev only).
#   - Console-only logger by default (no WandB required for this smoke script).
#
# Optional environment overrides:
#   CUDA_VISIBLE_DEVICES — limits which GPUs the process sees; must match trainer.n_gpus_per_node below.
#   N_GPUS_PER_NODE — overrides auto count from CUDA_VISIBLE_DEVICES (comma-separated IDs).
#   DATA_DIR, WAND_PROJECT, BASE_MODEL (student), TEACHER_MODEL, EXPERIMENT_NAME
#   ADV_ESTIMATOR (token_reward_direct | token_reward_direct_plus_grpo), GRPO_OUTCOME_WEIGHT
#   LOG_PROB_TOP_K, TOP_K_STRATEGY, REWARD_WEIGHT_MODE, TEACHER_TEMPERATURE (rollout teacher temp)
#   RETRIEVER_URL — full retrieve endpoint (default http://127.0.0.1:8000/retrieve)
#   TMPDIR, PYTORCH_CUDA_ALLOC_CONF, VLLM_ATTENTION_BACKEND

# check if the wandb api key is set
if [ -z "$WANDB_API_KEY_SEARCH_R1" ]; then
    echo "WANDB_API_KEY_SEARCH_R1 is not set"
    exit 1
fi

# W&B: default cloud (no custom host). Unset avoids the literal string "None", which breaks the client.
unset WANDB_BASE_URL
unset WANDB_ENTITY
export WANDB_API_KEY="${WANDB_API_KEY_SEARCH_R1}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
_CVD="${CUDA_VISIBLE_DEVICES// /}"
IFS=',' read -ra _CVD_IDS <<< "${_CVD}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-${#_CVD_IDS[@]}}"
export TMPDIR="${TMPDIR:-/tmp}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUMBER_THREADS:-${OMP_NUM_THREADS:-1}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DATA_DIR="${DATA_DIR:-/data/nq_hotpotqa_train}"
TRAIN_FILE="${DATA_DIR}/train.parquet"
VAL_FILE="${DATA_DIR}/test.parquet"

if [[ ! -f "${TRAIN_FILE}" ]]; then
  echo "Missing train parquet: ${TRAIN_FILE}"
  exit 1
fi
if [[ ! -f "${VAL_FILE}" ]]; then
  echo "Missing val parquet: ${VAL_FILE}"
  exit 1
fi

WAND_PROJECT="${WAND_PROJECT:-Search-R1}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B}"
TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-nq_opd-qwen2.5-3b-em}"

ADV_ESTIMATOR="${ADV_ESTIMATOR:-token_reward_direct}"
GRPO_OUTCOME_WEIGHT="${GRPO_OUTCOME_WEIGHT:-1.0}"
LOG_PROB_TOP_K="${LOG_PROB_TOP_K:-16}"
TOP_K_STRATEGY="${TOP_K_STRATEGY:-only_stu}"
REWARD_WEIGHT_MODE="${REWARD_WEIGHT_MODE:-student_p}"
TEACHER_TEMPERATURE="${TEACHER_TEMPERATURE:-1.0}"
RETRIEVER_URL="${RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}"

export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"

export NCCL_NET_PLUGIN=dummy_name

cd "${PROJECT_ROOT}"

export PYTHONUNBUFFERED=1
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator="${ADV_ESTIMATOR}" \
    algorithm.grpo_outcome_weight="${GRPO_OUTCOME_WEIGHT}" \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=8 \
    data.val_batch_size=8 \
    data.max_prompt_length=4096 \
    data.max_response_length=500 \
    data.max_start_length=2048 \
    data.max_obs_length=500 \
    data.shuffle_train_dataloader=True \
    actor_rollout_ref.model.path="${BASE_MODEL}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.log_prob_top_k="${LOG_PROB_TOP_K}" \
    actor_rollout_ref.rollout.top_k_strategy="${TOP_K_STRATEGY}" \
    actor_rollout_ref.rollout.reward_weight_mode="${REWARD_WEIGHT_MODE}" \
    actor_rollout_ref.rollout.teacher_temperature="${TEACHER_TEMPERATURE}" \
    algorithm.no_think_rl=false \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.actor.state_masking=true \
    reward_model.enable=true \
    reward_model.use_opd_teacher=true \
    reward_model.model.path="${TEACHER_MODEL}" \
    reward_model.model.input_tokenizer=null \
    reward_model.model.use_remove_padding=false \
    +reward_model.model.dtype=bfloat16 \
    reward_model.micro_batch_size=8 \
    reward_model.use_dynamic_bsz=false \
    trainer.logger=['console'] \
    +trainer.val_only=false \
    +trainer.val_before_train=false \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    trainer.project_name="${WAND_PROJECT}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.total_epochs=15 \
    trainer.total_training_steps=1005 \
    trainer.default_local_dir="verl_checkpoints/${EXPERIMENT_NAME}" \
    do_search=true \
    max_turns=4 \
    retriever.url="${RETRIEVER_URL}" \
    retriever.topk=3 \
    2>&1 | tee "${EXPERIMENT_NAME}.log"
