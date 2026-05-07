#!/usr/bin/env bash
set -euo pipefail

# Eval-only (GRPO stack: main_ppo + adv_estimator=grpo) loading weights from Hugging Face:
#   https://huggingface.co/PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-grpo-v0.2
#
# Data: always your locally processed train/test parquet (same official-repo data pipeline you already ran); this is
# treated as equivalent to pulling the same split from the Hub. Defaults match fan/train_grpo_mix_v02.sh
# (data/mix_train_nq_hotpotqa_test7/{train,test}.parquet). val_only still requires train.parquet on disk for the dataloader.
# Override: DATA_DIR=... or TRAIN_FILE=... VAL_FILE=...
#
# Rollout / FSDP knobs match fan/train_grpo_mix_v02.sh (same as on-the-fly _validate() during training).
# If you still OOM while training does not, check nvidia-smi for another large GPU job; retriever + eval share the same 8 cards.
#
# Requires: retrieval server (default http://127.0.0.1:8001/retrieve), searchr1 conda env, 8 GPUs by default.
#
# Usage:
#   bash fan/evaluate_hf_official_nq_hotpotqa_grpo_v02.sh
#   DATA_DIR=/path/to/your/processed_dir bash fan/evaluate_hf_official_nq_hotpotqa_grpo_v02.sh
#   EVAL_LOGGER=console  # only if you want to skip wandb

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data/mix_train_nq_hotpotqa_test7}"
TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/train.parquet}"
VAL_FILE="${VAL_FILE:-${DATA_DIR}/test.parquet}"

BASE_MODEL="${BASE_MODEL:-PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-grpo-v0.2}"

echo "evaluate_hf_official: local data (HF-equivalent pipeline)"
echo "  TRAIN_FILE=${TRAIN_FILE}"
echo "  VAL_FILE=${VAL_FILE}"
echo "  BASE_MODEL=${BASE_MODEL}"

if [[ ! -f "${TRAIN_FILE}" ]]; then
  echo "Missing train parquet: ${TRAIN_FILE}"
  echo "Set DATA_DIR (or TRAIN_FILE/VAL_FILE) to the directory where your processed train.parquet lives."
  exit 1
fi
if [[ ! -f "${VAL_FILE}" ]]; then
  echo "Missing val parquet: ${VAL_FILE}"
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

WAND_PROJECT="${WAND_PROJECT:-Search-R1}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-eval-official-ckpt-qwen2.5-3b-it-grpo-v0.2-on-mix_train_nq_hotpotqa_test7}"
RETRIEVER_URL="${RETRIEVER_URL:-http://127.0.0.1:8001/retrieve}"

VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-256}"
VAL_DATA_NUM="${VAL_DATA_NUM:-}"

EVAL_LOGGER="${EVAL_LOGGER:-wandb}"
LOGGER_ARG="['${EVAL_LOGGER//,/','}']"

export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"

cd "${PROJECT_ROOT}"

VAL_NUM_ARG=()
if [[ -n "${VAL_DATA_NUM}" ]]; then
  VAL_NUM_ARG=(data.val_data_num="${VAL_DATA_NUM}")
else
  VAL_NUM_ARG=(data.val_data_num=null)
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
    actor_rollout_ref.rollout.n_agent=5 \
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
