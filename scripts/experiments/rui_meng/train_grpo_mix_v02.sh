#!/usr/bin/env bash (WIP)
set -euo pipefail

# Personal GRPO training script based on scripts/nq_hotpotqa/v0.2/train_grpo.sh.
# Default data dir: repo data/nq_search (train.parquet, test.parquet). Override DATA_DIR
# for nq_hotpotqa_train or other parquet trees (see scripts/nq_hotpotqa/README.md).
#
# Usage:
#   From repo root (recommended):
#     ./scripts/experiments/rui_meng/train_grpo_mix_v02.sh
#   Or from anywhere:
#     bash /path/to/Search-R1/scripts/experiments/rui_meng/train_grpo_mix_v02.sh
#
# Prerequisites:
#   - Working directory for training is the repo root (script cd's there).
#   - Parquet files under data/nq_search/ (default), or set DATA_DIR. To build NQ parquets
#     see scripts/download_nq_data.sh (writes under SAVE_PATH/nq_search by default).
#   - Retrieval HTTP server on retriever.url (default matches v0.2 train_grpo.sh port 8000;
#     use RETRIEVER_URL=http://127.0.0.1:8001/retrieve if you run the server on 8001).
#   - WandB configured if you use the default trainer.logger (wandb).
#
# Optional environment overrides:
#   CUDA_VISIBLE_DEVICES   GPU list (default 0-7)
#   DATA_DIR             Directory with train.parquet / test.parquet
#   WAND_PROJECT         WandB project (default Search-R1)
#   BASE_MODEL           HF model id (default Qwen/Qwen2.5-3B)
#   EXPERIMENT_NAME      Run name + checkpoint subdir under verl_checkpoints/
#   RETRIEVER_URL        Full retrieve endpoint (default http://127.0.0.1:8000/retrieve)
#   VLLM_ATTENTION_BACKEND  (default XFORMERS; matches v0.2 train_grpo.sh note)
#   TMPDIR                 Default /tmp avoids cross-fs temp vs ~/.cache pip/Ray edge cases on /mnt-hosted TMPDIR.
#   PYTORCH_CUDA_ALLOC_CONF  Default adds expandable_segments; can reduce fragmentation OOM during FSDP/ref init.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
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
EXPERIMENT_NAME="${EXPERIMENT_NAME:-nq_search-search-r1-grpo-qwen2.5-3b-em}"
RETRIEVER_URL="${RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}"

# Keep v0.2 recommendation for Qwen with vLLM.
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"

cd "${PROJECT_ROOT}"

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=512 \
    data.val_batch_size=256 \
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
    trainer.logger=['wandb'] \
    +trainer.val_only=false \
    +trainer.val_before_train=true \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    trainer.project_name="${WAND_PROJECT}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.total_epochs=15 \
    trainer.total_training_steps=1005 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir="verl_checkpoints/${EXPERIMENT_NAME}" \
    max_turns=4 \
    retriever.url="${RETRIEVER_URL}" \
    retriever.topk=3 \
    2>&1 | tee "${EXPERIMENT_NAME}.log"
