#!/usr/bin/env bash
set -euo pipefail

# SDPO v02 variant: frozen ref teacher + no distillation IS clipping + training replay JSONL export.
# Based on train_sdpo_v02_noclip_replay.sh; see README_SDPO.md.
#
# Usage:
#   From repo root (recommended):
#     BASE_MODEL=Qwen/Qwen2.5-3B REF_MODEL=Qwen/Qwen2.5-7B-Instruct \
#     EXPERIMENT_NAME=sdpo_v02_noclip_replay_ref_qwen2.5-3b-qwen2.5-7b-instruct \
#     ./scripts/experiments/rui_meng/train_sdpo_v02_noclip_replay_ref.sh
#   Or from anywhere:
#     bash /path/to/Search-R1/scripts/experiments/rui_meng/train_sdpo_v02_noclip_replay_ref.sh
# Prerequisites:
#   - conda env searchr1 (see scripts/install_envs.sh)
#   - Cwd resolves to repo root (script cd's there).
#   - Parquet train/test under DATA_DIR (default ./data/nq_hotpotqa_train).
#   - Retrieval HTTP server on retriever.url (default http://127.0.0.1:8000/retrieve; override RETRIEVER_URL).
#   - WandB if using default trainer.logger.
#
# Optional environment overrides:
#   CUDA_VISIBLE_DEVICES — limits which GPUs the process sees; must match trainer.n_gpus_per_node below.
#   N_GPUS_PER_NODE — overrides auto count from CUDA_VISIBLE_DEVICES (comma-separated IDs).
#   DATA_DIR, WAND_PROJECT, BASE_MODEL (student), REF_MODEL (frozen teacher), EXPERIMENT_NAME
#   TEACHER_REG — self-distillation teacher (default ref); use ref for this script
#   TOTAL_TRAINING_STEPS — max training steps (default 200)
#   RETRIEVER_URL — full retrieve endpoint (default http://127.0.0.1:8000/retrieve)
#   TRAIN_REPLAY_DIR — default: res/train_replays/${EXPERIMENT_NAME} (writes step_NNNN.jsonl per step)
#   TMPDIR, PYTORCH_CUDA_ALLOC_CONF, VLLM_ATTENTION_BACKEND

_CONDA_BASE="${CONDA_BASE:-}"
if [[ -z "${_CONDA_BASE}" ]] && command -v conda &>/dev/null; then
  _CONDA_BASE="$(conda info --base)"
elif [[ -z "${_CONDA_BASE}" && -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  _CONDA_BASE="${HOME}/miniconda3"
fi
if [[ -n "${_CONDA_BASE}" && -f "${_CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "${_CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate searchr1
else
  echo "Could not activate conda env searchr1; set CONDA_BASE or ensure conda is on PATH" >&2
  exit 1
fi

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
DATA_DIR="${DATA_DIR:-./data/nq_hotpotqa_train}"
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
REF_MODEL="${REF_MODEL:-${BASE_MODEL}}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-nq_sdpo-${BASE_MODEL}-noclip-replay-ref-${REF_MODEL}}"
TEACHER_REG="${TEACHER_REG:-ref}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-200}"
RETRIEVER_URL="${RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}"
TRAIN_REPLAY_DIR="${TRAIN_REPLAY_DIR:-res/train_replays/${EXPERIMENT_NAME}}"

echo "train_sdpo_v02_noclip_replay_ref:"
echo "  TRAIN_FILE=${TRAIN_FILE}"
echo "  VAL_FILE=${VAL_FILE}"
echo "  BASE_MODEL=${BASE_MODEL}"
echo "  REF_MODEL=${REF_MODEL}"
echo "  EXPERIMENT_NAME=${EXPERIMENT_NAME}"
echo "  TEACHER_REG=${TEACHER_REG}"
echo "  TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS}"
echo "  TRAIN_REPLAY_DIR=${TRAIN_REPLAY_DIR}"
echo "  ppo_clip=false (no PPO-clipped SDPO distillation clipping)"

export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"

export NCCL_NET_PLUGIN=dummy_name

EXTRA_ARGS=()
if [[ "${TEACHER_REG}" == "ema" || "${TEACHER_REG}" == "ref" ]]; then
  EXTRA_ARGS+=(
    "actor_rollout_ref.ref.fsdp_config.param_offload=true"
    "actor_rollout_ref.actor.self_distillation.teacher_regularization=${TEACHER_REG}"
  )
fi
if [[ "${TEACHER_REG}" == "ref" ]]; then
  EXTRA_ARGS+=(
    "actor_rollout_ref.ref.model.path=${REF_MODEL}"
  )
fi

cd "${PROJECT_ROOT}"

export PYTHONUNBUFFERED=1
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    --config-name sdpo \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_data_num=null \
    data.val_data_num=null \
    data.return_raw_chat=true \
    data.train_batch_size=512 \
    data.val_batch_size=256 \
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
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size=64 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.actor.self_distillation.ppo_clip=false \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=128 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.n_agent=5 \
    algorithm.no_think_rl=false \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.actor.state_masking=true \
    actor_rollout_ref.actor.calculate_entropy=true \
    reward_model.enable=false \
    trainer.logger=['wandb'] \
    +trainer.val_only=false \
    trainer.val_before_train=false \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=100 \
    trainer.save_train_replay=true \
    trainer.train_replay_path="${TRAIN_REPLAY_DIR}" \
    trainer.project_name="${WAND_PROJECT}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.total_epochs=15 \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
    trainer.default_local_dir="verl_checkpoints/${EXPERIMENT_NAME}" \
    do_search=true \
    max_turns=4 \
    retriever.url="${RETRIEVER_URL}" \
    retriever.topk=3 \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "${EXPERIMENT_NAME}.log"
