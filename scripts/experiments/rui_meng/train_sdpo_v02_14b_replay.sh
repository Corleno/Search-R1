#!/usr/bin/env bash
# End-to-end SDPO launch for Qwen/Qwen2.5-14B:
#   1) download model if missing
#   2) start wiki-e5 retriever in background if not already up
#   3) run pure SDPO (v02 hyperparameters) with training replay JSONL export
#   4) record experiment metadata under exps/<EXPERIMENT_NAME>/
#
# Usage (from repo root):
#   bash scripts/experiments/rui_meng/train_sdpo_v02_14b_replay.sh
#
# Optional env overrides:
#   CUDA_VISIBLE_DEVICES, N_GPUS_PER_NODE, DATA_DIR, WAND_PROJECT,
#   EXPERIMENT_NAME, TEACHER_REG, RETRIEVER_URL, RETRIEVAL_PORT,
#   MODEL_DIR, HF_MODEL_ID, RETRIEVER_WAIT_SECS, WANDB_API_KEY_SEARCH_R1,
#   TRAIN_REPLAY_DIR — default: res/train_replays/${EXPERIMENT_NAME} (writes step_NNNN.jsonl per step)
set -euo pipefail

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
# ~/.bashrc references PS1 and other interactive-only vars; with `set -u` that
# aborts the script before any logs. Temporarily allow unbound vars while sourcing.
set +u
# shellcheck disable=SC1090
source "${HOME}/.bashrc" 2>/dev/null || true
if ! command -v conda >/dev/null 2>&1; then
  if [[ -f "/root/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "/root/miniconda3/etc/profile.d/conda.sh"
  elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  else
    set -u
    echo "ERROR: conda not found. Install miniconda or fix PATH."
    exit 1
  fi
fi
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
_CVD="${CUDA_VISIBLE_DEVICES// /}"
IFS=',' read -ra _CVD_IDS <<< "${_CVD}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-${#_CVD_IDS[@]}}"
export TMPDIR="${TMPDIR:-/tmp}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUMBER_THREADS:-${OMP_NUM_THREADS:-1}}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
export PYTHONUNBUFFERED=1

# A3+/TCPXO hosts put NCCL 2.28 + net/tuner shims on LD_LIBRARY_PATH. PyTorch
# 2.4 bundles NCCL 2.20.5; loading the host net plugin into that NCCL SIGSEGVs
# during the first real collective (FSDP wrap in ref/actor init_model).
# Prefer torch's libnccl, disable host net/shim plugins, force single-node P2P.
_TORCH_NCCL_LIB="${CONDA_PREFIX:-/root/miniconda3/envs/searchr1}/lib/python3.9/site-packages/nvidia/nccl/lib"
if [[ ! -d "${_TORCH_NCCL_LIB}" ]]; then
  _TORCH_NCCL_LIB="/root/miniconda3/envs/searchr1/lib/python3.9/site-packages/nvidia/nccl/lib"
fi
if [[ -d "${_TORCH_NCCL_LIB}" ]]; then
  export LD_LIBRARY_PATH="${_TORCH_NCCL_LIB}:${LD_LIBRARY_PATH:-/usr/local/nvidia/lib64}"
fi
export NCCL_NET_PLUGIN="${NCCL_NET_PLUGIN:-none}"
export NCCL_SHIMNET_SHIM_LAYERS="${NCCL_SHIMNET_SHIM_LAYERS:-UNUSED}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
unset NCCL_TUNER_PLUGIN || true
unset NCCL_TUNER_CONFIG_PATH || true
# Host image often sets NCCL_PROTO=Simple while guest checker wants Simple,LL128;
# with shim disabled this only needs to be a valid proto for torch's NCCL.
export NCCL_PROTO="${NCCL_PROTO:-Simple}"
export NCCL_ALGO="${NCCL_ALGO:-Ring}"

DATA_DIR="${DATA_DIR:-./data/nq_hotpotqa_train}"
TRAIN_FILE="${DATA_DIR}/train.parquet"
VAL_FILE="${DATA_DIR}/test.parquet"
HF_MODEL_ID="${HF_MODEL_ID:-Qwen/Qwen2.5-14B}"
MODEL_DIR="${MODEL_DIR:-${PROJECT_ROOT}/models/Qwen2.5-14B}"
WAND_PROJECT="${WAND_PROJECT:-Search-R1}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-nq_sdpo-qwen2.5-14b-em-replay}"
TEACHER_REG="${TEACHER_REG:-actor}"
RETRIEVAL_PORT="${RETRIEVAL_PORT:-8000}"
RETRIEVER_URL="${RETRIEVER_URL:-http://127.0.0.1:${RETRIEVAL_PORT}/retrieve}"
RETRIEVER_DOCS_URL="http://127.0.0.1:${RETRIEVAL_PORT}/docs"
RETRIEVER_WAIT_SECS="${RETRIEVER_WAIT_SECS:-1800}"
TRAIN_REPLAY_DIR="${TRAIN_REPLAY_DIR:-res/train_replays/${EXPERIMENT_NAME}}"
EXP_DIR="${PROJECT_ROOT}/exps/${EXPERIMENT_NAME}"
CKPT_DIR="verl_checkpoints/${EXPERIMENT_NAME}"
INDEX_DIR="${PROJECT_ROOT}/data/wiki18_e5"
INDEX_FILE="${INDEX_DIR}/e5_Flat.index"
CORPUS_FILE="${INDEX_DIR}/wiki-18.jsonl"

# ---------------------------------------------------------------------------
# Prerequisite checks
# ---------------------------------------------------------------------------
if [[ ! -f "${TRAIN_FILE}" ]]; then
  echo "Missing train parquet: ${TRAIN_FILE}"
  exit 1
fi
if [[ ! -f "${VAL_FILE}" ]]; then
  echo "Missing val parquet: ${VAL_FILE}"
  exit 1
fi
if [[ ! -f "${INDEX_FILE}" ]]; then
  echo "Missing retrieval index: ${INDEX_FILE}"
  exit 1
fi
if [[ ! -f "${CORPUS_FILE}" ]]; then
  echo "Missing retrieval corpus: ${CORPUS_FILE}"
  exit 1
fi

# ---------------------------------------------------------------------------
# Experiment directory + metadata helpers
# ---------------------------------------------------------------------------
mkdir -p "${EXP_DIR}"
START_TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
GIT_COMMIT="$(git -C "${PROJECT_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
RETRIEVER_STARTED_BY_SCRIPT=false
RETRIEVER_WAS_ALREADY_UP=false

write_info_json() {
  local status="${1:-running}"
  local end_ts="${2:-}"
  local exit_code="${3:-}"
  python3 - "${EXP_DIR}/info.json" <<PY
import json, os, sys
path = sys.argv[1]
end_ts = """${end_ts}"""
exit_code = """${exit_code}"""
info = {
    "status": """${status}""",
    "experiment_name": """${EXPERIMENT_NAME}""",
    "config_name": "sdpo",
    "hf_model_id": """${HF_MODEL_ID}""",
    "model_path": """${MODEL_DIR}""",
    "train_file": os.path.abspath("""${TRAIN_FILE}"""),
    "val_file": os.path.abspath("""${VAL_FILE}"""),
    "retriever_url": """${RETRIEVER_URL}""",
    "retrieval_port": int("""${RETRIEVAL_PORT}"""),
    "retriever_started_by_script": """${RETRIEVER_STARTED_BY_SCRIPT}""" == "true",
    "retriever_was_already_up": """${RETRIEVER_WAS_ALREADY_UP}""" == "true",
    "cuda_visible_devices": """${CUDA_VISIBLE_DEVICES}""",
    "n_gpus_per_node": int("""${N_GPUS_PER_NODE}"""),
    "wand_project": """${WAND_PROJECT}""",
    "teacher_regularization": """${TEACHER_REG}""",
    "train_replay_dir": os.path.abspath("""${TRAIN_REPLAY_DIR}"""),
    "checkpoint_dir": os.path.abspath("""${CKPT_DIR}"""),
    "exp_dir": """${EXP_DIR}""",
    "git_commit": """${GIT_COMMIT}""",
    "start_timestamp": """${START_TS}""",
    "command": "bash scripts/experiments/rui_meng/train_sdpo_v02_14b_replay.sh",
}
if end_ts:
    info["end_timestamp"] = end_ts
if exit_code != "":
    info["exit_code"] = int(exit_code)
with open(path, "w", encoding="utf-8") as f:
    json.dump(info, f, indent=2)
    f.write("\n")
PY
}

cat > "${EXP_DIR}/README.md" <<EOF
# ${EXPERIMENT_NAME}

Pure SDPO (Search-R1) with **${HF_MODEL_ID}**.

| Artifact | Path |
|----------|------|
| Checkpoints | \`${CKPT_DIR}/\` |
| Train replays | \`${TRAIN_REPLAY_DIR}/\` (\`step_NNNN.jsonl\`) |
| Train log | \`exps/${EXPERIMENT_NAME}/train.log\` |
| Experiment info | \`exps/${EXPERIMENT_NAME}/info.json\` |
| Launcher | \`scripts/experiments/rui_meng/train_sdpo_v02_14b_replay.sh\` |

Re-run:

\`\`\`bash
cd ${PROJECT_ROOT}
bash scripts/experiments/rui_meng/train_sdpo_v02_14b_replay.sh
\`\`\`
EOF

cp "${SCRIPT_DIR}/train_sdpo_v02_14b_replay.sh" "${EXP_DIR}/run.sh"
chmod +x "${EXP_DIR}/run.sh"

# ---------------------------------------------------------------------------
# 1) Download model if missing
# ---------------------------------------------------------------------------
echo "==> Ensuring model at ${MODEL_DIR}"
if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
  mkdir -p "${MODEL_DIR}"
  conda run -n searchr1 --no-capture-output huggingface-cli download \
    "${HF_MODEL_ID}" \
    --local-dir "${MODEL_DIR}"
else
  echo "    Model already present (config.json found)."
fi

# ---------------------------------------------------------------------------
# 2) Retriever: probe, start in background if needed, wait until ready
# ---------------------------------------------------------------------------
retriever_is_up() {
  curl -s -m 2 "${RETRIEVER_DOCS_URL}" >/dev/null 2>&1
}

echo "==> Checking retriever at ${RETRIEVER_DOCS_URL}"
if retriever_is_up; then
  echo "    Retriever already running."
  RETRIEVER_WAS_ALREADY_UP=true
else
  echo "    Retriever not up; starting in background (conda env: retriever)."
  RETRIEVER_STARTED_BY_SCRIPT=true
  # Always source conda.sh (bash -lc PATH alone is not enough for `conda activate`).
  _CONDA_SH=""
  if [[ -f "/root/miniconda3/etc/profile.d/conda.sh" ]]; then
    _CONDA_SH="/root/miniconda3/etc/profile.d/conda.sh"
  elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    _CONDA_SH="${HOME}/miniconda3/etc/profile.d/conda.sh"
  fi
  if [[ -z "${_CONDA_SH}" ]]; then
    echo "ERROR: conda.sh not found; cannot activate retriever env."
    write_info_json "failed" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" 1
    exit 1
  fi
  nohup bash -c "
    set -euo pipefail
    # shellcheck disable=SC1090
    source \"${_CONDA_SH}\"
    conda activate retriever
    cd \"${PROJECT_ROOT}\"
    exec bash retrieval_launch.sh
  " > "${EXP_DIR}/retriever.log" 2>&1 &
  echo $! > "${EXP_DIR}/retriever.pid"
  echo "    Retriever PID $(cat "${EXP_DIR}/retriever.pid"); logging to ${EXP_DIR}/retriever.log"

  echo "==> Waiting up to ${RETRIEVER_WAIT_SECS}s for retriever readiness..."
  _elapsed=0
  _interval=10
  while ! retriever_is_up; do
    if [[ "${_elapsed}" -ge "${RETRIEVER_WAIT_SECS}" ]]; then
      echo "ERROR: Retriever did not become ready within ${RETRIEVER_WAIT_SECS}s."
      echo "       See ${EXP_DIR}/retriever.log"
      write_info_json "failed" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" 1
      exit 1
    fi
    if [[ -f "${EXP_DIR}/retriever.pid" ]] && ! kill -0 "$(cat "${EXP_DIR}/retriever.pid")" 2>/dev/null; then
      echo "ERROR: Retriever process exited before becoming ready."
      echo "       See ${EXP_DIR}/retriever.log"
      write_info_json "failed" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" 1
      exit 1
    fi
    sleep "${_interval}"
    _elapsed=$((_elapsed + _interval))
    echo "    ... still waiting (${_elapsed}s)"
  done
  echo "    Retriever is ready."
fi

write_info_json "running"

# ---------------------------------------------------------------------------
# 3) WandB (optional) + EXTRA_ARGS
# ---------------------------------------------------------------------------
LOGGER_ARG="['console']"
if [[ -n "${WANDB_API_KEY_SEARCH_R1:-}" ]]; then
  unset WANDB_BASE_URL
  unset WANDB_ENTITY
  export WANDB_API_KEY="${WANDB_API_KEY_SEARCH_R1}"
  LOGGER_ARG="['console','wandb']"
  echo "==> WandB enabled (project=${WAND_PROJECT})"
else
  echo "==> WANDB_API_KEY_SEARCH_R1 unset; logging to console only"
fi

EXTRA_ARGS=()
if [[ "${TEACHER_REG}" == "ema" || "${TEACHER_REG}" == "ref" ]]; then
  EXTRA_ARGS+=(
    "actor_rollout_ref.ref.fsdp_config.param_offload=true"
    "actor_rollout_ref.actor.self_distillation.teacher_regularization=${TEACHER_REG}"
  )
fi

echo "==> Starting SDPO training"
echo "    EXPERIMENT_NAME=${EXPERIMENT_NAME}"
echo "    MODEL=${MODEL_DIR}"
echo "    TRAIN_FILE=${TRAIN_FILE}"
echo "    VAL_FILE=${VAL_FILE}"
echo "    N_GPUS_PER_NODE=${N_GPUS_PER_NODE}"
echo "    EXP_DIR=${EXP_DIR}"
echo "    CKPT_DIR=${CKPT_DIR}"
echo "    TRAIN_REPLAY_DIR=${TRAIN_REPLAY_DIR}"

mkdir -p "${TRAIN_REPLAY_DIR}"

# ---------------------------------------------------------------------------
# 4) Train under searchr1
# ---------------------------------------------------------------------------
set +e
conda run -n searchr1 --no-capture-output python3 -m verl.trainer.main_ppo \
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
    actor_rollout_ref.model.path="${MODEL_DIR}" \
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
    actor_rollout_ref.rollout.log_prob_micro_batch_size=128 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.n_agent=5 \
    algorithm.no_think_rl=false \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.actor.state_masking=true \
    reward_model.enable=false \
    "trainer.logger=${LOGGER_ARG}" \
    +trainer.val_only=false \
    trainer.val_before_train=true \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    trainer.save_train_replay=true \
    trainer.train_replay_path="${TRAIN_REPLAY_DIR}" \
    trainer.project_name="${WAND_PROJECT}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.total_epochs=15 \
    trainer.total_training_steps=1005 \
    trainer.default_local_dir="${CKPT_DIR}" \
    do_search=true \
    max_turns=4 \
    retriever.url="${RETRIEVER_URL}" \
    retriever.topk=3 \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "${EXP_DIR}/train.log"
TRAIN_EXIT=${PIPESTATUS[0]}
set -e

END_TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [[ "${TRAIN_EXIT}" -eq 0 ]]; then
  write_info_json "completed" "${END_TS}" "${TRAIN_EXIT}"
  echo "==> Training finished successfully. Artifacts in ${EXP_DIR}"
else
  write_info_json "failed" "${END_TS}" "${TRAIN_EXIT}"
  echo "==> Training failed with exit code ${TRAIN_EXIT}. See ${EXP_DIR}/train.log"
fi

# Leave retriever running (do not kill).
exit "${TRAIN_EXIT}"
