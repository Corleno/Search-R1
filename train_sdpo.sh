#!/usr/bin/env bash
# Self-Distilled Policy Optimization (SDPO) with Search-R1 multi-turn search.
# No external teacher model — the policy distills from reprompted self-context.
set -euo pipefail

export PYTHONUNBUFFERED=1

: "${ACTOR_MODEL:?Set ACTOR_MODEL to policy checkpoint path}"
: "${TRAIN_PARQUET:?Set TRAIN_PARQUET to train parquet}"
: "${VAL_PARQUET:?Set VAL_PARQUET to validation parquet}"

N_RESPONSES="${N_RESPONSES:-4}"
MINI_BATCH="${MINI_BATCH:-64}"
WANDB_PROJECT="${WANDB_PROJECT:-Search-R1-SDPO}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-sdpo_search_run}"
TEACHER_REG="${TEACHER_REG:-actor}"  # actor | ema | ref

EXTRA_ARGS=()
if [[ "${TEACHER_REG}" == "ema" || "${TEACHER_REG}" == "ref" ]]; then
  EXTRA_ARGS+=(
    "actor_rollout_ref.ref.fsdp_config.param_offload=true"
    "actor_rollout_ref.actor.self_distillation.teacher_regularization=${TEACHER_REG}"
  )
fi

python3 -m verl.trainer.main_ppo \
    --config-name sdpo \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_PARQUET}" \
    data.val_files="${VAL_PARQUET}" \
    data.return_raw_chat=true \
    data.train_batch_size=$((MINI_BATCH * 2)) \
    data.val_batch_size=256 \
    data.shuffle_train_dataloader=true \
    actor_rollout_ref.model.path="${ACTOR_MODEL}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.ppo_mini_batch_size="${MINI_BATCH}" \
    actor_rollout_ref.actor.ppo_micro_batch_size="${MINI_BATCH}" \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n="${N_RESPONSES}" \
    reward_model.enable=false \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${WANDB_PROJECT}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE:-8}" \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.total_epochs=15 \
    do_search=true \
    algorithm.state_masking=true \
    actor_rollout_ref.actor.state_masking=true \
    "${EXTRA_ARGS[@]}"
