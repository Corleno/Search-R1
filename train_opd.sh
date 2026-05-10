#!/usr/bin/env bash
# On-policy distillation (OPD) training with Search-R1 / verl.
# Set ACTOR_MODEL, TEACHER_MODEL, and data paths before running.
set -euo pipefail

export PYTHONUNBUFFERED=1

: "${ACTOR_MODEL:?Set ACTOR_MODEL to student checkpoint path}"
: "${TEACHER_MODEL:?Set TEACHER_MODEL to teacher causal LM path}"
: "${TRAIN_PARQUET:?Set TRAIN_PARQUET to train parquet}"
: "${VAL_PARQUET:?Set VAL_PARQUET to validation parquet}"

ADV_ESTIMATOR="${ADV_ESTIMATOR:-token_reward_direct}"
LOG_PROB_TOP_K="${LOG_PROB_TOP_K:-16}"
TOP_K_STRATEGY="${TOP_K_STRATEGY:-only_stu}"
REWARD_WEIGHT_MODE="${REWARD_WEIGHT_MODE:-student_p}"
TEACHER_TEMPERATURE="${TEACHER_TEMPERATURE:-1.0}"
N_RESPONSES="${N_RESPONSES:-4}"
MINI_BATCH="${MINI_BATCH:-64}"
WANDB_PROJECT="${WANDB_PROJECT:-Search-R1-OPD}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-opd_run}"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator="${ADV_ESTIMATOR}" \
    algorithm.grpo_outcome_weight="${GRPO_OUTCOME_WEIGHT:-1.0}" \
    data.train_files="${TRAIN_PARQUET}" \
    data.val_files="${VAL_PARQUET}" \
    data.train_batch_size=$((MINI_BATCH * 1)) \
    data.val_batch_size=256 \
    data.shuffle_train_dataloader=True \
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
    actor_rollout_ref.rollout.log_prob_top_k="${LOG_PROB_TOP_K}" \
    actor_rollout_ref.rollout.top_k_strategy="${TOP_K_STRATEGY}" \
    actor_rollout_ref.rollout.reward_weight_mode="${REWARD_WEIGHT_MODE}" \
    actor_rollout_ref.rollout.teacher_temperature="${TEACHER_TEMPERATURE}" \
    reward_model.enable=true \
    reward_model.use_opd_teacher=true \
    reward_model.model.path="${TEACHER_MODEL}" \
    reward_model.model.input_tokenizer=null \
    reward_model.model.use_remove_padding=false \
    +reward_model.model.dtype=bfloat16 \
    reward_model.micro_batch_size=64 \
    reward_model.use_dynamic_bsz=false \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${WANDB_PROJECT}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE:-8}" \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.total_epochs=1 \
    do_search=true
