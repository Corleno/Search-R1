# On-policy distillation (OPD)

This repo bundles an **on-policy distillation** path adapted from [thunlp/OPD](https://github.com/thunlp/OPD): the **student** (policy rolled out with vLLM) is trained with **dense token rewards** derived from **teacher log-probs** over student (or union) vocabulary, optionally mixed with auxiliary signals.

Training still goes through Search-R1’s **veRL fork** (`verl/`): rollout, Ray actors, FSDP actor, and a dedicated **teacher reward worker**.

## Pipeline (high level)

1. **Rollout** — Same as elsewhere: prompts → responses (`do_search=false` in the provided script; interleaved search + OPD is not the default focus of this README).
2. **Student log-probs** — `compute_log_prob` with **`log_prob_top_k > 0`** keeps top‑K logits per token (and metadata for aligning with the teacher).
3. **Teacher scores** — With `reward_model.use_opd_teacher=true`, the reward group runs **`OPDTeacherRewardWorker`** (causal LM), not the token-classification RM. It adds teacher log-probs / masks needed for rewards.
4. **Distillation (optional)** — If `log_prob_top_k > 0`, **`compute_distillation_reward`** merges student/teacher stats into reward-related tensors (`reward_weight_mode`, `top_k_strategy`).
5. **Reward function** — Your configured **`reward_fn`** turns the batch into **`token_level_scores`** / **`token_level_rewards`**.
6. **Advantages** — For `adv_estimator` in **`token_reward_direct`** or **`token_reward_direct_plus_grpo`**, the trainer uses those token rewards directly (dense) as advantages; **no critic** is spawned for these modes.

## Quick start

1. Install Search-R1 as in the root [README](./README.md) (`pip install -e .`, vLLM, etc.).
2. Set paths and parquet files, then run:

```bash
export ACTOR_MODEL=/path/to/student
export TEACHER_MODEL=/path/to/teacher_causal_lm
export TRAIN_PARQUET=/path/to/train.parquet
export VAL_PARQUET=/path/to/val.parquet
export N_GPUS_PER_NODE=8   # optional; script default is 8

bash train_opd.sh
```

`train_opd.sh` wraps `python3 -m verl.trainer.main_ppo` with OPD-aligned Hydra overrides. Override behavior with environment variables documented at the top of that script (`ADV_ESTIMATOR`, `LOG_PROB_TOP_K`, `TOP_K_STRATEGY`, `REWARD_WEIGHT_MODE`, `TEACHER_TEMPERATURE`, `GRPO_OUTCOME_WEIGHT`, etc.).

## Main configuration knobs

| Area | What to set |
|------|--------------|
| **Algorithm** | `algorithm.adv_estimator`: `token_reward_direct` (pure dense) or `token_reward_direct_plus_grpo` (dense + GRPO-style outcome baseline). For the hybrid mode, **`algorithm.grpo_outcome_weight`** scales the GRPO term. |
| **Rollout / OPD** | Under `actor_rollout_ref.rollout`: `log_prob_top_k`, `top_k_strategy`, `reward_weight_mode`, `teacher_temperature` (defaults in `verl/trainer/config/ppo_trainer.yaml`; example Hydra overrides in `train_opd.sh`). |
| **Teacher RM** | `reward_model.enable=true`, **`reward_model.use_opd_teacher=true`**, `reward_model.model.path=<teacher checkpoint>`. |
| **Trainer** | `do_search=false` in the starter script; enable search only if you have verified your stack for that combo. |

**Critic:** For OPD advantage modes, **`init_workers` skips the critic** (same pattern as GRPO).

## Teacher worker constraints (`OPDTeacherRewardWorker`)

The causal-LM teacher path is narrower than generic veRL RM configs. In practice you should align with **`train_opd.sh`**:

- **`reward_model.model.use_remove_padding=false`** — rmpad is not supported on this worker in this port.
- **`reward_model.use_dynamic_bsz=false`** — dynamic micro-batching is not implemented for OPD RM.
- **`ulysses_sequence_parallel_size==1`** for the reward model worker.

The student.actor side can keep normal Search-R1 choices (including `use_remove_padding` on the actor) as long as tensors line up when unioned into the batch.

## Files to read next

| File | Role |
|------|------|
| `train_opd.sh` | One-command launch + env vars |
| `verl/trainer/config/ppo_trainer.yaml` | Rollout OPD defaults, `reward_model.use_opd_teacher`, `algorithm.grpo_outcome_weight` |
| `verl/workers/opd_teacher_reward_worker.py` | Teacher forward and batch keys |
| `verl/trainer/ppo/ray_trainer.py` | OPD branch in `fit` (`opd_log_prob`, `opd_teacher`, `opd_distill`) |
| `verl/trainer/ppo/core_algos.py` | `compute_token_reward_direct_advantage`, hybrid with GRPO |

## Citation context

Refer to Search-R1 and veRL citations in the root README. For **OPD methodology**, cite the upstream OPD repository / paper linked from [thunlp/OPD](https://github.com/thunlp/OPD).
