# Self-Distilled Policy Optimization (SDPO)

This repo integrates [SDPO](https://arxiv.org/abs/2601.20802) (Self-Distilled Policy Optimization) from `/mnt/task_runtime/SDPO` into Search-R1’s **multi-turn search** and **OPD-compatible** veRL trainer.

SDPO trains the policy by distilling to a **self-teacher**: the same model conditioned on successful peer rollouts (and optional feedback), without an external teacher LM like OPD.

## SDPO vs OPD vs GRPO

| Aspect | GRPO | OPD | SDPO |
|--------|------|-----|------|
| Teacher | — | External causal LM | Self (reprompted context ± EMA ref) |
| Loss | PPO clip + group advantage | PPO clip + dense teacher rewards | KL / reverse-KL distillation |
| Critic | No | No | No |
| Search | Yes | Yes | Yes |
| `reward_model` | Rule EM | `use_opd_teacher=true` | Not required |

## Pipeline

1. **Search rollout** — `LLMGenerationManager.run_llm_loop` (same as GRPO/OPD).
2. **Rewards** — Rule-based EM (`RewardManager`); sparse token scores for metrics / success mining.
3. **Reprompt batch** — Group by `uid`, pick successful demos, build `teacher_input_ids` (prompt + solution + feedback).
4. **Teacher log-probs** — On reprompted context (actor weights, or ref policy if `teacher_regularization=ema|ref`).
5. **Actor update** — `compute_self_distillation_loss` matches student log-probs (original prompt) to teacher log-probs (same response tokens).

## Quick start

```bash
export ACTOR_MODEL=Qwen/Qwen2.5-3B
export TRAIN_PARQUET=/path/to/train.parquet
export VAL_PARQUET=/path/to/val.parquet
export N_GPUS_PER_NODE=8

bash train_sdpo.sh
```

Requires a running retrieval server when `do_search=true` (see root README).

### Hydra preset

```bash
python3 -m verl.trainer.main_ppo --config-name sdpo \
  actor_rollout_ref.model.path="${ACTOR_MODEL}" \
  data.train_files=... data.val_files=...
```

## Key configuration

| Key | Description |
|-----|-------------|
| `actor_rollout_ref.actor.policy_loss.loss_mode` | Set to `sdpo` |
| `actor_rollout_ref.actor.self_distillation.*` | Reprompt templates, `success_reward_threshold`, `is_clip`, etc. |
| `data.return_raw_chat` | **Required** (`true`) for reprompting |
| `actor_rollout_ref.rollout.n` | Samples per prompt (e.g. 4–8) |
| `algorithm.adv_estimator` | Use `grpo` (advantages computed but not used in SDPO loss) |
| `actor_rollout_ref.actor.self_distillation.teacher_regularization` | `actor` (default): same weights; `ema`/`ref`: use colocated ref worker |

## Search compatibility

- Uses the same `do_search`, `max_turns`, retriever URL, and **`state_masking`** / `loss_mask` as GRPO scripts.
- `uid` comes from dataset `index` so group-wise success mining aligns with `rollout.n` repeats.

## OPD compatibility

SDPO and OPD are **mutually exclusive** training modes:

- OPD: `algorithm.adv_estimator=token_reward_direct*` + `reward_model.use_opd_teacher=true`
- SDPO: `policy_loss.loss_mode=sdpo` + `reward_model.enable=false`

You can compare checkpoints using the same search eval scripts (e.g. `eval_grpo_v02.sh`).

## Files

| File | Role |
|------|------|
| `train_sdpo.sh` | Launch script |
| `verl/trainer/config/sdpo.yaml` | Hydra preset |
| `verl/trainer/ppo/core_algos.py` | `compute_self_distillation_loss` |
| `verl/trainer/ppo/ray_trainer.py` | Reprompt batch + teacher log-probs |
| `verl/workers/actor/dp_actor.py` | SDPO actor update |
| `README_OPD.md` | On-policy distillation (external teacher) |

## Citation

SDPO: [arXiv:2601.20802](https://arxiv.org/abs/2601.20802). Search-R1 and veRL citations are in the root README.
