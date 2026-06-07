# Self-Distilled Policy Optimization (SDPO)

This repo integrates [SDPO](https://arxiv.org/abs/2601.20802) (Self-Distilled Policy Optimization) from `/mnt/task_runtime/SDPO` into Search-R1’s **multi-turn search** and **OPD-compatible** veRL trainer.

SDPO trains the policy by distilling to a **self-teacher**: the same model conditioned on successful peer rollouts (and optional feedback), without an external teacher LM like OPD.

## SDPO vs OPD vs GRPO vs SDPO+GRPO

| Aspect | GRPO | OPD | SDPO | SDPO+GRPO |
|--------|------|-----|------|-----------|
| Teacher | — | External causal LM | Self (reprompted context ± EMA ref) | Self (same as SDPO) |
| Loss | PPO clip + group advantage | PPO clip + dense teacher rewards | KL / reverse-KL distillation | `sdpo_coef * L_sdpo + grpo_coef * L_grpo` |
| Critic | No | No | No | No |
| Search | Yes | Yes | Yes | Yes |
| `reward_model` | Rule EM | `use_opd_teacher=true` | Not required | Not required |
| Actor batch | Full batch | Full batch | Reprompt-filtered | Full batch (SDPO masked) |

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

### SDPO+GRPO hybrid

Combines self-distillation with GRPO clipped policy loss on the **full actor batch**. SDPO loss applies only to reprompt-active samples (`self_distillation_mask`); GRPO loss applies to all samples.

```bash
python3 -m verl.trainer.main_ppo --config-name sdpo_grpo \
  actor_rollout_ref.model.path="${ACTOR_MODEL}" \
  data.train_files=... data.val_files=...
```

Or use the test launch script: `scripts/experiments/rui_meng/train_sdpo_grpo_v02_test.sh`.

## Key configuration

| Key | Description |
|-----|-------------|
| `actor_rollout_ref.actor.policy_loss.loss_mode` | `sdpo`, `sdpo_grpo`, or `vanilla` (GRPO) |
| `actor_rollout_ref.actor.policy_loss.sdpo_coef` | Weight for SDPO term in hybrid mode (default `1.0`) |
| `actor_rollout_ref.actor.policy_loss.grpo_coef` | Weight for GRPO term in hybrid mode (default `1.0`) |
| `actor_rollout_ref.actor.self_distillation.*` | Reprompt templates, `success_reward_threshold`, `is_clip`, etc. |
| `actor_rollout_ref.actor.self_distillation.filter_reprompt_before_update` | `true` for pure SDPO; `false` for SDPO+GRPO (full-batch updates) |
| `data.return_raw_chat` | **Required** (`true`) for reprompting |
| `actor_rollout_ref.rollout.n_agent` | Independent search trajectories per prompt when `do_search=true` (e.g. 4–8) |
| `actor_rollout_ref.rollout.n` | Keep at `1` for search; use `>1` only for single-shot (non-search) rollouts |
| `algorithm.adv_estimator` | Use `grpo` (advantages drive GRPO loss; unused in pure SDPO) |
| `actor_rollout_ref.actor.self_distillation.teacher_regularization` | `actor` (default): same weights; `ema`/`ref`: use colocated ref worker |
| `actor_rollout_ref.actor.use_kl_loss` | Recommended `true` for SDPO+GRPO (matches GRPO v02) |

## Search compatibility

- Uses the same `do_search`, `max_turns`, retriever URL, and **`state_masking`** / `loss_mask` as GRPO scripts.
- `uid` comes from dataset `index` so group-wise success mining aligns with `rollout.n_agent` repeats.

## OPD compatibility

SDPO and OPD are **mutually exclusive** training modes:

- OPD: `algorithm.adv_estimator=token_reward_direct*` + `reward_model.use_opd_teacher=true`
- SDPO: `policy_loss.loss_mode=sdpo` + `reward_model.enable=false`
- SDPO+GRPO: `policy_loss.loss_mode=sdpo_grpo` + `reward_model.enable=false`

You can compare checkpoints using the same search eval scripts (e.g. `eval_grpo_v02.sh`).

## Files

| File | Role |
|------|------|
| `train_sdpo.sh` | Launch script |
| `scripts/experiments/rui_meng/train_sdpo_grpo_v02_test.sh` | SDPO+GRPO test launch script |
| `verl/trainer/config/sdpo.yaml` | SDPO Hydra preset |
| `verl/trainer/config/sdpo_grpo.yaml` | SDPO+GRPO Hydra preset |
| `verl/trainer/ppo/core_algos.py` | `compute_self_distillation_loss` |
| `verl/trainer/ppo/ray_trainer.py` | Reprompt batch + teacher log-probs |
| `verl/workers/actor/dp_actor.py` | SDPO / SDPO+GRPO actor update |
| `README_OPD.md` | On-policy distillation (external teacher) |

## Citation

SDPO: [arXiv:2601.20802](https://arxiv.org/abs/2601.20802). Search-R1 and veRL citations are in the root README.
