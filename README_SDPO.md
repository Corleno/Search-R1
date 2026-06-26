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

## SDPO training replay (JSONL)

During SDPO training, persist student rollouts and eval-ready teacher reprompts as one JSONL file per training step. Requires `data.return_raw_chat=true` (enabled by the `sdpo` config preset).

```bash
trainer.save_train_replay=true \
trainer.train_replay_path=res/train_replays/my_sdpo_run \
bash train_sdpo.sh
```

Files are written as `step_NNNN.jsonl` under the experiment directory (e.g. `res/train_replays/my_sdpo_run/step_0001.jsonl`). Default directory: `{trainer.default_local_dir}/train_replays`.

Each line is one training sample with student `prompt` (raw chat), `teacher_prompt` (reprompt chat usable for eval without re-deriving), `trajectory`, `response`, `score`, `self_distillation_mask`, `solution_used`, and search-turn stats.

Inspect one record:

```bash
head -n 1 res/train_replays/my_sdpo_run/step_0001.jsonl | python -m json.tool
```

Example with replay export enabled: `scripts/experiments/rui_meng/train_sdpo_v02_noclip_replay.sh`.

## Eval from replay JSONL (student / teacher prompts)

Re-run live multi-turn search from a saved replay file or per-step replay directory, choosing which prompt seeds the rollout:

| Mode | Prompt field | Compatible replay |
|------|--------------|-------------------|
| **student** | `prompt` | train replay dir (`step_*.jsonl`), merged `train_replay.jsonl`, `val_replay.jsonl` |
| **teacher** | `teacher_prompt` | train replay dir or merged `train_replay.jsonl` only |

If `prompt` is null in older `val_replay` exports, student mode reconstructs the instruction from the `question` field.

Teacher mode fails fast on `val_replay` (no `teacher_prompt` field). Train replay is deduped by default (`index`, keep latest `step`) so `n_agent` duplicates are not evaluated multiple times unless you set `EVAL_REPLAY_DEDUPE_KEY=none`.

Set `EVAL_REPLAY_REQUIRE_DISTILLATION_MASK=true` (or `trainer.eval_replay_require_distillation_mask=true`) to evaluate only records where `self_distillation_mask > 0` — i.e. samples that received a peer solution or feedback reprompt during SDPO training. This filter applies before dedup and requires `train_replay.jsonl` (not `val_replay`).

```bash
# Student eval on SDPO train replay (all steps in experiment dir)
REPLAY_PATH=res/train_replays/my_sdpo_run \
EVAL_PROMPT_MODE=student \
BASE_MODEL=Qwen/Qwen2.5-3B \
bash scripts/experiments/rui_meng/eval_replay.sh

# Teacher eval (reprompt with peer solution context)
EVAL_PROMPT_MODE=teacher \
REPLAY_PATH=res/train_replays/my_sdpo_run \
bash scripts/experiments/rui_meng/eval_replay.sh

# Teacher eval on distillation-active samples only
EVAL_PROMPT_MODE=teacher \
EVAL_REPLAY_REQUIRE_DISTILLATION_MASK=true \
REPLAY_PATH=res/train_replays/my_sdpo_run \
bash scripts/experiments/rui_meng/eval_replay.sh

# Single-step or legacy merged JSONL still supported
REPLAY_PATH=res/train_replays/my_sdpo_run/step_0042.jsonl \
EVAL_PROMPT_MODE=student \
bash scripts/experiments/rui_meng/eval_replay.sh

# Val replay (student only)
REPLAY_PATH=val_replays/_grpo_v02_test_64.jsonl \
EVAL_PROMPT_MODE=student \
EVAL_REPLAY_LIMIT=8 \
EVAL_LOGGER=console \
bash scripts/experiments/rui_meng/eval_replay.sh
```

Hydra knobs (see `verl/trainer/config/ppo_trainer.yaml`):

- `trainer.eval_from_replay=true`
- `trainer.eval_replay_path=...`
- `trainer.eval_prompt_mode=student|teacher`
- `trainer.eval_replay_dedupe_key=index|id|none`
- `trainer.eval_replay_dedupe_strategy=latest_step|first`
- `trainer.eval_replay_limit` — optional subset cap
- `trainer.eval_replay_require_distillation_mask` — keep only `self_distillation_mask > 0` (train replay)
- `trainer.save_eval_replay=true` — export re-run trajectories to `trainer.eval_replay_output_path`

Outputs: EM metrics (`val/test_score/{data_source}`), plus `val/eval_prompt_mode`, `val/replay_path`, `val/eval_replay_require_distillation_mask`, and optional `val/teacher_prompt_truncated_frac`. When `save_eval_replay=true`, writes JSONL + `<stem>_metrics.json` sidecar (same layout as val replay export).

## Key configuration

| Key | Description |
|-----|-------------|
| `actor_rollout_ref.actor.policy_loss.loss_mode` | `sdpo`, `sdpo_grpo`, or `vanilla` (GRPO) |
| `actor_rollout_ref.actor.policy_loss.sdpo_coef` | Weight for SDPO term in hybrid mode (default `1.0`) |
| `actor_rollout_ref.actor.policy_loss.grpo_coef` | Weight for GRPO term in hybrid mode (default `1.0`) |
| `actor_rollout_ref.actor.self_distillation.*` | Reprompt templates, `success_reward_threshold`, `ppo_clip`, etc. |
| `actor_rollout_ref.actor.self_distillation.ppo_clip` | `true` (default): PPO-style two-sided clip on distillation loss using `actor.clip_ratio`; `false` to disable |
| `actor_rollout_ref.actor.clip_ratio` | ε for PPO clip on GRPO loss and, when `ppo_clip=true`, on SDPO distillation loss (default `0.2`) |
| `actor_rollout_ref.actor.self_distillation.filter_reprompt_before_update` | `true` for pure SDPO; `false` for SDPO+GRPO (full-batch updates) |
| `data.return_raw_chat` | **Required** (`true`) for reprompting |
| `actor_rollout_ref.rollout.n_agent` | Independent search trajectories per prompt when `do_search=true` (e.g. 4–8) |
| `actor_rollout_ref.rollout.n` | Keep at `1` for search; use `>1` only for single-shot (non-search) rollouts |
| `algorithm.adv_estimator` | Use `grpo` (advantages drive GRPO loss; unused in pure SDPO) |
| `actor_rollout_ref.actor.self_distillation.teacher_regularization` | `actor` (default): same weights; `ema`/`ref`: use colocated ref worker |
| `actor_rollout_ref.actor.use_kl_loss` | Recommended `true` for SDPO+GRPO (matches GRPO v02) |

## PPO-clipped SDPO distillation

When `ppo_clip=true`, the per-token distillation term
\(g_t = (\log\pi_\theta - \log\pi_T)_{\text{sg}} \cdot \log\pi_\theta\)
is wrapped with a PPO-style off-policy clip against the rollout policy \(\pi_{\text{old}}\):

\[
\ell_t = \max\!\big(r_t\, g_t,\; \mathrm{clip}(r_t, 1-\varepsilon, 1+\varepsilon)\, g_t\big),
\quad r_t = \frac{\pi_\theta}{\pi_{\text{old}}},
\quad \varepsilon = \texttt{clip\_ratio}.
\]

This replaces the legacy one-sided truncated-IS weight (`is_clip`, removed). Log `actor/sdpo_clipfrac` tracks how often the clipped branch is active.

**Migration:** `self_distillation.is_clip=null` → `self_distillation.ppo_clip=false`. Default clipped runs now use two-sided `clip_ratio=0.2` instead of `is_clip=2.0` (not equivalent).

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
| `scripts/experiments/rui_meng/train_sdpo_v02_noclip_replay.sh` | SDPO v02 + train replay export |
| `scripts/experiments/rui_meng/eval_replay.sh` | Replay-driven eval with student/teacher prompt modes |
| `verl/utils/dataset/replay_dataset.py` | JSONL replay loader for eval (single file or step_*.jsonl directory) |
| `verl/trainer/config/sdpo.yaml` | SDPO Hydra preset |
| `verl/trainer/config/sdpo_grpo.yaml` | SDPO+GRPO Hydra preset |
| `verl/trainer/ppo/core_algos.py` | `compute_self_distillation_loss` |
| `verl/trainer/ppo/ray_trainer.py` | Reprompt batch + teacher log-probs |
| `verl/workers/actor/dp_actor.py` | SDPO / SDPO+GRPO actor update |
| `README_OPD.md` | On-policy distillation (external teacher) |

## Citation

SDPO: [arXiv:2601.20802](https://arxiv.org/abs/2601.20802). Search-R1 and veRL citations are in the root README.
