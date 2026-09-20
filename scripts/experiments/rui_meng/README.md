# Plot & analysis scripts (`rui_meng`)

Scripts for analyzing training-replay JSONLs and comparing experiments on shared figures.

Run all commands from the **repo root**.

## Pipeline overview

```
train_replays/*.jsonl
        │
        ▼  analyze_reward_trend_*.sh
res/train_replay_analysis/<exp>/reward_summary.json
        │
        ├──► plot_reward_trend_compare_*.sh  → reward_trend_compare.{png,csv}
        └──► plot_mask_frac_compare.sh       → mask_frac_compare.{png,csv}
```

1. **Sync replays** (if needed):

```bash
bash scripts/sync_train_replays_from_gcs.sh --subpath nq_sdpo-qwen2.5-3b-em-noclip-replay
```

2. **Analyze** each experiment so `reward_summary.json` exists.
3. **Plot** comparisons across experiments.

---

## 1. Analyze reward trends (prerequisite)

### `analyze_reward_trend.py`

Core analyzer. Reads `step_*.jsonl` under a replay dir and writes:

| Output | Description |
|--------|-------------|
| `reward_summary.json` | Per-step pass rate, mask frac, by-source stats (used by plotters) |
| `reward_trend.png` / CSV | Single-experiment pass-rate trend |
| Snapshot / distribution plots | Score bars and histograms |

### `analyze_reward_trend_nq_sdpo_qwen25_3b_em_replay.sh`

Convenience wrapper for the NQ SDPO Qwen2.5-3B EM replay family (and related MOPD runs).

```bash
bash scripts/experiments/rui_meng/analyze_reward_trend_nq_sdpo_qwen25_3b_em_replay.sh [VARIANT]
```

| Variant | Experiment dir under `res/train_replays/` |
|---------|-------------------------------------------|
| `base` (default) | `nq_sdpo-qwen2.5-3b-em-noclip-replay` |
| `ema` | `nq_sdpo-qwen2.5-3b-em-noclip-replay-ema` |
| `ref` | `nq_sdpo-qwen2.5-3b-em-noclip-replay-ref` |
| `ppo` | `nq_sdpo-qwen2.5-3b-em-ppoclip-replay` |
| `mopd` | `sdpo_v02_noclip_replay_ref_qwen2.5-3b-qwen2.5-7b-instruct` |
| `fs_mopd` | `sdpo_v02_noclip_replay_ref_qwen2.5-3b-qwen2.5-7b-instruct-empty-reprompt` |

Analysis lands in `res/train_replay_analysis/<same-name>/`.

```bash
# Analyze all variants used by the compare plots
for v in base ema ref ppo mopd fs_mopd; do
  bash scripts/experiments/rui_meng/analyze_reward_trend_nq_sdpo_qwen25_3b_em_replay.sh "$v"
done
```

Useful env overrides: `REPLAY_DIR`, `OUTPUT_DIR`, `STEP_MIN`, `STEP_MAX`, `STEP_STRIDE`, `PLOT_X_MIN`, `PLOT_X_MAX`.

---

## 2. Compare pass-rate trends

### `plot_reward_trend_compare.py`

Overlays multiple experiments’ pass rates in three panels: **Overall**, **NQ**, **HotpotQA**.

Requires `LABEL=ANALYSIS_DIR` pairs (at least two). Each `ANALYSIS_DIR` must contain `reward_summary.json`.

```bash
python3 scripts/experiments/rui_meng/plot_reward_trend_compare.py \
  --experiments "a=res/train_replay_analysis/exp_a,b=res/train_replay_analysis/exp_b" \
  --output-dir res/train_replay_analysis/compare
```

**Outputs:** `reward_trend_compare.png`, `reward_trend_compare.csv`

| Flag | Default | Meaning |
|------|---------|---------|
| `--experiments` | fcsd vs fcsd-ppo | Comma-separated `LABEL=DIR` |
| `--output-dir` | `res/train_replay_analysis/compare` | Where PNG/CSV are written |
| `--plot-x-min` / `--plot-x-max` | `0` / `100` | Fixed x-axis range |
| `--step-stride` | `5` | Plot every Nth step |
| `--show-rolling` | off | Overlay rolling mean |
| `--rolling-window` | `10` | Window when rolling is on |
| `--no-plot-xlim` | off | Autoscale x-axis |

### Wrapper: FCSD vs FCSD-PPO

```bash
bash scripts/experiments/rui_meng/plot_reward_trend_compare_fcsd_variates.sh
```

Defaults:

- **Experiments:** `fcsd` (noclip) vs `fcsd-ppo` (ppoclip)
- **Output:** `res/train_replay_analysis/compare/`
- **X range:** `0`–`200`, stride `5`

### Wrapper: teacher variants (base / ema / ref)

```bash
bash scripts/experiments/rui_meng/plot_reward_trend_compare_teacher_variates.sh
```

Defaults:

- **Experiments:** `base`, `ema`, `ref`
- **Output:** `res/train_replay_analysis/compare/base-ema-ref/`

### Wrapper: MOPD variants (`mopd` / `fs_mopd`)

```bash
bash scripts/experiments/rui_meng/plot_reward_trend_compare_mopd_variates.sh
```

Defaults:

- **Experiments:** `mopd` vs `fs_mopd` (empty-reprompt)
- **Output:** `res/train_replay_analysis/compare/mopd-fs_mopd/`

### Shared env overrides (all wrappers)

```bash
OUTPUT_DIR=... EXPERIMENTS="a=dir_a,b=dir_b" \
PLOT_X_MIN=0 PLOT_X_MAX=200 STEP_STRIDE=5 \
SHOW_ROLLING=1 ROLLING_WINDOW=10 \
bash scripts/experiments/rui_meng/plot_reward_trend_compare_fcsd_variates.sh
```

Set `NO_PLOT_XLIM=1` to disable the fixed x-axis.

---

## 3. Compare mask fraction (effective sample ratio)

### `plot_mask_frac_compare.py` / `plot_mask_frac_compare.sh`

Same input (`reward_summary.json`) and CLI/env pattern as the reward-trend comparer, but plots per-step **`mask_frac`** (effective sample ratio) on a single panel.

```bash
bash scripts/experiments/rui_meng/plot_mask_frac_compare.sh
```

Defaults match the FCSD vs FCSD-PPO pair; output:

- `res/train_replay_analysis/compare/mask_frac_compare.png`
- `res/train_replay_analysis/compare/mask_frac_compare.csv`

---

## 4. Eval replay scores (optional)

Under `eval_replay_sdpo_v02_noclip/`:

```bash
bash scripts/experiments/rui_meng/eval_replay_sdpo_v02_noclip/plot_eval_test_scores.sh
```

Plots student/teacher eval test scores from eval-replay metrics (separate from train-replay analysis).

---

## Quick recipes

**FCSD clip comparison (pass rate + mask frac):**

```bash
bash scripts/experiments/rui_meng/analyze_reward_trend_nq_sdpo_qwen25_3b_em_replay.sh base
bash scripts/experiments/rui_meng/analyze_reward_trend_nq_sdpo_qwen25_3b_em_replay.sh ppo
bash scripts/experiments/rui_meng/plot_reward_trend_compare_fcsd_variates.sh
bash scripts/experiments/rui_meng/plot_mask_frac_compare.sh
```

**Teacher variant comparison:**

```bash
for v in base ema ref; do
  bash scripts/experiments/rui_meng/analyze_reward_trend_nq_sdpo_qwen25_3b_em_replay.sh "$v"
done
bash scripts/experiments/rui_meng/plot_reward_trend_compare_teacher_variates.sh
```

**MOPD vs empty-reprompt (`fs_mopd`):**

```bash
for v in mopd fs_mopd; do
  bash scripts/experiments/rui_meng/analyze_reward_trend_nq_sdpo_qwen25_3b_em_replay.sh "$v"
done
bash scripts/experiments/rui_meng/plot_reward_trend_compare_mopd_variates.sh
```
