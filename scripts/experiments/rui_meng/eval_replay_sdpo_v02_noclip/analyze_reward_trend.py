#!/usr/bin/env python3
"""Analyze reward trend and distribution from training replay JSONL files."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
DEFAULT_REPLAY_DIR = os.path.join(
    REPO_ROOT, "res", "train_replays", "exp_sdpo_searchr1_0620"
)
DEFAULT_OUTPUT_DIR = os.path.join(
    REPO_ROOT, "res", "train_replay_analysis", "exp_sdpo_searchr1_0620"
)
STEP_RE = re.compile(r"step_(\d+)\.jsonl$")


@dataclass
class StepStats:
    step: int
    count: int = 0
    score_sum: float = 0.0
    score_sq_sum: float = 0.0
    mask_count: int = 0
    turns_sum: float = 0.0
    valid_searches_sum: float = 0.0
    by_source: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def add(
        self,
        score: float,
        data_source: str,
        mask: float,
        turns: float,
        valid_searches: float,
    ) -> None:
        self.count += 1
        self.score_sum += score
        self.score_sq_sum += score * score
        if mask > 0:
            self.mask_count += 1
        self.turns_sum += turns
        self.valid_searches_sum += valid_searches

        source_stats = self.by_source.setdefault(
            data_source,
            {"count": 0.0, "score_sum": 0.0},
        )
        source_stats["count"] += 1
        source_stats["score_sum"] += score

    @property
    def mean_score(self) -> float:
        return self.score_sum / self.count if self.count else 0.0

    @property
    def std_score(self) -> float:
        if self.count < 2:
            return 0.0
        mean = self.mean_score
        var = max(self.score_sq_sum / self.count - mean * mean, 0.0)
        return var**0.5

    @property
    def pass_rate(self) -> float:
        return self.mean_score

    @property
    def mask_frac(self) -> float:
        return self.mask_count / self.count if self.count else 0.0

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "count": self.count,
            "mean_score": self.mean_score,
            "std_score": self.std_score,
            "pass_rate": self.pass_rate,
            "mask_frac": self.mask_frac,
            "mean_turns": self.turns_sum / self.count if self.count else 0.0,
            "mean_valid_searches": self.valid_searches_sum / self.count if self.count else 0.0,
            "by_source": {
                source: {
                    "count": int(stats["count"]),
                    "mean_score": stats["score_sum"] / stats["count"],
                    "pass_rate": stats["score_sum"] / stats["count"],
                }
                for source, stats in sorted(self.by_source.items())
            },
        }


def _step_sort_key(path: str) -> int:
    match = STEP_RE.search(os.path.basename(path))
    return int(match.group(1)) if match else 10**9


def discover_replay_files(replay_dir: str) -> List[str]:
    files = glob.glob(os.path.join(replay_dir, "step_*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No step_*.jsonl files found in {replay_dir}")
    return sorted(files, key=_step_sort_key)


def filter_replay_files(
    files: Sequence[str],
    step_min: Optional[int],
    step_max: Optional[int],
    step_stride: int,
) -> List[str]:
    selected: List[str] = []
    base_step: Optional[int] = None
    for path in files:
        match = STEP_RE.search(os.path.basename(path))
        if not match:
            continue
        step = int(match.group(1))
        if base_step is None:
            base_step = step
        if step_min is not None and step < step_min:
            continue
        if step_max is not None and step > step_max:
            continue
        if step_stride > 1 and (step - (step_min if step_min is not None else base_step)) % step_stride != 0:
            continue
        selected.append(path)
    return selected


def iter_records(path: str) -> Iterable[dict]:
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}") from exc


def collect_stats(files: Sequence[str]) -> Tuple[List[StepStats], List[float], Dict[str, List[float]]]:
    per_step: List[StepStats] = []
    all_scores: List[float] = []
    scores_by_source: Dict[str, List[float]] = defaultdict(list)

    for path in files:
        match = STEP_RE.search(os.path.basename(path))
        step = int(match.group(1)) if match else -1
        stats = StepStats(step=step)

        for record in iter_records(path):
            score = float(record.get("score", 0.0))
            data_source = str(record.get("data_source", "unknown"))
            mask = float(record.get("self_distillation_mask", 0.0) or 0.0)
            turns = float(record.get("turns", 0.0) or 0.0)
            valid_searches = float(record.get("valid_searches", 0.0) or 0.0)

            stats.add(score, data_source, mask, turns, valid_searches)
            all_scores.append(score)
            scores_by_source[data_source].append(score)

        per_step.append(stats)

    return per_step, all_scores, scores_by_source


def rolling_mean(values: Sequence[float], window: int) -> List[float]:
    if not values:
        return []
    window = max(1, window)
    out: List[float] = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        chunk = values[start : idx + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def write_csv(path: str, per_step: Sequence[StepStats], sources: Sequence[str]) -> None:
    fieldnames = [
        "step",
        "count",
        "mean_score",
        "std_score",
        "pass_rate",
        "mask_frac",
        "mean_turns",
        "mean_valid_searches",
    ]
    for source in sources:
        fieldnames.extend([f"{source}_count", f"{source}_mean_score"])

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for stats in per_step:
            row = {
                "step": stats.step,
                "count": stats.count,
                "mean_score": stats.mean_score,
                "std_score": stats.std_score,
                "pass_rate": stats.pass_rate,
                "mask_frac": stats.mask_frac,
                "mean_turns": stats.turns_sum / stats.count if stats.count else 0.0,
                "mean_valid_searches": stats.valid_searches_sum / stats.count if stats.count else 0.0,
            }
            for source in sources:
                source_stats = stats.by_source.get(source)
                if source_stats:
                    row[f"{source}_count"] = int(source_stats["count"])
                    row[f"{source}_mean_score"] = source_stats["score_sum"] / source_stats["count"]
                else:
                    row[f"{source}_count"] = 0
                    row[f"{source}_mean_score"] = 0.0
            writer.writerow(row)


def plot_reward_trend(
    output_path: str,
    per_step: Sequence[StepStats],
    sources: Sequence[str],
    rolling_window: int,
) -> None:
    steps = [stats.step for stats in per_step]
    mean_scores = [stats.mean_score for stats in per_step]
    rolling = rolling_mean(mean_scores, rolling_window)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax = axes[0]
    ax.plot(steps, mean_scores, marker="o", markersize=3, linewidth=1.2, label="mean score")
    ax.plot(
        steps,
        rolling,
        linewidth=2.0,
        label=f"rolling mean (window={rolling_window})",
    )
    for source in sources:
        source_scores = []
        for stats in per_step:
            source_stats = stats.by_source.get(source)
            if source_stats and source_stats["count"]:
                source_scores.append(source_stats["score_sum"] / source_stats["count"])
            else:
                source_scores.append(float("nan"))
        ax.plot(steps, source_scores, linewidth=1.0, alpha=0.8, label=f"{source} mean")

    ax.set_ylabel("Reward / pass rate")
    ax.set_title("Training replay reward trend")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    ax = axes[1]
    mask_fracs = [stats.mask_frac for stats in per_step]
    mean_turns = [stats.turns_sum / stats.count if stats.count else 0.0 for stats in per_step]
    ax.plot(steps, mask_fracs, marker="o", markersize=3, color="tab:orange", label="self_distillation_mask frac")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mask fraction", color="tab:orange")
    ax.tick_params(axis="y", labelcolor="tab:orange")
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(steps, mean_turns, color="tab:green", linewidth=1.2, label="mean turns")
    ax2.set_ylabel("Mean turns", color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green")

    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], loc="best", fontsize=9)
    ax.set_title("Distillation mask and trajectory length")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_score_distribution(
    output_path: str,
    all_scores: Sequence[float],
    scores_by_source: Dict[str, Sequence[float]],
    per_step: Sequence[StepStats],
    snapshot_steps: Sequence[int],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    pass_count = int(sum(all_scores))
    fail_count = len(all_scores) - pass_count
    counts = [fail_count, pass_count]
    labels = ["0 (fail)", "1 (pass)"]
    bars = ax.bar(labels, [100.0 * c / len(all_scores) for c in counts], edgecolor="black")
    ax.set_ylabel("Percent of samples")
    ax.set_title(f"Overall score distribution (n={len(all_scores):,})")
    ax.grid(True, axis="y", alpha=0.3)
    for bar, count in zip(bars, counts):
        pct = 100.0 * count / len(all_scores)
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{pct:.1f}%\n(n={count:,})", ha="center", va="bottom", fontsize=9)

    ax = axes[1]
    source_names = sorted(scores_by_source)
    pass_rates = [
        100.0 * sum(scores) / len(scores) if scores else 0.0
        for scores in (scores_by_source[name] for name in source_names)
    ]
    bars = ax.bar(source_names, pass_rates, edgecolor="black")
    ax.set_ylabel("Pass rate (%)")
    ax.set_title("Pass rate by data source")
    ax.grid(True, axis="y", alpha=0.3)
    for bar, rate in zip(bars, pass_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{rate:.1f}%", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    if not snapshot_steps:
        return

    step_lookup = {stats.step: stats for stats in per_step}
    selected = [step for step in snapshot_steps if step in step_lookup]
    if not selected:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.8 / len(selected)
    for idx, step in enumerate(selected):
        stats = step_lookup[step]
        pass_rate = 100.0 * stats.pass_rate
        ax.bar(idx, pass_rate, width=0.8, label=f"step {step}")
        ax.text(idx, pass_rate, f"{pass_rate:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(range(len(selected)))
    ax.set_xticklabels([f"step {step}" for step in selected])
    ax.set_ylabel("Pass rate (%)")
    ax.set_title("Pass rate at selected training steps")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    snapshot_path = output_path.replace(".png", "_snapshots.png")
    fig.savefig(snapshot_path, dpi=150)
    plt.close(fig)


def build_summary(
    replay_dir: str,
    per_step: Sequence[StepStats],
    all_scores: Sequence[float],
    scores_by_source: Dict[str, Sequence[float]],
) -> dict:
    overall_pass = sum(all_scores) / len(all_scores) if all_scores else 0.0
    return {
        "replay_dir": replay_dir,
        "num_steps": len(per_step),
        "num_samples": len(all_scores),
        "overall": {
            "mean_score": overall_pass,
            "pass_rate": overall_pass,
            "std_score": statistics.pstdev(all_scores) if len(all_scores) > 1 else 0.0,
            "pass_count": int(sum(all_scores)),
            "fail_count": len(all_scores) - int(sum(all_scores)),
        },
        "by_source": {
            source: {
                "count": len(scores),
                "mean_score": sum(scores) / len(scores),
                "pass_rate": sum(scores) / len(scores),
            }
            for source, scores in sorted(scores_by_source.items())
        },
        "per_step": [stats.to_dict() for stats in per_step],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replay-dir",
        default=DEFAULT_REPLAY_DIR,
        help=f"Directory with step_*.jsonl replays (default: {DEFAULT_REPLAY_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for plots and summary files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument("--step-min", type=int, default=None, help="Minimum step (inclusive)")
    parser.add_argument("--step-max", type=int, default=None, help="Maximum step (inclusive)")
    parser.add_argument(
        "--step-stride",
        type=int,
        default=1,
        help="Analyze every Nth step (default: 1 = all steps)",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=10,
        help="Window size for rolling mean in trend plot (default: 10)",
    )
    parser.add_argument(
        "--snapshot-steps",
        default="1,50,100,150,200",
        help="Comma-separated steps for snapshot pass-rate bars (default: 1,50,100,150,200)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    replay_dir = os.path.abspath(args.replay_dir)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    all_files = discover_replay_files(replay_dir)
    files = filter_replay_files(all_files, args.step_min, args.step_max, args.step_stride)
    if not files:
        raise SystemExit("No replay files matched the requested step filters.")

    print(f"Analyzing {len(files)} replay files from {replay_dir}")
    per_step, all_scores, scores_by_source = collect_stats(files)
    sources = sorted({source for stats in per_step for source in stats.by_source})

    summary = build_summary(replay_dir, per_step, all_scores, scores_by_source)
    summary_path = os.path.join(output_dir, "reward_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    csv_path = os.path.join(output_dir, "reward_per_step.csv")
    write_csv(csv_path, per_step, sources)

    trend_plot = os.path.join(output_dir, "reward_trend.png")
    dist_plot = os.path.join(output_dir, "reward_distribution.png")
    plot_reward_trend(trend_plot, per_step, sources, args.rolling_window)

    snapshot_steps = [int(item.strip()) for item in args.snapshot_steps.split(",") if item.strip()]
    plot_score_distribution(dist_plot, all_scores, scores_by_source, per_step, snapshot_steps)

    print(f"Wrote summary: {summary_path}")
    print(f"Wrote per-step CSV: {csv_path}")
    print(f"Wrote plots: {trend_plot}, {dist_plot}")
    print(
        "Overall pass rate: "
        f"{summary['overall']['pass_rate'] * 100:.2f}% "
        f"({summary['overall']['pass_count']}/{summary['num_samples']})"
    )
    for source, stats in summary["by_source"].items():
        print(f"  {source}: {stats['pass_rate'] * 100:.2f}% ({stats['count']} samples)")


if __name__ == "__main__":
    main()
