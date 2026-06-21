#!/usr/bin/env python3
"""Plot student and teacher eval test scores over training replay steps."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
DEFAULT_EVAL_DIR = os.path.join(
    REPO_ROOT, "res", "eval_replays", "exp_sdpo_searchr1_0620"
)
DEFAULT_OUTPUT_DIR = os.path.join(
    REPO_ROOT, "res", "eval_replay_analysis", "exp_sdpo_searchr1_0620"
)
DEFAULT_STEPS = list(range(1, 192, 10))
DATASETS = ("nq", "hotpotqa")
COMBINED_DATASET = "combined"
MODES = ("student", "teacher")
METRIC_KEYS = {dataset: f"val/test_score/{dataset}" for dataset in DATASETS}


def load_metrics(eval_dir: str, step: int, mode: str) -> Dict[str, float]:
    path = os.path.join(eval_dir, f"step_{step:04d}_{mode}_metrics.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing metrics file: {path}")

    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)

    metrics = payload.get("metrics", {})
    scores = {}
    for dataset, key in METRIC_KEYS.items():
        if key not in metrics:
            raise KeyError(f"{key} not found in {path}")
        scores[dataset] = float(metrics[key])
    return scores


def count_eval_replay_samples(eval_dir: str, step: int, mode: str) -> Dict[str, int]:
    """Count evaluated replay records (mask=1 subset after dedupe) by data source."""
    path = os.path.join(eval_dir, f"step_{step:04d}_{mode}")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing eval replay file: {path}")

    counts = Counter()
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}") from exc
            counts[str(record.get("data_source", "unknown"))] += 1

    return {dataset: int(counts.get(dataset, 0)) for dataset in DATASETS}


def collect_scores_and_mask_counts(
    eval_dir: str,
    steps: Sequence[int],
) -> Tuple[List[int], Dict[str, Dict[str, List[float]]], Dict[str, List[int]]]:
    ordered_steps = list(steps)
    scores: Dict[str, Dict[str, List[float]]] = {
        mode: {dataset: [] for dataset in DATASETS} for mode in MODES
    }
    mask_counts: Dict[str, List[int]] = {dataset: [] for dataset in DATASETS}

    for step in ordered_steps:
        student_counts = count_eval_replay_samples(eval_dir, step, "student")
        teacher_counts = count_eval_replay_samples(eval_dir, step, "teacher")
        if student_counts != teacher_counts:
            raise ValueError(
                f"Student/teacher distillation-mask sample counts differ at step {step}: "
                f"{student_counts} vs {teacher_counts}"
            )

        for dataset in DATASETS:
            mask_counts[dataset].append(student_counts[dataset])

        for mode in MODES:
            step_scores = load_metrics(eval_dir, step, mode)
            for dataset in DATASETS:
                scores[mode][dataset].append(step_scores[dataset])

    return ordered_steps, scores, mask_counts


def build_combined_series(
    scores: Dict[str, Dict[str, List[float]]],
    mask_counts: Dict[str, List[int]],
) -> Tuple[Dict[str, List[float]], List[int]]:
    num_steps = len(next(iter(mask_counts.values())))
    combined_scores = {mode: [] for mode in MODES}
    combined_mask_counts: List[int] = []

    for idx in range(num_steps):
        total_count = sum(mask_counts[dataset][idx] for dataset in DATASETS)
        combined_mask_counts.append(total_count)

        for mode in MODES:
            if total_count == 0:
                combined_scores[mode].append(float("nan"))
                continue
            weighted_score = sum(
                scores[mode][dataset][idx] * mask_counts[dataset][idx]
                for dataset in DATASETS
            ) / total_count
            combined_scores[mode].append(weighted_score)

    return combined_scores, combined_mask_counts


def write_csv(
    path: str,
    steps: Sequence[int],
    scores: Dict[str, Dict[str, List[float]]],
    mask_counts: Dict[str, List[int]],
    combined_scores: Dict[str, List[float]],
    combined_mask_counts: List[int],
) -> None:
    fieldnames = ["step", "distillation_mask_count_combined"]
    for dataset in DATASETS:
        fieldnames.append(f"distillation_mask_count_{dataset}")
    for mode in MODES:
        fieldnames.append(f"{mode}_combined")
        for dataset in DATASETS:
            fieldnames.append(f"{mode}_{dataset}")

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, step in enumerate(steps):
            row = {
                "step": step,
                "distillation_mask_count_combined": combined_mask_counts[idx],
            }
            for dataset in DATASETS:
                row[f"distillation_mask_count_{dataset}"] = mask_counts[dataset][idx]
            for mode in MODES:
                row[f"{mode}_combined"] = combined_scores[mode][idx]
                for dataset in DATASETS:
                    row[f"{mode}_{dataset}"] = scores[mode][dataset][idx]
            writer.writerow(row)


def _plot_panel(
    ax: plt.Axes,
    steps: Sequence[int],
    panel_scores: Dict[str, Sequence[float]],
    panel_mask_counts: Sequence[int],
    panel_title: str,
) -> None:
    style = {
        "student": {"color": "tab:blue", "marker": "o", "linestyle": "-"},
        "teacher": {"color": "tab:orange", "marker": "s", "linestyle": "--"},
    }

    for mode in MODES:
        ax.plot(
            steps,
            panel_scores[mode],
            label=f"{mode} score",
            linewidth=1.8,
            markersize=5,
            **style[mode],
        )
    ax.set_title(panel_title)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.bar(
        steps,
        panel_mask_counts,
        width=8.0,
        alpha=0.25,
        color="tab:gray",
    )
    ax2.set_ylabel("Effective training samples")
    ymax = max(panel_mask_counts) if panel_mask_counts else 1
    ax2.set_ylim(0.0, ymax * 1.15)

    mask_handle = Patch(facecolor="tab:gray", alpha=0.25, label="effective training samples")
    ax.legend(
        ax.get_lines() + [mask_handle],
        [line.get_label() for line in ax.get_lines()] + ["effective training samples"],
        loc="best",
        fontsize=9,
    )


def plot_test_scores(
    output_path: str,
    steps: Sequence[int],
    scores: Dict[str, Dict[str, List[float]]],
    mask_counts: Dict[str, List[int]],
    combined_scores: Dict[str, List[float]],
    combined_mask_counts: List[int],
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)

    labels = {
        "nq": "NQ",
        "hotpotqa": "HotpotQA",
        COMBINED_DATASET: "NQ + HotpotQA",
    }
    panel_data = [
        (DATASETS[0], {mode: scores[mode][DATASETS[0]] for mode in MODES}, mask_counts[DATASETS[0]]),
        (DATASETS[1], {mode: scores[mode][DATASETS[1]] for mode in MODES}, mask_counts[DATASETS[1]]),
        (COMBINED_DATASET, combined_scores, combined_mask_counts),
    ]

    for ax, (dataset, panel_scores, panel_mask_counts) in zip(axes, panel_data):
        _plot_panel(ax, steps, panel_scores, panel_mask_counts, labels[dataset])

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def parse_steps(raw: str) -> List[int]:
    if not raw.strip():
        return DEFAULT_STEPS
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-dir",
        default=DEFAULT_EVAL_DIR,
        help=f"Directory with step_*_{{student,teacher}}_metrics.json (default: {DEFAULT_EVAL_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for plots and CSV (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--steps",
        default=",".join(str(step) for step in DEFAULT_STEPS),
        help="Comma-separated training steps to plot (default: 1,11,...,191)",
    )
    parser.add_argument(
        "--title",
        default="Eval replay test scores (student vs teacher)",
        help="Plot title",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eval_dir = os.path.abspath(args.eval_dir)
    output_dir = os.path.abspath(args.output_dir)
    steps = parse_steps(args.steps)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading eval metrics from {eval_dir}")
    print(f"Steps: {steps}")
    ordered_steps, scores, mask_counts = collect_scores_and_mask_counts(eval_dir, steps)
    combined_scores, combined_mask_counts = build_combined_series(scores, mask_counts)

    csv_path = os.path.join(output_dir, "eval_test_scores.csv")
    plot_path = os.path.join(output_dir, "eval_test_scores.png")
    write_csv(csv_path, ordered_steps, scores, mask_counts, combined_scores, combined_mask_counts)
    plot_test_scores(
        plot_path,
        ordered_steps,
        scores,
        mask_counts,
        combined_scores,
        combined_mask_counts,
        args.title,
    )

    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote plot: {plot_path}")
    for dataset in DATASETS:
        student_last = scores["student"][dataset][-1]
        teacher_last = scores["teacher"][dataset][-1]
        mask_last = mask_counts[dataset][-1]
        print(
            f"  {dataset} @ step {ordered_steps[-1]}: "
            f"student={student_last:.3f}, teacher={teacher_last:.3f}, "
            f"distillation_mask=1 count={mask_last}"
        )
    print(
        f"  combined @ step {ordered_steps[-1]}: "
        f"student={combined_scores['student'][-1]:.3f}, "
        f"teacher={combined_scores['teacher'][-1]:.3f}, "
        f"distillation_mask=1 count={combined_mask_counts[-1]}"
    )


if __name__ == "__main__":
    main()
