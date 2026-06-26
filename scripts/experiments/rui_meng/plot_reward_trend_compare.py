#!/usr/bin/env python3
"""Plot training replay pass rates for multiple experiments on one figure."""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_OUTPUT_DIR = os.path.join(REPO_ROOT, "res", "train_replay_analysis", "compare")
DEFAULT_EXPERIMENTS = (
    (
        "fcsd",
        os.path.join(REPO_ROOT, "res", "train_replay_analysis", "exp_sdpo_searchr1_0620"),
    ),
    (
        "fcsd-ppo",
        os.path.join(
            REPO_ROOT,
            "res",
            "train_replay_analysis",
            "nq_sdpo-qwen2.5-3b-em-ppoclip-replay",
        ),
    ),
)
PANELS = ("combined", "nq", "hotpotqa")
PANEL_LABELS = {
    "combined": "Overall",
    "nq": "NQ",
    "hotpotqa": "HotpotQA",
}
EXPERIMENT_STYLES = (
    {"color": "tab:blue", "marker": "o", "linestyle": "-"},
    {"color": "tab:orange", "marker": "s", "linestyle": "-"},
)


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


def load_summary(analysis_dir: str) -> dict:
    path = os.path.join(analysis_dir, "reward_summary.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Missing {path}. Run analyze_reward_trend.sh for this experiment first."
        )
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def extract_series(
    summary: dict,
    dataset: str,
    step_min: Optional[int],
    step_max: Optional[int],
) -> Tuple[List[int], List[float]]:
    steps: List[int] = []
    values: List[float] = []
    for row in summary.get("per_step", []):
        step = int(row["step"])
        if step_min is not None and step < step_min:
            continue
        if step_max is not None and step > step_max:
            continue
        if dataset == "combined":
            value = float(row["pass_rate"])
        else:
            source_stats = row.get("by_source", {}).get(dataset)
            if not source_stats:
                continue
            value = float(source_stats["pass_rate"])
        steps.append(step)
        values.append(value)
    return steps, values


def parse_experiments(raw: str) -> List[Tuple[str, str]]:
    experiments: List[Tuple[str, str]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                f"Invalid experiment spec {item!r}; expected LABEL=ANALYSIS_DIR"
            )
        label, analysis_dir = item.split("=", 1)
        experiments.append((label.strip(), analysis_dir.strip()))
    if len(experiments) < 2:
        raise ValueError("At least two experiments are required.")
    return experiments


def default_experiments_arg() -> str:
    return ",".join(f"{label}={analysis_dir}" for label, analysis_dir in DEFAULT_EXPERIMENTS)


def write_compare_csv(
    output_path: str,
    experiments: Sequence[Tuple[str, dict]],
    step_min: Optional[int],
    step_max: Optional[int],
) -> None:
    fieldnames = ["step"]
    for label, _ in experiments:
        for panel in PANELS:
            fieldnames.append(f"{label}_{panel}_pass_rate")

    rows_by_step: Dict[int, Dict[str, float]] = {}
    for label, summary in experiments:
        for panel in PANELS:
            steps, values = extract_series(summary, panel, step_min, step_max)
            for step, value in zip(steps, values):
                rows_by_step.setdefault(step, {})[f"{label}_{panel}_pass_rate"] = value

    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for step in sorted(rows_by_step):
            row = {"step": step}
            row.update(rows_by_step[step])
            writer.writerow(row)


def plot_compare(
    output_path: str,
    experiments: Sequence[Tuple[str, dict]],
    rolling_window: int,
    plot_x_min: Optional[int],
    plot_x_max: Optional[int],
    show_rolling: bool,
) -> None:
    fig, axes = plt.subplots(1, len(PANELS), figsize=(18, 5), sharex=True)

    for ax, panel in zip(axes, PANELS):
        for idx, (label, summary) in enumerate(experiments):
            steps, values = extract_series(summary, panel, plot_x_min, plot_x_max)
            if not steps:
                continue
            style = EXPERIMENT_STYLES[idx % len(EXPERIMENT_STYLES)]
            ax.plot(
                steps,
                values,
                label=label,
                linewidth=1.6,
                markersize=4,
                alpha=0.85,
                **style,
            )
            if show_rolling:
                rolling = rolling_mean(values, rolling_window)
                ax.plot(
                    steps,
                    rolling,
                    linewidth=2.0,
                    alpha=0.9,
                    color=style["color"],
                    linestyle="--",
                    label=f"{label} rolling (w={rolling_window})",
                )

        ax.set_title(PANEL_LABELS[panel])
        ax.set_xlabel("Training step")
        ax.set_ylabel("Pass rate")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    if plot_x_min is not None and plot_x_max is not None:
        for ax in axes:
            ax.set_xlim(plot_x_min, plot_x_max)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiments",
        default=default_experiments_arg(),
        help=(
            "Comma-separated LABEL=ANALYSIS_DIR entries "
            f"(default: {default_experiments_arg()})"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for comparison plot and CSV (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=10,
        help="Window size for optional rolling mean overlay (default: 10)",
    )
    parser.add_argument(
        "--show-rolling",
        action="store_true",
        help="Overlay rolling mean lines for each experiment",
    )
    parser.add_argument(
        "--plot-x-min",
        type=int,
        default=0,
        help="Minimum training step for x-axis (default: 0)",
    )
    parser.add_argument(
        "--plot-x-max",
        type=int,
        default=200,
        help="Maximum training step for x-axis (default: 200)",
    )
    parser.add_argument(
        "--no-plot-xlim",
        action="store_true",
        help="Do not fix the x-axis range",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    experiment_specs = parse_experiments(args.experiments)
    loaded: List[Tuple[str, dict]] = []
    for label, analysis_dir in experiment_specs:
        analysis_dir = os.path.abspath(analysis_dir)
        summary = load_summary(analysis_dir)
        loaded.append((label, summary))
        print(f"Loaded {label} from {analysis_dir} ({summary.get('num_steps', 0)} steps)")

    plot_x_min = None if args.no_plot_xlim else args.plot_x_min
    plot_x_max = None if args.no_plot_xlim else args.plot_x_max

    plot_path = os.path.join(output_dir, "reward_trend_compare.png")
    csv_path = os.path.join(output_dir, "reward_trend_compare.csv")
    plot_compare(
        plot_path,
        loaded,
        args.rolling_window,
        plot_x_min,
        plot_x_max,
        args.show_rolling,
    )
    write_compare_csv(csv_path, loaded, plot_x_min, plot_x_max)

    print(f"Wrote comparison plot: {plot_path}")
    print(f"Wrote comparison CSV: {csv_path}")


if __name__ == "__main__":
    main()
