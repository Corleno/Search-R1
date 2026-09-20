#!/usr/bin/env python3
"""Analyze training collapse signals from replay JSONL files.

Measures (per step):
  1. Trajectory similarity across different questions
  2. Search-query diversity
  3. Input sensitivity (question↔output overlap and same−diff question traj gap)
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
STEP_RE = re.compile(r"step_(\d+)\.jsonl$")

SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.DOTALL | re.IGNORECASE)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
INFO_RE = re.compile(r"<information>.*?</information>", re.DOTALL | re.IGNORECASE)
IM_ASSIST_RE = re.compile(
    r"<\|im_start\|>assistant\n(.*?)(?:<\|im_end\|>|$)",
    re.DOTALL,
)

# Env boilerplate embeds literal tags; neutralize before query extraction.
INSTR_QUERY_RE = re.compile(
    r"put the query between <search> and </search>",
    re.IGNORECASE,
)
INSTR_ANSWER_RE = re.compile(
    r"put the answer between <answer> and </answer>",
    re.IGNORECASE,
)

PLACEHOLDER_QUERIES = frozenset({"query", "and"})
DEGENERATE_SNIPPETS = (
    "my previous action is invalid",
    "put the query between",
    "let me try again",
)

STOPWORDS = frozenset(
    "the a an of to in on for and or is was were be by with from who what when "
    "where which how many do does did are am it its this that these those".split()
)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def extract_assistant_text(trajectory: str) -> str:
    """Return assistant-only spans from a chat trajectory."""
    parts = IM_ASSIST_RE.findall(trajectory or "")
    if parts:
        return "\n".join(parts)
    return trajectory or ""


def clean_assistant_text(trajectory: str) -> str:
    """Strip retrieval blocks and neutralize env boilerplate tag phrases."""
    text = extract_assistant_text(trajectory)
    text = INFO_RE.sub(" ", text)
    text = INSTR_QUERY_RE.sub("put the query between SEARCH_TAGS", text)
    text = INSTR_ANSWER_RE.sub("put the answer between ANSWER_TAGS", text)
    return text


def extract_queries(text: str) -> List[str]:
    """Extract search queries, dropping placeholders and empties."""
    queries: List[str] = []
    for match in SEARCH_RE.findall(text or ""):
        query = " ".join(match.strip().split())
        if not query:
            continue
        if query.lower() in PLACEHOLDER_QUERIES:
            continue
        queries.append(query)
    return queries


def extract_answer(text: str) -> str:
    matches = ANSWER_RE.findall(text or "")
    if not matches:
        return ""
    return " ".join(matches[-1].strip().split())


def is_degenerate_query(query: str) -> bool:
    if len(query) < 2 or len(query) > 200:
        return True
    lower = query.lower()
    return any(snippet in lower for snippet in DEGENERATE_SNIPPETS)


def content_tokens(text: str) -> Set[str]:
    return {
        tok
        for tok in re.findall(r"[a-z0-9]+", (text or "").lower())
        if tok not in STOPWORDS and len(tok) > 2
    }


def char_ngrams(text: str, n: int = 4) -> Set[str]:
    cleaned = re.sub(r"\s+", " ", (text or "").lower()).strip()
    if not cleaned:
        return set()
    if len(cleaned) < n:
        return {cleaned}
    return {cleaned[i : i + n] for i in range(len(cleaned) - n + 1)}


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def token_overlap(question_tokens: Set[str], other_tokens: Set[str]) -> float:
    if not question_tokens:
        return float("nan")
    return len(question_tokens & other_tokens) / len(question_tokens)


# ---------------------------------------------------------------------------
# Replay I/O (mirrors analyze_reward_trend.py)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Per-step metrics
# ---------------------------------------------------------------------------


@dataclass
class ParsedSample:
    question: str
    assistant_text: str
    queries: List[str]
    answer: str
    question_tokens: Set[str] = field(repr=False)
    _traj_ngrams: Optional[Set[str]] = field(default=None, repr=False)

    @property
    def traj_ngrams(self) -> Set[str]:
        if self._traj_ngrams is None:
            self._traj_ngrams = char_ngrams(self.assistant_text)
        return self._traj_ngrams


@dataclass
class StepCollapseStats:
    step: int
    count: int

    # Trajectory similarity
    mean_traj_sim: float
    frac_traj_sim_ge_0_3: float
    frac_traj_sim_ge_0_5: float
    answer_mode_share: float
    unique_answer_ratio: float

    # Query diversity
    search_rate: float
    mean_queries: float
    first_query_unique_ratio: float
    first_query_entropy: float
    first_query_mode_share: float
    degenerate_query_rate: float
    query_type_token_ratio: float

    # Input sensitivity
    question_query_overlap: float
    question_answer_overlap: float
    # Same-question vs different-question traj similarity (low-variance gap).
    # traj_sim_high_q_sim = same-question pairs; traj_sim_low_q_sim = different-question pairs.
    traj_sim_low_q_sim: float
    traj_sim_high_q_sim: float
    conditioned_gap: float
    duplicate_question_traj_sim: float
    n_duplicate_question_pairs: int
    n_diff_question_pairs: int

    # Pair sampling diagnostics
    n_pair_samples: int
    n_pairs: int

    def to_dict(self) -> dict:
        return asdict(self)


def _mean(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return sum(values) / len(values)


def _shannon_entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log(p, 2)
    return entropy


def parse_sample(record: dict) -> ParsedSample:
    question = str(record.get("question", "") or "")
    text = clean_assistant_text(str(record.get("trajectory", "") or ""))
    queries = extract_queries(text)
    answer = extract_answer(text)
    return ParsedSample(
        question=question,
        assistant_text=text,
        queries=queries,
        answer=answer,
        question_tokens=content_tokens(question),
    )


def analyze_step(
    step: int,
    records: Sequence[dict],
    sample_size: int,
    pair_sample_size: int,
    rng: random.Random,
) -> StepCollapseStats:
    samples = [parse_sample(record) for record in records]
    n = len(samples)
    if n == 0:
        raise ValueError(f"No records for step {step}")

    # --- Query diversity (full step) ---
    first_queries: List[str] = []
    all_queries: List[str] = []
    n_with_search = 0
    n_queries_total = 0
    n_degenerate = 0
    q_overlaps: List[float] = []
    a_overlaps: List[float] = []
    answers: List[str] = []

    for sample in samples:
        answers.append(sample.answer)
        n_queries_total += len(sample.queries)
        if sample.queries:
            n_with_search += 1
            first_queries.append(sample.queries[0])
            all_queries.extend(sample.queries)
            for query in sample.queries:
                if is_degenerate_query(query):
                    n_degenerate += 1
            query_tokens = content_tokens(" ".join(sample.queries))
            overlap = token_overlap(sample.question_tokens, query_tokens)
            if not math.isnan(overlap):
                q_overlaps.append(overlap)
        answer_tokens = content_tokens(sample.answer)
        a_overlap = token_overlap(sample.question_tokens, answer_tokens)
        if not math.isnan(a_overlap):
            a_overlaps.append(a_overlap if answer_tokens else 0.0)

    first_counter = Counter(first_queries)
    first_total = sum(first_counter.values())
    answer_counter = Counter(answers)
    answer_mode_share = (
        answer_counter.most_common(1)[0][1] / n if answer_counter else 0.0
    )
    unique_answer_ratio = len(set(answers)) / n

    search_rate = n_with_search / n
    mean_queries = n_queries_total / n
    first_query_unique_ratio = len(set(first_queries)) / max(len(first_queries), 1)
    first_query_entropy = _shannon_entropy(first_counter)
    first_query_mode_share = (
        first_counter.most_common(1)[0][1] / first_total if first_total else 0.0
    )
    degenerate_query_rate = n_degenerate / max(n_queries_total, 1)
    query_type_token_ratio = len(set(all_queries)) / max(len(all_queries), 1)

    # --- Pairwise trajectory similarity (subsample of rows) ---
    idxs = list(range(n))
    if sample_size > 0 and len(idxs) > sample_size:
        idxs = rng.sample(idxs, sample_size)
    n_pair_samples = len(idxs)

    pairs: List[Tuple[int, int]] = [
        (i, j) for i in range(n_pair_samples) for j in range(i + 1, n_pair_samples)
    ]
    if pair_sample_size > 0 and len(pairs) > pair_sample_size:
        pairs = rng.sample(pairs, pair_sample_size)

    traj_sims: List[float] = []
    diff_q_traj: List[float] = []
    for i, j in pairs:
        si = samples[idxs[i]]
        sj = samples[idxs[j]]
        t_sim = jaccard(si.traj_ngrams, sj.traj_ngrams)
        traj_sims.append(t_sim)
        if si.question != sj.question:
            diff_q_traj.append(t_sim)

    mean_traj_sim = _mean(traj_sims)
    frac_ge_03 = (
        sum(1 for sim in traj_sims if sim >= 0.3) / len(traj_sims) if traj_sims else float("nan")
    )
    frac_ge_05 = (
        sum(1 for sim in traj_sims if sim >= 0.5) / len(traj_sims) if traj_sims else float("nan")
    )

    # Same-question pairs: all rollout pairs per question (512 * C(5,2) ≈ 5120).
    # Low-variance "matched input" bin — exact duplicate questions in the batch.
    by_question: Dict[str, List[ParsedSample]] = defaultdict(list)
    for sample in samples:
        by_question[sample.question].append(sample)
    same_q_traj: List[float] = []
    for group in by_question.values():
        if len(group) < 2:
            continue
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                same_q_traj.append(jaccard(group[i].traj_ngrams, group[j].traj_ngrams))

    traj_sim_same = _mean(same_q_traj)
    traj_sim_diff = _mean(diff_q_traj)
    if math.isnan(traj_sim_same) or math.isnan(traj_sim_diff):
        conditioned_gap = float("nan")
    else:
        # Positive: same input → more similar traj than different inputs (input-sensitive).
        # Collapse: different-question traj_sim rises toward same-question → gap → 0.
        conditioned_gap = traj_sim_same - traj_sim_diff

    return StepCollapseStats(
        step=step,
        count=n,
        mean_traj_sim=mean_traj_sim,
        frac_traj_sim_ge_0_3=frac_ge_03,
        frac_traj_sim_ge_0_5=frac_ge_05,
        answer_mode_share=answer_mode_share,
        unique_answer_ratio=unique_answer_ratio,
        search_rate=search_rate,
        mean_queries=mean_queries,
        first_query_unique_ratio=first_query_unique_ratio,
        first_query_entropy=first_query_entropy,
        first_query_mode_share=first_query_mode_share,
        degenerate_query_rate=degenerate_query_rate,
        query_type_token_ratio=query_type_token_ratio,
        question_query_overlap=_mean(q_overlaps),
        question_answer_overlap=_mean(a_overlaps),
        traj_sim_low_q_sim=traj_sim_diff,
        traj_sim_high_q_sim=traj_sim_same,
        conditioned_gap=conditioned_gap,
        duplicate_question_traj_sim=traj_sim_same,
        n_duplicate_question_pairs=len(same_q_traj),
        n_diff_question_pairs=len(diff_q_traj),
        n_pair_samples=n_pair_samples,
        n_pairs=len(pairs),
    )


def collect_collapse_stats(
    files: Sequence[str],
    sample_size: int,
    pair_sample_size: int,
    seed: int,
) -> List[StepCollapseStats]:
    per_step: List[StepCollapseStats] = []
    total = len(files)
    start_time = time.time()
    for idx, path in enumerate(files, start=1):
        match = STEP_RE.search(os.path.basename(path))
        step = int(match.group(1)) if match else -1
        records = list(iter_records(path))
        # Independent RNG per step so stride/filter changes don't reshuffle later steps.
        rng = random.Random(seed + step)
        per_step.append(
            analyze_step(step, records, sample_size, pair_sample_size, rng)
        )
        _print_progress(idx, total, step, len(records), start_time)
    if total:
        sys.stdout.write("\n")
        sys.stdout.flush()
    return per_step


def _print_progress(
    done: int,
    total: int,
    step: int,
    n_samples: int,
    start_time: float,
) -> None:
    """In-place progress line for step-level analysis."""
    elapsed = max(time.time() - start_time, 1e-6)
    rate = done / elapsed
    remaining = (total - done) / rate if rate > 0 else float("inf")
    pct = 100.0 * done / total if total else 100.0
    bar_width = 28
    filled = int(bar_width * done / total) if total else bar_width
    bar = "#" * filled + "-" * (bar_width - filled)
    eta = _format_duration(remaining)
    elapsed_s = _format_duration(elapsed)
    sys.stdout.write(
        f"\r[{bar}] {done}/{total} ({pct:5.1f}%) "
        f"step={step} n={n_samples} "
        f"elapsed={elapsed_s} eta={eta}   "
    )
    sys.stdout.flush()


def _format_duration(seconds: float) -> str:
    if seconds == float("inf") or seconds != seconds:  # nan
        return "?"
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "step",
    "count",
    "mean_traj_sim",
    "frac_traj_sim_ge_0_3",
    "frac_traj_sim_ge_0_5",
    "answer_mode_share",
    "unique_answer_ratio",
    "search_rate",
    "mean_queries",
    "first_query_unique_ratio",
    "first_query_entropy",
    "first_query_mode_share",
    "degenerate_query_rate",
    "query_type_token_ratio",
    "question_query_overlap",
    "question_answer_overlap",
    "traj_sim_low_q_sim",
    "traj_sim_high_q_sim",
    "conditioned_gap",
    "duplicate_question_traj_sim",
    "n_duplicate_question_pairs",
    "n_diff_question_pairs",
    "n_pair_samples",
    "n_pairs",
]


def write_csv(path: str, per_step: Sequence[StepCollapseStats]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for stats in per_step:
            row = stats.to_dict()
            writer.writerow({key: row[key] for key in CSV_FIELDS})


def plot_collapse_trend(
    output_path: str,
    per_step: Sequence[StepCollapseStats],
    plot_x_min: Optional[int],
    plot_x_max: Optional[int],
) -> None:
    steps = [s.step for s in per_step]
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Panel 1: trajectory similarity
    ax = axes[0]
    ax.plot(steps, [s.mean_traj_sim for s in per_step], marker="o", markersize=3, label="mean traj sim")
    ax.plot(
        steps,
        [s.frac_traj_sim_ge_0_5 for s in per_step],
        marker="o",
        markersize=3,
        label="frac traj-sim ≥ 0.5",
    )
    ax.plot(
        steps,
        [s.answer_mode_share for s in per_step],
        marker="o",
        markersize=3,
        label="answer mode share",
    )
    ax.set_ylabel("Similarity / share")
    ax.set_title("Trajectory similarity across questions")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    # Panel 2: query diversity
    ax = axes[1]
    ax.plot(steps, [s.search_rate for s in per_step], marker="o", markersize=3, label="search rate")
    ax.plot(
        steps,
        [s.degenerate_query_rate for s in per_step],
        marker="o",
        markersize=3,
        label="degenerate query rate",
    )
    ax.set_ylabel("Rate")
    ax2 = ax.twinx()
    ax2.plot(
        steps,
        [s.first_query_entropy for s in per_step],
        color="tab:green",
        marker="o",
        markersize=3,
        label="first-query entropy",
    )
    ax2.set_ylabel("Entropy (bits)", color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green")
    ax.set_title("Query diversity")
    ax.grid(True, alpha=0.3)
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], loc="best", fontsize=8)

    # Panel 3: input sensitivity
    ax = axes[2]
    ax.plot(
        steps,
        [s.question_query_overlap for s in per_step],
        marker="o",
        markersize=3,
        label="question↔query overlap",
    )
    ax.plot(
        steps,
        [s.question_answer_overlap for s in per_step],
        marker="o",
        markersize=3,
        label="question↔answer overlap",
    )
    ax.plot(
        steps,
        [s.conditioned_gap for s in per_step],
        marker="o",
        markersize=3,
        label="same−diff question traj gap",
    )
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Overlap / gap")
    ax.set_title("Input sensitivity")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    if plot_x_min is not None and plot_x_max is not None:
        axes[0].set_xlim(plot_x_min, plot_x_max)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def build_summary(
    replay_dir: str,
    per_step: Sequence[StepCollapseStats],
    config: dict,
) -> dict:
    return {
        "replay_dir": replay_dir,
        "num_steps": len(per_step),
        "num_samples": sum(s.count for s in per_step),
        "config": config,
        "per_step": [s.to_dict() for s in per_step],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", required=True, help="Directory with step_*.jsonl replays")
    parser.add_argument("--output-dir", required=True, help="Directory for plots and summary files")
    parser.add_argument("--step-min", type=int, default=None)
    parser.add_argument("--step-max", type=int, default=None)
    parser.add_argument("--step-stride", type=int, default=1)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=256,
        help="Records subsampled per step for pairwise traj similarity (default: 256)",
    )
    parser.add_argument(
        "--pair-sample-size",
        type=int,
        default=2000,
        help="Max random pairs per step (default: 2000)",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for sampling")
    parser.add_argument("--plot-x-min", type=int, default=0)
    parser.add_argument("--plot-x-max", type=int, default=200)
    parser.add_argument("--no-plot-xlim", action="store_true")
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

    print(f"Analyzing collapse on {len(files)} replay files from {replay_dir}")
    per_step = collect_collapse_stats(
        files,
        sample_size=args.sample_size,
        pair_sample_size=args.pair_sample_size,
        seed=args.seed,
    )

    config = {
        "step_min": args.step_min,
        "step_max": args.step_max,
        "step_stride": args.step_stride,
        "sample_size": args.sample_size,
        "pair_sample_size": args.pair_sample_size,
        "seed": args.seed,
        "conditioned_gap": "same_question_traj_sim - diff_question_traj_sim",
    }
    summary = build_summary(replay_dir, per_step, config)
    summary_path = os.path.join(output_dir, "collapse_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    csv_path = os.path.join(output_dir, "collapse_metrics.csv")
    write_csv(csv_path, per_step)

    plot_path = os.path.join(output_dir, "collapse_trend.png")
    plot_x_min = None if args.no_plot_xlim else args.plot_x_min
    plot_x_max = None if args.no_plot_xlim else args.plot_x_max
    plot_collapse_trend(plot_path, per_step, plot_x_min, plot_x_max)

    print(f"Wrote summary: {summary_path}")
    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote plot: {plot_path}")

    # Print a few milestone steps for quick sanity check
    milestones = {1, 20, 40, 50, 100, 150, 190}
    for stats in per_step:
        if stats.step not in milestones:
            continue
        print(
            f"  step {stats.step:3d}: traj_sim={stats.mean_traj_sim:.3f} "
            f"search_rate={stats.search_rate:.3f} "
            f"q↔query={stats.question_query_overlap:.3f} "
            f"gap={stats.conditioned_gap:.3f} "
            f"degen={stats.degenerate_query_rate:.3f}"
        )


if __name__ == "__main__":
    main()
