import os
import random
from typing import Any, Dict, List, Optional

import pandas as pd
import pyarrow.parquet as pq
from flask import Flask, jsonify, request, send_from_directory


DEFAULT_DATA_DIR = "/mnt/task_runtime/data/nq_search"


def _data_paths(data_dir: str) -> Dict[str, str]:
    return {
        "train": os.path.join(data_dir, "train.parquet"),
        "test": os.path.join(data_dir, "test.parquet"),
    }


def _read_df(path: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing parquet file: {path}")
    return pd.read_parquet(path, columns=columns)


def _available_columns(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    # IMPORTANT: use Arrow schema top-level names. The Parquet schema view
    # (`pf.schema.names`) flattens nested types and can contain repeated field
    # names like "element", which breaks column presence checks.
    return pq.ParquetFile(path).schema_arrow.names


def _safe_len(x: Any) -> Optional[int]:
    try:
        return len(x)  # type: ignore[arg-type]
    except Exception:
        return None


def _jsonify_row(row: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in row.items():
        # Pandas can materialize Arrow list columns (e.g. `golden_answers`) as
        # numpy arrays; convert those (and similar array-likes) into plain lists
        # so the frontend receives JSON arrays instead of strings.
        if hasattr(v, "tolist") and not isinstance(v, (list, dict, str, int, float, bool)) and v is not None:
            try:
                v = v.tolist()  # type: ignore[assignment]
            except Exception:
                pass
        if isinstance(v, (list, dict, str, int, float, bool)) or v is None:
            out[k] = v
        else:
            out[k] = str(v)
    return out


def _summary_for(df: pd.DataFrame) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "rows": int(len(df)),
        "columns": list(df.columns),
    }

    if "ability" in df.columns:
        vc = df["ability"].value_counts(dropna=False)
        summary["ability_counts"] = {str(k): int(v) for k, v in vc.items()}

    if "data_source" in df.columns:
        vc = df["data_source"].value_counts(dropna=False)
        summary["data_source_counts"] = {str(k): int(v) for k, v in vc.items()}

    if "question" in df.columns:
        qlen = df["question"].astype(str).map(len)
        summary["question_length"] = {
            "min": int(qlen.min()),
            "p50": float(qlen.quantile(0.5)),
            "p90": float(qlen.quantile(0.9)),
            "max": int(qlen.max()),
            "mean": float(qlen.mean()),
        }

    if "golden_answers" in df.columns:
        alen = df["golden_answers"].map(_safe_len)
        alen = alen.dropna()
        if len(alen) > 0:
            summary["answers_per_question"] = {
                "min": int(alen.min()),
                "p50": float(alen.quantile(0.5)),
                "p90": float(alen.quantile(0.9)),
                "max": int(alen.max()),
                "mean": float(alen.mean()),
            }

    return summary


def create_app() -> Flask:
    app = Flask(__name__, static_folder="static", static_url_path="/")

    @app.get("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    @app.get("/api/summary")
    def api_summary():
        data_dir = request.args.get("data_dir", DEFAULT_DATA_DIR)
        paths = _data_paths(data_dir)
        train = _read_df(paths["train"], columns=["ability", "data_source", "question", "golden_answers"])
        test = _read_df(paths["test"], columns=["ability", "data_source", "question", "golden_answers"])
        return jsonify(
            {
                "data_dir": data_dir,
                "train": _summary_for(train),
                "test": _summary_for(test),
            }
        )

    @app.get("/api/samples")
    def api_samples():
        data_dir = request.args.get("data_dir", DEFAULT_DATA_DIR)
        split = request.args.get("split", "train")
        n = int(request.args.get("n", "20"))
        seed = request.args.get("seed")
        if seed is not None:
            random.seed(int(seed))

        paths = _data_paths(data_dir)
        if split not in paths:
            return jsonify({"error": f"Invalid split: {split}"}), 400

        cols = ["id", "question", "golden_answers", "data_source", "ability", "extra_info"]
        available = set(_available_columns(paths[split]))
        selected = [c for c in cols if c in available]
        df = _read_df(paths[split], columns=selected if selected else None)
        if len(df) == 0:
            return jsonify({"split": split, "rows": []})

        n = max(1, min(n, 200))
        idxs = random.sample(range(len(df)), k=min(n, len(df)))
        rows = [_jsonify_row(df.iloc[i].to_dict()) for i in idxs]
        return jsonify({"data_dir": data_dir, "split": split, "rows": rows})

    return app


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    create_app().run(host="0.0.0.0", port=port, debug=True)
