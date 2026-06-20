import glob
import json
import os
import re
from typing import Any, Dict, List

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_REPLAY_PATH = os.path.join(REPO_ROOT, "res", "train_replays", "exp_sdpo_searchr1_0620")
UPLOAD_DIR = os.environ.get(
    "JSONL_VIZ_UPLOAD_DIR",
    os.path.join(REPO_ROOT, "res", "jsonl_viz_uploads"),
)
ALLOWED_ROOTS = [
    REPO_ROOT,
    "/mnt/task_runtime",
]


def _allowed_roots() -> List[str]:
    extra = os.environ.get("JSONL_VIZ_ALLOWED_ROOTS", "")
    roots = list(ALLOWED_ROOTS)
    for item in extra.split(os.pathsep):
        item = item.strip()
        if item:
            roots.append(os.path.abspath(item))
    return list(dict.fromkeys(os.path.abspath(r) for r in roots))


def _resolve_safe_path(path: str) -> str:
    if not path:
        raise ValueError("Path is required")
    expanded = os.path.expanduser(path)
    if not os.path.isabs(expanded):
        expanded = os.path.join(REPO_ROOT, expanded)
    resolved = os.path.abspath(expanded)
    for root in _allowed_roots():
        if resolved == root or resolved.startswith(root + os.sep):
            return resolved
    raise ValueError(f"Path not allowed (must be under repo or JSONL_VIZ_ALLOWED_ROOTS): {path}")


def _step_sort_key(path: str) -> int:
    match = re.search(r"step_(\d+)\.jsonl$", os.path.basename(path))
    return int(match.group(1)) if match else 10**9


def _jsonl_files_for_path(path: str) -> List[str]:
    if os.path.isfile(path):
        if not path.endswith(".jsonl"):
            raise ValueError(f"Not a .jsonl file: {path}")
        return [path]

    if not os.path.isdir(path):
        raise FileNotFoundError(f"Path not found: {path}")

    step_files = sorted(glob.glob(os.path.join(path, "step_*.jsonl")), key=_step_sort_key)
    if step_files:
        return step_files

    jsonl_files = sorted(
        f for f in glob.glob(os.path.join(path, "*.jsonl")) if os.path.isfile(f)
    )
    if not jsonl_files:
        raise FileNotFoundError(f"No .jsonl files found in directory: {path}")
    return jsonl_files


def _load_jsonl_records(jsonl_path: str) -> List[dict]:
    records = []
    source_name = os.path.basename(jsonl_path)
    with open(jsonl_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {jsonl_path}: {exc}") from exc
            record["_source_file"] = source_name
            records.append(record)
    return records


def _load_replay_path(path: str) -> Dict[str, Any]:
    resolved = _resolve_safe_path(path)
    files = _jsonl_files_for_path(resolved)
    records: List[dict] = []
    for jsonl_path in files:
        records.extend(_load_jsonl_records(jsonl_path))
    return {
        "path": resolved,
        "files": [os.path.basename(f) for f in files],
        "record_count": len(records),
        "records": records,
    }


def _browse_dir(path: str) -> Dict[str, Any]:
    resolved = _resolve_safe_path(path)
    if not os.path.isdir(resolved):
        raise FileNotFoundError(f"Not a directory: {resolved}")

    entries = []
    for name in sorted(os.listdir(resolved)):
        full = os.path.join(resolved, name)
        if os.path.isdir(full):
            entries.append({"name": name, "path": full, "type": "dir"})
        elif name.endswith(".jsonl"):
            entries.append(
                {
                    "name": name,
                    "path": full,
                    "type": "file",
                    "size": os.path.getsize(full),
                }
            )

    parent = os.path.dirname(resolved)
    parent_entry = None
    if parent != resolved:
        try:
            _resolve_safe_path(parent)
            parent_entry = parent
        except ValueError:
            parent_entry = None

    return {
        "path": resolved,
        "parent": parent_entry,
        "entries": entries,
    }


def create_app() -> Flask:
    app = Flask(__name__, static_folder="static", static_url_path="/")

    @app.get("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    @app.get("/api/ping")
    def api_ping():
        return jsonify({"ok": True, "repo_root": REPO_ROOT, "upload_dir": UPLOAD_DIR})

    @app.get("/api/defaults")
    def api_defaults():
        return jsonify(
            {
                "default_replay_path": DEFAULT_REPLAY_PATH,
                "repo_root": REPO_ROOT,
                "upload_dir": UPLOAD_DIR,
            }
        )

    @app.get("/api/browse")
    def api_browse():
        path = request.args.get("path", REPO_ROOT)
        try:
            return jsonify(_browse_dir(path))
        except (ValueError, FileNotFoundError) as exc:
            return jsonify({"error": str(exc)}), 400

    @app.get("/api/load")
    def api_load():
        path = request.args.get("path")
        if not path:
            return jsonify({"error": "Missing path query parameter"}), 400
        try:
            payload = _load_replay_path(path)
            return jsonify(payload)
        except (ValueError, FileNotFoundError) as exc:
            return jsonify({"error": str(exc)}), 400

    @app.post("/api/upload")
    def api_upload():
        files = request.files.getlist("files")
        if not files:
            return jsonify({"error": "No files uploaded (use form field name 'files')"}), 400

        os.makedirs(UPLOAD_DIR, exist_ok=True)
        saved = []
        for f in files:
            if not f.filename:
                continue
            name = secure_filename(os.path.basename(f.filename))
            if not name.endswith(".jsonl"):
                return jsonify({"error": f"Only .jsonl files are allowed: {f.filename}"}), 400
            dest = os.path.join(UPLOAD_DIR, name)
            f.save(dest)
            saved.append(dest)

        if not saved:
            return jsonify({"error": "No valid files uploaded"}), 400

        records: List[dict] = []
        for path in sorted(saved):
            records.extend(_load_jsonl_records(path))

        return jsonify(
            {
                "upload_dir": UPLOAD_DIR,
                "files": [os.path.basename(p) for p in saved],
                "paths": saved,
                "record_count": len(records),
                "records": records,
            }
        )

    return app


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8001"))
    create_app().run(host="0.0.0.0", port=port, debug=True)
