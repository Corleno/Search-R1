# JSONL Replay Viewer

A small Flask app (`app.py`) and static UI (`static/`) for browsing training replay JSONL files produced by SDPO / Search-R1 training runs.

Load replays from the server filesystem, browse directories, upload `.jsonl` files, or open files locally in the browser (offline fallback).

## Prerequisites

- **Python**: 3.10+ recommended
- **Dependencies**: Flask (included in repo `requirements.txt`)

## Install

From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Run

From the repo root:

```bash
PORT=8001 python3 scripts/jsonl_viz/app.py
```

Then open:

- `http://127.0.0.1:8001/`

The default port is `8001` (override with the `PORT` environment variable).

## Data sources

The viewer expects one JSON object per line. Typical fields include `question`, `prompt`, `teacher_prompt`, `trajectory`, `response`, `score`, `turns`, `valid_searches`, `step`, `data_source`, and `ground_truth`.

### Server path (recommended)

Enter a path relative to the repo root or an absolute path under an allowed root, then click **Load**:

- **Single file**: `res/train_replays/my_run/step_0001.jsonl`
- **Directory**: `res/train_replays/my_run` — loads all `step_*.jsonl` files in step order; if none match, loads all `*.jsonl` in the directory

Use **Browse** to navigate directories on the server and click a file or folder to load it.

Default path on startup: `res/train_replays/exp_sdpo_searchr1_0620`.

### Upload to server

Choose one or more `.jsonl` files and click **Upload & load**. Files are saved under `res/jsonl_viz_uploads/` by default.

### Local browser upload (offline)

Expand **Local browser upload** to pick files or a folder, or drag-and-drop `.jsonl` files. This works without the Flask backend (open `static/index.html` directly), but server browse/load is unavailable.

## URL parameters

- `?path=res/train_replays/my_run` — pre-fill the server path input
- `?path=...&autoload=1` — load that path on page load

Example:

- `http://127.0.0.1:8001/?path=res/train_replays/my_sdpo_run&autoload=1`

## UI overview

After loading data, the dashboard shows:

- **Summary KPIs**: record count, accuracy (score > 0), average turns, average searches
- **Score distribution** chart
- **Filterable table**: filter by question text, data source, training step, and score; sort by index, score, turns, or searches
- **Detail panel**: metadata, side-by-side `prompt` vs `teacher_prompt`, and highlighted trajectory (`<search>`, `<information>`, `<answer>`, `<think>`, etc.)

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8001` | HTTP port for the Flask server |
| `JSONL_VIZ_UPLOAD_DIR` | `res/jsonl_viz_uploads` | Directory for uploaded `.jsonl` files |
| `JSONL_VIZ_ALLOWED_ROOTS` | (empty) | Extra allowed filesystem roots, separated by `:` (Linux) or `;` (Windows) |

By default, paths must resolve under the repo root or `/mnt/task_runtime`. Use `JSONL_VIZ_ALLOWED_ROOTS` to permit additional directories.

## API endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/ping` | GET | Health check |
| `/api/defaults` | GET | Default replay path and directories |
| `/api/browse?path=...` | GET | List subdirectories and `.jsonl` files |
| `/api/load?path=...` | GET | Load a file or directory of JSONL records |
| `/api/upload` | POST | Upload `.jsonl` files (form field: `files`) |

## Related docs

Training replay files are written when `trainer.save_train_replay=true` is set during SDPO training. See [README_SDPO.md](../../README_SDPO.md) for replay format and export options.

## Troubleshooting

- **Path not allowed**
  - Use a path under the repo root, `/mnt/task_runtime`, or add the parent directory to `JSONL_VIZ_ALLOWED_ROOTS`.
- **No .jsonl files found**
  - Confirm the directory contains `step_*.jsonl` or other `.jsonl` files.
- **Port already in use**
  - Use another port: `PORT=8002 python3 scripts/jsonl_viz/app.py`.
- **Local upload works but server load does not**
  - The Flask server is not running. Start `app.py` or use the local upload fallback.
