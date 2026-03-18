# NQ Search Visualization App

This folder contains a small Flask app (`app.py`) + static UI (`static/`) for browsing samples from the processed Natural Questions (NQ) parquet files.

## Prerequisites

- **Python**: 3.10+ recommended
- **System**: anything that can run Flask

## 1) Install Python dependencies

From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## 2) Prepare the NQ parquet data

The app expects:

- `train.parquet`
- `test.parquet`

By default it looks under:

- `/mnt/task_runtime/data/nq_search`

### Option A: Use the helper script (recommended)

From the repo root:

```bash
bash scripts/download_nq_data.sh
```

This downloads the necessary artifacts and produces:

- `/mnt/task_runtime/data/nq_search/train.parquet`
- `/mnt/task_runtime/data/nq_search/test.parquet`

### Option B: If you already have parquet files

Put `train.parquet` and `test.parquet` into a directory, e.g.:

```bash
mkdir -p /path/to/nq_search
cp train.parquet test.parquet /path/to/nq_search/
```

## 3) Run the visualization server

From the repo root:

```bash
PORT=8000 python3 scripts/nq_search_viz/app.py
```

Then open:

- `http://127.0.0.1:8000/`

## 4) Point the UI at a custom data directory (optional)

If your parquet files are not in the default location, you can override the directory via a query param:

- `http://127.0.0.1:8000/?data_dir=/path/to/nq_search`

The API endpoints accept the same parameter, for example:

- `http://127.0.0.1:8000/api/samples?split=train&n=20&data_dir=/path/to/nq_search`

## Troubleshooting

- **FileNotFoundError: Missing parquet file**
  - Ensure your `data_dir` contains both `train.parquet` and `test.parquet`.
- **Port already in use**
  - Pick another port, e.g. `PORT=8001 python3 scripts/nq_search_viz/app.py`.
- **`golden_answers` shows up empty**
  - Verify your parquet schema includes a top-level `golden_answers` list column.
  - If you’re using the repo’s processing script (`scripts/data_process/nq_search.py`), it should be present.

