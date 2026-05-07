#!/usr/bin/env bash
# Sanity checks for local dense retriever setup and optional HTTP probe.
#
# Run with the same Python env you use for the server (e.g. after `conda activate retriever`).
#
# Env:
#   RETRIEVAL_SAVE_PATH   Directory with e5_Flat.index and wiki-18.jsonl (default: auto; see below)
#   RETRIEVAL_PORT        Port for /docs probe (default: 8000)
#   RETRIEVAL_HOST        Host for probe (default: 127.0.0.1)
#
# If RETRIEVAL_SAVE_PATH is unset, pick the first layout that contains e5_Flat.index:
#   /mnt/task_runtime/data/wiki18_e5, then /mnt/task_runtime/data, else REPO_ROOT/data/wiki18_e5.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ -z "${RETRIEVAL_SAVE_PATH:-}" ]]; then
  if [[ -f "/mnt/task_runtime/data/wiki18_e5/e5_Flat.index" ]]; then
    RETRIEVAL_SAVE_PATH="/mnt/task_runtime/data/wiki18_e5"
  elif [[ -f "/mnt/task_runtime/data/e5_Flat.index" ]]; then
    RETRIEVAL_SAVE_PATH="/mnt/task_runtime/data"
  else
    RETRIEVAL_SAVE_PATH="${REPO_ROOT}/data/wiki18_e5"
  fi
fi
INDEX_FILE="${RETRIEVAL_SAVE_PATH}/e5_Flat.index"
CORPUS_FILE="${RETRIEVAL_SAVE_PATH}/wiki-18.jsonl"
RETRIEVAL_PORT="${RETRIEVAL_PORT:-8000}"
RETRIEVAL_HOST="${RETRIEVAL_HOST:-127.0.0.1}"

echo "[1/4] Checking files..."
[[ -f "${INDEX_FILE}" ]] && echo "  OK index:  ${INDEX_FILE}" || { echo "  MISSING index: ${INDEX_FILE}"; exit 1; }
[[ -f "${CORPUS_FILE}" ]] && echo "  OK corpus: ${CORPUS_FILE}" || { echo "  MISSING corpus: ${CORPUS_FILE}"; exit 1; }

echo "[2/4] Checking Python packages..."
python - <<'PY'
import importlib
pkgs = ["torch", "faiss", "transformers", "fastapi", "uvicorn", "datasets"]
missing = []
for p in pkgs:
    try:
        importlib.import_module(p)
    except Exception:
        missing.append(p)
if missing:
    raise SystemExit(f"Missing packages: {missing}")
print("  OK packages:", ", ".join(pkgs))
PY

echo "[3/4] Checking CUDA status..."
python - <<'PY'
import torch
print("  torch:", torch.__version__)
print("  cuda built:", torch.version.cuda)
print("  cuda available:", torch.cuda.is_available())
PY

echo "[4/4] Checking retriever endpoint..."
if curl -s -m 2 "http://${RETRIEVAL_HOST}:${RETRIEVAL_PORT}/docs" >/dev/null 2>&1; then
  echo "  Retriever server is RUNNING at http://${RETRIEVAL_HOST}:${RETRIEVAL_PORT}"
else
  echo "  Retriever server is NOT running at http://${RETRIEVAL_HOST}:${RETRIEVAL_PORT} (expected if not started yet)."
fi

echo "Health check finished."
