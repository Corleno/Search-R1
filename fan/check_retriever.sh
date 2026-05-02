#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate retriever

SAVE_PATH="${REPO_ROOT}/data/wiki18_e5"
INDEX_FILE="${SAVE_PATH}/e5_Flat.index"
CORPUS_FILE="${SAVE_PATH}/wiki-18.jsonl"

echo "[1/4] Checking files..."
[[ -f "${INDEX_FILE}" ]] && echo "  OK index:  ${INDEX_FILE}" || { echo "  MISSING index: ${INDEX_FILE}"; exit 1; }
[[ -f "${CORPUS_FILE}" ]] && echo "  OK corpus: ${CORPUS_FILE}" || { echo "  MISSING corpus: ${CORPUS_FILE}"; exit 1; }

echo "[2/4] Checking Python packages..."
python - <<'PY'
import importlib
pkgs = ["torch", "faiss", "transformers", "fastapi", "uvicorn"]
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
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in retriever env.")
PY

echo "[4/4] Checking retriever endpoint..."
if curl -s -m 2 "http://127.0.0.1:8000/docs" >/dev/null; then
  echo "  Retriever server is RUNNING at http://127.0.0.1:8000"
else
  echo "  Retriever server is NOT running (expected if not started yet)."
fi

echo "Health check finished."
