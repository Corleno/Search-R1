#!/usr/bin/env bash
# Start dense e5 + wiki-18 flat-index retriever (upstream server).
#
# Prerequisites: conda/virtualenv with torch, faiss, transformers, fastapi, uvicorn, datasets
# (see README "Retriever environment").
#
# Env:
#   RETRIEVAL_SAVE_PATH  Directory with e5_Flat.index and wiki-18.jsonl (default: auto; see check_retriever.sh)
#   RETRIEVAL_PORT       HTTP port (default: 8000)
#   USE_FAISS_GPU        If 1 (default): use --faiss_gpu when GPU faiss is available; if 0: CPU faiss only
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

if [[ ! -f "${INDEX_FILE}" ]]; then
  echo "Missing index file: ${INDEX_FILE}"
  echo "Set RETRIEVAL_SAVE_PATH to the directory containing e5_Flat.index and wiki-18.jsonl."
  exit 1
fi

if [[ ! -f "${CORPUS_FILE}" ]]; then
  echo "Missing corpus file: ${CORPUS_FILE}"
  exit 1
fi

echo "Starting retriever server with:"
echo "  index:  ${INDEX_FILE}"
echo "  corpus: ${CORPUS_FILE}"
echo "  url:    http://127.0.0.1:${RETRIEVAL_PORT}/retrieve"

FAISS_ARGS=()
if [[ "${USE_FAISS_GPU:-1}" == "1" ]]; then
  if python -c "import faiss, sys; sys.exit(0 if hasattr(faiss, 'GpuMultipleClonerOptions') else 1)" 2>/dev/null; then
    FAISS_ARGS=(--faiss_gpu)
    echo "  faiss:  GPU (--faiss_gpu)"
  else
    echo "[WARN] faiss has no GPU API (faiss-cpu or incomplete install). Starting CPU FAISS (slower)."
    echo "       Install faiss-gpu (README retriever env) or set USE_FAISS_GPU=0 to silence this."
  fi
else
  echo "  faiss:  CPU (USE_FAISS_GPU=0)"
fi

cd "${REPO_ROOT}"
python "${REPO_ROOT}/search_r1/search/retrieval_server.py" \
  --index_path "${INDEX_FILE}" \
  --corpus_path "${CORPUS_FILE}" \
  --topk 3 \
  --retriever_name e5 \
  --retriever_model intfloat/e5-base-v2 \
  --port "${RETRIEVAL_PORT}" \
  "${FAISS_ARGS[@]}"
