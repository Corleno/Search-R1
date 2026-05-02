#!/usr/bin/env bash
set -euo pipefail

# Resolve repository root from script location.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Load conda and activate retriever environment.
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate retriever

# Retrieval assets (downloaded in previous steps).
SAVE_PATH="${REPO_ROOT}/data/wiki18_e5"
INDEX_FILE="${SAVE_PATH}/e5_Flat.index"
CORPUS_FILE="${SAVE_PATH}/wiki-18.jsonl"

if [[ ! -f "${INDEX_FILE}" ]]; then
  echo "Missing index file: ${INDEX_FILE}"
  exit 1
fi

if [[ ! -f "${CORPUS_FILE}" ]]; then
  echo "Missing corpus file: ${CORPUS_FILE}"
  exit 1
fi

# HTTP port (shared nodes may have root using :8000). Override: RETRIEVAL_PORT=8001 bash ...
RETRIEVAL_PORT="${RETRIEVAL_PORT:-8000}"

echo "Starting retriever server with:"
echo "  index:  ${INDEX_FILE}"
echo "  corpus: ${CORPUS_FILE}"
echo "  url:    http://127.0.0.1:${RETRIEVAL_PORT}/retrieve"

# --faiss_gpu requires faiss-gpu (see faiss.GpuMultipleClonerOptions in retrieval_server_fan.py).
FAISS_ARGS=()
if [[ "${USE_FAISS_GPU:-1}" == "1" ]]; then
  if python -c "import faiss, sys; sys.exit(0 if hasattr(faiss, 'GpuMultipleClonerOptions') else 1)" 2>/dev/null; then
    FAISS_ARGS=(--faiss_gpu)
    echo "  faiss:  GPU (--faiss_gpu)"
  else
    echo "[WARN] faiss has no GPU API (faiss-cpu or incomplete install). Starting CPU FAISS (slower)."
    echo "       To fix: conda activate retriever && conda install -y faiss-gpu=1.8.0 -c pytorch -c nvidia"
    echo "       To force CPU even with GPU faiss: USE_FAISS_GPU=0 bash $0"
  fi
else
  echo "  faiss:  CPU (USE_FAISS_GPU=0)"
fi

cd "${REPO_ROOT}"
python "${SCRIPT_DIR}/retrieval_server_fan.py" \
  --index_path "${INDEX_FILE}" \
  --corpus_path "${CORPUS_FILE}" \
  --topk 3 \
  --retriever_name e5 \
  --retriever_model intfloat/e5-base-v2 \
  --port "${RETRIEVAL_PORT}" \
  "${FAISS_ARGS[@]}"
