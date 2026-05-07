#!/usr/bin/env bash
# Thin wrapper around search_r1/search/retrieval_server.py (wiki-18 + e5 flat index).
#
# Env:
#   RETRIEVAL_SAVE_PATH  Directory containing e5_Flat.index and wiki-18.jsonl (default: auto; see scripts/retriever/check_retriever.sh)
#   RETRIEVAL_PORT       HTTP port (default: 8000)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"

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

retriever_name=e5
retriever_path=intfloat/e5-base-v2

if [[ ! -f "${INDEX_FILE}" ]]; then
  echo "Missing index: ${INDEX_FILE}"
  echo "Download the wiki-18 index (README Quick start step 1) and merge parts, or set:"
  echo "  RETRIEVAL_SAVE_PATH=/path/to/dir   # dir must contain e5_Flat.index and wiki-18.jsonl"
  exit 1
fi
if [[ ! -f "${CORPUS_FILE}" ]]; then
  echo "Missing corpus: ${CORPUS_FILE}"
  echo " gunzip the corpus in that directory (see README) or fix RETRIEVAL_SAVE_PATH."
  exit 1
fi

python search_r1/search/retrieval_server.py --index_path "$INDEX_FILE" \
                                            --corpus_path "$CORPUS_FILE" \
                                            --topk 3 \
                                            --retriever_name "$retriever_name" \
                                            --retriever_model "$retriever_path" \
                                            --port "${RETRIEVAL_PORT}" \
                                            --faiss_gpu
