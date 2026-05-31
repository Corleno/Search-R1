#!/usr/bin/env bash
# Download Search-R1 training and retrieval data under ./data.
#
# Downloads:
#   1. NQ + HotpotQA training set  ->  data/nq_hotpotqa_train/
#   2. Wiki-18 e5 index + corpus  ->  data/wiki18_e5/
#
# See README.md Quick start and scripts/nq_hotpotqa/README.md.
#
# Usage:
#   bash scripts/download_data.sh [OPTIONS]
#
# Options:
#   --train-only       Download only nq_hotpotqa_train
#   --retrieval-only   Download only wiki-18 index and corpus
#   -h, --help         Show this help
#
# Environment variables:
#   DATA_DIR           Root data directory (default: <repo>/data)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data}"
TRAIN_DIR="$DATA_DIR/nq_hotpotqa_train"
RETRIEVAL_DIR="$DATA_DIR/wiki18_e5"

DOWNLOAD_TRAIN=1
DOWNLOAD_RETRIEVAL=1

usage() {
  cat <<EOF
Usage: bash scripts/download_data.sh [OPTIONS]

Download Search-R1 data under ./data (override with DATA_DIR):

  data/nq_hotpotqa_train/   NQ + HotpotQA training parquet (Hugging Face dataset)
  data/wiki18_e5/           e5 flat index + wiki-18 corpus for local retrieval

Options:
  --train-only       Download only the NQ+HotpotQA training dataset
  --retrieval-only   Download only the wiki-18 index and corpus
  -h, --help         Show this help

Environment variables:
  DATA_DIR           Root data directory (default: $PROJECT_ROOT/data)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --train-only)
      DOWNLOAD_TRAIN=1
      DOWNLOAD_RETRIEVAL=0
      shift
      ;;
    --retrieval-only)
      DOWNLOAD_TRAIN=0
      DOWNLOAD_RETRIEVAL=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

download_hf_dataset() {
  local repo_id="$1"
  local local_dir="$2"

  if command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download --repo-type dataset "$repo_id" --local-dir "$local_dir"
  else
    python - <<PY
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="${repo_id}",
    repo_type="dataset",
    local_dir="${local_dir}",
)
PY
  fi
}

echo "==> Project root: $PROJECT_ROOT"
echo "==> Data directory: $DATA_DIR"
mkdir -p "$DATA_DIR"

if [[ "$DOWNLOAD_RETRIEVAL" -eq 1 ]]; then
  echo "==> Downloading wiki-18 e5 index and corpus to: $RETRIEVAL_DIR"
  mkdir -p "$RETRIEVAL_DIR"
  cd "$PROJECT_ROOT"

  python scripts/download.py --save_path "$RETRIEVAL_DIR"

  if [[ ! -f "$RETRIEVAL_DIR/e5_Flat.index" ]]; then
    echo "==> Merging index parts into e5_Flat.index..."
    cat "$RETRIEVAL_DIR"/part_* > "$RETRIEVAL_DIR/e5_Flat.index"
  else
    echo "==> e5_Flat.index already exists; skipping merge."
  fi

  if [[ -f "$RETRIEVAL_DIR/wiki-18.jsonl.gz" && ! -f "$RETRIEVAL_DIR/wiki-18.jsonl" ]]; then
    echo "==> Decompressing wiki-18.jsonl.gz..."
    gzip -d -f "$RETRIEVAL_DIR/wiki-18.jsonl.gz"
  elif [[ -f "$RETRIEVAL_DIR/wiki-18.jsonl" ]]; then
    echo "==> wiki-18.jsonl already exists; skipping decompression."
  else
    echo "Missing wiki-18 corpus under $RETRIEVAL_DIR" >&2
    exit 1
  fi

  echo "    - Index:  $RETRIEVAL_DIR/e5_Flat.index"
  echo "    - Corpus: $RETRIEVAL_DIR/wiki-18.jsonl"
fi

if [[ "$DOWNLOAD_TRAIN" -eq 1 ]]; then
  echo "==> Downloading NQ+HotpotQA training data to: $TRAIN_DIR"
  mkdir -p "$TRAIN_DIR"
  download_hf_dataset "PeterJinGo/nq_hotpotqa_train" "$TRAIN_DIR"
  echo "    - Training data: $TRAIN_DIR"
fi

echo "==> Done."
