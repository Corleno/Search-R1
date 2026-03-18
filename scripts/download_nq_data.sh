#!/usr/bin/env bash
# Download NQ data (indexing, corpus, and processed NQ dataset) to /mnt/task_runtime/data
# See README.md Quick start section.

set -euo pipefail

SAVE_PATH="${SAVE_PATH:-/mnt/task_runtime/data}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "==> NQ data will be saved to: $SAVE_PATH"
echo "==> Project root: $PROJECT_ROOT"

mkdir -p "$SAVE_PATH"
cd "$PROJECT_ROOT"

echo "==> (1) Downloading indexing and corpus from Hugging Face..."
python scripts/download.py --save_path "$SAVE_PATH"

echo "==> (2) Merging index parts into e5_Flat.index..."
cat "$SAVE_PATH"/part_* > "$SAVE_PATH/e5_Flat.index"

echo "==> (3) Decompressing wiki-18.jsonl.gz..."
gzip -d -f "$SAVE_PATH/wiki-18.jsonl.gz"

echo "==> (4) Processing NQ dataset to parquet..."
python scripts/data_process/nq_search.py --local_dir "$SAVE_PATH/nq_search"

echo "==> Done. NQ data is in $SAVE_PATH"
echo "    - Index: $SAVE_PATH/e5_Flat.index"
echo "    - Corpus: $SAVE_PATH/wiki-18.jsonl"
echo "    - Processed NQ: $SAVE_PATH/nq_search/train.parquet, $SAVE_PATH/nq_search/test.parquet"
