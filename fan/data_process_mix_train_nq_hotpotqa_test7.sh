#!/usr/bin/env bash
set -euo pipefail

# Explicit paths for clarity.
PROJECT_ROOT="/nfshome/fayang/workspace/Search-R1"
OUT_DIR="/nfshome/fayang/workspace/Search-R1/data/mix_train_nq_hotpotqa_test7"

# Dataset mixture config.
TRAIN_SOURCES="nq,hotpotqa"  # 2 sources for training
TEST_SOURCES="nq,triviaqa,popqa,hotpotqa,2wikimultihopqa,musique,bamboogle"  # 7 sources for evaluation

mkdir -p "${OUT_DIR}"

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "OUT_DIR=${OUT_DIR}"
echo "TRAIN_SOURCES=${TRAIN_SOURCES}"
echo "TEST_SOURCES=${TEST_SOURCES}"

python "${PROJECT_ROOT}/scripts/data_process/qa_search_train_merge.py" \
  --local_dir "${OUT_DIR}" \
  --data_sources "${TRAIN_SOURCES}"

python "${PROJECT_ROOT}/scripts/data_process/qa_search_test_merge.py" \
  --local_dir "${OUT_DIR}" \
  --data_sources "${TEST_SOURCES}"

echo "Done:"
echo "  ${OUT_DIR}/train.parquet"
echo "  ${OUT_DIR}/test.parquet"
