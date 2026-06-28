#!/usr/bin/env bash
# Upload verl_checkpoints and train replay data to GCS.
#
# Syncs local training artifacts to:
#   gs://permanent-us-central2-0rxn/ruimeng/research/search_opd/verl_checkpoints/
#   gs://permanent-us-central2-0rxn/ruimeng/research/search_opd/res/train_replays/
#
# Uses gsutil rsync for incremental updates (only changed files are uploaded).
#
# Usage:
#   bash scripts/upload_to_gcs.sh [OPTIONS]
#
# Options:
#   --checkpoints-only   Upload only verl_checkpoints/
#   --replays-only       Upload only res/train_replays/
#   --subpath PATH       Sync PATH under both dirs (e.g. my_experiment or nq_sdpo-Qwen/run1)
#   --dry-run            Print what would be uploaded without copying
#   --delete             Remove GCS objects that no longer exist locally
#   -h, --help           Show this help
#
# Environment variables:
#   GCS_PREFIX         GCS destination prefix (default: gs://permanent-us-central2-0rxn/ruimeng/research/search_opd)
#   CHECKPOINTS_DIR    Local checkpoints root (default: <repo>/verl_checkpoints)
#   REPLAYS_DIR        Local train replays root (default: <repo>/res/train_replays)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

GCS_PREFIX="${GCS_PREFIX:-gs://permanent-us-central2-0rxn/ruimeng/research/search_opd}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-$PROJECT_ROOT/verl_checkpoints}"
REPLAYS_DIR="${REPLAYS_DIR:-$PROJECT_ROOT/res/train_replays}"

UPLOAD_CHECKPOINTS=1
UPLOAD_REPLAYS=1
SUBPATH=""
DRY_RUN=0
DELETE=0

usage() {
  cat <<EOF
Usage: bash scripts/upload_to_gcs.sh [OPTIONS]

Upload verl_checkpoints and train replay data to GCS under:

  ${GCS_PREFIX}/verl_checkpoints/
  ${GCS_PREFIX}/res/train_replays/

Options:
  --checkpoints-only   Upload only verl_checkpoints/
  --replays-only       Upload only res/train_replays/
  --subpath PATH       Sync only PATH under each selected directory
  --dry-run            Print planned uploads without copying
  --delete             Delete remote files missing locally (use with care)
  -h, --help           Show this help

Environment variables:
  GCS_PREFIX         GCS destination prefix
  CHECKPOINTS_DIR    Local checkpoints root (default: $PROJECT_ROOT/verl_checkpoints)
  REPLAYS_DIR        Local train replays root (default: $PROJECT_ROOT/res/train_replays)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoints-only)
      UPLOAD_CHECKPOINTS=1
      UPLOAD_REPLAYS=0
      shift
      ;;
    --replays-only)
      UPLOAD_CHECKPOINTS=0
      UPLOAD_REPLAYS=1
      shift
      ;;
    --subpath)
      SUBPATH="${2:?--subpath requires a path argument}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --delete)
      DELETE=1
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

if ! command -v gsutil >/dev/null 2>&1; then
  echo "gsutil not found. Install the Google Cloud SDK and authenticate first." >&2
  exit 1
fi

GCS_PREFIX="${GCS_PREFIX%/}"

RSYNC_OPTS=(-m rsync -r)
if [[ "$DRY_RUN" -eq 1 ]]; then
  RSYNC_OPTS+=(-n)
fi
if [[ "$DELETE" -eq 1 ]]; then
  RSYNC_OPTS+=(-d)
fi

sync_dir() {
  local label="$1"
  local src_root="$2"
  local gcs_subdir="$3"

  local src="$src_root"
  if [[ -n "$SUBPATH" ]]; then
    src="$src_root/$SUBPATH"
  fi

  if [[ ! -d "$src" ]]; then
    echo "==> Skipping $label: directory not found ($src)"
    return 0
  fi

  local dst="${GCS_PREFIX}/${gcs_subdir}"
  if [[ -n "$SUBPATH" ]]; then
    dst="${dst}/${SUBPATH}"
  fi

  echo "==> Syncing $label"
  echo "    Local:  $src"
  echo "    Remote: $dst"
  gsutil "${RSYNC_OPTS[@]}" "$src" "$dst"
}

echo "==> Project root: $PROJECT_ROOT"
echo "==> GCS prefix:   $GCS_PREFIX"
if [[ -n "$SUBPATH" ]]; then
  echo "==> Subpath:      $SUBPATH"
fi
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "==> Dry run: no files will be copied"
fi

if [[ "$UPLOAD_CHECKPOINTS" -eq 1 ]]; then
  sync_dir "verl_checkpoints" "$CHECKPOINTS_DIR" "verl_checkpoints"
fi

if [[ "$UPLOAD_REPLAYS" -eq 1 ]]; then
  sync_dir "train_replays" "$REPLAYS_DIR" "res/train_replays"
fi

echo "==> Done."
