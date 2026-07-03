#!/usr/bin/env bash
# Download train replay data from GCS to res/train_replays.
#
# Syncs from:
#   gs://permanent-us-central2-0rxn/ruimeng/research/search_opd/res/train_replays/
# to:
#   <repo>/res/train_replays/
#
# Uses gsutil rsync for incremental updates (only changed files are downloaded).
#
# Usage:
#   bash scripts/sync_train_replays_from_gcs.sh [OPTIONS]
#
# Options:
#   --subpath PATH       Sync only PATH under train_replays (e.g. exp_sdpo_searchr1_0620)
#   --dry-run            Print what would be downloaded without copying
#   --delete             Remove local files that no longer exist on GCS
#   --force              Download even when matching paths already exist locally
#   -h, --help           Show this help
#
# Environment variables:
#   GCS_PREFIX         GCS source prefix (default: gs://permanent-us-central2-0rxn/ruimeng/research/search_opd)
#   REPLAYS_DIR        Local train replays root (default: <repo>/res/train_replays)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

GCS_PREFIX="${GCS_PREFIX:-gs://permanent-us-central2-0rxn/ruimeng/research/search_opd}"
REPLAYS_DIR="${REPLAYS_DIR:-$PROJECT_ROOT/res/train_replays}"
GCS_REPLAYS_DIR="${GCS_PREFIX%/}/res/train_replays"

SUBPATH=""
DRY_RUN=0
DELETE=0
FORCE=0

usage() {
  cat <<EOF
Usage: bash scripts/sync_train_replays_from_gcs.sh [OPTIONS]

Download train replay data from GCS:

  Remote: ${GCS_REPLAYS_DIR}/
  Local:  ${REPLAYS_DIR}/

Options:
  --subpath PATH       Sync only PATH under train_replays
  --dry-run            Print planned downloads without copying
  --delete             Delete local files missing on GCS (use with care)
  --force              Download even when matching paths already exist locally
  -h, --help           Show this help

Environment variables:
  GCS_PREFIX         GCS source prefix
  REPLAYS_DIR        Local train replays root (default: $PROJECT_ROOT/res/train_replays)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
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
    --force)
      FORCE=1
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

RSYNC_OPTS=(-m rsync -r)
if [[ "$DRY_RUN" -eq 1 ]]; then
  RSYNC_OPTS+=(-n)
fi
if [[ "$DELETE" -eq 1 ]]; then
  RSYNC_OPTS+=(-d)
fi

gcs_has_objects() {
  local gcs_uri="${1%/}/"
  gsutil ls "$gcs_uri" >/dev/null 2>&1
}

check_local_conflicts() {
  local src="$1"
  local dst_root="$2"

  if ! gcs_has_objects "$src"; then
    echo "Remote path not found or empty: $src" >&2
    exit 1
  fi

  local -a conflicts=()
  local -a conflict_locals=()
  local -a conflict_remotes=()

  if [[ -n "$SUBPATH" ]]; then
    local dst="${dst_root}/${SUBPATH}"
    if [[ -e "$dst" ]]; then
      conflicts+=("$SUBPATH")
      conflict_locals+=("$dst")
      conflict_remotes+=("$src")
    fi
  else
    local entry name dst
    local -a remote_entries=()
    while IFS= read -r entry; do
      remote_entries+=("$entry")
    done < <(gsutil ls "${src%/}/" 2>/dev/null || true)

    for entry in "${remote_entries[@]}"; do
      [[ -n "$entry" ]] || continue
      name="${entry%/}"
      name="${name##*/}"
      dst="${dst_root}/${name}"
      if [[ -e "$dst" ]]; then
        conflicts+=("$name")
        conflict_locals+=("$dst")
        conflict_remotes+=("${src%/}/${name}")
      fi
    done
  fi

  if [[ ${#conflicts[@]} -eq 0 ]]; then
    return 0
  fi

  echo "" >&2
  echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" >&2
  echo "!! ALARM: remote train_replays already exist locally (download will merge/overwrite):" >&2
  local i
  for i in "${!conflicts[@]}"; do
    echo "!!   remote: ${conflict_remotes[$i]}" >&2
    echo "!!   local:  ${conflict_locals[$i]}" >&2
  done
  echo "!! Use --force to skip this prompt, or --dry-run to preview only." >&2
  echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" >&2
  echo "" >&2

  if [[ "$DRY_RUN" -eq 1 || "$FORCE" -eq 1 ]]; then
    return 0
  fi

  local reply
  read -r -p "Continue download anyway? [y/N] " reply </dev/tty
  if [[ ! "$reply" =~ ^[Yy]$ ]]; then
    echo "Aborted." >&2
    exit 1
  fi
}

src="$GCS_REPLAYS_DIR"
dst="$REPLAYS_DIR"
if [[ -n "$SUBPATH" ]]; then
  src="${src}/${SUBPATH}"
  dst="${dst}/${SUBPATH}"
fi

mkdir -p "$dst"

echo "==> Project root: $PROJECT_ROOT"
echo "==> Remote:       $src"
echo "==> Local:        $dst"
if [[ -n "$SUBPATH" ]]; then
  echo "==> Subpath:      $SUBPATH"
fi
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "==> Dry run: no files will be copied"
fi

check_local_conflicts "$src" "$REPLAYS_DIR"

echo "==> Syncing train_replays from GCS"
gsutil "${RSYNC_OPTS[@]}" "$src" "$dst"

echo "==> Done."
