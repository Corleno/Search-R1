#!/usr/bin/env bash
# Bootstrap a fresh machine: clone Search-R1, checkout branch, install envs, download data.
#
# Usage (from anywhere):
#   export GITHUB_TOKEN=ghp_...   # required for private push remote; optional for public clone
#   bash scripts/bootstrap_install.sh
#
# Or one-shot after cloning publicly:
#   GITHUB_TOKEN=ghp_... bash -c "$(curl -fsSL ...)"   # if you host this script
#
# Environment variables:
#   REPO_URL          Clone URL (default: https://github.com/Corleno/Search-R1.git)
#   REPO_DIR          Target directory (default: ./Search-R1, or repo root if already inside)
#   BRANCH            Git branch (default: sdpo_search_dev)
#   GITHUB_USER       GitHub username for authenticated origin (default: Corleno)
#   GITHUB_TOKEN      PAT used only to set origin push URL (never commit this)
#   SKIP_CLONE=1      Skip clone; use existing checkout
#   SKIP_DOWNLOAD=1   Skip scripts/download_data.sh
#   INSTALL_ENVS_ARGS Extra args for install_envs.sh (default: --install-conda)
#
set -euo pipefail

# Search-R1 public WandB (wandb.ai). Train scripts read WANDB_API_KEY_SEARCH_R1.
export WANDB_API_KEY_SEARCH_R1="${WANDB_API_KEY_SEARCH_R1:-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}"
if ! grep -qF 'WANDB_API_KEY_SEARCH_R1' "${HOME}/.bashrc" 2>/dev/null; then
  echo "export WANDB_API_KEY_SEARCH_R1='${WANDB_API_KEY_SEARCH_R1}'" >> "${HOME}/.bashrc"
fi

REPO_URL="${REPO_URL:-https://github.com/Corleno/Search-R1.git}"
BRANCH="${BRANCH:-sdpo_search_dev}"
GITHUB_USER="${GITHUB_USER:-Corleno}"
INSTALL_ENVS_ARGS="${INSTALL_ENVS_ARGS:---install-conda}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# If this file already lives inside a Search-R1 checkout, prefer that root.
if [[ -f "${SCRIPT_DIR}/../scripts/install_envs.sh" ]]; then
  DEFAULT_REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
else
  DEFAULT_REPO_DIR="$(pwd)/Search-R1"
fi
REPO_DIR="${REPO_DIR:-${DEFAULT_REPO_DIR}}"

usage() {
  cat <<EOF
Usage: bash scripts/bootstrap_install.sh

Clone Search-R1, checkout ${BRANCH}, run install_envs.sh, then download_data.sh.

Environment variables:
  REPO_URL            Clone URL (default: ${REPO_URL})
  REPO_DIR            Checkout path (default: ${DEFAULT_REPO_DIR})
  BRANCH              Branch to checkout (default: ${BRANCH})
  GITHUB_USER         User for authenticated origin (default: ${GITHUB_USER})
  GITHUB_TOKEN        GitHub PAT for authenticated origin (do not hardcode)
  SKIP_CLONE=1        Skip git clone / checkout
  SKIP_DOWNLOAD=1     Skip data download
  INSTALL_ENVS_ARGS   Args for install_envs.sh (default: --install-conda)

Example:
  export GITHUB_TOKEN=ghp_...
  bash scripts/bootstrap_install.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "${SKIP_CLONE:-0}" != 1 ]]; then
  if [[ ! -d "${REPO_DIR}/.git" ]]; then
    echo "==> Cloning ${REPO_URL} into ${REPO_DIR} ..."
    git clone "${REPO_URL}" "${REPO_DIR}"
  else
    echo "==> Repo already present at ${REPO_DIR}; skipping clone."
  fi

  cd "${REPO_DIR}"

  if [[ -n "${GITHUB_TOKEN:-}" ]]; then
    echo "==> Setting origin to authenticated HTTPS URL (token from GITHUB_TOKEN) ..."
    git remote set-url origin "https://${GITHUB_USER}:${GITHUB_TOKEN}@github.com/${GITHUB_USER}/Search-R1.git"
  else
    echo "==> GITHUB_TOKEN unset; leaving origin as-is (public HTTPS is fine for pull)."
  fi

  echo "==> Checking out ${BRANCH} ..."
  git fetch origin "${BRANCH}" 2>/dev/null || true
  git checkout "${BRANCH}"
else
  cd "${REPO_DIR}"
  echo "==> SKIP_CLONE=1; using ${REPO_DIR}"
fi

if [[ ! -f "${REPO_DIR}/scripts/install_envs.sh" ]]; then
  echo "Missing ${REPO_DIR}/scripts/install_envs.sh" >&2
  exit 1
fi

echo "==> Installing conda envs ..."
# shellcheck disable=SC2086
bash "${REPO_DIR}/scripts/install_envs.sh" ${INSTALL_ENVS_ARGS}

# Prefer searchr1 python for huggingface_hub used by download_data.sh
if command -v conda &>/dev/null || [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "${CONDA_INSTALL_DIR:-${HOME}/miniconda3}/etc/profile.d/conda.sh" 2>/dev/null || true
  if conda env list 2>/dev/null | awk '{print $1}' | grep -qx searchr1; then
    # shellcheck disable=SC1091
    eval "$(conda shell.bash hook)"
    conda activate searchr1
  fi
fi

if [[ "${SKIP_DOWNLOAD:-0}" != 1 ]]; then
  echo "==> Downloading data ..."
  bash "${REPO_DIR}/scripts/download_data.sh"
else
  echo "==> SKIP_DOWNLOAD=1; skipping data download."
fi

echo ""
echo "Done. Repo: ${REPO_DIR}"
echo "  conda activate searchr1"
echo "  conda activate retriever"
