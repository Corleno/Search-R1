#!/usr/bin/env bash
# Create searchr1 and retriever conda envs per README.md Installation section.
#
# Usage:
#   bash scripts/install_envs.sh [OPTIONS]
#
# Examples:
#   bash scripts/install_envs.sh
#   bash scripts/install_envs.sh --install-conda
#   bash scripts/install_envs.sh --searchr1-only
#   bash scripts/install_envs.sh --retriever-only
#   SKIP_FLASH_ATTN=1 bash scripts/install_envs.sh --searchr1-only
#
# Options:
#   --searchr1-only   Install only the searchr1 environment (Python 3.9, RL training)
#   --retriever-only  Install only the retriever environment (Python 3.10, retrieval server)
#   --install-conda   Run scripts/install_conda.sh if conda is not on PATH
#   -h, --help        Print help and exit
#
# Environment variables:
#   CONDA_INSTALL_DIR   Miniconda install path (default: ~/miniconda3)
#   SKIP_FLASH_ATTN=1   Skip flash-attn in searchr1 (optional)
#   FLASH_ATTN_WHEEL    Override prebuilt flash-attn wheel URL (README default for torch 2.4 + cu121 + py3.9)
#
# After install:
#   conda activate searchr1
#   conda activate retriever
#
# When installing both envs (default), searchr1 runs first, then retriever
# (sequential avoids conda pkgs-cache corruption from parallel large downloads).
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# README: torch 2.4 + cu121 + python 3.9 prebuilt wheel (avoids cross-device link on pip install)
FLASH_ATTN_WHEEL_DEFAULT="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.4cxx11abiFALSE-cp39-cp39-linux_x86_64.whl"

INSTALL_SEARCHR1=1
INSTALL_RETRIEVER=1
INSTALL_CONDA=0

usage() {
  cat <<EOF
Usage: bash scripts/install_envs.sh [OPTIONS]

Create conda environments from README.md (run from repo root):
  - searchr1   Python 3.9 — torch 2.4 (cu121), vllm 0.6.3, verl, flash-attn wheel, wandb
  - retriever  Python 3.10 — pytorch+cuda 12.1, transformers<4.48, pyserini, faiss-gpu, fastapi

  Installs searchr1 first, then retriever (sequential; retriever uses large conda packages).

Options:
  --searchr1-only   Install only the searchr1 environment
  --retriever-only  Install only the retriever environment
  --install-conda   Run scripts/install_conda.sh if conda is missing
  -h, --help        Show this help

Examples:
  bash scripts/install_envs.sh
  bash scripts/install_envs.sh --install-conda
  bash scripts/install_envs.sh --searchr1-only
  bash scripts/install_envs.sh --retriever-only
  SKIP_FLASH_ATTN=1 bash scripts/install_envs.sh --searchr1-only

Environment variables:
  CONDA_INSTALL_DIR   Miniconda path (default: ~/miniconda3; see install_conda.sh)
  SKIP_FLASH_ATTN=1   Skip flash-attn in searchr1 (optional)
  FLASH_ATTN_WHEEL    Prebuilt flash-attn wheel URL (see README Installation)

After install:
  conda activate searchr1
  conda activate retriever
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --searchr1-only)
      INSTALL_SEARCHR1=1
      INSTALL_RETRIEVER=0
      shift
      ;;
    --retriever-only)
      INSTALL_SEARCHR1=0
      INSTALL_RETRIEVER=1
      shift
      ;;
    --install-conda)
      INSTALL_CONDA=1
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

# Conda 24+ requires accepting Anaconda default-channel ToS before create/install.
accept_conda_tos() {
  if ! conda tos --help &>/dev/null; then
    return 0
  fi
  local channels=(
    https://repo.anaconda.com/pkgs/main
    https://repo.anaconda.com/pkgs/r
  )
  for ch in "${channels[@]}"; do
    echo "Accepting conda Terms of Service for ${ch} ..."
    conda tos accept --override-channels --channel "${ch}"
  done
}

ensure_conda() {
  if command -v conda &>/dev/null; then
    return 0
  fi
  local install_dir="${CONDA_INSTALL_DIR:-${HOME}/miniconda3}"
  if [[ -f "${install_dir}/etc/profile.d/conda.sh" ]]; then
    # shellcheck source=/dev/null
    source "${install_dir}/etc/profile.d/conda.sh"
    return 0
  fi
  if [[ "${INSTALL_CONDA:-0}" == 1 ]]; then
    bash "${SCRIPT_DIR}/install_conda.sh"
    # shellcheck source=/dev/null
    source "${install_dir}/etc/profile.d/conda.sh"
    return 0
  fi
  echo "conda not found. Install Miniconda first:" >&2
  echo "  bash ${SCRIPT_DIR}/install_conda.sh" >&2
  echo "  source ~/miniconda3/etc/profile.d/conda.sh" >&2
  echo "Or re-run with: --install-conda" >&2
  exit 1
}

conda_env_exists() {
  conda env list | awk '{print $1}' | grep -qx "$1"
}

create_env_if_missing() {
  local name="$1"
  local python="$2"
  if conda_env_exists "${name}"; then
    echo "Conda env '${name}' already exists; skipping create."
  else
    echo "Creating conda env '${name}' (python=${python}) ..."
    conda create -n "${name}" "python=${python}" -y
  fi
}

# pip 25+ uses dataclass(slots=...) and breaks on Python 3.9 (conda 26 default).
ensure_pip_py39() {
  echo "Ensuring pip<25 for Python 3.9 compatibility ..."
  conda install -n searchr1 "pip<25" -y
}

install_searchr1() {
  echo "=== Installing searchr1 environment ==="
  create_env_if_missing searchr1 3.9
  ensure_pip_py39

  echo "Installing PyTorch (cu121) ..."
  conda run -n searchr1 pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

  echo "Installing vllm ..."
  conda run -n searchr1 pip3 install vllm==0.6.3

  echo "Installing verl (editable) from ${REPO_ROOT} ..."
  conda run -n searchr1 pip install -e "${REPO_ROOT}"

  if [[ "${SKIP_FLASH_ATTN:-0}" != 1 ]]; then
    local flash_attn_wheel="${FLASH_ATTN_WHEEL:-${FLASH_ATTN_WHEEL_DEFAULT}}"
    echo "Installing flash-attn from prebuilt wheel (README) ..."
    echo "  ${flash_attn_wheel}"
    if ! conda run -n searchr1 pip install "${flash_attn_wheel}"; then
      echo "Prebuilt flash-attn wheel install failed." >&2
      echo "If the URL does not match your torch/cuda/python build, set FLASH_ATTN_WHEEL or try:" >&2
      echo "  export TMPDIR=/path/on/same/filesystem/as/pip-cache" >&2
      echo "  export PIP_CACHE_DIR=/path/on/same/filesystem/as/pip-cache" >&2
      echo "  conda activate searchr1 && pip3 install flash-attn --no-build-isolation" >&2
      return 1
    fi
  else
    echo "Skipping flash-attn (SKIP_FLASH_ATTN=1)."
  fi

  echo "Installing wandb ..."
  conda run -n searchr1 pip install wandb

  echo "searchr1 environment ready."
}

install_retriever() {
  echo "=== Installing retriever environment ==="
  create_env_if_missing retriever 3.10

  echo "Installing PyTorch (cu121) via pip ..."
  conda install -n retriever -y -c pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia

  echo "Installing transformers, datasets, pyserini ..."
  # transformers 5.x needs a newer torch and breaks retrieval_server with torch 2.4 (README)
  conda run -n retriever pip install 'transformers>=4.44,<4.48' datasets pyserini

  echo "Installing faiss-gpu ..."
  conda install -n retriever -y -c pytorch -c nvidia faiss-gpu=1.8.0

  echo "Installing uvicorn, fastapi ..."
  conda run -n retriever pip install uvicorn fastapi

  echo "retriever environment ready."
}

run_installs() {
  if [[ "${INSTALL_SEARCHR1}" == 1 ]]; then
    install_searchr1
  fi
  if [[ "${INSTALL_RETRIEVER}" == 1 ]]; then
    install_retriever
  fi
}

ensure_conda
accept_conda_tos
conda --version

run_installs

echo ""
echo "Done. Activate with:"
[[ "${INSTALL_SEARCHR1}" == 1 ]] && echo "  conda activate searchr1"
[[ "${INSTALL_RETRIEVER}" == 1 ]] && echo "  conda activate retriever"
