#!/usr/bin/env bash
# Install Miniconda (conda) on Linux. Non-interactive; idempotent if already installed.
set -euo pipefail

INSTALL_DIR="${CONDA_INSTALL_DIR:-${HOME}/miniconda3}"

if command -v conda &>/dev/null; then
  echo "conda is already on PATH: $(command -v conda)"
  conda --version
  exit 0
fi

if [[ -x "${INSTALL_DIR}/bin/conda" ]]; then
  echo "Miniconda already installed at ${INSTALL_DIR}"
  "${INSTALL_DIR}/bin/conda" --version
  echo "Add to your shell: source ${INSTALL_DIR}/etc/profile.d/conda.sh"
  exit 0
fi

ARCH="$(uname -m)"
case "${ARCH}" in
  x86_64)
    INSTALLER_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    ;;
  aarch64|arm64)
    INSTALLER_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"
    ;;
  *)
    echo "Unsupported architecture: ${ARCH}" >&2
    exit 1
    ;;
esac

INSTALLER="$(mktemp /tmp/miniconda.XXXXXX.sh)"
trap 'rm -f "${INSTALLER}"' EXIT

echo "Downloading Miniconda from ${INSTALLER_URL} ..."
curl -fsSL -o "${INSTALLER}" "${INSTALLER_URL}"

echo "Installing to ${INSTALL_DIR} ..."
bash "${INSTALLER}" -b -p "${INSTALL_DIR}"

"${INSTALL_DIR}/bin/conda" config --set auto_activate_base false 2>/dev/null || true

# Conda 24+ blocks create/install until default-channel ToS is accepted.
if "${INSTALL_DIR}/bin/conda" tos --help &>/dev/null; then
  for ch in https://repo.anaconda.com/pkgs/main https://repo.anaconda.com/pkgs/r; do
    echo "Accepting conda Terms of Service for ${ch} ..."
    "${INSTALL_DIR}/bin/conda" tos accept --override-channels --channel "${ch}"
  done
fi

# Initialize common shells (ignore failures for shells not installed)
for shell in bash zsh fish; do
  if command -v "${shell}" &>/dev/null; then
    "${INSTALL_DIR}/bin/conda" init "${shell}" 2>/dev/null || true
  fi
done

echo ""
echo "Miniconda installed."
"${INSTALL_DIR}/bin/conda" --version
echo ""
echo "Activate in this session:"
echo "  source ${INSTALL_DIR}/etc/profile.d/conda.sh"
echo ""
echo "Or restart your shell after conda init."
