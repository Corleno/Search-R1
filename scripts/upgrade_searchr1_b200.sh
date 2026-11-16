#!/usr/bin/env bash
# Recreate the searchr1 conda env for NVIDIA Blackwell (B200 / sm_100).
#
# Strategy (keeps Search-R1's vendored vLLM 0.6.3 hybrid-engine shims):
#   - Python 3.10
#   - PyTorch 2.7.1+cu128 (native sm_100 kernels)
#   - flash-attn prebuilt for torch2.7 + cu12 + cp310 (training path)
#   - Rebuild vllm==0.6.3 from source with:
#       * CMake CUDA_SUPPORTED_ARCHS += 10.0
#       * TORCH_CUDA_ARCH_LIST=9.0;10.0
#       * FlashAttention runtime check patched for sm_10x
#       * pip install wheel --no-deps (so pip cannot downgrade torch)
#
# Usage:
#   bash scripts/upgrade_searchr1_b200.sh
#   bash scripts/upgrade_searchr1_b200.sh --keep-old   # clone old env to searchr1_cu121 first
#   bash scripts/upgrade_searchr1_b200.sh --skip-smoke
#
# Env overrides:
#   TORCH_VER=2.7.1
#   VLLM_VER=0.6.3
#   FLASH_ATTN_WHEEL=<url>
#   TORCH_CUDA_ARCH_LIST=9.0;10.0
#   MAX_JOBS=8
#   VLLM_BUILD_DIR=/tmp/vllm_src
#   VLLM_WHEEL_DIR=/tmp/vllm_wheels
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
KEEP_OLD=0
SKIP_SMOKE=0
for arg in "$@"; do
  case "${arg}" in
    --keep-old) KEEP_OLD=1 ;;
    --skip-smoke) SKIP_SMOKE=1 ;;
    -h|--help)
      sed -n '2,28p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown arg: ${arg}" >&2
      exit 1
      ;;
  esac
done

TORCH_VER="${TORCH_VER:-2.7.1}"
VLLM_VER="${VLLM_VER:-0.6.3}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0;10.0}"
MAX_JOBS="${MAX_JOBS:-8}"
FLASH_ATTN_WHEEL_DEFAULT="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post2/flash_attn-2.8.3.post2+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
FLASH_ATTN_WHEEL="${FLASH_ATTN_WHEEL:-${FLASH_ATTN_WHEEL_DEFAULT}}"
VLLM_BUILD_DIR="${VLLM_BUILD_DIR:-/tmp/vllm_src}"
VLLM_WHEEL_DIR="${VLLM_WHEEL_DIR:-/tmp/vllm_wheels}"
# Commit pinned by vllm 0.6.3 CMakeLists.txt FetchContent
VLLM_FLASH_ATTN_GIT_TAG="${VLLM_FLASH_ATTN_GIT_TAG:-013f0c4fc47e6574060879d9734c1df8c5c273bd}"

# Prefer system CUDA 12.8 toolkit (needed to compile vllm for sm_100).
if [[ -z "${CUDA_HOME:-}" ]]; then
  if [[ -d /usr/local/cuda-12.8 ]]; then
    export CUDA_HOME=/usr/local/cuda-12.8
  elif [[ -d /usr/local/cuda ]]; then
    export CUDA_HOME=/usr/local/cuda
  fi
fi
if [[ -n "${CUDA_HOME:-}" ]]; then
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
  # FindCUDA / nvcc link needs libcuda; stubs are enough at build time.
  export LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib64/stubs:/usr/lib64:${LIBRARY_PATH:-}"
fi
export CUDA_CUDA_LIBRARY="${CUDA_CUDA_LIBRARY:-/usr/lib64/libcuda.so}"
export TORCH_CUDA_ARCH_LIST
export MAX_JOBS
export VLLM_INSTALL_PUNICA_KERNELS="${VLLM_INSTALL_PUNICA_KERNELS:-0}"

echo "==> CUDA_HOME=${CUDA_HOME:-unset}"
echo "==> nvcc: $(command -v nvcc || echo missing)"
nvcc --version 2>/dev/null | tail -3 || true
echo "==> TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
echo "==> MAX_JOBS=${MAX_JOBS}"

# Do not source ~/.bashrc here (set -u + unbound PS1 aborts non-interactive shells).
set +u
# shellcheck source=/dev/null
if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
  source /root/miniconda3/etc/profile.d/conda.sh
elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
else
  echo "ERROR: conda.sh not found" >&2
  exit 1
fi
set -u
command -v conda >/dev/null

cd "${REPO_ROOT}"

# Ensure we are not inside searchr1 (cannot remove the active env).
conda deactivate >/dev/null 2>&1 || true
conda deactivate >/dev/null 2>&1 || true

if conda env list | awk '{print $1}' | grep -qx searchr1; then
  if [[ "${KEEP_OLD}" == 1 ]]; then
    if conda env list | awk '{print $1}' | grep -qx searchr1_cu121; then
      echo "searchr1_cu121 already exists; removing current searchr1 only."
    else
      echo "Renaming searchr1 -> searchr1_cu121 ..."
      conda create -n searchr1_cu121 --clone searchr1 -y
    fi
    conda env remove -n searchr1 -y
  else
    echo "Removing existing searchr1 env ..."
    conda env remove -n searchr1 -y
  fi
fi

echo "Creating searchr1 (python=3.10) ..."
conda create -n searchr1 python=3.10 -y

echo "Installing PyTorch ${TORCH_VER}+cu128 ..."
conda run -n searchr1 pip install --upgrade pip setuptools wheel
conda run -n searchr1 pip install \
  "torch==${TORCH_VER}" torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128

echo "Verifying PyTorch CUDA compute on this GPU ..."
conda run -n searchr1 --no-capture-output python -c '
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("arch_list", torch.cuda.get_arch_list())
assert torch.cuda.is_available(), "CUDA not available"
cap = torch.cuda.get_device_capability(0)
print("device", torch.cuda.get_device_name(0), "capability", cap)
x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
y = x @ x
torch.cuda.synchronize()
print("matmul_ok", float(y[0, 0]))
if cap[0] < 10:
    print("WARNING: GPU is not sm_100+; B200 upgrade still installed but may be unnecessary.")
'

echo "Installing build deps for vllm ${VLLM_VER} ..."
conda run -n searchr1 pip install ninja packaging cmake numpy

# ---------------------------------------------------------------------------
# Build a custom vllm 0.6.3 wheel with Blackwell (sm_100) support.
# ---------------------------------------------------------------------------
echo "Preparing vllm ${VLLM_VER} sources under ${VLLM_BUILD_DIR} ..."
rm -rf "${VLLM_BUILD_DIR}/vllm-${VLLM_VER}"
mkdir -p "${VLLM_BUILD_DIR}" "${VLLM_WHEEL_DIR}"
conda run -n searchr1 pip download "vllm==${VLLM_VER}" --no-binary=:all: -d "${VLLM_BUILD_DIR}" --no-deps
# Prefer sdist tarball
SDIST="$(ls "${VLLM_BUILD_DIR}"/vllm-"${VLLM_VER}"*.tar.gz 2>/dev/null | head -1 || true)"
if [[ -z "${SDIST}" ]]; then
  echo "ERROR: could not download vllm==${VLLM_VER} sdist" >&2
  exit 1
fi
tar -xzf "${SDIST}" -C "${VLLM_BUILD_DIR}"
VLLM_SRC="${VLLM_BUILD_DIR}/vllm-${VLLM_VER}"
if [[ ! -d "${VLLM_SRC}" ]]; then
  # Some sdists unpack as vllm-v0.6.3
  VLLM_SRC="$(find "${VLLM_BUILD_DIR}" -maxdepth 1 -type d -name 'vllm*' ! -name 'vllm_wheels' | head -1)"
fi
echo "vllm source: ${VLLM_SRC}"

echo "Patching CMakeLists.txt: CUDA_SUPPORTED_ARCHS += 10.0 ..."
python3 - <<PY
from pathlib import Path
p = Path("${VLLM_SRC}") / "CMakeLists.txt"
text = p.read_text()
old = 'set(CUDA_SUPPORTED_ARCHS "7.0;7.5;8.0;8.6;8.9;9.0")'
new = 'set(CUDA_SUPPORTED_ARCHS "7.0;7.5;8.0;8.6;8.9;9.0;10.0")'
if old not in text:
    if "10.0" in text and "CUDA_SUPPORTED_ARCHS" in text:
        print("CMakeLists already has 10.0")
    else:
        raise SystemExit(f"CMakeLists pattern not found in {p}")
else:
    p.write_text(text.replace(old, new, 1))
    print("patched CUDA_SUPPORTED_ARCHS")
PY

echo "Fetching vllm-flash-attn @ ${VLLM_FLASH_ATTN_GIT_TAG} and patching sm_10x checks ..."
FA_SRC="${VLLM_BUILD_DIR}/vllm-flash-attn-${VLLM_FLASH_ATTN_GIT_TAG}"
rm -rf "${FA_SRC}"
git clone --filter=blob:none https://github.com/vllm-project/flash-attention.git "${FA_SRC}"
git -C "${FA_SRC}" checkout "${VLLM_FLASH_ATTN_GIT_TAG}"
FA_SRC="${FA_SRC}" python3 - <<'PY'
from pathlib import Path
import os

p = Path(os.environ["FA_SRC"]) / "csrc/flash_attn/flash_api.cpp"
text = p.read_text()
if "is_sm10x" in text:
    print("FA already patched:", text.count("is_sm10x"), "is_sm10x refs")
else:
    old = (
        "    bool is_sm8x = dprops->major == 8 && dprops->minor >= 0;\n"
        "    bool is_sm90 = dprops->major == 9 && dprops->minor == 0;\n"
        '    TORCH_CHECK(is_sm90 || is_sm8x, "FlashAttention only supports Ampere GPUs or newer.");'
    )
    new = (
        "    bool is_sm8x = dprops->major == 8 && dprops->minor >= 0;\n"
        "    bool is_sm90 = dprops->major == 9 && dprops->minor == 0;\n"
        "    bool is_sm10x = dprops->major >= 10;  // Blackwell+\n"
        '    TORCH_CHECK(is_sm90 || is_sm8x || is_sm10x, "FlashAttention only supports Ampere GPUs or newer.");'
    )
    count = text.count(old)
    if count < 1:
        raise SystemExit(f"FA Ampere check pattern not found in {p}")
    text = text.replace(old, new)
    old2 = '        TORCH_CHECK(is_sm90 || is_sm8x, "bfloat16 is only supported on Ampere GPUs or newer");'
    new2 = '        TORCH_CHECK(is_sm90 || is_sm8x || is_sm10x, "bfloat16 is only supported on Ampere GPUs or newer");'
    text = text.replace(old2, new2)
    p.write_text(text)
    print("patched", p, "ampere_blocks", count)
PY

export VLLM_FLASH_ATTN_SRC_DIR="${FA_SRC}"
echo "VLLM_FLASH_ATTN_SRC_DIR=${VLLM_FLASH_ATTN_SRC_DIR}"

echo "Building vllm wheel (this takes a while; logging to /tmp/vllm_b200_build.log) ..."
rm -f "${VLLM_WHEEL_DIR}"/vllm-*.whl
# Use conda run so the env's torch/headers are used; --no-deps keeps torch pinned.
conda run -n searchr1 --no-capture-output \
  env CUDA_HOME="${CUDA_HOME:-}" \
      PATH="${PATH}" \
      LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
      LIBRARY_PATH="${LIBRARY_PATH:-}" \
      CUDA_CUDA_LIBRARY="${CUDA_CUDA_LIBRARY}" \
      TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
      MAX_JOBS="${MAX_JOBS}" \
      VLLM_INSTALL_PUNICA_KERNELS="${VLLM_INSTALL_PUNICA_KERNELS}" \
      VLLM_FLASH_ATTN_SRC_DIR="${VLLM_FLASH_ATTN_SRC_DIR}" \
  bash -lc "cd '${VLLM_SRC}' && pip wheel . --no-build-isolation --no-deps -w '${VLLM_WHEEL_DIR}' --no-cache-dir" \
  2>&1 | tee /tmp/vllm_b200_build.log

WHEEL="$(ls "${VLLM_WHEEL_DIR}"/vllm-"${VLLM_VER}"*.whl 2>/dev/null | head -1 || true)"
if [[ -z "${WHEEL}" ]]; then
  echo "ERROR: vllm wheel not produced; see /tmp/vllm_b200_build.log" >&2
  exit 1
fi
echo "Installing ${WHEEL} with --no-deps (preserves torch ${TORCH_VER}+cu128) ..."
conda run -n searchr1 pip install "${WHEEL}" --no-deps --force-reinstall --no-cache-dir

# vllm runtime deps minus torch (already installed). Do not pin xformers to the
# cu121-era 0.0.27.post2 — that pulls an older torch and breaks sm_100.
echo "Installing vllm runtime dependencies (excluding torch) ..."
conda run -n searchr1 pip install \
  'ray>=2.9' 'xformers' 'fastapi' 'uvicorn[standard]' 'pydantic>=2' \
  'prometheus_client>=0.18.0' 'openai>=1.40.0' 'einops' 'prometheus-fastapi-instrumentator>=7.0.0' \
  'tokenizers>=0.19.1' 'msgspec' 'gguf==0.10.0' 'importlib_metadata' 'mistral_common>=1.4.3' \
  'py-cpuinfo' 'soundfile' 'compressed-tensors==0.6.0' \
  || echo "WARNING: some optional vllm deps failed; continuing"
# Guard against accidental torch downgrade from any dependency resolver pass.
conda run -n searchr1 pip install \
  "torch==${TORCH_VER}" torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128

echo "Installing flash-attn wheel (training path) ..."
echo "  ${FLASH_ATTN_WHEEL}"
if ! conda run -n searchr1 pip install "${FLASH_ATTN_WHEEL}"; then
  echo "Prebuilt flash-attn failed; building from source (slow) ..."
  conda run -n searchr1 pip install flash-attn==2.8.3 --no-build-isolation
fi

echo "Installing Search-R1 / verl editable + runtime deps ..."
conda run -n searchr1 pip install -e "${REPO_ROOT}"
conda run -n searchr1 pip install 'transformers>=4.44,<4.48' wandb huggingface_hub
conda run -n searchr1 pip install 'tensordict<0.6'

echo "Re-asserting torch ${TORCH_VER}+cu128 was not downgraded ..."
conda run -n searchr1 --no-capture-output python -c "
import torch
from importlib.metadata import version
print('torch', torch.__version__)
print('vllm', version('vllm'))
assert torch.__version__.startswith('${TORCH_VER}'), torch.__version__
assert 'cu128' in torch.__version__ or torch.version.cuda.startswith('12.8'), (torch.__version__, torch.version.cuda)
"

echo "Smoke imports ..."
conda run -n searchr1 --no-capture-output python -c '
import torch, vllm, flash_attn, transformers
from importlib.metadata import version
print("torch", torch.__version__)
print("vllm", version("vllm"))
print("flash_attn", flash_attn.__version__)
print("transformers", transformers.__version__)
import verl
from verl.third_party.vllm import vllm_version, LLM
print("verl vllm shim version", vllm_version)
x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize()
print("cuda_ok", float((x @ x)[0, 0]))
'

if [[ "${SKIP_SMOKE}" != 1 ]]; then
  echo "Smoke vLLM generate (FLASH_ATTN on B200) ..."
  conda run -n searchr1 --no-capture-output \
    env VLLM_ATTENTION_BACKEND=FLASH_ATTN \
    python -c '
import os
from vllm import LLM, SamplingParams
print("backend", os.environ.get("VLLM_ATTENTION_BACKEND"))
llm = LLM(model="Qwen/Qwen2.5-3B-Instruct", dtype="bfloat16",
          gpu_memory_utilization=0.35, max_model_len=512, enforce_eager=True)
out = llm.generate(["Say hello in one short sentence."],
                   SamplingParams(max_tokens=16, temperature=0.0))
print("OUT:", out[0].outputs[0].text)
print("SMOKE_OK")
'
fi

echo ""
echo "Done. Activate with: conda activate searchr1"
echo "For rollouts, prefer: export VLLM_ATTENTION_BACKEND=FLASH_ATTN"
echo "Old cu121 env (if --keep-old): conda activate searchr1_cu121"
