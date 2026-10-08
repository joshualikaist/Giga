#!/usr/bin/env bash
# =============================================================================
# setup_learning.sh — 강화학습 패키지 설치 (선택, 1회). PyTorch + Gymnasium + Stable-Baselines3
#
# 사용법:  source scripts/activate.sh   (먼저)
#          bash scripts/setup_learning.sh          # CPU판 PyTorch (내려받는 양 약 200 MB) — 권장
#          bash scripts/setup_learning.sh --gpu    # GPU(CUDA)판 PyTorch (약 3 GB, NVIDIA 드라이버 필요)
#
# 이 환경의 학습 예제(밀기 버티기)는 CPU가 GPU보다 빠릅니다 (docs/07 §6 실측: CPU 228 s, GPU 284 s).
# GPU로 빠르게 하는 보행 학습은 PyTorch가 아니라 별도 환경(JAX)을 씁니다 → scripts/setup_gpu_learning.sh
# =============================================================================
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TORCH_VERSION="2.8.0"   # 바꾸기 전에 requirements-learning.txt의 버전 고정 이유를 읽을 것

info() { echo -e "\033[1;34m[learning]\033[0m $*"; }
fail() { echo -e "\033[1;31m[error]\033[0m $*"; exit 1; }

[[ "${VIRTUAL_ENV:-}" == "${REPO_ROOT}/.venv" ]] || fail "먼저:  source scripts/activate.sh"

MODE="cpu"
case "${1:-}" in
    --cpu|"") ;;
    --gpu) MODE="gpu" ;;
    *) fail "알 수 없는 옵션: $1  (사용 가능: --gpu)" ;;
esac

if [[ "${MODE}" == "gpu" ]]; then
    INDEX="https://download.pytorch.org/whl/cu126"
    info "GPU판 PyTorch ${TORCH_VERSION} (CUDA 12.6) 설치 — 약 3 GB, 몇 분 걸릴 수 있습니다"
    nvidia-smi -L || true
else
    INDEX="https://download.pytorch.org/whl/cpu"
    info "CPU판 PyTorch ${TORCH_VERSION} 설치 — 약 200 MB"
fi

export PYTHONNOUSERSITE=1
cd "${REPO_ROOT}"
python -m pip install "torch==${TORCH_VERSION}" "numpy<2" --index-url "${INDEX}" --extra-index-url https://pypi.org/simple
python -m pip install -r requirements-learning.txt

info "설치 확인"
python - <<'EOF'
import numpy, setuptools, torch, gymnasium, stable_baselines3
print(f"  torch {torch.__version__} | gymnasium {gymnasium.__version__} | stable-baselines3 {stable_baselines3.__version__}")
print(f"  numpy {numpy.__version__} | setuptools {setuptools.__version__}")
assert int(numpy.__version__.split(".")[0]) < 2, "numpy 2가 설치됨 → ROS2 Humble과 충돌"
if torch.cuda.is_available():
    x = torch.randn(256, 256, device="cuda")
    _ = (x @ x).sum().item()  # 실제로 GPU 계산이 되는지
    print(f"  GPU 사용 가능: {torch.cuda.get_device_name(0)}")
else:
    print("  CPU판 PyTorch → CPU로 학습합니다 (이 예제는 CPU가 더 빠름)")
EOF
info "완료! 다음:  python learning/train_balance.py   (docs/07_learning.md 참고)"
