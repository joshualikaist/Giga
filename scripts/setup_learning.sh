#!/usr/bin/env bash
# =============================================================================
# setup_learning.sh — 강화학습 패키지 설치 (선택, 1회). PyTorch + Gymnasium + Stable-Baselines3
#
# 사용법:  source scripts/activate.sh   (먼저)
#          bash scripts/setup_learning.sh          # NVIDIA GPU가 있으면 GPU판, 없으면 CPU판 PyTorch
#          bash scripts/setup_learning.sh --cpu    # CPU판 강제 (내려받는 양 약 200 MB)
#          bash scripts/setup_learning.sh --gpu    # GPU판 강제 (약 3 GB, NVIDIA 드라이버 필요)
#
# 학습 예제는 CPU만으로도 충분히 빠릅니다 (docs/07 §6 CPU vs GPU 실측 참고).
# =============================================================================
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TORCH_VERSION="2.8.0"   # 바꾸기 전에 requirements-learning.txt의 버전 고정 이유를 읽을 것

info() { echo -e "\033[1;34m[learning]\033[0m $*"; }
fail() { echo -e "\033[1;31m[error]\033[0m $*"; exit 1; }

[[ "${VIRTUAL_ENV:-}" == "${REPO_ROOT}/.venv" ]] || fail "먼저:  source scripts/activate.sh"

MODE="auto"
case "${1:-}" in
    --cpu) MODE="cpu" ;;
    --gpu) MODE="gpu" ;;
    "") ;;
    *) fail "알 수 없는 옵션: $1  (사용 가능: --cpu, --gpu)" ;;
esac
if [[ "${MODE}" == "auto" ]]; then
    if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then MODE="gpu"; else MODE="cpu"; fi
fi

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
    print("  GPU 없음 → CPU로 학습합니다 (이 예제는 CPU로 충분)")
EOF
info "완료! 다음:  python learning/train_balance.py   (docs/07_learning.md 참고)"
