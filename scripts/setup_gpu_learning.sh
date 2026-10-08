#!/usr/bin/env bash
# =============================================================================
# setup_gpu_learning.sh — GPU 대량 병렬 학습 환경(.venv-mjx) 설치 (선택, 1회). 약 3 GB
#
# 사용법 (ROS를 켜지 않은 새 터미널 권장):
#     bash scripts/setup_gpu_learning.sh
# 이후 매번:
#     source scripts/activate_gpu.sh
#
# MuJoCo MJX(JAX) + Brax PPO로 시뮬레이션 수천 개를 GPU에서 동시에 돌린다. NVIDIA GPU + 드라이버 필요.
# ROS2용 .venv와는 완전히 분리된 가상환경이다 (이유: requirements-gpu.txt 머리말).
# =============================================================================
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv-mjx"

info() { echo -e "\033[1;34m[gpu-learning]\033[0m $*"; }
fail() { echo -e "\033[1;31m[error]\033[0m $*"; exit 1; }

command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1 \
    || fail "NVIDIA GPU/드라이버를 찾지 못했습니다 (nvidia-smi). GPU 없이 학습하려면 scripts/setup_learning.sh (CPU) 사용"
[[ -n "${CONDA_PREFIX:-}" ]] && fail "conda가 켜져 있습니다 → conda deactivate"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | sed 's/^/    GPU: /'

if [[ -x "${VENV_DIR}/bin/python" ]]; then
    info "기존 가상환경 재사용: ${VENV_DIR}"
else
    info "가상환경 생성: ${VENV_DIR} (ROS2와 분리, 시스템 패키지 미사용)"
    /usr/bin/python3 -m venv "${VENV_DIR}"
fi
touch "${VENV_DIR}/COLCON_IGNORE"

# ROS 환경이 켜진 터미널이어도 섞이지 않도록 PYTHONPATH를 비우고 설치
export PYTHONNOUSERSITE=1
unset PYTHONPATH
info "패키지 설치 (JAX CUDA, MuJoCo MJX, Brax ...) — 몇 분 걸립니다"
"${VENV_DIR}/bin/python" -m pip install --quiet --upgrade pip
(cd "${REPO_ROOT}" && "${VENV_DIR}/bin/python" -m pip install -r requirements-gpu.txt -e .)

info "설치 확인 (GPU에서 MJX 시뮬레이션 한 번 실행)"
"${VENV_DIR}/bin/python" - <<'EOF'
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.7")
import jax, jax.numpy as jnp, mujoco, numpy
from mujoco import mjx
print(f"  jax {jax.__version__} | mujoco {mujoco.__version__} | numpy {numpy.__version__}")
print(f"  JAX 장치: {jax.devices()}")
assert jax.default_backend() == "gpu", "JAX가 GPU를 쓰지 못합니다"
m = mujoco.MjModel.from_xml_string("<mujoco><worldbody><body><freejoint/><geom size='.1'/></body></worldbody></mujoco>")
d = jax.jit(mjx.step)(mjx.put_model(m), mjx.make_data(m))
print(f"  MJX GPU 시뮬레이션 OK (z = {float(d.qpos[2]):.4f})")
EOF
info "완료! 다음:  source scripts/activate_gpu.sh  →  python learning/train_walk_gpu.py   (docs/08 참고)"
