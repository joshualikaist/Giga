#!/usr/bin/env bash
# =============================================================================
# setup_env.sh — 개발 환경 1회 설치 스크립트
#
# 사용법 (저장소 어디에서 실행해도 됨):
#     bash scripts/setup_env.sh
#
# 하는 일:
#   1) conda가 켜져 있지 않은지 확인 (ROS2 Humble과 충돌)
#   2) 시스템 Python 3.10으로 .venv 생성 (--system-site-packages: rclpy 등 ROS2 패키지를 볼 수 있게)
#   3) requirements.txt 설치 (mujoco, numpy<2, biped_sim 패키지)
#   4) scripts/check_env.py로 환경 점검
#
# 다시 실행해도 안전합니다 (이미 있는 .venv는 재사용).
# 왜 이렇게 구성했는지는 docs/01_environment.md 참고.
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
SYSTEM_PYTHON="/usr/bin/python3"
ROS_SETUP="/opt/ros/humble/setup.bash"

info() { echo -e "\033[1;34m[setup]\033[0m $*"; }
warn() { echo -e "\033[1;33m[warn]\033[0m  $*"; }
fail() { echo -e "\033[1;31m[error]\033[0m $*"; exit 1; }

# --- 1) conda 확인 -----------------------------------------------------------
if [[ -n "${CONDA_PREFIX:-}" ]]; then
    fail "conda 환경(${CONDA_PREFIX})이 켜져 있습니다. 'conda deactivate'를 (필요하면 여러 번) 실행한 뒤 다시 시도하세요."
fi

# --- 2) 시스템 Python 확인 -----------------------------------------------------
[[ -x "${SYSTEM_PYTHON}" ]] || fail "${SYSTEM_PYTHON} 이 없습니다. Ubuntu 22.04 기본 Python이 필요합니다."
PY_VER="$("${SYSTEM_PYTHON}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
[[ "${PY_VER}" == "3.10" ]] || warn "시스템 Python이 ${PY_VER} 입니다. ROS2 Humble은 3.10 기준입니다."
"${SYSTEM_PYTHON}" -c "import venv, ensurepip" 2>/dev/null \
    || fail "python3-venv 가 없습니다:  sudo apt install python3-venv"

if [[ -f "${ROS_SETUP}" ]]; then
    info "ROS2 Humble 발견: ${ROS_SETUP}"
else
    warn "ROS2 Humble(${ROS_SETUP})이 없습니다. MuJoCo 튜토리얼은 동작하지만 Phase 5(ROS2 연동) 전에 설치하세요."
fi

# --- 3) venv 생성 -------------------------------------------------------------
if [[ -x "${VENV_DIR}/bin/python" ]]; then
    info "기존 가상환경 재사용: ${VENV_DIR}"
else
    info "가상환경 생성: ${VENV_DIR} (시스템 패키지 공유 모드)"
    "${SYSTEM_PYTHON}" -m venv --system-site-packages "${VENV_DIR}"
fi
# colcon(ROS2 빌드 도구)이 .venv 안의 파이썬 패키지를 ROS 패키지로 오인하지 않도록
touch "${VENV_DIR}/COLCON_IGNORE"

# --- 4) 패키지 설치 -----------------------------------------------------------
# PYTHONNOUSERSITE=1: ~/.local 에 깔린 패키지가 섞여 들어오는 것을 막음
export PYTHONNOUSERSITE=1
info "pip 업그레이드"
"${VENV_DIR}/bin/python" -m pip install --quiet --upgrade pip
info "requirements.txt 설치 (mujoco, numpy<2, biped_sim)"
(cd "${REPO_ROOT}" && "${VENV_DIR}/bin/python" -m pip install --quiet -r requirements.txt)

# --- 5) 점검 -----------------------------------------------------------------
info "환경 점검 실행"
if [[ -f "${ROS_SETUP}" ]]; then
    # ROS setup 스크립트는 정의되지 않은 변수를 참조하므로 잠시 -u 해제
    set +u; source "${ROS_SETUP}"; set -u
fi
"${VENV_DIR}/bin/python" "${REPO_ROOT}/scripts/check_env.py"

echo
info "완료! 새 터미널마다 다음을 실행하세요:"
echo "        source ${REPO_ROOT}/scripts/activate.sh"
