# =============================================================================
# activate_gpu.sh — GPU 학습 환경(.venv-mjx) 켜기. 사용법:  source scripts/activate_gpu.sh
#
# ROS2용 activate.sh와 따로 씁니다. ROS를 켜지 않은 새 터미널에서 실행하는 것을 권장합니다.
# (같은 터미널에서 ROS를 켜 둔 경우, ROS의 PYTHONPATH가 섞이지 않게 이 터미널에서만 비웁니다)
# =============================================================================
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "이 스크립트는 'source scripts/activate_gpu.sh' 로 실행해야 합니다."
    exit 1
fi
_GIGA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -x "${_GIGA_ROOT}/.venv-mjx/bin/python" ]]; then
    echo "[error] .venv-mjx가 없습니다. 먼저:  bash scripts/setup_gpu_learning.sh"
    return 1
fi
if [[ -n "${ROS_DISTRO:-}" ]]; then
    echo "[warn] 이 터미널에 ROS2가 켜져 있어 PYTHONPATH를 비웁니다 (이 터미널에서는 ROS 명령을 쓰지 마세요)."
fi
[[ -n "${VIRTUAL_ENV:-}" ]] && type deactivate >/dev/null 2>&1 && deactivate
unset PYTHONPATH
source "${_GIGA_ROOT}/.venv-mjx/bin/activate"
export PYTHONNOUSERSITE=1
# JAX는 기본으로 GPU 메모리의 75 %를 미리 잡는다. 4 GB급 GPU에서 화면 등과 나눠 쓰도록 70 %로
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.7}"
echo "[biped_sim] GPU 학습 환경 활성화: $(command -v python)"
unset _GIGA_ROOT
