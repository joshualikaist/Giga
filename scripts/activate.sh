# =============================================================================
# activate.sh — 새 터미널마다 실행하는 환경 활성화 스크립트
#
# 사용법:  source scripts/activate.sh        (← "bash"나 "./"가 아니라 "source"!)
#
# 순서가 중요합니다:
#   1) ROS2 Humble setup  → rclpy, 메시지 패키지 경로(PYTHONPATH)와 ros2 명령 추가
#   2) .venv activate     → python/pip가 가상환경 것으로 바뀜 (mujoco 사용 가능)
#   3) PYTHONNOUSERSITE=1 → ~/.local 패키지가 섞이지 않게 차단
# =============================================================================

# source로 실행했는지 확인 (./activate.sh 로 실행하면 현재 터미널에 아무 효과가 없음)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "이 스크립트는 'source scripts/activate.sh' 로 실행해야 합니다."
    exit 1
fi

_BIPED_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -n "${CONDA_PREFIX:-}" ]]; then
    echo "[warn] conda 환경(${CONDA_PREFIX})이 켜져 있습니다. 'conda deactivate' 후 다시 source 하세요."
    return 1
fi

if [[ ! -x "${_BIPED_ROOT}/.venv/bin/python" ]]; then
    echo "[error] .venv가 없습니다. 먼저 'bash scripts/setup_env.sh'를 실행하세요."
    return 1
fi

if [[ -f /opt/ros/humble/setup.bash ]]; then
    source /opt/ros/humble/setup.bash
else
    echo "[warn] ROS2 Humble이 없어 ROS 설정은 건너뜁니다 (MuJoCo 튜토리얼은 동작)."
fi

source "${_BIPED_ROOT}/.venv/bin/activate"
export PYTHONNOUSERSITE=1

echo "[biped_sim] 환경 활성화 완료"
echo "  python : $(command -v python)"
echo "  ROS    : ${ROS_DISTRO:-없음}"
echo "  repo   : ${_BIPED_ROOT}"
unset _BIPED_ROOT
