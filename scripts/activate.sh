# =============================================================================
# activate.sh — 새 터미널마다 실행하는 환경 활성화 스크립트
#
# 사용법:  source scripts/activate.sh        (← "bash"나 "./"가 아니라 "source"!)
#
# 순서가 중요합니다:
#   1) ROS2 Humble setup  → rclpy, 메시지 패키지 경로(PYTHONPATH)와 ros2 명령 추가
#   2) .venv activate     → python/pip가 가상환경 것으로 바뀜 (mujoco 사용 가능)
#   3) PYTHONNOUSERSITE=1 → ~/.local 패키지가 섞이지 않게 차단
#   4) ros2_ws overlay    → 빌드돼 있으면 우리 ROS 패키지(giga_sim_ros 등)를 ros2 run으로 실행 가능
#   5) ROS_DOMAIN_ID=27   → 같은 네트워크의 다른 ROS2 PC와 토픽이 섞이지 않게 하는 "채널 번호"
#                           (이미 지정돼 있으면 그 값을 그대로 씀. docs/06_ros2_hands_on.md §1)
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

# 5) ROS2 통신 채널 번호. 같은 번호끼리만 서로 보인다 (0 = 모든 ROS2 PC의 기본값이라 피함).
#    이 프로젝트의 모든 터미널은 이 스크립트를 source하므로 자동으로 같은 번호가 된다.
#    바꾸고 싶으면 source 전에 export ROS_DOMAIN_ID=<0~101> 을 해 두면 그 값이 우선한다.
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-27}"

# 4) 우리 ROS2 워크스페이스 (bash scripts/build_ros.sh로 빌드한 뒤에만 존재)
#    local_setup.bash: 이 워크스페이스만 추가 (setup.bash는 /opt/ros를 다시 source함)
_WS_SETUP="${_BIPED_ROOT}/ros2_ws/install/local_setup.bash"
if [[ -n "${ROS_DISTRO:-}" && -f "${_WS_SETUP}" ]]; then
    source "${_WS_SETUP}"
    _WS_STATUS="ros2_ws 로드됨"
else
    _WS_STATUS="ros2_ws 아직 빌드 안 됨 (bash scripts/build_ros.sh)"
fi

echo "[biped_sim] 환경 활성화 완료"
echo "  python : $(command -v python)"
echo "  ROS    : ${ROS_DISTRO:-없음} (ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0})"
echo "  ros2_ws: ${_WS_STATUS}"
echo "  repo   : ${_BIPED_ROOT}"
unset _BIPED_ROOT _WS_SETUP _WS_STATUS
