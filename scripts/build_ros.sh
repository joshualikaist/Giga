#!/usr/bin/env bash
# =============================================================================
# build_ros.sh — ROS2 워크스페이스(ros2_ws) 빌드
#
# 사용법:  source scripts/activate.sh   (먼저)
#          bash scripts/build_ros.sh     (추가 인자는 colcon에 전달. 예: --packages-select giga_sim_ros)
#
# 왜 그냥 `colcon build`가 아닌가?
#   colcon은 자기가 실행된 Python으로 노드 실행 파일의 첫 줄(#!)을 만든다.
#     - `colcon build`           → #!/usr/bin/python3        → 노드 실행 시 "No module named 'mujoco'"
#     - `python -m colcon build` → #!/…/.venv/bin/python     → mujoco, biped_sim 사용 가능
#   (2026-10-08 이 PC에서 둘 다 실제로 확인. docs/05_ros2_bridge_plan.md §2)
# =============================================================================
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"

if [[ "${VIRTUAL_ENV:-}" != "${VENV_DIR}" ]]; then
    echo "[error] 프로젝트 venv가 켜져 있지 않습니다. 먼저:  source scripts/activate.sh"
    exit 1
fi
if [[ -z "${ROS_DISTRO:-}" ]]; then
    echo "[error] ROS2 환경이 없습니다. 먼저:  source scripts/activate.sh"
    exit 1
fi

cd "${REPO_ROOT}/ros2_ws"
# --symlink-install: 파이썬 파일을 고치면 다시 빌드하지 않아도 바로 반영됨
python -m colcon build --symlink-install "$@"

echo
echo "[build_ros] 완료. 현재 터미널에 반영하려면:  source scripts/activate.sh"
