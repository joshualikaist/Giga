#!/usr/bin/env bash
# =============================================================================
# install_system_deps.sh — 시스템(apt) 패키지 설치. 관리자 권한(sudo)이 필요합니다. PC당 1번.
#
# 사용법:
#     bash scripts/install_system_deps.sh            # 할 일을 보여 주고, 확인(y)을 받은 뒤 설치
#     bash scripts/install_system_deps.sh --dry-run  # 아무것도 바꾸지 않고 할 일만 보여 줌
#
# 하는 일 (ROS2 Humble 공식 설치 문서를 따름):
#   ROS2가 이미 있으면 → 빠진 추가 패키지만 설치
#   ROS2가 없으면     → ① universe 저장소 ② ROS2 apt 저장소 등록(ros2-apt-source)
#                       ③ apt update + apt upgrade ④ ROS2 Humble Desktop + 추가 패키지 설치
#
# 공식 문서: https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html
#   (주의) 새로 설치한 Ubuntu 22.04에서는 ROS2 설치 전에 apt upgrade를 해야 합니다.
#          안 하면 중요한 시스템 패키지가 지워질 수 있다고 공식 문서가 경고합니다.
# =============================================================================
set -eo pipefail

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

ROS_PREFIX="/opt/ros/humble"
# 이 프로젝트에 필요한 apt 패키지
EXTRA_PACKAGES=(
    git                                   # 저장소 받기
    python3-venv python3-pip              # 파이썬 가상환경 (.venv)
    ros-dev-tools                         # colcon 등 ROS 빌드 도구
    ros-humble-joint-state-publisher-gui  # RViz에서 관절 슬라이더 (display.launch.py)
    ros-humble-xacro                      # URDF 매크로 도구 (실제 로봇 URDF에서 자주 씀)
    ffmpeg                                # 걸음 영상(mp4) 저장 (learning/analyze_gait.py)
)

info() { echo -e "\033[1;34m[deps]\033[0m $*"; }
warn() { echo -e "\033[1;33m[warn]\033[0m $*"; }
fail() { echo -e "\033[1;31m[error]\033[0m $*"; exit 1; }
run()  { echo "    \$ $*"; if [[ ${DRY_RUN} -eq 0 ]]; then eval "$@"; fi; }

# --- 0) OS 확인 --------------------------------------------------------------
. /etc/os-release
if [[ "${VERSION_CODENAME:-}" != "jammy" ]]; then
    fail "Ubuntu 22.04 (jammy)가 필요합니다. 현재: ${PRETTY_NAME:-알 수 없음}. ROS2 Humble은 22.04용입니다."
fi
info "OS: ${PRETTY_NAME}"

if ! locale | grep -qi "utf-8"; then
    warn "locale이 UTF-8이 아닙니다. 공식 문서의 'Set locale' 절차를 먼저 하세요:"
    echo "    sudo apt update && sudo apt install locales"
    echo "    sudo locale-gen en_US en_US.UTF-8"
    echo "    sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8"
    echo "    export LANG=en_US.UTF-8"
fi

# --- 1) 무엇을 할지 결정 -------------------------------------------------------
if [[ -f "${ROS_PREFIX}/setup.bash" ]]; then
    ROS_INSTALLED=1
    info "ROS2 Humble이 이미 설치되어 있습니다 (${ROS_PREFIX}). 빠진 추가 패키지만 설치합니다."
else
    ROS_INSTALLED=0
    info "ROS2 Humble이 없습니다. 공식 절차대로 ROS2 Humble Desktop부터 설치합니다."
fi

MISSING=()
for pkg in "${EXTRA_PACKAGES[@]}"; do
    dpkg -s "${pkg}" >/dev/null 2>&1 || MISSING+=("${pkg}")
done
[[ ${ROS_INSTALLED} -eq 0 ]] && MISSING=(ros-humble-desktop "${MISSING[@]}")

if [[ ${#MISSING[@]} -eq 0 ]]; then
    info "필요한 패키지가 모두 설치되어 있습니다. 할 일이 없습니다. ✅"
    exit 0
fi
info "설치할 패키지: ${MISSING[*]}"

if [[ ${DRY_RUN} -eq 1 ]]; then
    info "--dry-run: 아래 명령을 보여 주기만 하고 실행하지 않습니다."
else
    echo
    read -r -p "위 패키지를 설치합니다 (관리자 비밀번호를 물어볼 수 있습니다). 진행할까요? [y/N] " answer
    [[ "${answer}" =~ ^[Yy]$ ]] || { echo "취소했습니다."; exit 0; }
fi

# --- 2) ROS2 apt 저장소 등록 (ROS2가 없을 때만) -----------------------------------
if [[ ${ROS_INSTALLED} -eq 0 ]]; then
    if grep -rqs "packages.ros.org/ros2" /etc/apt/sources.list.d/; then
        info "ROS2 apt 저장소는 이미 등록되어 있습니다."
    else
        info "① Ubuntu universe 저장소 켜기"
        run "sudo apt install -y software-properties-common"
        run "sudo add-apt-repository -y universe"
        info "② ROS2 apt 저장소 등록 (ros2-apt-source 패키지, 공식 방법)"
        run "sudo apt update && sudo apt install -y curl"
        run "export ROS_APT_SOURCE_VERSION=\$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F 'tag_name' | awk -F'\"' '{print \$4}')"
        run "curl -L -o /tmp/ros2-apt-source.deb \"https://github.com/ros-infrastructure/ros-apt-source/releases/download/\${ROS_APT_SOURCE_VERSION}/ros2-apt-source_\${ROS_APT_SOURCE_VERSION}.\$(. /etc/os-release && echo \${UBUNTU_CODENAME:-\${VERSION_CODENAME}})_all.deb\""
        run "sudo dpkg -i /tmp/ros2-apt-source.deb"
    fi
    info "③ 패키지 목록 갱신 + 시스템 업데이트 (공식 문서의 필수 경고 사항)"
    run "sudo apt update"
    run "sudo apt upgrade -y"
else
    info "패키지 목록 갱신"
    run "sudo apt update"
fi

# --- 3) 설치 -------------------------------------------------------------------
info "④ 설치"
run "sudo apt install -y ${MISSING[*]}"

if [[ ${DRY_RUN} -eq 1 ]]; then
    info "(dry-run) 패키지 이름 확인: apt 모의 설치 결과"
    apt-get -s install "${MISSING[@]}" 2>&1 | grep -E "^E:|is already the newest|^Inst" | head -20 || true
    exit 0
fi

echo
info "완료! 다음 단계:  bash scripts/setup_env.sh"
