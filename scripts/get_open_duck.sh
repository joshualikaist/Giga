#!/usr/bin/env bash
# =============================================================================
# get_open_duck.sh — 오픈소스 오리 로봇 Open Duck Mini v2의 URDF와 메시를 내려받는다 (선택, 1회, 약 19 MB)
#
# 사용법:  bash scripts/get_open_duck.sh
# 결과:    models/third_party/open_duck_mini_v2/  (robot.urdf, STL 45개, LICENSE) — git에는 올리지 않음
#
# 출처: https://github.com/apirrone/Open_Duck_Mini  (Apache-2.0, 디즈니 BD-X 드로이드를 본뜬 팬 프로젝트)
#   - 언제 받아도 같은 파일이 되도록 커밋을 고정한다 (COMMIT). 바꾸면 docs/09의 확인 결과도 다시 확인할 것.
#   - 받은 파일은 고치지 않는다. 시뮬레이션에 필요한 수정은 불러올 때 biped_sim이 한다
#     (biped_sim/robot_configs.py의 OPEN_DUCK_MINI, docs/09_open_source_robot.md).
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${REPO_ROOT}/models/third_party/open_duck_mini_v2"
COMMIT="b23317a485b3cec7d8417f352478778b3475173c"   # v2 브랜치, 2026-01-31
BASE="https://raw.githubusercontent.com/apirrone/Open_Duck_Mini/${COMMIT}"

info() { echo -e "\033[1;34m[open-duck]\033[0m $*"; }
fail() { echo -e "\033[1;31m[error]\033[0m $*"; exit 1; }
command -v curl >/dev/null || fail "curl이 없습니다: sudo apt install curl"

mkdir -p "${DEST}"
cd "${DEST}"
info "받는 곳: ${DEST}"
info "커밋: ${COMMIT}"
curl -fsSL -o LICENSE "${BASE}/LICENSE" || fail "LICENSE를 받지 못했습니다 (인터넷 연결 확인)"
curl -fsSL -o robot.urdf "${BASE}/mini_bdx/robots/open_duck_mini_v2/robot.urdf"

# URDF가 쓰는 메시 파일 목록: <mesh filename="package:///이름.stl"/>
mapfile -t MESHES < <(grep -o 'package:///[^"]*' robot.urdf | sed 's#package:///##' | sort -u)
info "메시 ${#MESHES[@]}개 받는 중..."
for f in "${MESHES[@]}"; do
    [[ -s "${f}" ]] && continue   # 이미 있으면 건너뜀 (다시 실행해도 안전)
    curl -fsSL -o "${f}" "${BASE}/mini_bdx/robots/open_duck_mini_v2/${f}" || fail "${f} 받기 실패"
done
echo "${COMMIT}" > COMMIT

info "완료: robot.urdf + STL ${#MESHES[@]}개 + LICENSE ($(du -sh . | cut -f1))"
info "다음:  python tutorials/t07_open_duck.py --headless   (docs/09_open_source_robot.md)"
