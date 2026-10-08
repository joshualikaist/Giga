# 외부 오픈소스 로봇 모델

이 폴더의 하위 폴더는 **저장소에 올리지 않습니다** (`.gitignore`). 아래 스크립트로 내려받으세요.
받은 파일은 고치지 않고, 시뮬레이션에 필요한 보완은 `biped_sim/robot_configs.py`에서 불러올 때 합니다.

| 폴더 | 받는 법 | 출처 | 라이선스 | 고정 커밋 |
|---|---|---|---|---|
| `open_duck_mini_v2/` | `bash scripts/get_open_duck.sh` | https://github.com/apirrone/Open_Duck_Mini (`mini_bdx/robots/open_duck_mini_v2/`) | Apache-2.0 (폴더 안 `LICENSE`) | `b23317a` |

Open Duck Mini는 디즈니 BD-X 드로이드를 본뜬 팬 프로젝트입니다. 외형을 상업적으로 쓰려면 따로 확인이 필요합니다.
불러오는 과정과 확인 결과: [docs/09_open_source_robot.md](../../docs/09_open_source_robot.md)
