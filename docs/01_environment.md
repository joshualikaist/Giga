# 01. 개발 환경 — 무엇을, 왜 이렇게 설치했나

## 1. 설치 (처음 1회)

```bash
cd ~/Giga                                 # clone한 저장소 루트 (본인 경로로)
bash scripts/install_system_deps.sh       # [sudo] ROS2 Humble 등 apt 패키지 (--dry-run으로 미리 보기)
conda deactivate                          # conda가 켜져 있다면 (프롬프트에 (base) 등이 보이면) 끄기
bash scripts/setup_env.sh                 # .venv 생성 + 패키지 설치 + 환경 점검
```

처음부터 끝까지의 단계별 안내(확인 방법 포함)는 [README의 "처음 시작하기"](../README.md)를 보세요.

## 2. 매 터미널마다

```bash
source scripts/activate.sh     # ROS2 Humble + .venv + PYTHONNOUSERSITE=1 + ros2_ws + ROS_DOMAIN_ID=27 을 한 번에
python scripts/check_env.py    # (문제 있을 때) 환경 점검
```

> `bash scripts/activate.sh` 나 `./scripts/activate.sh` 로 실행하면 **아무 효과가 없습니다**.
> 새로 뜬 하위 셸에만 적용되고 바로 사라지기 때문입니다. 반드시 `source`.

## 3. 검증된 환경 (2026-10 기준)

| 항목 | 값 | 비고 |
|---|---|---|
| OS | Ubuntu 22.04.5 LTS | ROS2 Humble 공식 지원 OS |
| Python | 3.10.12 (`/usr/bin/python3`) | ROS2 Humble이 이 Python으로 빌드됨 |
| ROS2 | Humble (`/opt/ros/humble`) | rclpy, urdf, robot_state_publisher, rviz2 설치됨 |
| MuJoCo | 3.15.0 (pip) | `requirements.txt`에 고정 |
| numpy | 1.21.5 (Ubuntu apt 기본) | `numpy<2` 로 제한 |
| GPU | GTX 1650, NVIDIA 535 드라이버, OpenGL 4.6 | 뷰어(GLFW)·오프스크린(EGL) 모두 동작 |

## 4. 왜 conda가 아니라 venv인가?

ROS2 Humble의 `rclpy`와 메시지 패키지(`sensor_msgs` 등)는 **시스템 Python 3.10용 C 확장 모듈**로 빌드되어 `/opt/ros/humble`에 설치되어 있습니다.

- conda 환경의 Python(버전·빌드가 다름)에서는 이 C 확장을 불러올 수 없거나, 불러와도 미묘하게 깨집니다.
- 그래서 **시스템 Python으로 가상환경(venv)** 을 만들고, `--system-site-packages` 옵션으로 시스템에 깔린 패키지(numpy, matplotlib, ROS가 의존하는 것들)를 그대로 보이게 했습니다.
- MuJoCo만 venv 안에 pip로 설치합니다.

```
python (= .venv/bin/python, 실체는 /usr/bin/python3)
 ├─ .venv/lib/python3.10/site-packages     ← mujoco, biped_sim(편집 가능 설치)
 ├─ /opt/ros/humble/...                    ← rclpy, sensor_msgs (activate.sh가 PYTHONPATH에 추가)
 └─ /usr/lib/python3/dist-packages         ← numpy 1.21.5, matplotlib, pytest, yaml (system-site-packages)
```

## 5. 왜 numpy < 2 인가?

ROS2 Humble의 메시지 모듈(C 확장)은 **numpy 1.x의 C API(ABI)** 로 컴파일되어 있습니다.
numpy 2.x는 C ABI가 바뀌어서, numpy 2가 설치되면 ROS 메시지를 만들거나 받을 때 import 에러/크래시가 납니다.
`pip install` 하다가 다른 패키지가 numpy 2를 끌고 들어오지 않도록 `requirements.txt`와 `pyproject.toml`에 `numpy<2`를 명시했습니다.

## 6. 왜 PYTHONNOUSERSITE=1 인가?

`--system-site-packages` venv에서는 `~/.local/lib/python3.10/site-packages` (예전에 `pip install --user`로 깐 것들)도 보입니다.
이 PC에도 torch 등 100개 이상의 패키지가 있었습니다. 지금 당장은 충돌이 없어도, 언젠가 거기 들어간 numpy 2 같은 것이
조용히 우선순위를 잡으면 원인을 찾기 매우 어렵습니다. `activate.sh`가 `PYTHONNOUSERSITE=1`로 이를 차단합니다.

> `PYTHONPATH`에 직접 추가한 경로(예: `~/.bashrc`의 다른 프로젝트 경로)는 여전히 섞입니다.
> `check_env.py`가 ROS 외의 `PYTHONPATH` 항목을 WARN으로 보여 주니, 이상한 import 에러가 나면 먼저 의심하세요.

## 7. `.venv/COLCON_IGNORE` 는 뭔가?

ROS2 빌드 도구 `colcon`은 폴더를 뒤져 패키지를 찾는데, `.venv` 안의 파이썬 패키지들을 ROS 패키지로 오인할 수 있습니다.
빈 파일 `COLCON_IGNORE`가 있으면 colcon이 그 폴더를 건너뜁니다 (ROS2 공식 문서의 venv 사용 권장 방식).

## 8. 화면 없는 환경 (원격 서버, SSH)

| 상황 | 방법 |
|---|---|
| 뷰어 없이 실행 | 모든 튜토리얼에 `--headless` 옵션 |
| 이미지 저장(스냅샷) | `export MUJOCO_GL=egl` (NVIDIA/Intel GPU) 또는 `MUJOCO_GL=osmesa` (CPU) 후 `--snapshot` |
| 기본값 | `MUJOCO_GL` 미지정 시 GLFW(화면 필요) |

## 9. 문제 해결

| 증상 | 원인 / 해결 |
|---|---|
| `ModuleNotFoundError: mujoco` | venv가 안 켜짐 → `source scripts/activate.sh` |
| `ModuleNotFoundError: rclpy` | ROS setup이 안 됨 → `source scripts/activate.sh` (내부에서 `/opt/ros/humble/setup.bash`) |
| `activate.sh`가 conda 경고 | `conda deactivate` (여러 번 필요할 수 있음) 후 다시 source |
| numpy 관련 `ImportError`/크래시 (ROS 메시지) | `python -c "import numpy; print(numpy.__version__, numpy.__file__)"` 로 numpy 2가 섞였는지 확인 → `pip install "numpy<2"` |
| 뷰어 창이 안 뜸 / GLFW 에러 | `echo $DISPLAY`가 비어 있으면 화면이 없는 것 → `--headless` 사용 |
| `ros2 run giga_sim_ros sim_node` → `No module named 'mujoco'` | 그냥 `colcon build`로 빌드한 것. `bash scripts/build_ros.sh`로 다시 빌드 (docs/05 §2) |
| `ros2 run`에서 `Package 'giga_sim_ros' not found` | 빌드 안 했거나 빌드 후 `source scripts/activate.sh`를 안 함 |
| 내 `ros2 topic list`에 모르는 토픽이 보임 | 같은 네트워크의 다른 PC와 `ROS_DOMAIN_ID`가 같음. `activate.sh`가 27로 맞추므로, 그 터미널에서 `source scripts/activate.sh`를 했는지 확인 |
| 시뮬레이터 토픽이 다른 터미널에서 안 보임 | 그 터미널에서 `source scripts/activate.sh`를 안 해서 `ROS_DOMAIN_ID`가 0 (시뮬레이터는 27) |
| 직접 만든 스크립트에서 뷰어를 닫을 때 세그폴트(exit 139) / `GLXBadContext` / 멈춤 | `mujoco.viewer.launch_passive` 대신 `biped_sim.passive_viewer` 사용 (그리기 스레드가 끝날 때까지 기다림) |
| 뷰어 실행 시 `Xlib: extension "NV-GLX" missing on display ":1"` | 무해한 메시지 (원격/가상 디스플레이에서 흔함). 뷰어는 정상 동작 |
| 종료 시 `EGLError ... Renderer.__del__` 경고 | 렌더러를 명시적으로 닫지 않아 생기는 무해한 경고. `biped_sim.save_snapshot`은 `close()`로 처리함 |
| VS Code에서 `rclpy`에 빨간 줄 | 에디터가 ROS 경로를 모르는 것뿐(실행은 정상). Python 인터프리터를 `.venv/bin/python`으로 고르고, 필요하면 `python.analysis.extraPaths`에 `/opt/ros/humble/lib/python3.10/site-packages`와 `/opt/ros/humble/local/lib/python3.10/dist-packages` 추가 |
| `ros2 run ... -p robot_description:="$(cat x.urdf)"` 가 파싱 에러 | URDF 안의 `:` 등을 YAML이 해석해서 생기는 문제. 파라미터 파일(`--params-file`)이나 launch 파일로 넘길 것 (`tests/test_tutorials_and_ros.py` 참고) |
