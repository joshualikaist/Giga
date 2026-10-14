# 11. 학습한 오리를 ROS2로 조종하기 — sim-to-sim

GPU(MJX)에서 학습한 명령 따라 걷기 정책(docs/09 §10)을 **ROS2 토픽만으로** 돌립니다. 정책 노드는 IMU·관절 상태를 받아
관절 목표를 보내므로, 상대가 MuJoCo 시뮬레이터든 실물 로봇의 서보 드라이버든 똑같이 동작합니다.
학습과 다른 시뮬레이터(일반 MuJoCo) + ROS 메시지의 지연·주기 흔들림에서도 걷는지 보는 것이 **sim-to-sim** 시험이고,
실물로 가기 전의 마지막 연습 단계입니다.

| 단계 | 내용 | 상태 |
|---|---|---|
| 1 | ROS로 정책 돌리기 (sim_node + policy_node), 키보드 조종 (key_teleop) | ✅ 이 문서 §1~§5 |
| 2 | 핸드폰 조이스틱 (같은 Wi-Fi, 브라우저만) | ✅ §2.1 |
| 3 | 직접 만드는 험지 (YAML 블록 코스, 흑백 PNG 높이맵) | |
| 4 | 조종 기록 → 넘어진 곳·조종 습관을 학습에 → 다시 확인 | |

## 1. 구조

```mermaid
flowchart LR
  K["조종기<br/>(키보드, 핸드폰)"] -- "/cmd_vel<br/>Twist" --> P["policy_node<br/>정책 (numpy) 50 Hz"]
  P -- "/joint_commands<br/>다리 10개 목표 각도" --> S["sim_node<br/>MuJoCo 오리 + 서보 PD 250 Hz"]
  S -- "/imu/data<br/>/joint_states" --> P
  S -- "/tf, /clock" --> R["기록·RViz·시험"]
```

- **sim_node** = 가짜 하드웨어. 오리는 GPU 학습과 **같은 모델**로 만듭니다 (서보 모델 kp 13.37·토크 ±3.23 N·m·관절 damping,
  서 있는 자세 무릎 0.6 rad, 발 충돌 상자, 물리 0.004 s). 정책의 학습 설정(.json)에서 읽어 오므로 따로 맞출 필요가 없습니다.
- **policy_node** = 로봇의 뇌. 관측은 학습 환경과 같은 함수(`walk_spec.base_observation`)로 만듭니다:
  IMU 자세 → 몸통에서 본 중력 방향, IMU 각속도, 관절 각도·속도, 직전 행동, 명령 (vx, vy, wz), 걸음 박자 sin·cos, 지난 3스텝 기록.
  실물에 없는 몸통 속도·높이는 쓰지 않는 정책(`--deployable-obs`)만 돌릴 수 있습니다.
- 명령은 ROS 표준 `/cmd_vel` (`geometry_msgs/Twist`, 몸통 기준: `linear.x` 앞뒤, `linear.y` 옆 [m/s], `angular.z` 회전 [rad/s]).
  학습한 범위(앞 −0.10~0.25, 옆 ±0.15, 회전 ±0.5)로 잘라서 정책에 넣고, 실제로 넣은 값은 `/policy/command`로 내보냅니다.
- 안전장치: `/cmd_vel`이 0.5초 끊기면 멈춤(제자리 걸음), 정책이 0.1초 동안 관절 명령을 안 보내면 sim_node가 감쇠 모드
  (실물 드라이버의 안전 정지), 넘어지면(기울기 > 0.6 rad) 1초 뒤 처음 자세로 (`auto_reset`, 시뮬레이션만).

## 2. 해 보기 (키보드)

```bash
source scripts/activate.sh
bash scripts/build_ros.sh && source scripts/activate.sh          # 처음 한 번 (새 노드 policy_node, key_teleop)
ros2 launch giga_sim_ros duck.launch.py                          # MuJoCo 창에 오리. 평지
```

다른 터미널에서 (**이 터미널을 클릭해 둔 채로** 키를 누름 — MuJoCo 창이 아니라):

```bash
source scripts/activate.sh
ros2 run giga_sim_ros key_teleop
```

| 키 | 동작 |
|---|---|
| ↑ / ↓ | 앞뒤 속도 +0.05 / −0.05 m/s (누를 때마다 쌓임) |
| ← / → | 회전 +0.1 / −0.1 rad/s (왼쪽 / 오른쪽) |
| PageUp / PageDown | 옆 속도 +0.04 / −0.04 m/s (왼쪽 / 오른쪽) |
| Space, Backspace | 멈춤 (제자리 걸음) |
| Home | 기본 명령 (앞 0.15) |
| r | 처음 자리로 (`/policy/reset`) |

키를 누를 때마다 터미널에 지금 명령(`명령  앞뒤 +0.10 m/s   옆 …`)이 보입니다. README 16단계(`play_walk.py --view`)와 같은 키입니다.

- **MuJoCo 창의 보기** (그 창을 클릭하고 숫자 키): `6` 따라가기(기본) · `7` 위에서 · `8` 옆에서 · `9` 지형 전체.
  위·옆 보기도 오리를 따라가고, 마우스로 돌리거나 확대해도 됩니다. 왼쪽 위에 지금 보기, 아래에 정책에 들어간 명령과 몸통 위치가
  보입니다 (MuJoCo 글꼴은 한글이 안 돼 영어). 설정 패널은 Tab / Shift+Tab.
- 따라가기 보기에서는 카메라가 오리와 함께 움직이므로, 걷는지는 **바닥 격자가 흘러가는 것**으로 보입니다 (0.1 m/s = 1초에 10 cm).
- 지형: `terrain:=hills | slopes | obstacles | mixed` (원점 주변 0.35 m는 평평한 출발 자리, 넓이 8 × 8 m),
  `terrain_difficulty:=1.5` (평가 시험 수준), `terrain_seed:=3`.
- 다른 정책: `policy:=output/learning/<실행>/policy.npz` — Brax 정책은 먼저 바꾸기:
  `python learning/export_policy.py --params output/learning/<실행>/params.pkl` (GPU 학습 환경에서)

**ROS 표준 조종기(`teleop_twist_keyboard`)를 쓰고 싶다면** — 글자 키(`i` 앞, `,` 뒤, `j`/`l` 돌기, `Shift`+`J`/`L` 옆, `k` 멈춤)를 쓰고,
키를 누를 때만 명령을 보내므로 `duck.launch.py cmd_timeout:=0`(마지막 명령 유지)으로 실행해야 합니다.
⚠ **한글 입력 상태면 동작하지 않습니다**: `i`를 누르면 `ㅑ`가 들어가고, 모르는 키는 '멈춤'으로 처리되며, 움직임 키를 눌러도
화면에 아무것도 표시되지 않아 고장처럼 보입니다 (한/영 키로 영어로 바꾸면 됨). 그래서 방향키를 쓰는 `key_teleop`을 만들었습니다.

### 2.1 핸드폰 조이스틱

```bash
ros2 launch giga_sim_ros duck.launch.py phone:=true        # terrain:=mixed 등과 함께 써도 됨
```

출력에 나오는 주소(예: `핸드폰 조이스틱: 같은 Wi-Fi의 핸드폰 브라우저에서  http://192.168.0.14:8080`)를 **PC와 같은 Wi-Fi에 연결된
핸드폰**의 브라우저에 입력합니다. 앱 설치는 필요 없습니다. 핸드폰을 **가로로** 들면 양손으로 쓰기 좋습니다.

<img src="images/phone_joystick.png" width="560">

| 화면 | 동작 |
|---|---|
| 왼쪽 스틱 | 위·아래 = 앞·뒤, 왼·오른 = 옆걸음. 끝까지 밀면 학습한 최대 명령 (앞 0.25, 뒤 0.15, 옆 0.15 m/s) × 속도 배율 |
| 오른쪽 스틱 | 왼·오른 = 왼쪽·오른쪽으로 돌기 (최대 0.5 rad/s × 속도 배율) |
| 멈춤 | 두 스틱 모두 0 |
| 처음으로 | 오리를 처음 자리로 (`/policy/reset`) |
| 속도 배율 | 스틱 끝의 명령 크기 (기본 60 %, 처음엔 작게) |

- 스틱을 놓으면 0. 핸드폰은 초당 약 15번 스틱 값을 보내고, **0.5초 동안 아무것도 오지 않으면**(화면 꺼짐, Wi-Fi 끊김) 멈춥니다.
- 화면 아래 글자: 실제로 보낸 명령과 오리 위치. "연결 끊김"이면 PC와 같은 Wi-Fi인지 확인. 그래도 안 되면 PC 방화벽
  (`sudo ufw allow 8080/tcp`) 또는 공유기의 '기기 간 통신 차단(AP 격리)' 설정.
- 웹 서버는 파이썬 기본 라이브러리(`http.server`)만 씀. 페이지가 m/s로 바꿔 보내고 노드가 학습 범위로 한 번 더 자름.

## 3. 무엇을 만들었나

| 파일 | 하는 일 |
|---|---|
| `biped_sim/envs/walk_spec.py` | 보행 과제 설정·서보 모델·학습과 같은 모델 만들기·관측 계산. **jax 없이** import 가능 (전에는 `walk_mjx.py` 안에 있어 GPU 환경에서만 됐음) |
| `biped_sim/policy.py`, `learning/export_policy.py` | Brax 정책 → numpy (`.npz`: 관측 정규화 + 신경망 층). 행동 차이 7 × 10⁻⁷ (변환할 때 확인). 실물 보드에서도 같은 방식 |
| `ros2_ws/.../sim_node.py` | `robot:=open_duck_mini`, `terrain:=…` 추가. 실제 시간 맞추기 (§4). 화면 보기 6~9, 명령·위치 표시 |
| `ros2_ws/.../policy_node.py` | 센서 토픽 + `/cmd_vel` → 관절 목표 50 Hz, `/policy/reset` |
| `ros2_ws/.../key_teleop.py` | 터미널 방향키 → `/cmd_vel` 10 Hz (한글 입력 상태에서도 동작, 지금 명령 표시) |
| `ros2_ws/.../phone_joystick.py` | 웹 페이지 조이스틱 (두 스틱·멈춤·처음으로·속도 배율) → `/cmd_vel` 20 Hz, 0.5초 끊기면 멈춤 |
| `ros2_ws/.../duck.launch.py` | 둘을 함께 실행. 정책의 학습 설정에서 로봇·서보·자세를 읽어 sim_node에 넘김 |

## 4. 결과: ROS를 거쳐도 학습한 대로 걷나

명령 11가지를 평지에서 10초씩 (2초 이후 평균, 시뮬레이션 시간 기준). **MuJoCo 단독** = ROS 없이 같은 모델에서 정책을 직접
부른 것 (`play_walk.py`와 같음), **ROS 경유** = sim_node·policy_node가 토픽으로 주고받은 것.

| 명령 (vx, vy, wz) | MuJoCo 단독 | ROS 경유 |
|---|---|---|
| 앞 0.05 | 0.045 | 0.043 |
| 앞 0.10 | 0.093 | 0.089 |
| 앞 0.20 | 0.193 | 0.183 |
| 앞 0.25 | 0.230 | 0.222 |
| 뒤 0.08 | −0.084 | −0.086 |
| 걸으며 돌기 ±0.4 | +0.360 / −0.344 | +0.343 / −0.352 |
| 제자리 돌기 ±0.4 | +0.359 / −0.355 | +0.368 / −0.364 |
| 옆 ±0.08 | +0.070 / −0.070 | +0.072 / −0.070 |

- **11가지 모두 성공** (허용 ±0.03 m/s, ±0.1 rad/s). 섞인 험지(언덕·경사로·장애물)에서도 앞 0.15 명령에 0.132 m/s로 넘어지지 않음.
- ROS를 거치면 앞으로 걷기가 조금 느려집니다 (0.193 → 0.183, 0.230 → 0.222). 정책이 받는 관절 상태가 최대 한 물리 스텝(4 ms)
  늦고 명령이 도착하는 시점이 흔들리기 때문으로 보입니다 — 실물에서는 이 지연이 더 커지므로, 4단계에서 재서 학습에 반영할 차이입니다.

**고친 문제: 화면을 켜면 시뮬레이션이 0.7배로 느려짐.** sim_node는 물리 주기(250 Hz) 타이머로 한 스텝씩 진행하는데, 화면
갱신(`viewer.sync`)이 한 번에 평균 5.5 ms씩 기다립니다 (그림자를 꺼도 같음 — 물리 한 스텝은 0.044 ms). 1초에 60번이면 0.33초를
놓치고, ROS 타이머는 놓친 호출을 따라잡지 않아 시뮬레이션 시간이 실제 시간의 0.62~0.70배가 됐습니다 (조이스틱 조종이 슬로모션).
→ 타이머가 불릴 때마다 **실제 시간이 지난 만큼** 물리 스텝을 진행 (밀렸으면 그 자리에서 따라잡음, 최대 0.1초): 0.70 → **1.00**.

## 5. 관련 파일

| 파일 | 역할 |
|---|---|
| `tests/test_ros_duck.py` | launch로 오리 + 정책 실행 → `/cmd_vel` 앞·옆 명령대로 움직임, `/policy/reset`, 깔끔한 종료. key_teleop: 키 해석, 가상 터미널에서 방향키 → `/cmd_vel`. 핸드폰 조이스틱: 페이지, HTTP 명령대로 걷기, 끊기면 멈춤, 처음으로 |
| `tests/test_open_duck.py::test_walk_spec_and_numpy_policy_run_without_jax` | ROS용 환경(jax 없음)에서 모델·numpy 정책, IMU 자세 → 중력 방향 |
| `tests/test_walk_mjx.py::test_numpy_policy_matches_brax` | numpy 정책 = Brax 정책 |
| `learning/pretrained/duck_cmd_policy.npz` (+ `.json`) | 명령 따라 걷기 챔피언 (docs/09 §10)의 numpy 판 |
