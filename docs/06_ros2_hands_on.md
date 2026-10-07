# 06. ROS2 실행 가이드 — 무엇을 실행하면 무슨 일이 일어나나

> 이 문서의 명령·출력·수치는 2026-10-08 이 PC에서 실제로 실행해 확인한 것입니다.
> 같은 동작이 `tests/test_ros_sim_node.py`, `tests/test_ros_launch.py`로 자동 테스트됩니다.

## 0. 한눈에 보기

| 하고 싶은 것 | 명령 | 화면에 보이는 것 |
|---|---|---|
| **A.** 로봇 모델(URDF)만 확인 | `ros2 launch giga_description display.launch.py` | RViz + 관절 슬라이더 창. 슬라이더를 움직이면 로봇이 따라 움직임 (물리 없음) |
| **B.** 시뮬레이터 + RViz | `ros2 launch giga_sim_ros sim.launch.py` | MuJoCo 창(물리) + RViz 창. 로봇이 바닥에 서 있음 |
| **C.** 제어기가 로봇을 움직임 | `ros2 launch giga_sim_ros sim.launch.py controller:=squat` | 로봇이 4초 주기로 앉았다 일어섬. MuJoCo와 RViz가 똑같이 움직임 |
| **D.** 공중에 매단 로봇 | `ros2 launch giga_sim_ros sim.launch.py fixed_base:=true` | 넘어질 걱정 없이 관절만 시험 |
| **E.** 화면 없이 | `ros2 launch giga_sim_ros sim.launch.py viewer:=false rviz:=false` | 창 없음 (원격 PC, 자동 테스트) |
| **F.** 터미널로 직접 조종 | `ros2 run giga_sim_ros sim_node` + `ros2 topic pub ...` | 명령 한 줄마다 관절이 움직임 (§3 실습) |

인자는 섞어 쓸 수 있습니다. 예: `ros2 launch giga_sim_ros sim.launch.py fixed_base:=true controller:=squat rviz:=false`

## 1. 준비

### 처음 한 번

```bash
cd ~/Giga                          # 저장소 루트 (본인 경로)
source scripts/activate.sh
bash scripts/build_ros.sh          # ros2_ws 빌드 (그냥 colcon build 금지 — 아래 주의)
```

> `build_ros.sh`는 내부에서 `python -m colcon build`를 실행합니다. 그냥 `colcon build`로 빌드하면
> 노드가 시스템 Python으로 실행되어 `No module named 'mujoco'` 에러가 납니다 (확인함).
> Python 파일만 고쳤다면 다시 빌드할 필요 없습니다 (`--symlink-install`). 새 파일·launch를 추가했을 때만 다시 빌드하세요.

### 새 터미널마다

```bash
source scripts/activate.sh
```

```
[biped_sim] 환경 활성화 완료
  python : /home/joshuali/giga/.venv/bin/python
  ROS    : humble (ROS_DOMAIN_ID=27)
  ros2_ws: ros2_ws 로드됨
```

`ros2_ws 로드됨`이 보여야 `ros2 launch giga_sim_ros ...`가 동작합니다.

### `ROS_DOMAIN_ID=27`은 무엇인가?

ROS2 노드들은 **같은 네트워크에 있으면 서로를 자동으로 찾아서** 토픽을 주고받습니다 (설정 없이 되는 대신, 원치 않는 상대와도 연결됨).
`ROS_DOMAIN_ID`는 일종의 **채널 번호**로, 번호가 같은 노드끼리만 서로 보입니다.

- 기본값은 0이라, 연구실 Wi-Fi에서 다른 사람도 ROS2를 기본값으로 쓰면 그 사람의 토픽이 내 `ros2 topic list`에 보이고, 반대로 내 시뮬레이터 명령이 남의 로봇에 갈 수도 있습니다.
- 그래서 `scripts/activate.sh`가 **27**로 맞춰 둡니다. 0~101 중 아무 숫자나 되는데, 남들이 잘 안 쓰는 숫자를 고른 것뿐입니다
  (2026-10-08 확인 당시 이 네트워크의 0, 1, 2, 3, 5, 7, 10, 11, 13, 17, 21, 30, 42번에는 ROS2 노드가 없었습니다).
- 이 프로젝트의 터미널은 모두 `activate.sh`를 쓰므로 자동으로 같은 번호가 됩니다.
  `activate.sh` 없이 연 터미널(번호 0)에서는 시뮬레이터 토픽이 **보이지 않습니다.**
- 바꾸고 싶으면 `source` 전에 `export ROS_DOMAIN_ID=<번호>`를 해 두면 그 값이 우선합니다.
- 나중에 **실제 로봇의 컴퓨터와 통신할 때는 로봇 쪽도 같은 번호**여야 합니다.

## 2. 실행 시나리오별로 일어나는 일

### A. 로봇 모델(URDF)만 확인 — `display.launch.py`

```bash
ros2 launch giga_description display.launch.py              # 슬라이더 창 + RViz
ros2 launch giga_description display.launch.py gui:=false   # 슬라이더 없이 (모든 관절 0)
```

**보이는 것**: RViz 창(회색 몸통, 파란 왼다리, 빨간 오른다리)과 "Joint State Publisher" 슬라이더 창(관절 6개).
슬라이더를 움직이면 RViz의 다리가 따라 움직입니다. **물리 계산은 없습니다** — 중력도, 바닥도, 넘어짐도 없음.

**내부 동작**

```
joint_state_publisher_gui ──/joint_states──▶ robot_state_publisher ──/tf (다리 링크 위치)──▶ RViz
     (슬라이더 값)                              (URDF로 각 링크 위치 계산) ──/robot_description──▶ RViz (모양)
```

**언제 쓰나**: URDF를 고친 뒤(특히 실제 로봇 URDF로 바꿀 때) 관절 축 방향, 부호, 가동 범위가 맞는지 눈으로 확인할 때.
고정 좌표계가 몸통(`base_link`)이라 몸통은 화면에 고정되고 바닥 격자는 몸통 높이에 그려집니다.

**종료**: 터미널에서 `Ctrl+C`.
(시작할 때 나오는 `KDL does not support a root link with an inertia` 경고는 무해합니다 — [03 §4](03_urdf_and_mujoco.md))

### B. 시뮬레이터 + RViz — `sim.launch.py`

```bash
ros2 launch giga_sim_ros sim.launch.py
```

**보이는 것**: 창 2개.
- **MuJoCo 창** = 물리 세계 그 자체. 로봇이 바닥에 무릎을 살짝 굽히고 서 있습니다.
- **RViz 창** = ROS가 알고 있는 로봇 상태. MuJoCo와 **같은 자세**여야 정상입니다.
  고정 좌표계가 바닥(`odom`)이라 로봇이 넘어지거나 움직이면 RViz에서도 그대로 보입니다.

**내부 동작** (실행되는 노드 3개)

```
            ┌──────────────── sim_node (MuJoCo, 500 Hz) ────────────────┐
            │ 명령 없으면 home 자세를 PD로 유지                          │
            └─┬──────────────┬──────────────┬───────────────┬──────────┘
   /joint_states 500 Hz   /imu/data 500 Hz   /tf odom→base_link 100 Hz   /clock
              │                                     │
              ▼                                     ▼
   robot_state_publisher ── /tf (다리 링크들) ──▶ RViz (use_sim_time=true)
```

**해 볼 것**: MuJoCo 창에서 몸통을 **더블클릭 → Ctrl + 마우스 오른쪽 드래그**로 밀어 넘어뜨려 보세요. RViz의 로봇도 똑같이 넘어집니다.
다시 세우려면 다른 터미널에서 `ros2 service call /sim/reset std_srvs/srv/Trigger`.

**종료**: launch 터미널에서 `Ctrl+C` **또는 MuJoCo 창 닫기** → 모든 노드가 함께 종료됩니다.

### C. 제어기가 로봇을 움직임 — `controller:=squat` / `controller:=stand`

```bash
ros2 launch giga_sim_ros sim.launch.py controller:=squat    # 앉았다 일어서기
ros2 launch giga_sim_ros sim.launch.py controller:=stand    # 서 있기
```

**보이는 것 (squat)**: 로봇이 4초 주기로 천천히 앉았다 일어섭니다. MuJoCo와 RViz가 똑같이 움직입니다.
실측: 몸통 높이 0.590 m ↔ 0.505 m 반복, 몸통 기울기 최대 1.4°, 넘어지지 않음.

**내부 동작**: B에 `demo_controller` 노드가 하나 더 붙습니다.

```
demo_controller ── /joint_commands (100 Hz: 목표 각도+속도) ──▶ sim_node ── PD 토크 → MuJoCo 물리
       ▲                                                           │
       └──────────────────── /joint_states (500 Hz) ──────────────┘
```

- `demo_controller`는 **mujoco도 biped_sim도 모릅니다.** 토픽과 관절 이름만 압니다.
  → 나중에 `sim_node` 자리에 실제 로봇 드라이버 노드가 같은 토픽을 제공하면 이 제어기를 **그대로** 쓸 수 있습니다.
- 터미널 로그 순서: `motion=squat, 100 Hz. /joint_states를 기다리는 중...` → `로봇 발견, 관절 6개 → 명령 시작`
- 제어기를 붙이면 `sim_node`의 **안전장치(명령 타임아웃 50 ms)** 가 자동으로 켜집니다 (G 참고).
- 앉는 깊이·속도는 파라미터로 바꿀 수 있습니다:
  `ros2 run giga_sim_ros demo_controller --ros-args -p motion:=squat -p squat_bend:=0.75 -p squat_period:=2.0 -p use_sim_time:=true`
  (이 경우 launch는 `controller:=none`으로 띄우고, 다른 터미널에서 위 명령 실행)

### D. 공중에 매단 로봇 — `fixed_base:=true`

```bash
ros2 launch giga_sim_ros sim.launch.py fixed_base:=true
ros2 launch giga_sim_ros sim.launch.py fixed_base:=true controller:=squat
```

몸통이 높이 1 m에 고정되어 넘어지지 않습니다. 관절 명령·제어기를 처음 시험할 때 안전합니다 (실제 로봇도 처음엔 거치대에 매달고 시험).
이 모드에서 IMU 가속도는 (0, 0, 9.81)로 나옵니다 — 받침대에 고정된 실제 IMU와 같은 값입니다.

### E. 화면 없이 — `viewer:=false rviz:=false`

창 없이 모든 노드가 돕니다. 원격 PC(SSH)나 자동 테스트용. 동작 확인은 §3의 `ros2 topic ...` 명령으로 합니다.

### F. 노드를 직접 실행하고 터미널로 조종 — §3 실습

launch 없이 `sim_node` 하나만 띄우고, 사람이 `ros2 topic pub`으로 명령을 보냅니다.
"제어기 노드가 하는 일을 사람이 손으로 해 보는 것"으로, ROS2 기초를 익히는 데 가장 좋습니다.

### G. 안전장치 — 제어기가 갑자기 죽으면?

C처럼 제어기와 함께 실행 중에 `demo_controller`가 죽으면(실제로는 제어 PC 다운, 통신 끊김):

```
[sim_node]: 50 ms 동안 명령 없음 → 감쇠 모드 (Kp=0, Kd만 작동)
```

`sim_node`는 버티는 힘(Kp)을 빼고 관절을 천천히 늘어뜨립니다 (실제 모터 드라이버의 안전 정지와 같은 동작).
**서 있던 로봇은 주저앉습니다** (실측: 몸통 높이 0.588 → 0.09 m). 2족 로봇에서 "제어가 끊기면 넘어진다"는 것을 그대로 보여 줍니다.
명령이 다시 오면 `명령 재수신 → PD 제어 재개` 로그와 함께 PD가 다시 켜집니다.

## 3. 터미널 실습 — `sim_node`를 직접 조종하기

### 무엇을 조종하나?

**`sim_node` 안에 있는 로봇의 관절 6개**(실제 로봇이라면 모터 6개)입니다.
ROS2에서는 노드를 직접 조작하지 않고, **노드가 구독하는 토픽에 메시지를 보내면 그 노드가 반응**합니다.
turtlesim 튜토리얼과 구조가 같습니다:

| turtlesim 튜토리얼 | 이 프로젝트 |
|---|---|
| `turtlesim_node` (화면 속 거북이) | `sim_node` (MuJoCo 속 2족 로봇) |
| `/turtle1/cmd_vel` (속도 명령) | `/joint_commands` (관절 목표 각도) |
| `/turtle1/pose` (거북이 위치) | `/joint_states` (관절 각도·속도·토크) |
| `teleop_turtle_key` (키보드 조종 노드) | 지금은 터미널 명령 → C의 `demo_controller` |

### 실습 1 — 로봇 켜기 (터미널 1)

```bash
ros2 run giga_sim_ros sim_node --ros-args -p fixed_base:=true
```

```
[sim_node]: 시작: 공중에 매단(fixed base) 로봇, 관절 6개, 물리 500 Hz, /joint_states·/imu/data 500 Hz, TF 100 Hz, 명령 타임아웃 없음
[sim_node]: 관절 이름: ['left_hip_pitch', 'left_knee', 'left_ankle_pitch', 'right_hip_pitch', 'right_knee', 'right_ankle_pitch']
```

### 실습 2 — 노드와 토픽 들여다보기 (터미널 2) · 노드, 토픽, 메시지

```bash
ros2 node list                    # → /sim_node
ros2 node info /sim_node
```

```
/sim_node
  Subscribers:
    /joint_commands: sensor_msgs/msg/JointState      ← 명령을 받는 입구
  Publishers:
    /clock: rosgraph_msgs/msg/Clock
    /imu/data: sensor_msgs/msg/Imu
    /joint_states: sensor_msgs/msg/JointState        ← 관절 상태를 내보내는 출구
    /tf: tf2_msgs/msg/TFMessage                      ← 몸통 위치 (odom → base_link)
  Service Servers:
    /sim/reset: std_srvs/srv/Trigger
```

```bash
ros2 topic hz /joint_states                       # → average rate: 499.987 (Ctrl+C로 멈춤)
ros2 topic echo --once /joint_states              # 메시지 한 개 보기
ros2 topic echo --once /imu/data                  # 자세 쿼터니언(x,y,z,w), 각속도, 가속도
ros2 interface show sensor_msgs/msg/JointState    # 메시지 형식 보기
```

`JointState`는 `name[]`, `position[]`, `velocity[]`, `effort[]` 배열이고 같은 위치끼리 짝입니다
(예: `name[1] = left_knee` 이면 `position[1]`이 왼쪽 무릎 각도 [rad]).

### 실습 3 — 관절 움직이기 · 발행(publish)

```bash
ros2 topic pub --once /joint_commands sensor_msgs/msg/JointState "{name: [left_knee], position: [1.5]}"
```

- **왼쪽 무릎만** 굽혀집니다 (실측: 0.794 → 1.485 rad, 오른쪽 무릎은 그대로). 보내지 않은 관절은 이전 목표를 유지합니다.
  목표 1.5보다 조금 덜 굽은 것은 중력 때문입니다 (PD 제어의 정상상태 오차, t05 참고).
- 더 해 보기:
  ```bash
  ros2 topic pub --once /joint_commands sensor_msgs/msg/JointState "{name: [left_hip_pitch], position: [-1.0]}"   # 왼다리 앞으로 (hip은 음수가 앞)
  ros2 topic pub --once /joint_commands sensor_msgs/msg/JointState "{name: [right_hip_pitch, right_knee], position: [-0.8, 1.6]}"
  ```

### 실습 4 — 잘못된 명령 · 노드가 스스로를 지키는 법

```bash
ros2 topic pub --once /joint_commands sensor_msgs/msg/JointState "{name: [left_elbow], position: [1.0]}"
ros2 topic pub --once /joint_commands sensor_msgs/msg/JointState "{name: [left_knee], position: [3.0]}"
ros2 topic pub --once /joint_commands sensor_msgs/msg/JointState "{name: [left_knee, right_knee], position: [1.0]}"
```

터미널 1의 경고 (실측):

```
[WARN] 명령 거부: 모르는 관절 이름 ['left_elbow'] (사용 가능: [...])
[WARN] left_knee: 목표 3.000 rad가 범위 [0.00, 2.36] 밖 → 2.360
[WARN] 명령 거부: position 개수(1)가 name 개수(2)와 다름
```

### 실습 5 — 초기화 · 서비스

```bash
ros2 service call /sim/reset std_srvs/srv/Trigger    # → success=True, message='home 자세로 초기화했습니다'
```

토픽은 "계속 흘려보내는 방송", 서비스는 "한 번 요청하고 응답을 받는 전화"입니다.

### 실습 6 — 설정 · 파라미터

```bash
ros2 param list /sim_node            # command_timeout, fixed_base, publish_rate, tf_rate, use_sim_time, viewer
ros2 param get /sim_node fixed_base  # → Boolean value is: True
```

| 파라미터 | 기본값 | 의미 |
|---|---|---|
| `fixed_base` | false | true = 공중에 매단 로봇 |
| `viewer` | true | false = MuJoCo 창 없이 |
| `publish_rate` | 500.0 | `/joint_states`, `/imu/data` 발행 주기 [Hz] |
| `tf_rate` | 100.0 | `odom → base_link` 발행 주기 [Hz] |
| `command_timeout` | 0.0 | 마지막 명령 후 이 시간[s]이 지나면 감쇠 모드. 0 = 끔 (launch에서 제어기를 붙이면 0.05) |

파라미터는 실행할 때 `--ros-args -p 이름:=값`으로 바꿉니다 (현재는 시작할 때만 읽음 — 실행 중 `ros2 param set`은 반영 안 됨).

### 실습 7 — 바닥에 세워서

```bash
ros2 run giga_sim_ros sim_node                     # 터미널 1 (Ctrl+C로 앞의 것을 끄고)
```

```bash
# 터미널 2: 쪼그려 앉기 (발바닥 수평 조건: hip = -a, knee = 2a, ankle = -a)
ros2 topic pub --once /joint_commands sensor_msgs/msg/JointState \
  "{name: [left_hip_pitch, left_knee, left_ankle_pitch, right_hip_pitch, right_knee, right_ankle_pitch],
    position: [-0.6, 1.2, -0.6, -0.6, 1.2, -0.6]}"
```

실측: 관절이 (−0.400, 0.825, −0.410) → (−0.600, 1.236, −0.615)로 바뀌며 로봇이 낮게 앉습니다.
일부러 넘어뜨리고(한쪽 다리만 크게 움직이기) `/sim/reset`으로 되돌려 보세요.

### 실습 8 — 연결 구조 그림

```bash
# 터미널 3: 명령을 1초에 한 번씩 계속 보내기 (--once는 순식간에 끝나서 그림에 안 잡힐 수 있음)
ros2 topic pub -r 1 /joint_commands sensor_msgs/msg/JointState "{name: [left_knee], position: [1.0]}"
# 터미널 2:
rqt_graph
```

`/sim_node`와 `ros2 topic pub`이 만든 노드(`/_ros2cli_...`)가 `/joint_commands`로 연결된 그림이 보입니다
(비어 있으면 rqt_graph 왼쪽 위 새로고침). C 시나리오를 띄워 놓고 rqt_graph를 보면 `demo_controller` ↔ `sim_node` ↔ `robot_state_publisher` 구조가 보입니다.

## 4. 개념 정리

| 개념 | 이번에 한 것 | 명령 |
|---|---|---|
| 노드 | `sim_node`, `demo_controller`, `robot_state_publisher` | `ros2 run`, `ros2 node list/info` |
| 토픽 | 상태 구독, 명령 발행 | `ros2 topic list/echo/hz/pub` |
| 메시지 | `JointState`, `Imu`, `TFMessage` | `ros2 interface show` |
| 서비스 | 초기화 요청 | `ros2 service list/call` |
| 파라미터 | 매단/선 모드, 제어기 동작 | `--ros-args -p`, `ros2 param list/get` |
| launch | 여러 노드를 인자와 함께 한 번에 | `ros2 launch 패키지 파일 인자:=값` |
| TF | 좌표계 트리: `odom → base_link → 다리 링크` | RViz의 TF 표시 |
| sim time | 모든 노드가 `/clock`(시뮬레이션 시간) 기준 | `use_sim_time:=true` |
| 패키지·빌드 | `giga_description`, `giga_sim_ros` | `bash scripts/build_ros.sh` |

## 5. 문제 해결

| 증상 | 원인 / 해결 |
|---|---|
| `Package 'giga_sim_ros' not found` | 빌드 안 했거나, 빌드 후 `source scripts/activate.sh`를 안 함 |
| 노드 실행 시 `No module named 'mujoco'` | 그냥 `colcon build`로 빌드함 → `bash scripts/build_ros.sh` |
| 다른 터미널에서 토픽이 안 보임 | 그 터미널에서 `source scripts/activate.sh`를 안 해서 `ROS_DOMAIN_ID`가 다름 (27 vs 0) |
| RViz에 로봇이 안 보이거나 "No transform" | `sim.launch.py`로 띄웠는지 확인 (RViz와 robot_state_publisher가 `use_sim_time:=true`여야 시뮬레이션 시간 TF를 받음) |
| 새로 만든 launch 파일이 안 보임 | launch·새 파일 추가 후에는 `bash scripts/build_ros.sh` 다시 실행 |
| `ros2 topic pub`으로 `/joint_states`를 직접 보냈는데 RViz 로봇이 빨간 오류 | 메시지의 시간(stamp)이 0이라 RViz가 오래된 정보로 버림 → `"{header: auto, name: [...], position: [...]}"`처럼 `header: auto`를 넣으면 현재 시각이 채워짐 (확인함). `/joint_commands`는 시간을 안 쓰므로 상관없음 |

## 6. 다음 단계

- 서 있기 제어기에 IMU를 이용한 **균형 제어** 추가 (발목 토크로 몸통 기울기 보정) → 40 N 밀기 견디기 ([04 §5](04_simple_biped_spec.md))
- **12-DoF 모델**(hip roll·yaw, ankle roll)로 확장 → 좌우 체중 이동, 3D 보행 준비 (로드맵 Phase 4)
- 전체 계획과 진행 상태: [05_ros2_bridge_plan.md](05_ros2_bridge_plan.md)
