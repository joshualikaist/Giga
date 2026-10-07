# 06. ROS2 첫 실습 — 우리 로봇(`sim_node`)을 터미널에서 조종하기

> 이 문서의 명령과 결과값은 2026-10-08 이 PC에서 실제로 실행해 확인한 것입니다.
> 같은 내용이 `tests/test_ros_sim_node.py`로 자동 테스트됩니다.

## 0. 그래서 "무엇을" 조종하나?

**`sim_node` 안에 있는 로봇의 관절 6개**입니다. 실제 로봇이라면 모터 6개에 해당합니다.

ROS2에서는 노드를 직접 조작하지 않습니다. **노드가 구독하고 있는 토픽에 메시지를 보내면, 그 노드가 반응합니다.**

```
 터미널 2 (사람, 나중엔 제어기 노드)                         터미널 1
 ┌──────────────────────────────┐   /joint_commands    ┌──────────────────────────────┐
 │ ros2 topic pub ...           │ ───────────────────▶ │ sim_node                     │
 │  "왼쪽 무릎을 1.5 rad로"     │                      │  = MuJoCo 로봇 + 모터 드라이버│
 │                              │   /joint_states      │  명령 → PD 토크 → 물리 계산  │
 │ ros2 topic echo ...          │ ◀─────────────────── │  500 Hz로 관절 상태 발행     │
 └──────────────────────────────┘                      └──────────────────────────────┘
```

- 지금은 **사람이 터미널에서** 명령을 보냅니다.
- 다음 단계(Step 5-5)에서는 이 역할을 **제어기 노드(Python 코드)** 가 맡습니다. 제어기는 `/joint_states`를 읽고, 계산해서, `/joint_commands`를 보냅니다.
- 나중에 실제 로봇으로 갈 때는 `sim_node` 자리에 **실제 모터 드라이버 노드**가 같은 토픽으로 들어갑니다. 제어기 코드는 그대로입니다.

ROS2 공식 튜토리얼의 turtlesim과 구조가 똑같습니다:

| turtlesim 튜토리얼 | 이 프로젝트 |
|---|---|
| `turtlesim_node` (화면 속 거북이) | `sim_node` (MuJoCo 속 2족 로봇) |
| `/turtle1/cmd_vel` (속도 명령) | `/joint_commands` (관절 목표 각도) |
| `/turtle1/pose` (거북이 위치) | `/joint_states` (관절 각도·속도·토크) |
| `teleop_turtle_key` (키보드 조종 노드) | 지금은 터미널 명령 → 나중에 제어기 노드 |

## 1. 준비 (처음 한 번)

```bash
cd ~/giga                          # 저장소 루트 (본인 경로)
source scripts/activate.sh
bash scripts/build_ros.sh          # ros2_ws 빌드 (그냥 colcon build 금지 — 아래 참고)
source scripts/activate.sh         # 빌드한 패키지를 현재 터미널에 반영
```

> `bash scripts/build_ros.sh`는 내부에서 `python -m colcon build`를 실행합니다.
> 그냥 `colcon build`로 빌드하면 노드가 시스템 Python으로 실행되어 `No module named 'mujoco'` 에러가 납니다 (확인함).

**네트워크 주의**: 지금 `ROS_DOMAIN_ID`가 기본값(0)이라, 같은 연구실 네트워크에서 ROS2를 쓰는 다른 PC와 토픽이 섞일 수 있습니다.
0~101 중 숫자 하나를 정해 `~/.bashrc`에 넣어 두세요. 모든 터미널이 같은 값이어야 서로 통신합니다.

```bash
echo 'export ROS_DOMAIN_ID=42' >> ~/.bashrc     # 42는 예시
```

새 터미널마다 `source scripts/activate.sh`를 실행해야 합니다 (터미널 1, 2 모두).

## 2. 실습

### 실습 1 — 로봇 켜기 (터미널 1)

```bash
ros2 run giga_sim_ros sim_node --ros-args -p fixed_base:=true
```

- MuJoCo 창에 **공중에 매달린** 로봇이 뜹니다. 처음엔 넘어질 걱정이 없는 매단 모드로 연습합니다.
- 터미널 1에 로그가 나옵니다:
  ```
  [sim_node]: 시작: 공중에 매단(fixed base) 로봇, 관절 6개, 물리 500 Hz, /joint_states 500 Hz, 명령 타임아웃 없음
  [sim_node]: 관절 이름: ['left_hip_pitch', 'left_knee', 'left_ankle_pitch', 'right_hip_pitch', 'right_knee', 'right_ankle_pitch']
  ```
- 끄기: 터미널 1에서 `Ctrl+C` 또는 MuJoCo 창 닫기.

### 실습 2 — 노드와 토픽 들여다보기 (터미널 2) · 개념: 노드, 토픽, 메시지

```bash
ros2 node list                    # → /sim_node
ros2 node info /sim_node          # 이 노드가 무엇을 받고 내보내는지
```

```
/sim_node
  Subscribers:
    /joint_commands: sensor_msgs/msg/JointState      ← 명령을 받는 입구
  Publishers:
    /clock: rosgraph_msgs/msg/Clock
    /joint_states: sensor_msgs/msg/JointState        ← 상태를 내보내는 출구
  Service Servers:
    /sim/reset: std_srvs/srv/Trigger
```

```bash
ros2 topic hz /joint_states                       # → average rate: 499.987 (Ctrl+C로 멈춤)
ros2 topic echo --once /joint_states              # 메시지 한 개 보기
ros2 interface show sensor_msgs/msg/JointState    # 메시지 형식(필드) 보기
```

`JointState`는 `name[]`, `position[]`, `velocity[]`, `effort[]` 배열로 되어 있고, 같은 위치끼리 짝입니다
(예: `name[1] = left_knee` 이면 `position[1]`이 왼쪽 무릎 각도 [rad]).

### 실습 3 — 관절 하나 움직이기 · 개념: 발행(publish)

```bash
ros2 topic pub --once /joint_commands sensor_msgs/msg/JointState "{name: [left_knee], position: [1.5]}"
```

- MuJoCo 창에서 **왼쪽 무릎만** 굽혀집니다. 보내지 않은 관절은 이전 목표를 유지합니다.
- 확인 (실측): 왼쪽 무릎 0.794 → **1.485 rad**, 오른쪽 무릎은 0.794 그대로.
  목표 1.5보다 조금 덜 굽은 것은 중력 때문입니다 (PD 제어의 정상상태 오차, t05 참고).
- 직접 해 보기:
  ```bash
  # 왼다리를 앞으로 들기 (hip은 음수가 앞쪽 — docs/04 §3 부호 규약)
  ros2 topic pub --once /joint_commands sensor_msgs/msg/JointState "{name: [left_hip_pitch], position: [-1.0]}"
  # 여러 관절 동시에
  ros2 topic pub --once /joint_commands sensor_msgs/msg/JointState "{name: [right_hip_pitch, right_knee], position: [-0.8, 1.6]}"
  ```

### 실습 4 — 잘못된 명령 보내 보기 · 노드가 스스로를 지키는 법

```bash
ros2 topic pub --once /joint_commands sensor_msgs/msg/JointState "{name: [left_elbow], position: [1.0]}"
ros2 topic pub --once /joint_commands sensor_msgs/msg/JointState "{name: [left_knee], position: [3.0]}"
ros2 topic pub --once /joint_commands sensor_msgs/msg/JointState "{name: [left_knee, right_knee], position: [1.0]}"
```

터미널 1에 이런 경고가 나옵니다 (실측):

```
[WARN] 명령 거부: 모르는 관절 이름 ['left_elbow'] (사용 가능: [...])
[WARN] left_knee: 목표 3.000 rad가 범위 [0.00, 2.36] 밖 → 2.360
[WARN] 명령 거부: position 개수(1)가 name 개수(2)와 다름
```

실제 로봇 드라이버도 이렇게 명령을 검사해야 합니다. 잘못된 명령을 조용히 무시하면 디버깅이 매우 어려워집니다.

### 실습 5 — 초기화 · 개념: 서비스

```bash
ros2 service list                                    # /sim/reset 이 보임
ros2 service call /sim/reset std_srvs/srv/Trigger    # → success=True, message='home 자세로 초기화했습니다'
```

토픽은 "계속 흘려보내는 방송", 서비스는 "한 번 요청하고 응답을 받는 전화"입니다.

### 실습 6 — 설정 · 개념: 파라미터

```bash
ros2 param list /sim_node            # command_timeout, fixed_base, publish_rate, viewer (+ use_sim_time)
ros2 param get /sim_node fixed_base  # → Boolean value is: True
```

파라미터는 실행할 때 `--ros-args -p 이름:=값`으로 바꿉니다.
(현재 버전은 시작할 때만 파라미터를 읽습니다. 실행 중 `ros2 param set`으로 바꿔도 반영되지 않습니다.)

| 파라미터 | 기본값 | 의미 |
|---|---|---|
| `fixed_base` | false | true = 공중에 매단 로봇 |
| `viewer` | true | false = MuJoCo 창 없이 실행 |
| `publish_rate` | 500.0 | `/joint_states` 발행 주기 [Hz] |
| `command_timeout` | 0.0 | 마지막 명령 후 이 시간[s]이 지나면 감쇠 모드(Kp=0). 0 = 끔 |

### 실습 7 — 바닥에 세워서 해 보기

터미널 1에서 `Ctrl+C`로 끄고, 이번엔 바닥에 선 로봇으로:

```bash
ros2 run giga_sim_ros sim_node                     # 터미널 1
```

```bash
# 터미널 2: 쪼그려 앉기 (발바닥이 수평이 되려면 hip = -a, knee = 2a, ankle = -a)
ros2 topic pub --once /joint_commands sensor_msgs/msg/JointState \
  "{name: [left_hip_pitch, left_knee, left_ankle_pitch, right_hip_pitch, right_knee, right_ankle_pitch],
    position: [-0.6, 1.2, -0.6, -0.6, 1.2, -0.6]}"
```

실측: 관절이 (−0.400, 0.825, −0.410) → **(−0.600, 1.236, −0.615)** 로 바뀌며 로봇이 낮게 앉습니다.
일부러 넘어뜨려 보고(예: 한쪽 다리만 크게 움직이기) `ros2 service call /sim/reset std_srvs/srv/Trigger`로 되돌려 보세요.

### 실습 8 — 연결 구조 그림 보기

```bash
# 터미널 3: 명령을 1초에 한 번씩 계속 보내기 (--once는 순식간에 끝나서 그림에 안 잡힐 수 있음)
ros2 topic pub -r 1 /joint_commands sensor_msgs/msg/JointState "{name: [left_knee], position: [1.0]}"
# 터미널 2:
rqt_graph
```

`/sim_node`와 `ros2 topic pub`이 만든 노드(`/_ros2cli_...`)가 `/joint_commands`로 연결된 그림이 보입니다.
(그림이 비어 있으면 rqt_graph 왼쪽 위의 새로고침 버튼을 누르세요. 끝나면 터미널 3은 `Ctrl+C`)

## 3. 이번 실습에서 쓴 ROS2 개념 정리

| 개념 | 이번에 한 것 | 명령 |
|---|---|---|
| 노드 | `sim_node` 실행, 정보 보기 | `ros2 run`, `ros2 node list/info` |
| 토픽 | 상태 구독, 명령 발행 | `ros2 topic list/echo/hz/pub` |
| 메시지 | `JointState` 구조 이해 | `ros2 interface show` |
| 서비스 | 초기화 요청 | `ros2 service list/call` |
| 파라미터 | 매단/선 모드 전환 | `--ros-args -p`, `ros2 param get` |
| 패키지·빌드 | `giga_sim_ros` 빌드 | `bash scripts/build_ros.sh` |

## 4. 다음 단계

- **RViz에서 같은 로봇 보기** (Step 5-1, 5-3): `robot_state_publisher` + RViz로 `/joint_states`를 시각화
- **제어기 노드 만들기** (Step 5-5): 실습 3의 `ros2 topic pub`을 Python 코드로 → 서 있기 제어기
- 전체 계획: [05_ros2_bridge_plan.md](05_ros2_bridge_plan.md)
