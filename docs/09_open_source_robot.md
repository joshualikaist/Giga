# 09. 오픈소스 로봇 불러오기 — Open Duck Mini v2 (오리 모양 2족 로봇)

> 이 문서의 결과는 2026-10-08 이 PC에서 Open Duck Mini 저장소 커밋 `b23317a`(v2 브랜치)로 실제로 확인한 것입니다.

인터넷에서 받은 로봇 URDF는 **그대로 시뮬레이션에 넣으면 거의 항상 문제가 있습니다.**
이 문서는 실제 오픈소스 로봇 하나를 받아 문제를 하나씩 찾고, 받은 파일은 고치지 않은 채 불러올 때 보완하는 과정입니다.
다른 로봇을 가져올 때도 §2의 점검표를 그대로 쓰면 됩니다.

## 0. 한눈에 보기

```bash
source scripts/activate.sh
bash scripts/get_open_duck.sh                         # 처음 한 번: URDF + 메시 45개 + 라이선스 (약 19 MB)
python tutorials/t07_open_duck.py --headless          # 문제 확인 → 보완 → 좌우 부호 → 서 있기 검증
python tutorials/t07_open_duck.py                     # 뷰어: PD로 서 있는 오리 (Ctrl+오른쪽 드래그로 밀어 보기)
python -m pytest tests/test_open_duck.py              # 자동 점검 (파일이 없으면 건너뜀)
```

![PD로 서 있는 Open Duck Mini v2](images/open_duck_stand.png)

## 1. Open Duck Mini v2

디즈니의 BD-X 드로이드를 작게 본뜬 팬 프로젝트입니다 ([저장소](https://github.com/apirrone/Open_Duck_Mini)).

| 항목 | 값 |
|---|---|
| 크기 / 질량 | 다리를 폈을 때 약 42 cm / 2.06 kg (URDF 합계) — simple_biped(8.2 kg)의 4분의 1 |
| 관절 16개 | 다리 5개 × 2 (고관절 yaw·roll·pitch, 무릎, 발목) + 목·머리 4개 + 안테나 2개 |
| 모터 | Feetech STS3215 서보 (위치 제어형), 부품값 약 $400 목표 |
| 학습 방식 | MuJoCo MJX + Brax PPO (이 저장소의 docs/08과 같은 방식) + 참고 동작을 따라 하는 모방 보상 |
| 파일 | `mini_bdx/robots/open_duck_mini_v2/` 에 `robot.urdf`, MuJoCo용 `robot.xml`·`robot_motors.xml`, STL 45개 |

## 2. 오픈소스 URDF를 받을 때 점검표

| # | 점검 | 왜 | Open Duck에서 실제로 | 이 저장소의 대응 |
|---|---|---|---|---|
| 1 | **라이선스** | 받아 쓰고 고치고 배포해도 되는지 | 주 저장소 Apache-2.0 ✅. 학습 코드 저장소(Open_Duck_Playground)는 라이선스 없음 → 읽고 배우기만, 코드 복사 금지. 외형은 디즈니 캐릭터 → 상업적 사용은 따로 확인 | 파일을 저장소에 넣지 않고 내려받기 스크립트로. LICENSE를 함께 받음 |
| 2 | **버전 고정** | 원본이 바뀌면 결과가 달라짐 | 활발히 바뀌는 저장소 | 커밋 `b23317a`로 고정 (`scripts/get_open_duck.sh`) |
| 3 | **메시 경로** | `package://`는 ROS 전용 표기 | `package:///이름.stl` → MuJoCo가 `package:/이름.stl`을 열다 실패 | `mesh_dir` 옵션 (meshdir + strippath, docs/03 §3) |
| 4 | **몸통 고정·모터 없음** | URDF에는 시뮬레이션 요소가 없음 | 그대로 열면 nq=16(관절만), 모터 0개 | 빌더가 freejoint·모터·IMU·바닥 추가 (t04와 같음) |
| 5 | **관절 순서** | 코드가 순서를 가정하면 틀림 | URDF에 발목 → 무릎 → 고관절 순(아래부터)으로 적혀 있음 | 이름으로만 다룸 (`RobotInterface`) |
| 6 | **관절 축과 좌우 부호** | 대칭 동작·보상의 기준 | 모든 축이 `0 0 1`(관절마다 좌표계를 돌려 둠). **hip_pitch만 좌우 부호가 반대**, knee·ankle은 같음 | FK로 직접 확인 (t07 §3, 테스트) |
| 7 | **관절 범위** | 자세·보행 방식을 정함 | hip_pitch: 앞 +0.52 / 뒤 −1.22 rad로 비대칭 | 무릎을 새처럼 뒤로 굽히는 자세 (§4) |
| 8 | **모터 사양** | 토크 한계·제어 방식이 다르면 정책이 실물에서 안 됨 | 모든 관절 `effort=1`, `velocity=20` — CAD 내보내기 기본값 | 토크 한계 3.23 N·m로 덮어씀 (`effort_limits`) |
| 9 | **질량·관성** | CAD 값과 실물이 다름 | 3D 프린팅 부품은 채움 비율에 따라 무게가 달라 제작자가 슬라이서 값으로 고침 | 실물이 있으면 저울로 잰 총 질량과 비교할 것 |
| 10 | **충돌 형상** | 계산 속도와 가짜 접촉 | 충돌 형상 131개가 전부 CAD 메시, 꼭짓점 18만 6천 개. 처음 자세에서 부품끼리 겹친 가짜 접촉 154개 (빌더의 부모-자식 제외 후에도 26개) | 발만 충돌 (`collision_bodies`) → 가짜 접촉 0개 |
| 11 | **시각용 형상** | 충돌을 끈 형상이 사라질 수 있음 | MuJoCo는 URDF를 읽을 때 충돌하지 않는 형상을 버림(`discardvisual` 기본값) → 발만 보이는 로봇 | 빌더가 항상 `discardvisual=False` |
| 12 | **바닥에 놓기** | 시작부터 떨어지거나 박힘 | 메시를 경계구로 근사하면 발이 5 cm 떠서 시작해 떨어짐 | 메시는 꼭짓점으로 가장 낮은 점을 정확히 계산 → 1 mm 위 |
| 13 | **시뮬레이션과 실물 차이** | 학습한 정책이 실물에서 동작하려면 | 제작자 메모: "시뮬레이션의 모터가 실물과 다르게 움직이면 정책이 동작하지 않는다". 서보 특성을 측정 도구(BAM)로 따로 잼 | 걷기 학습 전에 서보 모델을 맞춰야 함 (§6) |

## 3. biped_sim에서 한 일 — 받은 파일은 그대로

받은 파일은 한 글자도 고치지 않았습니다. 필요한 보완은 모두 `biped_sim/robot_configs.py`의 `OPEN_DUCK_MINI.sim_defaults`에 있고,
`build_robot_model(cfg.urdf, cfg.sim_config(...))`로 불러올 때 적용됩니다.

| `sim_defaults` | 값 | 근거 |
|---|---|---|
| `mesh_dir` | `models/third_party/open_duck_mini_v2` | 메시 경로가 `package:///` (점검 3) |
| `effort_limits` | 3.23 N·m (모든 관절) | URDF 값 1은 임시값. Open Duck 프로젝트가 측정한 STS3215(7.4 V) 값 |
| `joint_armature`, `joint_frictionloss`, `joint_damping` | 0.027, 0.083, 0 | 서보 감속기 관성·마찰. Open Duck 저장소의 `robot_motors.xml` (토크 모터 모델) 값 |
| `collision_bodies` | `foot_assembly`, `foot_assembly_2` | 발만 충돌 (점검 10) |
| `kp` / `kd` | 9.5 / 0.5 | kp는 Open Duck `robot.xml`의 위치 서보 값. 서보 내부의 위치 PD를 "토크 모터 + 바깥 PD"로 흉내 냄 |

빌더(`biped_sim/builder.py`)에 새로 생긴 옵션(`SimConfig`): `mesh_dir`, `effort_limits`, `joint_damping`, `joint_frictionloss`,
`collision_bodies`. 그리고 이 로봇을 불러오다 발견한 두 가지 문제(점검 11·12)를 빌더에서 고쳐 모든 로봇에 적용했습니다.

## 4. 서 있는 자세 찾기

모든 관절이 0이면 다리를 거의 편 자세라 균형 잡기 어렵습니다. 무릎을 굽힌 자세를 직접 찾습니다.

**① 부호 규칙 확인 (t07 §3).** 매단 상태에서 관절 하나만 +0.3 rad 돌려 발이 어디로 가는지 봅니다.

| 관절 | 왼발 이동 | 오른발 이동 | 좌우 부호 |
|---|---|---|---|
| hip_pitch | +5.7 cm (앞) | −5.7 cm (뒤) | **반대** |
| knee | +3.4 cm | +3.4 cm | 같음 |
| ankle | +1.1 cm | +1.1 cm | 같음 |

**② 발바닥 수평 + 발이 엉덩이 아래.** 세 관절이 모두 같은 축(pitch)이므로 발바닥이 수평이려면 세 각도의 합이 0이어야 합니다
(오른쪽 hip은 부호를 뒤집어서). 이 조건에서 무릎 각도마다 발이 엉덩이 아래에 오는 hip 각도를 풀면 `hip = ankle = −knee/2`입니다.
무릎을 앞으로(−) 굽힐 수도, 뒤로(+) 굽힐 수도 있는데, hip_pitch의 앞쪽 범위가 0.52 rad밖에 안 돼서
무릎을 앞으로 굽히면(hip +0.4) 다리를 앞으로 내밀 여유가 0.12 rad밖에 남지 않습니다.
→ **무릎을 새처럼 뒤로 0.8 rad 굽히는 자세**를 골랐습니다 (BD-X 다리 모양과도 같음).

| | hip_pitch | knee | ankle |
|---|---|---|---|
| 왼쪽 | −0.4 | +0.8 | −0.4 |
| 오른쪽 | **+0.4** | +0.8 | −0.4 |

## 5. 결과: 관절 PD로 서 있기

`python tutorials/t07_open_duck.py --headless` (5초):

```
t=5.0s 몸통 높이 0.178 m, 기울기 4.2°, 발 하중 10.27 + 9.96 N = 무게(20.22 N)의 1.000배, 최대 토크 0.42 / 3.23 N·m
→ 서 있음 ✅
```

- 두 발 하중의 합이 무게와 정확히 같습니다 (t06과 같은 힘의 평형 검증).
- 서 있는 데는 서보 한계의 13 %(0.42 N·m)만 씁니다.

PD 게인을 바꿔 보면 (5초 뒤, 같은 home 자세에서 시작):

| kp / kd | 몸통 높이 | 기울기 | 최대 토크 |
|---|---|---|---|
| 5.0 / 0.5 | 0.171 m | 11.5° | 0.52 N·m |
| **9.5 / 0.5 (기본)** | 0.178 m | 4.2° | 0.42 N·m |
| 9.5 / 1.0 | 0.179 m | 4.0° | 0.41 N·m |
| 13.4 / 0.5 | 0.180 m | 2.8° | 0.40 N·m |

게인이 약하면 무게에 눌려 더 기웁니다. 기본 게인에서도 몸통이 앞으로 4.2° 기우는데(pitch +), 무게중심이 처음부터
발 중심보다 0.8 cm 앞에 있기 때문입니다. 머리가 0.54 kg(전체의 26 %)이고 몸통 원점보다 3.4 cm 앞에 있습니다 (실측).
실제 로봇에서도 이 기울기가 비슷한지 비교해 보면 질량 분포가 맞는지 확인할 수 있습니다.

## 6. 다음 단계: 오리 걷기 학습

docs/08의 보행 학습(MJX + Brax)을 이 로봇에 쓰려면 바꿀 것:

| 항목 | simple_biped | Open Duck Mini |
|---|---|---|
| 관절 수 (행동 크기) | 6 | 다리 10 (+ 머리 4, 안테나 2는 고정하거나 따로) |
| 좌우 대칭 보상 | 좌우 부호 같음 | **hip_pitch 부호 반대** → 관절별 부호 표가 필요 (§4 ①) |
| 고관절 yaw·roll | 없음 (옆 방향 균형을 발 폭으로만) | 있음 → 옆 방향 균형·방향 전환 가능 |
| 발 충돌 형상 | 상자 | STL 메시 4개씩 (MJX는 볼록 껍질로 계산, 상자보다 느림 → 상자로 바꾸는 것도 방법) |
| 모터 | 토크 모터 + PD | 위치 서보: 실물과 맞추려면 서보 특성(지연, 백래시, 토크-속도 곡선)을 모델에 넣어야 함 |
| 크기 | 몸통 높이 0.59 m, 목표 0.3 m/s | 몸통 높이 0.18 m → 목표 속도·걸음 박자도 작게 |

## 7. 관련 파일

| 파일 | 역할 |
|---|---|
| `scripts/get_open_duck.sh` | 고정 커밋에서 URDF·메시·LICENSE 내려받기 → `models/third_party/open_duck_mini_v2/` (git 제외) |
| `models/third_party/README.md` | 외부 로봇 파일의 출처·라이선스 |
| `biped_sim/robot_configs.py` | `OPEN_DUCK_MINI`: home 자세, 게인, `sim_defaults` (근거 주석 포함) |
| `biped_sim/builder.py` | `SimConfig`의 `mesh_dir`, `effort_limits`, `collision_bodies` 등 |
| `biped_sim/utils.py` | `lowest_collision_z`: 메시의 가장 낮은 점 정확히 |
| `tutorials/t07_open_duck.py` | 문제 확인 → 보완 → 좌우 부호 → 서 있기 |
| `tests/test_open_duck.py` | 빌더 보완, 바닥 1 mm 위 시작, 좌우 부호, 서 있기 |

출처: [Open_Duck_Mini](https://github.com/apirrone/Open_Duck_Mini) (Apache-2.0),
[sim2real 메모](https://github.com/apirrone/Open_Duck_Mini/blob/v2/docs/sim2real.md),
[Open_Duck_Playground](https://github.com/apirrone/Open_Duck_Playground) (참고만, 라이선스 없음)
