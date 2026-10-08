# 튜토리얼 (t01 ~ t07)

**순서대로** 진행하세요. 각 튜토리얼은 `--headless`(숫자 출력)와 뷰어 모드를 모두 지원합니다.
먼저 `--headless`로 결과를 읽고, 그다음 뷰어로 눈으로 확인하는 것을 권장합니다.

```bash
source scripts/activate.sh          # 매 터미널마다 (저장소 루트에서)
python tutorials/t01_hello_mujoco.py --headless
```

| # | 파일 | 주제 | 핵심 질문 | 관련 문서 |
|---|---|---|---|---|
| t01 | `t01_hello_mujoco.py` | MJCF로 상자·공 떨어뜨리기 | 시뮬레이션 루프란? | [02](../docs/02_mujoco_concepts.md) §1 |
| t02 | `t02_model_and_data.py` | MjModel / MjData 해부 | 왜 `nq ≠ nv`? 쿼터니언 순서는? | [02](../docs/02_mujoco_concepts.md) §2–6 |
| t03 | `t03_urdf_pendulum.py` | 첫 URDF + 이론 검증 | URDF를 그냥 넣으면 무엇이 틀어지나? | [03](../docs/03_urdf_and_mujoco.md) §2 |
| t04 | `t04_biped_inspect.py` | 2족 URDF 해부 | 빌더는 무엇을 추가하나? | [03](../docs/03_urdf_and_mujoco.md) §3, [04](../docs/04_simple_biped_spec.md) |
| t05 | `t05_biped_hanging_pd.py` | **매단 로봇 관절 PD 추종** | Kp/Kd/피드포워드의 효과는? | [02](../docs/02_mujoco_concepts.md) §8 |
| t06 | `t06_biped_stand.py` | 바닥에 세우기 | 시뮬레이션이 물리적으로 맞나? 어디까지 버티나? | [04](../docs/04_simple_biped_spec.md) §5 |
| t07 | `t07_open_duck.py` | 남이 만든 로봇(오리) 불러오기 | 받은 URDF를 그대로 넣으면 무엇이 틀어지나? | [09](../docs/09_open_source_robot.md) |

결과물(그래프, 이미지, 내보낸 MJCF)은 모두 `output/`에 저장됩니다.

---

## t01 — Hello MuJoCo

```bash
python tutorials/t01_hello_mujoco.py --headless
python tutorials/t01_hello_mujoco.py              # 뷰어
```

**관찰**: 상자는 z = 0.100 m(반 변 길이), 공은 z = 0.080 m(반지름)에서 멈춥니다.

**과제**
1. `models/mjcf/falling_box.xml`의 `timestep`을 0.01, 0.0005로 바꿔 결과가 어떻게 달라지는지 비교하세요.
2. 공의 `<geom>`에 `solref="-5000 -5"`를 추가하면? (MJCF 문서에서 `solref`의 음수 의미를 찾아보세요)
3. 뷰어에서 상자를 더블클릭 → `Ctrl` + 오른쪽 드래그로 밀어 보세요.

## t02 — MjModel과 MjData

```bash
python tutorials/t02_model_and_data.py
```

**관찰**: freejoint 하나가 qpos 7칸, qvel 6칸. MJCF `euler="20 30 0"`이 URDF 방식 rpy로는 (22.8°, 28.0°, 11.2°). 접촉력의 합 = 무게.

**과제**
1. `falling_box.xml`에 `<compiler eulerseq="XYZ"/>`를 넣고 다시 실행하면 3절 출력이 어떻게 바뀌나요? 왜 그런가요?
2. 공에 hinge 관절을 하나 더 단 바디를 추가하면 `nq`, `nv`는 얼마가 되나요? 실행해서 확인하세요.

## t03 — 첫 URDF와 검증

```bash
python tutorials/t03_urdf_pendulum.py --headless
python tutorials/t03_urdf_pendulum.py --raw       # 뷰어: 잘못된 모델
python tutorials/t03_urdf_pendulum.py             # 뷰어: 고친 모델
```

**관찰**: URDF를 그대로 읽으면 주기 오차 59%(FAIL) — 에러 메시지는 없음! 빌더로 읽으면 0.015%(PASS).

**과제**
1. `pendulum.urdf`의 `<dynamics damping>`을 0.05로 바꾸면 4단계(적분기 비교) 결과가 어떻게 달라지나요?
2. `pendulum.urdf` 루트에 `<mujoco><compiler fusestatic="false"/></mujoco>`만 추가하고 `--raw`로 다시 보세요.
   base_link는 살아나지만 막대는 여전히 붙잡힙니다. 왜일까요? ([02](../docs/02_mujoco_concepts.md) §6)
3. 막대 질량을 2 kg으로 바꾸면 주기는? 이론적으로 예측한 뒤 실행해서 확인하세요.

## t04 — 2족 URDF 해부

```bash
python tutorials/t04_biped_inspect.py --headless
python tutorials/t04_biped_inspect.py             # 뷰어: Control 슬라이더로 관절 구동
```

**관찰**: URDF 그대로는 nq=6, nu=0 (몸통이 공중에 용접됨). 빌더 결과는 nq=13, nv=12, nu=6, 센서 3개.
`output/simple_biped_floating.xml`을 열어 빌더가 추가한 요소를 직접 찾아보세요.

**과제**
1. 뷰어의 Control 슬라이더로 `left_knee`에 +토크를 주면 무릎이 어느 쪽으로 굽나요? [04](../docs/04_simple_biped_spec.md) §3 부호 규약과 맞나요?
2. 뷰어 왼쪽 "Rendering" 패널에서 geom group 3을 켜서 숨겨진 충돌 형상을 확인하세요.
3. 왼쪽 패널에서 Inertia 표시를 켜 보세요. 각 링크의 관성 상자가 형상과 비슷한 크기인가요?

## t05 — 매단 로봇의 관절 PD 추종 (제일 쉬운 URDF 제어 실험)

```bash
python tutorials/t05_biped_hanging_pd.py --headless          # 그래프: output/t05_tracking.png
python tutorials/t05_biped_hanging_pd.py --compare           # 6가지 설정 비교
python tutorials/t05_biped_hanging_pd.py                     # 뷰어
```

**관찰** (`--compare`): 기본 0.48° → Kp×3 0.17° / 중력 보상 0.22° / 속도 피드포워드 없음 1.78° / Kd×8 4.1° + 채터링.

**과제**
1. `--freq 2`로 빠르게 흔들면 어떤 관절이 먼저 토크 포화되나요?
2. Kd를 몇 배까지 키우면 채터링이 시작되나요? `--kd-scale`로 찾고, `Kd·dt/I` 공식으로 설명해 보세요.
   (힌트: `SimConfig(joint_armature=...)`를 바꾸면 경계가 어떻게 움직이나요?)
3. 중력 보상(`--gravity-comp`)을 켜면 Kp를 얼마까지 낮춰도 기본 설정만큼 정확한가요?

## t06 — 바닥에 세우기

```bash
python tutorials/t06_biped_stand.py --headless
python tutorials/t06_biped_stand.py --headless --no-control
python tutorials/t06_biped_stand.py --headless --push 20
python tutorials/t06_biped_stand.py --headless --push 40
python tutorials/t06_biped_stand.py                          # 뷰어에서 직접 밀어 보기
```

**관찰**: 서 있을 때 두 발 수직력 합/무게 = 1.000, IMU 가속도 z = 9.81. 20 N은 버티고 40 N은 넘어짐.

**과제**
1. `robot_configs.py`에서 발목 Kp를 30으로 낮추면 `--push 20`의 결과는? ([04](../docs/04_simple_biped_spec.md) §5 표와 비교)
2. 옆으로(y 방향) 밀면 어떻게 될까요? `control_step`의 힘 방향을 바꿔 실험하고, hip roll이 없는 이 로봇의 한계를 설명하세요.
3. home 자세의 `_A`(무릎 굽힘 정도)를 0.2, 0.6으로 바꾸면 버틸 수 있는 밀기 힘이 어떻게 변하나요? 이유는?

## t07 — 남이 만든 오픈소스 로봇 불러오기 (Open Duck Mini v2)

```bash
bash scripts/get_open_duck.sh                   # 처음 한 번 (약 19 MB)
python tutorials/t07_open_duck.py --headless
python tutorials/t07_open_duck.py               # 뷰어: 서 있는 오리를 Ctrl+오른쪽 드래그로 밀어 보기
python tutorials/t07_open_duck.py --hang        # 공중에 매단 오리
```

관찰 포인트
- 1번: 받은 URDF를 그대로 열면 메시 경로 오류 → 메시 폴더를 알려 줘도 몸통 고정·모터 없음·가짜 접촉 154개·토크 한계 1 N·m
- 2번: `RobotConfig.sim_defaults`의 각 줄이 1번의 어떤 문제를 고치는지 짝지어 보기
- 3번: 같은 이름의 관절인데 hip_pitch만 좌우 부호가 반대. 관절 이름만 보고 짐작하면 왜 위험한가?
- 4번: 두 발 하중의 합이 무게와 같은지 (t06과 같은 검증)

과제
1. `biped_sim/robot_configs.py`에서 `OPEN_DUCK_MINI`의 `collision_bodies`를 지우고 실행해 보세요. 접촉 수와 실행 속도가 어떻게 바뀌나요?
2. `kp`를 5로 낮추면 몸통이 얼마나 기우나요? (docs/09 §5의 표와 비교)
3. home 자세의 무릎을 앞으로 굽히도록(knee −0.8, 왼쪽 hip +0.4 ...) 바꾸면 서 있을 수 있나요? 걸을 때는 어떤 문제가 생길까요? (docs/09 §4)

