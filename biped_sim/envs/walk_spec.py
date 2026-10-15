"""보행 과제 설정 — 로봇마다 다른 값(WalkSpec), 서보 모델, 학습과 똑같은 MuJoCo 모델 만들기, 정책 관측 계산.

jax가 필요 없어서 ROS용 .venv(ROS2 노드: sim_node·policy_node)와 GPU 학습 환경(.venv-mjx) 모두에서 import할 수 있습니다.
GPU 학습 환경(walk_mjx.BipedWalkMjxEnv)과 ROS2 노드가 같은 정의를 쓰므로, 학습한 로봇과 ROS로 돌리는 로봇이 똑같습니다.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass, field, replace

import mujoco
import numpy as np

from ..builder import FREEJOINT_NAME, build_robot_spec
from ..robot_configs import OPEN_DUCK_MINI, SIMPLE_BIPED, RobotConfig

PHYSICS_DT = 0.004          # [s]
CONTROL_DT = 0.02           # [s] 정책 50 Hz → 물리 5스텝마다 행동 1번
FALL_TILT = 0.6             # [rad]

REWARD_WEIGHTS = {
    "forward_velocity": 2.0,    # exp(−(vx − 목표)² / σ²): 목표 속도에 가까울수록 최대 1 (σ²는 로봇 크기에 맞춤)
    "lateral_velocity": -1.0,   # vy² : 옆으로 미끄러지지 않기
    "yaw_rate": -0.5,           # wz² : 제자리에서 돌지 않기
    "upright": -2.0,            # 기울기²
    "height": -10.0,            # (높이 − 기준)² : 무릎을 너무 굽혀 기어가지 않기
    "feet_air_time": 2.0,       # 발이 공중에 있던 시간이 (박자 × 0.25)보다 길면 착지할 때 보상 → 발을 들어 딛게 유도
    "action_rate": -0.02,       # |Δ행동|² : 떨림 방지
    "torque": -2e-4,            # |토크|² : 에너지 절약
    "alive": 0.5,               # 살아 있으면 매 스텝 +
    "flight": -1.0,             # 두 발이 모두 공중이면 1 : 뛰지 말고 걷기
    "gait_phase": 1.0,          # 걸음 박자대로 좌우 발이 번갈아 딛으면 1 (두 발 평균)
    "heading": -3.0,            # (방향 각도)² : 처음 방향(+x)을 유지하기
    "symmetry": -2.0,           # Σ(왼다리 관절 − 반 박자 전 오른다리 관절 × 좌우 부호)² + 반대 [rad²] : 두 다리가
                                #   같은 동작을 반 박자 어긋나게 하기 (절뚝임 방지). 좌우 부호는 FK로 자동 계산
    "step_length": 5.0,         # 한 걸음 착지할 때 반대 발보다 앞에 놓은 거리 / 목표 보폭 (목표에서 1, 넘어도 1)
                                #   (박자 × 0.125 이상 공중에 있던 착지만. 짧게 튕긴 착지로 보상을 두 번 받는 꼼수 방지)
    "reference": 0.0,           # exp(−Σ(관절 − 참고 동작)² / (n·0.15²)) : 걸음 박자에 맞춘 참고 동작 따라 하기 (모방 보상).
                                #   기본 0 = 꺼짐. WalkSpec.reference_gait가 있는 로봇만 (docs/09 §7)
    "com_over_stance": 0.0,     # 한 발로 설 때 무게중심이 디딤 쪽 고관절(hip_roll 축)로 옮겨 간 거리 / COM_SHIFT_TARGET (0~1).
                                #   발목 roll이 없는 오리는 hip_roll이 옆 기울기를 혼자 버티므로, 옮길수록 hip_roll 토크가 줄어듦
                                #   (0 → 0.90, 2.4 cm → 0.50 N·m). 기본 0 = 꺼짐. WalkSpec.hip_roll_joints가 있는 로봇만 (docs/09 §9.6)
    "servo_overload": 0.0,      # Σ(관절마다 데이터시트 토크-속도 한계를 넘은 만큼)² [N·m²], 물리 스텝 평균. 넘지 않으면 0.
                                #   모터 토크 = 명령 − 관절 damping × 속도, 한계 = 정지 토크 − (정지/무부하)·부호(τ)·ω (docs/09 §8.7).
                                #   기본 0 = 꺼짐. 실물 서보(STS3215 7.4 V)가 못 내는 토크를 쓰지 않게 (docs/10 2차)
    "yaw_tracking": 0.0,        # exp(−(회전 속도 − 회전 명령)² / 0.1): 명령 모드에서 돌기를 따라가기 (기본 0 = 꺼짐, docs/09 §10)
    "foot_clearance": 0.0,      # 한 걸음 착지할 때 그 흔듦 동안 발을 가장 높이 든 높이 / WalkSpec.clearance_target (최대 1).
                                #   0.1초 이상 공중에 있던 착지만. 기본 0 = 꺼짐 (오리 장애물용, docs/09 §9.7)
    "roll_excess": 0.0,         # max(|몸통 roll| − roll_free, 0)² [rad²]: 꼭 필요한 만큼(roll_free)을 넘게 구르면 벌점.
                                #   head_sway와 달리 서 있거나 제자리 돌기처럼 안 구르는 명령에는 0이라 '서 있기'를 부추기지 않음.
                                #   발목 roll이 없는 오리는 발을 들려고 골반을 기울이는데(측정: 발 들기의 96 %), 그 양을 줄이려는 항 (docs/10 §12)
    "head_sway": 0.0,           # (머리의 옆 속도 − 옆 명령)² [m²/s²], 몸통 방향 기준. 머리 속도 ≈ 몸통 속도 + 각속도 × (머리 − 몸통)
                                #   (목 관절은 home 유지라 거의 안 움직임). 몸통이 좌우로 크게 구르면(roll 32°) 높이 0.38 m의
                                #   머리가 12 cm씩 흔들림 → 구르기를 줄이게. 기본 0 = 꺼짐. WalkSpec.head_body가 있는 로봇만 (docs/10 5차)
}
# 실물과 시뮬레이션의 차이 대비 (--domain-rand): 환경마다 리셋할 때 무작위로 정함 (docs/10 3차)
DOMAIN_RAND = {"mass": 0.15,        # 몸통 질량 ±15 % (관성도 같은 비율)
               "motor": 0.15,       # 모터 세기 ±15 % (PD 토크와 토크 한계 모두)
               "com": 0.01,         # 몸통 무게중심 앞뒤·좌우 ±1 cm
               "delay": 2,          # 제어 지연 0 ~ 2 제어 스텝 (0 / 20 / 40 ms): 그만큼 전에 낸 행동이 지금 적용
               "joint_offset": 0.03,   # 관절마다 서보 영점 오차 ±0.03 rad (≈2°, 서보 혼을 한 톱니 틀리게 끼운 정도): 읽기·명령 모두
               "imu_tilt": 0.035}      # IMU를 기울게 붙인 오차 (roll·pitch 각각 ±2°): 중력 방향·각속도를 그만큼 돌려서 읽음


OBS_NOISE = {"gravity": 0.02, "gyro": 0.05, "joint_pos": 0.01, "joint_vel": 0.5}   # 표준편차 (IMU BNO055, 서보 엔코더·백래시 정도)
SERVO_DATASHEET = {"stall": 1.91, "noload": 4.4}   # [N·m], [rad/s] STS3215 7.4 V (19.5 kg·cm, 0.238 s/60°) = feasibility.SERVO
COM_SHIFT_TARGET = 0.024    # [m] 이만큼 옮기면 com_over_stance 만점 (§8.6 LIPM 상한, hip_roll 정적 토크가 정격 아래로)
# 보상을 하나씩 더해 온 과정 (simple_biped 실측, docs/08 §5~6):
#   flight·gait_phase가 없으면 → 두 발을 다 띄우고 깡충깡충 뜀 (공중 70 %)
#   symmetry·step_length·heading이 없으면 → 오른다리는 늘 앞, 왼다리는 늘 뒤인 절뚝이는 걸음 (절뚝임 점수 40 %)
#   모두 켜고 처음부터 학습 → 대칭으로 걸음 (절뚝임 1.8 %, 좌우 다리 차이 0.8°)
# 보상과 상관없이 늘 기록하는 걸음 지표 (가중치 0인 항목도 실제로 어떤지 보이게). 학습 스크립트가 에피소드 평균
# 비율로 바꿔 TensorBoard walk/gait_<이름>_pct, walk/abs_heading_deg 로 기록한다.
GAIT_METRICS = ("flight", "single", "double", "phase_match", "abs_heading", "leg_asymmetry", "com_shift",
                "landing", "clearance", "servo_over", "track_lin", "track_yaw", "head_sway", "roll_abs")
# 명령 따라 걷기 (--cmd-*): 명령 = (앞뒤 속도 vx, 옆 속도 vy [m/s], 회전 속도 wz [rad/s]), 몸통이 향한 방향 기준.
# 학습 때는 환경마다 이 범위에서 뽑고, resample초마다 새로 뽑음 (걷는 중에 명령이 바뀌는 것도 연습)
# zero_prob: 성분마다 이 확률로 정확히 0 (범위가 0을 포함할 때만) → '제자리 돌기', '옆으로만', '뒤로만' 같은 명령이 자주 나옴
COMMANDS_DEFAULT = {"vx": (0.05, 0.25), "vy": (0.0, 0.0), "wz": (0.0, 0.0), "resample": 5.0, "zero_prob": 0.0}


@dataclass(frozen=True)
class WalkSpec:
    """로봇마다 다른 보행 과제 설정."""
    name: str
    robot: RobotConfig
    policy_joints: tuple[str, ...]      # 정책이 움직이는 관절. 왼쪽 관절들 다음 오른쪽 관절들, 같은 순서 (대칭 보상용)
    foot_frames: tuple[str, str]        # (왼, 오른) 발 기준 바디: 높이로 접촉 판정, x로 보폭 계산
    foot_contact_z: float               # [m] 발 기준 바디 높이가 이보다 낮으면 '딛음'
    nominal_height: float               # [m] 서 있을 때 몸통 높이 (관측·height 보상의 기준)
    fall_height: float                  # [m] 몸통이 이보다 낮으면 넘어짐
    target_speed: float                 # [m/s] 기본 목표 속도 (--speed로 바꿈)
    gait_period: float                  # [s] 걸음 박자 (한 쌍의 걸음). 정책에 sin/cos로 알려 줌
    action_scale: float                 # [rad] 행동 1.0 = home에서 이만큼
    view_distance: float = 1.5          # [m] 화면·영상 카메라 거리 (로봇 전체가 보이게)
    # 참고 동작 (reference 보상용). {관절 이름 뒷부분: (앞뒤 계수, 들어올림 계수)}, 진폭 fwd·lift [rad]. None = 없음
    reference_gait: dict | None = None
    reward_weights: dict = field(default_factory=lambda: dict(REWARD_WEIGHTS))
    sim_overrides: dict = field(default_factory=dict)   # 이 과제용 SimConfig (충돌 단순화 등)
    pd_gains: tuple[float, float] | None = None          # (kp, kd) 모든 구동 관절에. None = 로봇 설정(RobotConfig) 값
    servo: str = "ideal"                                  # 서보 모델 이름 (SERVO_MODELS, with_servo로 바꿈)
    home_knee: float | None = None                        # 서 있는 자세의 무릎 각도를 바꿨으면 그 값 (with_home_knee)
    hip_roll_joints: tuple[str, str] | None = None        # (왼, 오른) 옆으로 기울이는 고관절 (com_over_stance 보상용)
    clearance_target: float = 0.05                        # [m] foot_clearance 보상 만점 발 높이 (발 기준점 높이)
    # 실물에서 잴 수 있는 관측만: 몸통 속도(3)·땅 위 높이(1)는 시뮬레이션에서만 아는 값이라 0으로 가림 (오리는 IMU·서보 각도뿐)
    deployable_obs: bool = False
    # 관측 기록: 지금 관측 뒤에 지난 몇 제어 스텝의 관측을 이어 붙임 (지난 움직임에서 무게·모터·지연 같은 실물 차이를 짐작하게)
    obs_history: int = 0
    # 명령 따라 걷기: 관측의 '목표 속도' 1칸 대신 명령 3칸 (vx, vy, wz)
    command_obs: bool = False
    head_body: str | None = None                          # 머리 바디 (head_sway 보상·흔들림 측정용)
    # head_sway 벌점을 빼 주는 명령 (뒤 속도, 회전 속도). None = 늘 벌점. (0.05, 0.3)이면 뒤로 0.05 m/s 이상이나 회전 0.3 rad/s
    # 이상인 명령에서는 벌점 0 (사이는 직선). 벌점은 걸을 때마다 거의 같은 크기라, 추종 이득이 작은 뒤로·제자리 돌기 명령에서
    # '서 있기'를 고르게 만들었음 (docs/10 5차 R1b) → 그 명령들은 빼 줌
    head_sway_gate: tuple[float, float] | None = None
    roll_free: float = 0.14               # [rad] roll_excess 벌점이 봐주는 몸통 구르기 (8°). 무게중심을 옮기는 데 필요한 양은 §8.6 기준 2.6°
    # 명령 추종 오차에서 봐주는 옆 속도 [m/s] (0 = 지금까지처럼 전부 벌점). 몸통을 수평으로 두고 무게중심을 좌우로 옮기는
    # 정상 걸음은 옆 속도가 최대 약 0.15 m/s인데, dead-band가 없으면 그 자체가 추종 오차로 잡혀 '구르기'가 더 싸진다 (docs/10 §12)
    lateral_deadband: float = 0.0
    # gait_phase 보상의 양발 지지 구간: 박자가 바뀌는 앞뒤 double_support 주기씩 (기본 0.1 → 한 주기의 40 %가 양발,
    # 발마다 디딤 비율 β = 0.7). 서보 속도로 낼 수 있는 최고 속도는 β로 정해짐 (docs/09 §8.5: β 0.7 → 0.22~0.28 m/s,
    # 0.6 → 0.34~0.43 m/s). double_support_fast를 주면 앞뒤 명령 속도가 target_speed → fast_speed로 갈수록 그 값으로 줄임
    double_support: float = 0.1
    double_support_fast: float | None = None
    fast_speed: float = 0.375

    @property
    def n(self) -> int:
        return len(self.policy_joints)

    @property
    def base_obs_size(self) -> int:   # 한 스텝의 관측 크기 (명령 모드면 명령 3칸이라 2칸 더)
        return 13 + 3 * self.n + (2 if self.command_obs else 0)

    @property
    def obs_size(self) -> int:        # 정책이 받는 관측 크기 = 지금 + 지난 obs_history 스텝
        return self.base_obs_size * (1 + self.obs_history)

    @property
    def velocity_sigma2(self) -> float:   # 속도 보상의 폭: 목표 속도에 비례 (0.3 m/s에서 0.05)
        return 0.05 * (self.target_speed / 0.3) ** 2

    @property
    def half_period_steps(self) -> int:   # 반 박자 = 제어 스텝 수 (symmetry 보상용 기록 길이)
        return round(self.gait_period / 2 / CONTROL_DT)


_SIMPLE_LEGS = ("hip_pitch", "knee", "ankle_pitch")
_DUCK_LEGS = ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle")
_DUCK_FEET = {"left_foot": "foot_assembly", "right_foot": "foot_assembly_2"}   # 발바닥 기준점 ← 발 메시
SPECS = {
    "simple_biped": WalkSpec(
        name="simple_biped", robot=SIMPLE_BIPED,
        policy_joints=tuple(f"{s}_{j}" for s in ("left", "right") for j in _SIMPLE_LEGS),
        foot_frames=("left_foot", "right_foot"), foot_contact_z=0.035,   # 발 바디 원점 = 발목 높이
        nominal_height=0.588, fall_height=0.35, target_speed=0.3, gait_period=0.8, action_scale=0.5,
        sim_overrides=dict(collision_bodies=SIMPLE_BIPED.foot_bodies),
    ),
    # 오리: 다리 10개만 정책이, 머리·안테나는 PD로 home 유지. 크기가 simple_biped의 약 0.3배라
    # 속도·박자를 다리 길이에 맞춰 줄임 (같은 '걷는 느낌'이 되는 비율: 속도 ∝ √길이, 박자 ∝ √길이)
    "open_duck_mini": WalkSpec(
        name="open_duck_mini", robot=OPEN_DUCK_MINI,
        policy_joints=tuple(f"{s}_{j}" for s in ("left", "right") for j in _DUCK_LEGS),
        foot_frames=("left_foot", "right_foot"), foot_contact_z=0.01,    # 발바닥 기준점 (바닥에서 2.6 mm 위)
        nominal_height=0.18, fall_height=0.11, target_speed=0.15, gait_period=0.5, action_scale=0.3,
        view_distance=1.0,                                               # 몸통은 낮지만 머리까지 약 0.42 m
        # 참고 동작: 디딤 동안 발을 앞(+fwd)에서 뒤(−fwd)로 밀고, 흔듦 동안 무릎을 굽혀(lift) 들어 앞으로.
        # 계수는 서 있는 자세를 찾을 때의 관계에서 나옴 (docs/09 §4): hip+knee+ankle=0이면 발바닥 수평,
        # hip=ankle=−knee/2면 발이 엉덩이 아래 → 무릎 lift, hip·ankle −lift/2 / 앞뒤는 hip +fwd, ankle −fwd
        reference_gait=dict(fwd=0.13, lift=0.55,  # 디딤 동안 발이 3.7 cm 뒤로 (0.15 m/s × 0.25 s), 발 들기 약 2 cm (FK로 확인)
                            roles={"hip_pitch": (1.0, -0.5), "knee": (0.0, 1.0), "ankle": (-1.0, -0.5)}),
        sim_overrides=dict(collision_boxes=_DUCK_FEET, collision_bodies=tuple(_DUCK_FEET)),
        hip_roll_joints=("left_hip_roll", "right_hip_roll"),
        head_body="head_assembly",                                       # 머리 + 목 약 0.53 kg (전체 2.06 kg), 몸통 위 0.19 m
        clearance_target=0.03,   # 장애물 2.75 cm(험한 정도 1.5)를 넘을 높이. §8.3: 흔듦 0.15 s로는 서보 속도상 약 1.7 cm가 한계
    ),
}


# 서보 모델 (docs/09 §8~9). ideal = 지금까지의 학습: 빨리 돌아도 최대 토크를 그대로 냄 (관절 damping 0).
# open_duck = Open Duck 팀이 실물(STS3215 7.4 V)을 Rhoban BAM으로 측정해 학습에 쓰는 모델 (Open_Duck_Playground
#   xmls/open_duck_mini_v2.xml의 class sts3215): 위치 kp 13.37, kv 0, 토크 ±3.23, 관절 damping 0.56 (역기전력 —
#   빨리 돌수록 토크가 줄어 약 5.8 rad/s에서 0), frictionloss 0.068, armature 0.027
SERVO_MODELS = {
    "ideal": {},
    "open_duck": dict(pd_gains=(13.37, 0.0), sim=dict(effort_limits=3.23, joint_damping=0.56, joint_frictionloss=0.068,
                                                      joint_armature=0.027)),
}


def with_servo(spec: WalkSpec, servo: str) -> WalkSpec:
    """같은 보행 과제를 다른 서보 모델로. 학습 설정(config.json)의 'servo'로 재생·분석도 같은 모델을 쓴다."""
    if servo not in SERVO_MODELS:
        raise ValueError(f"모르는 서보 모델 '{servo}' (사용 가능: {', '.join(SERVO_MODELS)})")
    model = SERVO_MODELS[servo]
    if not model:
        return spec
    return replace(spec, servo=servo, pd_gains=model["pd_gains"], sim_overrides={**spec.sim_overrides, **model["sim"]})


@functools.lru_cache(maxsize=8)
def _home_base_height(spec_name: str, home_items: tuple) -> float:
    """home 자세로 바닥에 세웠을 때 몸통(베이스) 높이 [m] (빌더가 발바닥이 바닥에 닿게 맞춘 키프레임)."""
    robot = replace(SPECS[spec_name].robot, home_pose=dict(home_items))
    model = build_robot_spec(robot.urdf, robot.sim_config(fixed_base=False, **SPECS[spec_name].sim_overrides)).compile()
    return float(model.key("home").qpos[2])


def with_home_knee(spec: WalkSpec, knee: float) -> WalkSpec:
    """서 있는 자세를 무릎 knee [rad]로 바꾼 같은 과제 (오리처럼 hip·ankle = ∓knee/2로 발바닥 수평, 발이 엉덩이 아래).
    정책 행동은 home ± action_scale이므로 운동 범위도 함께 옮겨 간다. 몸통 기준 높이·넘어짐 높이는 서 있는 높이 변화만큼 옮김.
    무릎을 덜 굽히면 한 발로 버티는 무릎 토크가 줄어든다 (docs/09 §8.2, 0.8 rad 0.74 N·m → 0.6 rad 약 0.62 N·m)."""
    base = SPECS[spec.name]
    pose = dict(base.robot.home_pose)
    hip_sign = {"left": -1.0, "right": 1.0} if spec.name == "open_duck_mini" else None
    if hip_sign is None:
        raise ValueError("자세 바꾸기(home_knee)는 지금 오리 로봇(open_duck_mini)만 지원합니다")
    for side, sign in hip_sign.items():   # 오리는 hip_pitch만 좌우 부호가 반대 (docs/09 §4)
        pose.update({f"{side}_hip_pitch": sign * knee / 2, f"{side}_knee": knee, f"{side}_ankle": -knee / 2})
    shift = (_home_base_height(spec.name, tuple(sorted(pose.items())))
             - _home_base_height(spec.name, tuple(sorted(base.robot.home_pose.items()))))
    return replace(spec, robot=replace(spec.robot, home_pose=pose), home_knee=knee,
                   nominal_height=spec.nominal_height + shift, fall_height=spec.fall_height + shift)


def mirror_table(model: mujoco.MjModel, joints, foot_frames, delta: float = 0.2) -> tuple[list, list, list]:
    """정책 관절의 좌우 짝과 부호. 매단 자세(home)에서 왼쪽 관절을 +delta 돌렸을 때 왼발이 움직인 것을 좌우로 뒤집은 것이
    오른쪽 관절 +delta 때 오른발의 움직임과 같으면 +1, 반대면 −1. (관절 이름만 보고 짐작하면 틀림: 오리의 hip_pitch는 −1)"""
    data = mujoco.MjData(model)
    home = model.key("home").qpos

    def foot_move(joint, frame):
        data.qpos[:] = home
        mujoco.mj_kinematics(model, data)
        before = data.xpos[model.body(frame).id].copy()
        data.qpos[model.joint(joint).qposadr[0]] += delta
        mujoco.mj_kinematics(model, data)
        return data.xpos[model.body(frame).id] - before

    left, right, sign = [], [], []
    for i, name in enumerate(joints):
        if not name.startswith("left_"):
            continue
        j = joints.index("right_" + name[len("left_"):])
        move_l = foot_move(name, foot_frames[0]) * np.array([1.0, -1.0, 1.0])   # 좌우(y) 뒤집기
        move_r = foot_move(joints[j], foot_frames[1])
        left.append(i)
        right.append(j)
        sign.append(1.0 if move_l @ move_r >= 0 else -1.0)
    return left, right, sign


def build_mjx_model(spec: WalkSpec = SPECS["simple_biped"], terrain=None) -> tuple[mujoco.MjModel, dict]:
    """GPU 학습용 MuJoCo 모델 (발-바닥 충돌만) + 관절 인덱스 정보. terrain: biped_sim.terrain.Terrain (None = 평지)"""
    cfg = spec.robot
    mj_spec = build_robot_spec(cfg.urdf, cfg.sim_config(fixed_base=False, timestep=PHYSICS_DT, terrain=terrain,
                                                        **spec.sim_overrides))
    model = mj_spec.compile()
    model.opt.iterations = 4
    model.opt.ls_iterations = 8

    act_joints = [int(model.actuator_trnid[a, 0]) for a in range(model.nu)]
    act_names = [model.joint(j).name for j in act_joints]
    policy = [act_names.index(n) for n in spec.policy_joints]     # 정책 관절이 액추에이터 몇 번인지
    pj = [act_joints[a] for a in policy]
    feet = [model.body(b).id for b in spec.foot_frames]
    left, right, sign = mirror_table(model, list(spec.policy_joints), spec.foot_frames)
    info = {
        "joint_names": list(spec.policy_joints),
        "qpos_idx": np.array([model.jnt_qposadr[j] for j in pj]),        # 정책 관절
        "qvel_idx": np.array([model.jnt_dofadr[j] for j in pj]),
        "q_home": np.array([cfg.home_pose[n] for n in spec.policy_joints]),
        "policy_act": np.array(policy),
        "act_qpos_idx": np.array([model.jnt_qposadr[j] for j in act_joints]),   # 모든 구동 관절 (PD)
        "act_qvel_idx": np.array([model.jnt_dofadr[j] for j in act_joints]),
        "act_home": np.array([cfg.home_pose[n] for n in act_names]),
        "act_range": model.jnt_range[act_joints].copy(),
        "kp": np.array([cfg.kp[n] if spec.pd_gains is None else spec.pd_gains[0] for n in act_names]),
        "kd": np.array([cfg.kd[n] if spec.pd_gains is None else spec.pd_gains[1] for n in act_names]),
        "tau_limit": model.actuator_ctrlrange.copy(),
        "base_body": int(model.joint(FREEJOINT_NAME).bodyid[0]),
        "foot_bodies": np.array(feet),
        "home_qpos": model.key("home").qpos.copy(),
        "mirror_left": np.array(left), "mirror_right": np.array(right), "mirror_sign": np.array(sign),
    }
    if spec.head_body:
        info["head_body"] = model.body(spec.head_body).id
    if spec.hip_roll_joints:   # home 자세에서 무게중심 ~ 디딤 쪽 hip_roll 축의 옆 거리 (com_over_stance의 기준)
        data = mujoco.MjData(model)
        data.qpos[:] = info["home_qpos"]
        mujoco.mj_forward(model, data)
        hips = [model.joint(j).id for j in spec.hip_roll_joints]
        info["hip_roll_joints"] = np.array(hips)
        info["com_hip_nominal"] = float(data.xanchor[hips[0], 1] - data.subtree_com[info["base_body"], 1])
    return model, info


def spec_from_config(config: dict) -> WalkSpec:
    """학습 설정(config.json)의 로봇 보행 과제. 재생·평가·ROS2 policy_node가 학습과 같은 과제를 만들 때 씀.
    (예전 학습은 robot 항목이 없음 = simple_biped, servo 항목이 없음 = ideal, home_knee 항목이 없음 = 로봇 설정의 서 있는 자세)"""
    spec = with_servo(SPECS[config.get("robot", "simple_biped")], config.get("servo", "ideal"))
    if config.get("home_knee") is not None:
        spec = with_home_knee(spec, config["home_knee"])
    if config.get("gait_period") and config["gait_period"] != spec.gait_period:   # --gait-period로 바꾼 걸음 박자
        spec = replace(spec, gait_period=config["gait_period"])
    if config.get("deployable_obs"):                                             # --deployable-obs
        spec = replace(spec, deployable_obs=True)
    if config.get("command_obs"):                                                # --cmd-* (명령 따라 걷기)
        spec = replace(spec, command_obs=True)
    if config.get("obs_history"):                                                # --obs-history
        spec = replace(spec, obs_history=int(config["obs_history"]))
    if config.get("clearance_target"):                                           # --clearance-target
        spec = replace(spec, clearance_target=float(config["clearance_target"]))
    if config.get("lateral_deadband"):                                           # --lateral-deadband
        spec = replace(spec, lateral_deadband=float(config["lateral_deadband"]))
    if config.get("roll_free") is not None:                                      # --roll-free
        spec = replace(spec, roll_free=float(config["roll_free"]))
    if config.get("head_sway_gate"):                                             # --head-sway-gate
        spec = replace(spec, head_sway_gate=tuple(float(x) for x in config["head_sway_gate"]))
    if config.get("double_support_fast") is not None:                            # --double-support-fast
        spec = replace(spec, double_support_fast=float(config["double_support_fast"]),
                       fast_speed=float(config.get("fast_speed", spec.fast_speed)))
    return spec


def base_observation(spec: WalkSpec, gravity_body, gyro, q_rel, qvel, last_action, command, t: float,
                     vel_body=None, height: float = 0.0) -> np.ndarray:
    """한 스텝의 정책 관측 (numpy, 학습 환경 walk_mjx._observation과 같은 순서·크기 = spec.base_obs_size).

    gravity_body : 몸통 좌표계의 중력 방향 단위 벡터 (똑바로 서면 (0, 0, −1)) — IMU 자세에서 계산
    gyro         : 몸통 각속도 [rad/s] (IMU)            q_rel : 정책 관절 각도 − home [rad]
    qvel         : 정책 관절 속도 [rad/s]               last_action : 직전 행동 (n개)
    command      : 명령 모드면 (vx, vy, wz), 아니면 목표 속도 (1개)    t : 에피소드 시작 뒤 시간 [s] (걸음 박자)
    vel_body, height : 시뮬레이션만 아는 몸통 속도·높이 오차. spec.deployable_obs면 쓰지 않음 (0)"""
    if spec.deployable_obs or vel_body is None:
        vel_body, height = np.zeros(3), 0.0
    phase = 2 * np.pi * t / spec.gait_period
    return np.concatenate([gravity_body, gyro, vel_body, [height], q_rel, 0.1 * np.asarray(qvel), last_action,
                           np.atleast_1d(command), [np.sin(phase), np.cos(phase)]]).astype(np.float32)


def double_support_width(spec: WalkSpec, speed_x):
    """gait_phase 보상의 양발 지지 폭 (박자 전환 앞뒤로 한 주기의 몇 배). 앞뒤 명령 속도 |vx|가 target_speed 이하면
    double_support, fast_speed 이상이면 double_support_fast, 사이는 직선으로. numpy·jax 모두 (abs·min·max만 씀)."""
    if spec.double_support_fast is None:
        return spec.double_support
    frac = (abs(speed_x) - spec.target_speed) / (spec.fast_speed - spec.target_speed)
    frac = frac * (frac > 0)                       # clip 0~1 (jax 배열에도 되게 연산자로)
    frac = frac - (frac - 1) * (frac > 1)
    return spec.double_support + (spec.double_support_fast - spec.double_support) * frac


def head_sway_gate(spec: WalkSpec, command):
    """head_sway 벌점에 곱하는 값 (0~1): 뒤로 가는 명령·도는 명령일수록 작게. spec.head_sway_gate가 없으면 1.
    numpy·jax 모두 (곱셈·비교만 씀). command = (vx, vy, wz)."""
    if spec.head_sway_gate is None:
        return 1.0
    back, turn = spec.head_sway_gate
    b = -command[0] / back
    b = b * (b > 0) - (b - 1) * (b > 1)            # clip 0~1
    w = abs(command[2]) / turn
    w = w - (w - 1) * (w > 1)
    return (1 - b) * (1 - w)
