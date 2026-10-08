"""BipedWalkMjxEnv — GPU(MuJoCo MJX)에서 수천 개를 동시에 돌리는 보행 학습 환경. Brax PPO용.

⚠ GPU 학습 환경(.venv-mjx)에서만 import할 수 있습니다 (jax, mujoco.mjx, brax 필요).
   설치: bash scripts/setup_gpu_learning.sh  /  사용: source scripts/activate_gpu.sh

로봇마다 다른 값은 WalkSpec 하나에 모여 있다 (SPECS: simple_biped, open_duck_mini).
    python learning/train_walk_gpu.py --robot open_duck_mini

과제
    앞(+x)으로 목표 속도로 걷기. 넘어지면(기울기 > 0.6 rad 또는 몸통이 fall_height보다 낮음) 에피소드 종료.

행동 (n개, −1~1) → 정책 관절 목표 = home + action_scale × 행동, 나머지 관절(오리의 머리 등)은 home 유지
              → 모든 관절은 PD가 토크 계산 (CPU 환경·ROS2 sim_node와 같은 구조)
관측 (13 + 3n) 중력 방향(3) 각속도(3) 몸통 속도(3) 높이 오차(1) 관절 각도−home(n) 관절 속도×0.1(n) 직전 행동(n)
              목표 속도(1) 걸음 박자 sin/cos(2)        (simple_biped: n=6 → 31, 오리: 다리 n=10 → 43)
보상          REWARD_WEIGHTS의 항목 합. 항목별 에피소드 합계가 TensorBoard eval/episode_reward/ 에 기록된다.
              가중치는 학습 명령에서 바꿀 수 있다:  --reward heading=-1.0 --reward flight=-1.0

GPU(MJX)를 위한 단순화 (CPU MuJoCo와 다른 점) — docs/08 §3
    - 충돌: 발바닥(상자) ↔ 바닥만 계산. MJX는 원기둥-상자 충돌을 지원하지 않고, 다리·몸통이 바닥에 닿으면 이미 넘어진 것
      (오리처럼 발이 CAD 메시면 메시를 감싸는 상자로 바꿈: SimConfig.collision_boxes)
    - 물리 시간 간격 0.004 s (CPU 기본 0.002 s), 접촉 풀이 반복 4회 (MJX 권장값) → 속도를 위해
"""
from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

# MJX는 처음 import될 때(Brax를 import해도 같이 됨) 선택 백엔드(NVIDIA Warp)가 없으면
# "Failed to import warp ..."를 print한다 (오류 아님). 매번 나오는 이 안내만 숨긴다.
# → 보행 학습 스크립트들은 Brax보다 이 모듈을 먼저 import한다.
with contextlib.redirect_stdout(io.StringIO()):
    from brax.envs.base import Env, State
    from mujoco import mjx

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
}
# 보상을 하나씩 더해 온 과정 (simple_biped 실측, docs/08 §5~6):
#   flight·gait_phase가 없으면 → 두 발을 다 띄우고 깡충깡충 뜀 (공중 70 %)
#   symmetry·step_length·heading이 없으면 → 오른다리는 늘 앞, 왼다리는 늘 뒤인 절뚝이는 걸음 (절뚝임 점수 40 %)
#   모두 켜고 처음부터 학습 → 대칭으로 걸음 (절뚝임 1.8 %, 좌우 다리 차이 0.8°)
# 보상과 상관없이 늘 기록하는 걸음 지표 (가중치 0인 항목도 실제로 어떤지 보이게). 학습 스크립트가 에피소드 평균
# 비율로 바꿔 TensorBoard walk/gait_<이름>_pct, walk/abs_heading_deg 로 기록한다.
GAIT_METRICS = ("flight", "single", "double", "phase_match", "abs_heading", "leg_asymmetry")


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

    @property
    def n(self) -> int:
        return len(self.policy_joints)

    @property
    def obs_size(self) -> int:
        return 13 + 3 * self.n

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
    ),
}


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


def build_mjx_model(spec: WalkSpec = SPECS["simple_biped"]) -> tuple[mujoco.MjModel, dict]:
    """GPU 학습용 MuJoCo 모델 (발-바닥 충돌만) + 관절 인덱스 정보."""
    cfg = spec.robot
    mj_spec = build_robot_spec(cfg.urdf, cfg.sim_config(fixed_base=False, timestep=PHYSICS_DT, **spec.sim_overrides))
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
        "kp": np.array([cfg.kp[n] for n in act_names]),
        "kd": np.array([cfg.kd[n] for n in act_names]),
        "tau_limit": model.actuator_ctrlrange.copy(),
        "base_body": int(model.joint(FREEJOINT_NAME).bodyid[0]),
        "foot_bodies": np.array(feet),
        "home_qpos": model.key("home").qpos.copy(),
        "mirror_left": np.array(left), "mirror_right": np.array(right), "mirror_sign": np.array(sign),
    }
    return model, info


class BipedWalkMjxEnv(Env):
    def __init__(self, spec: WalkSpec | str = "simple_biped", target_speed: float | None = None,
                 reward_weights: dict | None = None):
        """spec: SPECS의 이름 또는 WalkSpec. target_speed: None이면 spec 기본값.
        reward_weights: 바꿀 항목만 {이름: 가중치}로 (예: {"heading": -1.0})."""
        self.spec = SPECS[spec] if isinstance(spec, str) else spec
        sp = self.spec
        unknown = set(reward_weights or {}) - set(sp.reward_weights)
        if unknown:
            raise ValueError(f"모르는 보상 항목 {sorted(unknown)} (사용 가능: {sorted(sp.reward_weights)})")
        self.reward_weights = {**sp.reward_weights, **(reward_weights or {})}
        self.mj_model, info = build_mjx_model(sp)
        self.mx = mjx.put_model(self.mj_model)
        self.target_speed = sp.target_speed if target_speed is None else target_speed
        self.step_target = max(self.target_speed * sp.gait_period / 2, 0.02)   # 목표 보폭 [m] = 속도 × 반 박자
        self.air_time_target = 0.25 * sp.gait_period      # feet_air_time 기준 (simple_biped 0.2 s)
        self.min_step_air = 0.125 * sp.gait_period        # 이보다 짧게 뜬 착지는 걸음이 아님 (simple_biped 0.1 s)
        self.n_substeps = round(CONTROL_DT / PHYSICS_DT)
        f32 = lambda x: jnp.asarray(x, dtype=jnp.float32)  # noqa: E731
        self.qpos_idx, self.qvel_idx = jnp.asarray(info["qpos_idx"]), jnp.asarray(info["qvel_idx"])
        self.q_home = f32(info["q_home"])
        self.policy_act = jnp.asarray(info["policy_act"])
        self.act_qpos_idx, self.act_qvel_idx = jnp.asarray(info["act_qpos_idx"]), jnp.asarray(info["act_qvel_idx"])
        self.act_home = f32(info["act_home"])
        self.act_low, self.act_high = f32(info["act_range"][:, 0]), f32(info["act_range"][:, 1])
        self.kp, self.kd = f32(info["kp"]), f32(info["kd"])
        self.tau_low, self.tau_high = f32(info["tau_limit"][:, 0]), f32(info["tau_limit"][:, 1])
        self.base_body = info["base_body"]
        self.foot_bodies = jnp.asarray(info["foot_bodies"])
        self.home_qpos = f32(info["home_qpos"])
        self.mirror_left, self.mirror_right = jnp.asarray(info["mirror_left"]), jnp.asarray(info["mirror_right"])
        self.mirror_sign = f32(info["mirror_sign"])
        self.joint_names = info["joint_names"]
        self.ref_fwd, self.ref_lift, self.ref_is_right = self._reference_coefficients(info)

    # ------------------------------------------------------------------ Brax Env API
    @property
    def observation_size(self) -> int:
        return self.spec.obs_size

    @property
    def action_size(self) -> int:
        return self.spec.n

    @property
    def backend(self) -> str:
        return "mjx"

    def reset(self, rng: jax.Array) -> State:
        rng, k1 = jax.random.split(rng)
        n = self.spec.n
        qpos = self.home_qpos.at[self.qpos_idx].add(jax.random.uniform(k1, (n,), minval=-0.03, maxval=0.03))
        data = mjx.make_data(self.mj_model).replace(qpos=qpos)
        data = mjx.forward(self.mx, data)
        info = {
            "rng": rng,
            "last_action": jnp.zeros(n),
            "time": jnp.zeros(()),
            "feet_air_time": jnp.zeros(2),
            "q_hist": jnp.tile(qpos[self.qpos_idx], (self.spec.half_period_steps, 1)),   # 반 박자 동안 관절 각도
        }
        metrics = {f"reward/{k}": jnp.zeros(()) for k in self.reward_weights}
        metrics.update({f"gait/{k}": jnp.zeros(()) for k in GAIT_METRICS})
        metrics.update({"forward_speed": jnp.zeros(())})
        obs = self._observation(data, info)
        return State(pipeline_state=data, obs=obs, reward=jnp.zeros(()), done=jnp.zeros(()),
                     metrics=metrics, info=info)

    def step(self, state: State, action: jax.Array) -> State:
        sp = self.spec
        action = jnp.clip(action, -1.0, 1.0)
        # 정책 관절은 home + 행동, 나머지(머리 등)는 home. 관절 범위 밖은 자름
        q_des = self.act_home.at[self.policy_act].add(sp.action_scale * action)
        q_des = jnp.clip(q_des, self.act_low, self.act_high)

        def substep(data, _):  # 물리 1스텝마다 PD로 토크 계산 (모터 드라이버 역할)
            q = data.qpos[self.act_qpos_idx]
            dq = data.qvel[self.act_qvel_idx]
            tau = jnp.clip(self.kp * (q_des - q) - self.kd * dq, self.tau_low, self.tau_high)
            data = mjx.step(self.mx, data.replace(ctrl=tau))
            return data, tau

        data, taus = jax.lax.scan(substep, state.pipeline_state, None, length=self.n_substeps)
        info = dict(state.info)
        info["time"] = state.info["time"] + CONTROL_DT

        # --- 상태 계산
        rot = data.xmat[self.base_body]                    # 몸통 → world
        vel_world = data.qvel[0:3]
        vel_body = rot.T @ vel_world
        ang_body = data.qvel[3:6]                          # freejoint 각속도는 몸통 좌표계
        tilt_cos = rot[2, 2]
        tilt = jnp.arccos(jnp.clip(tilt_cos, -1.0, 1.0))
        yaw = jnp.arctan2(rot[1, 0], rot[0, 0])
        height = data.xpos[self.base_body, 2]
        foot_z = data.xpos[self.foot_bodies, 2]
        contact = foot_z < sp.foot_contact_z

        # --- 발 공중 시간: 착지하는 순간 (공중에 있던 시간 − 기준)만큼 보상
        air_time = state.info["feet_air_time"] + CONTROL_DT
        first_contact = contact & (state.info["feet_air_time"] > 0.0)
        air_reward = jnp.sum((air_time - self.air_time_target) * first_contact)
        info["feet_air_time"] = jnp.where(contact, 0.0, air_time)

        # --- 좌우 대칭: 지금 왼다리 관절 각도 vs 반 박자 전 오른다리 (그리고 반대). 좌우 부호는 mirror_table
        q = data.qpos[self.qpos_idx]
        q_half = state.info["q_hist"][-1]                    # 반 박자 전
        s = self.mirror_sign
        asym = (jnp.sum(jnp.square(q[self.mirror_left] - s * q_half[self.mirror_right]))
                + jnp.sum(jnp.square(q[self.mirror_right] - s * q_half[self.mirror_left])))
        info["q_hist"] = jnp.concatenate([q[None], state.info["q_hist"][:-1]])

        # --- 보폭: 착지하는 발이 반대 발보다 얼마나 앞에 놓였나 (목표 보폭에서 최대 1점, 그 이상은 똑같이 1점)
        foot_x = data.xpos[self.foot_bodies, 0]
        ahead = foot_x - foot_x[::-1]
        real_step = first_contact & (air_time >= self.min_step_air)   # 살짝 튕긴 착지는 걸음이 아님
        step_reward = jnp.sum(real_step * jnp.clip(ahead, -0.2, self.step_target) / self.step_target)

        terms = {
            "forward_velocity": jnp.exp(-jnp.square(vel_world[0] - self.target_speed) / sp.velocity_sigma2),
            "lateral_velocity": jnp.square(vel_world[1]),
            "yaw_rate": jnp.square(ang_body[2]),
            "upright": jnp.square(tilt),
            "height": jnp.square(height - sp.nominal_height),
            "feet_air_time": air_reward,
            "action_rate": jnp.sum(jnp.square(action - state.info["last_action"])),
            "torque": jnp.sum(jnp.square(taus[-1])),
            "alive": jnp.ones(()),
            "heading": jnp.square(yaw),
            "flight": jnp.where(contact.any(), 0.0, 1.0),
            "gait_phase": self._gait_phase_match(info["time"], contact, sp.gait_period),
            "symmetry": asym,
            "step_length": step_reward,
            "reference": self._reference_match(q, info["time"]),
        }
        weighted = {k: self.reward_weights[k] * v for k, v in terms.items()}
        reward = sum(weighted.values()) * CONTROL_DT          # 스텝 길이로 나눠 에피소드 합이 '초당' 의미가 되게
        done = jnp.where((tilt > FALL_TILT) | (height < sp.fall_height), 1.0, 0.0)

        info["last_action"] = action
        # Brax 래퍼가 넣어 둔 항목(예: 'reward')을 지우지 않도록, 들어온 metrics를 복사해서 값만 갱신
        metrics = dict(state.metrics)
        metrics.update({f"reward/{k}": v * CONTROL_DT for k, v in weighted.items()})
        both, none = contact.all(), ~contact.any()
        metrics.update({
            "gait/flight": none.astype(jnp.float32),                 # 두 발 모두 공중
            "gait/double": both.astype(jnp.float32),                 # 두 발 모두 땅
            "gait/single": (~both & ~none).astype(jnp.float32),      # 한 발만 땅 (걷기의 핵심)
            "gait/phase_match": terms["gait_phase"],                 # 걸음 박자와 일치한 정도
            "gait/abs_heading": jnp.abs(yaw),                        # 처음 방향에서 돌아간 각도 [rad]
            "gait/leg_asymmetry": asym,                              # 좌우 다리 동작 차이 [rad², 정책 관절 합]
        })
        metrics["forward_speed"] = vel_world[0]
        obs = self._observation(data, info, vel_body=vel_body, ang_body=ang_body)
        return state.replace(pipeline_state=data, obs=obs, reward=reward, done=done,
                             metrics=metrics, info=info)

    # ------------------------------------------------------------------ 참고 동작 (모방 보상)
    def _reference_coefficients(self, info):
        """정책 관절마다 (앞뒤 계수 × 진폭, 들어올림 계수 × 진폭, 오른쪽 다리인가). 오른쪽은 좌우 부호표를 곱함."""
        n = self.spec.n
        fwd, lift, right = np.zeros(n), np.zeros(n), np.zeros(n)
        gait = self.spec.reference_gait
        if gait:
            sign = np.ones(n)
            sign[info["mirror_right"]] = info["mirror_sign"]
            for i, name in enumerate(self.spec.policy_joints):
                side, joint = name.split("_", 1)
                c_fwd, c_lift = gait["roles"].get(joint, (0.0, 0.0))
                speed_ratio = self.target_speed / self.spec.target_speed   # 빠르게 걸으면 보폭도 크게
                fwd[i] = sign[i] * c_fwd * gait["fwd"] * speed_ratio
                lift[i] = sign[i] * c_lift * gait["lift"]
                right[i] = side == "right"
        return jnp.asarray(fwd, jnp.float32), jnp.asarray(lift, jnp.float32), jnp.asarray(right, jnp.float32)

    def reference_pose(self, t):
        """시각 t의 참고 관절 각도. 걸음 박자의 앞 절반은 왼발 디딤·오른발 흔듦, 뒤 절반은 반대 (gait_phase와 같은 규칙).
        다리마다: 디딤 동안 앞뒤 값이 +1 → −1 (발을 뒤로 밂), 흔듦 동안 −1 → +1 (앞으로 가져옴), 들어올림은 흔듦 중 sin."""
        phase = jnp.mod(t / self.spec.gait_period, 1.0)
        leg_phase = jnp.mod(phase + 0.5 * self.ref_is_right, 1.0)       # 오른다리는 반 박자 늦게
        stance = leg_phase < 0.5
        progress = jnp.where(stance, leg_phase, leg_phase - 0.5) / 0.5  # 디딤/흔듦 안에서 0 → 1
        fwd = jnp.where(stance, 1.0 - 2.0 * progress, -jnp.cos(jnp.pi * progress))
        lift = jnp.where(stance, 0.0, jnp.sin(jnp.pi * progress))
        return self.q_home + self.ref_fwd * fwd + self.ref_lift * lift

    def _reference_match(self, q, t):
        if not self.spec.reference_gait:
            return jnp.zeros(())
        err = jnp.sum(jnp.square(q - self.reference_pose(t)))
        return jnp.exp(-err / (self.spec.n * 0.15 ** 2))

    @staticmethod
    def _gait_phase_match(t, contact, period):
        """걸음 박자의 앞 절반은 왼발만, 뒤 절반은 오른발만 땅을 딛는 것이 '정답'.
        박자가 바뀌는 앞뒤 0.1 주기(양발 지지 구간)는 두 발 모두 딛는 것이 정답. 발마다 맞으면 0.5점."""
        phase = jnp.mod(t / period, 1.0)
        double = (phase < 0.1) | ((phase > 0.4) & (phase < 0.6)) | (phase > 0.9)
        want_left = (phase < 0.5) | double
        want_right = (phase >= 0.5) | double
        match_left = jnp.where(want_left, contact[0], ~contact[0])
        match_right = jnp.where(want_right, contact[1], ~contact[1])
        return 0.5 * (match_left.astype(jnp.float32) + match_right.astype(jnp.float32))

    # ------------------------------------------------------------------ 관측
    def _observation(self, data, info, vel_body=None, ang_body=None) -> jax.Array:
        rot = data.xmat[self.base_body]
        if vel_body is None:
            vel_body = rot.T @ data.qvel[0:3]
            ang_body = data.qvel[3:6]
        gravity_body = rot.T @ jnp.array([0.0, 0.0, -1.0])
        phase = 2.0 * jnp.pi * info["time"] / self.spec.gait_period
        return jnp.concatenate([
            gravity_body,
            ang_body,
            vel_body,
            jnp.array([data.xpos[self.base_body, 2] - self.spec.nominal_height]),
            data.qpos[self.qpos_idx] - self.q_home,
            0.1 * data.qvel[self.qvel_idx],
            info["last_action"],
            jnp.array([self.target_speed]),
            jnp.array([jnp.sin(phase), jnp.cos(phase)]),
        ])
