"""BipedWalkMjxEnv — GPU(MuJoCo MJX)에서 수천 개를 동시에 돌리는 보행 학습 환경. Brax PPO용.

⚠ GPU 학습 환경(.venv-mjx)에서만 import할 수 있습니다 (jax, mujoco.mjx, brax 필요).
   설치: bash scripts/setup_gpu_learning.sh  /  사용: source scripts/activate_gpu.sh

과제
    앞(+x)으로 목표 속도(기본 0.3 m/s)로 걷기. 넘어지면(기울기 > 0.6 rad 또는 몸통 높이 < 0.35 m) 에피소드 종료.

행동 (6개, −1~1) → 관절 목표 = home 자세 + ACTION_SCALE × 행동  →  관절 PD가 토크 계산 (CPU 환경·ROS2 sim_node와 같은 구조)
관측 (31개)   중력 방향(3) 각속도(3) 몸통 속도(3) 높이 오차(1) 관절 각도−home(6) 관절 속도×0.1(6) 직전 행동(6)
              목표 속도(1) 걸음 박자 sin/cos(2)
보상          아래 REWARD_WEIGHTS의 항목 합. 항목별 에피소드 합계가 TensorBoard eval/episode_reward/ 에 기록된다.
              가중치는 학습 명령에서 바꿀 수 있다:  --reward heading=-1.0 --reward flight=-1.0

GPU(MJX)를 위한 단순화 (CPU MuJoCo와 다른 점) — docs/08 §3
    - 충돌: 발바닥(box) ↔ 바닥만 계산. MJX는 원기둥-상자 충돌을 지원하지 않고, 다리·몸통이 바닥에 닿으면 이미 넘어진 것
    - 물리 시간 간격 0.004 s (CPU 기본 0.002 s), 접촉 풀이 반복 4회 (MJX 권장값) → 속도를 위해
"""
from __future__ import annotations

import contextlib
import io

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

from ..builder import FREEJOINT_NAME, SimConfig, build_robot_spec
from ..robot_configs import SIMPLE_BIPED

PHYSICS_DT = 0.004          # [s]
CONTROL_DT = 0.02           # [s] 정책 50 Hz → 물리 5스텝마다 행동 1번
ACTION_SCALE = 0.5          # [rad] 보행은 다리를 크게 움직여야 해서 균형 과제(0.3)보다 크게
TARGET_SPEED = 0.3          # [m/s] 앞으로
GAIT_PERIOD = 0.8           # [s] 걸음 박자 (한 쌍의 걸음). 정책에 sin/cos로 알려 줌
NOMINAL_HEIGHT = 0.588      # [m]
FALL_TILT = 0.6             # [rad]
FALL_HEIGHT = 0.35          # [m]
FOOT_CONTACT_Z = 0.035      # [m] 발목(발 바디 원점) 높이가 이보다 낮으면 '발이 땅에 닿음'으로 판단
OBS_SIZE = 31
ACTION_SIZE = 6
HALF_PERIOD_STEPS = round(GAIT_PERIOD / 2 / CONTROL_DT)   # 반 박자 = 20 제어 스텝 (symmetry 보상용 기록 길이)
MIN_STEP_AIR = 0.1          # [s] 이만큼 이상 공중에 있다가 착지해야 '한 걸음' (step_length 보상).
                            #   없으면 발을 살짝 튕겨 두 번 착지해 보상을 두 번 받는 꼼수를 배움 (실측, docs/08 §6)

REWARD_WEIGHTS = {
    "forward_velocity": 2.0,    # exp(−(vx − 목표)² / 0.05): 목표 속도에 가까울수록 최대 1
    "lateral_velocity": -1.0,   # vy² : 옆으로 미끄러지지 않기
    "yaw_rate": -0.5,           # wz² : 제자리에서 돌지 않기
    "upright": -2.0,            # 기울기²
    "height": -10.0,            # (높이 − 기준)² : 무릎을 너무 굽혀 기어가지 않기
    "feet_air_time": 2.0,       # 발이 공중에 있던 시간이 0.2 s보다 길면 착지할 때 보상 → 발을 들어 딛게 유도
    "action_rate": -0.02,       # |Δ행동|² : 떨림 방지
    "torque": -2e-4,            # |토크|² : 에너지 절약
    "alive": 0.5,               # 살아 있으면 매 스텝 +
    "flight": -1.0,             # 두 발이 모두 공중이면 1 : 뛰지 말고 걷기
    "gait_phase": 1.0,          # 걸음 박자대로 좌우 발이 번갈아 딛으면 1 (두 발 평균)
    "heading": 0.0,             # (방향 각도)² : 처음 방향(+x)을 유지하기. 기본 0 = 꺼짐 (docs/08 §5 실험)
    "symmetry": 0.0,            # Σ(왼다리 관절 − 반 박자 전 오른다리 관절)² + 반대 [rad²] : 두 다리가 같은 동작을
                                #   반 박자 어긋나게 하기 (절뚝임 방지). 기본 0 = 꺼짐 (docs/08 §6)
    "step_length": 0.0,         # 한 걸음 착지할 때 반대 발보다 앞에 놓은 거리 / 목표 보폭 (목표에서 1, 넘어도 1)
                                #   (MIN_STEP_AIR 이상 공중에 있던 착지만) (docs/08 §6)
}
# flight·gait_phase가 없으면(0으로 두면) 두 발을 다 띄우고 깡충깡충 뛰는 걸음을 배움 (실측, docs/08 §5.1).
# 보상과 상관없이 늘 기록하는 걸음 지표 (가중치 0인 항목도 실제로 어떤지 보이게). 학습 스크립트가 에피소드 평균
# 비율로 바꿔 TensorBoard walk/gait_<이름>_pct, walk/abs_heading_deg 로 기록한다.
GAIT_METRICS = ("flight", "single", "double", "phase_match", "abs_heading", "leg_asymmetry")


def build_mjx_model() -> tuple[mujoco.MjModel, dict]:
    """GPU 학습용 MuJoCo 모델 (발-바닥 충돌만) + 관절 인덱스 정보."""
    cfg = SIMPLE_BIPED
    spec = build_robot_spec(cfg.urdf, SimConfig(fixed_base=False, timestep=PHYSICS_DT,
                                                home_joint_pos=cfg.home_pose))
    for g in spec.geoms:
        if (g.contype or g.conaffinity) and g.name != "floor" and g.parent.name not in cfg.foot_bodies:
            g.contype = 0
            g.conaffinity = 0
    model = spec.compile()
    model.opt.iterations = 4
    model.opt.ls_iterations = 8

    joint_ids = [int(model.actuator_trnid[a, 0]) for a in range(model.nu)]
    names = [model.joint(j).name for j in joint_ids]
    info = {
        "joint_names": names,
        "qpos_idx": np.array([model.jnt_qposadr[j] for j in joint_ids]),
        "qvel_idx": np.array([model.jnt_dofadr[j] for j in joint_ids]),
        "q_home": np.array([cfg.home_pose[n] for n in names]),
        "q_range": model.jnt_range[joint_ids].copy(),
        "kp": np.array([cfg.kp[n] for n in names]),
        "kd": np.array([cfg.kd[n] for n in names]),
        "tau_limit": model.actuator_ctrlrange.copy(),
        "base_body": int(model.joint(FREEJOINT_NAME).bodyid[0]),
        "foot_bodies": np.array([model.body(b).id for b in cfg.foot_bodies]),
        "home_qpos": model.key("home").qpos.copy(),
    }
    return model, info


class BipedWalkMjxEnv(Env):
    def __init__(self, target_speed: float = TARGET_SPEED, reward_weights: dict | None = None):
        """reward_weights: REWARD_WEIGHTS 중 바꿀 항목만 {이름: 가중치}로 (예: {"heading": -1.0})."""
        unknown = set(reward_weights or {}) - set(REWARD_WEIGHTS)
        if unknown:
            raise ValueError(f"모르는 보상 항목 {sorted(unknown)} (사용 가능: {sorted(REWARD_WEIGHTS)})")
        self.reward_weights = {**REWARD_WEIGHTS, **(reward_weights or {})}
        self.mj_model, info = build_mjx_model()
        self.mx = mjx.put_model(self.mj_model)
        self.target_speed = target_speed
        self.step_target = max(target_speed * GAIT_PERIOD / 2, 0.05)   # 목표 보폭 [m] = 속도 × 반 박자 (0.3 → 0.12 m)
        self.n_substeps = round(CONTROL_DT / PHYSICS_DT)
        self.qpos_idx = jnp.asarray(info["qpos_idx"])
        self.qvel_idx = jnp.asarray(info["qvel_idx"])
        self.q_home = jnp.asarray(info["q_home"], dtype=jnp.float32)
        self.q_low = jnp.asarray(info["q_range"][:, 0], dtype=jnp.float32)
        self.q_high = jnp.asarray(info["q_range"][:, 1], dtype=jnp.float32)
        self.kp = jnp.asarray(info["kp"], dtype=jnp.float32)
        self.kd = jnp.asarray(info["kd"], dtype=jnp.float32)
        self.tau_low = jnp.asarray(info["tau_limit"][:, 0], dtype=jnp.float32)
        self.tau_high = jnp.asarray(info["tau_limit"][:, 1], dtype=jnp.float32)
        self.base_body = info["base_body"]
        self.foot_bodies = jnp.asarray(info["foot_bodies"])
        self.home_qpos = jnp.asarray(info["home_qpos"], dtype=jnp.float32)
        self.joint_names = info["joint_names"]

    # ------------------------------------------------------------------ Brax Env API
    @property
    def observation_size(self) -> int:
        return OBS_SIZE

    @property
    def action_size(self) -> int:
        return ACTION_SIZE

    @property
    def backend(self) -> str:
        return "mjx"

    def reset(self, rng: jax.Array) -> State:
        rng, k1 = jax.random.split(rng)
        qpos = self.home_qpos.at[self.qpos_idx].add(jax.random.uniform(k1, (6,), minval=-0.03, maxval=0.03))
        data = mjx.make_data(self.mj_model).replace(qpos=qpos)
        data = mjx.forward(self.mx, data)
        info = {
            "rng": rng,
            "last_action": jnp.zeros(6),
            "time": jnp.zeros(()),
            "feet_air_time": jnp.zeros(2),
            "q_hist": jnp.tile(qpos[self.qpos_idx], (HALF_PERIOD_STEPS, 1)),   # 최근 20스텝 관절 각도 [0]=직전
        }
        metrics = {f"reward/{k}": jnp.zeros(()) for k in REWARD_WEIGHTS}
        metrics.update({f"gait/{k}": jnp.zeros(()) for k in GAIT_METRICS})
        metrics.update({"forward_speed": jnp.zeros(())})
        obs = self._observation(data, info)
        return State(pipeline_state=data, obs=obs, reward=jnp.zeros(()), done=jnp.zeros(()),
                     metrics=metrics, info=info)

    def step(self, state: State, action: jax.Array) -> State:
        action = jnp.clip(action, -1.0, 1.0)
        q_des = jnp.clip(self.q_home + ACTION_SCALE * action, self.q_low, self.q_high)

        def substep(data, _):  # 물리 1스텝마다 PD로 토크 계산 (모터 드라이버 역할)
            q = data.qpos[self.qpos_idx]
            dq = data.qvel[self.qvel_idx]
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
        contact = foot_z < FOOT_CONTACT_Z

        # --- 발 공중 시간: 착지하는 순간 (공중에 있던 시간 − 0.2 s)만큼 보상
        air_time = state.info["feet_air_time"] + CONTROL_DT
        first_contact = contact & (state.info["feet_air_time"] > 0.0)
        air_reward = jnp.sum((air_time - 0.2) * first_contact)
        info["feet_air_time"] = jnp.where(contact, 0.0, air_time)

        # --- 좌우 대칭: 지금 왼다리 관절 각도 vs 반 박자 전 오른다리 (그리고 반대). 피치 관절이라 좌우 부호가 같음
        q = data.qpos[self.qpos_idx]
        q_half = state.info["q_hist"][-1]                    # 반 박자(20스텝) 전
        asym = jnp.sum(jnp.square(q[:3] - q_half[3:])) + jnp.sum(jnp.square(q[3:] - q_half[:3]))
        info["q_hist"] = jnp.concatenate([q[None], state.info["q_hist"][:-1]])

        # --- 보폭: 착지하는 발이 반대 발보다 얼마나 앞에 놓였나 (목표 보폭에서 최대 1점, 그 이상은 똑같이 1점)
        foot_x = data.xpos[self.foot_bodies, 0]
        ahead = foot_x - foot_x[::-1]
        real_step = first_contact & (air_time >= MIN_STEP_AIR)   # 살짝 튕긴 착지는 걸음이 아님
        step_reward = jnp.sum(real_step * jnp.clip(ahead, -0.2, self.step_target) / self.step_target)

        terms = {
            "forward_velocity": jnp.exp(-jnp.square(vel_world[0] - self.target_speed) / 0.05),
            "lateral_velocity": jnp.square(vel_world[1]),
            "yaw_rate": jnp.square(ang_body[2]),
            "upright": jnp.square(tilt),
            "height": jnp.square(height - NOMINAL_HEIGHT),
            "feet_air_time": air_reward,
            "action_rate": jnp.sum(jnp.square(action - state.info["last_action"])),
            "torque": jnp.sum(jnp.square(taus[-1])),
            "alive": jnp.ones(()),
            "heading": jnp.square(yaw),
            "flight": jnp.where(contact.any(), 0.0, 1.0),
            "gait_phase": self._gait_phase_match(info["time"], contact),
            "symmetry": asym,
            "step_length": step_reward,
        }
        weighted = {k: self.reward_weights[k] * v for k, v in terms.items()}
        reward = sum(weighted.values()) * CONTROL_DT          # 스텝 길이로 나눠 에피소드 합이 '초당' 의미가 되게
        done = jnp.where((tilt > FALL_TILT) | (height < FALL_HEIGHT), 1.0, 0.0)

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
            "gait/leg_asymmetry": asym,                              # 좌우 다리 동작 차이 [rad², 관절 6개 합]
        })
        metrics["forward_speed"] = vel_world[0]
        obs = self._observation(data, info, vel_body=vel_body, ang_body=ang_body)
        return state.replace(pipeline_state=data, obs=obs, reward=reward, done=done,
                             metrics=metrics, info=info)

    @staticmethod
    def _gait_phase_match(t, contact):
        """걸음 박자(GAIT_PERIOD)의 앞 절반은 왼발만, 뒤 절반은 오른발만 땅을 딛는 것이 '정답'.
        박자가 바뀌는 앞뒤 0.1 주기(양발 지지 구간)는 두 발 모두 딛는 것이 정답. 발마다 맞으면 0.5점."""
        phase = jnp.mod(t / GAIT_PERIOD, 1.0)
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
        phase = 2.0 * jnp.pi * info["time"] / GAIT_PERIOD
        return jnp.concatenate([
            gravity_body,
            ang_body,
            vel_body,
            jnp.array([data.xpos[self.base_body, 2] - NOMINAL_HEIGHT]),
            data.qpos[self.qpos_idx] - self.q_home,
            0.1 * data.qvel[self.qvel_idx],
            info["last_action"],
            jnp.array([self.target_speed]),
            jnp.array([jnp.sin(phase), jnp.cos(phase)]),
        ])
