"""BipedWalkMjxEnv — GPU(MuJoCo MJX)에서 수천 개를 동시에 돌리는 보행 학습 환경. Brax PPO용.

⚠ GPU 학습 환경(.venv-mjx)에서만 import할 수 있습니다 (jax, mujoco.mjx, brax 필요).
   설치: bash scripts/setup_gpu_learning.sh  /  사용: source scripts/activate_gpu.sh

로봇마다 다른 값은 WalkSpec 하나에 모여 있다 (SPECS: simple_biped, open_duck_mini). 과제 설정·모델 만들기는
jax 없이 쓸 수 있게 walk_spec.py에 있고(ROS2 노드도 씀), 여기서 그대로 다시 내보낸다.
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

from ..builder import FREEJOINT_NAME
from .walk_spec import (CONTROL_DT, COM_SHIFT_TARGET, COMMANDS_DEFAULT, DOMAIN_RAND, FALL_TILT,  # noqa: F401 (다른 모듈이 walk_mjx.이름으로 씀)
                        GAIT_METRICS, OBS_NOISE, PHYSICS_DT, REWARD_WEIGHTS, SERVO_DATASHEET, SERVO_MODELS, SPECS,
                        WalkSpec, _home_base_height, base_observation, build_mjx_model, double_support_width,
                        head_sway_gate, mirror_table, spec_from_config, with_home_knee, with_servo)


def tilt_matrix(roll, pitch):
    """IMU 장착 기울기 (작은 roll·pitch) → 회전 행렬. 읽은 값 = 이 행렬 × 실제 값. jax·numpy 모두 (xp로 고름)."""
    xp = jnp if isinstance(roll, jax.Array) else np
    cr, sr, cp, sp = xp.cos(roll), xp.sin(roll), xp.cos(pitch), xp.sin(pitch)
    rx = xp.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = xp.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    return rx @ ry


class BipedWalkMjxEnv(Env):
    def __init__(self, spec: WalkSpec | str = "simple_biped", target_speed: float | None = None,
                 reward_weights: dict | None = None, terrain=None, spawn_area=None, friction_range=None, push=None,
                 obs_noise: bool = False, privileged_critic: bool = False, domain_rand: bool = False,
                 commands: dict | None = None):
        """spec: SPECS의 이름 또는 WalkSpec. target_speed: None이면 spec 기본값.
        reward_weights: 바꿀 항목만 {이름: 가중치}로 (예: {"heading": -1.0}).
        terrain: 울퉁불퉁한 지형 (biped_sim.terrain.Terrain). 몸통 높이·발 접촉은 그 자리 땅을 기준으로 잰다.
        spawn_area: ((x_min, x_max), (y_min, y_max)) — 에피소드마다 이 안의 무작위 위치에서 출발 (넓은 지형의 여러 곳 경험)
        friction_range: (최소, 최대) — 환경마다 바닥 마찰 계수를 이 안에서 무작위로 (None = MuJoCo 기본 1.0).
            Brax의 자동 리셋은 첫 리셋 상태로 되돌리므로 환경마다 학습 내내 같은 값 (1024개 환경이 범위를 골고루 나눠 가짐)
        push: (최대 속도 [m/s], 간격 [s]) — 간격마다 몸통을 수평 무작위 방향으로 0~최대 속도만큼 툭 밈 (외란 버티기). None = 없음
        obs_noise: 관측에 센서 잡음 (OBS_NOISE: 중력 방향 0.02, 각속도 0.05 rad/s, 관절 각도 0.01 rad, 관절 속도 0.5 rad/s)
        privileged_critic: 비대칭 actor-critic. 관측을 {"state": 정책이 보는 것(실물 관측·잡음), "privileged_state": 가리지 않은
            전체 값(잡음 없음)}로 줌 → 학습 중 점수만 매기는 critic(가치 함수)은 시뮬레이션의 모든 정보를 씀 (실물에 올리는 것은 정책뿐)
        domain_rand: 환경마다 몸통 질량·모터 세기·무게중심·제어 지연을 DOMAIN_RAND 범위에서 무작위로 (실물과의 차이 대비)
        commands: 명령 따라 걷기의 명령 범위 {"vx": (최소, 최대), "vy": ..., "wz": ..., "resample": 초} (spec.command_obs 필요).
            None이면 명령 = (target_speed, 0, 0) 고정"""
        self.spec = SPECS[spec] if isinstance(spec, str) else spec
        sp = self.spec
        unknown = set(reward_weights or {}) - set(sp.reward_weights)
        if unknown:
            raise ValueError(f"모르는 보상 항목 {sorted(unknown)} (사용 가능: {sorted(sp.reward_weights)})")
        self.reward_weights = {**sp.reward_weights, **(reward_weights or {})}
        self.terrain, self.spawn_area, self.friction_range, self.push = terrain, spawn_area, friction_range, push
        self.privileged_critic = privileged_critic
        self.domain_rand = domain_rand
        if commands and not sp.command_obs:
            raise ValueError("명령 범위(commands)는 command_obs인 과제에서만 씁니다")
        self.commands = {**COMMANDS_DEFAULT, **commands} if commands else None
        n = sp.n
        self.noise_scale = None if not obs_noise else jnp.asarray(np.concatenate([   # 한 스텝 관측(base_obs_size)에
            [OBS_NOISE["gravity"]] * 3, [OBS_NOISE["gyro"]] * 3, [0.0] * 4,          # 중력, 각속도, (속도·높이)
            [OBS_NOISE["joint_pos"]] * n, [0.1 * OBS_NOISE["joint_vel"]] * n,     # 관절 각도, 관절 속도 × 0.1
            [0.0] * (sp.base_obs_size - 10 - 2 * n)]), dtype=jnp.float32)          # 직전 행동, 목표 속도(명령), 박자
        self.mj_model, info = build_mjx_model(sp, terrain)
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
        self.act_damping = f32(self.mj_model.dof_damping[info["act_qvel_idx"]])   # 실측 서보 모델이면 0.56
        self.base_body = info["base_body"]
        self.foot_bodies = jnp.asarray(info["foot_bodies"])
        self.home_qpos = f32(info["home_qpos"])
        self.mirror_left, self.mirror_right = jnp.asarray(info["mirror_left"]), jnp.asarray(info["mirror_right"])
        self.mirror_sign = f32(info["mirror_sign"])
        self.joint_names = info["joint_names"]
        self.ref_fwd, self.ref_lift, self.ref_is_right = self._reference_coefficients(info)
        self.hip_roll_joints = jnp.asarray(info["hip_roll_joints"]) if "hip_roll_joints" in info else None
        self.com_hip_nominal = info.get("com_hip_nominal", 0.0)
        self.head_body = info.get("head_body")
        if terrain is not None:
            self.ground_heights = f32(terrain.heights)
            self.ground_origin = f32([terrain.center[0] - terrain.size_x, terrain.center[1] - terrain.size_y])
            self.ground_step = f32([2 * terrain.size_x / (terrain.ncol - 1), 2 * terrain.size_y / (terrain.nrow - 1)])

    def ground(self, xy):
        """위치 xy [..., 2]의 땅 높이 (지형 격자 선형 보간 = terrain.Terrain.height_at). 평지면 0."""
        if self.terrain is None:
            return jnp.zeros(xy.shape[:-1])
        g = (xy - self.ground_origin) / self.ground_step                 # [..., (열, 행)]
        h = self.ground_heights
        g = jnp.clip(g, 0.0, jnp.array([h.shape[1], h.shape[0]], jnp.float32) - 1.000001)
        c0, r0 = jnp.floor(g[..., 0]).astype(int), jnp.floor(g[..., 1]).astype(int)
        fc, fr = g[..., 0] - c0, g[..., 1] - r0
        return ((1 - fr) * ((1 - fc) * h[r0, c0] + fc * h[r0, c0 + 1])
                + fr * ((1 - fc) * h[r0 + 1, c0] + fc * h[r0 + 1, c0 + 1]))

    # ------------------------------------------------------------------ Brax Env API
    @property
    def observation_size(self):
        if self.privileged_critic:   # critic은 지금 스텝의 전체 관측만 (기록 없이)
            return {"state": (self.spec.obs_size,), "privileged_state": (self.spec.base_obs_size,)}
        return self.spec.obs_size

    @property
    def action_size(self) -> int:
        return self.spec.n

    @property
    def backend(self) -> str:
        return "mjx"

    def reset(self, rng: jax.Array) -> State:
        rng, k1, k2, k3 = jax.random.split(rng, 4)
        n = self.spec.n
        qpos = self.home_qpos.at[self.qpos_idx].add(jax.random.uniform(k1, (n,), minval=-0.03, maxval=0.03))
        if self.spawn_area is not None:   # 넓은 지형의 무작위 위치에서 출발: 두 발 밑의 땅 중 높은 쪽에 맞춰 올려놓음
            (x0, x1), (y0, y1) = self.spawn_area
            xy = jax.random.uniform(k2, (2,), minval=jnp.array([x0, y0]), maxval=jnp.array([x1, y1]))
            feet = xy + jnp.array([[0.0, 0.1], [0.0, -0.1], [0.06, 0.1], [0.06, -0.1], [-0.06, 0.1], [-0.06, -0.1]])
            qpos = qpos.at[0:2].set(xy).at[2].add(jnp.max(self.ground(feet)) + 0.003)
        data = mjx.make_data(self.mj_model).replace(qpos=qpos)
        data = mjx.forward(self.mx, data)
        info = {
            "rng": rng,
            "last_action": jnp.zeros(n),
            "time": jnp.zeros(()),
            "feet_air_time": jnp.zeros(2),
            "swing_peak": jnp.zeros(2),       # 이번 흔듦 동안 발을 가장 높이 든 높이 [m] (땅에 닿으면 0)
            "q_hist": jnp.tile(qpos[self.qpos_idx], (self.spec.half_period_steps, 1)),   # 반 박자 동안 관절 각도
        }
        if self.friction_range is not None:
            info["friction"] = jax.random.uniform(k3, (), minval=self.friction_range[0], maxval=self.friction_range[1])
        if self.spec.command_obs:   # 명령과, 회전 명령을 쌓은 '가야 할 방향' (heading 보상의 기준)
            info["command"] = (self._sample_command(jax.random.fold_in(k3, 13)) if self.commands
                               else jnp.array([self.target_speed, 0.0, 0.0]))
            info["heading_target"] = jnp.zeros(())
        if self.domain_rand:   # 이 환경의 '실물 오차' (자동 리셋은 첫 리셋 상태로 되돌리므로 환경마다 학습 내내 같은 값)
            km, ks, kc, kd = jax.random.split(jax.random.fold_in(k3, 7), 4)
            r = DOMAIN_RAND
            info["mass_scale"] = jax.random.uniform(km, (), minval=1 - r["mass"], maxval=1 + r["mass"])
            info["motor_scale"] = jax.random.uniform(ks, (), minval=1 - r["motor"], maxval=1 + r["motor"])
            info["com_offset"] = jax.random.uniform(kc, (2,), minval=-r["com"], maxval=r["com"])
            info["delay"] = jax.random.randint(kd, (), 0, r["delay"] + 1)
            info["act_hist"] = jnp.zeros((r["delay"], n))   # 지난 행동들 [1스텝 전, 2스텝 전, ...]
            kj, ki = jax.random.split(jax.random.fold_in(k3, 11))
            info["joint_offset"] = jax.random.uniform(kj, (n,), minval=-r["joint_offset"], maxval=r["joint_offset"])
            tilt = jax.random.uniform(ki, (2,), minval=-r["imu_tilt"], maxval=r["imu_tilt"])
            info["imu_rot"] = tilt_matrix(tilt[0], tilt[1])
        metrics = {f"reward/{k}": jnp.zeros(()) for k in self.reward_weights}
        metrics.update({f"gait/{k}": jnp.zeros(()) for k in GAIT_METRICS})
        metrics.update({"forward_speed": jnp.zeros(())})
        obs = self._observation(data, info)
        if self.spec.obs_history:   # 처음엔 지난 기록 = 지금 관측
            info["obs_hist"] = jnp.tile(obs, (self.spec.obs_history, 1))
            obs = jnp.concatenate([obs, info["obs_hist"].ravel()])
        if self.privileged_critic:
            obs = {"state": obs, "privileged_state": self._observation(data, info, full=True)}
        return State(pipeline_state=data, obs=obs, reward=jnp.zeros(()), done=jnp.zeros(()),
                     metrics=metrics, info=info)

    def step(self, state: State, action: jax.Array) -> State:
        sp = self.spec
        action = jnp.clip(action, -1.0, 1.0)
        applied = action
        if self.domain_rand:   # 제어 지연: 이 환경의 지연(0~2 스텝)만큼 전에 낸 행동이 지금 적용됨
            applied = jnp.concatenate([action[None], state.info["act_hist"]])[state.info["delay"]]
        # 정책 관절은 home + 행동, 나머지(머리 등)는 home. 관절 범위 밖은 자름
        q_des = self.act_home.at[self.policy_act].add(sp.action_scale * applied)
        if self.domain_rand:   # 서보 영점 오차: 서보 기준의 목표 각도가 실제로는 그만큼 어긋난 자리
            q_des = q_des.at[self.policy_act].add(state.info["joint_offset"])
        q_des = jnp.clip(q_des, self.act_low, self.act_high)

        mx = self.mx
        if self.friction_range is not None:   # 접촉 마찰 = 두 geom 중 큰 값 → 충돌하는 geom(발·바닥) 모두 같은 값으로
            mx = mx.replace(geom_friction=mx.geom_friction.at[:, 0].set(state.info["friction"]))
        motor = 1.0
        if self.domain_rand:   # 몸통 질량·관성, 무게중심 위치를 이 환경의 값으로
            b, ms = self.base_body, state.info["mass_scale"]
            mx = mx.replace(body_mass=mx.body_mass.at[b].multiply(ms), body_inertia=mx.body_inertia.at[b].multiply(ms),
                            body_ipos=mx.body_ipos.at[b, 0:2].add(state.info["com_offset"]))
            motor = state.info["motor_scale"]

        def substep(data, _):  # 물리 1스텝마다 PD로 토크 계산 (모터 드라이버 역할)
            q = data.qpos[self.act_qpos_idx]
            dq = data.qvel[self.act_qvel_idx]
            tau = motor * jnp.clip(self.kp * (q_des - q) - self.kd * dq, self.tau_low, self.tau_high)
            data = mjx.step(mx, data.replace(ctrl=tau))
            return data, (tau, dq)

        start = state.pipeline_state
        info = dict(state.info)
        if self.push is not None:   # 외란: 간격마다 몸통 수평 속도에 무작위 방향으로 더함 (freejoint qvel 0:2 = world 속도)
            max_speed, interval = self.push
            rng, k1, k2 = jax.random.split(state.info["rng"], 3)
            info["rng"] = rng
            t = state.info["time"]
            now = jnp.floor((t + CONTROL_DT) / interval) > jnp.floor(t / interval)
            angle = jax.random.uniform(k1, (), minval=-jnp.pi, maxval=jnp.pi)
            kick = jax.random.uniform(k2, (), maxval=max_speed) * jnp.array([jnp.cos(angle), jnp.sin(angle)])
            start = start.replace(qvel=start.qvel.at[0:2].add(jnp.where(now, kick, 0.0)))
        data, (taus, dqs) = jax.lax.scan(substep, start, None, length=self.n_substeps)
        overload, over_frac = self._servo_overload(taus, dqs)
        info["time"] = state.info["time"] + CONTROL_DT

        # --- 상태 계산
        rot = data.xmat[self.base_body]                    # 몸통 → world
        vel_world = data.qvel[0:3]
        vel_body = rot.T @ vel_world
        ang_body = data.qvel[3:6]                          # freejoint 각속도는 몸통 좌표계
        tilt_cos = rot[2, 2]
        tilt = jnp.arccos(jnp.clip(tilt_cos, -1.0, 1.0))
        yaw = jnp.arctan2(rot[1, 0], rot[0, 0])
        roll = jnp.arctan2(rot[2, 1], rot[2, 2])           # 몸통 구르기 (walk_tools의 rpy[0]과 같은 정의)
        height = data.xpos[self.base_body, 2] - self.ground(data.xpos[self.base_body, :2])   # 그 자리 땅 기준
        foot_z = data.xpos[self.foot_bodies, 2] - self.ground(data.xpos[self.foot_bodies, :2])
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
        peak = state.info["swing_peak"]                                 # 착지 직전까지의 최고 높이
        clearance_reward = jnp.sum(real_step * jnp.clip(peak / sp.clearance_target, 0.0, 1.0))
        info["swing_peak"] = jnp.where(contact, 0.0, jnp.maximum(peak, foot_z))
        step_reward = jnp.sum(real_step * jnp.clip(ahead, -0.2, self.step_target) / self.step_target)

        com_shift = self._com_shift(data, yaw, contact)

        # --- 머리 흔들림: 머리의 옆 속도 (몸통 방향 기준). 몸통 속도 + 각속도 × (머리 − 몸통) — 목 관절은 home 유지라 무시.
        #     (위치 차분은 쓰지 않음: Brax 자동 리셋은 info를 되돌리지 않아 리셋 직후 큰 값이 나옴)
        cy, sy = jnp.cos(yaw), jnp.sin(yaw)
        if self.head_body is not None:
            v_head_world = vel_world + jnp.cross(rot @ ang_body, data.xpos[self.head_body] - data.xpos[self.base_body])
            head_lat = -sy * v_head_world[0] + cy * v_head_world[1]
        else:
            head_lat = jnp.zeros(())

        # --- 명령 따라 걷기: 몸통 방향 기준 속도로 명령과 비교 (명령 모드가 아니면 아래 terms의 원래 식 그대로)
        heading_err = yaw
        cmd_terms = {}
        if sp.command_obs:
            c = state.info["command"]
            # v_heading = 몸통이 향한 방향 기준 (앞뒤, 옆) 속도 — 머리 속도(v_head_world)와 다름
            v_heading = jnp.array([cy * vel_world[0] + sy * vel_world[1], -sy * vel_world[0] + cy * vel_world[1]])
            lat_err = v_heading[1] - c[1]
            # 옆 오차는 dead-band만큼 봐줌: 몸통을 수평으로 두고 무게중심을 좌우로 옮기는 정상 걸음을 벌하지 않도록.
            # ⚠ 밴드는 옆 명령 크기만큼 좁힌다: 안 그러면 밴드보다 작은 옆 명령(시험값 0.08 < 0.1)은 가만히 있어도 오차 0이 되어
            #   정책이 옆걸음을 아예 포기한다 (7차 R7에서 옆걸음 10 → 0 / 10으로 확인, docs/10 §12)
            band = jnp.maximum(sp.lateral_deadband - jnp.abs(c[1]), 0.0)
            lat_err = jnp.sign(lat_err) * jnp.maximum(jnp.abs(lat_err) - band, 0.0)
            lin_err2 = jnp.square(v_heading[0] - c[0]) + jnp.square(lat_err)
            yaw_err = ang_body[2] - c[2]
            info["heading_target"] = state.info["heading_target"] + c[2] * CONTROL_DT   # 회전 명령을 쌓은 방향
            d = yaw - state.info["heading_target"]
            heading_err = jnp.arctan2(jnp.sin(d), jnp.cos(d))
            speed = jnp.linalg.norm(c[:2])   # 보폭: 명령한 방향으로 반대 발보다 얼마나 나갔나 (뒤로·옆으로도)
            direction = c[:2] / jnp.maximum(speed, 1e-6)
            d_world = jnp.array([cy * direction[0] - sy * direction[1], sy * direction[0] + cy * direction[1]])
            feet_xy = data.xpos[self.foot_bodies, :2]
            target = jnp.maximum(speed * sp.gait_period / 2, 0.02)
            ahead_cmd = (feet_xy - feet_xy[::-1]) @ d_world
            cmd_terms = {
                "forward_velocity": jnp.exp(-lin_err2 / sp.velocity_sigma2),
                "lateral_velocity": jnp.square(lat_err),
                "yaw_rate": jnp.square(yaw_err),
                "yaw_tracking": jnp.exp(-jnp.square(yaw_err) / 0.1),
                "step_length": (speed > 0.03) * jnp.sum(real_step * jnp.clip(ahead_cmd, -0.2, target) / target),
                # 돌 때·옆걸음 때는 좌우가 다른 게 당연 → 그만큼 대칭 요구를 낮춤. 옆 게이트가 없으면 대칭 가중치를 올릴 때
                # 정책이 옆걸음을 통째로 포기한다 (7차 R9에서 symmetry −24로 옆걸음 9·8 → 0·0 확인, docs/10 §12)
                "symmetry": asym * jnp.maximum(0.0, 1.0 - jnp.abs(c[2]) / 0.3) * jnp.maximum(0.0, 1.0 - jnp.abs(c[1]) / 0.08),
                "head_sway": jnp.square(head_lat - c[1]) * head_sway_gate(sp, c),   # 옆걸음 명령만큼은 움직여도 됨.
                                                                                       # 게이트: 뒤로·돌기 명령은 빼 줌 (6차)
                "gait_phase": self._gait_phase_match(info["time"], contact, sp.gait_period,
                                                     double_support_width(sp, c[0])),
            }
            track_lin, track_yaw = jnp.sqrt(lin_err2), jnp.abs(yaw_err)
        else:
            track_lin, track_yaw = jnp.abs(vel_world[0] - self.target_speed), jnp.abs(ang_body[2])

        terms = {
            "forward_velocity": jnp.exp(-jnp.square(vel_world[0] - self.target_speed) / sp.velocity_sigma2),
            "lateral_velocity": jnp.square(vel_world[1]),
            "yaw_rate": jnp.square(ang_body[2]),
            "upright": jnp.square(tilt),
            "height": jnp.square(height - sp.nominal_height),
            "feet_air_time": air_reward,
            "action_rate": jnp.sum(jnp.square(action - state.info["last_action"])),
            "torque": jnp.sum(jnp.square(taus[-1])),
            "servo_overload": overload,
            "alive": jnp.ones(()),
            "heading": jnp.square(yaw),
            "flight": jnp.where(contact.any(), 0.0, 1.0),
            "gait_phase": self._gait_phase_match(info["time"], contact, sp.gait_period,
                                                 double_support_width(sp, self.target_speed)),
            "symmetry": asym,
            "step_length": step_reward,
            "reference": self._reference_match(q, info["time"]),
            "com_over_stance": jnp.clip(com_shift / COM_SHIFT_TARGET, 0.0, 1.0),
            "foot_clearance": clearance_reward,
            "yaw_tracking": jnp.zeros(()),
            "head_sway": jnp.square(head_lat),
            # 꼭 필요한 만큼(roll_free)을 넘는 몸통 구르기만 벌점 → 안 구르는 명령(정지·제자리 돌기)에는 0
            "roll_excess": jnp.square(jnp.maximum(jnp.abs(roll) - sp.roll_free, 0.0)),
        }
        terms.update(cmd_terms)
        # 방향: 명령 모드에서는 1 − cos(오차) (최대 2). 제곱이면 돌기 명령을 못 따라갈 때 오차가 180°까지 쌓여 다른 보상을
        # 모두 덮음 (실측: 에피소드당 −36점, 4차 C2 첫 시도가 이것 때문에 배우지 못함)
        terms["heading"] = (1.0 - jnp.cos(heading_err)) if sp.command_obs else jnp.square(heading_err)
        weighted = {k: self.reward_weights[k] * v for k, v in terms.items()}
        reward = sum(weighted.values()) * CONTROL_DT          # 스텝 길이로 나눠 에피소드 합이 '초당' 의미가 되게
        done = jnp.where((tilt > FALL_TILT) | (height < sp.fall_height), 1.0, 0.0)

        info["last_action"] = action
        if self.commands is not None:   # resample초마다 새 명령 (정책은 다음 스텝부터 새 명령을 봄)
            rng, key = jax.random.split(info["rng"])
            info["rng"] = rng
            t, period = state.info["time"], self.commands["resample"]
            new = jnp.floor((t + CONTROL_DT) / period) > jnp.floor(t / period)
            info["command"] = jnp.where(new, self._sample_command(key), state.info["command"])
            if sp.command_obs:   # 새 명령이면 '가야 할 방향'도 지금 방향에서 다시 시작
                info["heading_target"] = jnp.where(new, yaw, info["heading_target"])
        if self.domain_rand:
            info["act_hist"] = jnp.concatenate([action[None], state.info["act_hist"][:-1]])
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
            "gait/com_shift": com_shift,                             # 한 발 지지 때 무게중심 이동 [m] (그 외 0)
            "gait/landing": jnp.sum(real_step).astype(jnp.float32),  # 이번 스텝에 걸음으로 착지한 발 수
            "gait/clearance": jnp.sum(real_step * peak),             # 그 착지들의 흔듦 최고 높이 합 [m]
            "gait/servo_over": over_frac,                            # 정책 관절이 데이터시트 한계 밖인 비율 (물리 스텝 평균)
            "gait/track_lin": track_lin,                             # 속도 명령과의 차이 [m/s]
            "gait/track_yaw": track_yaw,                             # 회전 명령과의 차이 [rad/s]
            "gait/head_sway": jnp.square(head_lat),                  # 머리 옆 속도² [m²/s²] (명령 빼지 않음)
            "gait/roll_abs": jnp.abs(roll),                          # 몸통 구르기 각도 [rad]
        })
        metrics["forward_speed"] = vel_world[0]
        obs = self._observation(data, info, vel_body=vel_body, ang_body=ang_body)
        if self.noise_scale is not None:   # 센서 잡음 (학습 때만, 실물 센서처럼)
            rng, key = jax.random.split(info["rng"])
            info["rng"] = rng
            obs = obs + self.noise_scale * jax.random.normal(key, obs.shape)
        if self.spec.obs_history:   # 지금 관측 + 지난 기록 (기록은 잡음까지 그대로 = 실물 센서에서 쌓은 것처럼)
            hist = state.info["obs_hist"]
            info["obs_hist"] = jnp.concatenate([obs[None], hist[:-1]])
            obs = jnp.concatenate([obs, hist.ravel()])
        if self.privileged_critic:   # critic용: 가리지 않고 잡음도 없는 전체 관측
            obs = {"state": obs, "privileged_state": self._observation(data, info, vel_body=vel_body,
                                                                      ang_body=ang_body, full=True)}
        return state.replace(pipeline_state=data, obs=obs, reward=reward, done=done,
                             metrics=metrics, info=info)

    def _sample_command(self, key):
        """명령 범위에서 (vx, vy, wz)를 고르게 뽑음. 성분마다 zero_prob 확률로 0 (범위가 0을 포함하면)."""
        lo = jnp.array([self.commands[k][0] for k in ("vx", "vy", "wz")])
        hi = jnp.array([self.commands[k][1] for k in ("vx", "vy", "wz")])
        k1, k2 = jax.random.split(key)
        cmd = jax.random.uniform(k1, (3,), minval=lo, maxval=hi)
        zero = (jax.random.uniform(k2, (3,)) < self.commands["zero_prob"]) & (lo <= 0) & (hi >= 0)
        return jnp.where(zero, 0.0, cmd)

    def _servo_overload(self, taus, dqs):
        """제어 스텝 동안(물리 스텝 n개) 모터 토크가 서보 데이터시트 한계를 넘은 정도.
        taus, dqs: [n, 구동 관절]. 돌려줌: (Σ넘은 만큼² 의 물리 스텝 평균 [N·m²], 정책 관절이 넘은 비율)"""
        net = taus - self.act_damping * dqs                                   # 역기전력 damping을 뺀 모터 토크
        k = SERVO_DATASHEET["stall"] / SERVO_DATASHEET["noload"]
        avail = jnp.clip(SERVO_DATASHEET["stall"] - k * jnp.sign(net) * dqs, 0.0, None)
        excess = jax.nn.relu(jnp.abs(net) - avail)
        return jnp.mean(jnp.sum(jnp.square(excess), axis=-1)), jnp.mean(excess[:, self.policy_act] > 0)

    def _com_shift(self, data, yaw, contact):
        """한 발로 설 때 무게중심이 home 자세보다 디딤 쪽 hip_roll 축으로 얼마나 옮겨 갔나 [m] (몸통 방향 기준 옆으로).
        두 발이 다 땅에 있거나 공중이면 0. hip_roll_joints가 없는 로봇은 늘 0."""
        if self.hip_roll_joints is None:
            return jnp.zeros(())
        lateral = jnp.array([-jnp.sin(yaw), jnp.cos(yaw)])               # 몸통 왼쪽 방향 (수평)
        com = data.subtree_com[self.base_body, :2]
        hips = data.xanchor[self.hip_roll_joints, :2]                     # (왼, 오른)
        inboard = jnp.array([(hips[0] - com) @ lateral, (com - hips[1]) @ lateral])   # 고관절 축이 무게중심보다 바깥으로 떨어진 거리
        shift = self.com_hip_nominal - inboard                            # 양수 = 디딤 쪽으로 옮겨 감
        left_only, right_only = contact[0] & ~contact[1], contact[1] & ~contact[0]
        return jnp.where(left_only, shift[0], jnp.where(right_only, shift[1], 0.0))

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
    def _gait_phase_match(t, contact, period, ds=0.1):
        """걸음 박자의 앞 절반은 왼발만, 뒤 절반은 오른발만 땅을 딛는 것이 '정답'.
        박자가 바뀌는 앞뒤 ds 주기(양발 지지 구간, 기본 0.1)는 두 발 모두 딛는 것이 정답. 발마다 맞으면 0.5점."""
        phase = jnp.mod(t / period, 1.0)
        double = (phase < ds) | ((phase > 0.5 - ds) & (phase < 0.5 + ds)) | (phase > 1.0 - ds)
        want_left = (phase < 0.5) | double
        want_right = (phase >= 0.5) | double
        match_left = jnp.where(want_left, contact[0], ~contact[0])
        match_right = jnp.where(want_right, contact[1], ~contact[1])
        return 0.5 * (match_left.astype(jnp.float32) + match_right.astype(jnp.float32))

    # ------------------------------------------------------------------ 관측
    def _observation(self, data, info, vel_body=None, ang_body=None, full: bool = False) -> jax.Array:
        """full=True면 deployable_obs여도 가리지 않음 (비대칭 critic의 privileged_state)."""
        rot = data.xmat[self.base_body]
        if vel_body is None:
            vel_body = rot.T @ data.qvel[0:3]
            ang_body = data.qvel[3:6]
        gravity_body = rot.T @ jnp.array([0.0, 0.0, -1.0])
        q_rel = data.qpos[self.qpos_idx] - self.q_home
        if self.domain_rand and not full:   # 조립 오차: IMU가 기울게 붙음, 서보는 영점이 어긋난 각도를 읽음
            gravity_body, ang_body = info["imu_rot"] @ gravity_body, info["imu_rot"] @ ang_body
            q_rel = q_rel - info["joint_offset"]
        phase = 2.0 * jnp.pi * info["time"] / self.spec.gait_period
        height = data.xpos[self.base_body, 2] - self.ground(data.xpos[self.base_body, :2]) - self.spec.nominal_height
        if self.spec.deployable_obs and not full:   # 실물에 없는 센서 값은 0 (MujocoRunner.obs와 같게)
            vel_body, height = jnp.zeros(3), jnp.zeros(())
        return jnp.concatenate([
            gravity_body,
            ang_body,
            vel_body,
            jnp.array([height]),
            q_rel,
            0.1 * data.qvel[self.qvel_idx],
            info["last_action"],
            info["command"] if self.spec.command_obs else jnp.array([self.target_speed]),
            jnp.array([jnp.sin(phase), jnp.cos(phase)]),
        ])
