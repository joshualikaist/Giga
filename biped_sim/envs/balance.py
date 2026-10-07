"""BipedBalanceEnv — "밀려도 넘어지지 않기"를 배우는 가장 단순한 강화학습 환경.

과제
    로봇이 home 자세로 서 있다가, 에피소드 중간에 몸통을 앞 또는 뒤로 0.1초 동안 세게 밀린다.
    4초 동안 넘어지지 않고 버티면 성공.

행동 (action, 6개, 각 −1 ~ 1)
    관절 목표 각도를 home 자세에서 얼마나 바꿀지. 목표 = home + 0.3 rad × action
    → 실제 토크는 기존 관절 PD 제어기(robot_configs의 게인)가 계산한다 (실제 로봇의 모터 드라이버와 같은 구조).
    → action = 0 이면 t06의 "PD로 home 자세 유지"와 똑같다. 학습은 이 기본 동작을 '고쳐 나가는' 것.

관측 (observation, 28개)
    중력 방향 (몸통 기준, 3)   ← IMU 자세에서 계산: 얼마나 기울었나
    몸통 각속도 (몸통 기준, 3) ← IMU 자이로
    몸통 선속도 (몸통 기준, 3) ← 시뮬레이션 정답값 (실제 로봇에선 추정해야 함)
    몸통 높이 − 기준 높이 (1)
    관절 각도 − home (6), 관절 속도 × 0.1 (6)
    직전 행동 (6)

보상 (reward, 매 제어 스텝)
    +1           살아 있음 (넘어지지 않음)
    −2 × 기울기²  몸통이 기울수록 감점 [rad²]
    −0.05 × |Δaction|²  행동이 급격히 바뀌면 감점 (떨림 방지)
넘어짐 판정 (에피소드 종료): 몸통 기울기 > 0.6 rad(약 34°) 또는 몸통 높이 < 0.40 m
"""
from __future__ import annotations

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from ..builder import SimConfig, build_robot_model
from ..controllers import JointPDController
from ..robot import RobotInterface
from ..robot_configs import SIMPLE_BIPED

ACTION_SCALE = 0.3          # [rad] action 1.0 = home에서 0.3 rad
CONTROL_HZ = 50             # 정책이 행동을 고르는 주기 (물리는 500 Hz → 10 스텝마다)
EPISODE_SECONDS = 4.0
NOMINAL_BASE_HEIGHT = 0.588  # [m] PD로 서 있을 때의 몸통 높이 (t06 실측)
FALL_TILT = 0.6             # [rad]
FALL_HEIGHT = 0.40          # [m]
PUSH_DURATION = 0.1         # [s]


class BipedBalanceEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": CONTROL_HZ}

    def __init__(self, push_force=(20.0, 60.0), push_time=(0.5, 1.5), push_directions=(1.0, -1.0),
                 render_mode: str | None = None):
        """
        push_force      : 미는 힘 [N]의 범위 (매 에피소드 균등 분포에서 뽑음). (F, F)로 주면 고정
        push_time       : 미는 시각 [s]의 범위
        push_directions : 미는 방향 후보 (+1 = 앞(+x), −1 = 뒤). (1.0,)이면 앞으로만
        """
        super().__init__()
        self.push_force = push_force
        self.push_time = push_time
        self.push_directions = np.asarray(push_directions, dtype=float)
        self.render_mode = render_mode

        cfg = SIMPLE_BIPED
        self.model, _ = build_robot_model(cfg.urdf, SimConfig(fixed_base=False, home_joint_pos=cfg.home_pose))
        self.data = mujoco.MjData(self.model)
        self.robot = RobotInterface(self.model, self.data)
        self.home_key = self.model.key("home").id
        self.q_home = self.robot.to_joint_vector(cfg.home_pose)
        self.q_low, self.q_high = self.model.jnt_range[self.robot.joint_ids].T
        self.pd = JointPDController(self.robot.to_joint_vector(cfg.kp), self.robot.to_joint_vector(cfg.kd),
                                    self.robot.torque_limits)
        self.n_substeps = round(1.0 / (CONTROL_HZ * self.model.opt.timestep))  # 500 Hz / 50 Hz = 10
        self.max_steps = round(EPISODE_SECONDS * CONTROL_HZ)

        n = self.robot.num_joints
        self.action_space = spaces.Box(-1.0, 1.0, shape=(n,), dtype=np.float32)
        obs_dim = 3 + 3 + 3 + 1 + n + n + n
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self._renderer = None

    # ------------------------------------------------------------------ Gymnasium API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_key)
        # 매번 아주 조금씩 다른 자세에서 시작 (항상 같은 상황만 외우지 않도록)
        self.data.qpos[self.robot.qpos_idx] += self.np_random.uniform(-0.02, 0.02, self.robot.num_joints)
        mujoco.mj_forward(self.model, self.data)

        options = options or {}
        lo, hi = self.push_force
        self.push_newtons = float(options.get("push_force", self.np_random.uniform(lo, hi)))
        self.push_dir = float(options.get("push_direction", self.np_random.choice(self.push_directions)))
        self.push_start = float(options.get("push_time", self.np_random.uniform(*self.push_time)))
        self.prev_action = np.zeros(self.robot.num_joints)
        self.steps = 0
        return self._observation(), self._info(fell=False)

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        q_des = np.clip(self.q_home + ACTION_SCALE * action, self.q_low, self.q_high)

        for _ in range(self.n_substeps):  # 물리 500 Hz, 그동안 같은 목표로 PD
            t = self.data.time
            pushing = self.push_start <= t < self.push_start + PUSH_DURATION
            self.data.xfrc_applied[self.robot.base_body_id, 0] = self.push_dir * self.push_newtons if pushing else 0.0
            tau = self.pd.compute(self.robot.joint_positions(), self.robot.joint_velocities(), q_des)
            self.robot.set_joint_torques(tau)
            mujoco.mj_step(self.model, self.data)
        self.steps += 1

        tilt = self.robot.base_tilt()
        height = self.robot.base_position()[2]
        fell = tilt > FALL_TILT or height < FALL_HEIGHT
        reward = 1.0 - 2.0 * tilt**2 - 0.05 * float(np.sum((action - self.prev_action) ** 2))
        self.prev_action = action
        truncated = self.steps >= self.max_steps
        return self._observation(), reward, fell, truncated, self._info(fell)

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=360, width=480)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = self.robot.base_body_id
        cam.distance, cam.azimuth, cam.elevation = 2.0, 90.0, -10.0
        self._renderer.update_scene(self.data, camera=cam)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------------ 내부
    def _observation(self) -> np.ndarray:
        rot = self.data.xmat[self.robot.base_body_id].reshape(3, 3)   # 몸통 → world 회전
        gravity_body = rot.T @ np.array([0.0, 0.0, -1.0])              # 몸통에서 본 '아래' 방향
        lin_vel_body = rot.T @ self.robot.base_linear_velocity()
        imu = self.robot.imu()
        obs = np.concatenate([
            gravity_body,
            imu["gyro"],
            lin_vel_body,
            [self.robot.base_position()[2] - NOMINAL_BASE_HEIGHT],
            self.robot.joint_positions() - self.q_home,
            0.1 * self.robot.joint_velocities(),
            self.prev_action,
        ])
        return obs.astype(np.float32)

    def _info(self, fell: bool) -> dict:
        return {"fell": fell, "push_force": self.push_newtons, "push_direction": self.push_dir,
                "tilt": self.robot.base_tilt(), "time": self.data.time}


# ---------------------------------------------------------------------- 평가
EVAL_FORCES = (20, 30, 40, 50, 60)      # [N]
EVAL_DIRECTIONS = (1.0, -1.0)           # 앞, 뒤
EVAL_PUSH_TIMES = (0.5, 1.0, 1.5)       # [s]


def zero_policy(obs: np.ndarray) -> np.ndarray:
    """학습 전 기준선: 항상 0 → 관절 PD로 home 자세만 유지 (t06과 같음)."""
    return np.zeros(6, dtype=np.float32)


def evaluate_push_recovery(policy, forces=EVAL_FORCES, directions=EVAL_DIRECTIONS,
                           push_times=EVAL_PUSH_TIMES) -> dict[int, tuple[int, int]]:
    """세기별로 (버틴 횟수, 시도 횟수)를 돌려준다. policy: 관측 → 행동 함수.

    무작위성 없이 항상 같은 조건(앞/뒤 × 미는 시각 3가지, 같은 시드)으로 시험하므로
    학습 전후·모델끼리 공정하게 비교할 수 있다.
    """
    env = BipedBalanceEnv()
    results = {}
    for force in forces:
        survived = trials = 0
        for direction in directions:
            for push_time in push_times:
                obs, _ = env.reset(seed=1, options={"push_force": force, "push_direction": direction,
                                                    "push_time": push_time})
                while True:
                    obs, _, terminated, truncated, info = env.step(policy(obs))
                    if terminated or truncated:
                        break
                survived += not info["fell"]
                trials += 1
        results[force] = (survived, trials)
    env.close()
    return results


def format_results(baseline: dict, trained: dict | None = None) -> str:
    """평가 결과를 표 문자열로."""
    lines = ["밀기 세기   PD만(학습 전)" + ("   학습한 정책" if trained else "")]
    for force, (ok, n) in baseline.items():
        row = f"  {force:3d} N     {ok}/{n} 버팀"
        if trained:
            ok2, n2 = trained[force]
            row += f"        {ok2}/{n2} 버팀"
        lines.append(row)
    return "\n".join(lines)
