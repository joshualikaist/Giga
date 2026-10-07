"""MuJoCo 배열 인덱스를 '관절 이름'으로 다루게 해 주는 로봇 인터페이스.

왜 필요한가?
    MuJoCo의 상태는 큰 배열 하나에 들어 있습니다.
        data.qpos : 위치   (길이 nq)  — freejoint가 있으면 앞 7칸이 [x y z qw qx qy qz]
        data.qvel : 속도   (길이 nv)  — freejoint가 있으면 앞 6칸이 [vx vy vz wx wy wz]
        data.ctrl : 제어입력 (길이 nu) — 액추에이터 순서
    nq ≠ nv 이기 때문에 "관절 i의 위치는 qpos[i]"라고 가정하면 바로 버그가 납니다.
    이 클래스는 모델에서 인덱스를 한 번 계산해 두고, 항상 같은 '관절 순서'로 값을 돌려줍니다.

    이 관절 순서/이름은 나중에 ROS2의 sensor_msgs/JointState(name[], position[], velocity[], effort[])에
    그대로 대응됩니다 (as_joint_state 참고).
"""
from __future__ import annotations

from typing import Mapping, Sequence

import mujoco
import numpy as np

from .builder import FREEJOINT_NAME, IMU_SITE_NAME
from .utils import lowest_collision_z, quat_to_rpy


class RobotInterface:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data

        if model.nu == 0:
            raise ValueError("액추에이터가 없는 모델입니다. builder.build_robot_model()로 만든 모델을 쓰세요.")

        # 액추에이터 순서 = 관절 순서 로 고정 (builder가 URDF 관절 순서대로 모터를 추가함)
        joint_ids = []
        for a in range(model.nu):
            if model.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT:
                raise ValueError(f"관절 구동이 아닌 액추에이터: {model.actuator(a).name}")
            joint_ids.append(int(model.actuator_trnid[a, 0]))
        self.joint_ids = np.array(joint_ids)
        self.joint_names = [model.joint(j).name for j in joint_ids]
        self.qpos_idx = model.jnt_qposadr[self.joint_ids]   # 각 관절의 qpos 위치
        self.qvel_idx = model.jnt_dofadr[self.joint_ids]    # 각 관절의 qvel 위치 (≠ qpos_idx!)
        self.torque_limits = model.actuator_ctrlrange.copy()

        # 루트 바디(world의 자식) = 로봇 몸통
        roots = [b for b in range(1, model.nbody) if model.body_parentid[b] == 0]
        self.base_body_id = roots[0]
        self.has_floating_base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, FREEJOINT_NAME) >= 0
        self.total_mass = float(model.body_subtreemass[self.base_body_id])

    # ------------------------------------------------------------------ 관절
    @property
    def num_joints(self) -> int:
        return len(self.joint_names)

    def to_joint_vector(self, values: Mapping[str, float] | Sequence[float]) -> np.ndarray:
        """{이름: 값} 딕셔너리 또는 순서 있는 배열 → 관절 순서 배열."""
        if isinstance(values, Mapping):
            missing = set(self.joint_names) - set(values)
            if missing:
                raise KeyError(f"값이 없는 관절: {sorted(missing)}")
            return np.array([values[n] for n in self.joint_names], dtype=float)
        vec = np.asarray(values, dtype=float)
        if vec.shape != (self.num_joints,):
            raise ValueError(f"길이 {self.num_joints} 배열이 필요합니다. 받은 shape: {vec.shape}")
        return vec

    def joint_positions(self) -> np.ndarray:
        return self.data.qpos[self.qpos_idx].copy()

    def joint_velocities(self) -> np.ndarray:
        return self.data.qvel[self.qvel_idx].copy()

    def joint_torques(self) -> np.ndarray:
        """직전 스텝에서 모터가 실제로 낸 토크 (ctrl이 한계로 잘린 뒤의 값)."""
        return self.data.actuator_force.copy()

    def set_joint_torques(self, tau: np.ndarray) -> None:
        """모터 토크 명령. 다음 mj_step()에서 적용된다. (한계 초과분은 MuJoCo가 ctrlrange로 자름)"""
        self.data.ctrl[:] = tau

    # ------------------------------------------------------------------ 몸통(베이스)
    def base_position(self) -> np.ndarray:
        """몸통 원점의 world 좌표 [m]."""
        return self.data.xpos[self.base_body_id].copy()

    def base_quaternion(self) -> np.ndarray:
        """몸통 자세 쿼터니언 (w, x, y, z) — MuJoCo 순서. ROS로 보낼 땐 utils.quat_wxyz_to_xyzw."""
        return self.data.xquat[self.base_body_id].copy()

    def base_rpy(self) -> np.ndarray:
        return quat_to_rpy(self.base_quaternion())

    def base_tilt(self) -> float:
        """몸통 z축이 수직(world z)에서 기운 각도 [rad]. 0 = 똑바로, π/2 = 옆으로 누움.

        넘어짐 판정에는 roll/pitch보다 이 값이 안전합니다. 크게 기울면 오일러각은
        roll=±180° 같은 다른 표현으로 뒤집혀 보일 수 있기 때문입니다.
        """
        rot = self.data.xmat[self.base_body_id].reshape(3, 3)
        return float(np.arccos(np.clip(rot[2, 2], -1.0, 1.0)))

    def base_linear_velocity(self) -> np.ndarray:
        """몸통 선속도 [m/s], world 좌표계. (freejoint qvel[0:3]은 world 기준)"""
        if not self.has_floating_base:
            return np.zeros(3)
        adr = self.model.joint(FREEJOINT_NAME).dofadr[0]
        return self.data.qvel[adr:adr + 3].copy()

    # ------------------------------------------------------------------ 센서
    def imu(self) -> dict[str, np.ndarray]:
        """IMU 센서값. quat=(w,x,y,z), gyro=[rad/s], acc=[m/s²] (정지 상태에서 +9.81 z: 중력 반작용 포함)."""
        acc = self.data.sensor("imu_acc").data.copy()
        if not self.has_floating_base:
            # MuJoCo 3.15는 world에 용접된(움직이지 않는) 바디의 가속도계 값을 0으로 낸다 (실측).
            # 실제 IMU는 받침대에 고정돼 있어도 중력 반작용 −g를 측정하므로, 그 값을 IMU 좌표계로 넣는다.
            rot = self.data.site_xmat[self.model.site(IMU_SITE_NAME).id].reshape(3, 3)
            acc = rot.T @ (-self.model.opt.gravity)
        return {
            "quat": self.data.sensor("imu_quat").data.copy(),
            "gyro": self.data.sensor("imu_gyro").data.copy(),
            "acc": acc,
        }

    def contact_normal_force(self, body_name: str) -> float:
        """해당 바디가 다른 물체로부터 받는 접촉 수직력의 합 [N] (예: 발바닥 하중)."""
        body_id = self.model.body(body_name).id
        total = 0.0
        wrench = np.zeros(6)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if body_id in (self.model.geom_bodyid[c.geom1], self.model.geom_bodyid[c.geom2]):
                mujoco.mj_contactForce(self.model, self.data, i, wrench)
                total += wrench[0]  # 접촉 좌표계의 첫 성분 = 법선(수직) 방향 힘
        return total

    # ------------------------------------------------------------------ 상태 초기화
    def reset(self, joint_pos: Mapping[str, float] | Sequence[float] | None = None,
              base_pos: Sequence[float] | None = None,
              on_ground: bool = False, clearance: float = 0.001) -> None:
        """시뮬레이션 상태를 초기화하고 원하는 자세로 둔다.

        on_ground=True면 가장 낮은 충돌점이 z=clearance가 되도록 몸통 높이를 맞춘다 (floating base 전용).
        """
        mujoco.mj_resetData(self.model, self.data)
        if joint_pos is not None:
            self.data.qpos[self.qpos_idx] = self.to_joint_vector(joint_pos)
        if base_pos is not None or on_ground:
            if not self.has_floating_base:
                raise ValueError("fixed_base 모델은 몸통 위치를 바꿀 수 없습니다 (SimConfig.base_pos로 지정).")
            adr = self.model.joint(FREEJOINT_NAME).qposadr[0]
            if base_pos is not None:
                self.data.qpos[adr:adr + 3] = base_pos
            if on_ground:
                mujoco.mj_forward(self.model, self.data)
                self.data.qpos[adr + 2] -= lowest_collision_z(self.model, self.data) - clearance
        mujoco.mj_forward(self.model, self.data)  # 위치가 바뀌었으니 파생량(xpos, 센서 등) 재계산

    # ------------------------------------------------------------------ ROS2 미리보기
    def as_joint_state(self) -> dict[str, list]:
        """sensor_msgs/JointState와 같은 모양의 딕셔너리. (Phase 5에서 그대로 메시지에 담으면 됨)"""
        return {
            "name": list(self.joint_names),
            "position": self.joint_positions().tolist(),
            "velocity": self.joint_velocities().tolist(),
            "effort": self.joint_torques().tolist(),
        }
