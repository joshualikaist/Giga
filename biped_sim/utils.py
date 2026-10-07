"""작은 수학/기하 유틸리티.

⚠ 쿼터니언 순서 주의 (ROS2 연동 시 가장 흔한 버그)
    MuJoCo : (w, x, y, z)   ← data.qpos, data.xquat, framequat 센서 모두 이 순서
    ROS2   : (x, y, z, w)   ← geometry_msgs/Quaternion, tf2
"""
import itertools

import mujoco
import numpy as np

# 박스 꼭짓점 8개의 부호 조합: (±1, ±1, ±1)
_BOX_CORNER_SIGNS = np.array(list(itertools.product((-1.0, 1.0), repeat=3)))


def quat_wxyz_to_xyzw(q):
    """MuJoCo 순서(w,x,y,z) → ROS 순서(x,y,z,w)."""
    return np.array([q[1], q[2], q[3], q[0]])


def quat_xyzw_to_wxyz(q):
    """ROS 순서(x,y,z,w) → MuJoCo 순서(w,x,y,z)."""
    return np.array([q[3], q[0], q[1], q[2]])


def quat_to_rpy(q_wxyz):
    """MuJoCo 쿼터니언(w,x,y,z) → (roll, pitch, yaw) [rad].

    ROS의 tf2 / URDF rpy와 같은 정의: 고정축 X→Y→Z 회전 (= 이동축 Z-Y-X).
    """
    w, x, y, z = q_wxyz
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([roll, pitch, yaw])


def lowest_collision_z(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """로봇의 충돌 geom 중 가장 낮은 점의 world z 좌표 [m].

    로봇을 바닥 "바로 위"에 올려놓을 때 사용합니다 (mj_forward 이후 호출해야 함).
    - world에 붙은 geom(바닥 등)과 시각용 geom(contype=conaffinity=0)은 제외
    - box/sphere/capsule/cylinder는 정확히 계산, 그 외(mesh 등)는 경계구(rbound)로 보수적 근사
    """
    z_min = np.inf
    for g in range(model.ngeom):
        if model.geom_bodyid[g] == 0:
            continue
        if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
            continue
        gtype = model.geom_type[g]
        size = model.geom_size[g]
        pos = data.geom_xpos[g]
        rot = data.geom_xmat[g].reshape(3, 3)
        axis_z = rot[2, 2]  # geom 로컬 z축의 world z 성분

        if gtype == mujoco.mjtGeom.mjGEOM_BOX:
            # MJCF box size = 절반 길이 → 꼭짓점 = 중심 ± R·size
            corners = pos + (_BOX_CORNER_SIGNS * size) @ rot.T
            z = corners[:, 2].min()
        elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
            z = pos[2] - size[0]
        elif gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
            z = pos[2] - abs(axis_z) * size[1] - size[0]
        elif gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
            z = pos[2] - abs(axis_z) * size[1] - size[0] * np.sqrt(max(0.0, 1.0 - axis_z**2))
        else:
            z = pos[2] - model.geom_rbound[g]
        z_min = min(z_min, z)
    return float(z_min)
