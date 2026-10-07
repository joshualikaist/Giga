"""시뮬레이터 회귀 테스트.

실행:  source scripts/activate.sh && pytest -v

이 테스트들은 "지금 확인된 사실"을 고정해 둡니다. MuJoCo 버전을 올리거나 URDF/빌더를 고친 뒤
테스트가 깨지면, 무엇이 달라졌는지 바로 알 수 있습니다. (docs/03_urdf_and_mujoco.md의 표와 대응)
"""
import mujoco
import numpy as np
import pytest

from biped_sim import (SIMPLE_BIPED, JointPDController, RobotInterface, SimConfig, build_robot_model,
                       paths, simulate)
from biped_sim.utils import lowest_collision_z, quat_to_rpy, quat_wxyz_to_xyzw, quat_xyzw_to_wxyz

EXPECTED_JOINTS = ["left_hip_pitch", "left_knee", "left_ankle_pitch",
                   "right_hip_pitch", "right_knee", "right_ankle_pitch"]
TOTAL_MASS = 8.2


# ----------------------------------------------------------------------------- fixtures
@pytest.fixture
def floating():
    model, _ = build_robot_model(SIMPLE_BIPED.urdf,
                                 SimConfig(fixed_base=False, home_joint_pos=SIMPLE_BIPED.home_pose))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    return model, data, RobotInterface(model, data)


def make_pd(robot):
    return JointPDController(robot.to_joint_vector(SIMPLE_BIPED.kp),
                             robot.to_joint_vector(SIMPLE_BIPED.kd), robot.torque_limits)


# ----------------------------------------------------------------------------- 기본 동작
def test_falling_box_settles_at_rest_height():
    model = mujoco.MjModel.from_xml_path(str(paths.FALLING_BOX_XML))
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data, nstep=1500)  # 3초
    assert data.body("box").xpos[2] == pytest.approx(0.1, abs=2e-3)    # 반 변 길이
    assert data.body("ball").xpos[2] == pytest.approx(0.08, abs=2e-3)  # 반지름


def test_quaternion_helpers():
    q_wxyz = np.array([0.9513, 0.1677, 0.2549, -0.0449])
    np.testing.assert_allclose(quat_xyzw_to_wxyz(quat_wxyz_to_xyzw(q_wxyz)), q_wxyz)
    # URDF rpy=(20°, 30°, 0°)를 MuJoCo가 변환한 쿼터니언 → 다시 rpy
    np.testing.assert_allclose(np.degrees(quat_to_rpy(q_wxyz / np.linalg.norm(q_wxyz))),
                               [20, 30, 0], atol=0.05)


# ----------------------------------------------------------------------------- MuJoCo의 URDF 변환 동작
def test_raw_urdf_import_behavior():
    """<mujoco> 태그 없는 URDF를 그대로 읽으면: 루트 병합, visual 폐기, 액추에이터 없음."""
    model = mujoco.MjModel.from_xml_path(str(paths.PENDULUM_URDF))
    assert [model.body(i).name for i in range(model.nbody)] == ["world", "rod_link"]
    assert model.ngeom == 2  # collision만
    assert model.nu == 0
    assert model.jnt_actfrcrange[0].tolist() == [-5.0, 5.0]  # <limit effort> → actuatorfrcrange


def test_simple_biped_urdf_parsed_as_designed():
    model = mujoco.MjModel.from_xml_path(str(SIMPLE_BIPED.urdf))
    assert [model.joint(j).name for j in range(model.njnt)] == EXPECTED_JOINTS
    assert model.body_subtreemass[0] == pytest.approx(TOTAL_MASS)
    assert model.body("base_link").id == 1  # <mujoco fusestatic="false"> 덕분에 보존
    n_visual = int(np.sum((model.geom_contype == 0) & (model.geom_conaffinity == 0)))
    assert n_visual == 7  # discardvisual="false" 덕분에 visual 보존
    assert model.dof_damping.tolist() == [0.1] * 6
    assert model.dof_frictionloss.tolist() == [0.02] * 6


# ----------------------------------------------------------------------------- 빌더
def test_builder_floating_model(floating):
    model, data, robot = floating
    assert (model.nq, model.nv, model.nu, model.nsensor) == (13, 12, 6, 3)
    assert robot.joint_names == EXPECTED_JOINTS
    assert robot.has_floating_base
    assert robot.total_mass == pytest.approx(TOTAL_MASS)
    np.testing.assert_array_equal(robot.qpos_idx, np.arange(7, 13))
    np.testing.assert_array_equal(robot.qvel_idx, np.arange(6, 12))
    np.testing.assert_allclose(robot.torque_limits[:, 1], [20, 20, 10, 20, 20, 10])
    assert model.nexclude == 6  # 부모-자식 링크 쌍
    assert np.all(model.dof_armature[6:] == pytest.approx(0.01))


def test_home_keyframe_places_feet_on_ground(floating):
    model, data, robot = floating
    assert lowest_collision_z(model, data) == pytest.approx(0.001, abs=1e-6)
    np.testing.assert_allclose(robot.joint_positions(), robot.to_joint_vector(SIMPLE_BIPED.home_pose))


def test_fixed_base_has_no_spurious_self_contacts():
    """몸통이 world에 고정되면 MuJoCo의 부모-자식 필터가 안 먹힘 → 빌더의 exclude가 막아야 함."""
    model, _ = build_robot_model(SIMPLE_BIPED.urdf,
                                 SimConfig(fixed_base=True, home_joint_pos=SIMPLE_BIPED.home_pose))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    assert model.nq == 6
    assert data.ncon == 0


# ----------------------------------------------------------------------------- 물리 검증
def test_pendulum_period_matches_theory():
    model, _ = build_robot_model(paths.PENDULUM_URDF, SimConfig(
        fixed_base=True, joint_armature=0.0, add_floor=False, add_imu=False))
    data = mujoco.MjData(model)
    m, L, r = 1.0, 0.5, 0.02
    i_pivot = m * (3 * r**2 + L**2) / 12 + m * (L / 2) ** 2
    t_theory = 2 * np.pi * np.sqrt(i_pivot / (m * 9.81 * L / 2))
    data.qpos[0] = 0.05
    crossings, prev = [], data.qpos[0]
    while data.time < 6.0:
        mujoco.mj_step(model, data)
        if prev < 0 <= data.qpos[0]:
            crossings.append(data.time)
        prev = data.qpos[0]
    assert np.mean(np.diff(crossings)) == pytest.approx(t_theory, rel=5e-3)


def test_biped_stands_and_contact_force_equals_weight(floating):
    model, data, robot = floating
    pd = make_pd(robot)
    q_home = robot.to_joint_vector(SIMPLE_BIPED.home_pose)
    simulate(model, data, lambda m, d: robot.set_joint_torques(
        pd.compute(robot.joint_positions(), robot.joint_velocities(), q_home)), duration=3.0, headless=True)
    assert robot.base_position()[2] > 0.55
    assert np.degrees(robot.base_tilt()) < 3.0
    force = sum(robot.contact_normal_force(b) for b in SIMPLE_BIPED.foot_bodies)
    assert force == pytest.approx(TOTAL_MASS * 9.81, rel=1e-2)
    assert robot.imu()["acc"][2] == pytest.approx(9.81, rel=1e-2)  # 정지 시 가속도계 = +g


def test_biped_collapses_without_control(floating):
    model, data, robot = floating
    simulate(model, data, lambda m, d: robot.set_joint_torques(np.zeros(6)), duration=3.0, headless=True)
    assert robot.base_position()[2] < 0.3


def test_joint_state_matches_raw_arrays(floating):
    model, data, robot = floating
    js = robot.as_joint_state()
    assert js["name"] == EXPECTED_JOINTS
    for i, name in enumerate(js["name"]):
        assert js["position"][i] == data.joint(name).qpos[0]


def test_imu_accelerometer_reads_gravity_reaction_on_fixed_base():
    """MuJoCo는 world에 용접된 바디의 가속도계를 0으로 낸다 → RobotInterface.imu()가 +g로 보정해야 함."""
    model, _ = build_robot_model(SIMPLE_BIPED.urdf, SimConfig(fixed_base=True))
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data, nstep=10)
    assert np.allclose(data.sensor("imu_acc").data, 0.0)  # MuJoCo 원시값 (이 동작이 바뀌면 알려 줌)
    np.testing.assert_allclose(RobotInterface(model, data).imu()["acc"], [0.0, 0.0, 9.81])
