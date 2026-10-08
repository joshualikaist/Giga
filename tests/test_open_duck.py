"""외부 오픈소스 로봇(Open Duck Mini v2) 불러오기 테스트. 파일이 없으면 건너뜀 → bash scripts/get_open_duck.sh

받은 URDF를 고치지 않고 RobotConfig.sim_defaults만으로 시뮬레이션할 수 있는지 확인한다 (docs/09).
"""
import mujoco
import numpy as np
import pytest

from biped_sim import OPEN_DUCK_MINI as CFG
from biped_sim import JointPDController, RobotInterface, build_robot_model
from biped_sim.utils import lowest_collision_z

pytestmark = pytest.mark.skipif(not CFG.urdf.exists(), reason="Open Duck Mini 파일 없음 → bash scripts/get_open_duck.sh")


@pytest.fixture(scope="module")
def floating():
    model, _ = build_robot_model(CFG.urdf, CFG.sim_config(fixed_base=False))
    return model


def test_builder_fixes_raw_urdf_problems(floating):
    m = floating
    assert m.nu == 16 and m.nq == 7 + 16                          # freejoint + 모터 16개
    np.testing.assert_allclose(m.actuator_ctrlrange[:, 1], 3.23)   # URDF effort=1 대신 서보 사양
    colliding = {m.body(m.geom_bodyid[g]).name for g in range(m.ngeom) if m.geom_contype[g] and m.geom_bodyid[g]}
    assert colliding == set(CFG.foot_bodies)                       # 발만 충돌
    assert m.ngeom > 100                                           # 충돌을 끈 몸통·다리 형상도 보이도록 남아 있음


def test_home_pose_starts_just_above_floor(floating):
    """메시 발의 가장 낮은 점을 정확히 계산 → home 자세에서 바닥 1 mm 위 (경계구 근사면 5 cm 위에서 떨어짐)."""
    d = mujoco.MjData(floating)
    mujoco.mj_resetDataKeyframe(floating, d, floating.key("home").id)
    mujoco.mj_forward(floating, d)
    assert lowest_collision_z(floating, d) == pytest.approx(0.001, abs=1e-6)
    assert d.ncon == 0


def test_hip_pitch_sign_is_mirrored_left_right():
    """같은 +0.3 rad에 hip_pitch는 좌우 발이 반대 방향으로, knee·ankle은 같은 방향으로 움직인다 (home 자세의 근거)."""
    m, _ = build_robot_model(CFG.urdf, CFG.sim_config(fixed_base=True))
    d = mujoco.MjData(m)

    def foot_dx(side, joint):
        d.qpos[:] = 0
        mujoco.mj_forward(m, d)
        x0 = d.xpos[m.body(f"{side}_foot").id, 0]
        d.qpos[m.joint(f"{side}_{joint}").qposadr[0]] = 0.3
        mujoco.mj_forward(m, d)
        return d.xpos[m.body(f"{side}_foot").id, 0] - x0

    assert np.sign(foot_dx("left", "hip_pitch")) == -np.sign(foot_dx("right", "hip_pitch"))
    for joint in ("knee", "ankle"):
        assert np.sign(foot_dx("left", joint)) == np.sign(foot_dx("right", joint))


def test_stands_with_joint_pd(floating):
    """관절 PD로 3초 서 있기: 넘어지지 않고, 두 발 하중의 합 = 무게."""
    m = floating
    d = mujoco.MjData(m)
    robot = RobotInterface(m, d)
    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
    mujoco.mj_forward(m, d)
    q_home = robot.to_joint_vector(CFG.home_pose)
    pd = JointPDController(robot.to_joint_vector(CFG.kp), robot.to_joint_vector(CFG.kd), robot.torque_limits)
    for _ in range(int(3.0 / m.opt.timestep)):
        robot.set_joint_torques(pd.compute(robot.joint_positions(), robot.joint_velocities(), q_home))
        mujoco.mj_step(m, d)
    assert np.degrees(robot.base_tilt()) < 10
    load = sum(robot.contact_normal_force(b) for b in CFG.foot_bodies)
    assert load == pytest.approx(robot.total_mass * 9.81, rel=0.02)
