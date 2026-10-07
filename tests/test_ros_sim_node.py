"""giga_sim_ros/sim_node 통합 테스트 (ROS2 토픽·서비스로 시뮬레이터를 조종).

실행 조건: source scripts/activate.sh  +  bash scripts/build_ros.sh 로 빌드되어 있을 것.
(조건이 안 맞으면 건너뜀)

docs/06_ros2_hands_on.md 의 터미널 실습을 그대로 자동화한 것입니다.
다른 PC/터미널의 ROS2 노드와 섞이지 않도록 별도 ROS_DOMAIN_ID를 씁니다.
"""
import os
import signal
import subprocess
import time

import pytest

from biped_sim import paths

SIM_NODE = paths.REPO_ROOT / "ros2_ws/install/giga_sim_ros/lib/giga_sim_ros/sim_node"
DOMAIN_ID = 87
JOINTS = ["left_hip_pitch", "left_knee", "left_ankle_pitch",
          "right_hip_pitch", "right_knee", "right_ankle_pitch"]

pytestmark = pytest.mark.skipif(
    not os.environ.get("ROS_DISTRO") or not SIM_NODE.exists(),
    reason="ROS2 환경 미로드 또는 ros2_ws 미빌드 (source scripts/activate.sh; bash scripts/build_ros.sh)")


@pytest.fixture
def ros():
    """sim_node(화면 없음, 매단 모드)를 띄우고, 테스트용 ROS 클라이언트 노드를 만든다."""
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from sensor_msgs.msg import JointState
    from std_srvs.srv import Trigger

    env = dict(os.environ, ROS_DOMAIN_ID=str(DOMAIN_ID))
    proc = subprocess.Popen(
        [str(SIM_NODE), "--ros-args", "-p", "viewer:=false", "-p", "fixed_base:=true"],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    ctx = rclpy.Context()
    rclpy.init(context=ctx, domain_id=DOMAIN_ID)
    node = rclpy.create_node("test_client", context=ctx)
    executor = SingleThreadedExecutor(context=ctx)
    executor.add_node(node)

    states = []
    node.create_subscription(JointState, "joint_states", states.append, 100)
    cmd_pub = node.create_publisher(JointState, "joint_commands", 10)
    reset_cli = node.create_client(Trigger, "sim/reset")

    def spin_for(seconds):
        end = time.time() + seconds
        while time.time() < end:
            executor.spin_once(timeout_sec=0.01)

    def send(names, positions, seconds=0.5):
        # 연결(discovery) 직후 첫 메시지가 유실될 수 있어 잠시 반복 발행
        msg = JointState(name=names, position=positions)
        end = time.time() + seconds
        while time.time() < end:
            cmd_pub.publish(msg)
            spin_for(0.05)

    def latest_position(name):
        return states[-1].position[states[-1].name.index(name)]

    # sim_node가 뜨고 첫 메시지가 올 때까지 대기
    deadline = time.time() + 15
    while not states and time.time() < deadline:
        spin_for(0.1)
    assert states, "sim_node에서 /joint_states를 받지 못함:\n" + _drain(proc)

    yield dict(states=states, spin_for=spin_for, send=send, latest=latest_position,
               reset_cli=reset_cli, Trigger=Trigger, proc=proc, executor=executor)

    executor.shutdown()
    node.destroy_node()
    rclpy.shutdown(context=ctx)
    if proc.poll() is None:
        proc.send_signal(signal.SIGINT)
    out = proc.communicate(timeout=15)[0]
    ros_exit = proc.returncode
    assert ros_exit == 0, f"sim_node가 정상 종료되지 않음 (exit {ros_exit}):\n{out}"
    assert "Traceback" not in out, out


def _drain(proc):
    proc.kill()
    return proc.communicate(timeout=5)[0]


def test_joint_states_rate_and_names(ros):
    ros["states"].clear()
    ros["spin_for"](2.0)
    rate = len(ros["states"]) / 2.0
    assert rate > 400, f"/joint_states {rate:.0f} Hz (기대: 약 500 Hz)"
    assert list(ros["states"][-1].name) == JOINTS
    stamps = [s.header.stamp.sec + s.header.stamp.nanosec * 1e-9 for s in ros["states"]]
    assert all(b > a for a, b in zip(stamps, stamps[1:])), "stamp(시뮬레이션 시간)가 증가해야 함"


def test_command_moves_only_that_joint_and_reset(ros):
    home_right_knee = ros["latest"]("right_knee")
    ros["send"](["left_knee"], [1.5])
    ros["spin_for"](1.0)
    assert ros["latest"]("left_knee") == pytest.approx(1.5, abs=0.05)
    assert ros["latest"]("right_knee") == pytest.approx(home_right_knee, abs=0.01)  # 안 보낸 관절은 그대로

    # 잘못된 명령(모르는 관절)은 거부되고 상태가 바뀌지 않아야 함
    ros["send"](["left_elbow"], [0.0])
    ros["spin_for"](0.5)
    assert ros["latest"]("left_knee") == pytest.approx(1.5, abs=0.05)

    # /sim/reset → home 자세 (매단 상태 home 무릎 ≈ 0.8 rad)
    cli = ros["reset_cli"]
    assert cli.wait_for_service(timeout_sec=5.0)
    future = cli.call_async(ros["Trigger"].Request())
    end = time.time() + 5
    while not future.done() and time.time() < end:
        ros["spin_for"](0.05)
    assert future.result().success
    ros["spin_for"](1.0)
    assert ros["latest"]("left_knee") == pytest.approx(0.8, abs=0.05)
