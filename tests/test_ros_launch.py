"""sim.launch.py 통합 테스트: 시뮬레이터 + robot_state_publisher + demo_controller를 launch로 함께 실행.

실행 조건: source scripts/activate.sh  +  bash scripts/build_ros.sh (조건이 안 맞으면 건너뜀)
화면 없이(viewer:=false rviz:=false) 실행하며, 다른 ROS2 노드와 섞이지 않도록 별도 ROS_DOMAIN_ID를 쓴다.
"""
import math
import os
import signal
import subprocess
import time

import pytest

from biped_sim import paths

LAUNCH_FILE = paths.REPO_ROOT / "ros2_ws/install/giga_sim_ros/share/giga_sim_ros/launch/sim.launch.py"
DOMAIN_ID = 88

pytestmark = pytest.mark.skipif(
    not os.environ.get("ROS_DISTRO") or not LAUNCH_FILE.exists(),
    reason="ROS2 환경 미로드 또는 ros2_ws 미빌드 (source scripts/activate.sh; bash scripts/build_ros.sh)")


@pytest.fixture(autouse=True)
def require_workspace_overlay():
    """빌드는 했는데 현재 터미널에 로드하지 않은 경우(`ros2 launch`가 패키지를 못 찾음)를 분명히 알려 준다."""
    this_install = str(paths.REPO_ROOT / "ros2_ws" / "install" / "giga_sim_ros")
    if this_install not in os.environ.get("AMENT_PREFIX_PATH", "").split(":"):
        pytest.fail("ros2_ws는 빌드됐지만 현재 터미널에 로드되지 않았습니다 → "
                    "`source scripts/activate.sh`를 다시 실행한 뒤 pytest를 돌리세요.")


class Observer:
    """테스트 쪽 ROS 노드: 몸통 TF와 명령 토픽을 관찰한다."""

    def __init__(self):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from sensor_msgs.msg import JointState
        from tf2_msgs.msg import TFMessage

        self.rclpy = rclpy
        self.ctx = rclpy.Context()
        rclpy.init(context=self.ctx, domain_id=DOMAIN_ID)
        self.node = rclpy.create_node("launch_observer", context=self.ctx)
        self.executor = SingleThreadedExecutor(context=self.ctx)
        self.executor.add_node(self.node)
        self.base = []      # (z, tilt_deg)
        self.n_commands = 0
        self.node.create_subscription(TFMessage, "tf", self._on_tf, 100)
        self.node.create_subscription(JointState, "joint_commands", self._on_cmd, 100)

    def _on_tf(self, msg):
        for t in msg.transforms:
            if t.header.frame_id == "odom" and t.child_frame_id == "base_link":
                q = t.transform.rotation
                tilt = math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (q.x ** 2 + q.y ** 2)))))
                self.base.append((t.transform.translation.z, tilt))

    def _on_cmd(self, msg):
        self.n_commands += 1

    def spin_for(self, seconds):
        end = time.time() + seconds
        while time.time() < end:
            self.executor.spin_once(timeout_sec=0.01)

    def wait_for_robot(self, timeout=20.0):
        end = time.time() + timeout
        while not self.base and time.time() < end:
            self.spin_for(0.1)
        return bool(self.base)

    def close(self):
        self.executor.shutdown()
        self.node.destroy_node()
        self.rclpy.shutdown(context=self.ctx)


def start_launch(*launch_args):
    env = dict(os.environ, ROS_DOMAIN_ID=str(DOMAIN_ID))
    return subprocess.Popen(
        ["ros2", "launch", "giga_sim_ros", "sim.launch.py", "viewer:=false", "rviz:=false", *launch_args],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        start_new_session=True)  # Ctrl+C(SIGINT)를 launch 프로세스에만 보내기 위해


def stop_launch(proc):
    if proc.poll() is None:
        proc.send_signal(signal.SIGINT)
    try:
        return proc.communicate(timeout=20)[0]
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        return proc.communicate()[0] + "\n[test] launch가 20초 안에 끝나지 않아 강제 종료함"


@pytest.fixture
def observer():
    obs = Observer()
    yield obs
    obs.close()


def test_squat_controller_through_ros(observer):
    """ROS 토픽만으로 앉았다 일어서기: 몸통이 오르내리되 넘어지지 않고, 모든 노드가 깔끔히 종료."""
    proc = start_launch("controller:=squat")
    try:
        assert observer.wait_for_robot(), "odom→base_link TF가 오지 않음"
        observer.base.clear()
        observer.n_commands = 0
        observer.spin_for(5.0)  # squat 주기 4 s → 한 번 이상 앉았다 일어남
    finally:
        out = stop_launch(proc)

    z = [b[0] for b in observer.base]
    tilt = [b[1] for b in observer.base]
    assert min(z) < 0.53, f"충분히 앉지 않음 (min z {min(z):.3f})"   # 가장 깊이: 약 0.505 m
    assert max(z) > 0.58, f"다시 일어서지 않음 (max z {max(z):.3f})"  # 서 있을 때: 약 0.588 m
    assert max(tilt) < 5.0, f"몸통이 기울어짐 ({max(tilt):.1f}°)"
    assert observer.n_commands / 5.0 > 80, "demo_controller 명령 주기가 100 Hz 근처여야 함"
    assert proc.returncode == 0, out
    assert out.count("process has finished cleanly") == 3, out  # sim_node, rsp, demo_controller
    assert "Traceback" not in out, out


def test_command_timeout_goes_limp_when_controller_dies(observer):
    """제어기가 죽으면 50 ms 뒤 감쇠 모드(Kp=0) → 경고 로그, 서 있던 로봇은 주저앉음 (실제 드라이버의 안전 정지)."""
    proc = start_launch("controller:=stand")
    try:
        assert observer.wait_for_robot()
        observer.spin_for(2.0)
        assert observer.base[-1][0] == pytest.approx(0.588, abs=0.01)  # 서 있음
        ctrl = subprocess.run(["pgrep", "-f", "lib/giga_sim_ros/demo_controller"],
                              capture_output=True, text=True).stdout.split()
        assert ctrl, "demo_controller 프로세스를 찾지 못함"
        for pid in ctrl:
            os.kill(int(pid), signal.SIGKILL)
        observer.spin_for(3.0)
    finally:
        out = stop_launch(proc)

    assert "명령 없음 → 감쇠 모드" in out, out
    assert observer.base[-1][0] < 0.3, "감쇠 모드에서는 버티지 않고 주저앉아야 함"
