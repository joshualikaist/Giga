"""duck.launch.py 통합 테스트: 학습한 오리 정책을 ROS2로 (sim_node + policy_node), /cmd_vel로 조종 (docs/11).

실행 조건: source scripts/activate.sh  +  bash scripts/build_ros.sh  +  오리 로봇 파일 (조건이 안 맞으면 건너뜀)
화면 없이(viewer:=false) 실행하며, 다른 ROS2 노드와 섞이지 않도록 별도 ROS_DOMAIN_ID를 쓴다.
"""
import math
import os
import signal
import subprocess
import time

import pytest

from biped_sim import OPEN_DUCK_MINI, paths

LAUNCH_FILE = paths.REPO_ROOT / "ros2_ws/install/giga_sim_ros/share/giga_sim_ros/launch/duck.launch.py"
DOMAIN_ID = 89

pytestmark = [
    pytest.mark.skipif(not os.environ.get("ROS_DISTRO") or not LAUNCH_FILE.exists(),
                       reason="ROS2 환경 미로드 또는 ros2_ws 미빌드 (source scripts/activate.sh; bash scripts/build_ros.sh)"),
    pytest.mark.skipif(not OPEN_DUCK_MINI.urdf.exists(), reason="오리 로봇 파일 없음 → bash scripts/get_open_duck.sh"),
]


class Driver:
    """테스트 쪽 ROS 노드: /cmd_vel을 보내고 몸통 TF를 관찰, /policy/reset 호출."""

    def __init__(self):
        import rclpy
        from geometry_msgs.msg import Twist
        from rclpy.executors import SingleThreadedExecutor
        from std_srvs.srv import Trigger
        from tf2_msgs.msg import TFMessage

        self.rclpy, self.Twist, self.Trigger = rclpy, Twist, Trigger
        self.ctx = rclpy.Context()
        rclpy.init(context=self.ctx, domain_id=DOMAIN_ID)
        self.node = rclpy.create_node("duck_driver", context=self.ctx)
        self.executor = SingleThreadedExecutor(context=self.ctx)
        self.executor.add_node(self.node)
        self.base = []      # (x, y, z, tilt_deg)
        self.node.create_subscription(TFMessage, "tf", self._on_tf, 100)
        self.cmd_pub = self.node.create_publisher(Twist, "cmd_vel", 10)
        self.reset = self.node.create_client(Trigger, "policy/reset")

    def _on_tf(self, msg):
        for t in msg.transforms:
            if t.header.frame_id == "odom" and t.child_frame_id == "base_link":
                q, p = t.transform.rotation, t.transform.translation
                tilt = math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (q.x ** 2 + q.y ** 2)))))
                self.base.append((p.x, p.y, p.z, tilt))

    def drive(self, vx, vy, wz, seconds):
        """seconds 동안 20 Hz로 /cmd_vel 발행하며 TF 수신."""
        msg = self.Twist()
        msg.linear.x, msg.linear.y, msg.angular.z = float(vx), float(vy), float(wz)
        end = time.time() + seconds
        next_pub = 0.0
        while time.time() < end:
            if time.time() >= next_pub:
                self.cmd_pub.publish(msg)
                next_pub = time.time() + 0.05
            self.executor.spin_once(timeout_sec=0.01)

    def wait_for(self, cond, timeout):
        end = time.time() + timeout
        while not cond() and time.time() < end:
            self.executor.spin_once(timeout_sec=0.05)
        return cond()

    def call_reset(self):
        future = self.reset.call_async(self.Trigger.Request())
        assert self.wait_for(future.done, 10.0), "/policy/reset 응답 없음"
        return future.result()

    def close(self):
        self.executor.shutdown()
        self.node.destroy_node()
        self.rclpy.shutdown(context=self.ctx)


def start_launch(*launch_args):
    env = dict(os.environ, ROS_DOMAIN_ID=str(DOMAIN_ID))
    return subprocess.Popen(["ros2", "launch", "giga_sim_ros", "duck.launch.py", "viewer:=false", *launch_args],
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            start_new_session=True)


def stop_launch(proc):
    if proc.poll() is None:
        proc.send_signal(signal.SIGINT)
    try:
        return proc.communicate(timeout=20)[0]
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        return proc.communicate()[0] + "\n[test] launch가 20초 안에 끝나지 않아 강제 종료함"


@pytest.fixture
def driver():
    d = Driver()
    yield d
    d.close()


def test_duck_policy_follows_cmd_vel_through_ros(driver):
    """/cmd_vel 앞 0.15 m/s → 학습한 정책이 ROS 토픽만으로 앞으로 걷고 (넘어지지 않음), 옆 명령이면 옆으로.
    /policy/reset이면 처음 자리로. 모든 노드가 깔끔히 종료."""
    proc = start_launch()
    try:
        assert driver.wait_for(lambda: driver.base and driver.reset.service_is_ready(), 30.0), "오리·정책이 시작되지 않음"
        driver.drive(0.15, 0.0, 0.0, 6.0)
        walk = list(driver.base)
        assert driver.call_reset().success
        driver.base.clear()
        driver.drive(0.0, 0.0, 0.0, 1.0)
        after_reset = list(driver.base)
        driver.base.clear()
        driver.drive(0.0, 0.08, 0.0, 6.0)
        side = list(driver.base)
    finally:
        out = stop_launch(proc)

    assert max(b[3] for b in walk) < 30, "앞으로 걷다 넘어짐"
    assert walk[-1][0] - walk[0][0] > 0.5, f"앞으로 충분히 가지 않음 ({walk[-1][0] - walk[0][0]:.2f} m / 6 s)"
    assert abs(after_reset[-1][0]) < 0.1, "reset 뒤 처음 자리가 아님"
    assert max(b[3] for b in side) < 30, "옆걸음하다 넘어짐"
    assert side[-1][1] - side[0][1] > 0.2, f"왼쪽으로 충분히 가지 않음 ({side[-1][1] - side[0][1]:.2f} m / 6 s)"
    assert proc.returncode == 0, out
    assert out.count("process has finished cleanly") == 2, out   # sim_node, policy_node
    assert "Traceback" not in out, out
