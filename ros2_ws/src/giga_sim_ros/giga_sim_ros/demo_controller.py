"""demo_controller — 토픽만으로 로봇을 조종하는 가장 단순한 제어기 노드.

docs/06 실습에서 사람이 터미널로 하던 `ros2 topic pub /joint_commands ...`를 코드로 옮긴 것입니다.

    구독  /joint_states    로봇이 가진 관절 이름을 알아내고, 로봇이 살아 있는지 확인
    발행  /joint_commands  목표 관절각(position)과 목표 속도(velocity)를 rate Hz로 계속 보냄

    motion:=stand  home 자세(무릎을 살짝 굽힌 자세) 유지
    motion:=squat  무릎을 천천히 굽혔다 폈다 반복 (앉았다 일어서기)

이 노드는 mujoco도 biped_sim도 import하지 않습니다. 관절 이름과 토픽만 알면 되므로,
sim_node 대신 실제 로봇 드라이버 노드가 같은 토픽을 제공하면 코드 수정 없이 그대로 동작합니다.

자세 규칙 (docs/04 §4): hip = −a, knee = 2a, ankle = −a 이면 발바닥이 수평이고 발목이 고관절 바로 아래.
"""
import math

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState

from .shutdown import init_with_stop_flag

# 관절 이름 끝부분 → 굽힘 정도 a에 곱할 계수 (hip = −a, knee = 2a, ankle = −a). left_/right_ 공통
BEND_RULE = {'hip_pitch': -1.0, 'knee': 2.0, 'ankle_pitch': -1.0}


class DemoController(Node):

    def __init__(self):
        super().__init__('demo_controller')
        self.motion = self.declare_parameter('motion', 'squat').value       # stand | squat
        rate = self.declare_parameter('rate', 100.0).value                  # [Hz] 명령 주기
        self.a_home = self.declare_parameter('home_bend', 0.4).value        # [rad] 서 있을 때 a
        self.a_deep = self.declare_parameter('squat_bend', 0.7).value       # [rad] 가장 깊이 앉았을 때 a
        self.period = self.declare_parameter('squat_period', 4.0).value     # [s] 앉았다 일어서는 한 주기
        if self.motion not in ('stand', 'squat'):
            raise ValueError(f"motion은 'stand' 또는 'squat'이어야 합니다: {self.motion!r}")

        self.joint_names = None  # 첫 /joint_states를 받으면 채워짐
        self.t_start = None
        self.cmd_pub = self.create_publisher(JointState, 'joint_commands', 10)
        self.create_subscription(JointState, 'joint_states', self.on_joint_states, 10)
        self.create_timer(1.0 / rate, self.on_timer)
        self.get_logger().info(f"motion={self.motion}, {rate:.0f} Hz. /joint_states를 기다리는 중...")

    def on_joint_states(self, msg: JointState):
        if self.joint_names is None:
            self.joint_names = list(msg.name)
            self.t_start = self.get_clock().now()
            self.get_logger().info(f'로봇 발견, 관절 {len(self.joint_names)}개 → 명령 시작')

    def bend(self, t: float):
        """시간 t[s]에서의 굽힘 정도 a와 그 변화율 da/dt."""
        if self.motion == 'stand':
            return self.a_home, 0.0
        # a_home에서 시작해 a_deep까지 갔다가 돌아오는 코사인 곡선 (시작할 때 튀지 않게)
        w = 2.0 * math.pi / self.period
        half = (self.a_deep - self.a_home) / 2.0
        a = self.a_home + half * (1.0 - math.cos(w * t))
        da = half * w * math.sin(w * t)
        return a, da

    def on_timer(self):
        if self.joint_names is None:
            return  # 아직 로봇이 안 보임
        # use_sim_time:=true로 실행하면 이 시계는 시뮬레이션 시간(/clock)을 따른다
        t = (self.get_clock().now() - self.t_start).nanoseconds * 1e-9
        a, da = self.bend(t)
        names, pos, vel = [], [], []
        for name in self.joint_names:
            for suffix, k in BEND_RULE.items():
                if name.endswith(suffix):
                    names.append(name)
                    pos.append(k * a)
                    vel.append(k * da)
                    break
        self.cmd_pub.publish(JointState(name=names, position=pos, velocity=vel))


def main(args=None):
    stop = init_with_stop_flag(args)  # Ctrl+C / SIGTERM → stop.set() (giga_sim_ros/shutdown.py)
    node = DemoController()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        while not stop.is_set():
            executor.spin_once(timeout_sec=0.05)
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
