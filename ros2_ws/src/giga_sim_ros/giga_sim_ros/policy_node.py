"""policy_node — 학습한 보행 정책을 ROS2로 돌리는 노드 = 로봇의 "뇌".

센서 토픽만 보고 관절 목표를 내므로, 상대가 sim_node(MuJoCo)든 실물 로봇의 드라이버든 똑같이 동작합니다.
정책은 GPU(MJX)에서 학습했고 여기서는 일반 MuJoCo + ROS2 메시지를 거쳐 돌리므로 = sim-to-sim 시험 (docs/11).

    구독  /imu/data       sensor_msgs/Imu         자세 → 몸통 좌표계의 중력 방향, 각속도
          /joint_states   sensor_msgs/JointState  정책 관절 각도·속도 (이름으로 찾음)
          /cmd_vel        geometry_msgs/Twist     명령 (몸통 기준): linear.x 앞뒤, linear.y 옆 [m/s], angular.z 회전 [rad/s]
    발행  /joint_commands sensor_msgs/JointState  정책 관절 목표 각도 = home + action_scale × 행동, 50 Hz (학습과 같은 주기)
          /policy/command geometry_msgs/Twist     실제로 정책에 넣은 명령 (학습 범위로 자른 값, 기록·확인용)
    서비스 /policy/reset  std_srvs/Trigger        sim_node를 처음 자세로 + 정책 상태(걸음 박자·직전 행동·관측 기록)도 처음으로
    서비스 호출 /sim/reset (auto_reset일 때) 넘어지면 1초 뒤 sim_node를 처음 자세로

실행 (보통은 duck.launch.py가 sim_node와 함께 실행)
    ros2 run giga_sim_ros policy_node --ros-args -p policy:=learning/pretrained/duck_cmd_policy.npz -p use_sim_time:=true

관측은 학습 환경과 같은 함수(biped_sim.envs.walk_spec.base_observation)로 만들고, 정책은 numpy로 계산한다
(biped_sim.policy.NumpyPolicy, learning/export_policy.py로 변환한 .npz). 실물에 없는 센서 값(몸통 속도·높이)을 쓰는
정책(--deployable-obs 없이 학습)은 돌릴 수 없다.
"""
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from std_srvs.srv import Trigger

from biped_sim import paths
from biped_sim.envs import walk_spec
from biped_sim.policy import NumpyPolicy

from .shutdown import init_with_stop_flag


def gravity_in_body(q_xyzw) -> np.ndarray:
    """IMU 자세 쿼터니언 (x, y, z, w; 몸통 → 월드) → 몸통 좌표계에서 본 중력 방향 단위 벡터 (똑바로 서면 (0, 0, −1))."""
    x, y, z, w = q_xyzw
    # 회전 행렬 R의 셋째 행 = R.T @ (0, 0, 1)
    up = np.array([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)])
    return -up


class PolicyNode(Node):

    def __init__(self):
        super().__init__('policy_node')
        policy_path = Path(self.declare_parameter('policy', 'learning/pretrained/duck_cmd_policy.npz').value)
        if not policy_path.is_absolute() and not policy_path.exists():
            policy_path = paths.REPO_ROOT / policy_path     # 저장소 기준 상대 경로도 받음
        # [s] 이 시간 동안 새 명령이 없으면 멈춤 (0, 0, 0) — 조이스틱처럼 계속 보내는 조종기용 안전장치.
        # 0이면 마지막 명령 유지 (키를 누를 때만 보내는 teleop_twist_keyboard용)
        self.cmd_timeout = self.declare_parameter('cmd_timeout', 0.5).value
        self.auto_reset = self.declare_parameter('auto_reset', True).value     # 넘어지면 /sim/reset (시뮬레이션만)

        self.policy = NumpyPolicy.load(policy_path)
        config = self.policy.config
        self.spec = walk_spec.spec_from_config(config)
        if not self.spec.deployable_obs:
            raise ValueError(f'{policy_path.name}: 실물 센서만 쓰는 정책(--deployable-obs로 학습)만 돌릴 수 있습니다 '
                             '(몸통 속도·높이는 IMU로 잴 수 없음)')
        _, info = walk_spec.build_mjx_model(self.spec)       # 정책 관절 이름·home·가동 범위 (학습과 같은 값)
        self.joint_names = info['joint_names']
        self.q_home = info['q_home']
        self.q_range = info['act_range'][info['policy_act']]     # 가동 범위 (구동 관절 순서 → 정책 관절 순서)
        self.target_speed = config.get('target_speed', self.spec.target_speed)
        commands = config.get('commands')
        self.cmd_lo = np.array([commands[k][0] for k in ('vx', 'vy', 'wz')]) if commands else None
        self.cmd_hi = np.array([commands[k][1] for k in ('vx', 'vy', 'wz')]) if commands else None

        self.imu = None                 # (중력 방향, 각속도)
        self.joints = None              # (각도, 속도) 정책 관절 순서
        self.command = np.zeros(3)
        self.last_cmd_time = None
        self.fallen_at = None
        self.reset_future = None
        self.reset_state()

        self.create_subscription(Imu, 'imu/data', self.on_imu, 10)
        self.create_subscription(JointState, 'joint_states', self.on_joint_states, 10)
        self.create_subscription(Twist, 'cmd_vel', self.on_cmd_vel, 10)
        self.command_pub = self.create_publisher(JointState, 'joint_commands', 10)
        self.applied_pub = self.create_publisher(Twist, 'policy/command', 10)
        self.reset_client = self.create_client(Trigger, 'sim/reset')
        self.create_service(Trigger, 'policy/reset', self.on_reset_request)
        self.create_timer(walk_spec.CONTROL_DT, self.control_step)    # use_sim_time이면 시뮬레이션 시간 기준 50 Hz

        rng = (f'명령 범위 vx {self.cmd_lo[0]:g}~{self.cmd_hi[0]:g}, vy {self.cmd_lo[1]:g}~{self.cmd_hi[1]:g} m/s, '
               f'wz {self.cmd_lo[2]:g}~{self.cmd_hi[2]:g} rad/s' if commands else f'명령 없음 (앞 {self.target_speed} m/s만)')
        self.get_logger().info(f'정책 {policy_path.name}: {self.spec.name}, 관측 {self.spec.obs_size} → 관절 {self.spec.n}개, '
                               f'{1 / walk_spec.CONTROL_DT:.0f} Hz, {rng}')

    # -------------------------------------------------------------------- 상태
    def reset_state(self):
        """에피소드 처음: 걸음 박자 0, 직전 행동 0, 관측 기록은 첫 관측으로 채움 (학습과 같게)."""
        self.t = 0.0
        self.last_action = np.zeros(self.spec.n)
        self.obs_hist = None

    # -------------------------------------------------------------------- 구독
    def on_imu(self, msg: Imu):
        q = msg.orientation
        gyro = msg.angular_velocity
        self.imu = (gravity_in_body((q.x, q.y, q.z, q.w)), np.array([gyro.x, gyro.y, gyro.z]))

    def on_joint_states(self, msg: JointState):
        index = {n: i for i, n in enumerate(msg.name)}
        if not all(n in index for n in self.joint_names):
            return
        k = [index[n] for n in self.joint_names]
        self.joints = (np.array(msg.position)[k], np.array(msg.velocity)[k] if msg.velocity else np.zeros(len(k)))

    def on_cmd_vel(self, msg: Twist):
        self.command = np.array([msg.linear.x, msg.linear.y, msg.angular.z])
        self.last_cmd_time = self.get_clock().now()

    # -------------------------------------------------------------------- 제어
    def current_command(self) -> np.ndarray:
        """학습한 범위로 자른 명령. cmd_timeout 동안 새 명령이 없으면 (0, 0, 0) = 제자리 걸음."""
        cmd = self.command
        if self.last_cmd_time is None or (self.cmd_timeout > 0 and
                                          (self.get_clock().now() - self.last_cmd_time).nanoseconds * 1e-9 > self.cmd_timeout):
            cmd = np.zeros(3)
        if self.cmd_lo is not None:
            cmd = np.clip(cmd, self.cmd_lo, self.cmd_hi)
        return cmd

    def control_step(self):
        if self.imu is None or self.joints is None:
            return                                       # 센서 값을 아직 못 받음
        if self.reset_future is not None:                # /sim/reset을 기다리는 중
            if not self.reset_future.done():
                return
            self.reset_future, self.fallen_at, self.imu, self.joints = None, None, None, None
            self.reset_state()
            return
        gravity, gyro = self.imu
        if gravity[2] > -np.cos(walk_spec.FALL_TILT):    # 기울기 > 0.6 rad = 학습의 '넘어짐'
            self.on_fall()
            return

        cmd = self.current_command()
        q, dq = self.joints
        base = walk_spec.base_observation(self.spec, gravity, gyro, q - self.q_home, dq, self.last_action,
                                          cmd if self.spec.command_obs else self.target_speed, self.t)
        if self.obs_hist is None:
            self.obs_hist = np.tile(base, (self.spec.obs_history, 1))
        obs = np.concatenate([base, self.obs_hist.ravel()])
        if self.spec.obs_history:
            self.obs_hist = np.concatenate([base[None], self.obs_hist[:-1]])

        action = np.clip(self.policy(obs), -1.0, 1.0)
        target = np.clip(self.q_home + self.spec.action_scale * action, self.q_range[:, 0], self.q_range[:, 1])
        msg = JointState(name=self.joint_names, position=target.tolist())
        msg.header.stamp = self.get_clock().now().to_msg()
        self.command_pub.publish(msg)
        applied = Twist()
        applied.linear.x, applied.linear.y, applied.angular.z = (float(c) for c in cmd)
        self.applied_pub.publish(applied)
        self.last_action = action
        self.t += walk_spec.CONTROL_DT

    def request_reset(self) -> bool:
        """sim_node를 처음 자세로 (비동기). 끝나면 control_step이 정책 상태도 처음으로 돌림."""
        if self.reset_future is not None:
            return True
        if not self.reset_client.service_is_ready():
            return False
        self.reset_future = self.reset_client.call_async(Trigger.Request())
        return True

    def on_reset_request(self, request, response):
        response.success = self.request_reset()
        response.message = '처음 자세로 되돌립니다' if response.success else '/sim/reset 서비스가 없습니다'
        return response

    def on_fall(self):
        now = self.get_clock().now()
        if self.fallen_at is None:
            self.fallen_at = now
            self.get_logger().warn('넘어짐 (기울기 > 0.6 rad)' + (' → 1초 뒤 처음 자세로' if self.auto_reset else ''))
        if self.auto_reset and (now - self.fallen_at).nanoseconds * 1e-9 > 1.0:
            self.request_reset()


def main(args=None):
    stop = init_with_stop_flag(args)
    node = PolicyNode()
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
