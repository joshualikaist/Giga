"""sim_node — MuJoCo 시뮬레이터를 감싼 ROS2 노드 = "가짜 로봇 하드웨어".

실제 로봇에서 모터 드라이버가 하는 일을 똑같이 합니다:
    명령(목표 각도 등)을 받아 → 매 스텝 PD로 토크를 계산해 모터에 넣고 → 관절 상태를 내보낸다.
그래서 제어기 노드는 상대가 이 시뮬레이터인지 실제 로봇인지 몰라도 됩니다.

    발행  /joint_states    sensor_msgs/JointState  관절 각도·속도·토크 (stamp = 시뮬레이션 시간)
          /clock           rosgraph_msgs/Clock     시뮬레이션 시간 (다른 노드는 use_sim_time:=true)
    구독  /joint_commands  sensor_msgs/JointState  position=목표각, velocity=목표속도, effort=추가 토크
                                                   (일부 관절만 보내도 됨. 안 보낸 관절은 이전 목표 유지)
    서비스 /sim/reset      std_srvs/Trigger        home 자세로 초기화

실행
    ros2 run giga_sim_ros sim_node                                   # 바닥에 서 있는 로봇 + 뷰어
    ros2 run giga_sim_ros sim_node --ros-args -p fixed_base:=true    # 공중에 매단 로봇
    ros2 run giga_sim_ros sim_node --ros-args -p viewer:=false       # 화면 없이

구조 (docs/05_ros2_bridge_plan.md §2에서 측정으로 검증한 방식)
    한 스레드(SingleThreadedExecutor)에서 500 Hz 타이머가 [PD → mj_step → 발행]을 반복하고,
    뷰어를 켜면 60 Hz 타이머가 화면만 갱신한다.
"""
import signal

import mujoco
import numpy as np
import rclpy
from builtin_interfaces.msg import Time
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger

from biped_sim import (SIMPLE_BIPED, JointPDController, RobotInterface, SimConfig,
                       build_robot_model, passive_viewer, track_body_camera)

VIEWER_FPS = 60.0


def to_time_msg(t: float) -> Time:
    """시뮬레이션 시간 [s] → builtin_interfaces/Time."""
    sec = int(t)
    return Time(sec=sec, nanosec=int((t - sec) * 1e9))


class SimNode(Node):

    def __init__(self):
        super().__init__('sim_node')

        # ---------------------------------------------------------------- 파라미터
        # 실행할 때 --ros-args -p 이름:=값 으로 바꿀 수 있다. (ros2 param list /sim_node 로 확인)
        self.fixed_base = self.declare_parameter('fixed_base', False).value
        self.use_viewer = self.declare_parameter('viewer', True).value
        publish_rate = self.declare_parameter('publish_rate', 500.0).value  # [Hz]
        # 마지막 명령 후 이 시간[s]이 지나면 감쇠 모드(Kp=0). 0이면 끔 = 마지막 명령을 계속 유지.
        # (터미널에서 ros2 topic pub --once 로 연습할 땐 0, 제어기 노드를 붙일 땐 0.05 정도 권장)
        self.command_timeout = self.declare_parameter('command_timeout', 0.0).value

        # ---------------------------------------------------------------- MuJoCo 모델
        robot_cfg = SIMPLE_BIPED
        self.model, _ = build_robot_model(
            robot_cfg.urdf,
            SimConfig(fixed_base=self.fixed_base, home_joint_pos=robot_cfg.home_pose))
        self.data = mujoco.MjData(self.model)
        self.robot = RobotInterface(self.model, self.data)
        self.pd = JointPDController(self.robot.to_joint_vector(robot_cfg.kp),
                                    self.robot.to_joint_vector(robot_cfg.kd),
                                    self.robot.torque_limits)
        self.joint_index = {name: i for i, name in enumerate(self.robot.joint_names)}
        self.joint_ranges = self.model.jnt_range[self.robot.joint_ids].copy()
        self.home_key_id = self.model.key('home').id
        self.reset_simulation()

        # ---------------------------------------------------------------- ROS 인터페이스
        self.joint_state_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.clock_pub = self.create_publisher(Clock, 'clock', 10)
        self.create_subscription(JointState, 'joint_commands', self.on_command, 10)
        self.create_service(Trigger, 'sim/reset', self.on_reset)

        dt = self.model.opt.timestep
        self.publish_every = max(1, round((1.0 / dt) / publish_rate))
        self.step_timer = self.create_timer(dt, self.step)
        self.viewer = None
        self.viewer_timer = None

        self.get_logger().info(
            f"시작: {'공중에 매단(fixed base)' if self.fixed_base else '바닥에 선(floating base)'} 로봇, "
            f"관절 {self.robot.num_joints}개, 물리 {1.0 / dt:.0f} Hz, "
            f"/joint_states {1.0 / dt / self.publish_every:.0f} Hz, "
            f"명령 타임아웃 {'없음' if self.command_timeout <= 0 else f'{self.command_timeout} s'}")
        self.get_logger().info(f"관절 이름: {self.robot.joint_names}")

    # -------------------------------------------------------------------- 시뮬레이션
    def reset_simulation(self):
        """home 키프레임으로 되돌리고, 목표도 home 자세로."""
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_key_id)
        mujoco.mj_forward(self.model, self.data)
        n = self.robot.num_joints
        self.q_des = self.robot.joint_positions()  # = home 자세
        self.dq_des = np.zeros(n)
        self.tau_ff = np.zeros(n)
        self.last_command_time = None
        self.damping_mode = False
        self.step_count = 0

    def step(self):
        """500 Hz: 드라이버 PD → 물리 한 스텝 → 상태 발행."""
        q, dq = self.robot.joint_positions(), self.robot.joint_velocities()

        if self.command_timeout > 0 and self.last_command_time is not None:
            elapsed = (self.get_clock().now() - self.last_command_time).nanoseconds * 1e-9
            if elapsed > self.command_timeout and not self.damping_mode:
                self.damping_mode = True
                self.get_logger().warn(
                    f'{elapsed * 1000:.0f} ms 동안 명령 없음 → 감쇠 모드 (Kp=0, Kd만 작동)')

        if self.damping_mode:  # 실제 모터 드라이버의 안전 정지와 같은 동작: 버티지 않고 천천히 늘어짐
            tau = np.clip(-self.pd.kd * dq, self.pd.torque_limits[:, 0], self.pd.torque_limits[:, 1])
        else:
            tau = self.pd.compute(q, dq, self.q_des, self.dq_des, self.tau_ff)
        self.robot.set_joint_torques(tau)
        mujoco.mj_step(self.model, self.data)
        self.step_count += 1

        stamp = to_time_msg(self.data.time)
        self.clock_pub.publish(Clock(clock=stamp))
        if self.step_count % self.publish_every == 0:
            state = self.robot.as_joint_state()
            msg = JointState(name=state['name'], position=state['position'],
                             velocity=state['velocity'], effort=state['effort'])
            msg.header.stamp = stamp
            self.joint_state_pub.publish(msg)

    # -------------------------------------------------------------------- 콜백
    def on_command(self, msg: JointState):
        """명령 수신: 이름으로 관절을 찾아 목표를 갱신. 잘못된 명령은 통째로 거부하고 경고."""
        unknown = [n for n in msg.name if n not in self.joint_index]
        n = len(msg.name)
        problem = None
        if unknown:
            problem = f'모르는 관절 이름 {unknown} (사용 가능: {self.robot.joint_names})'
        elif len(msg.position) != n:
            problem = f'position 개수({len(msg.position)})가 name 개수({n})와 다름'
        elif msg.velocity and len(msg.velocity) != n:
            problem = f'velocity 개수({len(msg.velocity)})가 name 개수({n})와 다름'
        elif msg.effort and len(msg.effort) != n:
            problem = f'effort 개수({len(msg.effort)})가 name 개수({n})와 다름'
        if problem:
            self.get_logger().warn(f'명령 거부: {problem}', throttle_duration_sec=1.0)
            return

        for k, name in enumerate(msg.name):
            i = self.joint_index[name]
            lo, hi = self.joint_ranges[i]
            target = float(np.clip(msg.position[k], lo, hi))  # 관절 가동범위 밖 목표는 잘라냄
            if target != msg.position[k]:
                self.get_logger().warn(
                    f'{name}: 목표 {msg.position[k]:.3f} rad가 범위 [{lo:.2f}, {hi:.2f}] 밖 → {target:.3f}',
                    throttle_duration_sec=1.0)
            self.q_des[i] = target
            self.dq_des[i] = msg.velocity[k] if msg.velocity else 0.0
            self.tau_ff[i] = msg.effort[k] if msg.effort else 0.0

        self.last_command_time = self.get_clock().now()
        if self.damping_mode:
            self.damping_mode = False
            self.get_logger().info('명령 재수신 → PD 제어 재개')

    def on_reset(self, request, response):
        self.reset_simulation()
        response.success = True
        response.message = 'home 자세로 초기화했습니다'
        self.get_logger().info(response.message)
        return response

    # -------------------------------------------------------------------- 뷰어
    def attach_viewer(self, viewer):
        self.viewer = viewer
        with viewer.lock():
            track_body_camera(self.robot.base_body_id, distance=2.0)(viewer)
        self.viewer_timer = self.create_timer(1.0 / VIEWER_FPS, self.viewer.sync)

    def detach_viewer(self):
        if self.viewer_timer is not None:
            self.destroy_timer(self.viewer_timer)
        self.viewer_timer = None
        self.viewer = None


def _raise_keyboard_interrupt(signum, frame):
    raise KeyboardInterrupt


def main(args=None):
    # 종료 신호(Ctrl+C = SIGINT, launch/kill = SIGTERM)는 rclpy가 아니라 파이썬이 받아서
    # KeyboardInterrupt 하나의 경로로 순서대로 정리한다.
    # rclpy 기본 신호 처리를 쓰면 신호 순간 rclpy가 먼저 종료돼 버려서, 타이머가 발행 중이면
    # "RCLError: ... context is not valid" 트레이스가 남는 경합이 생김 (SIGINT·SIGTERM 모두 실제로 확인).
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    node = SimNode()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        if node.use_viewer:
            # 뷰어는 biped_sim.passive_viewer로 열고(닫을 때 그리기 스레드가 끝날 때까지 기다림),
            # ROS 종료보다 먼저 닫는다. launch_passive를 그대로 쓰면 종료 시 세그폴트나
            # GLXBadContext 후 멈춤이 생기는 것을 확인함 (docs/05_ros2_bridge_plan.md §2)
            with passive_viewer(node.model, node.data) as viewer:
                node.attach_viewer(viewer)
                while rclpy.ok() and viewer.is_running():
                    executor.spin_once(timeout_sec=0.01)
                node.detach_viewer()
        else:
            executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass  # Ctrl+C / 종료 신호: 아래에서 정리
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
