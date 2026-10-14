"""sim_node — MuJoCo 시뮬레이터를 감싼 ROS2 노드 = "가짜 로봇 하드웨어".

실제 로봇에서 모터 드라이버가 하는 일을 똑같이 합니다:
    명령(목표 각도 등)을 받아 → 매 스텝 PD로 토크를 계산해 모터에 넣고 → 관절 상태를 내보낸다.
그래서 제어기 노드는 상대가 이 시뮬레이터인지 실제 로봇인지 몰라도 됩니다.

    발행  /joint_states    sensor_msgs/JointState  관절 각도·속도·토크 (stamp = 시뮬레이션 시간)
          /imu/data        sensor_msgs/Imu         몸통 IMU: 자세(x,y,z,w), 각속도, 가속도(중력 포함)
          /tf              odom → base_link        몸통 위치·자세 (시뮬레이션만 아는 정답값, RViz 표시용)
          /clock           rosgraph_msgs/Clock     시뮬레이션 시간 (다른 노드는 use_sim_time:=true)
    구독  /joint_commands  sensor_msgs/JointState  position=목표각, velocity=목표속도, effort=추가 토크
                                                   (일부 관절만 보내도 됨. 안 보낸 관절은 이전 목표 유지)
    서비스 /sim/reset      std_srvs/Trigger        home 자세로 초기화

실행
    ros2 run giga_sim_ros sim_node                                   # 바닥에 서 있는 로봇 + 뷰어
    ros2 run giga_sim_ros sim_node --ros-args -p fixed_base:=true    # 공중에 매단 로봇
    ros2 run giga_sim_ros sim_node --ros-args -p viewer:=false       # 화면 없이
    ros2 run giga_sim_ros sim_node --ros-args -p robot:=open_duck_mini -p terrain:=hills   # 오리 + 언덕 (docs/11)

화면 (MuJoCo 창에서 숫자 키): 6 따라가기(기본) · 7 위에서 · 8 옆에서 · 9 지형 전체. 마우스로 돌리고 확대해도 됨.
    왼쪽 위에 지금 보기, 아래에 policy_node가 넣은 명령(/policy/command)과 몸통 위치를 표시 (MuJoCo 글꼴은 한글이 안 돼 영어).

오리(robot:=open_duck_mini)는 GPU 학습과 똑같은 모델로 만든다 (biped_sim.envs.walk_spec.build_mjx_model:
    서보 모델·서 있는 자세·발 충돌 상자·물리 0.004 s). 학습한 정책은 policy_node가 돌린다 (duck.launch.py).

구조 (docs/05_ros2_bridge_plan.md §2에서 측정으로 검증한 방식)
    한 스레드(SingleThreadedExecutor)에서 물리 주기(500 Hz, 오리 250 Hz) 타이머가 [PD → mj_step → 발행]을 반복하고,
    뷰어를 켜면 60 Hz 타이머가 화면만 갱신한다.
    실제 시간 맞추기: 타이머가 늦게 불리면(화면 갱신 sync가 한 번에 약 5 ms씩 기다림) 밀린 물리 스텝을 그 자리에서
    따라잡는다. 안 하면 놓친 스텝만큼 시뮬레이션이 느려져 오리 + 화면에서 실제 시간의 0.7배였음 (docs/11).
"""
import time

import mujoco
import numpy as np
import rclpy
from builtin_interfaces.msg import Time
from geometry_msgs.msg import TransformStamped, Twist
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Imu, JointState
from std_srvs.srv import Trigger
from tf2_ros import TransformBroadcaster

from biped_sim import (SIMPLE_BIPED, JointPDController, RobotInterface, SimConfig,
                       build_robot_model, passive_viewer, track_body_camera)
from biped_sim.envs import walk_spec
from biped_sim.terrain import KINDS, make_terrain
from biped_sim.utils import quat_wxyz_to_xyzw  # MuJoCo (w,x,y,z) → ROS (x,y,z,w)

from .shutdown import init_with_stop_flag

VIEWER_FPS = 60.0
# 화면 보기 (숫자 키 → 이름, 카메라). 따라가는 보기는 몸통을 따라가는 MuJoCo 추적 카메라의 각도만 바꾼 것.
# azimuth 0 = 위에서 볼 때 화면 위쪽이 +x (출발할 때 로봇의 앞)
VIEWS = {54: ('follow', dict(azimuth=135.0, elevation=-15.0, zoom=1.0)),     # '6'
         55: ('top', dict(azimuth=0.0, elevation=-89.0, zoom=2.2)),          # '7'
         56: ('side', dict(azimuth=90.0, elevation=-8.0, zoom=1.5)),         # '8'
         57: ('map', dict(azimuth=0.0, elevation=-89.0, zoom=None))}         # '9' 지형 전체 (따라가지 않음)
MAX_CATCH_UP = 25   # 한 번에 따라잡는 최대 물리 스텝 (오리 0.1 s). 이보다 밀리면 나머지는 버림 (시뮬레이션이 잠깐 느려짐)


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
        publish_rate = self.declare_parameter('publish_rate', 500.0).value  # [Hz] joint_states, imu
        tf_rate = self.declare_parameter('tf_rate', 100.0).value            # [Hz] odom → base_link
        # 마지막 명령 후 이 시간[s]이 지나면 감쇠 모드(Kp=0). 0이면 끔 = 마지막 명령을 계속 유지.
        # (터미널에서 ros2 topic pub --once 로 연습할 땐 0, 제어기 노드를 붙일 땐 0.05 정도 권장)
        self.command_timeout = self.declare_parameter('command_timeout', 0.0).value
        # 로봇: simple_biped(기본) | open_duck_mini. 오리는 학습 설정과 같은 서보 모델·서 있는 자세로
        # (duck.launch.py가 정책의 학습 설정에서 읽어 넘김)
        robot_name = self.robot_name = self.declare_parameter('robot', 'simple_biped').value
        servo = self.declare_parameter('servo', 'open_duck').value          # 오리만: ideal | open_duck
        home_knee = self.declare_parameter('home_knee', 0.0).value          # 오리만: 서 있는 무릎 각도 [rad], 0 = 기본
        # 지형: flat | hills | slopes | obstacles | mixed (biped_sim.terrain, 원점 주변은 평평한 출발 자리)
        terrain_kind = self.declare_parameter('terrain', 'flat').value
        terrain_difficulty = self.declare_parameter('terrain_difficulty', 1.0).value
        terrain_seed = self.declare_parameter('terrain_seed', 0).value
        terrain_size = self.declare_parameter('terrain_size', 4.0).value    # [m] 원점에서 ±이만큼
        if terrain_kind not in KINDS:
            raise ValueError(f"terrain은 {KINDS} 중 하나: {terrain_kind!r}")
        terrain = None if terrain_kind == 'flat' else make_terrain(
            terrain_kind, seed=terrain_seed, difficulty=terrain_difficulty,
            size_x=terrain_size, size_y=terrain_size, center=(0.0, 0.0))

        # ---------------------------------------------------------------- MuJoCo 모델
        if robot_name == 'simple_biped':
            robot_cfg = SIMPLE_BIPED
            self.model, _ = build_robot_model(
                robot_cfg.urdf,
                SimConfig(fixed_base=self.fixed_base, home_joint_pos=robot_cfg.home_pose, terrain=terrain))
            kp, kd = robot_cfg.kp, robot_cfg.kd
            self.view_distance = 2.0
        else:   # 학습한 로봇: GPU 학습과 같은 모델 (서보 모델은 PD 게인 + 관절 damping·마찰로 들어 있음)
            if self.fixed_base:
                raise ValueError(f'{robot_name}은 fixed_base를 지원하지 않습니다 (학습 모델은 바닥에 선 로봇)')
            spec = walk_spec.spec_from_config({'robot': robot_name, 'servo': servo,
                                               'home_knee': home_knee if home_knee > 0 else None})
            self.model, info = walk_spec.build_mjx_model(spec, terrain)
            act_names = [self.model.joint(int(self.model.actuator_trnid[a, 0])).name for a in range(self.model.nu)]
            kp, kd = dict(zip(act_names, info['kp'])), dict(zip(act_names, info['kd']))
            self.view_distance = spec.view_distance
            # 그림자·바닥 반사 끄기: 오리 메시는 한 장 그리는 데 8.6 → 2.1 ms (실측). 뷰어가 그리는 동안 물리 스텝이
            # 기다리므로, 켜 두면 시뮬레이션이 실제 시간의 0.7배로 느려짐 (조이스틱 조종이 슬로모션처럼)
            self.model.light_castshadow[:] = 0
            self.model.mat_reflectance[:] = 0
        self.data = mujoco.MjData(self.model)
        self.robot = RobotInterface(self.model, self.data)
        self.pd = JointPDController(self.robot.to_joint_vector(kp),
                                    self.robot.to_joint_vector(kd),
                                    self.robot.torque_limits)
        self.joint_index = {name: i for i, name in enumerate(self.robot.joint_names)}
        self.joint_ranges = self.model.jnt_range[self.robot.joint_ids].copy()
        self.home_key_id = self.model.key('home').id
        self.reset_simulation()

        # ---------------------------------------------------------------- ROS 인터페이스
        self.joint_state_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.imu_pub = self.create_publisher(Imu, 'imu/data', 10)
        self.clock_pub = self.create_publisher(Clock, 'clock', 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.create_subscription(JointState, 'joint_commands', self.on_command, 10)
        self.shown_command = None   # 화면 표시용: policy_node가 실제로 넣은 명령
        self.create_subscription(Twist, 'policy/command', self.on_policy_command, 10)
        self.terrain_extent = terrain_size if terrain is not None else 4.0
        self.create_service(Trigger, 'sim/reset', self.on_reset)

        dt = self.model.opt.timestep
        self.publish_every = max(1, round((1.0 / dt) / publish_rate))
        self.tf_every = max(1, round((1.0 / dt) / tf_rate))
        self.step_timer = self.create_timer(dt, self.step)
        self.viewer = None
        self.viewer_timer = None

        self.get_logger().info(
            f"시작: {robot_name}, {'공중에 매단(fixed base)' if self.fixed_base else '바닥에 선(floating base)'} 로봇, "
            f"지형 {terrain_kind}, "
            f"관절 {self.robot.num_joints}개, 물리 {1.0 / dt:.0f} Hz, "
            f"/joint_states·/imu/data {1.0 / dt / self.publish_every:.0f} Hz, "
            f"TF {1.0 / dt / self.tf_every:.0f} Hz, "
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
        self.wall_start = time.perf_counter()   # 실제 시간 맞추기의 기준 (이때 시뮬레이션 시간 0)

    def step(self):
        """타이머마다: 실제 시간이 지난 만큼 물리 스텝 (보통 1번, 늦게 불렸으면 밀린 만큼)."""
        dt = self.model.opt.timestep
        due = int((time.perf_counter() - self.wall_start) / dt) - self.step_count
        if due > MAX_CATCH_UP:
            self.wall_start += (due - MAX_CATCH_UP) * dt
            due = MAX_CATCH_UP
        for _ in range(due):
            self.physics_step()

    def physics_step(self):
        """드라이버 PD → 물리 한 스텝 → 상태 발행."""
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
            self.publish_joint_states(stamp)
            self.publish_imu(stamp)
        if self.step_count % self.tf_every == 0:
            self.publish_base_tf(stamp)

    # -------------------------------------------------------------------- 발행
    def publish_joint_states(self, stamp: Time):
        state = self.robot.as_joint_state()
        msg = JointState(name=state['name'], position=state['position'],
                         velocity=state['velocity'], effort=state['effort'])
        msg.header.stamp = stamp
        self.joint_state_pub.publish(msg)

    def publish_imu(self, stamp: Time):
        imu = self.robot.imu()  # IMU site = base_link 원점, 같은 방향
        msg = Imu()
        msg.header.stamp = stamp
        msg.header.frame_id = 'base_link'
        x, y, z, w = quat_wxyz_to_xyzw(imu['quat'])  # ⚠ 순서 변환을 빼먹으면 자세가 엉뚱해짐
        msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w = x, y, z, w
        msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z = imu['gyro']
        msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z = imu['acc']
        # covariance는 0 = "모름" (REP-145). 센서 노이즈 모델은 Phase 4에서 추가
        self.imu_pub.publish(msg)

    def publish_base_tf(self, stamp: Time):
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = \
            self.robot.base_position()
        x, y, z, w = quat_wxyz_to_xyzw(self.robot.base_quaternion())
        t.transform.rotation.x, t.transform.rotation.y = x, y
        t.transform.rotation.z, t.transform.rotation.w = z, w
        self.tf_broadcaster.sendTransform(t)

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

    def on_policy_command(self, msg: Twist):
        self.shown_command = (msg.linear.x, msg.linear.y, msg.angular.z)

    def on_reset(self, request, response):
        self.reset_simulation()
        response.success = True
        response.message = 'home 자세로 초기화했습니다'
        self.get_logger().info(response.message)
        return response

    # -------------------------------------------------------------------- 뷰어
    def attach_viewer(self, viewer):
        self.viewer = viewer
        self.view, self.pending_view = 'follow', 54
        self.viewer_timer = self.create_timer(1.0 / VIEWER_FPS, self.update_viewer)

    def on_viewer_key(self, keycode: int):
        """MuJoCo 창에서 키 (화면 스레드에서 불림): 숫자 6~9면 보기 바꾸기를 예약."""
        if keycode in VIEWS:
            self.pending_view = keycode

    def update_viewer(self):
        with self.viewer.lock():
            if self.pending_view is not None:
                self.view, cam = VIEWS[self.pending_view]
                self.pending_view = None
                if cam['zoom'] is None:   # 지형 전체: 원점 위에서 내려다보기
                    self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                    self.viewer.cam.lookat[:] = (0.0, 0.0, 0.0)
                    self.viewer.cam.distance = 2.6 * self.terrain_extent
                else:
                    track_body_camera(self.robot.base_body_id, distance=cam['zoom'] * self.view_distance)(self.viewer)
                self.viewer.cam.azimuth, self.viewer.cam.elevation = cam['azimuth'], cam['elevation']
            font, grid = mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos
            x, y, _ = self.robot.base_position()
            cmd = ('cmd  vx {:+.2f} m/s  vy {:+.2f} m/s  wz {:+.1f} rad/s   |   '.format(*self.shown_command)
                   if self.shown_command is not None else '')
            self.viewer.set_texts([
                (font, grid.mjGRID_TOPLEFT, f'view: {self.view}', '6 follow  7 top  8 side  9 map'),
                (font, grid.mjGRID_BOTTOMLEFT, f'{cmd}position  x {x:+.2f} m  y {y:+.2f} m', ''),
            ])
        self.viewer.sync()

    def detach_viewer(self):
        if self.viewer_timer is not None:
            self.destroy_timer(self.viewer_timer)
        self.viewer_timer = None
        self.viewer = None


def main(args=None):
    stop = init_with_stop_flag(args)  # Ctrl+C / SIGTERM → stop.set() (giga_sim_ros/shutdown.py)
    node = SimNode()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        if node.use_viewer:
            # 뷰어는 biped_sim.passive_viewer로 열고(닫을 때 그리기 스레드가 끝날 때까지 기다림),
            # ROS 종료보다 먼저 닫는다. launch_passive를 그대로 쓰면 종료 시 세그폴트나
            # GLXBadContext 후 멈춤이 생기는 것을 확인함 (docs/05_ros2_bridge_plan.md §2)
            # 학습한 로봇(오리)은 설정 패널을 숨겨 로봇·글자를 크게 (창에서 Tab / Shift+Tab으로 다시 켬)
            with passive_viewer(node.model, node.data, show_ui=node.robot_name == 'simple_biped',
                                key_callback=node.on_viewer_key) as viewer:
                node.attach_viewer(viewer)
                while not stop.is_set() and viewer.is_running():
                    executor.spin_once(timeout_sec=0.01)
                node.detach_viewer()
        else:
            while not stop.is_set():
                executor.spin_once(timeout_sec=0.05)
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
