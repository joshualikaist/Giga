"""시뮬레이터 + ROS2 도구를 한 번에 실행.

    ros2 launch giga_sim_ros sim.launch.py                        # 바닥에 선 로봇 + MuJoCo 뷰어 + RViz
    ros2 launch giga_sim_ros sim.launch.py controller:=squat      # + 앉았다 일어서기 제어기
    ros2 launch giga_sim_ros sim.launch.py controller:=stand      # + 서 있기 제어기
    ros2 launch giga_sim_ros sim.launch.py fixed_base:=true       # 공중에 매단 로봇
    ros2 launch giga_sim_ros sim.launch.py viewer:=false rviz:=false   # 화면 없이

실행되는 노드
    sim_node               MuJoCo 로봇 (가짜 하드웨어): /joint_states, /imu/data, /tf(odom→base_link), /clock
    robot_state_publisher  URDF + /joint_states → 다리 링크들의 TF
    rviz2                  (rviz:=true) ROS가 알고 있는 로봇 상태를 3D로 표시
    demo_controller        (controller:=stand|squat) /joint_commands 발행

sim_node를 뺀 모든 노드는 use_sim_time:=true → 시뮬레이션 시간(/clock)을 기준으로 동작한다.
종료: 이 터미널에서 Ctrl+C, 또는 MuJoCo 창 닫기 → 모든 노드가 함께 종료된다.

주의: launch 파일은 /usr/bin/python3로 실행되므로 mujoco/biped_sim을 import하지 않는다 (docs/05 §2).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, Shutdown
from launch_ros.actions import Node

CONTROLLERS = ('none', 'stand', 'squat')


def launch_setup(context):
    arg = {name: context.launch_configurations[name]
           for name in ('fixed_base', 'viewer', 'rviz', 'controller')}
    if arg['controller'] not in CONTROLLERS:
        raise ValueError(f"controller는 {CONTROLLERS} 중 하나: {arg['controller']!r}")
    use_controller = arg['controller'] != 'none'
    as_bool = {'true': True, 'false': False}

    share = get_package_share_directory('giga_description')
    with open(os.path.join(share, 'urdf', 'simple_biped.urdf')) as f:
        robot_description = f.read()

    nodes = [
        # MuJoCo 창을 닫는 등 sim_node가 끝나면 launch 전체(RViz, 제어기 포함)를 함께 끝낸다
        Node(package='giga_sim_ros', executable='sim_node', output='screen', on_exit=Shutdown(),
             parameters=[{
                 'fixed_base': as_bool[arg['fixed_base']],
                 'viewer': as_bool[arg['viewer']],
                 # 제어기 노드가 붙으면 안전장치를 켠다: 50 ms 동안 명령이 끊기면 감쇠 모드
                 'command_timeout': 0.05 if use_controller else 0.0,
             }]),
        Node(package='robot_state_publisher', executable='robot_state_publisher',
             parameters=[{'robot_description': robot_description, 'use_sim_time': True}]),
    ]
    if as_bool[arg['rviz']]:
        # sim.rviz: 고정 좌표계 = odom (바닥). 로봇이 넘어지거나 움직이는 것이 보임
        nodes.append(Node(package='rviz2', executable='rviz2',
                          arguments=['-d', os.path.join(share, 'rviz', 'sim.rviz')],
                          parameters=[{'use_sim_time': True}]))
    if use_controller:
        nodes.append(Node(package='giga_sim_ros', executable='demo_controller', output='screen',
                          parameters=[{'motion': arg['controller'], 'use_sim_time': True}]))
    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('fixed_base', default_value='false', description='true = 공중에 매단 로봇'),
        DeclareLaunchArgument('viewer', default_value='true', description='MuJoCo 뷰어 창'),
        DeclareLaunchArgument('rviz', default_value='true', description='RViz 창'),
        DeclareLaunchArgument('controller', default_value='none',
                              description='none | stand | squat (demo_controller 실행)'),
        OpaqueFunction(function=launch_setup),
    ])
