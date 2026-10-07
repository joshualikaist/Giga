"""URDF를 RViz로 보기 — 물리 시뮬레이션 없음 (로봇 모델이 맞게 생겼는지 확인하는 용도).

    ros2 launch giga_description display.launch.py              # 관절 슬라이더 창 + RViz
    ros2 launch giga_description display.launch.py gui:=false   # 슬라이더 없이 (모든 관절 0 rad)

실행되는 노드
    robot_state_publisher      URDF + /joint_states → 모든 링크의 TF 계산·발행, /robot_description 발행
    joint_state_publisher_gui  관절마다 슬라이더 → /joint_states 발행 (gui:=false면 슬라이더 없는 버전)
    rviz2                      /robot_description과 TF를 받아 3D로 그림

주의: 이 launch 파일은 /usr/bin/python3로 실행되므로 mujoco나 biped_sim을 import하면 안 된다.
      (docs/05_ros2_bridge_plan.md §2) URDF는 ROS 표준 방식(패키지 share 폴더)으로 찾는다.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('giga_description')
    with open(os.path.join(share, 'urdf', 'simple_biped.urdf')) as f:
        robot_description = f.read()

    gui = LaunchConfiguration('gui')
    rviz = LaunchConfiguration('rviz')

    return LaunchDescription([
        DeclareLaunchArgument('gui', default_value='true',
                              description='관절 슬라이더 창(joint_state_publisher_gui) 표시'),
        DeclareLaunchArgument('rviz', default_value='true', description='RViz 실행'),

        Node(package='robot_state_publisher', executable='robot_state_publisher',
             parameters=[{'robot_description': robot_description}]),
        Node(package='joint_state_publisher_gui', executable='joint_state_publisher_gui',
             condition=IfCondition(gui)),
        Node(package='joint_state_publisher', executable='joint_state_publisher',
             condition=UnlessCondition(gui)),
        # display.rviz: 고정 좌표계 = base_link (몸통이 화면 중심에 고정되고 다리가 그 아래로)
        Node(package='rviz2', executable='rviz2', condition=IfCondition(rviz),
             arguments=['-d', os.path.join(share, 'rviz', 'display.rviz')]),
    ])
