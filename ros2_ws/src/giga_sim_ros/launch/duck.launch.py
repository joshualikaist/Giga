"""학습한 오리 보행 정책을 ROS2로: MuJoCo 오리(sim_node) + 정책(policy_node). 조종은 /cmd_vel (docs/11).

    ros2 launch giga_sim_ros duck.launch.py                          # 평지, 저장소의 명령 따라 걷기 정책
    ros2 launch giga_sim_ros duck.launch.py terrain:=mixed           # 언덕·경사로·장애물 (원점 주변은 평평)
    ros2 launch giga_sim_ros duck.launch.py policy:=output/learning/<실행>/policy.npz
    다른 터미널에서 조종:  ros2 run giga_sim_ros key_teleop          (방향키, 그 터미널을 클릭해 둔 채로)
    ros2 launch giga_sim_ros duck.launch.py phone:=true               # + 핸드폰 조이스틱 (출력에 나오는 주소를 핸드폰 브라우저로)
    ros2 launch giga_sim_ros duck.launch.py cmd_timeout:=0            # 키를 누를 때만 보내는 조종기용 (teleop_twist_keyboard)

실행되는 노드
    sim_node     MuJoCo 오리 (가짜 하드웨어): 학습과 같은 서보 모델·서 있는 자세 — 정책의 학습 설정(.json)에서 읽음
    policy_node  /imu/data + /joint_states + /cmd_vel → 50 Hz /joint_commands. 넘어지면 1초 뒤 처음 자세로
    phone_joystick (phone:=true) 웹 페이지 조이스틱 → /cmd_vel. 스틱 끝 = 정책의 학습 명령 범위

종료: 이 터미널에서 Ctrl+C, 또는 MuJoCo 창 닫기.
주의: launch 파일은 /usr/bin/python3로 실행되므로 mujoco/biped_sim을 import하지 않는다 (설정은 json으로만 읽음).
"""
import json
import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, Shutdown
from launch_ros.actions import Node


def repo_root() -> Path:
    """이 launch 파일(src 또는 install 안, 둘 다 저장소 안)에서 위로 올라가며 biped_sim 폴더가 있는 곳."""
    for parent in Path(__file__).resolve().parents:
        if (parent / 'biped_sim').is_dir():
            return parent
    return Path.cwd()


def resolve_policy(path: str) -> Path:
    p = Path(os.path.expanduser(path))
    if p.is_absolute() or p.exists():
        return p.resolve()
    return repo_root() / p


def launch_setup(context):
    arg = {name: context.launch_configurations[name]
           for name in ('policy', 'terrain', 'terrain_difficulty', 'terrain_seed', 'viewer', 'auto_reset', 'cmd_timeout',
                        'phone', 'port')}
    policy = resolve_policy(arg['policy'])
    config_path = policy.with_suffix('.json')
    if not policy.exists() or not config_path.exists():
        raise FileNotFoundError(f'정책 파일 {policy} 또는 학습 설정 {config_path}가 없습니다. Brax 정책(.pkl)이면 먼저 '
                                f'python learning/export_policy.py --params <정책.pkl> 로 .npz를 만드세요')
    config = json.loads(config_path.read_text())
    as_bool = {'true': True, 'false': False}
    sim_time = {'use_sim_time': True}

    nodes = [
        Node(package='giga_sim_ros', executable='sim_node', output='screen', on_exit=Shutdown(),
             parameters=[{
                 'robot': config.get('robot', 'simple_biped'),
                 'servo': config.get('servo', 'ideal'),
                 'home_knee': float(config.get('home_knee') or 0.0),
                 'terrain': arg['terrain'],
                 'terrain_difficulty': float(arg['terrain_difficulty']),
                 'terrain_seed': int(arg['terrain_seed']),
                 'viewer': as_bool[arg['viewer']],
                 'command_timeout': 0.1,   # 정책이 0.1초 동안 명령을 안 보내면 감쇠 모드 (실물 드라이버의 안전 정지)
             }]),
        Node(package='giga_sim_ros', executable='policy_node', output='screen',
             parameters=[{'policy': str(policy), 'auto_reset': as_bool[arg['auto_reset']],
                          'cmd_timeout': float(arg['cmd_timeout']), **sim_time}]),
    ]
    if as_bool[arg['phone']]:
        rng = config.get('commands') or {}
        limits = {'vx_min': rng.get('vx', [0.0, 0.25])[0], 'vx_max': rng.get('vx', [0.0, 0.25])[1],
                  'vy_max': rng.get('vy', [0.0, 0.0])[1], 'wz_max': rng.get('wz', [0.0, 0.0])[1]}
        nodes.append(Node(package='giga_sim_ros', executable='phone_joystick', output='screen',
                          parameters=[{'port': int(arg['port']), **{k: float(v) for k, v in limits.items()}}]))
    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('policy', default_value='learning/pretrained/duck_cmd_policy.npz',
                              description='numpy 정책 (.npz, 같은 이름 .json = 학습 설정). 저장소 기준 상대 경로 가능'),
        DeclareLaunchArgument('terrain', default_value='flat', description='flat | hills | slopes | obstacles | mixed'),
        DeclareLaunchArgument('terrain_difficulty', default_value='1.0', description='험한 정도 (평가 시험은 1.5, 2.0)'),
        DeclareLaunchArgument('terrain_seed', default_value='0', description='지형 무작위 시드'),
        DeclareLaunchArgument('viewer', default_value='true', description='MuJoCo 뷰어 창'),
        DeclareLaunchArgument('auto_reset', default_value='true', description='넘어지면 1초 뒤 처음 자세로'),
        DeclareLaunchArgument('phone', default_value='false', description='핸드폰 조이스틱 웹 페이지도 띄움'),
        DeclareLaunchArgument('port', default_value='8080', description='핸드폰 조이스틱 웹 페이지 포트'),
        DeclareLaunchArgument('cmd_timeout', default_value='0.5',
                              description='[s] /cmd_vel이 이만큼 끊기면 멈춤. 0 = 마지막 명령 유지 (키보드 조종)'),
        OpaqueFunction(function=launch_setup),
    ])
