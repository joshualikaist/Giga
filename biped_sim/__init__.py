"""biped_sim — MuJoCo 기반 2족 보행 로봇 시뮬레이터 (교육용 baseline).

모듈 구성 (의존 방향: 위 → 아래)
    robot_configs : 로봇별 설정 (URDF 경로, 기본 자세, 게인). SIMPLE_BIPED, OPEN_DUCK_MINI(외부 오픈소스)
    runner        : 시뮬레이션 루프 (뷰어/헤드리스), 안전한 뷰어 열기/닫기, 스냅샷 저장
    controllers   : 관절 PD 제어기
    robot         : 관절 '이름' 기반 상태 읽기/토크 쓰기 (ROS2 JointState와 1:1 대응)
    builder       : URDF → MuJoCo 모델 (freejoint, 바닥, 모터, IMU 추가)
    utils         : 쿼터니언 변환, 지면 높이 계산
    paths         : 파일 경로 상수
    envs          : 강화학습 환경 (balance: CPU/Gymnasium, walk_mjx: GPU/MJX)
"""
from . import paths
from .builder import SimConfig, build_robot_model, build_robot_spec, export_mjcf
from .controllers import JointPDController
from .robot import RobotInterface
from .robot_configs import OPEN_DUCK_MINI, SIMPLE_BIPED, RobotConfig
from .runner import add_common_args, passive_viewer, save_snapshot, simulate, track_body_camera

__all__ = [
    "paths",
    "SimConfig", "build_robot_model", "build_robot_spec", "export_mjcf",
    "JointPDController",
    "RobotInterface",
    "RobotConfig", "SIMPLE_BIPED", "OPEN_DUCK_MINI",
    "add_common_args", "passive_viewer", "save_snapshot", "simulate", "track_body_camera",
]
