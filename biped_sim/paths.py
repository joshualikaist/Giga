"""프로젝트 경로 상수.

모든 코드는 "어디서 실행하든" 같은 파일을 찾도록 이 모듈의 경로만 사용합니다.
(상대 경로 "models/..."를 코드에 직접 쓰면, 실행 위치에 따라 파일을 못 찾는 버그가 생깁니다.)
"""
from pathlib import Path

# biped_sim/paths.py 기준으로 한 단계 위 = 저장소 루트
REPO_ROOT = Path(__file__).resolve().parents[1]

MODELS_DIR = REPO_ROOT / "models"
MJCF_DIR = MODELS_DIR / "mjcf"
URDF_DIR = MODELS_DIR / "urdf"
OUTPUT_DIR = REPO_ROOT / "output"  # 생성물(내보낸 MJCF, 그래프, 스냅샷). git에서 제외됨

# 로봇 "설명" ROS2 패키지. 실제 로봇 URDF의 단일 원본은 여기에 둔다 (ROS2·RViz·MuJoCo 공용).
DESCRIPTION_DIR = REPO_ROOT / "ros2_ws" / "src" / "giga_description"

# 개별 모델 파일
FALLING_BOX_XML = MJCF_DIR / "falling_box.xml"                  # 교육용 MJCF 예제
PENDULUM_URDF = URDF_DIR / "pendulum" / "pendulum.urdf"         # 교육용 최소 URDF
SIMPLE_BIPED_URDF = DESCRIPTION_DIR / "urdf" / "simple_biped.urdf"


def output_path(filename: str) -> Path:
    """output/ 아래 파일 경로를 돌려준다 (폴더가 없으면 만든다)."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / filename
