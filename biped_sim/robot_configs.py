"""로봇별 설정 (URDF 경로, 기본 자세, 제어 게인, 발 링크 이름).

새 로봇(예: 실제 제작 중인 로봇의 URDF)을 추가할 때는 이 파일에 RobotConfig 하나만 더 만들면
튜토리얼/테스트 코드를 거의 그대로 재사용할 수 있습니다.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import paths


@dataclass(frozen=True)
class RobotConfig:
    name: str
    urdf: Path
    home_pose: dict[str, float]   # 관절 이름 → 각도 [rad]
    kp: dict[str, float]          # 관절 이름 → P 게인 [N·m/rad]
    kd: dict[str, float]          # 관절 이름 → D 게인 [N·m·s/rad]
    foot_bodies: tuple[str, ...]  # 접촉력을 측정할 발 링크 이름


# 무릎을 살짝 굽힌 자세: hip = -a, knee = 2a, ankle = -a → 발바닥 수평, 발목이 고관절 바로 아래
_A = 0.4

SIMPLE_BIPED = RobotConfig(
    name="simple_biped",
    urdf=paths.SIMPLE_BIPED_URDF,
    home_pose={
        "left_hip_pitch": -_A, "left_knee": 2 * _A, "left_ankle_pitch": -_A,
        "right_hip_pitch": -_A, "right_knee": 2 * _A, "right_ankle_pitch": -_A,
    },
    # 게인 근거: 서 있는 로봇은 발목을 축으로 한 역진자. 넘어지려는 중력 강성 ≈ m·g·h
    #   = 8.2 kg × 9.81 × 0.41 m(home 자세 무게중심 높이) ≈ 33 N·m/rad.
    #   두 발목 강성 합(2×Kp)이 이보다 충분히 커야 한다. (무릎·고관절도 직렬로 휘므로 여유 필요)
    #   발목 Kp=30이면 가만히는 서 있지만 20 N 밀기에 넘어짐 → 80으로 상향 (t06에서 직접 실험해 볼 것)
    kp={
        "left_hip_pitch": 100.0, "left_knee": 100.0, "left_ankle_pitch": 80.0,
        "right_hip_pitch": 100.0, "right_knee": 100.0, "right_ankle_pitch": 80.0,
    },
    kd={
        "left_hip_pitch": 4.0, "left_knee": 4.0, "left_ankle_pitch": 2.0,
        "right_hip_pitch": 4.0, "right_knee": 4.0, "right_ankle_pitch": 2.0,
    },
    foot_bodies=("left_foot", "right_foot"),
)
