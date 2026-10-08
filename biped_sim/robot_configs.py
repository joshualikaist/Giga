"""로봇별 설정 (URDF 경로, 기본 자세, 제어 게인, 발 링크 이름).

새 로봇(예: 실제 제작 중인 로봇의 URDF)을 추가할 때는 이 파일에 RobotConfig 하나만 더 만들면
튜토리얼/테스트 코드를 거의 그대로 재사용할 수 있습니다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import paths
from .builder import SimConfig


@dataclass(frozen=True)
class RobotConfig:
    name: str
    urdf: Path
    home_pose: dict[str, float]   # 관절 이름 → 각도 [rad]
    kp: dict[str, float]          # 관절 이름 → P 게인 [N·m/rad]
    kd: dict[str, float]          # 관절 이름 → D 게인 [N·m·s/rad]
    foot_bodies: tuple[str, ...]  # 접촉력을 측정할 발 링크 이름
    # 이 로봇에 필요한 SimConfig 기본값 (메시 폴더, 토크 한계 덮어쓰기 등). 외부 URDF를 고치지 않고 불러올 때 쓴다
    sim_defaults: dict[str, Any] = field(default_factory=dict)

    def sim_config(self, **overrides) -> SimConfig:
        """이 로봇용 SimConfig: home 키프레임 + sim_defaults + 직접 준 값(우선).
            build_robot_model(cfg.urdf, cfg.sim_config(fixed_base=True))"""
        return SimConfig(**{"home_joint_pos": self.home_pose, **self.sim_defaults, **overrides})


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


# ---------------------------------------------------------------------------- Open Duck Mini v2 (외부 오픈소스)
# 디즈니 BD-X 드로이드를 본뜬 팬 프로젝트 https://github.com/apirrone/Open_Duck_Mini (Apache-2.0).
# 파일은 저장소에 넣지 않고 내려받는다: bash scripts/get_open_duck.sh → models/third_party/open_duck_mini_v2/
# 받은 URDF는 고치지 않고, 불러올 때 아래 sim_defaults로 보완한다. 확인 과정은 docs/09_open_source_robot.md.
OPEN_DUCK_DIR = paths.THIRD_PARTY_DIR / "open_duck_mini_v2"
_DUCK_LEG = ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle")
_DUCK_HEAD = ("neck_pitch", "head_pitch", "head_yaw", "head_roll", "left_antenna", "right_antenna")
_DUCK_FEET = ("foot_assembly", "foot_assembly_2")   # 발 링크 (URDF의 left_foot/right_foot은 질량 없는 기준점)
# 서 있는 자세: 무릎을 새처럼 뒤로 0.8 rad 굽히고, 발바닥이 수평·발이 엉덩이 아래에 오도록 hip/ankle = −0.4.
# ⚠ hip_pitch만 좌우 부호가 반대 (같은 동작이 왼쪽 −, 오른쪽 +). knee·ankle은 좌우 부호가 같다 (FK로 확인, docs/09)
_D = 0.4
OPEN_DUCK_MINI = RobotConfig(
    name="open_duck_mini_v2",
    urdf=OPEN_DUCK_DIR / "robot.urdf",
    home_pose={
        **{f"left_{j}": v for j, v in zip(_DUCK_LEG, (0.0, 0.0, -_D, 2 * _D, -_D))},
        **{f"right_{j}": v for j, v in zip(_DUCK_LEG, (0.0, 0.0, +_D, 2 * _D, -_D))},
        **{j: 0.0 for j in _DUCK_HEAD},
    },
    # 서보(Feetech STS3215)는 내부에서 위치 PD를 한다. 같은 동작을 토크 모터 + 바깥 PD로 흉내 낸다.
    # kp 9.5 = Open Duck 저장소 MuJoCo 모델(robot.xml)의 위치 서보 kp. kd는 서 있기 실험으로 정함 (docs/09)
    kp={j: 9.5 for j in (*[f"{s}_{j}" for s in ("left", "right") for j in _DUCK_LEG], *_DUCK_HEAD)},
    kd={j: 0.5 for j in (*[f"{s}_{j}" for s in ("left", "right") for j in _DUCK_LEG], *_DUCK_HEAD)},
    foot_bodies=_DUCK_FEET,
    sim_defaults=dict(
        mesh_dir=OPEN_DUCK_DIR,          # URDF 메시 경로가 package:///이름.stl
        effort_limits=3.23,              # [N·m] URDF는 effort=1(내보내기 기본값). Open Duck 프로젝트가 측정한 STS3215(7.4 V) 값
        joint_armature=0.027,            # 서보 감속기 관성·마찰: Open Duck 저장소 robot_motors.xml 값
        joint_frictionloss=0.083,
        joint_damping=0.0,
        collision_bodies=_DUCK_FEET,     # CAD 메시 131개 대신 발만 충돌 (부품끼리 가짜 접촉 154개 → 0)
    ),
)
