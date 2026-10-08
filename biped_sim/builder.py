"""URDF → MuJoCo 시뮬레이션 모델 빌더.

왜 이 단계가 필요한가?
    URDF는 "로봇의 생김새(기구학 + 관성)"만 기술합니다. 시뮬레이션에 필요한 나머지는 없습니다.

        URDF에 있는 것                     URDF에 없는 것 (이 모듈이 추가)
        ─────────────────────────         ─────────────────────────────────────
        링크 질량/관성, 형상               바닥, 조명, 하늘
        관절 종류/축/범위/최대토크         모터(액추에이터)  ← 없으면 로봇에 힘을 줄 수 없음
        관절 감쇠/마찰                     떠 있는 베이스(freejoint) ← 없으면 몸통이 공중에 용접됨
                                           IMU 센서, 모터 회전자 관성(armature), 적분기 설정

    그래서 "URDF = 로봇 설명의 단일 원본(ROS2와 공유)", "MJCF = 시뮬레이션용 파생물"로 나누고,
    그 변환을 이 파일 한 곳에서만 합니다. 결과 MJCF는 export_mjcf()로 저장해 눈으로 확인할 수 있습니다.

검증된 MuJoCo 3.15 URDF 변환 동작은 docs/03_urdf_and_mujoco.md 참고.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco

from .utils import lowest_collision_z

FREEJOINT_NAME = "floating_base"
IMU_SITE_NAME = "imu"
IMU_SENSORS = {  # 센서 이름 → MuJoCo 센서 타입
    "imu_quat": mujoco.mjtSensor.mjSENS_FRAMEQUAT,       # 자세 (w,x,y,z), world 기준
    "imu_gyro": mujoco.mjtSensor.mjSENS_GYRO,            # 각속도 [rad/s], IMU 좌표계
    "imu_acc": mujoco.mjtSensor.mjSENS_ACCELEROMETER,    # 가속도 [m/s²], IMU 좌표계 (중력 포함!)
}
COLLISION_GROUP = 3  # 뷰어는 기본적으로 geom 그룹 0,1,2만 그림 → 충돌 형상은 3번에 숨김

INTEGRATORS = {
    "euler": mujoco.mjtIntegrator.mjINT_EULER,
    "rk4": mujoco.mjtIntegrator.mjINT_RK4,
    "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT,
    "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
}


@dataclass
class SimConfig:
    """시뮬레이션 모델을 만들 때의 선택 사항."""

    # True : 몸통을 공중(base_pos)에 고정 → 넘어지지 않음. 관절/제어기 디버깅용 ("매달기")
    # False: 몸통에 freejoint(6자유도) 추가 → 중력으로 떨어지고, 바닥에 서고, 넘어질 수 있음
    fixed_base: bool = False
    base_pos: tuple[float, float, float] = (0.0, 0.0, 1.0)

    timestep: float = 0.002            # [s] 0.002 = 500 Hz 물리 스텝
    integrator: str = "implicitfast"   # euler / rk4 / implicit / implicitfast

    # 모터 회전자 관성 [kg·m²]. 실제 감속기 모터는 (회전자 관성 × 감속비²)만큼 관절이 "무거워" 보임.
    # URDF에는 이 항목이 없으며, 가벼운 링크(발 등)에서 PD 제어를 수치적으로 안정하게 해 줌.
    joint_armature: float = 0.01
    joint_damping: float | None = None       # 관절 점성 감쇠 [N·m·s/rad]. None = URDF <dynamics> 값 그대로
    joint_frictionloss: float | None = None  # 관절 건마찰 [N·m]. None = URDF 값 그대로
    # 모터 토크 한계 덮어쓰기 [N·m]: 숫자 하나(모든 관절) 또는 {관절 이름: 한계}. None = URDF <limit effort>.
    # 오픈소스 URDF의 effort는 CAD 내보내기 기본값(예: 1)인 경우가 많아 실제 모터 사양으로 바꿔야 함 (docs/09)
    effort_limits: float | dict[str, float] | None = None

    # URDF의 메시 경로가 package://... 이면 MuJoCo가 파일을 못 찾는다 → 경로를 떼고 파일 이름만 이 폴더에서 찾음
    mesh_dir: str | Path | None = None
    # 이 바디들만 충돌 계산 (예: 발). None = 전부. CAD 메시를 그대로 충돌시키면 느리고 부품끼리 가짜 접촉이 생김
    collision_bodies: tuple[str, ...] | None = None

    add_floor: bool = True
    add_imu: bool = True
    hide_collision_geoms: bool = True  # visual geom이 있을 때 충돌 geom을 그룹 3으로 숨김
    exclude_parent_child_contacts: bool = True  # 관절로 이어진 두 링크끼리는 충돌 검사 안 함

    # 지정하면 "home" 키프레임을 만든다. {관절 이름: 각도[rad]}
    home_joint_pos: dict[str, float] | None = None
    home_on_ground: bool = True        # home 자세에서 발바닥이 바닥에 닿도록 베이스 높이 자동 조정
    ground_clearance: float = 0.001    # [m] 바닥과의 여유 간격 (0이면 시작부터 접촉 → 튐 가능)


def build_robot_spec(urdf_path: str | Path, cfg: SimConfig | None = None) -> mujoco.MjSpec:
    """URDF를 읽어 시뮬레이션 요소가 추가된 MjSpec을 돌려준다 (아직 컴파일 전)."""
    cfg = cfg or SimConfig()
    spec = mujoco.MjSpec.from_file(str(urdf_path))
    if cfg.mesh_dir is not None:   # package://pkg/meshes/x.stl → x.stl 을 mesh_dir에서 (docs/03 §3)
        spec.meshdir = str(cfg.mesh_dir)
        spec.strippath = True

    # ① 관절 없는 링크 병합 끄기. 기본값(True)이면 루트 링크(base_link)가 world에 흡수되어
    #    freejoint를 붙일 대상 자체가 사라진다. (URDF의 <mujoco> 태그가 없어도 안전하도록 여기서 강제)
    spec.compiler.fusestatic = False
    # ①-b 충돌하지 않는 형상(시각용)도 남기기. URDF를 읽을 때 MuJoCo 기본값은 discardvisual=True라서,
    #    collision_bodies로 충돌을 끈 몸통·다리 형상이 컴파일 때 통째로 사라진다 (Open Duck Mini에서 확인, docs/09)
    spec.compiler.discardvisual = False

    # ② 물리 옵션
    spec.option.timestep = cfg.timestep
    spec.option.integrator = INTEGRATORS[cfg.integrator]

    # ③ 루트 바디(= URDF 루트 링크) 찾기
    roots = list(spec.worldbody.bodies)
    if len(roots) != 1:
        raise ValueError(f"URDF 루트 링크는 1개여야 합니다. 발견: {[b.name for b in roots]}")
    base = roots[0]
    base.pos = list(cfg.base_pos)
    if not cfg.fixed_base:
        base.add_freejoint(name=FREEJOINT_NAME)

    # ③-b 부모-자식 링크 사이 충돌 제외.
    #    MuJoCo는 기본적으로 부모-자식 충돌을 걸러 주지만(filterparent), "world에 용접된 바디"
    #    (= fixed_base의 몸통, 또는 병합된 루트 링크)는 world로 취급해서 거르지 않는다.
    #    → 허벅지 위쪽 끝이 몸통 박스와 맞닿아 있으면 매달기 모드에서 가짜 접촉이 생겨 관절이 끼임.
    #    CAD에서 뽑은 메쉬도 관절 부위가 겹치는 경우가 많으므로 명시적으로 제외해 둔다. (t03에서 실험)
    if cfg.exclude_parent_child_contacts:
        for body in spec.bodies:
            parent = body.parent
            if body.name == "world" or parent.name == "world":
                continue
            spec.add_exclude(bodyname1=parent.name, bodyname2=body.name)

    # ④ 관절마다 armature + 토크 모터 추가 (URDF의 <transmission>은 MuJoCo가 읽지 않음)
    for joint in spec.joints:
        if joint.type not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            continue  # freejoint 등은 구동하지 않음
        joint.armature = cfg.joint_armature
        if cfg.joint_damping is not None:
            joint.damping = [cfg.joint_damping, 0.0, 0.0]   # MuJoCo 3.15: [선형 감쇠, 비선형 항 2개]
        if cfg.joint_frictionloss is not None:
            joint.frictionloss = cfg.joint_frictionloss
        if cfg.effort_limits is not None:
            limit = (cfg.effort_limits[joint.name] if isinstance(cfg.effort_limits, dict)
                     else float(cfg.effort_limits))
            joint.actfrcrange = [-limit, limit]
        lo, hi = joint.actfrcrange
        if not hi > lo:
            raise ValueError(f"관절 '{joint.name}'에 토크 한계가 없습니다. URDF <limit effort=...>를 지정하세요.")
        act = spec.add_actuator(name=joint.name, target=joint.name,
                                trntype=mujoco.mjtTrn.mjTRN_JOINT)
        # motor = 입력(ctrl)이 그대로 토크 [N·m]. (실제 모터 드라이버의 토크 모드와 같음)
        act.set_to_motor()
        act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.ctrlrange = [lo, hi]

    # ⑤ 충돌 형상 숨기기 (visual이 있을 때만. 없으면 로봇이 통째로 안 보이게 되므로)
    geoms = list(spec.geoms)
    has_visual = any(g.contype == 0 and g.conaffinity == 0 for g in geoms)
    if cfg.hide_collision_geoms and has_visual:
        for g in geoms:
            if g.contype != 0 or g.conaffinity != 0:
                g.group = COLLISION_GROUP
                g.rgba = [0.9, 0.6, 0.1, 0.4]  # 켜서 볼 때 반투명 주황색
    # ⑤-b 지정한 바디만 충돌 (나머지는 보이기만 함)
    if cfg.collision_bodies is not None:
        for g in geoms:
            if g.parent.name not in cfg.collision_bodies:
                g.contype = 0
                g.conaffinity = 0

    # ⑥ IMU: 몸통 원점에 site(=좌표계 표식)를 달고, 그 site를 기준으로 센서를 정의
    if cfg.add_imu:
        base.add_site(name=IMU_SITE_NAME, pos=[0, 0, 0], size=[0.01, 0.01, 0.01])
        for name, stype in IMU_SENSORS.items():
            spec.add_sensor(name=name, type=stype,
                            objtype=mujoco.mjtObj.mjOBJ_SITE, objname=IMU_SITE_NAME)

    # ⑦ 환경: 바닥, 조명, 하늘
    if cfg.add_floor:
        _add_environment(spec)

    # ⑧ home 키프레임 (뷰어의 "Key" 버튼 또는 mj_resetDataKeyframe으로 불러올 수 있음)
    if cfg.home_joint_pos:
        _add_home_keyframe(spec, cfg)

    return spec


def build_robot_model(urdf_path: str | Path, cfg: SimConfig | None = None
                      ) -> tuple[mujoco.MjModel, mujoco.MjSpec]:
    """URDF → (컴파일된 MjModel, 원본 MjSpec)."""
    spec = build_robot_spec(urdf_path, cfg)
    return spec.compile(), spec


def export_mjcf(spec: mujoco.MjSpec, path: str | Path) -> Path:
    """최종 시뮬레이션 모델을 MJCF(XML)로 저장. 빌더가 무엇을 만들었는지 눈으로 확인하는 용도."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(spec.to_xml())
    return path


# ---------------------------------------------------------------------------
# 내부 함수
# ---------------------------------------------------------------------------
def _add_environment(spec: mujoco.MjSpec) -> None:
    spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.2, 0.3, 0.4], rgb2=[0.1, 0.15, 0.2], width=512, height=512)
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.4, 0.6, 0.8], rgb2=[0.0, 0.0, 0.0], width=512, height=3072)
    # texuniform=True면 texrepeat은 "1 m당 반복 횟수". checker 1장 = 2×2칸 → 2회/m면 한 칸 = 25 cm (크기 가늠용)
    floor_mat = spec.add_material(name="grid", texrepeat=[2, 2], texuniform=True, reflectance=0.1)
    floor_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"

    # size 앞 두 값이 0이면 "무한 평면"으로 그림. 충돌은 크기와 무관하게 항상 무한 평면.
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                            size=[0, 0, 0.05], material="grid")
    spec.worldbody.add_light(name="sun", pos=[0, 0, 3], dir=[0, 0, -1],
                             type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                             diffuse=[0.6, 0.6, 0.6], castshadow=True)


def _add_home_keyframe(spec: mujoco.MjSpec, cfg: SimConfig) -> None:
    """home 자세의 전체 qpos를 계산해 키프레임으로 저장."""
    model = spec.compile()
    data = mujoco.MjData(model)
    qpos = model.qpos0.copy()  # freejoint면 앞 7칸 = 베이스 위치(3) + 쿼터니언(4)
    for name, angle in cfg.home_joint_pos.items():
        qpos[model.joint(name).qposadr[0]] = angle
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)

    if not cfg.fixed_base and cfg.home_on_ground and cfg.add_floor:
        free_adr = model.joint(FREEJOINT_NAME).qposadr[0]
        qpos[free_adr + 2] -= lowest_collision_z(model, data) - cfg.ground_clearance

    spec.add_key(name="home", qpos=qpos.tolist())
