#!/usr/bin/env python3
"""t07 — 남이 만든 오픈소스 로봇 불러오기: Open Duck Mini v2 (오리 모양 2족 로봇)

인터넷에서 받은 URDF는 "그대로 시뮬레이션에 넣으면" 거의 항상 문제가 있습니다.
이 튜토리얼은 실제 오픈소스 로봇 하나로 문제를 하나씩 확인하고, biped_sim이 그것을 어떻게 보완하는지 봅니다.
받은 파일은 고치지 않습니다. 보완은 모두 불러올 때(RobotConfig.sim_defaults) 합니다.

학습 목표
    1. 받은 URDF를 그대로 불러올 때 생기는 문제를 직접 본다: 메시 경로, 몸통 고정, 모터 없음, 가짜 접촉, 토크 한계
    2. 빌더 옵션(mesh_dir, effort_limits, collision_bodies ...)으로 보완하고 전후를 비교한다
    3. 좌우 관절의 부호 규칙을 순운동학(FK)으로 직접 확인한다 (관절 이름만 보고 짐작하면 틀림)
    4. 관절 PD로 바닥에 세우고, 두 발 하중의 합 = 무게인지 검증한다 (t06과 같은 방법)

준비 (처음 한 번, 약 19 MB)
    bash scripts/get_open_duck.sh

실행
    python tutorials/t07_open_duck.py --headless       # 1~4 전부 출력
    python tutorials/t07_open_duck.py                  # 뷰어: PD로 서 있는 오리 (Ctrl+오른쪽 드래그로 밀어 보기)
    python tutorials/t07_open_duck.py --hang           # 뷰어: 공중에 매단 오리
    python tutorials/t07_open_duck.py --headless --snapshot   # 서 있는 모습 이미지 저장

자세한 설명: docs/09_open_source_robot.md
출처: https://github.com/apirrone/Open_Duck_Mini (Apache-2.0) — 디즈니 BD-X 드로이드를 본뜬 팬 프로젝트
"""
import argparse

import mujoco
import numpy as np

from biped_sim import (OPEN_DUCK_MINI, JointPDController, RobotInterface, add_common_args, build_robot_model,
                       paths, save_snapshot, simulate, track_body_camera)

CFG = OPEN_DUCK_MINI


def banner(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def contact_count(model):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return data.ncon


def raw_import():
    banner("1. URDF를 그대로 불러오기")
    try:
        mujoco.MjModel.from_xml_path(str(CFG.urdf))
    except ValueError as e:
        print(f"실패: {str(e).splitlines()[0]}")
        print("→ 메시 경로가 'package:///이름.stl' 형식이라 MuJoCo가 파일을 못 찾습니다.")
    spec = mujoco.MjSpec.from_file(str(CFG.urdf))
    spec.meshdir, spec.strippath = str(CFG.urdf.parent), True   # 메시 폴더만 알려 주면 열림
    m = spec.compile()
    n_coll = int(np.sum((m.geom_contype > 0) | (m.geom_conaffinity > 0)))
    print(f"메시 폴더만 지정하면 열림: 관절 {m.njnt}개, nq={m.nq}, 모터 nu={m.nu}, 질량 {m.body_subtreemass[0]:.2f} kg")
    print("  - 몸통 freejoint 없음 (nq = 관절 수) → 공중에 용접된 상태, 모터 0개 (t04와 같은 문제)")
    print(f"  - 충돌 형상 {n_coll}개가 전부 CAD 메시, 꼭짓점 {m.mesh_vertnum.sum():,}개 → 계산이 무겁다 (특히 GPU 학습)")
    print(f"  - 처음 자세에서 접촉 {contact_count(m)}개: 붙어 있는 부품끼리 겹친 '가짜 접촉'")
    print(f"  - URDF 토크 한계(effort) = {m.jnt_actfrcrange[0, 1]:g} N·m: CAD 내보내기 기본값. 실제 서보는 약 3.2 N·m")


def builder_import():
    banner("2. biped_sim 빌더로 불러오기 (RobotConfig.sim_defaults로 보완)")
    for key, value in CFG.sim_defaults.items():
        print(f"  {key:20s} = {value}")
    model, _ = build_robot_model(CFG.urdf, CFG.sim_config(fixed_base=False))
    n_coll = [model.geom(g).name or model.body(model.geom_bodyid[g]).name
              for g in range(model.ngeom) if model.geom_contype[g] and model.geom_bodyid[g] != 0]
    print(f"결과: nq={model.nq} (freejoint 7 + 관절 {model.nq - 7}), 모터 nu={model.nu}, "
          f"토크 한계 ±{model.actuator_ctrlrange[0, 1]:g} N·m, 로봇 충돌 형상 {len(n_coll)}개 (발만), "
          f"home 자세 접촉 {contact_count(model)}개")
    return model


def sign_table():
    banner("3. 좌우 관절 부호 규칙 (매단 상태에서 관절 하나만 +0.3 rad 돌려 보기)")
    model, _ = build_robot_model(CFG.urdf, CFG.sim_config(fixed_base=True))
    data = mujoco.MjData(model)
    print(f"{'관절':10s} {'왼발 앞뒤 이동':>14s} {'오른발 앞뒤 이동':>16s}   좌우 부호")
    for joint in ("hip_pitch", "knee", "ankle"):
        moves = []
        for side in ("left", "right"):
            foot = model.body(f"{side}_foot").id
            data.qpos[:] = 0
            mujoco.mj_forward(model, data)
            x0 = data.xpos[foot, 0]
            data.qpos[model.joint(f"{side}_{joint}").qposadr[0]] = 0.3
            mujoco.mj_forward(model, data)
            moves.append(data.xpos[foot, 0] - x0)
        same = np.sign(moves[0]) == np.sign(moves[1])
        print(f"{joint:10s} {100 * moves[0]:+12.1f} cm {100 * moves[1]:+14.1f} cm   {'같음' if same else '반대 ⚠'}")
    print("→ hip_pitch만 좌우 부호가 반대. home 자세도, 좌우 대칭 보상(walk_mjx의 symmetry)도 이 규칙을 따라야 한다.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--hang", action="store_true", help="몸통을 공중에 고정 (fixed base)")
    args = parser.parse_args()
    if not CFG.urdf.exists():
        raise SystemExit(f"Open Duck Mini 파일이 없습니다: {CFG.urdf}\n→ 먼저 bash scripts/get_open_duck.sh")

    if args.headless:
        raw_import()
        builder_import()
        sign_table()
        banner("4. 관절 PD로 바닥에 세우기 (t06과 같은 검증)")

    model, _ = build_robot_model(CFG.urdf, CFG.sim_config(fixed_base=args.hang, base_pos=(0, 0, 0.45)))
    data = mujoco.MjData(model)
    robot = RobotInterface(model, data)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    q_home = robot.to_joint_vector(CFG.home_pose)
    pd = JointPDController(robot.to_joint_vector(CFG.kp), robot.to_joint_vector(CFG.kd), robot.torque_limits)
    weight = robot.total_mass * -model.opt.gravity[2]
    max_tau = [0.0]

    def control_step(m, d):
        tau = pd.compute(robot.joint_positions(), robot.joint_velocities(), q_home)
        max_tau[0] = max(max_tau[0], float(np.abs(tau).max()))
        robot.set_joint_torques(tau)

    duration = args.duration if args.duration is not None else (5.0 if args.headless else None)
    simulate(model, data, control_step, duration=duration, headless=args.headless,
             viewer_setup=track_body_camera(robot.base_body_id, distance=1.0, azimuth=135, elevation=-15))

    if args.headless and not args.hang:
        left, right = (robot.contact_normal_force(b) for b in CFG.foot_bodies)
        tilt = np.degrees(robot.base_tilt())
        print(f"t={data.time:.1f}s 몸통 높이 {robot.base_position()[2]:.3f} m, 기울기 {tilt:.1f}°, "
              f"발 하중 {left:.2f} + {right:.2f} N = 무게({weight:.2f} N)의 {(left + right) / weight:.3f}배, "
              f"최대 토크 {max_tau[0]:.2f} / {robot.torque_limits[0, 1]:g} N·m")
        print("→ 서 있음 ✅" if tilt < 20 else "→ 넘어짐 ❌")

    if args.snapshot:
        pos = robot.base_position()
        path = save_snapshot(model, data, paths.output_path("t07_open_duck.png"),
                             lookat=(pos[0], pos[1], 0.2), distance=0.9, azimuth=135, elevation=-12)
        print(f"스냅샷 저장: {path.relative_to(paths.REPO_ROOT)}")


if __name__ == "__main__":
    main()
