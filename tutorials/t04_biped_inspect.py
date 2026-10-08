#!/usr/bin/env python3
"""t04 — 2족 로봇 URDF 해부: URDF 그대로 vs 시뮬레이션용 모델

학습 목표
    1. URDF만으로는 2족 로봇이 "공중에 용접된 인형"이라는 것을 확인한다 (freejoint·모터 없음).
    2. 빌더가 추가한 것(freejoint, 모터, IMU, 바닥, 충돌 제외, home 키프레임)을 표로 확인한다.
    3. 관절별 qpos/qvel/ctrl 인덱스를 확인한다 (ROS2 연동 때 그대로 쓰임).
    4. 최종 MJCF를 output/에 저장해 눈으로 읽어 본다.

실행
    python tutorials/t04_biped_inspect.py --headless    # 표 출력 + MJCF 내보내기
    python tutorials/t04_biped_inspect.py               # 뷰어: 매달린 로봇을 슬라이더로 직접 구동
    python tutorials/t04_biped_inspect.py --headless --snapshot   # home 자세 이미지 저장

뷰어 사용법 (이 튜토리얼은 MuJoCo 기본 뷰어 'launch'를 씀 → 물리 루프를 뷰어가 직접 돌림)
    - 오른쪽 패널 "Control": 각 모터의 토크 슬라이더 [N·m]. 움직여서 다리를 흔들어 보세요.
    - 왼쪽 패널 "Rendering" > "Group enable"의 Geom 3번: 숨겨진 충돌 형상(주황 반투명) 표시
    - 왼쪽 패널 "Rendering" > "Model Elements"의 Joint, Inertia 등도 켜 보세요.
"""
import argparse
import os

import mujoco
import numpy as np

from biped_sim import SIMPLE_BIPED, SimConfig, build_robot_model, export_mjcf, paths, save_snapshot

def banner(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def print_raw_urdf():
    banner("1. URDF를 그대로 불러오기 (simple_biped.urdf 안의 <mujoco> 태그만 적용됨)")
    m = mujoco.MjModel.from_xml_path(str(SIMPLE_BIPED.urdf))
    print(f"바디 {m.nbody}개: {[m.body(i).name for i in range(m.nbody)]}")
    print(f"관절 {m.njnt}개, nq={m.nq}, nv={m.nv}, 액추에이터 nu={m.nu}")
    n_vis = int(np.sum((m.geom_contype == 0) & (m.geom_conaffinity == 0)))
    print(f"geom {m.ngeom}개 = 시각용 {n_vis}개(group 1) + 충돌용 {m.ngeom - n_vis}개(group 0)")
    print("→ base_link에 관절이 없으므로 몸통이 world에 용접된 상태입니다. nq=6은 다리 관절뿐.")
    print("→ 액추에이터도 0개. 이대로는 서지도, 넘어지지도, 걷지도 못합니다.")


def print_model_tables(model, title):
    banner(title)
    print(f"nq={model.nq}  nv={model.nv}  nu={model.nu}  nsensor={model.nsensor}  "
          f"nkey={model.nkey}  timestep={model.opt.timestep}s  "
          f"integrator={mujoco.mjtIntegrator(model.opt.integrator).name}")

    print("\n[바디 트리]")
    print(f"{'id':>3} {'body':14s} {'parent':12s} {'mass[kg]':>9} {'pos in parent [m]':>24}")
    for i in range(1, model.nbody):
        print(f"{i:3d} {model.body(i).name:14s} {model.body(model.body_parentid[i]).name:12s} "
              f"{model.body_mass[i]:9.3f} {np.array2string(model.body_pos[i], precision=3):>24}")
    print(f"총 질량 = {model.body_subtreemass[1]:.3f} kg")

    print("\n[관절]  ← qposadr/dofadr가 data.qpos/data.qvel에서의 위치")
    print(f"{'joint':18s} {'type':6s} {'qposadr':>7} {'dofadr':>6} {'range [deg]':>17} "
          f"{'damping':>7} {'friction':>8} {'armature':>8}")
    type_names = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}
    for j in range(model.njnt):
        rng = (f"[{np.degrees(model.jnt_range[j][0]):6.1f},{np.degrees(model.jnt_range[j][1]):6.1f}]"
               if model.jnt_limited[j] else "-")
        dof = model.jnt_dofadr[j]
        print(f"{model.joint(j).name:18s} {type_names[int(model.jnt_type[j])]:6s} "
              f"{model.jnt_qposadr[j]:7d} {dof:6d} {rng:>17} {model.dof_damping[dof]:7.2f} "
              f"{model.dof_frictionloss[dof]:8.3f} {model.dof_armature[dof]:8.3f}")

    print("\n[액추에이터]  ← data.ctrl[i]에 토크[N·m]를 쓰면 해당 관절에 적용")
    print(f"{'i':>2} {'actuator':18s} {'→ joint':18s} {'ctrlrange [N·m]':>18}")
    for a in range(model.nu):
        j = model.actuator_trnid[a, 0]
        print(f"{a:2d} {model.actuator(a).name:18s} {model.joint(j).name:18s} "
              f"{np.array2string(model.actuator_ctrlrange[a], precision=1):>18}")

    print("\n[센서]  ← data.sensordata[adr : adr+dim]")
    for s in range(model.nsensor):
        stype = mujoco.mjtSensor(int(model.sensor_type[s])).name  # 정수 → enum 이름
        print(f"  {model.sensor(s).name:10s} type={stype:22s} adr={model.sensor_adr[s]:2d} "
              f"dim={model.sensor_dim[s]}")

    print("\n[충돌 제외 쌍]  ← 관절로 연결된 링크끼리는 충돌 검사 안 함")
    pairs = [(model.body(sig >> 16).name, model.body(sig & 0xFFFF).name)
             for sig in model.exclude_signature]
    for p in pairs:
        print(f"  {p[0]} ↔ {p[1]}")

    if model.nkey:
        key = model.key("home")
        print("\n[home 키프레임 qpos]")
        print(f"  {np.array2string(key.qpos, precision=4, max_line_width=100)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--snapshot", action="store_true", help="home 자세 이미지를 output/에 저장")
    args = parser.parse_args()

    if not args.headless:
        # 매달린(fixed base) 로봇을 MuJoCo 기본 뷰어로 열기. 슬라이더로 토크를 직접 줘 보자.
        model, _ = build_robot_model(SIMPLE_BIPED.urdf, SimConfig(fixed_base=True, base_pos=(0, 0, 1.0)))
        data = mujoco.MjData(model)
        import signal

        import mujoco.viewer as mj_viewer
        # launch()는 메인 스레드에서 화면 루프를 직접 돌려서, 파이썬이 Ctrl+C를 처리할 틈이 없다.
        # Ctrl+C를 운영체제 기본 동작(즉시 종료)으로 돌려 두면 터미널에서도 끌 수 있다.
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        os.environ.setdefault("__GL_SYNC_TO_VBLANK", "1")  # 화면 갱신을 모니터 주기로 제한 (GPU 과다 사용 방지)
        mj_viewer.launch(model, data)  # 창을 닫거나 Ctrl+C를 누를 때까지 반환하지 않음
        return

    print_raw_urdf()

    floating, floating_spec = build_robot_model(
        SIMPLE_BIPED.urdf, SimConfig(fixed_base=False, home_joint_pos=SIMPLE_BIPED.home_pose))
    print_model_tables(floating, "2. 빌더로 만든 시뮬레이션 모델 (floating base: 바닥에 서는 모드)")
    print("\n→ freejoint 'floating_base'가 qpos 0~6 (위치 3 + 쿼터니언 4), qvel 0~5를 차지하고,")
    print("  다리 관절은 qpos 7~12, qvel 6~11 입니다. 관절 번호와 qpos 번호가 다르다는 점에 주의!")

    fixed, fixed_spec = build_robot_model(
        SIMPLE_BIPED.urdf, SimConfig(fixed_base=True, home_joint_pos=SIMPLE_BIPED.home_pose))
    banner("3. 비교: fixed base (매달기 모드)")
    print(f"nq={fixed.nq}, nv={fixed.nv}  → freejoint가 없으니 다리 관절 6개뿐 (qpos 0~5)")

    banner("4. 최종 MJCF 내보내기 — 빌더가 실제로 무엇을 만들었는지 직접 읽어 보세요")
    for name, spec in (("simple_biped_floating.xml", floating_spec), ("simple_biped_fixed.xml", fixed_spec)):
        path = export_mjcf(spec, paths.output_path(name))
        print(f"  저장: {path.relative_to(paths.REPO_ROOT)}")
    print("  내보낸 파일은 그대로 뷰어로 열 수 있습니다:")
    print("    python -m mujoco.viewer --mjcf=output/simple_biped_floating.xml")

    if args.snapshot:
        data = mujoco.MjData(floating)
        mujoco.mj_resetDataKeyframe(floating, data, floating.key("home").id)
        mujoco.mj_forward(floating, data)
        path = save_snapshot(floating, data, paths.output_path("t04_biped_home.png"),
                             lookat=(0, 0, 0.35), distance=1.6, azimuth=150, elevation=-10)
        print(f"\n스냅샷 저장: {path.relative_to(paths.REPO_ROOT)}")


if __name__ == "__main__":
    main()
