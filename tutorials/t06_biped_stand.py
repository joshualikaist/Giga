#!/usr/bin/env python3
"""t06 — 바닥에 세우기: freejoint(떠 있는 베이스) + 접촉 + 관절 PD

t05와 무엇이 다른가?
    몸통에 freejoint(6자유도)가 생겨, 로봇 전체가 중력으로 떨어지고 발바닥으로 바닥을 딛고 섭니다.
    이제 몸통 위치·자세는 "제어할 수 없는(underactuated)" 상태이고, 오직 발과 바닥 사이의
    접촉력을 통해서만 간접적으로 영향을 줄 수 있습니다. 이것이 보행 로봇 제어가 어려운 근본 이유입니다.

학습 목표
    1. home 키프레임으로 초기화하고, 관절 PD로 자세를 유지해 서 있게 한다.
    2. 시뮬레이션이 물리적으로 맞는지 검증: 두 발 접촉 수직력의 합 = 로봇 무게(m·g)
    3. IMU 값의 의미를 이해한다 (정지 상태 가속도계 = +9.81 z).
    4. 제어를 끄거나(--no-control) 몸통을 밀어서(--push) 한계를 관찰한다.

실행
    python tutorials/t06_biped_stand.py                          # 뷰어 (Ctrl+오른쪽 드래그로 직접 밀어 보기)
    python tutorials/t06_biped_stand.py --headless               # 상태 로그 + 판정
    python tutorials/t06_biped_stand.py --headless --no-control  # 토크 0 → 무너짐
    python tutorials/t06_biped_stand.py --headless --push 20     # t=2 s에 앞으로 20 N × 0.1 s
    python tutorials/t06_biped_stand.py --headless --push 40     # 40 N이면?

뷰어에서 몸통 밀기: 몸통을 더블클릭해 선택 → Ctrl + 마우스 오른쪽 드래그
"""
import argparse

import mujoco
import numpy as np

from biped_sim import (SIMPLE_BIPED, JointPDController, RobotInterface, SimConfig, add_common_args,
                       build_robot_model, paths, save_snapshot, simulate, track_body_camera)

PUSH_START, PUSH_DURATION = 2.0, 0.1  # [s]
FALL_HEIGHT = 0.4                     # [m] 몸통이 이보다 낮아지면 넘어진 것으로 판정
FALL_TILT = np.radians(30)            # [rad] 몸통이 수직에서 이보다 많이 기울면 넘어진 것으로 판정


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--no-control", action="store_true", help="모터 토크 0 (제어 없음)")
    parser.add_argument("--push", type=float, default=0.0,
                        help=f"t={PUSH_START}s에 몸통을 +x(앞)로 미는 힘 [N], {PUSH_DURATION}s 동안")
    args = parser.parse_args()
    duration = args.duration if args.duration is not None else (5.0 if args.headless else None)

    # ① 모델: floating base + home 키프레임(발바닥이 바닥 1 mm 위에 오도록 높이 자동 계산)
    cfg = SimConfig(fixed_base=False, home_joint_pos=SIMPLE_BIPED.home_pose)
    model, _ = build_robot_model(SIMPLE_BIPED.urdf, cfg)
    data = mujoco.MjData(model)
    robot = RobotInterface(model, data)

    # ② 초기화: 키프레임의 qpos(몸통 위치·자세 + 관절각)를 data에 복사
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)

    q_home = robot.to_joint_vector(SIMPLE_BIPED.home_pose)
    pd = JointPDController(robot.to_joint_vector(SIMPLE_BIPED.kp),
                           robot.to_joint_vector(SIMPLE_BIPED.kd), robot.torque_limits)
    weight = robot.total_mass * -model.opt.gravity[2]

    print(f"로봇 질량 {robot.total_mass:.2f} kg → 무게 {weight:.2f} N")
    print(f"초기 몸통 높이 {robot.base_position()[2]:.4f} m (home 키프레임)")
    print(f"제어: {'없음 (토크 0)' if args.no_control else '관절 PD로 home 자세 유지'}"
          + (f",  t={PUSH_START}s에 +x 방향 {args.push} N × {PUSH_DURATION}s 밀기" if args.push else ""))

    next_print = [0.0]
    if args.headless:
        print(f"\n{'t[s]':>5} {'base z[m]':>9} {'roll[°]':>8} {'pitch[°]':>8} "
              f"{'L발[N]':>7} {'R발[N]':>7} {'합/무게':>7} {'IMU acc z':>9}")

    def control_step(m, d):
        # 밀기: xfrc_applied = 바디에 직접 가하는 외력 [Fx Fy Fz Tx Ty Tz] (world 좌표계)
        pushing = args.push and PUSH_START <= d.time < PUSH_START + PUSH_DURATION
        d.xfrc_applied[robot.base_body_id, :3] = (args.push, 0.0, 0.0) if pushing else (0.0, 0.0, 0.0)

        if args.no_control:
            robot.set_joint_torques(np.zeros(robot.num_joints))
        else:
            robot.set_joint_torques(pd.compute(robot.joint_positions(), robot.joint_velocities(), q_home))

        if args.headless and d.time >= next_print[0] - 1e-9:
            next_print[0] += 0.25
            left, right = (robot.contact_normal_force(b) for b in SIMPLE_BIPED.foot_bodies)
            roll, pitch, _ = np.degrees(robot.base_rpy())
            print(f"{d.time:5.2f} {robot.base_position()[2]:9.4f} {roll:8.2f} {pitch:8.2f} "
                  f"{left:7.2f} {right:7.2f} {(left + right) / weight:7.3f} {robot.imu()['acc'][2]:9.3f}"
                  + ("  ← 미는 중" if pushing else ""))

    simulate(model, data, control_step, duration=duration, headless=args.headless,
             viewer_setup=track_body_camera(robot.base_body_id, distance=2.0, azimuth=135, elevation=-15))

    if args.headless:
        z = robot.base_position()[2]
        tilt = robot.base_tilt()
        standing = z > FALL_HEIGHT and tilt < FALL_TILT
        print(f"\n판정 (t={data.time:.1f}s): 몸통 높이 {z:.3f} m, 몸통 기울기(tilt) {np.degrees(tilt):.1f}°"
              f" → {'서 있음 ✅' if standing else '넘어짐 ❌'}")
        if standing:
            print("검증 포인트: 정지해 있을 때 '합/무게'가 1.000 이어야 합니다 (힘의 평형: Σ접촉 수직력 = m·g).")

    if args.snapshot:
        pos = robot.base_position()
        path = save_snapshot(model, data, paths.output_path("t06_stand.png"),
                             lookat=(pos[0], pos[1], 0.35), distance=1.8, azimuth=135, elevation=-12)
        print(f"스냅샷 저장: {path.relative_to(paths.REPO_ROOT)}")


if __name__ == "__main__":
    main()
