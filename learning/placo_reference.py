#!/usr/bin/env python3
"""placo 걷기 엔진으로 오리의 '몸통이 수평인 참고 걸음'을 만들고, 학습 전에 그 걸음 자체를 잰다 (docs/10 §12, B 0단계)

placo(MIT, Rhoban)는 ZMP/LIPM으로 무게중심을 계획하는 걷기 엔진이다. 몸통은 수평으로 두고 무게중심은 좌우로 옮겨
한 발로 서는 걸음을 만든다 — Open Duck 공식 학습과 Disney BDX가 쓰는 '참고 걸음 모방'의 재료.
(Open Duck의 참고 동작 생성기는 라이선스가 없어 코드는 쓰지 않고, placo의 공개 API로 새로 짰다.)

⚠ 따로 만든 환경에서 실행 (placo가 numpy 2를 요구해 학습 환경 .venv-mjx와 섞으면 안 됨):
    python3 -m venv .venv-placo && .venv-placo/bin/pip install placo==0.10.1
    .venv-placo/bin/python learning/placo_reference.py                         # 기본 명령들 → output/references/
    .venv-placo/bin/python learning/placo_reference.py --feet-spacing 0.14 --single-support 0.15
    .venv-placo/bin/python learning/placo_reference.py --commands 0.15,0,0 0,0,0.4

재는 것 (명령마다, 출발 구간 제외) — analyze_sway.py와 같은 정의라 학습한 정책과 바로 비교된다
    몸통 roll 범위, 머리 좌우 흔들림 [cm], 무게중심 좌우 [cm], 흔듦 최고 발 높이 [cm], 실제 속도,
    관절마다 최대 각속도 [rad/s] (서보 무부하 4.4 rad/s와 비교), hip_roll·hip_pitch·무릎 사용 범위 [°]
결과: output/references/<이름>.npz (50 Hz 관절 각도·몸통 자세·발·머리 위치) + summary.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import placo

REPO = Path(__file__).resolve().parents[1]
URDF = REPO / "models" / "third_party" / "open_duck_mini_v2" / "robot.urdf"
LEGS = [f"{s}_{j}" for s in ("left", "right") for j in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle")]
HEAD = ["neck_pitch", "head_pitch", "head_yaw", "head_roll", "left_antenna", "right_antenna"]
SERVO_NOLOAD = 4.4       # [rad/s] STS3215 7.4 V 무부하 속도 (docs/09 §8.1)
# 학습 모델의 서 있는 자세 (walk_spec.with_home_knee(0.6)과 같음: hip_pitch만 좌우 부호가 반대, 무릎·발목은 같음)
HOME = {"left_hip_pitch": -0.3, "left_knee": 0.6, "left_ankle": -0.3,
        "right_hip_pitch": 0.3, "right_knee": 0.6, "right_ankle": -0.3}
# 무릎은 앞으로만 굽음 (두 무릎 모두 +). placo는 다리를 쭉 편 자세(관절 0)에서 IK를 시작하는데, 그 자세에서는 무릎이
# 어느 쪽으로든 꺾일 수 있어 한계를 안 걸면 오른 무릎이 −58°로 거꾸로 꺾인 해가 나왔다 (그러면 hip_roll도 안 씀)
KNEE_LIMITS = (0.05, 1.5)
DT = 0.005               # [s] 운동학 풀이 간격
REC_DT = 0.02            # [s] 기록 간격 = 학습 제어 주기 (50 Hz)
STEADY_FROM = 2.0        # [s] 출발 구간 제외


def make_parameters(args) -> placo.HumanoidParameters:
    p = placo.HumanoidParameters()
    p.single_support_duration = args.single_support
    p.single_support_timesteps = 10
    p.double_support_ratio = args.double_support_ratio
    p.startend_double_support_ratio = 1.5
    p.planned_timesteps = 48
    p.walk_com_height = args.com_height
    p.walk_foot_height = args.foot_height
    p.walk_trunk_pitch = np.radians(args.trunk_pitch_deg)
    p.walk_foot_rise_ratio = 0.2
    p.foot_length = 0.06
    p.foot_width = 0.04
    p.feet_spacing = args.feet_spacing
    p.zmp_margin = 0.0
    p.foot_zmp_target_x = 0.0
    p.foot_zmp_target_y = args.zmp_target_y        # 음수 = 발 중심보다 안쪽 (무게중심을 덜 옮김)
    p.walk_max_dtheta = 1.0
    p.walk_max_dy = 0.1
    p.walk_max_dx_forward = 0.12
    p.walk_max_dx_backward = 0.06
    return p


def step_duration(p) -> float:
    ds = p.double_support_duration
    return p.single_support_duration + (ds() if callable(ds) else ds)


def generate(args, command) -> dict:
    """명령 (vx, vy, wz)로 seconds초 걷는 참고 걸음. 50 Hz 기록."""
    robot = placo.HumanoidRobot(str(URDF))
    for side in ("left", "right"):
        robot.set_joint_limits(f"{side}_knee", *KNEE_LIMITS)
    for j, v in HOME.items():                      # 쭉 편 자세가 아니라 우리 서 있는 자세에서 IK 시작
        robot.set_joint(j, v)
    robot.update_kinematics()
    params = make_parameters(args)
    solver = placo.KinematicsSolver(robot)
    solver.enable_joint_limits(True)
    # 관절 속도 한계: 기본은 끔 (필요한 속도를 그대로 재려고). --servo-limit을 주면 IK가 그 안에서만 움직인다
    # (URDF의 20 rad/s는 실제 서보의 4.4배라 쓸모없음)
    if args.servo_limit:
        for j in LEGS:
            robot.set_velocity_limit(j, args.servo_limit)
    solver.enable_velocity_limits(bool(args.servo_limit))
    solver.dt = DT
    tasks = placo.WalkTasks()
    tasks.initialize_tasks(solver, robot)
    # 과제 가중치 (placo 기본은 모두 soft 1). 이 오리는 발목 roll이 없어서, 발바닥을 평평히 두고 몸통까지 수평이면
    # 무게중심을 옆으로 옮길 수 없다 (실측: 3 cm 요청에 0.28 cm). 셋이 다투면 계획한 무게중심을 못 따라가
    # 동역학적으로 불가능한 걸음이 된다 → 발은 단단히, 몸통 수평은 --trunk-weight로 얼마나 양보할지 고른다
    tasks.left_foot_task.configure("left_foot", "soft", args.foot_weight)
    tasks.right_foot_task.configure("right_foot", "soft", args.foot_weight)
    tasks.trunk_orientation_task.configure("trunk_orientation", "soft", args.trunk_weight)
    tasks.com_task.configure("com", "soft", args.com_weight)
    head = solver.add_joints_task()                # 머리·안테나는 학습 모델처럼 home(0)에 고정
    head.set_joints({j: 0.0 for j in HEAD})
    head.configure("head", "soft", 1.0)
    tasks.reach_initial_pose(np.eye(4), params.feet_spacing, params.walk_com_height, params.walk_trunk_pitch)

    vx, vy, wz = command
    T = step_duration(params)                      # 한 걸음 시간 (한 발 지지 + 양발 지지)
    planner = placo.FootstepsPlannerRepetitive(params)
    planner.configure(vx * T, vy * T, wz * T, 10)        # 10걸음씩 계획하고 걸으면서 다시 계획 (placo 예제와 같이)
    footsteps = planner.plan(placo.HumanoidRobot_Side.left, placo.flatten_on_floor(robot.get_T_world_left()),
                             placo.flatten_on_floor(robot.get_T_world_right()))
    supports = placo.FootstepsPlanner.make_supports(footsteps, 0.0, True, params.has_double_support(), True)
    walk = placo.WalkPatternGenerator(robot, params)
    traj = walk.plan(supports, robot.com_world(), 0.0)

    rec = {k: [] for k in ("t", "q", "trunk_pos", "trunk_rot", "head", "com", "com_plan", "feet", "support")}
    t, last_replan, next_rec = 0.0, 0.0, 0.0
    while t < args.seconds:
        tasks.update_tasks_from_trajectory(traj, t)
        robot.update_kinematics()
        solver.solve(True)
        if not traj.support_is_both(t):
            robot.update_support_side(str(traj.support_side(t)))
            robot.ensure_on_floor()
        if t - last_replan > 0.1 and walk.can_replan_supports(traj, t):
            supports = walk.replan_supports(planner, traj, t, last_replan)
            last_replan = t
            traj = walk.replan(supports, traj, t)
        if t >= next_rec - 1e-9:
            robot.update_kinematics()
            T_trunk = robot.get_T_world_frame("trunk")
            rec["t"].append(t)
            rec["q"].append([robot.get_joint(j) for j in LEGS])
            rec["trunk_pos"].append(T_trunk[:3, 3].copy())
            rec["trunk_rot"].append(T_trunk[:3, :3].copy())
            rec["head"].append(robot.get_T_world_frame("head_assembly")[:3, 3].copy())
            rec["com"].append(np.asarray(robot.com_world()).copy())
            rec["com_plan"].append(np.asarray(traj.get_p_world_CoM(t)).copy())
            rec["feet"].append([robot.get_T_world_frame("left_foot")[:3, 3].copy(),
                                robot.get_T_world_frame("right_foot")[:3, 3].copy()])
            rec["support"].append(0 if traj.support_is_both(t) else (1 if str(traj.support_side(t)) == "left" else 2))
            next_rec += REC_DT
        t += DT
    return {k: np.asarray(v) for k, v in rec.items()}


def measure(rec, command) -> dict:
    """analyze_sway.py와 같은 정의로 참고 걸음을 잰다."""
    t = rec["t"]
    k = t >= STEADY_FROM
    R = rec["trunk_rot"]
    roll = np.degrees(np.arctan2(R[:, 2, 1], R[:, 2, 2]))
    yaw = np.unwrap(np.arctan2(R[:, 1, 0], R[:, 0, 0]))

    def lateral(p):     # 몸통 방향 기준 옆 위치, 직선 추세(천천히 흐르는 것)를 뺀 흔들림 폭 5~95 %
        y = -np.sin(yaw[k].mean()) * p[k, 0] + np.cos(yaw[k].mean()) * p[k, 1]
        n = np.arange(len(y))
        res = y - np.polyval(np.polyfit(n, y, 1), n)
        return float(100 * (np.percentile(res, 95) - np.percentile(res, 5)))

    q = rec["q"]
    qd = np.abs(np.diff(q, axis=0)) / REC_DT
    feet_z = rec["feet"][:, :, 2]
    dist = rec["trunk_pos"][k][-1] - rec["trunk_pos"][k][0]
    dur = t[k][-1] - t[k][0]
    h = yaw[k].mean()
    out = {
        "command": list(command),
        "achieved": [float((np.cos(h) * dist[0] + np.sin(h) * dist[1]) / dur),
                     float((-np.sin(h) * dist[0] + np.cos(h) * dist[1]) / dur),
                     float((yaw[k][-1] - yaw[k][0]) / dur)],
        "roll_range_deg": float(np.ptp(roll[k])), "roll_rms_deg": float(np.sqrt(np.mean(roll[k] ** 2))),
        "head_sway_cm": lateral(rec["head"]), "com_sway_cm": lateral(rec["com"]),
        "com_plan_sway_cm": lateral(rec["com_plan"]),
        # 계획(ZMP가 발 안에 있는 동역학적으로 맞는 무게중심)과 실제 자세의 무게중심 차이 — 크면 따라 할 수 없는 걸음
        "com_track_err_cm": float(100 * np.sqrt(np.mean(np.sum((rec["com"][k] - rec["com_plan"][k])[:, :2] ** 2, 1)))),
        "swing_peak_cm": float(100 * (feet_z[k].max(axis=0) - feet_z[k].min(axis=0)).mean()),
        "max_joint_speed": {j: float(qd[k[1:], i].max()) for i, j in enumerate(LEGS)},
        "range_deg": {j: float(np.degrees(np.ptp(q[k, i]))) for i, j in enumerate(LEGS)},
    }
    out["max_speed_overall"] = max(out["max_joint_speed"].values())
    out["over_servo_pct"] = float(100 * np.mean(qd[k[1:]] > SERVO_NOLOAD))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--commands", nargs="+",
                        default=["0.15,0,0", "0.30,0,0", "-0.08,0,0", "0,0.08,0", "0,0,0.4", "0.1,0,0.4", "0,0,0"])
    parser.add_argument("--com-height", type=float, default=0.205, help="걸을 때 무게중심 높이 [m] (home 0.227)")
    parser.add_argument("--feet-spacing", type=float, default=0.16, help="두 발 간격 [m] (지금 학습 모델 home 0.18)")
    parser.add_argument("--single-support", type=float, default=0.18, help="한 발 지지 시간 [s]")
    parser.add_argument("--double-support-ratio", type=float, default=0.5, help="양발 지지 / 한 발 지지")
    parser.add_argument("--foot-height", type=float, default=0.04, help="흔드는 발 높이 [m]")
    parser.add_argument("--zmp-target-y", type=float, default=-0.03, help="ZMP 목표 (발 중심 기준, 음수 = 안쪽) [m]")
    parser.add_argument("--trunk-pitch-deg", type=float, default=0.0)
    parser.add_argument("--servo-limit", type=float, default=0.0, help="IK 관절 속도 한계 [rad/s] (0 = 끔, 4.4 = 서보 무부하)")
    parser.add_argument("--trunk-weight", type=float, default=1.0, help="몸통 수평 과제 가중 (작을수록 구르기 허용)")
    parser.add_argument("--com-weight", type=float, default=1.0, help="무게중심 과제 가중")
    parser.add_argument("--foot-weight", type=float, default=1.0, help="발 위치·자세 과제 가중")
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--name", default="default")
    parser.add_argument("--out-dir", type=Path, default=REPO / "output" / "references")
    args = parser.parse_args()

    out = args.out_dir / args.name
    out.mkdir(parents=True, exist_ok=True)
    params = make_parameters(args)
    T = step_duration(params)
    print(f"참고 걸음 '{args.name}': 한 걸음 {T:.3f} s (박자 {2 * T:.3f} s), 발 간격 {args.feet_spacing} m, "
          f"무게중심 높이 {args.com_height} m, 발 높이 {args.foot_height} m, ZMP {args.zmp_target_y:+.3f} m, "
          f"가중 발 {args.foot_weight} · 몸통 수평 {args.trunk_weight} · 무게중심 {args.com_weight}")
    summary = {"params": vars(args) | {"out_dir": str(args.out_dir)}, "step_s": T, "results": []}
    for c in args.commands:
        cmd = tuple(float(x) for x in c.split(","))
        rec = generate(args, cmd)
        np.savez(out / f"cmd_{c.replace(',', '_')}.npz", joints=np.array(LEGS), **rec)
        m = measure(rec, cmd)
        summary["results"].append(m)
        worst = max(m["max_joint_speed"], key=m["max_joint_speed"].get)
        print(f"  명령 {cmd}: 실제 ({m['achieved'][0]:+.3f}, {m['achieved'][1]:+.3f}, {m['achieved'][2]:+.2f}) | "
              f"roll 범위 {m['roll_range_deg']:.1f}° | 머리 {m['head_sway_cm']:.1f} cm | "
              f"무게중심 {m['com_sway_cm']:.1f} (계획 {m['com_plan_sway_cm']:.1f}, 오차 {m['com_track_err_cm']:.1f}) cm | "
              f"발 들기 {m['swing_peak_cm']:.1f} cm | 최대 관절 속도 {m['max_speed_overall']:.1f} rad/s ({worst}), "
              f"서보 무부하 초과 {m['over_servo_pct']:.0f} % | hip_roll {m['range_deg']['left_hip_roll']:.0f}° "
              f"hip_pitch {m['range_deg']['left_hip_pitch']:.0f}° 무릎 {m['range_deg']['left_knee']:.0f}°", flush=True)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
