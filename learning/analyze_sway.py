#!/usr/bin/env python3
"""몸통 구르기(roll)와 머리 흔들림이 어디서 오는지 뜯어보기 — 보상을 바꾸기 전에 원인을 재는 도구 (docs/10 §12)

실행 (GPU 학습 환경에서: source scripts/activate_gpu.sh)
    python learning/analyze_sway.py --params output/learning/<실행>/params.pkl
    python learning/analyze_sway.py --params A.pkl --baseline B.pkl        # 두 정책 비교
    python learning/analyze_sway.py --params <정책> --commands 0.15,0,0 0.3,0,0 0,0.08,0

재는 것 (명령마다 평지 10초, 출발 2초 제외)
    몸통 roll      RMS·최대-최소 [°]
    머리 흔들림     좌우 폭 [cm] (gait_metrics와 같은 정의) + **회전축 높이**: 머리 옆 위치를 몸통 roll로 회귀한 기울기.
                   머리(발 위 0.38 m)가 몸통과 함께 굴러서 흔들리면 기울기 ≈ 머리에서 회전축까지의 거리.
    발 들기 분해     흔드는 발을 디딤 발보다 얼마나 들었나 = (골반 기울기 몫: 흔드는 쪽 hip_roll 축이 디딤 쪽보다 높아진 양)
                   + (다리 관절 몫: 무릎·고관절로 발을 끌어올린 양). 어느 쪽으로 발을 드는지 가림
    무게중심        좌우 이동 폭 [cm] (걸음 주기 이동평균을 뺀 값 — docs/09 §8.6과 같은 정의)
    각운동량        몸통 x축 둘레 L_x RMS [kg·m²/s]: 구르기를 '플라이휠'로 쓰고 있는지
    디딤 발         발바닥 roll 각도 [°] (발 모서리로 기우는지)
    hip_roll       토크 RMS [N·m], 정격(0.64) 초과 비율 [%]
    발 들기 실측     흔듦 중 최고 높이 [cm] (clearance_target과 비교)
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from walk_tools import head_sway_cm, load_policy, make_runner, run_episode

RATED_TORQUE = 0.64   # [N·m] STS3215 7.4 V 정격 (docs/09 §8.1)
STEADY_FROM = 2.0     # [s] 출발 구간 제외


def measure(config, policy, command, seconds=10.0) -> dict:
    """명령 하나로 평지 10초 걷고, 구르기·머리 흔들림의 출처를 뜯어본다."""
    runner = make_runner(config)
    runner.command = np.array(command, dtype=float)
    d, info, m = runner.data, runner.info, runner.model
    hips = info.get("hip_roll_joints")            # (왼, 오른) hip_roll 관절 id
    feet = info["foot_bodies"]
    foot_geoms = [next(g for g in range(m.ngeom) if m.geom_bodyid[g] == b and m.geom_contype[g]) for b in feet]
    hip_roll_act = [i for i, n in enumerate(info["joint_names"]) if n.endswith("hip_roll")]
    extra = {k: [] for k in ("hip_z", "foot_z", "com_y", "Lx", "foot_roll", "hip_tau", "contact")}

    def on_step(state):
        mujoco.mj_subtreeVel(m, d)                 # subtree_angmom 채우기 (각운동량)
        extra["hip_z"].append([d.xanchor[j, 2] for j in hips] if hips is not None else [0.0, 0.0])
        extra["foot_z"].append(d.xpos[feet, 2].copy())
        extra["com_y"].append(float(d.subtree_com[info["base_body"], 1]))
        rot = d.xmat[info["base_body"]].reshape(3, 3)
        extra["Lx"].append(float((rot.T @ d.subtree_angmom[info["base_body"]])[0]))   # 몸통 x축 둘레
        # 발바닥 roll: 발 geom의 y축이 수평에서 기운 각도
        extra["foot_roll"].append([float(np.degrees(np.arcsin(np.clip(d.geom_xmat[g].reshape(3, 3)[2, 1], -1, 1))))
                                   for g in foot_geoms])
        extra["hip_tau"].append([float(d.ctrl[info["policy_act"][i]]) for i in hip_roll_act])
        extra["contact"].append(d.xpos[feet, 2] - np.array([runner.ground(*d.xpos[b, :2]) for b in feet])
                                < runner.spec.foot_contact_z)

    rec = run_episode(runner, policy, seconds=seconds, on_step=on_step)
    n = len(rec["t"])
    ex = {k: np.asarray(v[:n]) for k, v in extra.items()}
    steady = rec["t"] >= min(STEADY_FROM, rec["t"][-1] / 2)
    roll = np.degrees(rec["rpy"][:, 0])
    head_y_body = head_lateral(rec)                 # 몸통 방향 기준 머리 옆 위치 [m]

    # 회전축 높이: 머리 옆 위치 = −(지렛대) × sin(roll) + 상수  (roll이 +면 머리는 −y로)
    s = np.sin(np.radians(roll[steady]))
    lever = float(-np.polyfit(s, head_y_body[steady], 1)[0]) if np.ptp(s) > 1e-6 else float("nan")

    # 발 들기 분해: 흔드는 발이 디딤 발보다 높은 양 = 골반 기울기 몫 + 다리 관절 몫
    use = steady & (ex["contact"].sum(1) == 1)      # 한 발만 딛는 구간 (흔드는 발이 정해지는 때)
    if use.sum() < 5:                                # 제자리걸음처럼 늘 양발이면 분해할 수 없음
        lift_total = lift_pelvis = lift_leg = np.array([np.nan])
    else:
        swing = np.argmin(ex["contact"][use], axis=1)   # 공중에 있는 발 (0 = 왼, 1 = 오른)
        k = np.arange(use.sum())
        hip_z, foot_z = ex["hip_z"][use], ex["foot_z"][use]
        stance = 1 - swing
        lift_total = foot_z[k, swing] - foot_z[k, stance]
        lift_pelvis = hip_z[k, swing] - hip_z[k, stance]        # 골반을 기울여 흔드는 쪽 고관절을 올린 양
        lift_leg = lift_total - lift_pelvis                      # 무릎·고관절로 발을 끌어올린 양

    # 무게중심 좌우 이동: 걸음 주기 이동평균(천천히 흘러가는 성분)을 뺌
    win = max(3, int(round(rec["gait_period"] / rec["dt"])))
    com = ex["com_y"][steady]
    trend = np.convolve(com, np.ones(win) / win, mode="same")
    com_osc = com - trend

    tau = ex["hip_tau"][steady]
    peak = np.array([rec["foot_rel_z"][steady, i].max() for i in (0, 1)])
    return {
        "command": list(map(float, command)),
        "fell": bool(rec["fell"]), "survived_s": float(rec["t"][-1]),
        "roll_rms_deg": float(np.sqrt(np.mean(roll[steady] ** 2))),
        "roll_range_deg": float(np.ptp(roll[steady])),
        "head_sway_cm": float(head_sway_cm(rec, steady)),
        "head_pivot_lever_m": lever,
        "lift_total_cm": float(100 * np.mean(lift_total)), "lift_pelvis_cm": float(100 * np.mean(lift_pelvis)),
        "lift_leg_cm": float(100 * np.mean(lift_leg)),
        "pelvis_share_pct": float(100 * np.mean(lift_pelvis) / max(np.mean(lift_total), 1e-9)),
        "com_sway_cm": float(100 * (np.percentile(com_osc, 95) - np.percentile(com_osc, 5))),
        "Lx_rms": float(np.sqrt(np.mean(ex["Lx"][steady] ** 2))),
        "foot_roll_range_deg": float(np.ptp(ex["foot_roll"][steady])),
        "hip_roll_tau_rms": float(np.sqrt(np.mean(tau ** 2))),
        "hip_roll_over_rated_pct": float(100 * np.mean(np.abs(tau) > RATED_TORQUE)),
        "swing_peak_cm": float(100 * peak.mean()),
    }


def head_lateral(rec) -> np.ndarray:
    """몸통이 향한 방향 기준, 머리의 좌우 위치 [m] (몸통 중심 기준)."""
    rel = rec["head"][:, :2] - rec["base"][:, :2]
    yaw = rec["rpy"][:, 2]
    return -np.sin(yaw) * rel[:, 0] + np.cos(yaw) * rel[:, 1]


ROWS = [("몸통 roll RMS / 범위 [°]", lambda r: f"{r['roll_rms_deg']:.1f} / {r['roll_range_deg']:.0f}"),
        ("머리 좌우 흔들림 [cm]", lambda r: f"{r['head_sway_cm']:.1f}"),
        ("회전축까지 거리 [m]", lambda r: f"{r['head_pivot_lever_m']:.2f}"),
        ("발 들기 합 [cm]", lambda r: f"{r['lift_total_cm']:.1f}"),
        ("  골반 기울기 몫 [cm]", lambda r: f"{r['lift_pelvis_cm']:.1f} ({r['pelvis_share_pct']:.0f} %)"),
        ("  다리 관절 몫 [cm]", lambda r: f"{r['lift_leg_cm']:.1f}"),
        ("흔듦 최고 발 높이 [cm]", lambda r: f"{r['swing_peak_cm']:.1f}"),
        ("무게중심 좌우 [cm]", lambda r: f"{r['com_sway_cm']:.1f}"),
        ("각운동량 Lx RMS", lambda r: f"{r['Lx_rms']:.4f}"),
        ("디딤 발 roll 범위 [°]", lambda r: f"{r['foot_roll_range_deg']:.1f}"),
        ("hip_roll 토크 RMS [N·m]", lambda r: f"{r['hip_roll_tau_rms']:.2f}"),
        ("hip_roll 정격 초과 [%]", lambda r: f"{r['hip_roll_over_rated_pct']:.0f}")]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, help="비교할 기준 정책")
    parser.add_argument("--commands", nargs="+", default=["0.15,0,0", "0.30,0,0", "0,0.08,0", "0,0,0.4"],
                        help="명령 vx,vy,wz (쉼표로)")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--out", type=Path, help="결과 JSON")
    args = parser.parse_args()

    commands = [tuple(float(x) for x in c.split(",")) for c in args.commands]
    runs = {"새 정책": args.params} | ({"기준": args.baseline} if args.baseline else {})
    results = {}
    for label, path in runs.items():
        policy, config = load_policy(path)
        results[label] = [measure(config, policy, c, args.seconds) for c in commands]
        print(f"[{label}] {path}", flush=True)

    for i, c in enumerate(commands):
        print(f"\n명령 (vx, vy, wz) = {c}")
        head = "  {:28s}" + "".join(f" {l:>12s}" for l in runs)
        print(head.format(""))
        for name, fmt in ROWS:
            print(f"  {name:28s}" + "".join(f" {fmt(results[l][i]):>12s}" for l in runs))
    if args.out:
        args.out.write_text(json.dumps(results, ensure_ascii=False, indent=1))
        print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()
