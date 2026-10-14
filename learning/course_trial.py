#!/usr/bin/env python3
"""직접 만든 코스(models/courses)를 학습한 정책으로 곧장 걸어 보기 — 어디까지 가고 어디서 넘어지나 (docs/11 §3.1)

실행 (GPU 학습 환경에서: source scripts/activate_gpu.sh)
    python learning/course_trial.py --params learning/pretrained/duck_cmd_policy.pkl
    python learning/course_trial.py --params <정책.pkl> --courses stairs stones --speeds 0.1 0.2
    python learning/course_trial.py --params <정책.pkl> --no-assist      # 방향 보정 없이 (정책 혼자 방향 지키기)

명령은 앞 속도 하나. 기본은 조종자처럼 방향을 보정 (회전 명령 = −1.5 × 방향 오차 − 0.8 × 옆으로 벗어난 거리, 최대 0.3 rad/s) —
정책이 스스로 방향을 못 지켜 코스 옆으로 빠지는 것(5차 R1: 험지에서 10초에 −60~−70°)과 지형 자체의 한계를 나눠 보려고.
결과: 코스마다 속도마다 넘어짐 / 통과 / 멈춤, 마지막 위치와 그 구간 이름. --out에 JSON.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from walk_tools import load_policy, make_runner, run_episode

from biped_sim.course import load_course

# 예시 코스의 구간 (시작 x [m], 이름) — 마지막은 끝
SECTIONS = {
    "stairs": [(0.8, "1 cm 계단"), (4.6, "1.5 cm 계단"), (7.8, "2 cm 계단"), (11.0, "끝")],
    "ramp": [(0.8, "4° 경사"), (3.8, "6° 경사"), (6.8, "9° 경사"), (9.6, "옆 기운 길"), (11.4, "끝")],
    "stones": [(0.8, "0.8 cm 디딤돌"), (2.6, "1.3 cm 디딤돌"), (4.6, "턱 0.5"), (5.3, "턱 1"), (6.0, "턱 1.5"),
               (6.7, "턱 2"), (7.5, "끝")],
    "gravel": [(0.8, "잔자갈 0.8 cm"), (2.8, "자갈 1.2 cm"), (4.8, "굵은 돌 2 cm"), (6.6, "끝")],
    "drawing": [(0.5, "블록 길"), (1.3, "언덕"), (4.4, "DUCK 글자"), (5.4, "끝")],
}


def trial(config, policy, terrain, speed: float, end: float, assist: bool) -> dict:
    runner = make_runner(config, terrain=terrain)
    runner.command = np.array([speed, 0.0, 0.0])

    def steer(state):   # 조종자처럼: 앞을 보도록, 옆으로 벗어나면 가운데로
        d = runner.data
        rot = d.xmat[runner.info["base_body"]].reshape(3, 3)
        yaw, y = np.arctan2(rot[1, 0], rot[0, 0]), d.xpos[runner.info["base_body"], 1]
        runner.command[2] = float(np.clip(-1.5 * yaw - 0.8 * y, -0.3, 0.3))

    rec = run_episode(runner, policy, seconds=min(120.0, end / speed * 1.3), on_step=steer if assist else None)
    base = np.asarray(rec["base"])
    x, y = base[-1, :2]
    passed = bool(base[:, 0].max() >= end)          # 끝을 한 번이라도 넘었으면 통과 (그 뒤 코스 끝 낭떠러지에서 떨어져도)
    return {"fell": bool(rec["fell"]) and not passed, "x": float(min(x, end) if passed else x), "y": float(y),
            "passed": passed, "time": float(rec["t"][-1])}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--courses", nargs="+", default=list(SECTIONS))
    parser.add_argument("--speeds", type=float, nargs="+", default=[0.10, 0.15, 0.20])
    parser.add_argument("--no-assist", action="store_true", help="방향 보정 없이")
    parser.add_argument("--out", type=Path, help="결과 JSON")
    args = parser.parse_args()
    policy, config = load_policy(args.params)
    results = {}
    print(f"정책 {args.params} | 방향 보정 {'없음' if args.no_assist else '있음'}")
    for name in args.courses:
        secs = SECTIONS[name]
        terrain, end = load_course(name), secs[-1][0]
        results[name] = {}
        row = []
        for v in args.speeds:
            r = trial(config, policy, terrain, v, end, not args.no_assist)
            where = [s for x0, s in secs if r["x"] >= x0 - 0.1]
            r["section"] = where[-1] if where else "출발"
            results[name][f"{v:g}"] = r
            status = "통과" if r["passed"] else ("넘어짐" if r["fell"] else "멈춤")
            row.append(f"{v:.2f}: {status} x={r['x']:.1f} ({r['section']}) y={r['y']:+.1f}")
        print(f"{name:8s} " + " | ".join(row), flush=True)
    if args.out:
        args.out.write_text(json.dumps(results, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
