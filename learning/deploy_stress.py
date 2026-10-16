#!/usr/bin/env python3
"""실물에 올리기 전 시험: 실물과 시뮬의 차이를 하나씩, 그리고 한꺼번에 키우면 정책이 어디서 무너지나 (docs/10 §14)

실행 (GPU 학습 환경에서: source scripts/activate_gpu.sh)
    python learning/deploy_stress.py --params learning/pretrained/duck_cmd_policy.pkl
    python learning/deploy_stress.py --params A.pkl --params B.pkl          # 정책 여러 개 비교

한 조건마다: 평지 명령 4가지 (앞 0.15·0.30, 뒤 0.08, 제자리 돌기 0.4)를 허용(±0.03 m/s, ±0.1 rad/s) 안에서 따라가나 +
섞인 코스 1.0에서 앞 0.15로 10번 중 몇 번 10초 버티나.
조건: 제어 지연 (20 ms 단위), 모터 세기 (배터리 전압 처짐·서보 개체 차), 몸통 질량, 서보 영점 오차, IMU 장착 기울기,
센서 잡음 (학습 때 OBS_NOISE의 배수), 관절 속도를 엔코더(4096칸/바퀴) 차분으로 (실물 서보는 속도를 직접 못 잼), 그리고 이것들을 묶은 '실물 묶음'.
"""
import argparse
from pathlib import Path

import numpy as np

from walk_tools import load_policy, make_runner, run_episode
from evaluate_walk import command_tracking, COMMAND_TOL
from biped_sim.terrain import make_terrain
from biped_sim.envs.walk_mjx import tilt_matrix
from biped_sim.envs.walk_spec import OBS_NOISE

CMDS = [(0.15, 0, 0), (0.30, 0, 0), (-0.08, 0, 0), (0, 0, 0.4)]
TICK = 2 * np.pi / 4096          # STS3215 엔코더 한 칸 [rad]


def patch(r, noise=0.0, fd_vel=False, rng_seed=0):
    """재생기의 관측에 센서 잡음 (OBS_NOISE × noise, 학습과 같은 자리)·엔코더 차분 속도를 넣는다."""
    rng = np.random.default_rng(rng_seed); n = r.spec.n; orig = r._base_obs
    scale = np.concatenate([[OBS_NOISE["gravity"]] * 3, [OBS_NOISE["gyro"]] * 3, [0.0] * 4,
                            [OBS_NOISE["joint_pos"]] * n, [0.1 * OBS_NOISE["joint_vel"]] * n])
    state = {"q": None}
    def base_obs():
        o = orig().copy()
        jp = slice(10, 10 + n); jv = slice(10 + n, 10 + 2 * n)
        if fd_vel:   # 관절 각도는 엔코더 칸으로 반올림, 속도는 지난 제어 스텝과의 차분 (관측 속도는 × 0.1로 들어감)
            q = np.round(o[jp] / TICK) * TICK
            o[jv] = 0.1 * (0.0 if state["q"] is None else (q - state["q"]) / 0.02)
            state["q"] = q; o[jp] = q
        o[:len(scale)] += noise * scale * rng.normal(size=len(scale))
        return o
    r._base_obs = base_obs


def setup(r, cond):
    m, b = r.model, r.info["base_body"]
    r.delay = cond.get("delay", 0); r.motor_scale = cond.get("motor", 1.0)
    m.body_mass[b] *= cond.get("mass", 1.0); m.body_inertia[b] *= cond.get("mass", 1.0)
    if cond.get("joint_off"): r.joint_offset = np.random.default_rng(42).uniform(-1, 1, r.spec.n) * np.radians(cond["joint_off"])
    if cond.get("imu_tilt"): r.imu_rot = tilt_matrix(np.radians(cond["imu_tilt"]), np.radians(cond["imu_tilt"]))
    patch(r, cond.get("noise", 0.0), cond.get("fd_vel", False))


def trial(config, policy, cond):
    ok = 0
    for c in CMDS:
        r = make_runner(config); setup(r, cond); r.command = np.array(c)
        rec = run_episode(r, policy); vv, w = command_tracking(rec)
        ok += (not rec["fell"]) and np.linalg.norm(vv - c[:2]) <= COMMAND_TOL["lin"] and abs(w - c[2]) <= COMMAND_TOL["yaw"]
    alive = 0
    for s in range(10):
        r = make_runner(config, terrain=make_terrain("mixed", seed=300 + s, difficulty=1.0)); setup(r, cond)
        r.command = np.array([0.15, 0, 0]); alive += not run_episode(r, policy)["fell"]
    return ok, alive


REAL = dict(delay=2, motor=0.85, mass=1.15, joint_off=2, imu_tilt=2, noise=1.0, fd_vel=True)
CONDS = ([("기본", {})] + [(f"지연 {20 * k} ms", dict(delay=k)) for k in (2, 3, 4)]
         + [(f"모터 {m}", dict(motor=m)) for m in (0.85, 0.7, 0.6)] + [("질량 1.3", dict(mass=1.3))]
         + [("영점 6°", dict(joint_off=6)), ("IMU 6°", dict(imu_tilt=6)), ("잡음 3×", dict(noise=3)),
            ("속도 = 엔코더 차분 + 잡음 1×", dict(fd_vel=True, noise=1))]
         + [("실물 묶음 (40 ms·모터 0.85·질량 1.15·영점 2°·IMU 2°·잡음·차분)", REAL),
            ("실물 묶음 + 모터 0.7", {**REAL, "motor": 0.7}), ("실물 묶음 + 지연 60 ms", {**REAL, "delay": 3})])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--params", type=Path, action="append", required=True)
    args = parser.parse_args()
    policies = [load_policy(p) for p in args.params]
    print("조건 | " + " | ".join(p.parent.name if p.name == "params.pkl" else p.stem for p in args.params)
          + "   (평지 명령 /4, 험지 버팀 /10)")
    for name, cond in CONDS:
        cells = [f"{a}/4, {b}/10" for policy, config in policies for a, b in [trial(config, policy, cond)]]
        print(f"{name} | " + " | ".join(cells), flush=True)


if __name__ == "__main__":
    main()
