#!/usr/bin/env python3
"""울퉁불퉁한 지형 실험 — 학습한 보행 정책을 언덕·불규칙한 경사로·장애물 위에서 걸려 보기 (MuJoCo 화면)

실행 (GPU 학습 환경에서: source scripts/activate_gpu.sh)
    python learning/terrain_trial.py --params <정책.pkl>                       # 화면: 지형마다 에피소드 3번, 매번 새 지형
    python learning/terrain_trial.py --params <정책.pkl> --difficulty 0.8      # 더 험하게 (0~1)
    python learning/terrain_trial.py --params <정책.pkl> --headless --episodes 20   # 화면 없이 통계만 (빠름)
    python learning/terrain_trial.py --params <정책.pkl> --friction 0.5        # 미끄러운 바닥 (기본 마찰 1.0)
    python learning/terrain_trial.py --params <정책.pkl> --servo open_duck     # 다른 서보 모델로 (기본: 학습 때 모델)

지형 (biped_sim/terrain.py, 크기는 오리 로봇 기준. 출발 자리 지름 0.7 m는 평평)
    flat       평지 (비교 기준)
    hills      산악 언덕: 파장 약 1 m, 높이 ±1~3 cm, 경사 최대 4~12°
    slopes     불규칙한 경사로: 0.4 m 칸마다 기울기가 다름, 경사 최대 2~9°
    obstacles  장애물: 높이 0.5~2 cm, 한 변 6~24 cm의 납작한 상자들
    mixed      언덕 → 경사로 → 장애물을 이어 붙인 코스
에피소드마다 지형을 새로 만든다 (같은 종류, 다른 무작위 배치). 결과: 10초 버팀 여부, 앞으로 간 거리.
"""
import argparse
import time
from pathlib import Path

import mujoco
import numpy as np
from walk_tools import LATEST, get_spec, load_policy, make_runner, run_episode

from biped_sim import passive_viewer, track_body_camera
from biped_sim.terrain import KINDS, make_terrain, max_slope_deg


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--params", type=Path, default=LATEST, help="정책 파일")
    parser.add_argument("--kinds", nargs="+", default=list(KINDS), choices=KINDS, help="시험할 지형 종류")
    parser.add_argument("--difficulty", type=float, default=0.5, help="지형 험한 정도 0~1")
    parser.add_argument("--episodes", type=int, default=3, help="지형 종류마다 에피소드 수 (매번 새 지형)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--headless", action="store_true", help="화면 없이 빠르게 (통계용)")
    parser.add_argument("--friction", type=float, default=None, help="바닥 마찰 계수 (기본: MuJoCo 1.0)")
    parser.add_argument("--servo", default=None, help="서보 모델을 바꿔 시험 (기본: 학습 때와 같음). ideal / open_duck")
    args = parser.parse_args()

    policy, config = load_policy(args.params.resolve())
    if args.servo:   # 예: ideal 서보로 배운 정책을 실물 측정 모델(open_duck)에서
        config = {**config, "servo": args.servo}
    spec = get_spec(config)
    runner = make_runner(config, terrain=make_terrain("flat"))   # 지형 격자를 가진 모델 (지형은 에피소드마다 바꿔 끼움)
    hfield = runner.model.hfield("terrain").id
    if args.friction is not None:   # 접촉 마찰 = 두 geom 중 큰 값 → 발·바닥 모두
        runner.model.geom_friction[:, 0] = args.friction
    print(f"정책: {args.params} | 로봇 {spec.name} (서보 {spec.servo}) | 지형 {', '.join(args.kinds)} | 험한 정도 "
          f"{args.difficulty} | 마찰 {runner.model.geom_friction[0, 0]:g} | 종류마다 {args.episodes}번")

    results = {k: [] for k in args.kinds}

    def trial(viewer=None):
        for kind in args.kinds:
            for ep in range(args.episodes):
                seed = args.seed * 1000 + ep
                terrain = make_terrain(kind, seed=seed, difficulty=args.difficulty)
                terrain.apply(runner.model)           # 같은 모델에 새 지형 (격자 크기 같음)
                runner.terrain = terrain
                hud = {}
                if viewer is not None:
                    viewer.update_hfield(hfield)      # 화면에도 새 지형
                    hud = dict(text=f"{kind}  (difficulty {args.difficulty}, max slope {max_slope_deg(terrain):.0f} deg)"
                                    f"\nepisode {ep + 1} / {args.episodes}")

                    def show(state, hud=hud):
                        if round(state["t"] / 0.02) % 5 == 0:
                            viewer.set_texts((mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                              "terrain\n\ntime\ndistance\nspeed",
                                              f"{hud['text']}\n{state['t']:4.1f} s\n{state['distance']:+.2f} m\n"
                                              f"{state['vx']:+.2f} m/s"))
                rec = run_episode(runner, policy, viewer, on_step=show if viewer is not None else None)
                fell = rec["fell"]
                dist = float(rec["base"][-1, 0] - rec["base"][0, 0])
                results[kind].append((not fell, dist, float(rec["t"][-1])))
                print(f"  {kind:9s} #{ep + 1}: {'넘어짐 ❌' if fell else '10초 버팀 ✅'}  앞으로 {dist:+.2f} m  "
                      f"({rec['t'][-1]:.1f} s)  최대 경사 {max_slope_deg(terrain):.0f}°", flush=True)
                if viewer is not None:
                    if not viewer.is_running():
                        return
                    time.sleep(0.8)

    if args.headless:
        trial()
    else:
        print("화면: 에피소드마다 새 지형. 종료: 창 닫기 또는 Ctrl+C")
        try:
            with passive_viewer(runner.model, runner.data, show_ui=False) as viewer:
                with viewer.lock():
                    track_body_camera(runner.info["base_body"], distance=1.6 * spec.view_distance,
                                      azimuth=140, elevation=-20)(viewer)
                trial(viewer)
        except KeyboardInterrupt:
            print("\nCtrl+C — 종료합니다.")

    print(f"\n{'지형':10s} {'10초 버팀':>9s} {'평균 거리':>9s} {'평균 버틴 시간':>12s}")
    for kind, rows in results.items():
        if rows:
            ok, dist, t = np.array(rows).T
            print(f"{kind:10s} {int(ok.sum()):>4d} / {len(rows):<3d} {dist.mean():+8.2f} m {t.mean():10.1f} s")


if __name__ == "__main__":
    main()
