#!/usr/bin/env python3
"""GPU로 학습한 보행 정책을 MuJoCo 화면으로 보기

실행 (GPU 학습 환경에서: source scripts/activate_gpu.sh)
    python learning/play_walk.py --view              # 최근 학습 정책을 반복 재생 (넘어지거나 10초가 지나면 다시 시작)
    python learning/play_walk.py --live              # 학습 화면: 학습 중인 최신 정책을 계속 불러와 재생 + 학습 그래프
    python learning/play_walk.py --view --params learning/pretrained/walk_policy.pkl   # 다듬어 둔 걸음
    python learning/play_walk.py --view --backend mjx   # 학습 때와 똑같은 GPU 시뮬레이터(MJX)로 재생
    python learning/play_walk.py                     # 화면 없이 10초 걸어 보고 요약만 (자세한 분석: analyze_gait.py)

화면에서 Enter 키: [학습 현황](그래프 + 가운데 로봇) ↔ [로봇 보기](로봇 크게 + 발 접촉 그래프)
    --live는 학습 현황, --view는 로봇 보기로 시작. 그래프는 learning/live_dashboard.py

기본(--backend mujoco)은 일반 MuJoCo — ROS2 sim_node와 같은 시뮬레이터 — 로 돌린다.
GPU(MJX)에서 배운 걸음이 다른 시뮬레이터에서도 통하는지(sim-to-sim) 확인하는 의미가 있다.
학습이 진행 중이어도 실행할 수 있다 (평가 때마다 최신 정책이 저장됨).
"""
import argparse
import json
import time
from pathlib import Path

from walk_tools import (LATEST, find_run_dir, gait_metrics, get_spec, load_policy, make_runner, run_episode,
                        summary_lines)

from biped_sim import passive_viewer, track_body_camera


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--params", type=Path, default=LATEST, help="정책 파일 (기본: 최근 학습)")
    parser.add_argument("--view", action="store_true", help="MuJoCo 화면으로 반복 재생")
    parser.add_argument("--live", action="store_true",
                        help="학습 화면: 정책 파일이 바뀔 때마다(학습이 저장할 때마다) 다시 불러와 계속 재생 + 학습 그래프")
    parser.add_argument("--backend", choices=["mujoco", "mjx"], default="mujoco",
                        help="mujoco: 일반 MuJoCo (ROS2 sim_node와 같은 시뮬레이터) / mjx: 학습과 같은 GPU 시뮬레이터")
    args = parser.parse_args()

    params_path = args.params.resolve()
    if params_path.exists():
        policy, config = load_policy(params_path)
    elif args.live:  # 학습을 막 시작함: 첫 정책이 저장될 때까지 행동 0 (PD로 제자리에 서 있음)
        policy, config = None, {}
        config_path = params_path.with_name("config.json")
        if params_path.name == "params.pkl" and config_path.exists():  # 학습 설정은 시작하자마자 저장됨
            config = json.loads(config_path.read_text())
        print(f"정책 파일을 기다리는 중: {params_path} (학습 시작 후 약 3~4분)")
    else:
        raise SystemExit(f"정책이 없습니다: {args.params}\n→ 먼저 python learning/train_walk_gpu.py")
    runner = make_runner(config, args.backend)
    if args.live:   # 학습 화면: 그림자·바닥 반사를 끄면 한 장 그리는 시간 1/4 (오리 8.6 → 2.1 ms, 실측) → GPU 학습을 덜 방해
        runner.model.light_castshadow[:] = 0
        runner.model.mat_reflectance[:] = 0
    print(f"정책: {args.params} | 로봇 {runner.spec.name} | 목표 속도 {runner.target_speed} m/s | 시뮬레이터: {args.backend}")

    if not (args.view or args.live):
        print("\n".join(summary_lines(gait_metrics(run_episode(runner, policy), runner.info))))
        print("영상·그래프·좌우 비교: python learning/analyze_gait.py --params " + str(args.params))
        return

    from live_dashboard import Dashboard  # 같은 learning/ 폴더

    loaded_mtime = params_path.stat().st_mtime if policy is not None else None
    dash = Dashboard(get_spec(config), config, find_run_dir(params_path, config), mode="graphs" if args.live else "robot")
    print("화면 재생: 넘어지거나 10초가 지나면 처음부터 다시. Enter = 그래프 ↔ 로봇 화면 전환. "
          "종료: 창 닫기 또는 Ctrl+C")
    if args.live:
        print("실시간 모드: 학습이 새 정책을 저장하면 다음 에피소드부터 자동으로 바뀝니다.")
    try:
        # 설정 패널은 숨김 (그래프와 겹치지 않게). 창에서 Tab / Shift+Tab으로 켤 수 있음
        with passive_viewer(runner.model, runner.data, show_ui=False, key_callback=dash.key_callback) as viewer:
            with viewer.lock():
                # 로봇 크기에 맞춘 거리 (simple_biped 2.5 m, 오리 약 1.7 m)
                track_body_camera(runner.info["base_body"], distance=1.67 * runner.spec.view_distance,
                                  azimuth=120, elevation=-15)(viewer)
            episode = 0
            while viewer.is_running():
                if args.live and params_path.exists() and params_path.stat().st_mtime != loaded_mtime:
                    try:  # 학습 프로세스가 파일을 바꾸는 중이면 실패할 수 있음 → 다음 에피소드에 다시 시도
                        mtime = params_path.stat().st_mtime
                        policy, config = load_policy(params_path)
                        loaded_mtime = mtime  # 성공했을 때만 기록 (실패하면 다음에 다시 읽음)
                        dash.set_policy(config, find_run_dir(params_path, config))
                        print(f"  ↻ 새 정책 불러옴 (학습 스텝 {config.get('saved_step', '?')})")
                    except Exception as e:  # noqa: BLE001
                        print(f"  (정책 다시 읽기 실패, 다음에 재시도: {type(e).__name__})")
                episode += 1
                print(f"[에피소드 {episode}]" + ("" if policy is not None else " (첫 정책 기다리는 중: 행동 0)"))
                dash.start_episode()
                rec = run_episode(runner, policy, viewer, on_step=lambda state: dash.on_step(viewer, state))
                print("\n".join(summary_lines(gait_metrics(rec, runner.info))))
                time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nCtrl+C — 종료합니다.")


if __name__ == "__main__":
    main()
