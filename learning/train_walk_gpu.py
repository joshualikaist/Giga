#!/usr/bin/env python3
"""GPU 보행 학습 — MuJoCo MJX(GPU 병렬 시뮬레이션) + Brax PPO

실행 (GPU 학습 환경에서. 처음 한 번: bash scripts/setup_gpu_learning.sh)
    source scripts/activate_gpu.sh
    python learning/train_walk_gpu.py --watch               # 기본 3,000만 스텝 + 학습 화면(그래프·로봇) 함께
    python learning/train_walk_gpu.py                       # 화면 없이 학습만
    python learning/train_walk_gpu.py --steps 2000000       # 짧게 동작 확인
    python learning/train_walk_gpu.py --reward heading=-1 --name walk_heading   # 보상 바꿔 실험

학습 화면 (--watch, = 다른 터미널에서 python learning/play_walk.py --live)
    MuJoCo 창 하나에 학습 그래프와 최신 정책으로 걷는 로봇. Enter 키로 [학습 현황] ↔ [로봇 보기] 전환

학습 과정을 TensorBoard로 (다른 터미널: source scripts/activate.sh 후)
    tensorboard --logdir output/learning      → http://localhost:6006
    eval/episode_reward/<항목> : 보상 항목별 에피소드 합계 (어떤 보상이 오르고 내리는지)
    walk/forward_speed_mps     : 평균 앞으로 걷는 속도 (목표 0.3 m/s)
    walk/episode_seconds       : 평균 버틴 시간 (최대 10 s)
    walk/gait_<flight|single|double>_pct : 걸음 방식 — 두 발 공중 / 한 발 / 두 발 땅 시간 비율 [%]

결과 (output/learning/walk_<날짜_시각>/)
    params.pkl   학습된 정책 (평가할 때마다 갱신 → 학습 중에도 화면으로 볼 수 있음)
    config.json  학습 설정
    events.*     TensorBoard 기록
    → 화면으로 보기: python learning/play_walk.py --view

과제·보상 설명: biped_sim/envs/walk_mjx.py, docs/08_gpu_walking.md
"""
import argparse
import functools
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from biped_sim import paths

RUNS_DIR = paths.OUTPUT_DIR / "learning"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=30_000_000, help="총 학습 스텝 (제어 스텝, 1스텝 = 0.02 s)")
    parser.add_argument("--envs", type=int, default=1024, help="GPU에서 동시에 돌릴 시뮬레이션 개수")
    parser.add_argument("--evals", type=int, default=31, help="학습 중 평가 횟수 (= 그래프 점 개수, 모델 저장 횟수)")
    parser.add_argument("--speed", type=float, default=0.3, help="목표 걷기 속도 [m/s]")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reward", action="append", default=[], metavar="이름=가중치",
                        help="보상 가중치 바꾸기 (여러 번 가능). 예: --reward heading=-1.0 --reward flight=-1.0")
    parser.add_argument("--name", default="walk", help="결과 폴더 이름 앞부분 (TensorBoard에서 실행 구분용)")
    parser.add_argument("--output-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--watch", action="store_true",
                        help="학습 화면을 함께 띄움 (그래프 + 최신 정책으로 걷는 로봇, Enter로 전환) = play_walk.py --live")
    args = parser.parse_args()

    try:
        import jax
        from brax.io import model as brax_model
        from brax.training.agents.ppo import networks as ppo_networks
        from brax.training.agents.ppo import train as ppo
        from tensorboardX import SummaryWriter

        from biped_sim.envs import walk_mjx
    except ImportError as e:
        raise SystemExit(f"GPU 학습 환경이 아닙니다 ({e}) → bash scripts/setup_gpu_learning.sh 후 "
                         "source scripts/activate_gpu.sh")

    if jax.default_backend() != "gpu":
        print(f"[warn] JAX가 GPU가 아니라 {jax.default_backend()}를 씁니다. 매우 느릴 수 있습니다.")

    overrides = {}
    for item in args.reward:
        name, _, value = item.partition("=")
        overrides[name.strip()] = float(value)
    run_dir = args.output_dir.resolve() / f"{args.name}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    latest = args.output_dir.resolve() / "walk_latest.pkl"
    config = {
        "run_dir": str(run_dir),
        "target_speed": args.speed, "steps": args.steps, "envs": args.envs, "seed": args.seed,
        "physics_dt": walk_mjx.PHYSICS_DT, "control_dt": walk_mjx.CONTROL_DT,
        "action_scale": walk_mjx.ACTION_SCALE, "reward_weights": {**walk_mjx.REWARD_WEIGHTS, **overrides},
        "policy_hidden": [128, 128, 128], "value_hidden": [256, 256, 256],
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))

    print("=" * 76)
    print(f"과제: 앞으로 {args.speed} m/s로 걷기 (넘어지면 끝, 에피소드 최대 10 s)")
    print(f"학습: Brax PPO, {args.steps:,} 스텝, GPU 병렬 환경 {args.envs}개, 장치 {jax.devices()[0]}")
    print(f"보상 가중치: {config['reward_weights']}")
    print(f"저장: {run_dir}")
    print(f"실시간 그래프: 다른 터미널(.venv)에서  tensorboard --logdir {args.output_dir}  → http://localhost:6006")
    print("처음 1~3분은 GPU용 코드 컴파일(JIT) 시간이라 출력이 없습니다.")
    if args.watch:
        print("학습 화면: 곧 MuJoCo 창이 열립니다 (첫 정책이 나올 때까지 로봇은 제자리에 서 있음). "
              "Enter = 그래프 ↔ 로봇 전환")
    print("=" * 76, flush=True)
    if args.watch:  # 화면은 별도 프로세스(CPU)로: 창을 닫아도 학습은 계속되고, 학습이 끝나도 창은 남음
        watch_log = open(run_dir / "watch.log", "w")
        subprocess.Popen([sys.executable, "-u", str(Path(__file__).with_name("play_walk.py")), "--live",
                          "--params", str(run_dir / "params.pkl")], stdout=watch_log, stderr=subprocess.STDOUT)

    env = walk_mjx.BipedWalkMjxEnv(target_speed=args.speed, reward_weights=overrides)
    # flush_secs=5: 기록을 5초마다 파일에 씀 (기본 120초면 TensorBoard·학습 화면 그래프가 최대 2분 늦게 보임)
    writer = SummaryWriter(str(run_dir), flush_secs=5)
    t_start = time.perf_counter()
    episode_len = 500  # 10 s

    def progress(step, metrics):
        for key, value in metrics.items():
            try:
                writer.add_scalar(key, float(value), step)
            except (TypeError, ValueError):
                pass
        if "eval/episode_reward" not in metrics:
            return
        length = float(metrics["eval/avg_episode_length"])
        speed = float(metrics.get("eval/episode_forward_speed", 0.0)) / max(length, 1.0)
        writer.add_scalar("walk/forward_speed_mps", speed, step)
        writer.add_scalar("walk/episode_seconds", length * walk_mjx.CONTROL_DT, step)
        gait = {k: float(metrics.get(f"eval/episode_gait/{k}", 0.0)) / max(length, 1.0) for k in walk_mjx.GAIT_METRICS}
        for k in ("flight", "single", "double", "phase_match"):
            writer.add_scalar(f"walk/gait_{k}_pct", 100.0 * gait[k], step)
        writer.add_scalar("walk/abs_heading_deg", float(np.degrees(gait["abs_heading"])), step)
        writer.flush()
        print(f"  스텝 {step:>11,} / {args.steps:,} | 평균 보상 {float(metrics['eval/episode_reward']):7.2f} | "
              f"버틴 시간 {length * walk_mjx.CONTROL_DT:5.2f} s / 10 s | 앞으로 속도 {speed:+.2f} m/s | "
              f"두 발 공중 {100 * gait['flight']:3.0f}% | 경과 {time.perf_counter() - t_start:5.0f} s", flush=True)

    def save_policy(step, make_policy, params):  # 평가 때마다 최신 정책 저장 (play_walk.py --live가 자동으로 다시 읽음)
        # 설정(json)을 먼저, 정책(pkl)을 나중에 바꿔야 재생 쪽이 '새 pkl + 옛 json'을 읽는 일이 없음
        (run_dir / "config.json").write_text(json.dumps({**config, "saved_step": int(step)}, indent=2,
                                                        ensure_ascii=False))
        shutil.copy(run_dir / "config.json", latest.with_suffix(".json"))
        tmp = run_dir / "params.pkl.tmp"
        brax_model.save_params(str(tmp), params)
        shutil.copy(tmp, latest.with_suffix(".pkl.tmp"))
        # 원자적 교체(replace): 재생 쪽이 반쯤 쓰인 파일을 읽지 않게
        latest.with_suffix(".pkl.tmp").replace(latest)
        tmp.replace(run_dir / "params.pkl")

    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=tuple(config["policy_hidden"]),
        value_hidden_layer_sizes=tuple(config["value_hidden"]))

    _, params, _ = ppo.train(
        environment=env,
        num_timesteps=args.steps,
        num_envs=args.envs,
        episode_length=episode_len,
        num_evals=args.evals,
        num_eval_envs=128,
        learning_rate=3e-4,
        entropy_cost=1e-2,
        discounting=0.97,
        gae_lambda=0.95,
        unroll_length=20,
        batch_size=256,
        num_minibatches=16,          # 256 × 16 = 4096 = 4 × 환경 1024개
        num_updates_per_batch=4,
        clipping_epsilon=0.2,
        max_grad_norm=1.0,
        normalize_observations=True,
        reward_scaling=1.0,
        network_factory=network_factory,
        seed=args.seed,
        log_training_metrics=True,
        progress_fn=progress,
        policy_params_fn=save_policy,
    )
    save_policy(args.steps, None, params)
    writer.close()
    print(f"\n학습 완료: {time.perf_counter() - t_start:.0f} s")
    print(f"정책 저장: {run_dir / 'params.pkl'}  (최신: {latest})")
    print("화면으로 보기:  python learning/play_walk.py --view"
          + ("   (--watch 학습 화면은 창을 닫으면 끝납니다)" if args.watch else ""))


if __name__ == "__main__":
    main()
