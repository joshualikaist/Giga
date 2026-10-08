#!/usr/bin/env python3
"""GPU 보행 학습 — MuJoCo MJX(GPU 병렬 시뮬레이션) + Brax PPO

실행 (GPU 학습 환경에서. 처음 한 번: bash scripts/setup_gpu_learning.sh)
    source scripts/activate_gpu.sh
    python learning/train_walk_gpu.py --watch               # 기본 3,000만 스텝 + 학습 화면(그래프·로봇) 함께
    python learning/train_walk_gpu.py --robot open_duck_mini --watch   # 오리 로봇 (bash scripts/get_open_duck.sh 먼저)
    python learning/train_walk_gpu.py --robot open_duck_mini --terrain rough --init-from <평지 정책.pkl>  # 울퉁불퉁한 지형
    python learning/train_walk_gpu.py                       # 화면 없이 학습만
    python learning/train_walk_gpu.py --steps 2000000       # 짧게 동작 확인
    python learning/train_walk_gpu.py --reward symmetry=0 --name walk_nosym     # 보상 바꿔 실험 (대칭 보상 끄기)
    python learning/train_walk_gpu.py --init-from output/learning/<YYMMDD_HHMMSS>_walk/params.pkl \
        --reward heading=-5 --steps 10000000 --name walk_heading               # 이전 걸음에서 이어서 다듬기

학습 화면 (--watch, = 다른 터미널에서 python learning/play_walk.py --live)
    MuJoCo 창 하나에 학습 그래프와 최신 정책으로 걷는 로봇. Enter 키로 [학습 현황] ↔ [로봇 보기] 전환

학습 과정을 TensorBoard로 (다른 터미널: source scripts/activate.sh 후)
    tensorboard --logdir output/learning      → http://localhost:6006
    eval/episode_reward/<항목> : 보상 항목별 에피소드 합계 (어떤 보상이 오르고 내리는지)
    walk/forward_speed_mps     : 평균 앞으로 걷는 속도 (목표 0.3 m/s)
    walk/episode_seconds       : 평균 버틴 시간 (최대 10 s)
    walk/gait_<flight|single|double>_pct : 걸음 방식 — 두 발 공중 / 한 발 / 두 발 땅 시간 비율 [%]

결과 (output/learning/<YYMMDD_HHMMSS>_<이름>/, 예: 261008_140644_walk — TensorBoard Runs에 이 이름으로 보임)
    params.pkl     학습된 정책 (평가할 때마다 갱신 → 학습 중에도 화면으로 볼 수 있음)
    checkpoints/   평가마다의 정책 step_<스텝>.pkl (중간 정책이 더 좋을 수 있음 → analyze_gait.py --run)
    config.json    학습 설정 (보상 가중치 등)
    events.*       TensorBoard 기록
    watch.log      --watch 학습 화면의 출력
    → 화면으로 보기: python learning/play_walk.py --view,  걸음 분석: python learning/analyze_gait.py --run <폴더>

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
    parser.add_argument("--robot", default="simple_biped", help="로봇: simple_biped, open_duck_mini (docs/09)")
    parser.add_argument("--speed", type=float, default=None, help="목표 걷기 속도 [m/s] (기본: 로봇별, simple 0.3 / 오리 0.15)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reward", action="append", default=[], metavar="이름=가중치",
                        help="보상 가중치 바꾸기 (여러 번 가능). 예: --reward heading=-1.0 --reward flight=-1.0")
    parser.add_argument("--name", default=None,
                        help="실험 이름 (기본: walk, 오리는 duck_walk). 결과 폴더 = <YYMMDD_HHMMSS>_<이름> "
                             "(날짜가 앞이라 TensorBoard Runs가 시간순)")
    parser.add_argument("--output-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--terrain", choices=["flat", "rough"], default="flat",
                        help="rough: 12 × 12 m 울퉁불퉁한 지형(언덕·경사로·장애물)의 무작위 위치에서 출발 (biped_sim/terrain.py)")
    parser.add_argument("--terrain-seed", type=int, default=0, help="rough 지형을 만드는 난수 시드")
    parser.add_argument("--init-from", type=Path, default=None, metavar="정책.pkl",
                        help="이전 학습의 정책에서 이어서 학습 (보상을 바꿔 걸음 다듬기). "
                             "예: output/learning/<YYMMDD_HHMMSS>_walk/params.pkl 또는 .../checkpoints/step_<스텝>.pkl")
    parser.add_argument("--watch", action="store_true",
                        help="학습 화면을 함께 띄움 (그래프 + 최신 정책으로 걷는 로봇, Enter로 전환) = play_walk.py --live")
    args = parser.parse_args()

    try:
        import jax
        from biped_sim.envs import walk_mjx   # Brax보다 먼저 (MJX import 안내 문구를 숨김)
        from brax.io import model as brax_model
        from brax.training.agents.ppo import networks as ppo_networks
        from brax.training.agents.ppo import train as ppo
        from tensorboardX import SummaryWriter
    except ImportError as e:
        raise SystemExit(f"GPU 학습 환경이 아닙니다 ({e}) → bash scripts/setup_gpu_learning.sh 후 "
                         "source scripts/activate_gpu.sh")

    if jax.default_backend() != "gpu":
        print(f"[warn] JAX가 GPU가 아니라 {jax.default_backend()}를 씁니다. 매우 느릴 수 있습니다.")

    if args.robot not in walk_mjx.SPECS:
        raise SystemExit(f"모르는 로봇 '{args.robot}'. 사용 가능: {', '.join(walk_mjx.SPECS)}")
    spec = walk_mjx.SPECS[args.robot]
    if not spec.robot.urdf.exists():
        raise SystemExit(f"로봇 파일이 없습니다: {spec.robot.urdf} (오리 로봇이면 bash scripts/get_open_duck.sh)")
    speed = spec.target_speed if args.speed is None else args.speed
    name = args.name or ("walk" if args.robot == "simple_biped" else "duck_walk")
    if args.init_from and not args.init_from.exists():
        raise SystemExit(f"--init-from 정책 파일이 없습니다: {args.init_from}")
    overrides = {}
    for item in args.reward:   # 결과 폴더를 만들기 전에 확인 (오타로 빈 폴더가 생기지 않게)
        key, _, value = item.partition("=")
        try:
            overrides[key.strip()] = float(value)
        except ValueError:
            raise SystemExit(f"--reward는 이름=가중치 형식입니다 (예: --reward symmetry=-2). 받은 값: {item}")
    unknown = sorted(set(overrides) - set(spec.reward_weights))
    if unknown:
        raise SystemExit(f"모르는 보상 항목 {unknown}. 사용 가능: {', '.join(spec.reward_weights)}")
    run_dir = args.output_dir.resolve() / f"{datetime.now():%y%m%d_%H%M%S}_{name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    latest = args.output_dir.resolve() / "walk_latest.pkl"
    config = {
        "run_dir": str(run_dir),
        "init_from": str(args.init_from.resolve()) if args.init_from else None,
        "robot": args.robot,
        "target_speed": speed, "steps": args.steps, "envs": args.envs, "seed": args.seed,
        "physics_dt": walk_mjx.PHYSICS_DT, "control_dt": walk_mjx.CONTROL_DT, "gait_period": spec.gait_period,
        "action_scale": spec.action_scale, "reward_weights": {**spec.reward_weights, **overrides},
        "terrain": None if args.terrain == "flat" else {"kind": "training", "seed": args.terrain_seed, "half_size": 6.0,
                                                        "resolution": 0.08},
        "policy_hidden": [128, 128, 128], "value_hidden": [256, 256, 256],
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))

    print("=" * 76)
    print(f"과제: {args.robot} 로봇이 앞으로 {speed} m/s로 걷기 (넘어지면 끝, 에피소드 최대 10 s)"
          + (" — 울퉁불퉁한 지형 12 × 12 m, 무작위 출발 위치" if args.terrain == "rough" else ""))
    print(f"학습: Brax PPO, {args.steps:,} 스텝, GPU 병렬 환경 {args.envs}개, 장치 {jax.devices()[0]}")
    print(f"보상 가중치: {config['reward_weights']}")
    if args.init_from:
        print(f"이어서 학습: {args.init_from} 의 정책에서 시작 (처음 평가 = 그 정책의 실력)")
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

    terrain, spawn_area = None, None
    if args.terrain == "rough":   # 넓은 지형의 무작위 위치에서 출발. 앞으로 2.5 m는 지형 안이도록
        from biped_sim.terrain import make_training_terrain
        # 격자 8 cm: 4 cm보다 GPU에서 2배 빠름 (발 상자가 검사할 칸이 25 → 9개, 실측 7,800 대 3,850 스텝/s)
        terrain = make_training_terrain(seed=args.terrain_seed, half_size=6.0, resolution=0.08)
        spawn_area = ((-5.5, 3.5), (-5.5, 5.5))
    env = walk_mjx.BipedWalkMjxEnv(spec, target_speed=speed, reward_weights=overrides, terrain=terrain,
                                   spawn_area=spawn_area)
    # flush_secs=5: 기록을 5초마다 파일에 씀 (기본 120초면 TensorBoard·학습 화면 그래프가 최대 2분 늦게 보임)
    writer = SummaryWriter(str(run_dir), flush_secs=5)
    t_start = time.perf_counter()
    episode_len = round(10.0 / walk_mjx.CONTROL_DT)   # 10 s = 500 제어 스텝

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
        # 좌우 다리 동작 차이: 반 박자 어긋나게 비교한 관절 각도 차이의 RMS [°] (0이면 두 다리가 똑같이 움직임)
        leg_asym = float(np.degrees(np.sqrt(gait["leg_asymmetry"] / spec.n)))
        writer.add_scalar("walk/leg_asymmetry_deg", leg_asym, step)
        writer.flush()
        print(f"  스텝 {step:>11,} / {args.steps:,} | 평균 보상 {float(metrics['eval/episode_reward']):7.2f} | "
              f"버틴 시간 {length * walk_mjx.CONTROL_DT:5.2f} s / 10 s | 앞으로 속도 {speed:+.2f} m/s | "
              f"두 발 공중 {100 * gait['flight']:3.0f}% | 좌우 다리 차이 {leg_asym:4.1f}° | "
              f"경과 {time.perf_counter() - t_start:5.0f} s", flush=True)

    def save_policy(step, make_policy, params):  # 평가 때마다 최신 정책 저장 (play_walk.py --live가 자동으로 다시 읽음)
        # 설정(json)을 먼저, 정책(pkl)을 나중에 바꿔야 재생 쪽이 '새 pkl + 옛 json'을 읽는 일이 없음
        (run_dir / "config.json").write_text(json.dumps({**config, "saved_step": int(step)}, indent=2,
                                                        ensure_ascii=False))
        shutil.copy(run_dir / "config.json", latest.with_suffix(".json"))
        tmp = run_dir / "params.pkl.tmp"
        brax_model.save_params(str(tmp), params)
        # 평가마다 따로 보관 → 학습 중간의 걸음이 더 좋으면 그것을 골라 쓸 수 있음 (analyze_gait.py --run)
        (run_dir / "checkpoints").mkdir(exist_ok=True)
        shutil.copy(tmp, run_dir / "checkpoints" / f"step_{int(step):011d}.pkl")
        shutil.copy(tmp, latest.with_suffix(".pkl.tmp"))
        # 원자적 교체(replace): 재생 쪽이 반쯤 쓰인 파일을 읽지 않게
        latest.with_suffix(".pkl.tmp").replace(latest)
        tmp.replace(run_dir / "params.pkl")

    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=tuple(config["policy_hidden"]),
        value_hidden_layer_sizes=tuple(config["value_hidden"]))

    ppo.train(
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
        policy_params_fn=save_policy,   # Brax가 평가 직전마다 부름 (마지막 평가 포함 → 끝난 뒤 따로 저장할 필요 없음)
        restore_params=brax_model.load_params(str(args.init_from)) if args.init_from else None,
    )
    writer.close()
    print(f"\n학습 완료: {time.perf_counter() - t_start:.0f} s")
    print(f"정책 저장: {run_dir / 'params.pkl'}  (최신: {latest})")
    print("화면으로 보기:  python learning/play_walk.py --view"
          + ("   (--watch 학습 화면은 창을 닫으면 끝납니다)" if args.watch else ""))


if __name__ == "__main__":
    main()
