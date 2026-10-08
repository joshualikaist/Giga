#!/usr/bin/env python3
"""가장 간단한 강화학습 예제 — "밀려도 넘어지지 않기"를 PPO로 학습

실행 (먼저 bash scripts/setup_learning.sh 로 학습 패키지 설치)
    python learning/train_balance.py                  # 50만 스텝, CPU로 약 4분
    python learning/train_balance.py --steps 20000    # 빠른 동작 확인 (약 15초, 학습 효과는 거의 없음)
    python learning/train_balance.py --steps 1000000  # 더 오래 학습
    python learning/train_balance.py --push-range 40 80   # 더 세게 밀며 학습

학습 과정을 실시간 그래프로 (다른 터미널에서, source scripts/activate.sh 후)
    tensorboard --logdir output/learning      → 웹 브라우저에서 http://localhost:6006
    보상 항목별 값(reward_terms/), 버틴 시간(episode/), 세기별 시험 결과(eval/), PPO 내부 값(train/)

결과 (output/learning/balance_<날짜_시각>/)
    model.zip           학습된 정책 (신경망)
    progress.csv        학습 기록 (스텝별 평균 보상, 에피소드 길이 등)
    events.out.tfevents.*  TensorBoard 기록
    learning_curve.png  학습 곡선 그래프
    → 다음: python learning/evaluate_balance.py --view      (학습한 정책을 MuJoCo 화면으로 보기)

과제·관측·행동·보상 설명: biped_sim/envs/balance.py, docs/07_learning.md
"""
import argparse
import shutil
import time
from datetime import datetime
from pathlib import Path

from biped_sim import paths
from biped_sim.envs import BipedBalanceEnv, evaluate_push_recovery, format_results, zero_policy

RUNS_DIR = paths.OUTPUT_DIR / "learning"   # 기본 저장 위치. 최신 모델은 balance_latest.zip (evaluate_balance.py 기본값)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=500_000, help="총 학습 스텝 수 (제어 스텝 기준, 1스텝 = 0.02 s)")
    parser.add_argument("--envs", type=int, default=8, help="동시에 돌릴 시뮬레이션 개수 (CPU 코어 수 이하 권장)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                        help="신경망 계산 장치. 이 예제는 cpu가 더 빠름 (docs/07 §6)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--push-range", type=float, nargs=2, default=(20.0, 60.0), metavar=("MIN", "MAX"),
                        help="학습 중 미는 힘 범위 [N] (기본 20 60)")
    parser.add_argument("--output-dir", type=Path, default=RUNS_DIR, help="결과 저장 폴더")
    parser.add_argument("--eval-every", type=int, default=50_000,
                        help="이 스텝마다 30/40/50 N 시험을 해서 TensorBoard eval/에 기록 (0이면 안 함)")
    args = parser.parse_args()

    # 학습 패키지는 무겁고 선택 설치라 여기서 import (없으면 친절한 안내)
    try:
        import torch
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.logger import configure
        from stable_baselines3.common.vec_env import SubprocVecEnv
    except ImportError:
        raise SystemExit("학습 패키지가 없습니다 → bash scripts/setup_learning.sh")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA(GPU)를 쓸 수 없습니다 → --device cpu")
    torch.set_num_threads(1)  # 작은 신경망은 스레드 1개가 오히려 빠름 (병렬은 환경 쪽에서)

    run_dir = args.output_dir.resolve() / f"balance_{datetime.now():%Y%m%d_%H%M%S}"
    latest_model = args.output_dir.resolve() / "balance_latest.zip"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    lo, hi = args.push_range
    print(f"과제: 몸통을 앞/뒤로 {lo:g}~{hi:g} N × 0.1초 밀 때 4초 동안 넘어지지 않기")
    print(f"학습: PPO, {args.steps:,} 스텝, 환경 {args.envs}개 병렬, 장치 {args.device}")
    print(f"저장: {run_dir}")
    print(f"실시간 그래프: 다른 터미널에서  tensorboard --logdir {args.output_dir}  → http://localhost:6006")
    print("=" * 72)

    # ① 환경: 같은 시뮬레이션을 여러 개(프로세스) 동시에 돌려 경험을 빨리 모은다
    venv = make_vec_env(BipedBalanceEnv, n_envs=args.envs, seed=args.seed, vec_env_cls=SubprocVecEnv,
                        env_kwargs={"push_force": (lo, hi)})

    # ② 알고리즘: PPO. 정책 = 관측(28) → [128, 128] 신경망 → 행동(6)
    #    log_std_init=-1: 처음 탐색 잡음을 작게(표준편차 0.37) → 학습 초반에도 PD 기본 동작 근처에서 시작
    model = PPO("MlpPolicy", venv, n_steps=256, batch_size=512, n_epochs=10, learning_rate=3e-4,
                gamma=0.99, gae_lambda=0.95,
                policy_kwargs=dict(net_arch=[128, 128], log_std_init=-1.0),
                device=args.device, seed=args.seed, verbose=0)
    # 기록: progress.csv + TensorBoard (rollout/, train/, time/ 은 PPO가, 나머지는 아래 콜백이 기록)
    model.set_logger(configure(str(run_dir), ["csv", "tensorboard"]))

    # ③ 학습. 진행 상황을 한 줄씩 출력하는 콜백
    class Progress(BaseCallback):
        def __init__(self):
            super().__init__()
            self.start = time.perf_counter()
            self.next_print = 0

        def _on_step(self):
            if self.num_timesteps >= self.next_print:
                self.next_print += max(args.steps // 20, 1)
                ep = self.model.ep_info_buffer
                if ep:
                    length = sum(e["l"] for e in ep) / len(ep)
                    reward = sum(e["r"] for e in ep) / len(ep)
                    print(f"  스텝 {self.num_timesteps:>9,} / {args.steps:,} | "
                          f"평균 버틴 시간 {length / 50:4.2f} s / 4.00 s | 평균 보상 {reward:6.1f} | "
                          f"경과 {time.perf_counter() - self.start:5.0f} s", flush=True)
            return True

    # ④ TensorBoard용 기록 콜백: 에피소드가 끝날 때마다 보상 항목별 합계 등을 모아, PPO가 매 롤아웃 끝에 기록
    class RewardTermsLogger(BaseCallback):
        def __init__(self):
            super().__init__()
            self.sums = [dict() for _ in range(args.envs)]
            self.next_eval = args.eval_every

        def _on_step(self):
            for i, (info, done) in enumerate(zip(self.locals["infos"], self.locals["dones"])):
                for name, value in info.get("reward_terms", {}).items():
                    self.sums[i][name] = self.sums[i].get(name, 0.0) + value
                if done:
                    # 에피소드 전체 동안 각 보상 항목이 얼마나 쌓였나 (어떤 항목이 점수를 좌우하는지 보임)
                    for name, total in self.sums[i].items():
                        self.logger.record_mean(f"reward_terms/{name}", total)
                    self.logger.record_mean("episode/survival_time_s", info["time"])
                    self.logger.record_mean("episode/fell", float(info["fell"]))
                    self.sums[i] = {}
            if args.eval_every and self.num_timesteps >= self.next_eval:
                self.next_eval += args.eval_every
                policy = lambda obs: self.model.predict(obs, deterministic=True)[0]  # noqa: E731
                results = evaluate_push_recovery(policy, forces=(30, 40, 50), push_times=(1.0,))
                for force, (ok, n) in results.items():
                    self.logger.record(f"eval/survive_{force}N", ok / n)
            return True

    t0 = time.perf_counter()
    model.learn(total_timesteps=args.steps, callback=[Progress(), RewardTermsLogger()])
    elapsed = time.perf_counter() - t0
    venv.close()

    model.save(run_dir / "model.zip")
    shutil.copy(run_dir / "model.zip", latest_model)
    print(f"\n학습 완료: {elapsed:.0f} s ({args.steps / elapsed:.0f} 스텝/s)")
    print(f"모델 저장: {run_dir / 'model.zip'}  (최신 모델 복사본: {latest_model})")

    plot = save_learning_curve(run_dir, args.push_range)
    print(f"학습 곡선: {plot}")

    # ⑤ 평가: 같은 조건(앞/뒤 × 미는 시각 3가지)으로 학습 전(PD만)과 비교
    print("\n평가 (세기마다 앞/뒤 × 미는 시각 3가지 = 6번 시도)")
    trained = evaluate_push_recovery(lambda obs: model.predict(obs, deterministic=True)[0])
    print(format_results(evaluate_push_recovery(zero_policy), trained))
    print("\n다음: python learning/evaluate_balance.py --view --push 40   (MuJoCo 화면으로 보기)")


def save_learning_curve(run_dir, push_range=(20.0, 60.0)):
    """progress.csv → 학습 곡선 그림 (x: 학습 스텝, y: 평균 버틴 시간)."""
    import csv

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(run_dir / "progress.csv") as f:
        rows = [r for r in csv.DictReader(f) if r.get("rollout/ep_len_mean")]
    steps = [int(float(r["time/total_timesteps"])) for r in rows]
    survive = [float(r["rollout/ep_len_mean"]) / 50 for r in rows]   # 제어 스텝 → 초
    reward = [float(r["rollout/ep_rew_mean"]) for r in rows]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.plot(steps, survive)
    ax1.axhline(4.0, ls="--", c="gray")
    ax1.set_ylabel("mean survival time [s]\n(max 4 s)")
    ax1.grid(alpha=0.3)
    ax2.plot(steps, reward, c="tab:orange")
    ax2.set_ylabel("mean episode reward")
    ax2.set_xlabel("training steps")
    ax2.grid(alpha=0.3)
    fig.suptitle(f"PPO learning curve: push recovery ({push_range[0]:g}-{push_range[1]:g} N)")
    fig.tight_layout()
    path = run_dir / "learning_curve.png"
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


if __name__ == "__main__":
    main()
