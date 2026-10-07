#!/usr/bin/env python3
"""학습한 정책 평가·재생 — "밀려도 넘어지지 않기"

실행
    python learning/evaluate_balance.py                        # 최근 학습 모델 vs PD만: 비교표
    python learning/evaluate_balance.py --pretrained           # 저장소에 들어 있는 미리 학습된 모델로 (학습 없이 바로)
    python learning/evaluate_balance.py --model 경로/model.zip  # 특정 모델로

    python learning/evaluate_balance.py --view --push 40              # MuJoCo 화면: 학습한 정책이 40 N을 버티는 모습
    python learning/evaluate_balance.py --view --push 40 --baseline   # 같은 상황에서 PD만 쓰면? (넘어짐)
    python learning/evaluate_balance.py --pretrained --view --push 40 # 미리 학습된 모델로 바로 보기

화면 모드에서는 앞으로 밀기 / 뒤로 밀기를 번갈아 반복합니다. 끝내려면 창을 닫거나 Ctrl+C.
"""
import argparse
import time
from pathlib import Path

from biped_sim import paths, passive_viewer, track_body_camera
from biped_sim.envs import BipedBalanceEnv, evaluate_push_recovery, format_results, zero_policy

LATEST_MODEL = paths.OUTPUT_DIR / "learning" / "balance_latest.zip"       # train_balance.py가 저장
PRETRAINED_MODEL = paths.REPO_ROOT / "learning" / "pretrained" / "balance_ppo.zip"


def load_policy(model_path):
    try:
        from stable_baselines3 import PPO
    except ImportError:
        raise SystemExit("학습 패키지가 없습니다 → bash scripts/setup_learning.sh")
    if not model_path.exists():
        raise SystemExit(f"모델이 없습니다: {model_path}\n"
                         "→ 먼저 python learning/train_balance.py 로 학습하거나, --pretrained 를 붙이세요.")
    model = PPO.load(model_path, device="cpu")
    print(f"모델: {model_path}")
    return lambda obs: model.predict(obs, deterministic=True)[0]


def view(policy, force, label):
    """MuJoCo 화면으로 재생: 앞/뒤로 번갈아 밀기, 에피소드마다 결과 출력."""
    env = BipedBalanceEnv()
    dt = 1.0 / env.metadata["render_fps"]  # 제어 주기 0.02 s
    direction = 1.0
    print(f"\n[{label}] {force} N으로 0.1초 밀기를 앞/뒤 번갈아 반복합니다. 종료: 창 닫기 또는 Ctrl+C")
    try:
        with passive_viewer(env.model, env.data) as viewer:
            with viewer.lock():
                track_body_camera(env.robot.base_body_id, distance=2.2, azimuth=90, elevation=-10)(viewer)
            episode = 0
            while viewer.is_running():
                obs, info = env.reset(options={"push_force": force, "push_direction": direction, "push_time": 1.0})
                viewer.sync()
                episode += 1
                while viewer.is_running():
                    t0 = time.perf_counter()
                    obs, _, terminated, truncated, info = env.step(policy(obs))
                    viewer.sync()
                    time.sleep(max(0.0, dt - (time.perf_counter() - t0)))  # 실제 시간 속도로
                    if terminated or truncated:
                        break
                where = "앞으로" if direction > 0 else "뒤로"
                result = "넘어짐 ❌" if info["fell"] else "버팀 ✅"
                print(f"  #{episode}: {where} {force} N → {result}  (t = {info['time']:.2f} s)")
                time.sleep(0.8)  # 결과를 볼 수 있게 잠깐 멈춤
                direction = -direction
    except KeyboardInterrupt:
        print("\nCtrl+C — 종료합니다.")
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=str, default=None, help="평가할 모델 파일 (기본: 최근 학습 모델)")
    parser.add_argument("--pretrained", action="store_true", help="저장소의 미리 학습된 모델 사용")
    parser.add_argument("--view", action="store_true", help="MuJoCo 화면으로 재생")
    parser.add_argument("--push", type=float, default=40.0, help="화면 모드에서 미는 힘 [N]")
    parser.add_argument("--baseline", action="store_true", help="화면 모드에서 학습 정책 대신 PD만 사용")
    args = parser.parse_args()

    if args.view and args.baseline:
        view(zero_policy, args.push, "PD만 (학습 전)")
        return

    if args.pretrained:
        model_path = PRETRAINED_MODEL
    elif args.model:
        model_path = Path(args.model).expanduser().resolve()  # 상대 경로는 현재 폴더 기준
    else:
        model_path = LATEST_MODEL
    policy = load_policy(model_path)

    if args.view:
        view(policy, args.push, "학습한 정책")
        return

    print("\n평가 (세기마다 앞/뒤 × 미는 시각 3가지 = 6번 시도, 매번 같은 조건)")
    print(format_results(evaluate_push_recovery(zero_policy), evaluate_push_recovery(policy)))


if __name__ == "__main__":
    main()
