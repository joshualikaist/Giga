#!/usr/bin/env python3
"""GPU 보행 학습 — MuJoCo MJX(GPU 병렬 시뮬레이션) + Brax PPO

실행 (GPU 학습 환경에서. 처음 한 번: bash scripts/setup_gpu_learning.sh)
    source scripts/activate_gpu.sh
    python learning/train_walk_gpu.py --watch               # 기본 3,000만 스텝 + 학습 화면(그래프·로봇) 함께
    python learning/train_walk_gpu.py --robot open_duck_mini --watch   # 오리 로봇 (bash scripts/get_open_duck.sh 먼저)
    python learning/train_walk_gpu.py --robot open_duck_mini --terrain rough --init-from <평지 정책.pkl>  # 울퉁불퉁한 지형
    python learning/train_walk_gpu.py --robot open_duck_mini --servo open_duck --friction 0.4 1.0 \
        --terrain rough --init-from <정책.pkl>             # 실물에 가까운 서보 모델 + 바닥 마찰 무작위 (docs/09 §9)
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
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from gpu_env import ensure_gpu_env

from biped_sim import paths

RUNS_DIR = paths.OUTPUT_DIR / "learning"
WATCH_PID_FILE = ".watch_window.pid"   # --watch로 띄운 학습 화면의 프로세스 번호 (다음 학습이 닫을 때 씀)


def close_previous_watch(pid_file: Path) -> bool:
    """지난 --watch 학습이 띄운 학습 화면(play_walk.py --live)이 아직 떠 있으면 닫음. 닫았으면 True.
    pid 파일에 적힌 프로세스만, 그리고 정말 play_walk.py --live일 때만 닫음 (사용자가 직접 띄운 창은 그대로)."""
    try:
        pid = int(pid_file.read_text())
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except (OSError, ValueError):   # 파일 없음 / 이미 꺼진 프로세스 / /proc 없는 OS
        return False
    if not any(arg.endswith(b"play_walk.py") for arg in cmdline) or b"--live" not in cmdline:
        return False                # 번호가 다른 프로그램에 다시 쓰인 경우
    try:
        os.kill(pid, 15)            # SIGTERM
    except OSError:
        return False
    return True


def add_command_inputs(params, base_old: int, history: int, privileged: bool, commands: dict):
    """목표 속도 1칸으로 학습한 정책 → 명령 3칸(vx, vy, wz) 정책. 관측 블록(지금 + 기록)마다 목표 속도 칸(= base_old − 3, vx로
    그대로 씀) 뒤에 vy·wz 두 칸을 끼움: 정책 첫 층 가중치는 0 행, 관측 통계는 명령 범위에 맞춤. 예전 목표 속도 칸은 늘 같은 값이라
    σ가 0에 가까웠으므로 명령 범위의 평균·σ로 바꿈 (vx = 그 평균이면 정규화 값 0 → 예전과 같은 입력). critic은 새로 만듦."""
    import jax.numpy as jnp
    import numpy as np
    norm, policy, value = params
    blocks = 1 + history

    def insert(x, values):   # 블록마다 목표 속도 칸 뒤(sin·cos 앞)에 values 2개를 끼움
        x = np.asarray(x)
        out = []
        for b in range(blocks):
            blk = x[b * base_old:(b + 1) * base_old]
            out.append(np.concatenate([blk[:base_old - 2], np.broadcast_to(values, (2,) + blk.shape[1:]), blk[base_old - 2:]]))
        return jnp.asarray(np.concatenate(out))

    layer = policy["params"]["hidden_0"]
    zeros_row = np.zeros((2, layer["kernel"].shape[1]))
    kernel = []
    k = np.asarray(layer["kernel"])
    for b in range(blocks):
        blk = k[b * base_old:(b + 1) * base_old]
        kernel.append(np.concatenate([blk[:base_old - 2], zeros_row, blk[base_old - 2:]]))
    policy = {"params": {**policy["params"], "hidden_0": {**layer, "kernel": jnp.asarray(np.concatenate(kernel))}}}

    # 명령 칸 통계: 범위 [lo, hi]의 균일 분포 평균·σ (σ가 0이면 0.05로)
    def stats(key):
        lo, hi = commands[key]
        return (lo + hi) / 2, max((hi - lo) / np.sqrt(12), 0.05)
    (mx, sx), (my, sy), (mw, sw) = stats("vx"), stats("vy"), stats("wz")

    def widen(mean, std, var, single):
        n_blocks = 1 if single else blocks
        mean, std, var = np.array(mean), np.array(std), np.array(var)
        count = float(var[10] / std[10] ** 2)   # 관절 각도 칸으로 지금까지의 표본 수 추정
        for b in range(n_blocks):               # 예전 목표 속도 칸 → vx 명령의 통계
            i = b * base_old + base_old - 3
            mean[i], std[i], var[i] = mx, sx, sx ** 2 * count
        out = []
        for b in range(n_blocks):
            sl = slice(b * base_old, (b + 1) * base_old)
            out.append([np.concatenate([a[sl][:base_old - 2], new, a[sl][base_old - 2:]])
                        for a, new in ((mean, [my, mw]), (std, [sy, sw]), (var, [sy ** 2 * count, sw ** 2 * count]))])
        return [jnp.asarray(np.concatenate([o[j] for o in out])) for j in range(3)]

    if privileged:
        st = widen(norm.mean["state"], norm.std["state"], norm.summed_variance["state"], single=False)
        pr = widen(norm.mean["privileged_state"], norm.std["privileged_state"], norm.summed_variance["privileged_state"],
                   single=True)
        norm = norm.replace(mean={"state": st[0], "privileged_state": pr[0]}, std={"state": st[1], "privileged_state": pr[1]},
                            summed_variance={"state": st[2], "privileged_state": pr[2]})
    else:
        st = widen(norm.mean, norm.std, norm.summed_variance, single=False)
        norm = norm.replace(mean=st[0], std=st[1], summed_variance=st[2])
    return norm, policy, value


def add_history_inputs(params, base_size: int, history: int, privileged: bool):
    """기록 없이 학습한 정책 → 관측 기록 history 스텝을 받는 정책. 정책 첫 층 가중치에 새 입력(지난 관측)용 행을 0으로
    붙이고(처음엔 같은 행동), 관측 통계는 지금 관측의 통계를 지난 관측 칸에도 복사 (같은 종류의 값이라서)."""
    import jax.numpy as jnp
    norm, policy, value = params
    layer = policy["params"]["hidden_0"]
    kernel = jnp.concatenate([layer["kernel"], jnp.zeros((base_size * history, layer["kernel"].shape[1]))])
    policy = {"params": {**policy["params"], "hidden_0": {**layer, "kernel": kernel}}}
    tile = lambda x: jnp.tile(x, 1 + history)   # noqa: E731

    def widen(stats):   # 비대칭 critic이면 정책 쪽("state")만 늘림 (critic은 지금 관측만)
        return {**stats, "state": tile(stats["state"])} if privileged else tile(stats)
    norm = norm.replace(mean=widen(norm.mean), std=widen(norm.std), summed_variance=widen(norm.summed_variance))
    return norm, policy, value


def to_privileged_params(params):
    """관측 하나로 학습한 정책 파라미터 → 비대칭 critic 형식 {"state", "privileged_state"}.
    정책·관측 통계는 그대로 'state'에 쓰고, critic용 'privileged_state' 통계도 같은 값에서 시작 (실물 관측 정책도 가린 칸의
    통계는 그 전 학습에서 이어받은 값이라 크기가 맞음: 속도 σ ≈ 0.05 m/s, 높이 σ ≈ 0.003 m). critic 가중치는 새로 (restore_value_fn=False)."""
    norm, policy, value = params
    both = lambda x: {"state": x, "privileged_state": x}   # noqa: E731
    return (norm.replace(mean=both(norm.mean), std=both(norm.std), summed_variance=both(norm.summed_variance)),
            policy, value)


def main():
    ensure_gpu_env()   # ROS용 .venv에서 실행했으면 .venv-mjx로 다시 실행
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
    parser.add_argument("--servo", default="ideal",
                        help="서보 모델: ideal(토크가 속도와 상관없음, 기본) / open_duck(실물 측정 모델: 빨리 돌수록 토크 감소). "
                             "biped_sim/envs/walk_mjx.py SERVO_MODELS")
    parser.add_argument("--friction", type=float, nargs=2, default=None, metavar=("최소", "최대"),
                        help="환경마다 바닥 마찰 계수를 이 범위에서 무작위로 (기본: 1.0 고정)")
    parser.add_argument("--gait-period", type=float, default=None, metavar="S",
                        help="걸음 박자 [s] (오리 기본 0.5). 길면 발을 들어 앞으로 가져올 시간이 늘어 더 높이 들 수 있음 (docs/09 §8.3)")
    parser.add_argument("--terrain-difficulty", type=float, nargs=2, default=None, metavar=("최소", "최대"),
                        help="rough 학습 지형의 험한 정도 범위 (기본 0.4 1.2). 시험(1.5, 2.0)에 가깝게 올리면 더 험한 곳을 연습")
    parser.add_argument("--push", type=float, nargs=2, default=None, metavar=("최대속도", "간격"),
                        help="외란: '간격'초마다 몸통을 무작위 방향으로 0~'최대속도'[m/s]만큼 밈 (예: 0.15 2). 실물에서 부딪힘·흔들림 대비")
    parser.add_argument("--deployable-obs", action="store_true",
                        help="실물에서 잴 수 있는 관측만 (몸통 속도·높이를 0으로 가림) + 센서 잡음. 실물 로봇에 올릴 정책용 (docs/10)")
    parser.add_argument("--privileged-critic", action="store_true",
                        help="비대칭 actor-critic: 정책은 실물 관측만, 학습 중 점수만 매기는 critic은 시뮬레이션의 모든 정보 "
                             "(--deployable-obs와 함께. 실물에 올리는 정책은 그대로)")
    parser.add_argument("--cmd-vx", type=float, nargs=2, default=None, metavar=("최소", "최대"),
                        help="명령 따라 걷기: 앞뒤 속도 명령 범위 [m/s] (음수 = 뒤로). --cmd-* 중 하나라도 주면 명령 모드")
    parser.add_argument("--cmd-vy", type=float, nargs=2, default=None, metavar=("최소", "최대"), help="옆 속도 명령 범위 [m/s]")
    parser.add_argument("--cmd-wz", type=float, nargs=2, default=None, metavar=("최소", "최대"), help="회전 속도 명령 범위 [rad/s]")
    parser.add_argument("--cmd-resample", type=float, default=5.0, metavar="S", help="학습 중 명령을 바꾸는 간격 [s]")
    parser.add_argument("--cmd-zero-prob", type=float, default=0.0, metavar="P",
                        help="명령 성분마다 이 확률로 정확히 0 (제자리 돌기·옆으로만·뒤로만 같은 명령을 자주 연습)")
    parser.add_argument("--double-support-fast", type=float, default=None, metavar="DS",
                        help="빠른 앞뒤 명령에서 gait_phase 보상의 양발 지지 폭 (기본 0.1 = 한 주기의 40 %%가 양발, 디딤 비율 0.7). "
                             "명령 |vx|가 목표 속도(오리 0.15) → --fast-speed로 갈수록 이 값으로 줄임 (0.05면 디딤 비율 0.6: "
                             "서보 속도로 낼 수 있는 최고 속도 0.22~0.28 → 0.34~0.43 m/s, docs/09 §8.5)")
    parser.add_argument("--clearance-target", type=float, default=None, metavar="M",
                        help="foot_clearance 보상이 만점이 되는 흔듦 최고 발 높이 [m] (오리 기본 0.03). 지금 정책은 1.85 cm까지만 들어"
                             "보상이 포화되지 않고, 그 들기의 96 %%가 골반 기울기(몸통 구르기)에서 나옴 → 0.017로 낮추면 압박이 사라짐")
    parser.add_argument("--lateral-deadband", type=float, default=None, metavar="V",
                        help="명령 추종 오차에서 봐주는 옆 속도 [m/s] (기본 0 = 전부 벌점). 몸통을 수평으로 두고 무게중심을 좌우로 "
                             "옮기는 정상 걸음은 옆 속도가 최대 약 0.15 m/s라, dead-band가 없으면 '구르기'가 더 싸짐 (docs/10 §12)")
    parser.add_argument("--roll-free", type=float, default=None, metavar="RAD",
                        help="roll_excess 벌점이 봐주는 몸통 구르기 [rad] (기본 0.14 = 8°)")
    parser.add_argument("--head-sway-gate", type=float, nargs=2, default=None, metavar=("뒤", "회전"),
                        help="head_sway 벌점을 빼 주는 명령: 뒤로 이 속도 [m/s] 이상, 회전 이 속도 [rad/s] 이상이면 벌점 0 "
                             "(사이는 직선). 예: 0.05 0.3 — 뒤로·제자리 돌기가 벌점을 피하려 '서 있기'를 고르지 않게 (docs/10 6차)")
    parser.add_argument("--fast-speed", type=float, default=0.375, metavar="V",
                        help="--double-support-fast가 다 적용되는 앞뒤 명령 속도 [m/s]")
    parser.add_argument("--obs-history", type=int, default=0, metavar="H",
                        help="정책 관측에 지난 H 제어 스텝의 관측을 이어 붙임 (지난 움직임에서 실물 차이를 짐작). "
                             "기록 없이 학습한 정책에서 이어 가면 새 입력의 가중치를 0으로 시작 (처음엔 같은 행동)")
    parser.add_argument("--domain-rand", action="store_true",
                        help="실물과의 차이 대비: 환경마다 몸통 질량 ±15 %%, 모터 세기 ±15 %%, 무게중심 ±1 cm, 제어 지연 0/20 ms "
                             "(walk_mjx.DOMAIN_RAND)")
    parser.add_argument("--motor-range", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                        help="--domain-rand의 모터 세기 배율 범위 (기본 0.85 1.15). 배터리 처짐 대비로 아래를 넓힘 (docs/10 §15)")
    parser.add_argument("--home-knee", type=float, default=None, metavar="RAD",
                        help="서 있는 자세의 무릎 각도 (오리 기본 0.8). 덜 굽히면 한 발 버티는 무릎 토크가 줄어듦 (docs/09 §9.5)")
    parser.add_argument("--init-from", type=Path, default=None, metavar="정책.pkl",
                        help="이전 학습의 정책에서 이어서 학습 (보상을 바꿔 걸음 다듬기). "
                             "예: output/learning/<YYMMDD_HHMMSS>_walk/params.pkl 또는 .../checkpoints/step_<스텝>.pkl")
    parser.add_argument("--no-eval", action="store_true",
                        help="학습이 끝난 뒤 자동 평가(evaluate_walk.py: 지형·토크·걸음, 이전 정책과 비교)를 하지 않음")
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
    if args.servo not in walk_mjx.SERVO_MODELS:
        raise SystemExit(f"모르는 서보 모델 '{args.servo}'. 사용 가능: {', '.join(walk_mjx.SERVO_MODELS)}")
    spec = walk_mjx.with_servo(walk_mjx.SPECS[args.robot], args.servo)
    if args.home_knee is not None:
        spec = walk_mjx.with_home_knee(spec, args.home_knee)
    if args.gait_period is not None:
        import dataclasses
        spec = dataclasses.replace(spec, gait_period=args.gait_period)
    if args.deployable_obs:
        import dataclasses
        spec = dataclasses.replace(spec, deployable_obs=True)
    if args.obs_history:
        import dataclasses
        spec = dataclasses.replace(spec, obs_history=args.obs_history)
    if args.double_support_fast is not None:
        import dataclasses
        spec = dataclasses.replace(spec, double_support_fast=args.double_support_fast, fast_speed=args.fast_speed)
    if args.head_sway_gate is not None:
        import dataclasses
        spec = dataclasses.replace(spec, head_sway_gate=tuple(args.head_sway_gate))
    import dataclasses
    for name in ("clearance_target", "lateral_deadband", "roll_free"):
        if getattr(args, name) is not None:
            spec = dataclasses.replace(spec, **{name: getattr(args, name)})
    commands = None
    if args.cmd_vx or args.cmd_vy or args.cmd_wz:   # 명령 따라 걷기
        import dataclasses
        spec = dataclasses.replace(spec, command_obs=True)
        commands = {"vx": tuple(args.cmd_vx or walk_mjx.COMMANDS_DEFAULT["vx"]), "vy": tuple(args.cmd_vy or (0.0, 0.0)),
                    "wz": tuple(args.cmd_wz or (0.0, 0.0)), "resample": args.cmd_resample, "zero_prob": args.cmd_zero_prob}
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
    # Brax는 평가 간격마다 '학습 단위'(배치 256 × 펼침 20 × 미니배치 16 = 81,920 스텝)를 정수 개로 올려서 돈다
    # → 실제 스텝은 요청보다 조금 많다 (예: 2,000만 요청 → 30 간격 × 9단위 × 81,920 = 22,118,400)
    unit, intervals = 256 * 20 * 16, max(args.evals - 1, 1)
    total_steps = intervals * math.ceil(args.steps / (intervals * unit)) * unit
    run_dir = args.output_dir.resolve() / f"{datetime.now():%y%m%d_%H%M%S}_{name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    latest = args.output_dir.resolve() / "walk_latest.pkl"
    config = {
        "run_dir": str(run_dir),
        "init_from": str(args.init_from.resolve()) if args.init_from else None,
        "robot": args.robot,
        "target_speed": speed, "steps": total_steps, "steps_requested": args.steps, "envs": args.envs, "seed": args.seed,
        "physics_dt": walk_mjx.PHYSICS_DT, "control_dt": walk_mjx.CONTROL_DT, "gait_period": spec.gait_period,
        "action_scale": spec.action_scale, "reward_weights": {**spec.reward_weights, **overrides},
        "servo": args.servo, "friction_range": args.friction, "home_knee": args.home_knee, "push": args.push,
        "deployable_obs": args.deployable_obs, "privileged_critic": args.privileged_critic,
        "domain_rand": ({**walk_mjx.DOMAIN_RAND, **({"motor_range": list(args.motor_range)} if args.motor_range else {})}
                        if args.domain_rand else None), "obs_history": args.obs_history,
        "command_obs": commands is not None, "commands": commands,
        "double_support_fast": args.double_support_fast, "fast_speed": args.fast_speed,
        "head_sway_gate": args.head_sway_gate, "clearance_target": spec.clearance_target,
        "lateral_deadband": spec.lateral_deadband, "roll_free": spec.roll_free,
        "terrain": None if args.terrain == "flat" else {"kind": "training", "seed": args.terrain_seed, "half_size": 6.0,
                                                        "resolution": 0.08,
                                                        "difficulty": list(args.terrain_difficulty or (0.4, 1.2))},
        "policy_hidden": [128, 128, 128], "value_hidden": [256, 256, 256],
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))

    print("=" * 76)
    print(f"과제: {args.robot} 로봇이 앞으로 {speed} m/s로 걷기 (넘어지면 끝, 에피소드 최대 10 s)"
          + (" — 울퉁불퉁한 지형 12 × 12 m, 무작위 출발 위치" if args.terrain == "rough" else ""))
    print(f"학습: Brax PPO, {total_steps:,} 스텝 (요청 {args.steps:,} → Brax가 평가 간격마다 {unit:,} 스텝 단위로 올림), "
          f"GPU 병렬 환경 {args.envs}개, 장치 {jax.devices()[0]}")
    print(f"보상 가중치: {config['reward_weights']}")
    print(f"서보 모델: {args.servo}" + (f", 바닥 마찰 {args.friction[0]}~{args.friction[1]} 무작위" if args.friction else "")
          + (f", 서 있는 자세 무릎 {args.home_knee} rad (기준 높이 {spec.nominal_height:.3f} m)" if args.home_knee else "")
          + (f", 밀기 {args.push[1]:g}초마다 최대 {args.push[0]:g} m/s" if args.push else "")
          + (", 실물 관측만(속도·높이 가림)+센서 잡음" if args.deployable_obs else "")
          + (", 비대칭 critic(전체 정보)" if args.privileged_critic else "")
          + (", 질량·모터·무게중심·지연 무작위" if args.domain_rand else "")
          + (f", 관측 기록 {args.obs_history}스텝" if args.obs_history else "")
          + (f"\n명령 따라 걷기: vx {commands['vx']}, vy {commands['vy']} m/s, wz {commands['wz']} rad/s, "
             f"{commands['resample']:g}초마다 바뀜" if commands else ""))
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
        pid_file = Path(args.output_dir) / WATCH_PID_FILE
        if close_previous_watch(pid_file):  # 지난 학습이 띄운 창이 남아 있으면 CPU를 나눠 쓰므로 닫음
            print("지난 학습의 학습 화면을 닫았습니다 (새 학습 화면으로 바꿈)")
        watch_log = open(run_dir / "watch.log", "w")
        viewer = subprocess.Popen([sys.executable, "-u", str(Path(__file__).with_name("play_walk.py")), "--live",
                                   "--params", str(run_dir / "params.pkl")], stdout=watch_log, stderr=subprocess.STDOUT)
        pid_file.write_text(str(viewer.pid))

    terrain, spawn_area = None, None
    if args.terrain == "rough":   # 넓은 지형의 무작위 위치에서 출발. 앞으로 2.5 m는 지형 안이도록
        from biped_sim.terrain import make_training_terrain
        # 격자 8 cm: 4 cm보다 GPU에서 2배 빠름 (발 상자가 검사할 칸이 25 → 9개, 실측 7,800 대 3,850 스텝/s)
        terrain = make_training_terrain(seed=args.terrain_seed, half_size=6.0, resolution=0.08,
                                        difficulty=tuple(args.terrain_difficulty or (0.4, 1.2)))
        spawn_area = ((-5.5, 3.5), (-5.5, 5.5))
    env = walk_mjx.BipedWalkMjxEnv(spec, target_speed=speed, reward_weights=overrides, terrain=terrain,
                                   spawn_area=spawn_area, friction_range=args.friction,
                                   push=tuple(args.push) if args.push else None, obs_noise=args.deployable_obs,
                                   privileged_critic=args.privileged_critic, domain_rand=args.domain_rand,
                                   motor_range=tuple(args.motor_range) if args.motor_range else None, commands=commands)
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
        com_shift_cm = 100.0 * gait["com_shift"] / max(gait["single"], 1e-6)   # 한 발 지지 동안 평균
        writer.add_scalar("walk/com_shift_cm", com_shift_cm, step)
        clearance_cm = 100.0 * gait["clearance"] / max(gait["landing"], 1e-6)  # 걸음마다 흔듦 최고 발 높이 평균
        writer.add_scalar("walk/foot_clearance_cm", clearance_cm, step)
        writer.add_scalar("walk/servo_over_pct", 100.0 * gait["servo_over"], step)   # 데이터시트 한계 밖 [%]
        writer.add_scalar("walk/track_lin_mps", gait["track_lin"], step)    # 속도 명령과의 차이 (명령 모드가 아니면 목표 속도와)
        writer.add_scalar("walk/track_yaw_rps", gait["track_yaw"], step)    # 회전 명령과의 차이
        head_sway = float(np.sqrt(gait["head_sway"]))                       # 머리 옆 속도 RMS [m/s]
        writer.add_scalar("walk/head_sway_mps", head_sway, step)
        roll_deg = float(np.degrees(gait["roll_abs"]))                      # 몸통 구르기 평균 |각도| [°]
        writer.add_scalar("walk/roll_abs_deg", roll_deg, step)
        writer.flush()
        print(f"  스텝 {step:>11,} / {total_steps:,} | 평균 보상 {float(metrics['eval/episode_reward']):7.2f} | "
              f"버틴 시간 {length * walk_mjx.CONTROL_DT:5.2f} s / 10 s | 앞으로 속도 {speed:+.2f} m/s | "
              f"두 발 공중 {100 * gait['flight']:3.0f}% | 좌우 다리 차이 {leg_asym:4.1f}° | 무게중심 이동 {com_shift_cm:4.1f} cm | 발 들기 {clearance_cm:4.1f} cm | 서보 한계 밖 {100 * gait['servo_over']:4.1f}% | "
              + (f"명령 오차 {gait['track_lin']:.3f} m/s·{gait['track_yaw']:.2f} rad/s | " if commands else "")
              + (f"머리 흔들림 {head_sway:.2f} m/s | 구르기 {roll_deg:.0f}° | " if spec.head_body else "")
              + f"경과 {time.perf_counter() - t_start:5.0f} s", flush=True)

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
        value_hidden_layer_sizes=tuple(config["value_hidden"]),
        **(dict(policy_obs_key="state", value_obs_key="privileged_state") if args.privileged_critic else {}))
    restore, restore_value = None, True
    if args.init_from:
        restore = brax_model.load_params(str(args.init_from))
        init_config = json.loads((args.init_from.parent.parent / "config.json" if args.init_from.parent.name == "checkpoints"
                                  else args.init_from.with_name("config.json")).read_text())
        if args.privileged_critic and not init_config.get("privileged_critic"):
            restore, restore_value = to_privileged_params(restore), False
            print("이어서 학습: 정책·관측 통계는 그대로, critic은 새로 (비대칭 critic으로 바꿈)")
        if commands is not None and not init_config.get("command_obs"):
            restore = add_command_inputs(restore, spec.base_obs_size - 2, args.obs_history, args.privileged_critic, commands)
            restore_value = False
            print("이어서 학습: 목표 속도 1칸 → 명령 3칸 (새 입력 가중치 0, 명령 vx = 0.15면 처음엔 같은 행동), critic은 새로")
        old_h = int(init_config.get("obs_history") or 0)
        if args.obs_history != old_h:
            if old_h:
                raise SystemExit(f"관측 기록 {old_h} → {args.obs_history}로는 이어 갈 수 없습니다 (0에서 늘리는 것만)")
            restore = add_history_inputs(restore, spec.base_obs_size, args.obs_history, args.privileged_critic)
            print(f"이어서 학습: 관측 기록 {args.obs_history}스텝 추가 (새 입력 가중치 0 → 처음엔 같은 행동)")

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
        restore_params=restore,
        restore_value_fn=restore_value,
    )
    writer.close()
    # 학습 화면(play_walk --live)이 '학습 끝, 마지막 정책 재생 중'으로 바꿔 보이게
    saved = json.loads((run_dir / "config.json").read_text())
    (run_dir / "config.json").write_text(json.dumps({**saved, "finished": True}, indent=2, ensure_ascii=False))
    print(f"\n학습 완료: {time.perf_counter() - t_start:.0f} s")
    print(f"정책 저장: {run_dir / 'params.pkl'}  (최신: {latest})")
    print("화면으로 보기:  python learning/play_walk.py --view"
          + ("   (--watch 학습 화면은 마지막 정책을 계속 재생합니다. 창을 닫으면 끝)" if args.watch else ""))
    if not args.no_eval:   # 자동 평가: 지형 시험·토크·걸음 분석, 이어서 학습했으면 시작 정책과 비교 (CPU, 약 5분)
        print("\n자동 평가 시작 (약 5분, 끄려면 --no-eval): 지형 시험·토크·걸음, "
              + ("시작 정책과 비교" if args.init_from else "비교 없이"), flush=True)
        done = subprocess.run([sys.executable, "-u", str(Path(__file__).with_name("evaluate_walk.py")),
                               "--params", str(run_dir / "params.pkl")], env={**os.environ, "JAX_PLATFORMS": "cpu"})
        if done.returncode == 0:
            print(f"평가 보고서: {run_dir / 'eval' / 'report.md'}  (TensorBoard TEXT 탭에도)")
        else:
            print(f"[알림] 자동 평가 실패 (exit {done.returncode}). 따로 실행: python learning/evaluate_walk.py "
                  f"--params {run_dir / 'params.pkl'}")


if __name__ == "__main__":
    main()
