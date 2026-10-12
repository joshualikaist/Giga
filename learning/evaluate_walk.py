#!/usr/bin/env python3
"""학습 뒤 자동 평가 — 지형 시험·토크(서보 한계)·걸음 분석을 한 번에 하고, 이전 정책(기준)과 비교한 보고서를 만든다.

train_walk_gpu.py가 학습을 마치면 자동으로 실행한다 (--no-eval로 끔). 따로 실행:
    python learning/evaluate_walk.py --params output/learning/<실행>/params.pkl          # 기준 = 학습 설정의 init_from
    python learning/evaluate_walk.py --params A.pkl --baseline B.pkl                     # 기준을 직접 고름
    python learning/evaluate_walk.py --params A.pkl --no-baseline --quick                # 빠르게 확인 (1분 안팎)

결과: <실행 폴더>/eval/  (체크포인트를 주면 <실행 폴더>/eval_<스텝>/)
    report.md        판정 + 표 (TensorBoard의 TEXT 탭에도 들어감)
    report.json      모든 수치
    filmstrip.png, gait.png, walk.mp4   평지 10초 걸음 (analyze_gait.py와 같음)
    torque_speed.png                    토크-속도 측정점과 서보 한계선 (오리 로봇만, feasibility.py와 같음)

공정한 비교를 위해 두 정책을 같은 지형(같은 시드), 같은 마찰, 같은 서보 모델(새 정책의 것)에서 시험한다.
서 있는 자세(home_knee)는 정책마다 자기 것을 쓴다 (정책의 행동이 그 자세를 기준으로 한 값이라서).
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from analyze_gait import record_with_frames, save_filmstrip, save_plots, save_video
from terrain_trial import run_trials
from walk_tools import gait_metrics, get_spec, load_policy, make_runner, run_episode

from biped_sim.terrain import KINDS, make_terrain

GROUPS = ("hip_roll", "hip_pitch", "knee", "ankle", "hip_yaw")   # 토크 표의 관절 순서 (오리)
# 기본 시험 조건. 이 조건으로 평가했을 때만 챔피언(지금까지 가장 잘 걷는 정책)과 비교·갱신한다 (점수를 견줄 수 있게)
DEFAULTS = dict(episodes=10, difficulties=[1.5, 2.0], frictions=[1.0, 0.5], kinds=list(KINDS), seed=0)
CHAMPION_DIR = Path(__file__).resolve().parents[1] / "output" / "learning"
PUSH_TEST = dict(speeds=(0.3, 0.4), interval=2.0)   # 밀기 시험: 평지에서 2초마다 몸통을 무작위 방향으로 툭 (보행 점수와 별도)
# 모델 오차 시험: 로봇을 일부러 시뮬레이션과 다르게 해서 섞인 코스(험한 정도 1.5)를 걸림 (실물과의 차이 대비, 보행 점수와 별도)
MODEL_ERROR_TEST = {
    "heavy_weak_slow": dict(mass=1.15, motor=0.85, com=(0.01, 0.0), delay=2),    # 몸통 +15 %, 모터 −15 %, 무게중심 앞 1 cm, 40 ms 지연
    "light_strong": dict(mass=0.85, motor=1.15, com=(-0.01, 0.0), delay=1),      # 몸통 −15 %, 모터 +15 %, 무게중심 뒤 1 cm, 20 ms 지연
    # 조립 오차: 관절마다 서보 영점 ±0.03 rad(≈2°, 고정된 무작위 패턴), IMU가 앞·옆으로 2°씩 기울게 붙음, 20 ms 지연
    "miscalibrated": dict(mass=1.0, motor=1.0, com=(0.0, 0.0), delay=1, joint_offset=0.03, imu_tilt=(0.035, 0.035)),
}
# 명령 시험 (명령 따라 걷기 정책): 명령마다 평지 1번 + 섞인 코스(험한 정도 1.0) 2번, 10초씩.
# 성공 = 10초 버팀 + 2초 뒤부터의 평균 속도(몸통 방향 기준)가 명령에서 ±0.03 m/s, 회전 속도가 ±0.1 rad/s 안
COMMAND_TEST = [("forward 0.05", (0.05, 0.0, 0.0)), ("forward 0.10", (0.10, 0.0, 0.0)), ("forward 0.20", (0.20, 0.0, 0.0)),
                ("forward 0.25", (0.25, 0.0, 0.0)), ("backward 0.08", (-0.08, 0.0, 0.0)),
                ("turn left walking", (0.10, 0.0, 0.4)), ("turn right walking", (0.10, 0.0, -0.4)),
                ("turn left in place", (0.0, 0.0, 0.4)), ("turn right in place", (0.0, 0.0, -0.4)),
                ("step left", (0.0, 0.08, 0.0)), ("step right", (0.0, -0.08, 0.0))]
COMMAND_TOL = {"lin": 0.03, "yaw": 0.1}
COMMAND_LABELS = {"forward 0.05": "앞 0.05", "forward 0.10": "앞 0.10", "forward 0.20": "앞 0.20", "forward 0.25": "앞 0.25",
                  "backward 0.08": "뒤 0.08", "turn left walking": "걸으며 왼쪽 돌기", "turn right walking": "걸으며 오른쪽 돌기",
                  "turn left in place": "제자리 왼쪽 돌기", "turn right in place": "제자리 오른쪽 돌기",
                  "step left": "왼쪽 옆걸음", "step right": "오른쪽 옆걸음"}


def command_tracking(rec, steady_s: float = 2.0) -> tuple[np.ndarray, float]:
    """기록 → 2초 뒤부터의 평균 (앞뒤, 옆) 속도 [m/s] (그때그때 몸통 방향 기준)와 평균 회전 속도 [rad/s]."""
    t, base, yaw = rec["t"], np.asarray(rec["base"]), np.unwrap(np.asarray(rec["rpy"])[:, 2])
    k = t >= min(steady_s, t[-1] / 2)
    v = np.diff(base[:, :2], axis=0) / np.diff(t)[:, None]
    c, s_ = np.cos(yaw[1:]), np.sin(yaw[1:])
    v_head = np.stack([c * v[:, 0] + s_ * v[:, 1], -s_ * v[:, 0] + c * v[:, 1]], axis=1)[k[1:]]
    duration = t[k][-1] - t[k][0]
    yaw_rate = (yaw[k][-1] - yaw[k][0]) / max(duration, 1e-6)
    return v_head.mean(axis=0), float(yaw_rate)


def command_trials(config, policy, seed=0) -> dict:
    """COMMAND_TEST의 명령마다 평지 1번 + 섞인 코스 1.0 두 번. {이름: {ok, trials, survived, err_lin, err_yaw}}"""
    out = {}
    for name, cmd in COMMAND_TEST:
        rows = []
        for terrain in (None, make_terrain("mixed", seed=seed * 1000 + 1, difficulty=1.0),
                        make_terrain("mixed", seed=seed * 1000 + 2, difficulty=1.0)):
            runner = make_runner(config, terrain=terrain)
            runner.command = np.array(cmd)
            rec = run_episode(runner, policy)
            v, w = command_tracking(rec)
            err_lin, err_yaw = float(np.linalg.norm(v - np.array(cmd[:2]))), abs(w - cmd[2])
            ok = (not rec["fell"]) and err_lin <= COMMAND_TOL["lin"] and err_yaw <= COMMAND_TOL["yaw"]
            rows.append((ok, not rec["fell"], err_lin, err_yaw))
        ok, alive, el, ey = zip(*rows)
        out[name] = {"ok": int(sum(ok)), "trials": len(rows), "survived": int(sum(alive)),
                     "err_lin": float(np.mean(el)), "err_yaw": float(np.mean(ey))}
    return out


MODEL_ERROR_LABELS = {"heavy_weak_slow": "무겁고·약하고·40 ms", "light_strong": "가볍고·세고·20 ms",
                      "miscalibrated": "조립 오차(관절·IMU 2°)·20 ms"}


def model_error_trials(config, policy, cond: dict, episodes: int, seed=0) -> int:
    """cond대로 몸통 질량·관성, 무게중심, 모터 세기, 제어 지연을 바꾼 로봇으로 섞인 코스 1.5를 episodes번. 10초 버틴 횟수."""
    runner = make_runner(config, terrain=make_terrain("flat"))
    m, b = runner.model, runner.info["base_body"]
    m.body_mass[b] *= cond["mass"]
    m.body_inertia[b] *= cond["mass"]
    m.body_ipos[b, 0:2] += cond["com"]
    runner.motor_scale, runner.delay = cond["motor"], cond["delay"]
    if cond.get("joint_offset"):   # 고정된 무작위 패턴 (정책끼리 같은 오차로 비교)
        runner.joint_offset = np.random.default_rng(42).uniform(-cond["joint_offset"], cond["joint_offset"], runner.spec.n)
    if cond.get("imu_tilt"):
        from biped_sim.envs.walk_mjx import tilt_matrix
        runner.imu_rot = tilt_matrix(*cond["imu_tilt"])
    rows = run_trials(runner, policy, ["mixed"], 1.5, episodes, seed + 7, log=None)
    return int(sum(ok for ok, _, _ in rows["mixed"]))


def push_trials(runner, policy, speed, interval, episodes, seed=0) -> list[bool]:
    """평지 10초 동안 interval초마다 몸통 수평 속도에 speed [m/s]를 무작위 방향으로 더함. 에피소드마다 10초 버텼나."""
    every = round(interval / runner.W.CONTROL_DT)
    results = []
    for ep in range(episodes):
        rng = np.random.default_rng(seed * 1000 + ep)

        def kick(state, k=[0]):   # run_episode가 제어 스텝마다 부름 (k = 스텝 수)
            k[0] += 1
            if k[0] % every == 0:
                angle = rng.uniform(-np.pi, np.pi)
                runner.data.qvel[0:2] += speed * np.array([np.cos(angle), np.sin(angle)])
        rec = run_episode(runner, policy, on_step=kick)
        results.append(not rec["fell"])
    return results


# ---------------------------------------------------------------------------- 점수·챔피언
def walking_score(result: dict, target_speed: float) -> dict:
    """보행 점수 = 울퉁불퉁한 지형(평지 제외) 전체 시도 중 10초 버틴 비율 [%].
    합격 조건: 평지에서 모두 버팀, 평지 10초 걸음에서 넘어지지 않음, 속도 목표 ±0.05, 절뚝임 점수 15 % 미만."""
    rough = [n for t in result["terrain"].values() for k, n in t.items() if k != "flat"]
    flat = [t["flat"] for t in result["terrain"].values() if "flat" in t]
    g = result["gait"]
    problems = []
    if flat and sum(flat) < len(flat) * result["episodes"]:
        problems.append(f"평지에서 넘어짐 ({sum(flat)} / {len(flat) * result['episodes']})")
    if g["fell"]:
        problems.append("평지 10초 걸음에서 넘어짐")
    if abs(g["speed_mps"] - target_speed) > 0.05:
        problems.append(f"속도 {g['speed_mps']:.2f} m/s (목표 {target_speed} ± 0.05 밖)")
    if g["limp_score_pct"] >= 15:
        problems.append(f"절뚝임 {g['limp_score_pct']:.0f} %")
    torque = None
    if result.get("torque"):
        torque = float(np.mean([v for grp in GROUPS for v in result["torque"]["flat"][grp]["over_datasheet_pct"]]))
    push, model = result.get("push") or {}, result.get("model_error") or {}
    return {"score": 100.0 * sum(rough) / max(result["trials_rough"], 1), "passed": not problems, "problems": problems,
            "datasheet_over_pct": torque, "trials_rough": result["trials_rough"],
            "push_ok": int(sum(push.values())), "push_trials": len(push) * result["episodes"],
            # 견고함 = 밀기 + 모델 오차 시험 (챔피언 동점 깨기용)
            "robust_ok": int(sum(push.values()) + sum(model.values())),
            "robust_trials": (len(push) + len(model)) * result["episodes"]}


def beats_champion(new: dict, old: dict | None) -> tuple[bool, str]:
    """챔피언을 바꿀지 (docs/10 §3). new·old = walking_score 결과 (old에 밀기 시험이 없으면 점수만 봄).
    ① 합격해야 함 ② 보행 점수가 높으면 바꿈 ③ 보행 점수 차이가 우연 범위(2σ) 안이고
    견고함 시험(밀기 + 모델 오차, 그것이 없는 예전 결과와는 밀기만)이 우연보다 크게 좋으면 바꿈"""
    if not new["passed"]:
        return False, "불합격"
    if old is None:
        return True, "첫 챔피언"
    if new["score"] > old["score"]:
        return True, "보행 점수가 높음"
    n = new.get("score_trials") or new.get("trials_rough") or 160
    a, b = old["score"] * n / 100, new["score"] * n / 100
    score_tie = not _pm(b - a, n, (a + b) / (2 * n))
    key = "robust" if new.get("robust_trials") and old.get("robust_trials") == new["robust_trials"] else "push"
    m = new.get(f"{key}_trials", 0)
    if score_tie and m and old.get(f"{key}_trials") == m:
        pa, pb = old[f"{key}_ok"], new[f"{key}_ok"]
        name = "밀기·모델 오차 시험" if key == "robust" else "밀기 시험"
        if pb > pa and _pm(pb - pa, m, (pa + pb) / (2 * m)):
            return True, f"보행 점수는 우연 범위 안에서 같고 {name}이 우연보다 크게 좋음 ({pa} → {pb} / {m})"
    return False, "보행 점수가 낮거나 같음"


def update_champion(robot: str, params: Path, run: str, step, score: dict,
                    champion_push: dict | None = None) -> tuple[bool, dict | None, str]:
    """기본 조건 평가라면 beats_champion 규칙으로 챔피언 갱신. (갱신했나, 이전 챔피언, 이유)
    champion_push: 이번 평가에서 기준 정책이 챔피언이었다면 그 밀기 시험 결과 (챔피언 파일에 없을 때 보충)"""
    path = CHAMPION_DIR / f"champion_{robot}.json"
    old = json.loads(path.read_text()) if path.exists() else None
    if old is not None and champion_push is not None:   # 챔피언 파일에 없는 시험 결과는 이번에 잰 값으로 채움
        old = {**old, **champion_push}
        if not beats_champion(score, old)[0]:            # 챔피언이 그대로면 채운 값을 파일에도 남김 (다음 비교용)
            path.write_text(json.dumps(old, indent=2, ensure_ascii=False))
    better, why = beats_champion(score, old)
    if better:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"params": str(params), "run": run, "step": step, **score}, indent=2, ensure_ascii=False))
    return better, old, why


# ---------------------------------------------------------------------------- 정책 하나 평가
def evaluate(policy, config, args, out: Path | None = None, label: str = "") -> dict:
    """지형 시험(험한 정도 × 마찰), 평지 걸음 수치, (오리면) 토크. out을 주면 그림·영상도 저장."""
    spec = get_spec(config)
    result = {"terrain": {}, "gait": None, "torque": None, "episodes": args.episodes,
              "trials_rough": args.episodes * len(args.difficulties) * len(args.frictions)
                              * len([k for k in args.kinds if k != "flat"])}
    for d in args.difficulties:
        for mu in args.frictions:
            runner = make_runner(config, terrain=make_terrain("flat"))   # 지형 격자를 가진 모델
            runner.model.geom_friction[:, 0] = mu                       # 접촉 마찰 = 두 geom 중 큰 값 → 모두
            rows = run_trials(runner, policy, args.kinds, d, args.episodes, args.seed, log=None)
            cond = f"{d:.1f} / {mu:.1f}"
            result["terrain"][cond] = {k: int(sum(ok for ok, _, _ in r)) for k, r in rows.items()}
            print(f"  [{label}] 험한 정도 {d:.1f}, 마찰 {mu:.1f}: "
                  + ", ".join(f"{k} {n}/{args.episodes}" for k, n in result["terrain"][cond].items()), flush=True)

    if spec.command_obs or args.commands:   # 명령 시험 (명령을 안 받는 예전 정책도 비교용으로 걸려 봄: 늘 0.15로 직진)
        result["commands"] = command_trials(config, policy, args.seed)
        n_ok = sum(c["ok"] for c in result["commands"].values())
        n_all = sum(c["trials"] for c in result["commands"].values())
        print(f"  [{label}] 명령 시험: {n_ok}/{n_all} 성공 — " + ", ".join(
            f"{COMMAND_LABELS[k]} {c['ok']}/{c['trials']}" for k, c in result["commands"].items()), flush=True)
    runner = make_runner(config)
    result["push"] = {f"{v:g}": int(sum(push_trials(runner, policy, v, PUSH_TEST["interval"], args.episodes, args.seed)))
                      for v in PUSH_TEST["speeds"]}
    print(f"  [{label}] 밀기 ({PUSH_TEST['interval']:g}초마다): "
          + ", ".join(f"{v} m/s {n}/{args.episodes}" for v, n in result["push"].items()), flush=True)
    result["model_error"] = {name: model_error_trials(config, policy, cond, args.episodes, args.seed)
                             for name, cond in MODEL_ERROR_TEST.items()}
    print(f"  [{label}] 모델 오차 (섞인 코스 1.5): "
          + ", ".join(f"{MODEL_ERROR_LABELS[k]} {n}/{args.episodes}" for k, n in result["model_error"].items()), flush=True)
    if out is not None and not args.no_video:
        rec, frames = record_with_frames(runner, policy)
    else:
        rec, frames = run_episode(runner, policy), []
    g = gait_metrics(rec, runner.info)
    knees = [i for i, n in enumerate(runner.info["joint_names"]) if n.endswith("knee")]
    g["knee_mean_deg"] = float(np.degrees(np.mean(np.asarray(rec["q"])[:, knees]))) if knees else None
    result["gait"] = g
    if out is not None:
        title = f"{Path(config.get('run_dir') or 'policy').name}_{config.get('saved_step')}"   # 그림 글꼴에 한글이 없어 영문
        save_plots(rec, g, runner.info, f"{title}  (robot: {spec.name}, servo: {spec.servo})", out / "gait.png")
        if frames:
            save_filmstrip(frames, rec, out / "filmstrip.png")
            save_video(frames, rec, out / "walk.mp4", fps=round(1 / runner.W.CONTROL_DT))

    if spec.name == "open_duck_mini":   # 토크·서보 한계 (feasibility.py: 관절·발 이름이 오리 기준)
        import feasibility as F
        meas = {"flat": F.measure(make_runner(config), policy)}
        for kind in ("hills", "obstacles"):
            meas[f"{kind} 1"] = F.measure(make_runner(config, terrain=make_terrain(kind, seed=0, difficulty=1.0)), policy)
        if out is not None:
            F.plot(meas, runner.info["joint_names"], out / "torque_speed.png")
        result["torque"] = {}
        for where, mm in meas.items():
            joints = mm["joints"]
            result["torque"][where] = {"fell": mm["fell"], **{
                grp: {key: [joints[f"{s}_{grp}"][key] for s in ("left", "right")]
                      for key in ("rms_torque", "over_datasheet_pct", "over_rated_pct", "peak_torque")}
                for grp in GROUPS}}
    return result


# ---------------------------------------------------------------------------- 보고서
def _pm(diff: float, n: int, p: float) -> bool:
    """두 정책의 성공 횟수 차이가 우연(이항분포 2σ)보다 큰가. n = 정책마다 시도 수, p = 두 정책 합친 성공률."""
    return abs(diff) > 2 * math.sqrt(max(2 * n * p * (1 - p), 1e-9))


def _cell(base, new, fmt="{}"):
    return fmt.format(new) if base is None else f"{fmt.format(base)} → **{fmt.format(new)}**"


def _rng(values, fmt="{:.0f}", unit=" %"):
    lo, hi = min(values), max(values)
    return (fmt.format(lo) if fmt.format(lo) == fmt.format(hi) else f"{fmt.format(lo)}~{fmt.format(hi)}") + unit


def verdict(new: dict, base: dict | None, args, target_speed: float) -> list[str]:
    """자동 판정 몇 줄: 평지에서 넘어지는지, 속도, 지형 합계·장애물(우연보다 큰 차이인지), 토크, 절뚝임."""
    lines = []
    n_cond = len(new["terrain"])
    flat_ok = sum(t.get("flat", args.episodes) for t in new["terrain"].values())
    if "flat" in args.kinds and flat_ok < args.episodes * n_cond:
        lines.append(f"⚠ **평지에서도 넘어짐** ({flat_ok} / {args.episodes * n_cond})")
    g = new["gait"]
    if g["fell"]:
        lines.append("⚠ **평지 10초 걸음에서 넘어짐**")
    if abs(g["speed_mps"] - target_speed) > 0.05:
        lines.append(f"⚠ 속도 {g['speed_mps']:.2f} m/s — 목표 {target_speed} ± 0.05 밖")
    if new.get("push"):   # 밀기 시험 (보행 점수와 별도)
        lines.append(f"밀기 버티기 (평지, {PUSH_TEST['interval']:g}초마다 무작위 방향, {args.episodes}번 중): " + ", ".join(
            f"{v} m/s " + (f"{base['push'][v]} → {n}" if base is not None and base.get("push") else f"{n}")
            for v, n in new["push"].items()))
    if new.get("model_error"):   # 모델 오차 시험 (보행 점수와 별도)
        lines.append(f"모델 오차 버티기 (섞인 코스 1.5, {args.episodes}번 중): " + ", ".join(
            f"{MODEL_ERROR_LABELS[k]} " + (f"{base['model_error'][k]} → {n}" if base is not None and base.get("model_error")
                                          else f"{n}") for k, n in new["model_error"].items()))
    if base is not None:
        for name, kinds in (("지형 전체(평지 제외)", [k for k in args.kinds if k != "flat"]),
                            ("장애물", [k for k in args.kinds if k == "obstacles"])):
            if not kinds:
                continue
            n = args.episodes * n_cond * len(kinds)
            a = sum(t[k] for t in base["terrain"].values() for k in kinds)
            b = sum(t[k] for t in new["terrain"].values() for k in kinds)
            real = _pm(b - a, n, (a + b) / (2 * n))
            word = "좋아짐" if b > a else "나빠짐" if b < a else "같음"
            lines.append(f"{name}: {a} → {b} / {n} — {word}"
                         + ("" if b == a else " (우연보다 큰 차이)" if real else " (우연 범위 안, ±2σ)"))
        if new["torque"] and base["torque"]:
            def leg_rms(r):
                return np.mean([v for grp in GROUPS for v in r["torque"]["flat"][grp]["rms_torque"]])

            def over(r):
                return np.mean([v for grp in GROUPS for v in r["torque"]["flat"][grp]["over_datasheet_pct"]])
            a, b = leg_rms(base), leg_rms(new)
            lines.append(f"평지 다리 토크 평균 RMS {a:.2f} → {b:.2f} N·m ({100 * (b - a) / a:+.0f} %), "
                         f"데이터시트 한계 밖 평균 {over(base):.1f} → {over(new):.1f} %")
        a, b = base["gait"]["limp_score_pct"], g["limp_score_pct"]
        lines.append(f"절뚝임 점수 {a:.1f} → {b:.1f} %" + (" ⚠ 15 % 넘음" if b > 15 else ""))
    elif not lines:
        lines.append("기준 정책 없이 평가함 (비교 없음)")
    return lines


def report_md(run: str, step, params: Path, baseline: Path | None, new: dict, base: dict | None,
              config: dict, base_config: dict | None, args, spec, champion_lines=()) -> str:
    target = config.get("target_speed", spec.target_speed)
    md = [f"# 평가: {run} (학습 스텝 {step:,})" if isinstance(step, int) else f"# 평가: {run}", ""]
    md.append(f"- 정책: `{params}`")
    md.append(f"- 기준: `{baseline}` (같은 지형·마찰·서보 모델에서)" if baseline else "- 기준: 없음")
    md.append(f"- 시험: 서보 `{spec.servo}`, 서 있는 자세 무릎 {spec.robot.home_pose.get('left_knee', '?')} rad, "
              f"지형 {', '.join(args.kinds)} × 험한 정도 {list(args.difficulties)} × 마찰 {list(args.frictions)}, "
              f"각 {args.episodes}번 (시드 {args.seed})")
    if base_config is not None:   # 기준과 달라진 학습 설정
        changes = []
        bw, nw = base_config.get("reward_weights", {}), config.get("reward_weights", {})
        changes += [f"보상 `{k}` {bw.get(k, 0)} → {nw[k]}" for k in nw if nw[k] != bw.get(k, 0)]
        for key in ("servo", "home_knee", "friction_range", "terrain", "target_speed"):
            if config.get(key) != base_config.get(key):
                changes.append(f"`{key}` {base_config.get(key)} → {config.get(key)}")
        md.append("- 기준과 달라진 학습 설정: " + ("; ".join(changes) if changes else "없음"))
    md += ["", "## 판정", ""] + [f"- {line}" for line in (*champion_lines, *verdict(new, base, args, target))]

    md += ["", f"## 1. 지형 ({args.episodes}번 중 10초 버팀" + (", 기준 → 새 정책)" if base else ")"), ""]
    md.append("| 험한 정도 / 마찰 | " + " | ".join(args.kinds) + " | 평지 제외 합계 |")
    md.append("|---" * (len(args.kinds) + 2) + "|")
    for cond, row in new["terrain"].items():
        brow = base["terrain"][cond] if base else None
        cells = [_cell(brow[k] if brow else None, row[k]) for k in args.kinds]
        rough = [k for k in args.kinds if k != "flat"]
        total = _cell(sum(brow[k] for k in rough) if brow else None, sum(row[k] for k in rough))
        md.append(f"| {cond} | " + " | ".join(cells) + f" | {total} |")

    if new["torque"]:
        md += ["", "## 2. 토크 (실측 서보 모델 기준, 좌우 다리 범위" + (", 기준 → 새 정책)" if base else ")"), ""]
        md.append("| 관절 | RMS 평지 [N·m] | RMS 장애물 1.0 | 데이터시트 한계 밖 (평지) | 정격 0.64 초과 (평지) |")
        md.append("|---|---|---|---|---|")
        for grp in GROUPS:
            def col(r, where, key, fmt, unit):
                return _rng(r["torque"][where][grp][key], fmt, unit)
            cells = []
            for where, key, fmt, unit in (("flat", "rms_torque", "{:.2f}", ""), ("obstacles 1", "rms_torque", "{:.2f}", ""),
                                          ("flat", "over_datasheet_pct", "{:.0f}", " %"),
                                          ("flat", "over_rated_pct", "{:.0f}", " %")):
                cells.append(_cell(col(base, where, key, fmt, unit) if base and base["torque"] else None,
                                   col(new, where, key, fmt, unit)))
            md.append(f"| {grp} | " + " | ".join(cells) + " |")

    g, bg = new["gait"], base["gait"] if base else None

    def gv(m, f):
        return None if m is None else f(m)
    rows = [("10초 버팀", lambda m: "예" if not m["fell"] else f"넘어짐 {m['survived_s']:.1f} s", "{}"),
            ("속도 [m/s]", lambda m: m["speed_mps"], "{:.3f}"),
            ("발 들기 [cm] (좌우 평균)", lambda m: 50 * (m["left"]["clearance_m"] + m["right"]["clearance_m"]), "{:.1f}"),
            ("걸음 길이 왼 / 오른 [cm]", lambda m: f"{100 * m['left']['step_length_m']:.1f} / "
                                              f"{100 * m['right']['step_length_m']:.1f}", "{}"),
            ("절뚝임 점수 [%]", lambda m: m["limp_score_pct"], "{:.1f}"),
            ("좌우 다리 동작 차이 [°]", lambda m: m["leg_asymmetry_deg"], "{:.1f}"),
            ("몸통 roll 범위 [°]", lambda m: m["body_roll_range_deg"], "{:.0f}"),
            ("방향 틀어짐 [°] / 옆으로 [m]", lambda m: f"{m['yaw_deg']:+.0f} / {m['drift_y_m']:+.2f}", "{}"),
            ("평균 무릎 각도 [°]", lambda m: m["knee_mean_deg"], "{:.0f}")]
    md += ["", "## 3. 평지 걸음 (10초)", "", "| | " + ("기준 | 새 정책 |" if base else "새 정책 |"),
           "|---|---|" + ("---|" if base else "")]
    for name, f, fmt in rows:
        md.append(f"| {name} | " + (f"{fmt.format(gv(bg, f))} | " if base else "") + f"{fmt.format(gv(g, f))} |")
    if new.get("commands"):
        md += ["", "## 4. 명령 따라 걷기 (명령마다 3번: 평지 1 + 섞인 코스 1.0 2, 성공 / 평균 오차" + (", 기준 → 새 정책)" if base else ")"), "",
               "| 명령 (vx, vy, wz) | 성공 | 넘어지지 않음 | 속도 오차 [m/s] | 회전 오차 [rad/s] |", "|---|---|---|---|---|"]
        for name, cmd in COMMAND_TEST:
            c, bc = new["commands"][name], (base or {}).get("commands", {}).get(name)
            md.append(f"| {COMMAND_LABELS[name]} {cmd} | " + " | ".join(
                _cell(bc[k] if bc else None, c[k], fmt) for k, fmt in (("ok", "{}"), ("survived", "{}"),
                                                                     ("err_lin", "{:.3f}"), ("err_yaw", "{:.2f}"))) + " |")
    md += ["", "![연속 사진](filmstrip.png)", "", "![걸음 그래프](gait.png)"]
    if new["torque"]:
        md += ["", "![토크-속도](torque_speed.png)"]
    md += ["", f"칸마다 {args.episodes}번이라 ±1~2번 차이는 우연일 수 있습니다. 판정의 '우연보다 큰 차이'는 이항분포 2σ 기준입니다."]
    return "\n".join(md) + "\n"


# ---------------------------------------------------------------------------- 실행
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--params", type=Path, required=True, help="평가할 정책 (보통 <실행 폴더>/params.pkl)")
    parser.add_argument("--baseline", type=Path, default=None, help="비교할 정책 (기본: 학습 설정의 init_from)")
    parser.add_argument("--no-baseline", action="store_true", help="비교 없이 새 정책만")
    parser.add_argument("--episodes", type=int, default=10, help="조건·지형마다 에피소드 수")
    parser.add_argument("--difficulties", type=float, nargs="+", default=[1.5, 2.0], help="지형 험한 정도")
    parser.add_argument("--frictions", type=float, nargs="+", default=[1.0, 0.5], help="바닥 마찰 계수")
    parser.add_argument("--kinds", nargs="+", default=list(KINDS), choices=KINDS)
    parser.add_argument("--seed", type=int, default=0, help="지형 시드 (같으면 같은 지형)")
    parser.add_argument("--quick", action="store_true", help="빠르게: 3번씩, 험한 정도 1.5, 마찰 1.0만")
    parser.add_argument("--commands", action="store_true", help="명령 시험도 (명령 따라 걷기 정책이면 저절로)")
    parser.add_argument("--no-video", action="store_true", help="연속 사진·영상 없이 (화면 렌더링이 없는 환경·테스트용)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.quick:
        args.episodes, args.difficulties, args.frictions = 3, [1.5], [1.0]

    params = args.params.resolve()
    policy, config = load_policy(params)
    spec = get_spec(config)
    run_dir = params.parent.parent if params.parent.name == "checkpoints" else params.parent
    step = config.get("saved_step")
    out = (args.out or run_dir / ("eval" if params.name == "params.pkl" else f"eval_{step}")).resolve()
    out.mkdir(parents=True, exist_ok=True)

    baseline = None if args.no_baseline else (args.baseline or (Path(config["init_from"]) if config.get("init_from") else None))
    if baseline is not None and not baseline.exists():
        print(f"[알림] 기준 정책이 없어 비교 없이 평가합니다: {baseline}")
        baseline = None
    print(f"평가: {params}\n기준: {baseline or '없음'}\n결과: {out}", flush=True)

    if spec.command_obs:   # 명령 따라 걷기 정책이면 기준 정책도 같은 명령 시험으로
        args.commands = True
    new = evaluate(policy, config, args, out, label="새 정책")
    base, base_config = None, None
    if baseline is not None:
        base_policy, base_config = load_policy(baseline.resolve())
        test_config = {**base_config, "servo": config.get("servo", "ideal")}   # 같은 서보 모델에서
        base = evaluate(base_policy, test_config, args, None, label="기준")

    target = config.get("target_speed", spec.target_speed)
    score = walking_score(new, target)
    lines = [f"**보행 점수 {score['score']:.1f}** (울퉁불퉁한 지형 {new['trials_rough']}번 중 10초 버팀 비율) — "
             + ("합격" if score["passed"] else "불합격: " + ", ".join(score["problems"]))]
    if spec.command_obs:   # 명령 따라 걷기: 챔피언은 명령 점수로 겨룸. 단 보행 점수가 실물 관측 챔피언보다 우연 이상 낮으면 불합격
        n_ok = sum(c["ok"] for c in new["commands"].values())
        n_all = sum(c["trials"] for c in new["commands"].values())
        walk = score["score"]
        dep_file = CHAMPION_DIR / f"champion_{spec.name}_deployable.json"
        if dep_file.exists():
            dep = json.loads(dep_file.read_text())
            n = score["trials_rough"]
            a, b = dep["score"] * n / 100, walk * n / 100
            if b < a and _pm(b - a, n, (a + b) / (2 * n)):
                score["passed"] = False
                score["problems"].append(f"보행 점수가 실물 관측 챔피언({dep['score']:.1f})보다 우연 이상 낮음")
        score = {**score, "score": 100.0 * n_ok / n_all, "score_trials": n_all, "walk_score": walk}
        lines.insert(0, f"**명령 점수 {score['score']:.1f}** (명령 {len(COMMAND_TEST)}가지 × 3번 = {n_all}번 중, 10초 버티며 "
                        f"속도 ±{COMMAND_TOL['lin']} m/s·회전 ±{COMMAND_TOL['yaw']} rad/s 안) — "
                        + ("합격" if score["passed"] else "불합격: " + ", ".join(score["problems"])))
    if base is not None:
        bs = walking_score(base, target)
        lines.append(f"기준 정책 보행 점수 {bs['score']:.1f}" + ("" if bs["passed"] else " (불합격: " + ", ".join(bs["problems"]) + ")"))
    is_default = all(getattr(args, k) == v for k, v in DEFAULTS.items())
    if is_default:
        # 실물 관측 정책(deployable_obs)은 따로 겨룸: 시뮬레이션에만 있는 센서 값을 쓰는 정책과는 비교 대상이 다름 (docs/10 §7)
        category = spec.name + ("_deployable" if spec.deployable_obs else "") + ("_cmd" if spec.command_obs else "")
        champion_file = CHAMPION_DIR / f"champion_{category}.json"
        champ = json.loads(champion_file.read_text()) if champion_file.exists() else None
        champion_push = None   # 챔피언 파일에 없는 견고함 시험(밀기·모델 오차) 결과를 채움
        expected = (len(PUSH_TEST["speeds"]) + len(MODEL_ERROR_TEST)) * args.episodes
        if champ is not None and champ.get("robust_trials") != expected:   # 없거나 시험 구성이 바뀌었으면 다시 잼
            if base is not None and baseline is not None and Path(champ["params"]).resolve() == baseline.resolve():
                bs = walking_score(base, target)   # 기준 정책이 바로 챔피언 → 이번에 잰 값
            else:                                  # 아니면 챔피언을 같은 시험으로 직접 잼 (약 1분)
                print(f"  챔피언({champ['run']})의 견고함 시험 결과가 없어 재는 중...", flush=True)
                cp, cc = load_policy(Path(champ["params"]))
                cc = {**cc, "servo": config.get("servo", "ideal")}
                measured = {"push": {f"{v:g}": int(sum(push_trials(make_runner(cc), cp, v, PUSH_TEST["interval"],
                                                                    args.episodes, args.seed)))
                                     for v in PUSH_TEST["speeds"]},
                            "model_error": {k: model_error_trials(cc, cp, c, args.episodes, args.seed)
                                            for k, c in MODEL_ERROR_TEST.items()}}
                push, model = measured["push"], measured["model_error"]
                bs = {"push_ok": sum(push.values()), "push_trials": len(push) * args.episodes,
                      "robust_ok": sum(push.values()) + sum(model.values()),
                      "robust_trials": (len(push) + len(model)) * args.episodes}
            champion_push = {k: bs[k] for k in ("push_ok", "push_trials", "robust_ok", "robust_trials")}
        updated, old, why = update_champion(category, params, run_dir.name, step, score, champion_push)
        if spec.command_obs:   # 명령 챔피언의 '점수'는 명령 점수
            why = why.replace("보행 점수", "명령 점수")
        kind = "명령 챔피언" if spec.command_obs else "실물 관측 챔피언" if spec.deployable_obs else "챔피언"
        lines.append((f"🏆 **{kind} 갱신** — {why}" + (f" (이전: {old['run']} {old['score']:.1f})" if old else ""))
                     if updated else f"{kind} 유지: {old['run']} {old['score']:.1f} ({why})" if old else f"{kind} 없음 ({why})")
        if base is not None and spec.deployable_obs and not get_spec(base_config).deployable_obs:
            lines.append("⚠ 기준 정책은 시뮬레이션에만 있는 센서 값(몸통 속도·높이)을 쓰는 정책이라 비교는 참고용")
    else:
        lines.append("기본 조건이 아니라 챔피언과 비교하지 않음")
    md = report_md(run_dir.name, step, params, baseline, new, base, config, base_config, args, spec, lines)
    (out / "report.md").write_text(md)
    (out / "report.json").write_text(json.dumps({"params": str(params), "baseline": str(baseline) if baseline else None,
                                                 "conditions": {"episodes": args.episodes, "difficulties": args.difficulties,
                                                                "frictions": args.frictions, "kinds": args.kinds,
                                                                "seed": args.seed, "servo": spec.servo},
                                                 "score": score, "new": new, "baseline_result": base},
                                                indent=2, ensure_ascii=False))
    try:   # TensorBoard TEXT 탭에서도 보이게 (학습 실행 폴더일 때만. 그림은 상대 경로라 TensorBoard에서는 안 보임)
        if args.out is not None or not any(run_dir.glob("events.out.tfevents*")):
            raise FileNotFoundError("학습 실행 폴더의 기본 결과 위치가 아님")   # --out으로 따로 뽑은 평가는 기록하지 않음
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(str(run_dir), filename_suffix=".eval")
        writer.add_text("evaluation", md, global_step=step if isinstance(step, int) else 0)
        writer.close()
    except Exception as e:  # noqa: BLE001  (기록 실패해도 보고서 파일은 있음)
        print(f"[알림] TensorBoard에 보고서를 쓰지 못했습니다: {type(e).__name__}")
    print("\n" + md)
    print(f"저장: {out}/report.md (+ report.json, 그림)")


if __name__ == "__main__":
    main()
