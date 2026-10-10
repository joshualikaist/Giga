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


# ---------------------------------------------------------------------------- 정책 하나 평가
def evaluate(policy, config, args, out: Path | None = None, label: str = "") -> dict:
    """지형 시험(험한 정도 × 마찰), 평지 걸음 수치, (오리면) 토크. out을 주면 그림·영상도 저장."""
    spec = get_spec(config)
    result = {"terrain": {}, "gait": None, "torque": None}
    for d in args.difficulties:
        for mu in args.frictions:
            runner = make_runner(config, terrain=make_terrain("flat"))   # 지형 격자를 가진 모델
            runner.model.geom_friction[:, 0] = mu                       # 접촉 마찰 = 두 geom 중 큰 값 → 모두
            rows = run_trials(runner, policy, args.kinds, d, args.episodes, args.seed, log=None)
            cond = f"{d:g} / {mu:.1f}"
            result["terrain"][cond] = {k: int(sum(ok for ok, _, _ in r)) for k, r in rows.items()}
            print(f"  [{label}] 험한 정도 {d:g}, 마찰 {mu:.1f}: "
                  + ", ".join(f"{k} {n}/{args.episodes}" for k, n in result["terrain"][cond].items()), flush=True)

    runner = make_runner(config)
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
              config: dict, base_config: dict | None, args, spec) -> str:
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
    md += ["", "## 판정", ""] + [f"- {line}" for line in verdict(new, base, args, target)]

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

    new = evaluate(policy, config, args, out, label="새 정책")
    base, base_config = None, None
    if baseline is not None:
        base_policy, base_config = load_policy(baseline.resolve())
        test_config = {**base_config, "servo": config.get("servo", "ideal")}   # 같은 서보 모델에서
        base = evaluate(base_policy, test_config, args, None, label="기준")

    md = report_md(run_dir.name, step, params, baseline, new, base, config, base_config, args, spec)
    (out / "report.md").write_text(md)
    (out / "report.json").write_text(json.dumps({"params": str(params), "baseline": str(baseline) if baseline else None,
                                                 "conditions": {"episodes": args.episodes, "difficulties": args.difficulties,
                                                                "frictions": args.frictions, "kinds": args.kinds,
                                                                "seed": args.seed, "servo": spec.servo},
                                                 "new": new, "baseline_result": base}, indent=2, ensure_ascii=False))
    try:   # TensorBoard TEXT 탭에서도 보이게 (학습 실행 폴더일 때만. 그림은 상대 경로라 TensorBoard에서는 안 보임)
        if not any(run_dir.glob("events.out.tfevents*")):
            raise FileNotFoundError("학습 실행 폴더가 아님")
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
