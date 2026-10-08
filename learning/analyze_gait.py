#!/usr/bin/env python3
"""걸음 분석 — 정책으로 10초 걸어 보고 영상, 연속 사진, 그래프, 좌우 대칭 수치(절뚝임 점수)를 만든다

실행 (GPU 학습 환경에서: source scripts/activate_gpu.sh)
    python learning/analyze_gait.py                                   # 가장 최근 학습 정책
    python learning/analyze_gait.py --params output/learning/<YYMMDD_HHMMSS>_walk/params.pkl
    python learning/analyze_gait.py --params output/learning/<YYMMDD_HHMMSS>_walk/checkpoints/step_00010649600.pkl
    python learning/analyze_gait.py --run output/learning/<YYMMDD_HHMMSS>_walk   # 저장된 정책을 모두 비교 → 가장 좋은 걸음

결과 (output/gait/<실행 폴더 이름>_<학습 스텝>/)
    walk.mp4        옆(왼쪽)·앞(오른쪽)에서 본 영상, 실시간 속도
    walk_slow.mp4   같은 영상 4배 느리게 (절뚝임을 눈으로 확인하기 좋음)
    filmstrip.png   걸음 한 주기(왼발 착지 → 다음 왼발 착지)의 옆·앞 모습 8장. 영상을 한 장으로 보는 용도
    gait.png        관절 각도 좌우 비교, 발 높이·접촉, 몸통 기울기, 관절 궤적 고리, 다리 막대 그림, 좌우 비교표
    gait.json       위 수치 전부 (실행끼리 비교용)

절뚝임 점수 = 좌우 차이(대칭 지수 SI)의 평균 [%]. SI = |왼쪽 − 오른쪽| / 평균 × 100
    걸음 길이, 디딤(땅에 닿아 있는) 시간, 흔듦(공중) 시간, 발 들어 올린 높이, 고관절·무릎·발목 움직임 범위로 계산한다.
    대략 5 % 미만이면 대칭, 5~15 %면 약간, 15 % 이상이면 눈에 띄게 절뚝인다 (사람 보행 분석의 경험적 기준, 참고용).
"""
import argparse
import functools
import json
import shutil
import subprocess
from pathlib import Path

import mujoco
import numpy as np
# walk_tools를 walk_mjx(→ JAX)보다 먼저: JAX가 CPU를 쓰도록 설정함 (학습 중인 GPU와 겹치지 않게)
from walk_tools import (LATEST, STEADY_FROM, _sagittal_joints, gait_metrics, load_policy, make_runner, run_episode,
                        si, summary_lines)

from biped_sim import paths

MAX_TAPS = 2               # --run에서 '좋은 걸음'으로 인정하는 발 튕김 횟수 (정상 상태 8초 동안)
FRAME_W, FRAME_H = 480, 360
SIDE, FRONT = dict(azimuth=90, elevation=-5), dict(azimuth=180, elevation=-5)   # 거리는 로봇별 (WalkSpec.view_distance)
COLORS = {"left": "tab:blue", "right": "tab:red"}   # 로봇 다리 색과 같게 (왼쪽 파랑, 오른쪽 빨강)
TILE_W = 300               # 연속 사진 한 칸 너비 (화면 가운데의 로봇 부분만 잘라 냄)


def record_with_frames(runner, policy):
    """한 에피소드를 기록하면서 매 제어 스텝 옆·앞 화면을 렌더링 (영상·연속 사진용)."""
    renderer = mujoco.Renderer(runner.model, FRAME_H, FRAME_W)
    cams = []
    for view in (SIDE, FRONT):
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = runner.info["base_body"]
        cam.distance = runner.spec.view_distance   # 로봇 크기에 맞춰 (simple_biped 1.5 m, 오리 1.0 m)
        cam.azimuth, cam.elevation = view["azimuth"], view["elevation"]
        cams.append(cam)
    frames = []

    def grab(_state):
        row = []
        for cam in cams:
            renderer.update_scene(runner.data, camera=cam)
            row.append(renderer.render())
        frames.append(np.concatenate(row, axis=1))

    try:
        rec = run_episode(runner, policy, on_step=grab)
    finally:
        renderer.close()
    return rec, frames


# ---------------------------------------------------------------------------- 그림
def stride_window(rec):
    """정상 상태의 한 주기(걸음 박자, simple_biped 0.8 s): t ≥ 6 s 이후 첫 왼발 착지부터 (착지가 없으면 6.0 s부터)."""
    c, t = rec["contact"][:, 0], rec["t"]
    start = min(6.0, t[-1] / 2)
    td = np.flatnonzero(c[1:] & ~c[:-1]) + 1
    td = td[t[td] >= start]
    a = int(td[0]) if len(td) else int(np.searchsorted(t, start))
    return a, min(a + round(rec["gait_period"] / rec["dt"]), len(t) - 1)


@functools.lru_cache(maxsize=1)
def _font():
    from PIL import ImageFont
    try:
        return ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        return ImageFont.load_default()


def label(img, text):
    from PIL import Image, ImageDraw
    im = Image.fromarray(img)
    draw = ImageDraw.Draw(im)
    font = _font()
    draw.rectangle((0, 0, im.width, 26), fill=(0, 0, 0))
    draw.text((6, 3), text, fill=(255, 255, 255), font=font)
    return np.asarray(im)


def save_filmstrip(frames, rec, path, n=8):
    a, b = stride_window(rec)
    idx = np.linspace(a, b, n).round().astype(int)
    tiles = []
    x0 = (FRAME_W - TILE_W) // 2
    for i in idx:
        cl, cr = rec["contact"][i]
        side, front = frames[i][:, x0:x0 + TILE_W], frames[i][:, FRAME_W + x0:FRAME_W + x0 + TILE_W]
        text = f"{rec['t'][i]:.2f}s L{'#' if cl else '.'} R{'#' if cr else '.'}"
        tiles.append(np.concatenate([label(side, text), label(front, "front")], axis=0))
        tiles.append(np.full((tiles[-1].shape[0], 4, 3), 255, np.uint8))   # 칸 사이 흰 줄
    from PIL import Image
    Image.fromarray(np.concatenate(tiles, axis=1)).save(path)


def save_video(frames, rec, path, fps):
    if shutil.which("ffmpeg") is None:
        print(f"  (ffmpeg가 없어 {path.name}을 건너뜀 → sudo apt install ffmpeg)")
        return
    h, w = frames[0].shape[:2]
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i, f in enumerate(frames):
        cl, cr = rec["contact"][i]
        proc.stdin.write(label(f, f"t={rec['t'][i]:5.2f}s   left foot {'DOWN' if cl else 'up  '}   "
                                  f"right foot {'DOWN' if cr else 'up'}").tobytes())
    proc.stdin.close()
    proc.wait()


def _side_joints(names) -> list[str]:
    return [n.split("_", 1)[1] for n in names if n.startswith("left_")]


def save_plots(rec, metrics, info, title, path):
    names = info["joint_names"]
    hip = names.index("left_hip_pitch")    # hip_pitch의 좌우 부호 (오리는 −1)
    mirror_hip = float(info["mirror_sign"][list(info["mirror_left"]).index(hip)])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = rec["t"]
    win = (t >= max(STEADY_FROM, t[-1] - 4.0))          # 마지막 4초
    a, b = stride_window(rec)
    colors = COLORS
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    fig.suptitle(title, fontsize=13)

    sagittal = _sagittal_joints(names)   # 옆에서 본 굽힘 관절: 고관절 pitch, 무릎, 발목 (로봇마다 이름이 다름)
    for ax, joint in zip(axes[0], sagittal):
        for side in ("left", "right"):
            j = names.index(f"{side}_{joint}")
            ax.plot(t[win], np.degrees(rec["q"][win, j]), color=colors[side], label=side)
        ax.set_title(f"{joint} angle [deg]")
        ax.set_xlabel("time [s]")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)

    ax = axes[1, 0]
    for i, side in enumerate(("left", "right")):
        ax.plot(t[win], 100 * rec["foot"][win, i, 2], color=colors[side], label=f"{side} ankle height")
        down = rec["contact"][win, i]
        ax.fill_between(t[win], -1.5 - 1.5 * i, -0.3 - 1.5 * i, where=down, color=colors[side], alpha=0.6, step="mid")
    ax.set_title("foot height [cm] (bars below: foot on ground)")
    ax.set_xlabel("time [s]")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    rpy = np.degrees(rec["rpy"])
    ax.plot(t[win], rpy[win, 1], label="pitch (+ = lean forward)")
    ax.plot(t[win], rpy[win, 0], label="roll (+ = lean left)")
    ax.set_title("body tilt [deg]")
    ax.set_xlabel("time [s]")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    ax = axes[1, 2]   # 관절 궤적 고리: 고관절-무릎 각도를 한 평면에. 좌우 고리가 겹치면 대칭
    for side in ("left", "right"):
        jh, jk = names.index(f"{side}_hip_pitch"), names.index(f"{side}_knee")
        sign = mirror_hip if side == "right" else 1.0   # 좌우 부호가 반대인 관절은 뒤집어 같은 그림에
        ax.plot(sign * np.degrees(rec["q"][win, jh]), np.degrees(rec["q"][win, jk]), color=colors[side], label=side, lw=1)
    ax.set_title("hip-knee loop (overlap = symmetric" + (", right hip mirrored)" if mirror_hip < 0 else ")"))
    ax.set_xlabel("hip pitch [deg]")
    ax.set_ylabel("knee [deg]")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    for ax, (i, side) in zip(axes[2, :2], enumerate(("left", "right"))):   # 다리 막대 그림 (옆에서, 몸통 기준)
        steps = np.linspace(a, b, 9).round().astype(int)
        for n, s in enumerate(steps):
            hip, knee, ankle = rec["legs"][s, i]
            toe, heel = rec["toe"][s, i], rec["heel"][s, i]
            x0 = rec["base"][s, 0]
            xs = np.array([hip[0], knee[0], ankle[0], toe[0], heel[0], ankle[0]]) - x0
            zs = np.array([hip[2], knee[2], ankle[2], toe[2], heel[2], ankle[2]])
            ax.plot(xs, zs, "-o", ms=3, color=plt.cm.viridis(n / (len(steps) - 1)), lw=1.5)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_aspect("equal")
        ax.set_title(f"{side} leg, one stride (side view; dark→light = time)")
        ax.set_xlabel("x relative to body [m]")
        ax.set_ylabel("height [m]")
        ax.grid(alpha=0.3)

    ax = axes[2, 2]
    ax.axis("off")
    L, R, S = metrics["left"], metrics["right"], metrics["symmetry_index_pct"]
    rows = [("step length [cm]", 100 * L["step_length_m"], 100 * R["step_length_m"], S["step_length_m"]),
            ("stance time [s]", L["stance_s"], R["stance_s"], S["stance_s"]),
            ("swing time [s]", L["swing_s"], R["swing_s"], S["swing_s"]),
            ("foot clearance [cm]", 100 * L["clearance_m"], 100 * R["clearance_m"], S["clearance_m"]),
            *[(f"{j} range [deg]", L[f"{j}_range_deg"], R[f"{j}_range_deg"], S[f"{j}_range_deg"])
              for j in _side_joints(names)],
            ("steps (taps)", f"{L['touchdowns']} ({L['taps']})", f"{R['touchdowns']} ({R['taps']})",
             si(L["touchdowns"], R["touchdowns"]))]
    fmt = lambda v: v if isinstance(v, str) else f"{v:.2f}"  # noqa: E731
    table = ax.table(cellText=[[r[0], fmt(r[1]), fmt(r[2]), f"{r[3]:.0f}"] for r in rows],
                     colLabels=["", "left (blue)", "right (red)", "diff %"], loc="center", cellLoc="center",
                     colWidths=[0.42, 0.2, 0.2, 0.18])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.7)
    phase = metrics["left_right_phase_pct"]
    ax.set_title(f"LIMP SCORE {metrics['limp_score_pct']:.1f} %\n(mean left-right diff; <5 symmetric, >15 visible limp)"
                 + (f"\nL→R touchdown at {phase:.0f}% of stride (50 = even)" if phase is not None else ""),
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def print_report(m):
    L, R, S = m["left"], m["right"], m["symmetry_index_pct"]
    print("\n".join(summary_lines(m)[:2]))
    print(f"  몸통: 앞뒤 기울기 평균 {m['body_pitch_mean_deg']:+.1f}° (흔들림 {m['body_pitch_range_deg']:.1f}°), "
          f"좌우 기울기 평균 {m['body_roll_mean_deg']:+.1f}° (흔들림 {m['body_roll_range_deg']:.1f}°), "
          f"높이 흔들림 {100 * m['body_height_range_m']:.1f} cm")
    print(f"  {'':22s} {'왼쪽':>8s} {'오른쪽':>8s} {'좌우 차이':>9s}")
    joints = [k[:-len("_range_deg")] for k in L if k.endswith("_range_deg")]
    for label_, key, scale in (("걸음 길이 [cm]", "step_length_m", 100), ("디딤 시간 [s]", "stance_s", 1),
                               ("흔듦(공중) 시간 [s]", "swing_s", 1), ("발 높이 [cm]", "clearance_m", 100),
                               *[(f"{j} 범위 [°]", f"{j}_range_deg", 1) for j in joints]):
        print(f"  {label_:22s} {scale * L[key]:8.2f} {scale * R[key]:8.2f} {S[key]:8.0f} %")
    print(f"  {'걸음 수 (튕김 제외)':22s} {L['touchdowns']:8d} {R['touchdowns']:8d}"
          + (f"   ⚠ 발 튕김 왼발 {L['taps']}회, 오른발 {R['taps']}회 (착지 후 살짝 떴다 다시 닿음)"
             if L["taps"] + R["taps"] else ""))
    print(f"  {'평균 각도 [°] ' + '/'.join(joints):22s} " + "/".join(f"{L[f'{j}_mean_deg']:+.0f}" for j in joints)
          + "   " + "/".join(f"{R[f'{j}_mean_deg']:+.0f}" for j in joints))
    print(f"  좌우 다리 동작 차이 (반 박자 어긋나게 비교한 관절 각도 RMS) {m['leg_asymmetry_deg']:.1f}°")
    print(f"  ▶ 절뚝임 점수 {m['limp_score_pct']:.1f} % (좌우 차이 평균. 5 % 미만 대칭, 15 % 이상 눈에 띄게 절뚝임)")


def compare_checkpoints(run_dir: Path) -> Path | None:
    """학습 중 평가마다 저장된 정책(checkpoints/)을 모두 10초씩 걸려 보고 표로 비교. 가장 좋은 것의 경로를 돌려줌.
    '좋은 걸음' = 10초 버팀, 속도가 목표 ±0.05 m/s, 두 발 공중 5 % 미만, 발 튕김 2회 이하인 것 중
    절뚝임 점수가 가장 낮은 것. (튕김 조건이 없으면 '살짝 튕기며 걷는' 꼼수 정책이 뽑힐 수 있음 — docs/08 §6)"""
    ckpts = sorted((run_dir / "checkpoints").glob("step_*.pkl"))
    if not ckpts:
        raise SystemExit(f"체크포인트가 없습니다: {run_dir / 'checkpoints'} (이 기능 이전의 학습이면 --params로 하나씩)")
    rows, runner = [], None
    print(f"{run_dir.name}: 체크포인트 {len(ckpts)}개 비교 (각 10초, 화면 없이)")
    print(f"  {'학습 스텝':>12s} {'버틴 s':>6s} {'속도':>6s} {'방향°':>5s} {'공중%':>5s} {'한발%':>5s} {'튕김':>4s} "
          f"{'다리차이°':>8s} {'절뚝임%':>7s}")
    for ck in ckpts:
        policy, config = load_policy(ck)
        if runner is None:   # 같은 실행의 체크포인트는 설정이 같으므로 재생기는 하나만 만들어 재사용
            runner = make_runner(config)
        m = gait_metrics(run_episode(runner, policy), runner.info)
        taps = m["left"]["taps"] + m["right"]["taps"]
        ok = (not m["fell"] and abs(m["speed_mps"] - config["target_speed"]) <= 0.05 and m["both_air_pct"] < 5
              and taps <= MAX_TAPS)
        rows.append({"params": str(ck), "step": int(ck.stem.split("_")[-1]), "ok": ok,
                     "taps": taps, "yaw_deg": m["yaw_deg"],
                     **{k: m[k] for k in ("survived_s", "speed_mps", "both_air_pct", "single_pct",
                                          "leg_asymmetry_deg", "limp_score_pct")}})
        r = rows[-1]
        print(f"  {r['step']:>12,} {r['survived_s']:6.1f} {r['speed_mps']:+6.2f} {r['yaw_deg']:+5.0f} "
              f"{r['both_air_pct']:5.0f} {r['single_pct']:5.0f} {r['taps']:4d} "
              f"{r['leg_asymmetry_deg']:8.1f} {r['limp_score_pct']:7.1f}"
              + ("" if ok else "   (조건 미달)"))
    good = [r for r in rows if r["ok"]]
    best = min(good, key=lambda r: r["limp_score_pct"]) if good else None
    out = paths.OUTPUT_DIR / "gait" / f"{run_dir.name}_checkpoints.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows, "best": best}, indent=2, ensure_ascii=False))
    if best is None:
        print(f"  조건(10초 버팀, 목표 속도 ±0.05, 두 발 공중 < 5 %, 튕김 ≤ {MAX_TAPS})을 만족하는 정책이 없습니다.")
        return None
    print(f"  ▶ 가장 좋은 걸음: 스텝 {best['step']:,} (절뚝임 {best['limp_score_pct']:.1f} %) → {best['params']}")
    return Path(best["params"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--params", type=Path, default=LATEST, help="정책 파일 (기본: 최근 학습)")
    parser.add_argument("--run", type=Path, default=None,
                        help="학습 실행 폴더: 저장된 정책을 모두 비교해 가장 좋은 걸음을 고르고, 그것을 영상까지 분석")
    parser.add_argument("--out", type=Path, default=None, help="결과 폴더 (기본: output/gait/<실행>_<스텝>)")
    parser.add_argument("--no-video", action="store_true", help="영상·사진 없이 수치와 그래프만 (빠름)")
    args = parser.parse_args()

    if args.run is not None:
        best = compare_checkpoints(args.run.resolve())
        if best is None:
            return
        args.params = best
        print()

    params_path = args.params.resolve()
    policy, config = load_policy(params_path)
    runner = make_runner(config)
    run = Path(config["run_dir"]).name if config.get("run_dir") else params_path.stem   # 미리 학습된 정책은 파일 이름
    step = config.get("saved_step")
    name = f"{run}_{step if step is not None else 'unknown'}"
    out = (args.out or paths.OUTPUT_DIR / "gait" / name).resolve()
    out.mkdir(parents=True, exist_ok=True)

    print(f"정책: {params_path} (학습 스텝 {step}) → {out}")
    rec, frames = (run_episode(runner, policy), []) if args.no_video else record_with_frames(runner, policy)
    metrics = gait_metrics(rec, runner.info)
    metrics.update({"params": str(params_path), "run": run, "step": step,
                    "reward_weights": config.get("reward_weights")})
    print_report(metrics)
    (out / "gait.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    save_plots(rec, metrics, runner.info, f"{name}  (robot: {runner.spec.name}, simulator: MuJoCo)", out / "gait.png")
    if frames:
        save_filmstrip(frames, rec, out / "filmstrip.png")
        save_video(frames, rec, out / "walk.mp4", fps=round(1 / runner.W.CONTROL_DT))
        save_video(frames, rec, out / "walk_slow.mp4", fps=round(1 / runner.W.CONTROL_DT) / 4)
    print("  저장: " + ", ".join(p.name for p in sorted(out.iterdir())))
    print(f"  이 걸음에서 이어서 보상을 바꿔 학습:  python learning/train_walk_gpu.py --init-from {params_path} "
          "--reward <이름>=<가중치> --steps 10000000")


if __name__ == "__main__":
    main()
