"""학습 대시보드 — MuJoCo 창 하나에 학습 그래프와 걷는 로봇을 함께 (play_walk.py --live / --view 가 사용)

Enter 키로 화면 전환
    [학습 현황]  양옆에 학습 그래프 4개, 가운데 위에 발 접촉 그래프, 그 아래 최신 정책으로 걷는 로봇   (--live 기본)
    [로봇 보기]  로봇을 크게, 아래에 발 접촉 그래프(보행 방식), 왼쪽 위에 상태 글자   (--view 기본)

그래프 데이터는 TensorBoard와 같은 기록 파일(output/learning/<실행>/events.out.tfevents.*)을 직접 읽는다.
그래프는 MuJoCo 자체 기능(mjvFigure, viewer.set_figures)으로 그린다. MuJoCo 글꼴은 한글을 못 그려서 화면 글자는 영어.
"""
from __future__ import annotations

import struct
import time
from collections import deque
from pathlib import Path

import mujoco
import numpy as np

KEY_ENTER, KEY_KP_ENTER = 257, 335   # GLFW 키 번호. MuJoCo 뷰어는 글자 키를 거의 다 단축키로 써서 Enter를 씀
GAIT_WINDOW_S = 4.0                   # 발 접촉 그래프에 보여 줄 최근 시간 [s]
DRAW_EVERY = 5                        # 제어 스텝 5번(0.1 s)마다 화면 글자·그래프 갱신
LOG_REFRESH_S = 2.0                   # 기록 파일 다시 읽는 주기 [s]
PALETTE = [(0.35, 0.85, 0.45), (1.0, 0.6, 0.2), (0.35, 0.65, 1.0), (1.0, 0.35, 0.35), (0.95, 0.9, 0.3),
           (0.75, 0.5, 1.0), (0.3, 0.9, 0.9), (1.0, 0.55, 0.8), (0.75, 0.75, 0.75), (0.6, 0.9, 0.3),
           (0.9, 0.75, 0.55), (0.55, 0.55, 1.0)]
DIM = (0.45, 0.45, 0.5)


class ScalarLog:
    """TensorBoard 기록 파일을 직접 읽어 {태그: [(스텝, 값), ...]} 로. 새로 쓰인 부분만 이어서 읽는다.

    파일 형식 (TFRecord): [길이 8바이트][검사값 4][내용(Event protobuf) 길이만큼][검사값 4] 이 반복된다.
    """

    def __init__(self, run_dir: Path | None):
        self.run_dir = Path(run_dir) if run_dir else None
        self.offsets: dict[Path, int] = {}
        self.scalars: dict[str, list[tuple[int, float]]] = {}

    def update(self) -> dict[str, list[tuple[int, float]]]:
        if self.run_dir is None:
            return self.scalars
        from tensorboardX.proto import event_pb2  # GPU 학습 환경(.venv-mjx)에 있음

        for path in sorted(self.run_dir.glob("events.out.tfevents.*")):
            with open(path, "rb") as f:
                f.seek(self.offsets.get(path, 0))
                data = f.read()
            pos = 0
            while pos + 12 <= len(data):
                length = struct.unpack("<Q", data[pos:pos + 8])[0]
                end = pos + 12 + length + 4
                if end > len(data):  # 학습 프로세스가 아직 쓰는 중인 기록 → 다음에 다시 읽음
                    break
                event = event_pb2.Event()
                event.ParseFromString(data[pos + 12:pos + 12 + length])
                for value in event.summary.value:
                    if value.HasField("simple_value"):
                        self.scalars.setdefault(value.tag, []).append((event.step, value.simple_value))
                pos = end
            self.offsets[path] = self.offsets.get(path, 0) + pos
        return self.scalars


def make_figure(title: str, xlabel: str = "", yformat: str = "%.1f") -> mujoco.MjvFigure:
    fig = mujoco.MjvFigure()
    mujoco.mjv_defaultFigure(fig)
    fig.title = title
    fig.xlabel = xlabel
    fig.xformat = "%.0f"
    fig.yformat = yformat
    fig.flg_extend = 0          # 축 범위는 직접 정함 (set_range)
    fig.flg_legend = 1
    fig.gridsize = (4, 3)
    fig.linewidth = 2.0
    fig.figurergba = (0.06, 0.06, 0.08, 0.85)
    fig.panergba = (0.12, 0.12, 0.15, 0.85)
    fig.legendrgba = (0.06, 0.06, 0.08, 0.6)
    return fig


def set_line(fig: mujoco.MjvFigure, i: int, name: str, xs, ys, rgb) -> None:
    """fig의 i번째 선에 점들을 넣는다. linedata[i] = [x0, y0, x1, y1, ...]"""
    n = min(len(xs), mujoco.mjMAXLINEPNT)
    fig.linename[i] = name
    fig.linergb[i] = rgb
    fig.linepnt[i] = n
    if n:
        fig.linedata[i, 0:2 * n:2] = np.asarray(xs, dtype=np.float32)[-n:]
        fig.linedata[i, 1:2 * n:2] = np.asarray(ys, dtype=np.float32)[-n:]


def set_range(fig: mujoco.MjvFigure, x_range, ys, y_min=None, y_max=None) -> None:
    ys = np.concatenate([np.ravel(y) for y in ys] + [np.zeros(0)])
    if ys.size == 0:
        ys = np.zeros(1)
    lo = float(np.min(ys)) if y_min is None else y_min
    hi = float(np.max(ys)) if y_max is None else y_max
    if hi - lo < 1e-6:
        lo, hi = lo - 1.0, hi + 1.0
    pad = 0.08 * (hi - lo)
    fig.range = (x_range, (lo - (pad if y_min is None else 0.0), hi + (pad if y_max is None else 0.0)))
    span = hi - lo   # 눈금 숫자 자릿수: 범위가 좁으면 소수점까지 (안 그러면 -1, -2, -2처럼 반올림돼 보임)
    fig.yformat = "%.0f" if span >= 6 else ("%.1f" if span >= 0.6 else "%.2f")


def gait_target(t: np.ndarray, period: float) -> tuple[np.ndarray, np.ndarray]:
    """gait_phase 보상이 원하는 발 접촉 (walk_mjx._gait_phase_match와 같은 규칙): 앞 절반 왼발, 뒤 절반 오른발."""
    phase = np.mod(t / period, 1.0)
    double = (phase < 0.1) | ((phase > 0.4) & (phase < 0.6)) | (phase > 0.9)
    return (phase < 0.5) | double, (phase >= 0.5) | double


class Dashboard:
    def __init__(self, W, config: dict, run_dir: Path | None, mode: str = "graphs"):
        """W: biped_sim.envs.walk_mjx 모듈 (제어 주기·걸음 박자 상수), mode: 'graphs' | 'robot'"""
        self.W = W
        self.mode = mode
        self.reloads = 0
        self.gait = deque(maxlen=round(GAIT_WINDOW_S / W.CONTROL_DT))   # (t, 왼발 접촉, 오른발 접촉)
        self.measured: list[tuple[float, float, float, float]] = []     # 이 창에서 잰 걸음 (스텝[백만], 공중, 한 발, 양발 %)
        self.set_policy(config, run_dir, reloaded=False)
        self.figs = {name: make_figure(title, xlabel, yfmt) for name, title, xlabel, yfmt in [
            ("reward", "Episode reward (eval)", "million steps", "%.0f"),
            ("speed", "Forward speed [m/s]", "million steps", "%.2f"),
            ("terms", "Reward terms (sum per episode)", "million steps", "%.0f"),
            ("gait_hist", "Gait over training [%]", "million steps", "%.0f"),
            ("gait", "Foot contact", "", "%.0f"),
        ]}
        gait = self.figs["gait"]
        gait.xformat = "%.1f"
        gait.flg_ticklabel = (1, 0)   # y축 숫자는 의미 없음 (위 = 왼발, 아래 = 오른발)
        gait.flg_legend = 0           # 칸이 낮아 범례가 한 줄만 보임 → 색 설명은 제목에
        gait.title = "Feet: L top, R bottom, gray=target"   # 칸 너비를 넘는 제목은 잘림
        self._last_log = 0.0
        self._k = 0
        self.recent = (0.0, 0.0)   # 최근 4초: 두 발 공중 %, 한 발 %
        self._last_toggle = 0.0

    # ------------------------------------------------------------------ 상태 갱신 (메인 스레드)
    def key_callback(self, keycode: int) -> None:
        """화면 스레드에서 호출됨 → 값 하나만 바꾸고, 실제 그리기는 메인 스레드의 draw()가 함.
        MuJoCo 뷰어는 키를 누를 때와 뗄 때 두 번 부르므로(실측), 0.4초 안의 두 번째 호출은 무시한다."""
        now = time.monotonic()
        if keycode in (KEY_ENTER, KEY_KP_ENTER) and now - self._last_toggle > 0.4:
            self._last_toggle = now
            self.mode = "robot" if self.mode == "graphs" else "graphs"

    def set_policy(self, config: dict, run_dir: Path | None, reloaded: bool = True) -> None:
        self.config = config
        self.step = config.get("saved_step")
        if reloaded:
            self.reloads += 1
        if run_dir is None or getattr(self, "log", None) is None or self.log.run_dir != Path(run_dir):
            self.log = ScalarLog(run_dir)   # 학습 실행(폴더)이 바뀌면 처음부터 다시 읽음
            self.measured = []
            self._last_log = 0.0

    def start_episode(self) -> None:
        self.gait.clear()
        self._k = 0

    def end_episode(self, result: dict) -> None:
        if self.step is not None:
            self.measured.append((self.step / 1e6, 100 * result["flight"], 100 * result["single"],
                                  100 * result["double"]))

    def on_step(self, viewer, state: dict, t: float, distance: float) -> None:
        """run_episode가 제어 스텝마다 호출."""
        contact = state["foot_z"] < self.W.FOOT_CONTACT_Z
        self.gait.append((t, bool(contact[0]), bool(contact[1])))
        self._k += 1
        if self._k % DRAW_EVERY == 0:
            self.draw(viewer, state, t, distance, contact)

    # ------------------------------------------------------------------ 그리기
    def draw(self, viewer, state, t, distance, contact) -> None:
        now = time.monotonic()
        if now - self._last_log > LOG_REFRESH_S:
            self._last_log = now
            try:
                self.log.update()
            except Exception:  # noqa: BLE001  (기록 파일을 읽다 실패해도 화면은 계속)
                pass
        self._fill_gait_figure()
        viewport = viewer.viewport
        if viewport is None or viewport.width < 100 or viewport.height < 100:
            return
        w, h = viewport.width, viewport.height
        font, grid = mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos
        step = f"{self.step:,}" if self.step is not None else "waiting for first policy (JIT compile ~3 min)"
        total = self.config.get("steps")
        progress = f" / {total:,}" if (self.step is not None and total) else ""

        if self.mode == "graphs":
            self._fill_training_figures()
            col_w, top, bottom = int(0.30 * w), int(0.06 * h), int(0.05 * h)
            fig_h = (h - top - bottom) // 2
            rects = []
            for col, names in ((0, ("reward", "speed")), (w - col_w, ("terms", "gait_hist"))):
                for row, name in enumerate(names):
                    rects.append((mujoco.MjrRect(col, h - top - (row + 1) * fig_h, col_w, fig_h), self.figs[name]))
            gait_h = int(0.31 * h)  # 가운데 위: 발 접촉 (로봇은 그 아래에 보임)
            rects.append((mujoco.MjrRect(col_w, h - top - gait_h, w - 2 * col_w, gait_h), self.figs["gait"]))
            viewer.set_figures(rects)
            run = Path(self.log.run_dir).name if self.log.run_dir else "?"
            alive = self.log.scalars.get("walk/episode_seconds")
            alive = f"   survives {alive[-1][1]:.1f} / 10 s" if alive else ""
            # 글자 칸(text1, text2)을 둘 다 쓰면 가운데 정렬 위치에서 서로 겹쳐서 한 칸에 모아 씀
            viewer.set_texts([
                (font, grid.mjGRID_TOP,
                 f"LIVE TRAINING   {run}   step {step}{progress}{alive}   policy reloads {self.reloads}", ""),
                (font, grid.mjGRID_BOTTOM,
                 f"[Enter] robot view   |   t {t:4.1f} s   {state['vx']:+.2f} m/s   |   last 4 s: both feet in air "
                 f"{self.recent[0]:.0f}%, one foot {self.recent[1]:.0f}%", ""),
            ])
        else:
            viewer.set_figures([(mujoco.MjrRect(0, 0, w, int(0.27 * h)), self.figs["gait"])])
            viewer.set_texts([
                (font, grid.mjGRID_TOPLEFT,
                 "training step\npolicy reloads\nepisode time\nforward speed\ndistance\nfeet on ground (L R)",
                 f"{step}{progress}\n{self.reloads}\n{t:5.2f} s / 10 s\n"
                 f"{state['vx']:+.2f} m/s (target {self.config['target_speed']:.2f})\n{distance:+.2f} m\n"
                 f"{'#' if contact[0] else '.'}   {'#' if contact[1] else '.'}"),
                (font, grid.mjGRID_TOPRIGHT, "[Enter] training graphs\n\nlast 4 s\nboth feet in air\none foot",
                 f"\n\n\n{self.recent[0]:.0f} %\n{self.recent[1]:.0f} %"),
            ])

    def _fill_gait_figure(self) -> None:
        """보행 방식: 위 = 왼발, 아래 = 오른발. 높으면 땅을 딛고 있음. 흐린 선 = gait_phase 보상이 원하는 박자."""
        fig = self.figs["gait"]
        if not self.gait:
            fig.linepnt[:] = 0
            return
        g = np.array(self.gait, dtype=np.float32)
        t, left, right = g[:, 0], g[:, 1], g[:, 2]
        want_l, want_r = gait_target(t, self.W.GAIT_PERIOD)
        # 선은 번호 순서대로 그려짐: 흐린 목표 선을 먼저 → 실제 선이 위에 덮여 겹치는 곳은 밝은 선이 보임
        set_line(fig, 0, "target L", t, 1.15 + 0.7 * want_l, DIM)
        set_line(fig, 1, "target R", t, 0.15 + 0.7 * want_r, DIM)
        set_line(fig, 2, "left foot", t, 1.15 + 0.7 * left, PALETTE[0])
        set_line(fig, 3, "right foot", t, 0.15 + 0.7 * right, PALETTE[1])
        fig.linepnt[4:] = 0
        self.recent = (np.mean((left == 0) & (right == 0)) * 100, np.mean(left != right) * 100)
        t_end = max(float(t[-1]), GAIT_WINDOW_S)
        fig.range = ((t_end - GAIT_WINDOW_S, t_end), (0.0, 2.1))

    def _fill_training_figures(self) -> None:
        s = self.log.scalars
        total_m = (self.config.get("steps") or 0) / 1e6
        last_m = max([v[-1][0] for v in s.values() if v] + [0]) / 1e6
        x_range = (0.0, max(total_m, last_m, 1.0))

        def series(tag):
            pts = s.get(tag, [])
            return [p[0] / 1e6 for p in pts], [p[1] for p in pts]

        fig = self.figs["reward"]
        xs, ys = series("eval/episode_reward")
        set_line(fig, 0, "", xs, ys, PALETTE[0])
        fig.linepnt[1:] = 0
        set_range(fig, x_range, [ys])
        fig.title = "Episode reward"
        fig.flg_legend = 0

        fig = self.figs["speed"]
        xs, ys = series("walk/forward_speed_mps")
        target = float(self.config.get("target_speed", self.W.TARGET_SPEED))
        set_line(fig, 0, "target", list(x_range), [target, target], DIM)
        set_line(fig, 1, "measured", xs, ys, PALETTE[2])
        fig.linepnt[2:] = 0
        set_range(fig, x_range, [ys, [target]], y_min=0.0)

        # 보상 항목: 최근 값의 크기가 큰 6개만 (색은 항목마다 고정)
        self.figs["terms"].title = "Reward terms (sum per episode, top 6)"
        fig = self.figs["terms"]
        names = list(self.W.REWARD_WEIGHTS)
        terms = [(n, series(f"eval/episode_reward/{n}")) for n in names if s.get(f"eval/episode_reward/{n}")]
        terms = sorted(terms, key=lambda item: -abs(item[1][1][-1]))[:6]
        all_y = []
        for i, (n, (xs, ys)) in enumerate(terms):
            set_line(fig, i, n, xs, ys, PALETTE[names.index(n) % len(PALETTE)])
            all_y.append(ys)
        fig.linepnt[len(terms):] = 0
        set_range(fig, x_range, all_y)

        # 걸음 방식의 변화: 학습 기록에 있으면 그것을(평가 128개 평균), 없으면(예전 학습) 이 창에서 잰 값
        fig = self.figs["gait_hist"]
        if s.get("walk/gait_flight_pct"):
            fig.title = "Gait over training [% of time]"
            lines = [(k, series(f"walk/gait_{k}_pct")) for k in ("flight", "single", "double")]
        else:
            fig.title = "Gait [%]  (measured in this window)"
            m = np.array(self.measured).reshape(-1, 4)
            lines = [(k, (m[:, 0], m[:, i + 1])) for i, k in enumerate(("flight", "single", "double"))]
        labels = {"flight": "both in air", "single": "one foot", "double": "both down"}
        for i, (k, (xs, ys)) in enumerate(lines):
            set_line(fig, i, labels[k], xs, ys, PALETTE[(3, 0, 2)[i]])
        fig.linepnt[3:] = 0
        set_range(fig, x_range, [], y_min=0.0, y_max=100.0)
