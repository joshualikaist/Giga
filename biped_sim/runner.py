"""시뮬레이션 실행 루프와 화면 저장 도구.

모든 시뮬레이션은 결국 아래 루프입니다:

    while 시간이 남았으면:
        step_fn(model, data)      # ① 센서 읽기 → 제어 계산 → data.ctrl 쓰기
        mujoco.mj_step(model, data)  # ② 물리 엔진이 timestep만큼 시간을 진행

뷰어 모드에서는 화면 갱신(60 Hz)과 실시간 맞추기(sleep)가 추가될 뿐 구조는 같습니다.
"""
from __future__ import annotations

import argparse
import contextlib
import os
import threading
import time
from pathlib import Path
from typing import Callable, Iterator

import mujoco

StepFn = Callable[[mujoco.MjModel, mujoco.MjData], None]
ViewerSetup = Callable[[object], None]


def add_common_args(parser: argparse.ArgumentParser, default_duration: float | None = None
                    ) -> argparse.ArgumentParser:
    parser.add_argument("--headless", action="store_true",
                        help="뷰어 없이 실행하고 결과만 출력 (원격 서버, 자동 테스트용)")
    parser.add_argument("--duration", type=float, default=default_duration,
                        help="시뮬레이션 시간 [s] (뷰어 모드에서 생략하면 창을 닫을 때까지)")
    parser.add_argument("--snapshot", action="store_true",
                        help="종료 시점 장면을 output/ 폴더에 PNG로 저장")
    return parser


@contextlib.contextmanager
def passive_viewer(model: mujoco.MjModel, data: mujoco.MjData, show_ui: bool = True,
                   key_callback: Callable[[int], None] | None = None) -> Iterator[object]:
    """mujoco.viewer.launch_passive를 안전하게 열고 닫는 with 블록.

        with passive_viewer(model, data) as viewer:
            while viewer.is_running(): ...

    왜 launch_passive를 직접 쓰지 않나?
        launch_passive는 화면 그리기를 daemon 스레드에서 돌리고, close()는 "종료 요청"만 보낸 뒤
        바로 돌아온다. 그 스레드가 끝나기 전에 프로그램이 종료되면 세그폴트(exit 139)나
        X 에러(GLXBadContext)가 나거나 프로세스가 멈춘다 (MuJoCo 3.15, 이 PC에서 반복 재현).
        → 띄울 때 생긴 스레드를 기억해 두었다가, 닫은 뒤 끝날 때까지 기다린다.
    """
    # 화면이 없는 서버에서도 헤드리스 모드는 동작하도록 뷰어는 여기서 import.
    # (주의: 함수 안에서 `import mujoco.viewer`라고 쓰면 `mujoco`가 지역 변수가 되어 위쪽 코드가 깨짐)
    import mujoco.viewer as mj_viewer

    # 화면 갱신을 모니터 주기(보통 60 Hz)에 맞춘다 (NVIDIA 드라이버 설정). 안 하면 뷰어가 쉬지 않고 다시 그려서
    # GTX 1650 기준 GPU를 49~96 % 차지함 (실측) → GPU 학습이 느려짐. 켜면 측정되지 않을 만큼 줄어듦.
    os.environ.setdefault("__GL_SYNC_TO_VBLANK", "1")

    before = set(threading.enumerate())
    # show_ui=False: 왼쪽/오른쪽 설정 패널을 숨겨 로봇을 크게 (창에서 Tab / Shift+Tab으로 다시 켤 수 있음)
    # key_callback(keycode): 창에서 키를 누를 때마다 화면 스레드에서 호출 (keycode는 GLFW 번호, 예: Enter = 257)
    viewer = mj_viewer.launch_passive(model, data, key_callback=key_callback,
                                      show_left_ui=show_ui, show_right_ui=show_ui)
    viewer_threads = [t for t in threading.enumerate() if t not in before]
    try:
        yield viewer
    finally:
        viewer.close()
        for t in viewer_threads:
            t.join(timeout=5.0)


def simulate(model: mujoco.MjModel, data: mujoco.MjData, step_fn: StepFn | None = None,
             duration: float | None = None, headless: bool = False, realtime: bool = True,
             viewer_setup: ViewerSetup | None = None) -> None:
    """step_fn을 매 물리 스텝 직전에 호출하며 시뮬레이션을 진행한다."""
    if headless:
        if duration is None:
            raise ValueError("헤드리스 모드에서는 duration이 필요합니다.")
        for _ in range(int(round(duration / model.opt.timestep))):
            if step_fn:
                step_fn(model, data)
            mujoco.mj_step(model, data)
        return

    # 물리는 500 Hz(0.002 s)로 돌지만 화면은 60 Hz면 충분 → 한 프레임에 여러 스텝을 묶어서 진행
    steps_per_frame = max(1, round((1.0 / 60.0) / model.opt.timestep))

    try:
        _viewer_loop(model, data, step_fn, duration, realtime, viewer_setup, steps_per_frame)
    except KeyboardInterrupt:  # 터미널에서 Ctrl+C: 뷰어를 정상적으로 닫고 돌아감
        print("\n[biped_sim] Ctrl+C — 시뮬레이션을 종료합니다.")


def _viewer_loop(model, data, step_fn, duration, realtime, viewer_setup, steps_per_frame) -> None:
    with passive_viewer(model, data) as viewer:
        if viewer_setup:
            with viewer.lock():
                viewer_setup(viewer)
        t_start = data.time
        wall_start = time.perf_counter()
        while viewer.is_running():
            if duration is not None and data.time - t_start >= duration:
                break
            prev_time = data.time
            for _ in range(steps_per_frame):
                if step_fn:
                    step_fn(model, data)
                mujoco.mj_step(model, data)
            viewer.sync()  # 화면 갱신 + 마우스로 가한 힘(Ctrl+드래그)을 data에 반영

            if data.time < prev_time:  # 뷰어에서 리셋(Backspace)하면 시간이 0으로 돌아감
                t_start, wall_start = data.time, time.perf_counter()
            if realtime:  # 시뮬레이션 시간이 실제 시간보다 앞서면 기다린다
                ahead = (data.time - t_start) - (time.perf_counter() - wall_start)
                if ahead > 0:
                    time.sleep(ahead)


def track_body_camera(body_id: int, distance: float = 2.0, azimuth: float = 135.0,
                      elevation: float = -15.0) -> ViewerSetup:
    """카메라가 특정 바디(보통 로봇 몸통)를 따라가게 하는 viewer_setup 함수를 만든다."""
    def setup(viewer) -> None:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = body_id
        viewer.cam.distance = distance
        viewer.cam.azimuth = azimuth
        viewer.cam.elevation = elevation
    return setup


def save_snapshot(model: mujoco.MjModel, data: mujoco.MjData, path: str | Path,
                  lookat=(0.0, 0.0, 0.5), distance: float = 2.0, azimuth: float = 135.0,
                  elevation: float = -15.0, width: int = 640, height: int = 480) -> Path:
    """현재 장면을 오프스크린 렌더링해 PNG로 저장한다.

    화면 없는 서버에서는 실행 전에 `export MUJOCO_GL=egl` (NVIDIA/Intel GPU) 또는
    `MUJOCO_GL=osmesa` (CPU 소프트웨어 렌더링)를 지정하세요.
    기본 오프스크린 버퍼는 640×480이며, 더 크게 하려면 모델의 <visual><global offwidth/offheight>를 키워야 합니다.
    """
    import matplotlib.image  # pyplot 없이 PNG만 저장

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation

    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        renderer.update_scene(data, camera=cam)
        image = renderer.render()
    finally:
        renderer.close()  # GL 컨텍스트를 명시적으로 해제 (안 하면 종료 시 EGL 경고가 뜰 수 있음)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    matplotlib.image.imsave(path, image)
    return path
