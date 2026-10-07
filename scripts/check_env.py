#!/usr/bin/env python3
"""개발 환경 점검 스크립트.

사용법:  source scripts/activate.sh && python scripts/check_env.py

각 항목을 [OK] / [WARN] / [FAIL]로 표시하고, FAIL이 하나라도 있으면 종료 코드 1을 돌려줍니다.
문제가 생기면 이 출력 전체를 그대로 공유하면 원인 파악이 빠릅니다.
"""
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MUJOCO = "3.15.0"
results = []


def report(status, name, detail=""):
    results.append(status)
    mark = {"OK": "\033[32m[OK]  \033[0m", "WARN": "\033[33m[WARN]\033[0m",
            "FAIL": "\033[31m[FAIL]\033[0m"}[status]
    print(f"{mark} {name}" + (f"  — {detail}" if detail else ""))


def check_python():
    exe = Path(sys.executable)
    ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    in_venv = sys.prefix != sys.base_prefix
    if in_venv and Path(sys.prefix).resolve() == (REPO_ROOT / ".venv").resolve():
        report("OK", f"Python {ver} (프로젝트 .venv)", str(exe))
    else:
        report("FAIL", f"Python {ver} — 프로젝트 .venv가 아님", f"{exe}  → source scripts/activate.sh")
    if sys.version_info[:2] != (3, 10):
        report("WARN", "Python 버전", "ROS2 Humble은 Python 3.10 기준입니다")
    if os.environ.get("CONDA_PREFIX"):
        report("FAIL", "conda 환경이 켜져 있음", f"{os.environ['CONDA_PREFIX']} → conda deactivate")
    if os.environ.get("PYTHONNOUSERSITE") == "1":
        report("OK", "~/.local 패키지 차단 (PYTHONNOUSERSITE=1)")
    else:
        report("WARN", "~/.local 패키지가 섞일 수 있음", "activate.sh로 활성화하면 자동 설정됩니다")
    # ~/.bashrc 등에서 추가한 PYTHONPATH는 venv보다 우선하므로 패키지 충돌의 흔한 원인
    extra = [p for p in os.environ.get("PYTHONPATH", "").split(":")
             if p and not p.startswith("/opt/ros/")]
    if extra:
        report("WARN", "ROS 외의 PYTHONPATH 항목이 있음 (충돌 시 의심)", ", ".join(extra))


def check_numpy():
    try:
        import numpy
    except ImportError as e:
        return report("FAIL", "numpy import 실패", str(e))
    major = int(numpy.__version__.split(".")[0])
    if major >= 2:
        report("FAIL", f"numpy {numpy.__version__}", "ROS2 Humble과 호환되지 않음 → pip install 'numpy<2'")
    else:
        report("OK", f"numpy {numpy.__version__}", numpy.__file__)


def check_mujoco():
    try:
        import mujoco
    except ImportError as e:
        return report("FAIL", "mujoco import 실패", f"{e} → bash scripts/setup_env.sh")
    status = "OK" if mujoco.__version__ == EXPECTED_MUJOCO else "WARN"
    report(status, f"mujoco {mujoco.__version__}",
           "" if status == "OK" else f"권장 버전 {EXPECTED_MUJOCO} (requirements.txt)")
    try:
        m = mujoco.MjModel.from_xml_string(
            "<mujoco><worldbody><body><freejoint/><geom size='.1'/></body></worldbody></mujoco>")
        d = mujoco.MjData(m)
        mujoco.mj_step(m, d, nstep=100)
        # 구는 z=0에서 시작 → 0.2초 뒤 낙하 거리 = -z (이론값 ½·g·t² = 0.196 m)
        report("OK", "mujoco 시뮬레이션 동작", f"구가 0.2초 동안 {-d.qpos[2]:.3f} m 낙하 (이론 0.196 m)")
    except Exception as e:  # noqa: BLE001 — 진단 도구이므로 모든 예외를 보고
        report("FAIL", "mujoco 시뮬레이션 실패", str(e))


def check_project():
    try:
        import mujoco
        from biped_sim import SIMPLE_BIPED, SimConfig, build_robot_model, paths
    except ImportError as e:
        return report("FAIL", "biped_sim import 실패", f"{e} → pip install -e .")
    for p in (paths.FALLING_BOX_XML, paths.PENDULUM_URDF, paths.SIMPLE_BIPED_URDF):
        if not p.exists():
            report("FAIL", "모델 파일 없음", str(p))
    try:
        model, _ = build_robot_model(SIMPLE_BIPED.urdf, SimConfig())
        report("OK", "simple_biped 빌드", f"nq={model.nq} nv={model.nv} nu={model.nu} "
               f"질량={model.body_subtreemass[1]:.2f} kg")
    except Exception as e:  # noqa: BLE001
        report("FAIL", "simple_biped 빌드 실패", str(e))


def check_ros2():
    distro = os.environ.get("ROS_DISTRO")
    if not distro:
        return report("WARN", "ROS2 환경이 로드되지 않음", "source scripts/activate.sh (Phase 5부터 필요)")
    try:
        import rclpy  # noqa: F401
        from sensor_msgs.msg import JointState
        msg = JointState()
        msg.name = ["a", "b"]
        msg.position = [0.1, 0.2]
        report("OK", f"ROS2 {distro}: rclpy + sensor_msgs", "venv 안에서 ROS2 메시지 생성 가능")
    except Exception as e:  # noqa: BLE001
        report("FAIL", f"ROS2 {distro} 패키지 import 실패", str(e))


def check_display():
    gl = os.environ.get("MUJOCO_GL", "(기본: glfw)")
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        report("OK", "화면(DISPLAY) 있음 → 뷰어 사용 가능", f"MUJOCO_GL={gl}")
    else:
        report("WARN", "화면 없음 → 뷰어 불가, --headless 사용", "스냅샷은 MUJOCO_GL=egl 로 가능")
    # 오프스크린 렌더링은 GL 컨텍스트 문제로 프로세스가 죽을 수 있어 별도 프로세스에서 시험
    code = ("import mujoco; m=mujoco.MjModel.from_xml_string("
            "'<mujoco><worldbody><geom size=\".1\"/></worldbody></mujoco>');"
            "r=mujoco.Renderer(m,64,64); r.update_scene(mujoco.MjData(m)); r.render(); r.close()")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    if proc.returncode == 0:
        report("OK", "오프스크린 렌더링 (스냅샷 저장 가능)")
    else:
        last = (proc.stderr.strip().splitlines() or ["?"])[-1]
        report("WARN", "오프스크린 렌더링 실패", f"{last} → MUJOCO_GL=egl 로 다시 시도")


def main():
    print(f"=== biped_sim 환경 점검 ({REPO_ROOT}) ===")
    check_python()
    check_numpy()
    check_mujoco()
    check_project()
    check_ros2()
    check_display()
    n_fail, n_warn = results.count("FAIL"), results.count("WARN")
    print(f"\n결과: FAIL {n_fail}개, WARN {n_warn}개")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
