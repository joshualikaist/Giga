"""GPU 학습 환경(.venv-mjx) 확인 — 다른 환경(.venv, 시스템 파이썬)에서 실행하면 .venv-mjx의 파이썬으로 다시 실행한다.

보행 학습·재생 스크립트(train_walk_gpu, play_walk, analyze_gait, terrain_trial)는 JAX·Brax가 필요한데,
이것들은 ROS2와 섞이지 않도록 별도 환경(.venv-mjx)에만 있다 (docs/08 §2). source scripts/activate_gpu.sh를
잊고 ROS용 .venv 터미널에서 실행해도 동작하도록, activate_gpu.sh와 같은 설정으로 스스로 다시 실행한다.
"""
import importlib.util
import os
import sys
from pathlib import Path

VENV = Path(__file__).resolve().parents[1] / ".venv-mjx"


def ensure_gpu_env() -> None:
    if importlib.util.find_spec("jax") and importlib.util.find_spec("brax"):
        return   # 이미 GPU 학습 환경
    python = VENV / "bin" / "python"
    if not python.exists():
        raise SystemExit("이 스크립트는 GPU 학습 환경(.venv-mjx)이 필요합니다 → 처음 한 번: bash scripts/setup_gpu_learning.sh")
    if Path(sys.prefix).resolve() == VENV.resolve():
        raise SystemExit("GPU 학습 환경에 jax/brax가 없습니다 → bash scripts/setup_gpu_learning.sh 를 다시 실행하세요")
    print("[알림] GPU 학습 환경(.venv-mjx)으로 다시 실행합니다 (직접 켜려면: source scripts/activate_gpu.sh)", flush=True)
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}   # ROS 경로가 섞이지 않게
    env.update(PYTHONNOUSERSITE="1", VIRTUAL_ENV=str(VENV), PATH=f"{VENV / 'bin'}:{env.get('PATH', '')}")
    env.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.7")
    os.execve(str(python), [str(python), *sys.argv], env)
