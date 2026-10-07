"""강화학습 예제 테스트 (학습 패키지가 없으면 건너뜀: bash scripts/setup_learning.sh)."""
import subprocess
import sys

import numpy as np
import pytest

from biped_sim import paths

try:
    import stable_baselines3  # noqa: F401
    from biped_sim.envs import BipedBalanceEnv, evaluate_push_recovery, zero_policy
    HAVE_LEARNING = True
except ImportError:
    HAVE_LEARNING = False

# 파일 전체를 건너뛰는 importorskip 대신 테스트마다 건너뜀 (pyproject.toml의 launch_testing 주석 참고)
pytestmark = pytest.mark.skipif(not HAVE_LEARNING, reason="학습 패키지 없음 → bash scripts/setup_learning.sh")

PRETRAINED = paths.REPO_ROOT / "learning" / "pretrained" / "balance_ppo.zip"


def test_env_follows_gymnasium_api():
    from gymnasium.utils.env_checker import check_env
    env = BipedBalanceEnv()
    check_env(env, skip_render_check=True)
    obs, info = env.reset(seed=0)
    assert obs.shape == (28,) and env.action_space.shape == (6,)
    assert info["fell"] is False


def test_zero_action_equals_pd_baseline():
    """행동 0 = PD로 home 자세 유지 (t06과 같음): 20 N은 버티고 30 N은 넘어진다."""
    result = evaluate_push_recovery(zero_policy, forces=(20, 30), push_times=(1.0,))
    assert result[20] == (2, 2)
    assert result[30] == (0, 2)


def test_pretrained_policy_beats_pd_baseline():
    from stable_baselines3 import PPO
    model = PPO.load(PRETRAINED, device="cpu")
    result = evaluate_push_recovery(lambda o: model.predict(o, deterministic=True)[0],
                                    forces=(30,), push_times=(1.0,))
    assert result[30] == (2, 2), "학습한 정책은 PD만으로는 넘어지는 30 N을 버텨야 함"


def test_train_script_runs(tmp_path):
    """학습 스크립트가 끝까지 돌고 모델·기록·그래프를 저장하는지 (아주 짧게)."""
    proc = subprocess.run([sys.executable, str(paths.REPO_ROOT / "learning" / "train_balance.py"),
                           "--steps", "2048", "--envs", "2", "--output-dir", str(tmp_path)],
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    run_dirs = list(tmp_path.glob("balance_2*"))
    assert len(run_dirs) == 1
    for name in ("model.zip", "progress.csv", "learning_curve.png"):
        assert (run_dirs[0] / name).exists(), name
    assert (tmp_path / "balance_latest.zip").exists()
    assert "PD만(학습 전)" in proc.stdout


def test_observation_is_finite_after_fall():
    env = BipedBalanceEnv()
    obs, _ = env.reset(seed=0, options={"push_force": 60.0, "push_direction": 1.0, "push_time": 0.5})
    for _ in range(env.max_steps):
        obs, _, terminated, truncated, info = env.step(np.zeros(6))
        if terminated or truncated:
            break
    assert info["fell"] and np.all(np.isfinite(obs))
