"""GPU 보행 학습 환경 테스트 — GPU 학습 환경(.venv-mjx)에서만 실행됨 (그 외에는 건너뜀).

실행:  source scripts/activate_gpu.sh && python -m pytest tests/test_walk_mjx.py
(학습 중인 GPU를 방해하지 않도록 JAX를 CPU로 실행한다. 첫 컴파일에 1분 정도 걸림)
"""
import os
import sys

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

try:
    import jax
    import jax.numpy as jnp

    from biped_sim.envs import walk_mjx
    HAVE_MJX = True
except ImportError:
    HAVE_MJX = False

pytestmark = pytest.mark.skipif(not HAVE_MJX, reason="GPU 학습 환경 아님 → bash scripts/setup_gpu_learning.sh")


@pytest.fixture(scope="module")
def env():
    return walk_mjx.BipedWalkMjxEnv()


def test_reset_and_standing_with_zero_action(env):
    """행동 0 = PD로 home 자세 유지 → 1초 동안 넘어지지 않고, 관측이 유한하고, 보상 항목이 기록된다."""
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    assert state.obs.shape == (walk_mjx.OBS_SIZE,)
    step = jax.jit(env.step)
    for _ in range(50):
        state = step(state, jnp.zeros(walk_mjx.ACTION_SIZE))
    assert float(state.done) == 0.0
    assert np.all(np.isfinite(np.asarray(state.obs)))
    height = float(state.pipeline_state.xpos[env.base_body, 2])
    assert height == pytest.approx(walk_mjx.NOMINAL_HEIGHT, abs=0.03)
    for name in walk_mjx.REWARD_WEIGHTS:
        assert f"reward/{name}" in state.metrics
    for name in walk_mjx.GAIT_METRICS:  # 보상과 별개로 늘 기록하는 걸음 지표
        assert f"gait/{name}" in state.metrics
    # 서 있기만 하면 두 발 모두 땅: 양발 1, 한 발 0, 공중 0
    assert float(state.metrics["gait/double"]) == 1.0
    assert float(state.metrics["gait/flight"]) == 0.0


def test_cpu_mujoco_runner_matches_mjx_observation(env):
    """learning/play_walk.py의 일반 MuJoCo 재생기가 학습 환경과 '같은 관측'을 만드는지 (sim-to-sim의 전제)."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "learning"))
    from play_walk import MujocoRunner

    runner = MujocoRunner(target_speed=env.target_speed)
    obs_cpu = runner.reset()
    # MJX 환경을 같은 상태(home, 잡음 없음)에서 시작
    state = env.reset(jax.random.PRNGKey(0))
    data = state.pipeline_state.replace(qpos=jnp.asarray(runner.data.qpos, dtype=jnp.float32))
    from mujoco import mjx
    data = mjx.forward(env.mx, data)
    obs_mjx = np.asarray(env._observation(data, state.info))
    np.testing.assert_allclose(obs_cpu, obs_mjx, atol=1e-4)

    # 같은 행동을 넣고 한 스텝 진행해도 비슷해야 함 (시뮬레이터 구현 차이 수준)
    action = np.full(walk_mjx.ACTION_SIZE, 0.2, dtype=np.float32)
    obs_cpu = runner.step(action)
    state = env.step(state.replace(pipeline_state=data), jnp.asarray(action))
    np.testing.assert_allclose(obs_cpu[:3], np.asarray(state.obs)[:3], atol=1e-2)       # 중력 방향
    np.testing.assert_allclose(obs_cpu[10:16], np.asarray(state.obs)[10:16], atol=1e-2)  # 관절 각도


def test_live_dashboard_reads_tensorboard_log_and_draws(tmp_path):
    """학습 화면(learning/live_dashboard.py): TensorBoard 기록을 직접 읽고, 두 화면 모드에서 그래프·글자를 만든다."""
    import mujoco
    from tensorboardX import SummaryWriter

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "learning"))
    from live_dashboard import KEY_ENTER, Dashboard, ScalarLog

    writer = SummaryWriter(str(tmp_path))
    for i, step in enumerate((0, 1_000_000, 2_000_000)):
        writer.add_scalar("eval/episode_reward", float(i), step)
        writer.add_scalar("walk/gait_flight_pct", 50.0 - 20 * i, step)
    writer.close()   # tensorboardX는 별도 스레드로 쓰므로 close()해야 파일에 다 들어감
    log = ScalarLog(tmp_path)
    assert log.update()["eval/episode_reward"] == [(0, 0.0), (1_000_000, 1.0), (2_000_000, 2.0)]
    writer = SummaryWriter(str(tmp_path), filename_suffix=".2")   # 나중에 더 쓴 기록
    writer.add_scalar("eval/episode_reward", 3.0, 3_000_000)
    writer.close()
    assert log.update()["eval/episode_reward"][-1] == (3_000_000, 3.0)
    assert len(log.scalars["eval/episode_reward"]) == 4      # 이미 읽은 부분은 다시 읽지 않음 (중복 없음)

    class FakeViewer:  # 창 없이 시험: 뷰어가 받는 값만 기록
        viewport = mujoco.MjrRect(0, 0, 1280, 720)

        def set_figures(self, figs):
            self.figs = figs

        def set_texts(self, texts):
            self.texts = texts

    viewer = FakeViewer()
    dash = Dashboard(walk_mjx, {"target_speed": 0.3, "steps": 3_000_000, "saved_step": 3_000_000}, tmp_path)
    dash.start_episode()
    for k in range(10):
        dash.on_step(viewer, {"foot_z": np.array([0.02, 0.08]), "vx": 0.3}, (k + 1) * 0.02, 0.0)
    titles = [fig.title for _, fig in viewer.figs]
    assert len(titles) == 5 and titles[0] == "Episode reward" and titles[4].startswith("Feet:")
    reward_fig = viewer.figs[0][1]
    assert reward_fig.linepnt[0] == 4                       # 기록 4개가 선 하나로
    gait_hist = viewer.figs[3][1]
    assert gait_hist.linepnt[0] == 3                        # 학습 기록의 walk/gait_flight_pct 3개
    gait_fig = viewer.figs[4][1]
    assert gait_fig.linepnt[2] == 10                        # 왼발 접촉 10스텝
    dash.key_callback(KEY_ENTER)                            # Enter → 로봇 보기: 발 접촉 그래프 하나만
    dash.key_callback(KEY_ENTER)                            # 뷰어는 키를 뗄 때도 부름 → 무시되어야 함
    assert dash.mode == "robot"
    for k in range(10, 15):
        dash.on_step(viewer, {"foot_z": np.array([0.02, 0.08]), "vx": 0.3}, (k + 1) * 0.02, 0.0)
    assert len(viewer.figs) == 1 and viewer.figs[0][1].title.startswith("Feet:")
