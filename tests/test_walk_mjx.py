"""GPU 보행 학습 환경 테스트 — GPU 학습 환경(.venv-mjx)에서만 실행됨 (그 외에는 건너뜀).

실행:  source scripts/activate_gpu.sh && python -m pytest tests/test_walk_mjx.py
(학습 중인 GPU를 방해하지 않도록 JAX를 CPU로 실행한다. 첫 컴파일에 1분 정도 걸림)
"""
import os
import sys
from pathlib import Path

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
    assert gait_fig.linepnt[0] == 10                        # 왼발 접촉 10스텝
    dash.key_callback(KEY_ENTER)                            # Enter → 로봇 보기: 발 접촉 그래프 하나만
    dash.key_callback(KEY_ENTER)                            # 뷰어는 키를 뗄 때도 부름 → 무시되어야 함
    assert dash.mode == "robot"
    for k in range(10, 15):
        dash.on_step(viewer, {"foot_z": np.array([0.02, 0.08]), "vx": 0.3}, (k + 1) * 0.02, 0.0)
    assert len(viewer.figs) == 1 and viewer.figs[0][1].title.startswith("Feet:")


def test_analyze_gait_merges_taps_and_measures_asymmetry():
    """learning/analyze_gait.py: 살짝 튕긴 착지는 걸음으로 세지 않고, 좌우 다리 차이는 반 박자 어긋나게 비교한다."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "learning"))
    from analyze_gait import leg_asymmetry_deg, merge_taps

    # 발 높이: 땅(0.02) → 크게 들기(0.10, 걸음) → 땅 → 살짝 들기(0.025, 튕김) → 땅
    z = np.array([0.02] * 5 + [0.10] * 5 + [0.02] * 5 + [0.025] * 2 + [0.02] * 5)
    foot_z = np.stack([z, np.full_like(z, 0.02)], axis=1)
    contact = foot_z < 0.022
    merged, taps = merge_taps(contact, foot_z)
    assert taps == [1, 0]
    assert merged[:, 0].sum() == contact[:, 0].sum() + 2      # 튕긴 2스텝이 '딛고 있음'으로 합쳐짐
    assert not merged[5:10, 0].any()                          # 진짜 걸음(높이 든 구간)은 그대로

    # 오른다리가 왼다리와 똑같은 동작을 반 박자(20스텝) 늦게 하면 차이 0
    t = np.arange(200) * 0.02
    left = np.stack([np.sin(2 * np.pi * t / 0.8)] * 3, axis=1)
    right = np.stack([np.sin(2 * np.pi * (t - 0.4) / 0.8)] * 3, axis=1)
    assert leg_asymmetry_deg(np.concatenate([left, right], axis=1), 20) < 1e-6
    assert leg_asymmetry_deg(np.concatenate([left, right + 0.2], axis=1), 20) > 5   # 한쪽만 0.2 rad 치우치면 큼


def test_pretrained_walk_policy_walks_symmetrically():
    """learning/pretrained/walk_policy.pkl (docs/08 §6의 결과): 일반 MuJoCo에서 10초 동안 넘어지지 않고
    목표 속도로, 절뚝이지 않고 걷는다."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "learning"))
    from analyze_gait import gait_metrics, record
    from play_walk import MujocoRunner, load_policy

    path = os.path.join(os.path.dirname(__file__), "..", "learning", "pretrained", "walk_policy.pkl")
    policy, config = load_policy(Path(path).resolve())
    runner = MujocoRunner(config["target_speed"])
    rec, _ = record(runner, policy, render=False)
    m = gait_metrics(rec, runner.info["joint_names"], runner.W.CONTROL_DT)
    assert not m["fell"]
    assert m["speed_mps"] == pytest.approx(0.30, abs=0.03)
    assert m["both_air_pct"] < 5
    assert m["limp_score_pct"] < 5          # 측정값 2.1 %
    assert abs(m["yaw_deg"]) < 10
