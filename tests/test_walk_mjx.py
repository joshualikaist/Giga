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
LEARNING_DIR = Path(__file__).resolve().parents[1] / "learning"
sys.path.insert(0, str(LEARNING_DIR))   # learning/의 스크립트 모듈(walk_tools, live_dashboard)을 import하려고

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
    state = _stands_with_zero_action(env)
    for name in walk_mjx.REWARD_WEIGHTS:
        assert f"reward/{name}" in state.metrics
    for name in walk_mjx.GAIT_METRICS:  # 보상과 별개로 늘 기록하는 걸음 지표
        assert f"gait/{name}" in state.metrics
    # 서 있기만 하면 두 발 모두 땅: 양발 1, 한 발 0, 공중 0
    assert float(state.metrics["gait/double"]) == 1.0
    assert float(state.metrics["gait/flight"]) == 0.0


def _stands_with_zero_action(env):
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    assert state.obs.shape == (env.spec.obs_size,)
    step = jax.jit(env.step)
    for _ in range(50):
        state = step(state, jnp.zeros(env.spec.n))
    assert float(state.done) == 0.0
    assert np.all(np.isfinite(np.asarray(state.obs)))
    height = float(state.pipeline_state.xpos[env.base_body, 2])
    assert height == pytest.approx(env.spec.nominal_height, abs=0.05 * env.spec.nominal_height)
    return state


def test_cpu_mujoco_runner_matches_mjx_observation(env):
    _cpu_runner_matches_mjx(env)


def _cpu_runner_matches_mjx(env):
    """learning/walk_tools.py의 일반 MuJoCo 재생기가 학습 환경과 '같은 관측'을 만드는지 (sim-to-sim의 전제)."""
    from walk_tools import MujocoRunner

    runner = MujocoRunner(env.spec, env.target_speed)
    obs_cpu = runner.reset()
    # MJX 환경을 같은 상태(home, 잡음 없음)에서 시작
    state = env.reset(jax.random.PRNGKey(0))
    data = state.pipeline_state.replace(qpos=jnp.asarray(runner.data.qpos, dtype=jnp.float32))
    from mujoco import mjx
    data = mjx.forward(env.mx, data)
    obs_mjx = np.asarray(env._observation(data, state.info))
    np.testing.assert_allclose(obs_cpu, obs_mjx, atol=1e-4)

    # 같은 행동을 넣고 한 스텝 진행해도 비슷해야 함 (시뮬레이터 구현 차이 수준)
    action = np.full(env.spec.n, 0.2, dtype=np.float32)
    obs_cpu = runner.step(action)
    state = env.step(state.replace(pipeline_state=data), jnp.asarray(action))
    n = env.spec.n
    np.testing.assert_allclose(obs_cpu[:3], np.asarray(state.obs)[:3], atol=1e-2)               # 중력 방향
    np.testing.assert_allclose(obs_cpu[10:10 + n], np.asarray(state.obs)[10:10 + n], atol=1e-2)  # 관절 각도


@pytest.mark.skipif(not HAVE_MJX or not walk_mjx.SPECS["open_duck_mini"].robot.urdf.exists(),
                    reason="오리 로봇 파일 없음 → bash scripts/get_open_duck.sh")
def test_open_duck_walk_env():
    """오리 보행 환경: 다리 10개만 정책, 머리는 PD로 고정. 행동 0으로 서 있고, 일반 MuJoCo 재생기와 관측이 같다.
    좌우 부호표는 FK로 자동 계산: hip yaw·roll·pitch는 반대(−1), knee·ankle은 같음(+1) (docs/09)."""
    duck = walk_mjx.BipedWalkMjxEnv("open_duck_mini")
    assert duck.spec.obs_size == 43 and duck.spec.n == 10
    assert list(np.asarray(duck.mirror_sign)) == [-1.0, -1.0, -1.0, 1.0, 1.0]
    _stands_with_zero_action(duck)
    _cpu_runner_matches_mjx(duck)


@pytest.mark.skipif(not HAVE_MJX or not walk_mjx.SPECS["open_duck_mini"].robot.urdf.exists(),
                    reason="오리 로봇 파일 없음 → bash scripts/get_open_duck.sh")
def test_open_duck_servo_model_and_random_friction():
    """실물 측정 서보 모델(--servo open_duck)과 바닥 마찰 무작위(--friction): 모델에 반영되고, 행동 0으로 서 있다."""
    spec = walk_mjx.with_servo(walk_mjx.SPECS["open_duck_mini"], "open_duck")
    duck = walk_mjx.BipedWalkMjxEnv(spec, friction_range=(0.4, 1.0))
    m = duck.mj_model
    assert float(duck.kp[0]) == pytest.approx(13.37) and float(duck.kd[0]) == 0.0
    assert m.dof_damping[6] == pytest.approx(0.56) and m.dof_frictionloss[6] == pytest.approx(0.068)
    state = _stands_with_zero_action(duck)
    assert 0.4 <= float(state.info["friction"]) <= 1.0
    assert walk_mjx.with_servo(walk_mjx.SPECS["open_duck_mini"], "ideal") is walk_mjx.SPECS["open_duck_mini"]


@pytest.mark.skipif(not HAVE_MJX or not walk_mjx.SPECS["open_duck_mini"].robot.urdf.exists(),
                    reason="오리 로봇 파일 없음 → bash scripts/get_open_duck.sh")
def test_open_duck_home_knee_posture():
    """서 있는 자세 바꾸기(--home-knee): 무릎을 덜 굽히면 몸통이 높아지고(기준 높이도 같이), 한 발 무릎 토크가 줄고,
    행동 0으로 서 있으면 새 기준 높이를 지킨다. 학습 설정에 저장되어 재생 때도 같은 자세 (walk_tools.get_spec)."""
    import mujoco

    import feasibility as F
    from walk_tools import MujocoRunner, get_spec

    base = walk_mjx.SPECS["open_duck_mini"]
    spec = get_spec({"robot": "open_duck_mini", "home_knee": 0.6})
    assert spec.robot.home_pose["left_knee"] == 0.6 and spec.robot.home_pose["right_hip_pitch"] == pytest.approx(0.3)
    assert 0.003 < spec.nominal_height - base.nominal_height < 0.008                  # 약 5 mm 높아짐
    assert base.robot.home_pose["left_knee"] == pytest.approx(0.8)                     # 원래 로봇 설정은 그대로
    runner = MujocoRunner(spec)
    data = mujoco.MjData(runner.model)
    data.qpos[:] = runner.info["home_qpos"]
    mujoco.mj_forward(runner.model, data)
    assert abs(F.static_torques(runner.model, data, runner.info["base_body"])["knee"]) < F.SERVO["rated"]   # 0.61 < 0.64
    runner.reset()
    for _ in range(50):
        runner.step(np.zeros(spec.n))
    height = runner.data.xpos[runner.info["base_body"], 2]
    assert height == pytest.approx(spec.nominal_height, abs=0.05 * spec.nominal_height)


@pytest.mark.skipif(not HAVE_MJX or not walk_mjx.SPECS["open_duck_mini"].robot.urdf.exists(),
                    reason="오리 로봇 파일 없음 → bash scripts/get_open_duck.sh")
def test_feasibility_estimates_for_open_duck():
    """동역학 한계 계산(learning/feasibility.py, docs/09 §8): 2026-10-08 실측값이 그대로 나오는지와 물리적으로 맞는 관계."""
    import feasibility as F
    from walk_tools import MujocoRunner

    runner = MujocoRunner("open_duck_mini")
    p = F.predictions(runner, {"duty": 0.7, "single_support_s": 0.15, "clearance_m": 0.02})
    t = p["static_torque_Nm"]
    assert abs(t["hip_yaw"]) < 1e-6                       # 연직 축 → 중력 토크 없음
    assert abs(t["hip_roll"]) == pytest.approx(0.90, abs=0.02) and abs(t["knee"]) == pytest.approx(0.74, abs=0.02)
    assert p["slope"]["slip_deg"] == pytest.approx(45.0)  # μ = 1
    assert p["slope"]["tip_rigid_uphill_deg"] < p["slope"]["tip_upright_deg"]
    lift = p["fold_lift"]["swing_s"]
    assert lift["0.15"][1] < lift["0.25"][1] <= p["step_up_max_m"] + 0.01   # 흔듦이 길수록 높이, 관절 범위가 상한
    w, (s1, s2) = p["stance_width_m"], p["lipm"]["sway_m"]
    assert 0 < s1 < s2 < w / 2                            # 무게중심은 디딤 발까지 가지 않음
    assert F.available_torque(np.array([1.0, 1.0]), np.array([0.0, F.SERVO["noload"]]), "datasheet") == \
        pytest.approx([F.SERVO["stall"], 0.0])
    assert F.available_torque(np.array([1.0]), np.array([-3.0]), "datasheet")[0] > F.SERVO["stall"]    # 브레이크 방향
    # 학습 보상 servo_overload가 쓰는 데이터시트 값 = 이 분석의 값
    assert walk_mjx.SERVO_DATASHEET == {"stall": F.SERVO["stall"], "noload": F.SERVO["noload"]}


def test_evaluate_walk_writes_report(tmp_path):
    """학습 뒤 자동 평가(learning/evaluate_walk.py): 미리 학습된 정책으로 아주 짧게 돌려 보고서(md·json)가 나오는지."""
    import json
    import subprocess

    params = LEARNING_DIR / "pretrained" / "walk_policy.pkl"
    done = subprocess.run([sys.executable, str(LEARNING_DIR / "evaluate_walk.py"), "--params", str(params),
                           "--no-baseline", "--no-video", "--episodes", "1", "--difficulties", "0.5",
                           "--frictions", "1.0", "--kinds", "flat", "hills", "--out", str(tmp_path)],
                          env={**os.environ, "JAX_PLATFORMS": "cpu"}, capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stderr[-2000:]
    md = (tmp_path / "report.md").read_text()
    assert "## 판정" in md and "## 1. 지형" in md and "| 0.5 / 1.0 |" in md
    result = json.loads((tmp_path / "report.json").read_text())
    assert set(result["new"]["terrain"]["0.5 / 1.0"]) == {"flat", "hills"}
    assert result["new"]["gait"]["survived_s"] == pytest.approx(10.0)
    assert (tmp_path / "gait.png").exists()
    assert not list(params.parent.glob("events.out.tfevents*"))   # 미리 학습된 정책 폴더에는 TensorBoard 기록을 쓰지 않음


@pytest.mark.skipif(not HAVE_MJX or not walk_mjx.SPECS["open_duck_mini"].robot.urdf.exists(),
                    reason="오리 로봇 파일 없음 → bash scripts/get_open_duck.sh")
def test_deployable_obs_hides_privileged_values():
    """실물 관측 모드(--deployable-obs): 몸통 속도·높이 칸이 0이고, 일반 MuJoCo 재생기와 관측이 같으며, 센서 잡음은 학습 환경에만."""
    import dataclasses

    from walk_tools import get_spec

    spec = get_spec({"robot": "open_duck_mini", "deployable_obs": True})
    assert spec.deployable_obs and dataclasses.replace(spec, deployable_obs=False) == walk_mjx.SPECS["open_duck_mini"]
    duck = walk_mjx.BipedWalkMjxEnv(spec)
    _cpu_runner_matches_mjx(duck)
    state = jax.jit(duck.reset)(jax.random.PRNGKey(0))
    state = jax.jit(duck.step)(state, jnp.full(spec.n, 0.3))
    assert np.all(np.asarray(state.obs[6:10]) == 0.0)                 # 속도(3)·높이(1)
    noisy = walk_mjx.BipedWalkMjxEnv(spec, obs_noise=True)
    s1 = jax.jit(noisy.step)(jax.jit(noisy.reset)(jax.random.PRNGKey(0)), jnp.zeros(spec.n))
    s0 = jax.jit(duck.step)(jax.jit(duck.reset)(jax.random.PRNGKey(0)), jnp.zeros(spec.n))
    diff = np.abs(np.asarray(s1.obs - s0.obs))
    assert diff[0:6].max() > 0 and diff[6:10].max() == 0.0             # IMU 칸엔 잡음, 가린 칸은 그대로 0
    # 비대칭 critic: 정책 관측은 가린 그대로, critic 관측(privileged_state)에는 실제 속도·높이
    asym = walk_mjx.BipedWalkMjxEnv(spec, privileged_critic=True)
    s2 = jax.jit(asym.step)(jax.jit(asym.reset)(jax.random.PRNGKey(0)), jnp.full(spec.n, 0.3))
    assert set(s2.obs) == {"state", "privileged_state"} and asym.observation_size["state"] == (spec.obs_size,)
    assert np.all(np.asarray(s2.obs["state"][6:10]) == 0.0) and np.abs(np.asarray(s2.obs["privileged_state"][6:10])).max() > 0
    np.testing.assert_allclose(np.asarray(s2.obs["state"]), np.asarray(state.obs), atol=1e-5)   # 정책 관측은 그대로


@pytest.mark.skipif(not HAVE_MJX or not walk_mjx.SPECS["open_duck_mini"].robot.urdf.exists(),
                    reason="오리 로봇 파일 없음 → bash scripts/get_open_duck.sh")
def test_domain_randomization_and_delay():
    """실물과의 차이 대비(--domain-rand): 환경마다 질량·모터·무게중심·지연이 범위 안에서 정해지고 서 있다.
    제어 지연 k 스텝이면 k 스텝 동안은 home 자세 명령 (일반 MuJoCo 재생기도 같은 방식)."""
    from walk_tools import MujocoRunner

    spec = walk_mjx.SPECS["open_duck_mini"]
    duck = walk_mjx.BipedWalkMjxEnv(spec, domain_rand=True)
    state = _stands_with_zero_action(duck)
    r, i = walk_mjx.DOMAIN_RAND, state.info
    assert abs(float(i["mass_scale"]) - 1) <= r["mass"] and abs(float(i["motor_scale"]) - 1) <= r["motor"]
    assert np.abs(np.asarray(i["com_offset"])).max() <= r["com"] and 0 <= int(i["delay"]) <= r["delay"]
    runner = MujocoRunner(spec)
    runner.delay = 2
    runner.reset()
    q0 = runner.data.qpos[runner.info["qpos_idx"]].copy()
    runner.step(np.ones(spec.n))       # 지연 2스텝: 아직 home 명령 → 거의 움직이지 않음
    moved_delayed = np.abs(runner.data.qpos[runner.info["qpos_idx"]] - q0).max()
    runner.delay = 0
    runner.reset()
    runner.step(np.ones(spec.n))
    moved_now = np.abs(runner.data.qpos[runner.info["qpos_idx"]] - q0).max()
    assert moved_delayed < 0.3 * moved_now


def test_obs_history_and_parameter_expansion(tmp_path):
    """관측 기록(--obs-history): ① 학습 환경은 지금 관측 뒤에 지난 관측을 한 칸씩 밀며 붙이고, 일반 MuJoCo 재생기도 같다.
    ② 기록 없이 학습한 정책을 기록 받는 정책으로 늘리면(새 입력 가중치 0) 같은 상황에서 같은 행동을 낸다."""
    import dataclasses
    import json

    from brax.io import model as brax_model
    from train_walk_gpu import add_history_inputs
    from walk_tools import MujocoRunner, load_policy

    spec = dataclasses.replace(walk_mjx.SPECS["simple_biped"], obs_history=3)
    env = walk_mjx.BipedWalkMjxEnv(spec)
    base = spec.base_obs_size
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    assert state.obs.shape == (base * 4,) and np.allclose(state.obs[base:2 * base], state.obs[:base])
    s1 = jax.jit(env.step)(state, jnp.full(spec.n, 0.2))
    np.testing.assert_allclose(np.asarray(s1.obs[base:2 * base]), np.asarray(state.obs[:base]), atol=1e-6)   # 한 칸 밀림
    runner = MujocoRunner(spec)   # 일반 MuJoCo 재생기도 같은 방식 (MJX는 리셋 때 관절에 잡음을 넣어 값 자체는 다름)
    o0 = runner.reset()
    assert o0.shape == (base * 4,) and np.allclose(o0[base:2 * base], o0[:base])
    o1 = runner.step(np.full(spec.n, 0.2))
    np.testing.assert_allclose(o1[base:2 * base], o0[:base], atol=1e-6)

    src = LEARNING_DIR / "pretrained" / "walk_policy.pkl"
    policy, config = load_policy(src)
    wide = add_history_inputs(brax_model.load_params(str(src)), base, 3, privileged=False)
    brax_model.save_params(str(tmp_path / "params.pkl"), wide)
    (tmp_path / "config.json").write_text(json.dumps({**config, "obs_history": 3}))
    wide_policy, _ = load_policy(tmp_path / "params.pkl")
    obs = runner.reset()
    for _ in range(5):
        action = wide_policy(obs)
        np.testing.assert_allclose(action, policy(obs[:base]), atol=1e-5)
        obs = runner.step(action)


def test_command_mode_conversion_and_tracking(tmp_path):
    """명령 따라 걷기: ① 학습 환경이 모든 학습 옵션(잡음·기록·비대칭 critic·무작위화)과 함께 명령 3칸 관측을 만들고 명령을 바꿈
    ② 목표 속도 1칸 정책을 명령 3칸으로 늘리면 명령 vx = 예전 목표 속도일 때 같은 행동 ③ 명령 시험의 속도 측정이 맞음."""
    import dataclasses
    import json

    from brax.io import model as brax_model
    from evaluate_walk import command_tracking
    from train_walk_gpu import add_command_inputs
    from walk_tools import load_policy, make_runner, run_episode

    spec = dataclasses.replace(walk_mjx.SPECS["simple_biped"], command_obs=True, deployable_obs=True, obs_history=2)
    env = walk_mjx.BipedWalkMjxEnv(spec, privileged_critic=True, domain_rand=True, obs_noise=True, push=(0.3, 1.0),
                                   commands={"vx": (0.2, 0.4), "wz": (-0.3, 0.3), "resample": 0.5})
    state = jax.jit(env.reset)(jax.random.PRNGKey(1))
    step = jax.jit(env.step)
    first = np.asarray(state.info["command"])
    for _ in range(30):
        state = step(state, jnp.zeros(spec.n))
    assert state.obs["state"].shape == (spec.base_obs_size * 3,) and spec.base_obs_size == 13 + 3 * spec.n + 2
    assert not np.allclose(np.asarray(state.info["command"]), first)          # 0.5초마다 새 명령

    src = LEARNING_DIR / "pretrained" / "walk_policy.pkl"
    policy, config = load_policy(src)                                          # 목표 속도 0.3으로 학습한 정책
    cmds = {"vx": (0.2, 0.4), "vy": (0.0, 0.0), "wz": (0.0, 0.0), "resample": 5.0}
    wide = add_command_inputs(brax_model.load_params(str(src)), 13 + 3 * 6, 0, False, cmds)
    brax_model.save_params(str(tmp_path / "params.pkl"), wide)
    (tmp_path / "config.json").write_text(json.dumps({**config, "command_obs": True, "commands": cmds}))
    cmd_policy, cmd_config = load_policy(tmp_path / "params.pkl")
    old, new = make_runner(config), make_runner(cmd_config)
    new.command[:] = (0.3, 0.0, 0.0)
    o_old, o_new = old.reset(), new.reset()
    for _ in range(50):
        a_old, a_new = policy(o_old), cmd_policy(o_new)
        np.testing.assert_allclose(a_new, a_old, atol=1e-5)
        o_old, o_new = old.step(a_old), new.step(a_new)

    v, w = command_tracking(run_episode(make_runner(config), policy))
    assert abs(v[0] - 0.3) < 0.05 and abs(v[1]) < 0.05 and abs(w) < 0.1      # 목표 0.3 m/s로 곧게 걷는 정책


def test_close_previous_watch_only_closes_live_viewer(tmp_path):
    """--watch 새 학습이 지난 학습 화면만 닫음: pid 파일의 프로세스가 play_walk.py --live일 때만 끄고, 다른 프로그램은 그대로."""
    import subprocess
    import time

    from train_walk_gpu import close_previous_watch

    pid_file = tmp_path / ".watch_window.pid"
    assert not close_previous_watch(pid_file)                       # 파일 없음
    sleeper = [sys.executable, "-c", "import time; time.sleep(30)"]
    other = subprocess.Popen(sleeper + ["other.py"])                # 다른 프로그램: 그대로 둠
    viewer = subprocess.Popen(sleeper + ["play_walk.py", "--live"])  # 학습 화면처럼 보이는 프로세스
    try:
        time.sleep(0.2)
        pid_file.write_text(str(other.pid))
        assert not close_previous_watch(pid_file) and other.poll() is None
        pid_file.write_text(str(viewer.pid))
        assert close_previous_watch(pid_file)
        assert viewer.wait(timeout=5) != 0
    finally:
        other.kill(), viewer.kill()


def test_gait_report_handles_joints_left_out_of_limp_score(tmp_path):
    """절뚝임 점수에서 뺀 관절(거의 안 움직임)이 있어도 걸음 그래프·출력이 동작 (2026-10-11 자동 평가가 여기서 멈춘 적 있음)."""
    from analyze_gait import print_report, save_plots
    from walk_tools import gait_metrics, load_policy, make_runner, run_episode

    policy, config = load_policy(LEARNING_DIR / "pretrained" / "walk_policy.pkl")
    runner = make_runner(config)
    rec = run_episode(runner, policy)
    m = gait_metrics(rec, runner.info)
    m["symmetry_index_pct"].pop("knee_range_deg")      # 오리의 hip_yaw처럼 빠진 관절이 있는 상황
    save_plots(rec, m, runner.info, "test", tmp_path / "gait.png")
    print_report(m)
    assert (tmp_path / "gait.png").exists()


def test_live_dashboard_reads_tensorboard_log_and_draws(tmp_path):
    """학습 화면(learning/live_dashboard.py): TensorBoard 기록을 직접 읽고, 두 화면 모드에서 그래프·글자를 만든다."""
    import mujoco
    from tensorboardX import SummaryWriter

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
    dash = Dashboard(walk_mjx.SPECS["simple_biped"], {"target_speed": 0.3, "steps": 3_000_000, "saved_step": 3_000_000},
                     tmp_path)
    dash.start_episode()
    for k in range(10):
        dash.on_step(viewer, {"t": (k + 1) * 0.02, "vx": 0.3, "distance": 0.0, "foot_z": np.array([0.02, 0.08])})
    titles = [fig.title for _, fig in viewer.figs]
    assert len(titles) == 5 and titles[0] == "Episode reward (eval)" and titles[4].startswith("Feet:")
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
        dash.on_step(viewer, {"t": (k + 1) * 0.02, "vx": 0.3, "distance": 0.0, "foot_z": np.array([0.02, 0.08])})
    assert len(viewer.figs) == 1 and viewer.figs[0][1].title.startswith("Feet:")


def test_gait_metrics_merge_taps_and_measure_asymmetry():
    """learning/walk_tools.py: 살짝 튕긴 착지는 걸음으로 세지 않고, 좌우 다리 차이는 반 박자 어긋나게 비교한다."""
    from walk_tools import leg_asymmetry_deg, merge_taps

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
    from walk_tools import gait_metrics, load_policy, make_runner, run_episode

    policy, config = load_policy(LEARNING_DIR / "pretrained" / "walk_policy.pkl")
    runner = make_runner(config)
    m = gait_metrics(run_episode(runner, policy), runner.info)
    assert not m["fell"]
    assert m["speed_mps"] == pytest.approx(0.30, abs=0.03)
    assert m["both_air_pct"] < 5
    assert m["limp_score_pct"] < 5          # 측정값 2.1 %
    assert abs(m["yaw_deg"]) < 10


def test_terrain_env_measures_height_from_local_ground():
    """지형 위 학습 환경: GPU 쪽 땅 높이 계산(ground)이 terrain.height_at과 같고, 무작위 출발 위치에서
    몸통이 그 자리 땅 위 기준 높이에 놓인다 (관측의 높이 오차가 작다)."""
    from biped_sim.terrain import make_training_terrain

    terrain = make_training_terrain(seed=0, half_size=2.0, resolution=0.08)
    env = walk_mjx.BipedWalkMjxEnv("simple_biped", terrain=terrain, spawn_area=((-1.0, 1.0), (-1.0, 1.0)))
    xy = np.random.default_rng(0).uniform(-1.9, 1.9, (50, 2))
    np.testing.assert_allclose(np.asarray(env.ground(jnp.asarray(xy, jnp.float32))),
                               terrain.height_at(xy[:, 0], xy[:, 1]), atol=1e-5)
    for seed in range(3):
        state = jax.jit(env.reset)(jax.random.PRNGKey(seed))
        base_xy = np.asarray(state.pipeline_state.xpos[env.base_body, :2])
        assert np.all(np.abs(base_xy) <= 1.0 + 1e-6)                     # 출발 구역 안
        assert abs(float(state.obs[9])) < 0.04                             # 높이 오차 (그 자리 땅 기준)
