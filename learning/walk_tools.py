"""보행 정책 공용 도구 — 정책 불러오기, 일반 MuJoCo/MJX로 재생, 한 에피소드 기록, 걸음 수치 계산

play_walk.py(화면), analyze_gait.py(영상·그래프 분석), 테스트가 함께 쓴다. GPU 학습 환경(.venv-mjx) 전용.

    policy, config = load_policy(Path("learning/pretrained/walk_policy.pkl"))
    runner = make_runner(config)                        # 학습 설정의 로봇(simple_biped / open_duck_mini)
    rec = run_episode(runner, policy)                   # 10초 걷기, 제어 스텝(0.02 s)마다 상태 기록
    m = gait_metrics(rec, runner.info)                  # 속도, 발 접촉 비율, 좌우 대칭(절뚝임 점수) ...
    print("\\n".join(summary_lines(m)))
"""
from __future__ import annotations

import functools
import json
import os
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")   # 정책 계산은 아주 작아서 CPU로 충분 (학습 중인 GPU와 겹치지 않게)

from gpu_env import ensure_gpu_env  # noqa: E402

ensure_gpu_env()   # ROS용 .venv에서 실행했으면 .venv-mjx로 다시 실행 (JAX·Brax가 거기에만 있음)

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from biped_sim import paths  # noqa: E402
from biped_sim.envs import walk_mjx  # noqa: E402  (Brax보다 먼저 import: MJX의 안내 문구를 숨김)

LATEST = paths.OUTPUT_DIR / "learning" / "walk_latest.pkl"   # train_walk_gpu.py가 평가마다 갱신
EPISODE_SECONDS = 10.0
STEADY_FROM = 2.0        # [s] 걸음 수치는 출발 구간을 빼고 계산
TAP_HEIGHT = 0.015       # [m] 발을 이보다 낮게 들었다 다시 닿으면 걸음이 아니라 '튕김'(tap)으로 봄


# ---------------------------------------------------------------------------- 정책
def get_spec(config: dict) -> walk_mjx.WalkSpec:
    """학습 설정의 로봇 보행 과제 (예전 학습은 robot 항목이 없음 = simple_biped, servo 항목이 없음 = ideal,
    home_knee 항목이 없음 = 로봇 설정의 서 있는 자세)."""
    spec = walk_mjx.with_servo(walk_mjx.SPECS[config.get("robot", "simple_biped")], config.get("servo", "ideal"))
    return spec if config.get("home_knee") is None else walk_mjx.with_home_knee(spec, config["home_knee"])


@functools.lru_cache(maxsize=4)
def _policy_apply(obs_size: int, action_size: int, policy_hidden: tuple, value_hidden: tuple):
    """신경망 구조별로 한 번만 컴파일되는 함수 apply(params, obs) → 행동.
    가중치(params)를 인자로 받으므로, 정책을 바꿔 끼워도(학습 화면의 갱신, 체크포인트 비교) 다시 컴파일하지 않는다."""
    import jax
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks

    networks = ppo_networks.make_ppo_networks(
        obs_size, action_size,
        preprocess_observations_fn=running_statistics.normalize,   # 학습 때 normalize_observations=True
        policy_hidden_layer_sizes=policy_hidden, value_hidden_layer_sizes=value_hidden)
    make_policy = ppo_networks.make_inference_fn(networks)
    key = jax.random.PRNGKey(0)   # deterministic=True라 쓰이지 않지만 인자로 필요
    return jax.jit(lambda params, obs: make_policy(params, deterministic=True)(obs, key)[0])


def load_policy(params_path: Path):
    """Brax PPO 정책 파일 → (관측 → 행동 함수, 학습 설정 dict). 설정 파일은 정책 파일 위치에서 찾는다:
    <실행>/params.pkl → <실행>/config.json,  <실행>/checkpoints/step_<스텝>.pkl → <실행>/config.json,
    그 밖의 이름.pkl(walk_latest, pretrained/walk_policy) → 이름.json"""
    from brax.io import model as brax_model

    is_checkpoint = params_path.parent.name == "checkpoints"
    if params_path.name == "params.pkl":
        config_path = params_path.with_name("config.json")
    elif is_checkpoint:
        config_path = params_path.parent.parent / "config.json"
    else:
        config_path = params_path.with_suffix(".json")
    config = json.loads(config_path.read_text())
    if is_checkpoint:   # 설정 파일의 saved_step은 마지막 저장 기준 → 파일 이름의 스텝으로
        config["saved_step"] = int(params_path.stem.split("_")[-1])
    spec = get_spec(config)
    apply = _policy_apply(spec.obs_size, spec.n, tuple(config["policy_hidden"]), tuple(config["value_hidden"]))
    params = brax_model.load_params(str(params_path))
    return (lambda obs: np.asarray(apply(params, obs))), config


def find_run_dir(params_path: Path, config: dict) -> Path | None:
    """정책 파일에 해당하는 학습 실행 폴더 (그래프용 TensorBoard 기록이 있는 곳). 없으면 None."""
    if params_path.name == "params.pkl":
        return params_path.parent
    if params_path.parent.name == "checkpoints":
        return params_path.parent.parent
    if config.get("run_dir"):
        return Path(config["run_dir"])
    return None


# ---------------------------------------------------------------------------- 재생기
class MujocoRunner:
    """일반 MuJoCo(C)로 학습 환경과 똑같은 규칙(관측·PD·제어 주기)을 재현. ROS2 sim_node와 같은 시뮬레이터.
    관측 계산이 학습 환경(walk_mjx._observation)과 같은지는 tests/test_walk_mjx.py가 확인한다."""

    def __init__(self, spec: walk_mjx.WalkSpec | str = "simple_biped", target_speed: float | None = None,
                 terrain=None):
        """terrain: biped_sim.terrain.Terrain (None = 평지). 몸통 높이·발 접촉은 그 자리 땅을 기준으로 잰다."""
        self.W = walk_mjx
        self.spec = walk_mjx.SPECS[spec] if isinstance(spec, str) else spec
        self.terrain = terrain
        self.model, self.info = walk_mjx.build_mjx_model(self.spec, terrain)
        self.data = mujoco.MjData(self.model)
        self.target_speed = self.spec.target_speed if target_speed is None else target_speed
        self.n_substeps = round(walk_mjx.CONTROL_DT / walk_mjx.PHYSICS_DT)
        self.on_substep = None   # 물리 스텝 직전(토크를 정한 뒤)마다 부를 함수 f(data) (feasibility.py가 토크 최댓값을 놓치지 않게)

    def reset(self) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.info["home_qpos"]
        mujoco.mj_forward(self.model, self.data)
        self.t = 0.0
        self.last_action = np.zeros(self.spec.n)
        return self.obs()

    def step(self, action) -> np.ndarray:
        d, i = self.data, self.info
        action = np.clip(action, -1, 1)
        q_des = i["act_home"].copy()              # 정책 관절은 home + 행동, 나머지(머리 등)는 home
        q_des[i["policy_act"]] += self.spec.action_scale * action
        q_des = np.clip(q_des, i["act_range"][:, 0], i["act_range"][:, 1])
        for _ in range(self.n_substeps):   # 물리 1스텝마다 PD로 토크 계산 (모터 드라이버 역할)
            q, dq = d.qpos[i["act_qpos_idx"]], d.qvel[i["act_qvel_idx"]]
            d.ctrl[:] = np.clip(i["kp"] * (q_des - q) - i["kd"] * dq, i["tau_limit"][:, 0], i["tau_limit"][:, 1])
            if self.on_substep:
                self.on_substep(d)
            mujoco.mj_step(self.model, d)
        self.t += walk_mjx.CONTROL_DT
        self.last_action = action
        return self.obs()

    def obs(self) -> np.ndarray:
        d, i = self.data, self.info
        rot = d.xmat[i["base_body"]].reshape(3, 3)
        phase = 2 * np.pi * self.t / self.spec.gait_period
        return np.concatenate([rot.T @ [0, 0, -1.0], d.qvel[3:6], rot.T @ d.qvel[0:3],
                               [d.xpos[i["base_body"], 2] - self.ground(*d.xpos[i["base_body"], :2])
                                - self.spec.nominal_height],
                               d.qpos[i["qpos_idx"]] - i["q_home"], 0.1 * d.qvel[i["qvel_idx"]],
                               self.last_action, [self.target_speed], [np.sin(phase), np.cos(phase)]]
                              ).astype(np.float32)


    def ground(self, x, y):
        """(x, y)의 땅 높이 [m]. 평지면 0."""
        return 0.0 if self.terrain is None else self.terrain.height_at(x, y)


class MjxRunner(MujocoRunner):
    """학습 때와 똑같은 MJX 환경으로 재생 (화면 표시·기록용으로 매 스텝 MuJoCo 데이터에 복사)."""

    def __init__(self, spec: walk_mjx.WalkSpec | str = "simple_biped", target_speed: float | None = None,
                 terrain=None):
        super().__init__(spec, target_speed, terrain)
        import jax
        self.jax = jax
        self.env = walk_mjx.BipedWalkMjxEnv(self.spec, target_speed=self.target_speed, terrain=terrain)
        self._reset, self._step = jax.jit(self.env.reset), jax.jit(self.env.step)

    def reset(self) -> np.ndarray:
        self.s = self._reset(self.jax.random.PRNGKey(0))
        self._sync()
        return np.asarray(self.s.obs)

    def step(self, action) -> np.ndarray:
        self.s = self._step(self.s, action)
        self._sync()
        return np.asarray(self.s.obs)

    def _sync(self):
        from mujoco import mjx
        mjx.get_data_into(self.data, self.model, self.s.pipeline_state)


def make_runner(config: dict, backend: str = "mujoco", terrain=None) -> MujocoRunner:
    """학습 설정(config.json)에 맞는 재생기. backend: mujoco(일반 MuJoCo) / mjx(학습과 같은 시뮬레이터)."""
    return (MjxRunner if backend == "mjx" else MujocoRunner)(get_spec(config), config.get("target_speed"), terrain)


# ---------------------------------------------------------------------------- 한 에피소드 기록
def run_episode(runner, policy, viewer=None, on_step=None) -> dict:
    """최대 10초 걷고, 제어 스텝(0.02 s)마다 상태를 기록해 돌려준다. 넘어지면 거기서 끝.

    policy  : 관측 → 행동 함수. None이면 행동 0 (= PD로 home 자세 유지. 첫 정책을 기다릴 때)
    viewer  : 주면 매 스텝 화면을 갱신하고 실제 시간 속도로 진행 (창을 닫으면 끝)
    on_step : 매 스텝 on_step(state) 호출. state = {t, vx, distance, foot_z} (학습 화면 그래프, 영상 녹화용)
    """
    W, sp, m, d, info = runner.W, runner.spec, runner.model, runner.data, runner.info
    # 다리 막대 그림용 관절 (옆에서 본 고관절·무릎·발목): 관절 축이 지나는 점(xanchor)
    leg_joints = [[m.joint(f"{side}_{j}").id for j in _sagittal_joints(info["joint_names"])]
                  for side in ("left", "right")]
    foot_geoms = [next(g for g in range(m.ngeom) if m.geom_bodyid[g] == b and m.geom_contype[g])
                  for b in info["foot_bodies"]]
    zero = np.zeros(sp.n, dtype=np.float32)

    obs = runner.reset()
    x0 = d.xpos[info["base_body"], 0]
    rec = {k: [] for k in ("t", "q", "tau", "foot", "foot_rel_z", "base", "rpy", "legs", "toe", "heel")}
    fell = False
    for k in range(round(EPISODE_SECONDS / W.CONTROL_DT)):
        t0 = time.perf_counter()
        obs = runner.step(policy(obs) if policy is not None else zero)
        rot = d.xmat[info["base_body"]].reshape(3, 3)
        t = (k + 1) * W.CONTROL_DT
        rec["t"].append(t)
        rec["q"].append(d.qpos[info["qpos_idx"]].copy())
        rec["tau"].append(d.ctrl.copy())
        rec["foot"].append(d.xpos[info["foot_bodies"]].copy())
        rec["foot_rel_z"].append([d.xpos[b, 2] - runner.ground(*d.xpos[b, :2]) for b in info["foot_bodies"]])
        rec["base"].append(d.xpos[info["base_body"]].copy())
        rec["rpy"].append([np.arctan2(rot[2, 1], rot[2, 2]), np.arcsin(-np.clip(rot[2, 0], -1, 1)),
                           np.arctan2(rot[1, 0], rot[0, 0])])
        # 고관절·무릎·발목 위치. .copy() 필수: d.xanchor[j]는 MjData 메모리를 가리키는 '창'이라 두면 마지막 값으로 바뀜
        rec["legs"].append([[d.xanchor[j].copy() for j in leg] for leg in leg_joints])
        # 발바닥 앞끝(발가락)·뒤끝(뒤꿈치): 발 상자 중심 ± 발 방향 × 반길이
        axes = [d.geom_xmat[g].reshape(3, 3)[:, 0] * m.geom_size[g, 0] for g in foot_geoms]
        rec["toe"].append([d.geom_xpos[g] + a for g, a in zip(foot_geoms, axes)])
        rec["heel"].append([d.geom_xpos[g] - a for g, a in zip(foot_geoms, axes)])
        if on_step is not None:
            on_step({"t": t, "vx": float(d.qvel[0]), "distance": float(d.xpos[info["base_body"], 0] - x0),
                     "foot_z": np.array(rec["foot_rel_z"][-1])})
        if viewer is not None:
            if not viewer.is_running():
                break
            viewer.sync()
            time.sleep(max(0.0, W.CONTROL_DT - (time.perf_counter() - t0)))   # 실제 시간 속도로
        base_height = d.xpos[info["base_body"], 2] - runner.ground(*d.xpos[info["base_body"], :2])
        if np.arccos(np.clip(rot[2, 2], -1, 1)) > W.FALL_TILT or base_height < sp.fall_height:
            fell = True
            break
    rec = {k: np.asarray(v) for k, v in rec.items()}
    rec["raw_contact"] = rec["foot_rel_z"] < sp.foot_contact_z      # 그 자리 땅 기준 발 높이로 접촉 판정
    rec["contact"], rec["taps"] = merge_taps(rec["raw_contact"], rec["foot_rel_z"], sp.foot_contact_z)
    rec["fell"] = fell
    rec["dt"] = W.CONTROL_DT
    rec["gait_period"] = sp.gait_period
    return rec


def _sagittal_joints(joint_names) -> list[str]:
    """옆에서 본 다리의 굽힘 관절 (고관절 pitch, 무릎, 발목) 이름 뒷부분. 로봇마다 발목 이름이 다름."""
    names = {n.split("_", 1)[1] for n in joint_names if n.startswith("left_")}
    return [j for j in ("hip_pitch", "knee", "ankle_pitch", "ankle") if j in names]


def merge_taps(contact, foot_z, contact_z: float = 0.035):
    """발을 살짝(TAP_HEIGHT × 로봇 크기) 들었다 다시 닿은 구간은 '계속 딛고 있음'으로 합친다 (걸음 수를 바르게 세려고).
    로봇 크기는 접촉 판정 높이로 가늠 (simple_biped 0.035 m 기준). 돌려주는 값: (합친 접촉, 발마다 튕긴 횟수)."""
    tap_height = TAP_HEIGHT * contact_z / 0.035
    contact = contact.copy()
    taps = [0, 0]
    for i in range(2):
        c = contact[:, i]
        ground = np.median(foot_z[c, i]) if c.any() else 0.0
        for a in np.flatnonzero(~c[1:] & c[:-1]) + 1:            # 발 떼는 순간
            down = np.flatnonzero(c[a:])
            if not len(down):
                break
            b = a + down[0]                                       # 다시 닿는 순간
            if foot_z[a:b, i].max() - ground < tap_height:
                c[a:b] = True
                taps[i] += 1
    return contact, taps


# ---------------------------------------------------------------------------- 걸음 수치
def si(left, right) -> float:
    """대칭 지수 [%] = |왼쪽 − 오른쪽| / 평균 × 100. 0이면 좌우 똑같음."""
    mean = 0.5 * (abs(left) + abs(right))
    return float(100 * abs(left - right) / mean) if mean > 1e-9 else 0.0


def leg_asymmetry_deg(q, half: int, left=(0, 1, 2), right=(3, 4, 5), sign=(1.0, 1.0, 1.0)) -> float:
    """좌우 다리 동작 차이 [°]: 왼다리 관절 각도 vs 반 박자(half 스텝) 전 오른다리 × 좌우 부호 (그리고 반대)의 RMS.
    TensorBoard walk/leg_asymmetry_deg, walk_mjx의 symmetry 보상과 같은 정의."""
    if len(q) <= half:
        return 0.0
    left, right, sign = list(left), list(right), np.asarray(sign)
    now, before = q[half:], q[:-half]
    diff = np.concatenate([now[:, left] - sign * before[:, right], now[:, right] - sign * before[:, left]], axis=1)
    return float(np.degrees(np.sqrt(np.mean(np.square(diff)))))


def gait_metrics(rec: dict, info: dict) -> dict:
    """run_episode 기록 → 걸음 수치. 좌우 비교와 절뚝임 점수(대칭 지수 평균) 포함 (analyze_gait.py 설명 참고).
    info: 재생기의 runner.info (정책 관절 이름, 좌우 짝·부호)"""
    joint_names = info["joint_names"]
    t, c, foot, dt = rec["t"], rec["contact"], rec["foot"], rec["dt"]
    steady = t >= min(STEADY_FROM, t[-1] / 2)       # 일찍 넘어진 에피소드도 계산되도록
    per_side, touchdowns = {}, {}
    for i, side in enumerate(("left", "right")):
        td = np.flatnonzero(c[1:, i] & ~c[:-1, i]) + 1      # 착지 순간
        lo = np.flatnonzero(~c[1:, i] & c[:-1, i]) + 1      # 발 떼는 순간
        td, lo = td[steady[td]], lo[steady[lo]]
        touchdowns[side] = td
        stance = [(lo[lo > a][0] - a) * dt for a in td if np.any(lo > a)]
        swing = [(td[td > a][0] - a) * dt for a in lo if np.any(td > a)]
        step_len = [foot[a, i, 0] - foot[a, 1 - i, 0] for a in td]     # 착지할 때 반대 발보다 얼마나 앞에
        on_ground = c[:, i] & steady
        rel_z = rec["foot_rel_z"][:, i]                                 # 그 자리 땅 기준 발 높이
        ground_z = np.median(rel_z[on_ground]) if on_ground.any() else 0.0
        clearance = [rel_z[a:b].max() - ground_z for a in lo for b in td[td > a][:1]]
        per_side[side] = {
            "touchdowns": int(len(td)), "taps": int(rec["taps"][i]),
            "stance_s": float(np.mean(stance)) if stance else 0.0,
            "swing_s": float(np.mean(swing)) if swing else 0.0,
            "step_length_m": float(np.mean(step_len)) if step_len else 0.0,
            "clearance_m": float(np.mean(clearance)) if clearance else 0.0,
        }
    q, tau = rec["q"][steady], rec["tau"][steady]
    for j, name in enumerate(joint_names):
        side, joint = name.split("_", 1)
        per_side[side][f"{joint}_range_deg"] = float(np.degrees(np.percentile(q[:, j], 95) - np.percentile(q[:, j], 5)))
        per_side[side][f"{joint}_mean_deg"] = float(np.degrees(q[:, j].mean()))
        per_side[side][f"{joint}_torque_Nm"] = float(np.abs(tau[:, j]).mean())

    L, R = per_side["left"], per_side["right"]
    joints = [n.split("_", 1)[1] for n in joint_names if n.startswith("left_")]
    symmetry = {k: si(L[k], R[k]) for k in ("step_length_m", "stance_s", "swing_s", "clearance_m",
                                            *[f"{j}_range_deg" for j in joints])}
    # 박자: 왼발 착지 → 오른발 착지까지가 한 주기의 몇 %인가 (번갈아 걸으면 50 %)
    phase = []
    for a, b in zip(touchdowns["left"][:-1], touchdowns["left"][1:]):
        r = touchdowns["right"][(touchdowns["right"] > a) & (touchdowns["right"] < b)]
        if len(r):
            phase.append((r[0] - a) / (b - a))
    cs, rpy, base = c[steady], np.degrees(rec["rpy"][steady]), rec["base"]
    duration = float(t[-1])
    return {
        "survived_s": duration, "fell": bool(rec["fell"]),
        "speed_mps": float((base[-1, 0] - base[0, 0]) / duration),
        "drift_y_m": float(base[-1, 1] - base[0, 1]), "yaw_deg": float(np.degrees(rec["rpy"][-1, 2])),
        "both_air_pct": float(100 * np.mean(~cs.any(1))), "single_pct": float(100 * np.mean(cs.sum(1) == 1)),
        "both_ground_pct": float(100 * np.mean(cs.all(1))),
        "left_right_phase_pct": float(100 * np.mean(phase)) if phase else None,
        "body_pitch_mean_deg": float(rpy[:, 1].mean()), "body_pitch_range_deg": float(np.ptp(rpy[:, 1])),
        "body_roll_mean_deg": float(rpy[:, 0].mean()), "body_roll_range_deg": float(np.ptp(rpy[:, 0])),
        "body_height_range_m": float(np.ptp(base[steady, 2])),
        "left": L, "right": R, "symmetry_index_pct": symmetry,
        "limp_score_pct": float(np.mean(list(symmetry.values()))),
        "leg_asymmetry_deg": leg_asymmetry_deg(q, round(0.5 * rec["gait_period"] / dt), info["mirror_left"],
                                               info["mirror_right"], info["mirror_sign"]),
    }


def summary_lines(m: dict) -> list[str]:
    """걸음 수치 요약 3줄 (play_walk.py 에피소드마다, analyze_gait.py 첫머리)."""
    L, R = m["left"], m["right"]
    phase = m["left_right_phase_pct"]
    taps = L["taps"] + R["taps"]
    return [
        f"  {m['survived_s']:4.1f} s {'넘어짐 ❌' if m['fell'] else '넘어지지 않음 ✅'} | 속도 {m['speed_mps']:+.2f} m/s | "
        f"옆으로 {m['drift_y_m']:+.2f} m | 방향 {m['yaw_deg']:+.0f}°",
        f"  발 접촉: 양발 {m['both_ground_pct']:.0f}% / 한 발 {m['single_pct']:.0f}% / 둘 다 공중 {m['both_air_pct']:.0f}%"
        f" | 걸음 수 왼발 {L['touchdowns']} 오른발 {R['touchdowns']}" + (f" (발 튕김 {taps}회)" if taps else "")
        + (f" | 왼발→오른발 착지 = 주기의 {phase:.0f}% (번갈아 걸으면 50%)" if phase is not None else ""),
        f"  절뚝임 점수 {m['limp_score_pct']:.1f} % (좌우 차이 평균, 5 % 미만 대칭) | "
        f"걸음 길이 왼 {100 * L['step_length_m']:.1f} / 오른 {100 * R['step_length_m']:.1f} cm",
    ]
