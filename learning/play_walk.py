#!/usr/bin/env python3
"""GPU로 학습한 보행 정책 재생·분석

실행 (GPU 학습 환경에서: source scripts/activate_gpu.sh)
    python learning/play_walk.py                     # 10초 걸어 보고 걸음 분석 (화면 없음)
    python learning/play_walk.py --view              # MuJoCo 화면으로 반복 재생 (넘어지거나 10초가 지나면 다시 시작)
    python learning/play_walk.py --live              # 학습 화면: 학습 중인 최신 정책을 계속 불러와 재생 + 학습 그래프
    python learning/play_walk.py --backend mjx       # 학습 때와 똑같은 GPU 시뮬레이터(MJX)로 재생
    python learning/play_walk.py --params output/learning/walk_<날짜_시각>/params.pkl   # 특정 정책

화면에서 Enter 키: [학습 현황](그래프 6개 + 가운데 로봇) ↔ [로봇 보기](로봇 크게 + 발 접촉 그래프)
    --live는 학습 현황, --view는 로봇 보기로 시작. 그래프는 learning/live_dashboard.py

기본(--backend mujoco)은 일반 MuJoCo — ROS2 sim_node와 같은 시뮬레이터 — 로 돌린다.
GPU(MJX)에서 배운 걸음이 다른 시뮬레이터에서도 통하는지(sim-to-sim) 확인하는 의미가 있다.
학습이 진행 중이어도 실행할 수 있다 (평가 때마다 최신 정책이 저장됨).
"""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # 정책 계산은 아주 작아서 CPU로 충분 (학습 중인 GPU와 겹치지 않게)

import numpy as np  # noqa: E402

from biped_sim import paths, passive_viewer, track_body_camera  # noqa: E402

LATEST = paths.OUTPUT_DIR / "learning" / "walk_latest.pkl"
EPISODE_SECONDS = 10.0


def load_policy(params_path: Path):
    import jax
    from brax.io import model as brax_model
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks

    from biped_sim.envs import walk_mjx

    config_path = params_path.with_name("config.json") if params_path.name == "params.pkl" \
        else params_path.with_suffix(".json")
    config = json.loads(config_path.read_text())
    networks = ppo_networks.make_ppo_networks(
        walk_mjx.OBS_SIZE, walk_mjx.ACTION_SIZE,
        preprocess_observations_fn=running_statistics.normalize,   # 학습 때 normalize_observations=True
        policy_hidden_layer_sizes=tuple(config["policy_hidden"]),
        value_hidden_layer_sizes=tuple(config["value_hidden"]))
    inference = ppo_networks.make_inference_fn(networks)(brax_model.load_params(str(params_path)),
                                                          deterministic=True)
    inference = jax.jit(inference)
    key = jax.random.PRNGKey(0)
    return (lambda obs: np.asarray(inference(obs, key)[0])), config


class MujocoRunner:
    """일반 MuJoCo(C)로 학습 환경과 똑같은 규칙(관측·PD·제어 주기)을 재현."""

    def __init__(self, target_speed):
        import mujoco

        from biped_sim.envs import walk_mjx as W
        self.W, self.mujoco = W, mujoco
        self.model, info = W.build_mjx_model()
        self.data = mujoco.MjData(self.model)
        self.info = info
        self.target_speed = target_speed
        self.n_substeps = round(W.CONTROL_DT / W.PHYSICS_DT)

    def reset(self):
        self.mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.info["home_qpos"]
        self.mujoco.mj_forward(self.model, self.data)
        self.t = 0.0
        self.last_action = np.zeros(6)
        return self.obs()

    def step(self, action):
        W, d, i = self.W, self.data, self.info
        action = np.clip(action, -1, 1)
        q_des = np.clip(i["q_home"] + W.ACTION_SCALE * action, i["q_range"][:, 0], i["q_range"][:, 1])
        for _ in range(self.n_substeps):
            q, dq = d.qpos[i["qpos_idx"]], d.qvel[i["qvel_idx"]]
            d.ctrl[:] = np.clip(i["kp"] * (q_des - q) - i["kd"] * dq, i["tau_limit"][:, 0], i["tau_limit"][:, 1])
            self.mujoco.mj_step(self.model, d)
        self.t += W.CONTROL_DT
        self.last_action = action
        return self.obs()

    def obs(self):
        W, d, i = self.W, self.data, self.info
        rot = d.xmat[i["base_body"]].reshape(3, 3)
        phase = 2 * np.pi * self.t / W.GAIT_PERIOD
        return np.concatenate([rot.T @ [0, 0, -1.0], d.qvel[3:6], rot.T @ d.qvel[0:3],
                               [d.xpos[i["base_body"], 2] - W.NOMINAL_HEIGHT],
                               d.qpos[i["qpos_idx"]] - i["q_home"], 0.1 * d.qvel[i["qvel_idx"]],
                               self.last_action, [self.target_speed], [np.sin(phase), np.cos(phase)]]
                              ).astype(np.float32)

    def state(self):
        i, d = self.info, self.data
        rot = d.xmat[i["base_body"]].reshape(3, 3)
        return {"pos": d.xpos[i["base_body"]].copy(), "vx": float(d.qvel[0]),
                "tilt": float(np.arccos(np.clip(rot[2, 2], -1, 1))),
                "yaw": float(np.arctan2(rot[1, 0], rot[0, 0])), "foot_z": d.xpos[i["foot_bodies"], 2].copy()}


class MjxRunner(MujocoRunner):
    """학습 때와 똑같은 MJX 환경으로 재생 (화면 표시용으로 MuJoCo 데이터에 복사)."""

    def __init__(self, target_speed):
        super().__init__(target_speed)
        import jax
        from mujoco import mjx
        self.jax, self.mjx = jax, mjx
        self.env = self.W.BipedWalkMjxEnv(target_speed=target_speed)
        self._reset, self._step = jax.jit(self.env.reset), jax.jit(self.env.step)

    def reset(self):
        self.s = self._reset(self.jax.random.PRNGKey(0))
        self._sync()
        return np.asarray(self.s.obs)

    def step(self, action):
        self.s = self._step(self.s, action)
        self._sync()
        return np.asarray(self.s.obs)

    def _sync(self):
        self.mjx.get_data_into(self.data, self.model, self.s.pipeline_state)


def run_episode(runner, policy, viewer=None, on_step=None):
    """한 에피소드(최대 10 s) 실행하고 걸음 분석 결과를 돌려준다.
    policy=None이면 행동 0 (= PD로 home 자세 유지, 첫 정책을 기다릴 때).
    on_step(viewer, 상태, 시간, 이동 거리): 화면 모드에서 제어 스텝마다 호출 (그래프·글자 갱신용)"""
    W = runner.W
    obs = runner.reset()
    start = runner.state()["pos"].copy()
    log, fell = [], False
    for k in range(round(EPISODE_SECONDS / W.CONTROL_DT)):
        t0 = time.perf_counter()
        obs = runner.step(policy(obs) if policy is not None else np.zeros(W.ACTION_SIZE, dtype=np.float32))
        s = runner.state()
        log.append(s)
        if viewer is not None:
            if not viewer.is_running():
                break
            if on_step is not None:
                on_step(viewer, s, (k + 1) * W.CONTROL_DT, s["pos"][0] - start[0])
            viewer.sync()
            time.sleep(max(0.0, W.CONTROL_DT - (time.perf_counter() - t0)))
        if s["tilt"] > W.FALL_TILT or s["pos"][2] < W.FALL_HEIGHT:
            fell = True
            break
    contact = np.array([s["foot_z"] < W.FOOT_CONTACT_Z for s in log])        # (T, 2)
    touchdowns = np.sum(contact[1:] & ~contact[:-1], axis=0)                  # 발이 다시 땅에 닿은 횟수
    duration = len(log) * W.CONTROL_DT
    end = log[-1]["pos"]
    return {
        "time": duration, "fell": fell,
        "distance_x": float(end[0] - start[0]), "drift_y": float(end[1] - start[1]),
        "speed": float(end[0] - start[0]) / max(duration, 1e-6), "yaw_deg": float(np.degrees(log[-1]["yaw"])),
        "double": float(np.mean(contact.all(1))), "single": float(np.mean(contact.sum(1) == 1)),
        "flight": float(np.mean(~contact.any(1))), "steps": touchdowns.tolist(),
    }


def report(r):
    print(f"  {r['time']:4.1f} s 동안 {'넘어짐 ❌' if r['fell'] else '넘어지지 않음 ✅'} | 앞으로 {r['distance_x']:+.2f} m "
          f"(평균 {r['speed']:+.2f} m/s), 옆으로 {r['drift_y']:+.2f} m, 방향 {r['yaw_deg']:+.0f}°")
    print(f"  발 접촉: 양발 {r['double']*100:3.0f}% / 한 발 {r['single']*100:3.0f}% / 둘 다 공중 {r['flight']*100:3.0f}% "
          f"| 발 딛은 횟수 왼발 {r['steps'][0]}회, 오른발 {r['steps'][1]}회")


def find_run_dir(params_path: Path, config: dict) -> Path | None:
    """정책 파일에 해당하는 학습 실행 폴더 (그래프용 TensorBoard 기록이 있는 곳)."""
    if params_path.name == "params.pkl":
        return params_path.parent
    if config.get("run_dir"):
        return Path(config["run_dir"])
    # 예전 학습(설정에 run_dir 없음): 가장 최근에 기록된 walk 실행 폴더
    events = sorted(params_path.parent.glob("walk*/events.out.tfevents.*"), key=lambda p: p.stat().st_mtime)
    return events[-1].parent if events else None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--params", type=Path, default=LATEST, help="정책 파일 (기본: 최근 학습)")
    parser.add_argument("--view", action="store_true", help="MuJoCo 화면으로 반복 재생")
    parser.add_argument("--live", action="store_true",
                        help="학습 화면: 정책 파일이 바뀔 때마다(학습이 저장할 때마다) 다시 불러와 계속 재생 + 학습 그래프")
    parser.add_argument("--backend", choices=["mujoco", "mjx"], default="mujoco",
                        help="mujoco: 일반 MuJoCo (ROS2 sim_node와 같은 시뮬레이터) / mjx: 학습과 같은 GPU 시뮬레이터")
    args = parser.parse_args()

    from biped_sim.envs import walk_mjx
    params_path = args.params.resolve()
    if params_path.exists():
        policy, config = load_policy(params_path)
    elif args.live:  # 학습을 막 시작함: 첫 정책이 저장될 때까지 행동 0 (PD로 제자리에 서 있음)
        policy, config = None, {"target_speed": walk_mjx.TARGET_SPEED}
        config_path = params_path.with_name("config.json")
        if params_path.name == "params.pkl" and config_path.exists():  # 학습 설정은 시작하자마자 저장됨
            config = json.loads(config_path.read_text())
        print(f"정책 파일을 기다리는 중: {params_path} (학습 시작 후 약 3~4분)")
    else:
        raise SystemExit(f"정책이 없습니다: {args.params}\n→ 먼저 python learning/train_walk_gpu.py")
    runner = (MjxRunner if args.backend == "mjx" else MujocoRunner)(config["target_speed"])
    print(f"정책: {args.params} | 목표 속도 {config['target_speed']} m/s | 시뮬레이터: {args.backend}")

    if not (args.view or args.live):
        report(run_episode(runner, policy))
        return

    from live_dashboard import Dashboard  # 같은 learning/ 폴더

    loaded_mtime = params_path.stat().st_mtime if policy is not None else None
    dash = Dashboard(walk_mjx, config, find_run_dir(params_path, config), mode="graphs" if args.live else "robot")
    print("화면 재생: 넘어지거나 10초가 지나면 처음부터 다시. Enter = 그래프 ↔ 로봇 화면 전환. "
          "종료: 창 닫기 또는 Ctrl+C")
    if args.live:
        print("실시간 모드: 학습이 새 정책을 저장하면 다음 에피소드부터 자동으로 바뀝니다.")
    try:
        # 설정 패널은 숨김 (그래프와 겹치지 않게). 창에서 Tab / Shift+Tab으로 켤 수 있음
        with passive_viewer(runner.model, runner.data, show_ui=False, key_callback=dash.key_callback) as viewer:
            with viewer.lock():
                track_body_camera(runner.info["base_body"], distance=2.5, azimuth=120, elevation=-15)(viewer)
            episode = 0
            while viewer.is_running():
                if args.live and params_path.exists() and params_path.stat().st_mtime != loaded_mtime:
                    try:  # 학습 프로세스가 파일을 바꾸는 중이면 실패할 수 있음 → 다음 에피소드에 다시 시도
                        mtime = params_path.stat().st_mtime
                        policy, config = load_policy(params_path)
                        loaded_mtime = mtime  # 성공했을 때만 기록 (실패하면 다음에 다시 읽음)
                        dash.set_policy(config, find_run_dir(params_path, config))
                        print(f"  ↻ 새 정책 불러옴 (학습 스텝 {config.get('saved_step', '?')})")
                    except Exception as e:  # noqa: BLE001
                        print(f"  (정책 다시 읽기 실패, 다음에 재시도: {type(e).__name__})")
                episode += 1
                print(f"[에피소드 {episode}]" + ("" if policy is not None else " (첫 정책 기다리는 중: 행동 0)"))
                dash.start_episode()
                result = run_episode(runner, policy, viewer, on_step=dash.on_step)
                dash.end_episode(result)
                report(result)
                time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nCtrl+C — 종료합니다.")


if __name__ == "__main__":
    main()
