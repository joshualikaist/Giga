#!/usr/bin/env python3
"""t05 — 제일 쉬운 URDF 제어 실험: 공중에 매단 2족 로봇의 관절 PD 궤적 추종

왜 "매달기(fixed base)"부터?
    바닥에 세우면 넘어짐·접촉·균형 문제가 한꺼번에 섞입니다.
    몸통을 공중에 고정하면 "관절 하나하나가 명령대로 움직이는가"만 따로 검증할 수 있습니다.
    실제 로봇도 처음엔 거치대에 매달아 놓고 관절 제어부터 확인합니다.

학습 목표
    1. 매 스텝: 상태 읽기(q, dq) → PD 토크 계산 → data.ctrl 쓰기 → mj_step  의 제어 루프를 이해한다.
    2. Kp, Kd를 바꾸며 추종 오차·진동·토크 포화를 관찰한다.
    3. 속도 피드포워드(dq_des)와 중력 보상(MuJoCo의 qfrc_bias)의 효과를 비교한다.

실행
    python tutorials/t05_biped_hanging_pd.py                     # 뷰어
    python tutorials/t05_biped_hanging_pd.py --headless          # 오차 리포트 + 그래프 저장
    python tutorials/t05_biped_hanging_pd.py --headless --kp-scale 0.2       # 약한 게인
    python tutorials/t05_biped_hanging_pd.py --headless --gravity-comp       # 중력 보상 추가
    python tutorials/t05_biped_hanging_pd.py --headless --compare            # 여러 설정 한 번에 비교

결과 그래프: output/t05_tracking.png
"""
import argparse

import mujoco
import numpy as np

from biped_sim import (SIMPLE_BIPED, JointPDController, RobotInterface, SimConfig, add_common_args,
                       build_robot_model, paths, save_snapshot, simulate, track_body_camera)

# 목표 궤적: home 자세를 중심으로 사인파. 왼발/오른발은 반대 위상(걷는 것처럼 번갈아)
AMPLITUDE = {"hip_pitch": 0.4, "knee": 0.4, "ankle_pitch": 0.2}  # [rad]
PHASE_OFFSET = {"hip_pitch": 0.0, "knee": np.pi / 2, "ankle_pitch": 0.0}
LEG_PHASE = {"left": 0.0, "right": np.pi}
TRANSIENT = 1.0  # [s] 처음 1초는 초기 과도응답이므로 오차 통계에서 제외


class SineTrajectory:
    """관절별 q_des(t) = q_home + A·sin(ωt + φ),  dq_des(t) = A·ω·cos(ωt + φ)"""

    def __init__(self, robot: RobotInterface, freq_hz: float):
        self.omega = 2 * np.pi * freq_hz
        self.center = robot.to_joint_vector(SIMPLE_BIPED.home_pose)
        self.amp = np.zeros(robot.num_joints)
        self.phase = np.zeros(robot.num_joints)
        for i, name in enumerate(robot.joint_names):  # 예: "left_knee" → leg="left", joint="knee"
            leg, joint = name.split("_", 1)
            self.amp[i] = AMPLITUDE[joint]
            self.phase[i] = LEG_PHASE[leg] + PHASE_OFFSET[joint]

    def __call__(self, t: float):
        arg = self.omega * t + self.phase
        return self.center + self.amp * np.sin(arg), self.amp * self.omega * np.cos(arg)


def run_experiment(model, kp_scale=1.0, kd_scale=1.0, velocity_ff=True, gravity_comp=False,
                   freq=0.5, duration=6.0, headless=True):
    """한 번의 추종 실험을 돌리고 로그를 돌려준다."""
    data = mujoco.MjData(model)
    robot = RobotInterface(model, data)
    traj = SineTrajectory(robot, freq)
    pd = JointPDController(kp=robot.to_joint_vector(SIMPLE_BIPED.kp) * kp_scale,
                           kd=robot.to_joint_vector(SIMPLE_BIPED.kd) * kd_scale,
                           torque_limits=robot.torque_limits)

    q0, _ = traj(0.0)
    robot.reset(joint_pos=q0)  # 궤적의 시작점에서 출발 (큰 초기 오차로 튀는 것 방지)
    log = {"t": [], "q": [], "q_des": [], "tau": []}

    def control_step(m, d):
        q, dq = robot.joint_positions(), robot.joint_velocities()
        q_des, dq_des = traj(d.time)
        if not velocity_ff:
            dq_des = np.zeros_like(dq_des)  # D항이 "멈춰라"라고 해서 움직임을 방해하게 됨
        # 중력 보상: MuJoCo가 매 스텝 계산하는 바이어스 힘(중력+코리올리)을 그대로 더해 줌.
        # (직전 스텝에서 계산된 값이라 2 ms 늦지만 충분히 정확)
        tau_ff = d.qfrc_bias[robot.qvel_idx] if gravity_comp else 0.0
        tau = pd.compute(q, dq, q_des, dq_des, tau_ff)
        robot.set_joint_torques(tau)
        log["t"].append(d.time)
        log["q"].append(q)
        log["q_des"].append(q_des)
        log["tau"].append(tau)

    simulate(model, data, control_step, duration=duration, headless=headless,
             viewer_setup=track_body_camera(robot.base_body_id, distance=2.2, azimuth=90, elevation=-5))
    log = {k: np.array(v) for k, v in log.items()}
    log["joint_names"] = robot.joint_names
    log["torque_limits"] = robot.torque_limits
    return log, data


def summarize(log, label=""):
    steady = log["t"] >= TRANSIENT
    err = log["q_des"][steady] - log["q"][steady]
    rms_deg = np.degrees(np.sqrt(np.mean(err**2, axis=0)))
    max_deg = np.degrees(np.max(np.abs(err), axis=0))
    tau_max = np.max(np.abs(log["tau"]), axis=0)
    limit = log["torque_limits"][:, 1]
    saturated = np.mean(np.isclose(np.abs(log["tau"]), limit, rtol=1e-6), axis=0) * 100
    if label:
        print(f"\n[{label}]")
    print(f"{'joint':18s} {'RMS err[deg]':>12} {'max err[deg]':>12} {'max|τ|[N·m]':>12} {'포화[%]':>8}")
    for i, name in enumerate(log["joint_names"]):
        print(f"{name:18s} {rms_deg[i]:12.3f} {max_deg[i]:12.3f} {tau_max[i]:12.2f} {saturated[i]:8.1f}")
    return rms_deg


def save_plot(log, path):
    import matplotlib
    matplotlib.use("Agg")  # 화면 없이 파일로만 그림
    # (그래프 글자는 영어로: matplotlib 기본 폰트에는 한글 글꼴이 없어 네모로 깨짐)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
    for ax, i in zip(axes[:3], range(3)):  # 왼다리 관절 3개
        name = log["joint_names"][i]
        ax.plot(log["t"], np.degrees(log["q_des"][:, i]), "--", label="q_des (target)")
        ax.plot(log["t"], np.degrees(log["q"][:, i]), label="q (actual)")
        ax.set_ylabel(f"{name}\n[deg]")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    for i in range(3):
        axes[3].plot(log["t"], log["tau"][:, i], label=log["joint_names"][i])
    axes[3].set_ylabel("torque [N·m]")
    axes[3].set_xlabel("time [s]")
    axes[3].grid(alpha=0.3)
    axes[3].legend(loc="upper right", fontsize=8)
    fig.suptitle("t05: hanging biped — joint PD tracking (left leg)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser, default_duration=None)
    parser.add_argument("--kp-scale", type=float, default=1.0, help="Kp 배율 (robot_configs 값 × 배율)")
    parser.add_argument("--kd-scale", type=float, default=1.0, help="Kd 배율")
    parser.add_argument("--freq", type=float, default=0.5, help="궤적 주파수 [Hz]")
    parser.add_argument("--no-velocity-ff", action="store_true", help="dq_des=0 으로 둠")
    parser.add_argument("--gravity-comp", action="store_true", help="중력 보상(qfrc_bias) 추가")
    parser.add_argument("--compare", action="store_true", help="여러 설정을 비교 (헤드리스)")
    args = parser.parse_args()

    # 몸통을 z=1.0 m에 고정. 다리 길이(0.63 m)보다 높으므로 발이 바닥에 닿지 않음
    model, _ = build_robot_model(SIMPLE_BIPED.urdf, SimConfig(fixed_base=True, base_pos=(0, 0, 1.0)))
    print(f"모델: fixed base, nq={model.nq}, nu={model.nu}, timestep={model.opt.timestep}s")
    print(f"궤적: home 자세 ± 진폭 {AMPLITUDE} rad, {args.freq} Hz, 왼/오른다리 반대 위상")

    if args.compare:
        duration = args.duration or 6.0
        cases = [
            ("① 기본 (Kp×1, Kd×1, 속도 피드포워드)", {}),
            ("② 약한 게인 Kp×0.2", {"kp_scale": 0.2}),
            ("③ 강한 게인 Kp×3", {"kp_scale": 3.0}),
            ("④ 속도 피드포워드 없음 (dq_des=0)", {"velocity_ff": False}),
            ("⑤ 기본 + 중력 보상", {"gravity_comp": True}),
            ("⑥ 과도한 Kd×8", {"kd_scale": 8.0}),
        ]
        # 한글은 터미널에서 2칸을 차지해 표가 어긋나므로 숫자를 앞에, 설명을 뒤에 둠
        print(f"\n{'RMS오차[deg]':>12} {'포화[%]':>8} {'스텝당 최대 토크변화[N·m]':>24}   설정")
        for label, kw in cases:
            log, _ = run_experiment(model, freq=args.freq, duration=duration, **kw)
            steady = log["t"] >= TRANSIENT
            rms = np.degrees(np.sqrt(np.mean((log["q_des"][steady] - log["q"][steady]) ** 2)))
            sat = np.mean(np.isclose(np.abs(log["tau"]), log["torque_limits"][:, 1])) * 100
            jump = np.abs(np.diff(log["tau"], axis=0)).max()
            print(f"{rms:12.3f} {sat:8.1f} {jump:24.2f}   {label}")
        print("""
관찰 포인트
  ①→②→③ Kp를 키울수록 오차가 줄어듭니다. 단, 이 시뮬레이션엔 센서 노이즈·통신 지연이 없어서
         고게인이 공짜처럼 보일 뿐, 실제 로봇에선 진동·발열·충격에 대한 위험이 커집니다.
  ④     목표 속도를 0으로 주면 D항이 '멈춰라'고 움직임을 방해해 오차가 커집니다.
  ⑤     중력 보상(MuJoCo의 qfrc_bias)은 게인을 올리지 않고(=관절을 부드럽게 유지한 채)
         오차를 절반 이하로 줄입니다. 모델 기반 피드포워드의 힘입니다.
  ⑥     Kd가 너무 크면 오히려 오차가 커지고 토크가 매 스텝 +한계↔−한계를 오가는 '채터링'이
         생깁니다. 이산 시간 PD에서 Kd·dt/I(I = 관절에서 본 관성)가 커지면 수치적으로 불안정해지기
         때문입니다. → timestep을 줄이거나, armature를 키우거나, Kd를 줄여야 합니다.""")
        return

    duration = args.duration if args.duration is not None else (6.0 if args.headless else None)
    log, data = run_experiment(model, args.kp_scale, args.kd_scale, not args.no_velocity_ff,
                                      args.gravity_comp, args.freq, duration, args.headless)
    if args.headless:
        print(f"\n{duration:.1f}초 실행, 처음 {TRANSIENT}초 제외한 추종 성능:")
        summarize(log)
        path = save_plot(log, paths.output_path("t05_tracking.png"))
        print(f"\n그래프 저장: {path.relative_to(paths.REPO_ROOT)}")
    if args.snapshot:
        path = save_snapshot(model, data, paths.output_path("t05_hanging.png"),
                             lookat=(0, 0, 0.7), distance=2.0, azimuth=90, elevation=-5)
        print(f"스냅샷 저장: {path.relative_to(paths.REPO_ROOT)}")


if __name__ == "__main__":
    main()
