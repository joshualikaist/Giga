#!/usr/bin/env python3
"""동역학적으로 어디까지 가능한가 — 로봇의 물리적 한계를 계산으로 예상하고, 학습한 걸음의 실제 값과 비교 (오리 로봇)

실행 (GPU 학습 환경: source scripts/activate_gpu.sh. 다른 터미널이면 알아서 그 환경으로 다시 실행)
    python learning/feasibility.py --params <오리 정책.pkl>                 # 예상 + 평지·언덕·장애물에서 측정
    python learning/feasibility.py --params <정책.pkl> --difficulty 1.5     # 측정할 지형의 험한 정도

1부: 예상 (모델의 질량·치수·관절 범위와 서보 사양만으로. 시뮬레이션 없이 손으로도 할 수 있는 계산)
    정적 토크   한 발로 설 때 관절마다 필요한 토크 = (그 관절보다 몸통 쪽 질량) × g × (관절 축까지 수평 거리)
    턱 높이     무릎을 굽히고 펴서 바꿀 수 있는 다리 길이 (그 자세에서 한 발로 버틸 토크가 되는지도)
    경사        미끄러짐: tan θ ≤ μ / 넘어짐: 무게중심의 연직선이 발바닥 밖으로 나가는 각도
    속도        프루드 수 Fr = v²/(g·L) (사람·동물은 Fr ≈ 0.5에서 뛰기로 바꿈) / 서보 최고 속도로 다리를 앞으로 가져올 수 있는 속도
    좌우 흔들림 선형 역진자(LIPM): 한 발로 서 있는 동안 무게중심이 디딤 발 쪽으로 쏠려야 하는 양
2부: 측정 (정책으로 10초 걸으며 물리 스텝마다: 관절 토크·속도, 발의 마찰 사용량, 무게중심)
    토크-속도 한계선 밖으로 나가는 순간이 있는지 — 두 가지 서보 모델로:
      데이터시트   τ ≤ 1.91·(1 − |ω|/4.4)  (STS3215 7.4 V: 정지 토크 19.5 kg·cm, 무부하 0.238 s/60°. 보수적)
      Open Duck    τ ≤ 3.23 − 0.56·|ω|     (Open Duck Playground가 실물을 측정해 맞춘 모델: 토크 한계 + 관절 damping)
    (토크와 속도 방향이 같을 때 = 모터가 일을 할 때 속도에 따라 줄어듦. 반대 방향(브레이크)이면 오히려 늘어남)
"""
import argparse
import itertools
import json
from pathlib import Path

import mujoco
import numpy as np
from walk_tools import gait_metrics, get_spec, load_policy, make_runner, run_episode

from biped_sim import paths
from biped_sim.terrain import make_terrain

G = 9.81
LEG = ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle")
# Feetech STS3215 7.4 V (Open Duck Mini v2 BOM). 정격 = 계속 내도 과열되지 않는 토크 (판매처마다 5~6.5 kg·cm)
SERVO = {"stall": 1.91, "rated": 0.64, "noload": 4.4, "noload_max": 5.5}
OPEN_DUCK_MODEL = {"max": 3.23, "damping": 0.56}   # Open_Duck_Playground xmls/open_duck_mini_v2.xml (class sts3215)
STEADY_S = 1.0                                      # 출발 직후(일어서며 토크가 튀는 구간)는 통계에서 뺌


def available_torque(tau, vel, model):
    """그 순간 속도에서 서보가 그 방향으로 낼 수 있는 최대 토크 크기. 전압으로 구동하는 DC 모터는
    τ ∈ [−τ_s − kω, τ_s − kω] (k = τ_s/ω_무부하): 모터가 일을 할 때(τ·ω > 0)는 빠를수록 줄고, 브레이크(τ·ω < 0)는 늘어난다."""
    if model == "datasheet":
        stall, k = SERVO["stall"], SERVO["stall"] / SERVO["noload"]
    else:
        stall, k = OPEN_DUCK_MODEL["max"], OPEN_DUCK_MODEL["damping"]
    return np.clip(stall - k * np.sign(tau) * vel, 0, None)


# ---------------------------------------------------------------------------- 1부: 예상
def fold_lift(m, d, info, delta, levels=7):
    """몸통을 고정하고 왼다리 4관절(hip_roll·hip_pitch·knee·ankle)을 home에서 ±delta 안에서 바꿔
    발 상자의 가장 낮은 모서리를 가장 높이 올릴 수 있는 높이 [m] (격자 탐색, 관절 범위 안에서만)."""
    box = m.geom("left_foot_box").id
    corners = np.array(list(itertools.product((-1, 1), repeat=3))) * m.geom_size[box]
    joints = [m.joint(f"left_{j}") for j in LEG[1:]]

    def lowest(offsets):
        d.qpos[:] = info["home_qpos"]
        for j, v in zip(joints, offsets):
            d.qpos[j.qposadr[0]] += v
        mujoco.mj_kinematics(m, d)
        return (d.geom_xpos[box] + corners @ d.geom_xmat[box].reshape(3, 3).T)[:, 2].min()

    z0, best = lowest((0,) * 4), 0.0
    for offsets in itertools.product(np.linspace(-delta, delta, levels), repeat=4):
        q = [info["home_qpos"][j.qposadr[0]] + v for j, v in zip(joints, offsets)]
        if all(j.range[0] <= x <= j.range[1] for j, x in zip(joints, q)):
            best = max(best, lowest(offsets) - z0)
    return float(best)


def static_torques(m, d, root, side="left"):
    """한 발(side)로 설 때 그 다리 관절마다 정적 토크 [N·m]. 발은 땅에 고정, 몸 나머지 전체를 그 다리가 든다.
    무게중심이 지금 자세 그대로(두 발 사이 가운데)라고 가정 → hip_roll은 가장 나쁜 경우."""
    M, com = m.body_subtreemass[root], d.subtree_com[root]
    out = {}
    for j in LEG:
        jid = m.joint(f"{side}_{j}").id
        b = m.jnt_bodyid[jid]                                  # 관절이 움직이는 바디부터 발까지 = 먼 쪽
        m_far, c_far = m.body_subtreemass[b], d.subtree_com[b]
        m_near = M - m_far
        c_near = (M * com - m_far * c_far) / m_near             # 몸통 쪽 질량의 무게중심
        out[j] = float(d.xaxis[jid] @ np.cross(c_near - d.xanchor[jid], m_near * np.array([0, 0, -G])))
    return out


def predictions(runner, gait):
    m, info = runner.model, runner.info
    d = mujoco.MjData(m)
    d.qpos[:] = info["home_qpos"]
    mujoco.mj_forward(m, d)
    root = info["base_body"]
    M, com = float(m.body_subtreemass[root]), d.subtree_com[root].copy()
    box = [m.geom(f"{f}_box").id for f in ("left_foot", "right_foot")]
    half = m.geom_size[box[0]]                                  # 발 충돌 상자 반길이 (x 앞뒤, y 좌우, z)
    feet = d.geom_xpos[box].copy()
    hip = m.joint("left_hip_pitch").id

    def leg_length():                                           # 고관절 축 ~ 발바닥(발 상자 밑면) 높이 차
        return float(d.xanchor[hip, 2] - (d.geom_xpos[box[0], 2] - half[2]))

    p = {"mass_kg": M, "com_height_m": float(com[2]), "leg_length_m": leg_length(),
         "foot_length_m": float(2 * half[0]), "foot_width_m": float(2 * half[1]),
         "stance_width_m": float(feet[0, 1] - feet[1, 1]), "static_torque_Nm": static_torques(m, d, root)}

    # 턱 높이: 무릎 k를 바꾸며 hip = ankle = −k/2 (발바닥 수평, 발이 고관절 바로 아래 — home 자세와 같은 규칙, docs/09 §4)
    sweep = []
    for k in np.linspace(0.0, m.jnt_range[m.joint("left_knee").id, 1], 32):   # 무릎 0 ~ 관절 범위 끝
        d.qpos[:] = info["home_qpos"]
        for side, s in (("left", 1.0), ("right", -1.0)):        # 오른쪽 hip_pitch는 부호 반대
            for j, v in zip(LEG, (0.0, 0.0, -k / 2, k, -k / 2)):
                d.qpos[m.joint(f"{side}_{j}").qposadr[0]] = s * v if j == "hip_pitch" else v
        mujoco.mj_forward(m, d)
        sweep.append((float(k), leg_length(), abs(static_torques(m, d, root)["knee"])))
    sweep = np.array(sweep)
    ok = sweep[sweep[:, 2] <= SERVO["stall"]]                   # 그 자세로 한 발 버티기가 데이터시트 정지 토크 이내
    p["knee_sweep"] = sweep.tolist()
    p["leg_length_range_m"] = (float(ok[:, 1].min()), float(sweep[:, 1].max()))
    p["swing_lift_max_m"] = p["leg_length_m"] - p["leg_length_range_m"][0]          # 디딤 다리는 home 그대로, 흔드는 발만 접기
    p["step_up_max_m"] = p["leg_length_range_m"][1] - p["leg_length_range_m"][0]    # 디딤 다리도 끝까지 펴면
    p["knee_torque_deep_crouch_Nm"] = float(ok[-1, 2])
    # 흔드는 동안 발을 얼마나 올릴 수 있나: 관절이 갔다 오는 시간 T 동안 최고 속도 ω이면 (사인 모양) 각도 폭 ≤ ω·T/π.
    # 정책은 home ± action_scale 안에서만 움직일 수 있다 (학습 환경의 행동 범위)
    t_sw = gait["single_support_s"]
    p["fold_lift"] = {"action_scale_rad": runner.spec.action_scale,
                      "action_scale_m": fold_lift(m, d, info, runner.spec.action_scale),
                      "swing_s": {f"{t:.2f}": [float(SERVO["noload"] * t / np.pi), fold_lift(m, d, info, SERVO["noload"] * t / np.pi)]
                                  for t in (t_sw, 0.25)}}

    # 경사 (위의 탐색이 자세를 바꿔 놓았으므로 다시 home 자세에서)
    d.qpos[:] = info["home_qpos"]
    mujoco.mj_forward(m, d)
    mu = float(max(m.geom_friction[box[0], 0], m.geom_friction[m.geom("floor").id, 0]))
    toe = feet[0, 0] + half[0] - com[0]                         # 무게중심 연직선 → 발끝까지 (앞)
    heel = com[0] - (feet[0, 0] - half[0])                      # → 뒤꿈치까지
    outer = (feet[0, 1] + half[1]) - com[1]                     # 두 발로 설 때 → 바깥 모서리까지 (옆)
    ankle_h = d.xanchor[m.joint("left_ankle").id][2] - (feet[0, 2] - half[2])
    p["slope"] = {
        "friction_mu": mu, "slip_deg": float(np.degrees(np.arctan(mu))),
        # 자세를 그대로 두고 통째로 기울면: 무게중심 연직선이 발 밖으로 나가는 각도 = atan(거리 / 무게중심 높이)
        "tip_rigid_uphill_deg": float(np.degrees(np.arctan(heel / com[2]))),
        "tip_rigid_downhill_deg": float(np.degrees(np.arctan(toe / com[2]))),
        "tip_rigid_sideways_deg": float(np.degrees(np.arctan(outer / com[2]))),
        # 발목만 꺾어 몸을 똑바로 세우면: 발목 바로 아래에서 압력 중심이 발목 높이 × tan θ만 밀림
        "tip_upright_deg": float(np.degrees(np.arctan(min(heel, toe) / ankle_h))),
    }

    # 속도. 고관절 각 = 다리가 연직에서 기운 각. 디딤 동안 ±A로 일정하게 쓸고(A = v·T_디딤 / 2L),
    # 흔듦 동안 사인 모양으로 되돌아오면 최대 각속도 = π·A / T_흔듦 = (v/L)·(π/2)·β/(1−β)  (β = 디딤 비율)
    L, v = p["leg_length_m"], runner.target_speed
    ratio = lambda beta: np.pi / 2 * beta / (1 - beta)
    p["speed"] = {"target_mps": v, "froude_now": float(v ** 2 / (G * L)), "walk_limit_mps": float(np.sqrt(0.5 * G * L)),
                  "duty": gait["duty"], "hip_speed_needed_now_radps": float(v / L * ratio(gait["duty"])),
                  "servo_limit_mps": {f"{b:.2f}": [float(w * L / ratio(b)) for w in (SERVO["noload"], SERVO["noload_max"])]
                                      for b in sorted({0.5, 0.6, round(gait["duty"], 2)})}}

    # 좌우 흔들림 (LIPM): y'' = ω²(y − p), 디딤 발 p = 너비/2. 두 발 사이 가운데에서 출발해 t초 뒤 가운데로 돌아오는
    # 대칭 궤적 y − p = −C·cosh(ω(t − t/2)) → 가장 많이 쏠린 양 = (너비/2)·(1 − 1/cosh(ω·t/2))
    w0 = np.sqrt(G / com[2])
    sway = lambda t: float(p["stance_width_m"] / 2 * (1 - 1 / np.cosh(w0 * t / 2)))
    t_single, t_half = gait["single_support_s"], runner.spec.gait_period / 2
    p["lipm"] = {"omega": float(w0), "time_constant_s": float(1 / w0), "single_support_s": t_single,
                 "half_period_s": t_half, "sway_m": (sway(t_single), sway(t_half))}
    return p


def gait_timing(runner, policy):
    """평지에서 한 번 걸어 보고 디딤 비율·한 발 지지 시간을 잰다 (속도·LIPM 예상에 씀)."""
    g = gait_metrics(run_episode(runner, policy), runner.info)
    stance = 0.5 * (g["left"]["stance_s"] + g["right"]["stance_s"])
    swing = 0.5 * (g["left"]["swing_s"] + g["right"]["swing_s"])
    return {"duty": stance / (stance + swing), "single_support_s": swing,    # 한 발이 공중 = 다른 발 혼자 지지
            "clearance_m": 0.5 * (g["left"]["clearance_m"] + g["right"]["clearance_m"])}


# ---------------------------------------------------------------------------- 2부: 측정
def measure(runner, policy, seconds=10.0):
    """정책으로 걸으며 물리 스텝(4 ms)마다 정책 관절 토크·속도, 발 접촉력(수직·수평)을, 제어 스텝마다 무게중심을 기록."""
    m, d, info, sp = runner.model, runner.data, runner.info, runner.spec
    act, root = info["policy_act"], info["base_body"]
    box = {m.geom(f"{f}_box").id for f in ("left_foot", "right_foot")}
    mu = float(m.geom_friction[min(box), 0])
    taus, vels, ratios, coms, rolls, force = [], [], [], [], [], np.zeros(6)

    damping = m.dof_damping[info["qvel_idx"]]

    def record(d):   # 물리 스텝 직전: 모터가 내는 토크 = 명령 토크(토크 한계로 자른 PD) − 관절 damping(역기전력 모델) × 속도
        taus.append(d.ctrl[act] - damping * d.qvel[info["qvel_idx"]])
        vels.append(d.qvel[info["qvel_idx"]].copy())
        for i in range(d.ncon):                                   # 발-바닥 접촉: 쓰는 마찰 = 수평력 / 수직력
            c = d.contact[i]
            if c.geom1 in box or c.geom2 in box:
                mujoco.mj_contactForce(m, d, i, force)
                if force[0] > 0.5:                                # 수직력 0.5 N 이상 (거의 안 닿은 순간 제외)
                    ratios.append(np.hypot(force[1], force[2]) / force[0])

    obs, fell = runner.reset(), False
    runner.on_substep = record
    for _ in range(round(seconds / runner.W.CONTROL_DT)):
        obs = runner.step(policy(obs))
        coms.append(d.subtree_com[root].copy())
        rot = d.xmat[root].reshape(3, 3)
        rolls.append(np.arctan2(rot[2, 1], rot[2, 2]))
        base_h = d.xpos[root, 2] - runner.ground(*d.xpos[root, :2])
        if np.arccos(np.clip(rot[2, 2], -1, 1)) > runner.W.FALL_TILT or base_h < sp.fall_height:
            fell = True
            break
    runner.on_substep = None

    n_sub = runner.n_substeps
    skip = round(STEADY_S / runner.W.CONTROL_DT)
    taus, vels = np.array(taus)[skip * n_sub:], np.array(vels)[skip * n_sub:]
    coms, rolls = np.array(coms)[skip:], np.array(rolls)[skip:]
    joints = {}
    for i, n in enumerate(info["joint_names"]):
        t, w = taus[:, i], vels[:, i]
        joints[n] = {"peak_torque": float(np.abs(t).max()), "rms_torque": float(np.sqrt(np.mean(t ** 2))),
                     "peak_speed": float(np.abs(w).max()),
                     "over_rated_pct": float(100 * np.mean(np.abs(t) > SERVO["rated"])),
                     "over_datasheet_pct": float(100 * np.mean(np.abs(t) > available_torque(t, w, "datasheet"))),
                     "over_open_duck_pct": float(100 * np.mean(np.abs(t) > available_torque(t, w, "open_duck") + 1e-3))}
    # 무게중심 좌우 흔들림: 걸음 한 주기 이동평균을 빼서 천천히 옆으로 흘러가는 것(drift)은 제외
    win = max(1, round(sp.gait_period / runner.W.CONTROL_DT))
    y = coms[:, 1] - np.convolve(coms[:, 1], np.ones(win) / win, mode="same")
    y = y[win:-win] if len(y) > 3 * win else y
    ratios = np.array(ratios)
    return {"fell": fell, "seconds": (len(coms) + skip) * runner.W.CONTROL_DT, "joints": joints,
            "friction_median": float(np.median(ratios)) if len(ratios) else 0.0,
            "friction_p90": float(np.percentile(ratios, 90)) if len(ratios) else 0.0,
            "slip_pct": float(100 * np.mean(ratios > 0.95 * mu)) if len(ratios) else 0.0,
            "com_sway_m": float(np.ptp(y) / 2), "roll_range_deg": float(np.degrees(np.ptp(rolls))),
            "samples": {"tau": taus, "vel": vels}}


# ---------------------------------------------------------------------------- 출력
def report(p, meas, gait, sim):
    s, sp, lp = p["slope"], p["speed"], p["lipm"]
    print(f"\n[로봇] 질량 {p['mass_kg']:.2f} kg, 무게중심 높이 {100 * p['com_height_m']:.1f} cm, 다리 길이(home) "
          f"{100 * p['leg_length_m']:.1f} cm, 발 {100 * p['foot_length_m']:.1f} × {100 * p['foot_width_m']:.1f} cm, "
          f"두 발 간격 {100 * p['stance_width_m']:.1f} cm")
    print(f"[서보] STS3215 7.4 V 데이터시트: 정지 {SERVO['stall']} N·m, 정격 {SERVO['rated']} N·m, 무부하 "
          f"{SERVO['noload']}~{SERVO['noload_max']} rad/s / Open Duck 모델: {OPEN_DUCK_MODEL['max']} N·m − "
          f"{OPEN_DUCK_MODEL['damping']}·ω / 이 시뮬레이션({sim['servo']}): {sim['max']:.2f} N·m, 관절 damping {sim['damping']:.2f}")

    print("\n1) 한 발로 설 때 정적 토크 (home 자세, 무게중심이 두 발 가운데 그대로 = 가장 나쁜 경우)")
    for j, t in p["static_torque_Nm"].items():
        print(f"   {j:9s} {abs(t):5.2f} N·m  = 데이터시트 정지 토크의 {100 * abs(t) / SERVO['stall']:3.0f} %, "
              f"정격의 {100 * abs(t) / SERVO['rated']:4.0f} %")
    print("\n2) 턱 높이")
    print(f"   다리 길이 {100 * p['leg_length_range_m'][0]:.1f} (무릎 끝까지 접음) ~ {100 * p['leg_length_range_m'][1]:.1f} cm "
          f"(완전히 폄). 끝까지 접은 자세로 한 발 버티는 무릎 토크 {p['knee_torque_deep_crouch_Nm']:.2f} N·m")
    print(f"   흔드는 발만 접어 올릴 수 있는 높이 {100 * p['swing_lift_max_m']:.1f} cm, 디딤 다리까지 펴면 최대 "
          f"{100 * p['step_up_max_m']:.1f} cm (지금 정책의 발 들기 {100 * gait['clearance_m']:.1f} cm)")
    fl = p["fold_lift"]
    print(f"   행동 범위(home ± {fl['action_scale_rad']} rad) 안에서 다리를 접어 올릴 수 있는 높이 {100 * fl['action_scale_m']:.1f} cm")
    for t, (delta, lift) in fl["swing_s"].items():
        print(f"   흔듦 {t} s 동안 서보 무부하 속도로 갔다 올 수 있는 각도 = 관절마다 {delta:.2f} rad → 발 올리기 최대 {100 * lift:.1f} cm")
    print("\n3) 경사")
    print(f"   미끄러짐: tan θ ≤ μ = {s['friction_mu']:.1f} → {s['slip_deg']:.0f}°")
    print(f"   자세 그대로 통째로 기울면 넘어지는 각도: 오르막 {s['tip_rigid_uphill_deg']:.0f}°, 내리막 "
          f"{s['tip_rigid_downhill_deg']:.0f}°, 옆 {s['tip_rigid_sideways_deg']:.0f}°")
    print(f"   발목으로 몸을 똑바로 세우면 {s['tip_upright_deg']:.0f}°, 몸을 경사 쪽으로 숙이면 넘어짐 한계 없음 → 미끄러짐({s['slip_deg']:.0f}°)이 한계")
    print("\n4) 속도")
    print(f"   지금 {sp['target_mps']} m/s = 프루드 수 {sp['froude_now']:.3f}. 걷기 한계 Fr 0.5 → {sp['walk_limit_mps']:.2f} m/s")
    print(f"   지금 필요한 고관절 최대 속도 {sp['hip_speed_needed_now_radps']:.1f} rad/s (디딤 비율 {sp['duty']:.2f})")
    for beta, (lo, hi) in sp["servo_limit_mps"].items():
        print(f"   서보 무부하 속도로 다리를 앞으로 가져올 수 있는 최고 속도 (디딤 비율 {beta}): {lo:.2f} ~ {hi:.2f} m/s")
    print("\n5) 좌우 흔들림 (선형 역진자)")
    print(f"   시정수 1/ω = {1000 * lp['time_constant_s']:.0f} ms. 한 발 지지 {lp['single_support_s']:.2f} ~ 반 주기 "
          f"{lp['half_period_s']:.2f} s 동안 무게중심이 디딤 발 쪽으로 {100 * lp['sway_m'][0]:.1f} ~ "
          f"{100 * lp['sway_m'][1]:.1f} cm 쏠려야 함")
    for label, mm in meas.items():
        print(f"\n[측정: {label}] {'넘어짐 (' + format(mm['seconds'], '.1f') + '초)' if mm['fell'] else '10초 버팀'}, "
              f"무게중심 좌우 ±{100 * mm['com_sway_m']:.1f} cm, 몸통 roll 범위 {mm['roll_range_deg']:.0f}°, "
              f"쓰는 마찰 중앙값 {mm['friction_median']:.2f} / 90 % {mm['friction_p90']:.2f}, 미끄러지는 접촉 {mm['slip_pct']:.0f} %")
        print(f"   {'관절':16s} {'최대 토크':>7s} {'RMS':>5s} {'최대 속도':>7s}  {'정격 초과':>6s}  {'한계 밖: 데이터시트':>12s} {'Open Duck':>9s}")
        for n, j in mm["joints"].items():
            print(f"   {n:16s} {j['peak_torque']:7.2f} {j['rms_torque']:6.2f} {j['peak_speed']:8.1f}   "
                  f"{j['over_rated_pct']:6.0f} %   {j['over_datasheet_pct']:12.1f} % {j['over_open_duck_pct']:9.1f} %")


def plot(meas, names, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = ("hip_roll", "hip_pitch", "knee", "ankle")
    fig, axes = plt.subplots(1, len(groups), figsize=(18, 4.6))
    w = np.linspace(0, 7, 50)
    for ax, joint in zip(axes, groups):
        idx = [i for i, n in enumerate(names) if n.endswith(joint)]
        for (label, mm), color in zip(meas.items(), ("tab:blue", "tab:orange", "tab:red")):
            tau, vel = mm["samples"]["tau"][::2, idx], mm["samples"]["vel"][::2, idx]
            motoring = tau * vel > 0                              # 모터가 일을 하는 순간만 (한계선과 비교할 수 있는 점)
            ax.scatter(np.abs(vel[motoring]), np.abs(tau[motoring]), s=3, alpha=0.3, color=color, label=label)
        ax.plot(w, available_torque(np.ones_like(w), w, "datasheet"), "k-", lw=1.5, label="STS3215 datasheet")
        ax.plot(w, available_torque(np.ones_like(w), w, "open_duck"), "k--", lw=1.5, label="Open Duck model")
        ax.axhline(SERVO["rated"], color="green", ls=":", lw=1.5, label="rated (continuous)")
        ax.set_title(f"{joint}: |torque| vs |speed| (motoring)")
        ax.set_xlabel("joint speed [rad/s]")
        ax.set_ylabel("torque [N·m]")
        ax.set_xlim(0, 6.5)
        ax.set_ylim(0, 3.4)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper right", markerscale=3)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--params", type=Path, required=True, help="보행 정책 (오리 로봇)")
    parser.add_argument("--difficulty", type=float, default=1.0, help="측정할 지형의 험한 정도")
    parser.add_argument("--seed", type=int, default=0, help="측정할 지형의 seed")
    parser.add_argument("--out", type=Path, default=None, help="결과 폴더 (기본 output/feasibility/<실행 이름>_<체크포인트>)")
    parser.add_argument("--servo", default=None, help="서보 모델을 바꿔 측정 (기본: 학습 때와 같음). ideal / open_duck")
    args = parser.parse_args()

    params = args.params.resolve()
    policy, config = load_policy(params)
    if args.servo:
        config = {**config, "servo": args.servo}
    if get_spec(config).name != "open_duck_mini":
        raise SystemExit("이 분석은 오리 로봇(open_duck_mini) 정책용입니다 (관절·발 이름이 오리 기준)")
    flat = make_runner(config)
    gait = gait_timing(flat, policy)
    p = predictions(flat, gait)
    meas = {"flat": measure(flat, policy)}
    for kind in ("hills", "obstacles"):
        rough = make_runner(config, terrain=make_terrain(kind, seed=args.seed, difficulty=args.difficulty))
        meas[f"{kind} {args.difficulty:g}"] = measure(rough, policy)
    sim = {"servo": flat.spec.servo, "max": float(flat.model.actuator_ctrlrange[0, 1]),
           "damping": float(flat.model.dof_damping[flat.info["qvel_idx"][0]])}
    report(p, meas, gait, sim)

    run = params.parent.parent.name if params.parent.name == "checkpoints" else params.parent.name
    suffix = f"_{args.servo}" if args.servo else ""
    out = (args.out or paths.OUTPUT_DIR / "feasibility" / f"{run}_{params.stem}{suffix}").resolve()
    out.mkdir(parents=True, exist_ok=True)
    plot(meas, flat.info["joint_names"], out / "torque_speed.png")
    for mm in meas.values():
        mm.pop("samples")
    (out / "report.json").write_text(json.dumps({"predictions": p, "gait": gait, "measured": meas, "servo": SERVO,
                                                 "open_duck_model": OPEN_DUCK_MODEL}, indent=2, ensure_ascii=False))
    print(f"\n저장: {out}/torque_speed.png, report.json")


if __name__ == "__main__":
    main()
