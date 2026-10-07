#!/usr/bin/env python3
"""t03 — 첫 URDF: 진자를 MuJoCo로 불러오고, 시뮬레이션이 "맞는지" 이론으로 검증하기

학습 목표
    1. MuJoCo는 URDF를 직접 읽을 수 있다 (MjModel.from_xml_path에 .urdf를 넣으면 됨).
    2. 하지만 그대로 읽으면 무엇이 사라지고 / 무엇이 "조용히" 틀어지는지 확인한다.
    3. 시뮬레이터 개발자의 기본기: 해석해가 있는 문제로 결과를 검증한다.
           물리 진자 주기  T = 2π·√(I_pivot / (m·g·d))
       검증 → 실패 → 원인 진단 → 수정 → 재검증 의 흐름을 직접 겪어 본다.
    4. 적분기(integrator)에 따라 에너지 보존 성능이 다름을 확인한다.

실행
    python tutorials/t03_urdf_pendulum.py --headless   # 검증 리포트 (먼저 이것부터!)
    python tutorials/t03_urdf_pendulum.py              # 뷰어: 수정된 모델이 흔들리는 모습
    python tutorials/t03_urdf_pendulum.py --raw        # 뷰어: URDF를 그대로 읽은 (잘못된) 모델

참고: 아무 코드 없이 뷰어로 보기만 하려면
    python -m mujoco.viewer --mjcf=models/urdf/pendulum/pendulum.urdf
"""
import argparse

import mujoco
import numpy as np

from biped_sim import SimConfig, build_robot_model, simulate
from biped_sim.builder import INTEGRATORS
from biped_sim.paths import PENDULUM_URDF

PASS_TOLERANCE_PERCENT = 0.5


def banner(title):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def body_names(model):
    return [model.body(i).name for i in range(model.nbody)]


def contact_pairs(model, data):
    return sorted({(model.body(model.geom_bodyid[c.geom1]).name,
                    model.body(model.geom_bodyid[c.geom2]).name) for c in data.contact[:data.ncon]})


def pivot_inertia_and_com(model, body_name="rod_link", axis=(0.0, 1.0, 0.0)):
    """회전축 기준 관성 I_pivot, 질량 m, 축~COM 거리 d 를 '모델에서' 직접 계산.

    주의: model.body_inertia는 '주관성 좌표계'(body_iquat로 회전된) 기준 대각값이며,
    MuJoCo가 크기순으로 재정렬하기도 합니다. 그래서 회전행렬로 바디 좌표계 텐서를 복원합니다.
    """
    b = model.body(body_name)
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, b.iquat)
    rot = rot.reshape(3, 3)
    inertia_body = rot @ np.diag(b.inertia) @ rot.T   # 바디 좌표계에서 본 COM 기준 관성 텐서
    a = np.asarray(axis)
    m = b.mass[0]
    com = b.ipos                                        # 바디 원점(=회전축) → COM 벡터
    d_perp = com - a * (com @ a)                        # 회전축에 수직인 성분
    i_pivot = a @ inertia_body @ a + m * (d_perp @ d_perp)  # 평행축 정리
    joint_id = model.body_jntadr[b.id]
    i_pivot += model.dof_armature[model.jnt_dofadr[joint_id]]  # 모터 회전자 관성(armature)도 더함
    return i_pivot, m, float(np.linalg.norm(d_perp))


def measure_period(model, data, theta0, duration=10.0):
    """theta0에서 놓고, 각도가 음→양으로 0을 지나는 시각들의 간격 평균 = 주기."""
    mujoco.mj_resetData(model, data)
    data.qpos[0] = theta0
    crossings, prev = [], data.qpos[0]
    while data.time < duration:
        mujoco.mj_step(model, data)
        cur = data.qpos[0]
        if prev < 0.0 <= cur:  # 선형 보간으로 교차 시각을 정밀하게
            crossings.append(data.time - model.opt.timestep * cur / (cur - prev))
        prev = cur
    return float(np.mean(np.diff(crossings))) if len(crossings) > 1 else float("nan")


def verify_period(label, model, data):
    """이론 주기와 비교하고 통과 여부를 돌려준다."""
    i_pivot, m, d = pivot_inertia_and_com(model)
    g = -model.opt.gravity[2]
    t_theory = 2 * np.pi * np.sqrt(i_pivot / (m * g * d))
    t_sim = measure_period(model, data, theta0=0.05)
    err = abs(t_sim - t_theory) / t_theory * 100
    ok = err < PASS_TOLERANCE_PERCENT
    print(f"[{label}] m={m:.3f} kg, d={d:.4f} m, I_pivot={i_pivot:.6f} kg·m²")
    print(f"  이론 주기 = {t_theory:.5f} s,  시뮬레이션 주기 = {t_sim:.5f} s,  오차 = {err:.3f} %"
          f"  → {'PASS ✅' if ok else 'FAIL ❌'}")
    return ok


def energy_drift(model, data, integrator, theta0=1.0, duration=10.0):
    """감쇠 0인 진자의 총에너지 변화 [J]. 완벽한 적분기라면 0."""
    model.opt.integrator = INTEGRATORS[integrator]
    model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY  # data.energy 계산 켜기
    mujoco.mj_resetData(model, data)
    data.qpos[0] = theta0
    mujoco.mj_forward(model, data)
    e0 = data.energy.sum()  # energy = [위치에너지, 운동에너지]
    e_min = e_max = e0
    while data.time < duration:
        mujoco.mj_step(model, data)
        e = data.energy.sum()
        e_min, e_max = min(e_min, e), max(e_max, e)
    return data.energy.sum() - e0, e_max - e_min


def load_raw():
    """URDF를 아무 설정 없이 그대로 읽기."""
    return mujoco.MjModel.from_xml_path(str(PENDULUM_URDF))


def load_fixed():
    """biped_sim 빌더로 읽기: 루트 링크 보존 + 부모-자식 충돌 제외 + 모터 추가.
    검증을 위해 armature(모터 관성)는 0, 바닥/IMU는 생략."""
    model, _ = build_robot_model(PENDULUM_URDF, SimConfig(
        fixed_base=True, base_pos=(0, 0, 1.0), joint_armature=0.0,
        add_floor=False, add_imu=False))
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--raw", action="store_true", help="뷰어에서 URDF를 그대로 읽은 모델을 봄")
    parser.add_argument("--theta0", type=float, default=0.8, help="뷰어 모드 초기 각도 [rad]")
    args = parser.parse_args()

    if not args.headless:
        model = load_raw() if args.raw else load_fixed()
        data = mujoco.MjData(model)
        data.qpos[0] = args.theta0
        print(f"뷰어: {'URDF 그대로 (잘못된 모델)' if args.raw else '빌더로 수정된 모델'}, "
              f"{args.theta0} rad에서 놓음. 감쇠가 0이라 계속 흔들려야 정상입니다.")
        simulate(model, data)
        return

    # ------------------------------------------------------------------ 1단계
    banner("1단계. URDF를 '그대로' 불러오기  —  MjModel.from_xml_path('pendulum.urdf')")
    raw = load_raw()
    raw_data = mujoco.MjData(raw)
    print(f"바디: {body_names(raw)}")
    print("  → URDF의 base_link가 사라졌습니다. 관절 없는 링크는 부모(여기선 world)에 '병합'됩니다.")
    print(f"관절: {[raw.joint(i).name for i in range(raw.njnt)]}, nq={raw.nq}, nv={raw.nv}")
    print(f"geom 개수: {raw.ngeom}  → URDF의 visual 2개 + collision 2개 중 collision만 남았습니다.")
    print(f"액추에이터 nu = {raw.nu}  → 토크를 줄 방법이 없습니다 (URDF에는 모터 개념이 없음).")
    print(f"integrator = {mujoco.mjtIntegrator(raw.opt.integrator).name} (MuJoCo 기본값)")

    # ------------------------------------------------------------------ 2단계
    banner("2단계. 검증: 작은 진폭(0.05 rad) 진자 주기 — 시뮬레이션 vs 이론")
    verify_period("URDF 그대로", raw, raw_data)
    raw_data.qpos[0] = 0.05
    mujoco.mj_forward(raw, raw_data)
    print("\n원인 진단: 진자가 무언가와 닿아 있는가?")
    print(f"  접촉 개수 = {raw_data.ncon},  접촉 쌍 = {contact_pairs(raw, raw_data)}")
    print("  → 막대(rod_link)가 world와 접촉 중! world에 병합된 받침대 상자의 아랫면과")
    print("    막대 윗면이 맞닿아 있어서, 흔들릴 때마다 마찰로 붙잡히고 있었습니다.")
    print("    MuJoCo는 부모-자식 바디끼리의 충돌을 자동으로 거르지만, 부모가 'world(또는 world에")
    print("    용접된 바디)'이면 거르지 않습니다 (물체가 바닥과는 충돌해야 하므로).")
    print("  교훈: 에러 메시지 없이 '그럴듯하게 틀린' 결과가 나옵니다. 검증하지 않으면 모릅니다.")

    # ------------------------------------------------------------------ 3단계
    banner("3단계. 수정: biped_sim 빌더로 불러오기  —  build_robot_model(...)")
    fixed = load_fixed()
    fixed_data = mujoco.MjData(fixed)
    print(f"바디: {body_names(fixed)}   ← base_link 보존 (fusestatic=False)")
    excluded = [(fixed.body(fixed.exclude_signature[i] >> 16).name,
                 fixed.body(fixed.exclude_signature[i] & 0xFFFF).name) for i in range(fixed.nexclude)]
    print(f"충돌 제외 쌍: {excluded}   ← 부모-자식 링크 충돌 명시적 제외")
    print(f"액추에이터: {[fixed.actuator(i).name for i in range(fixed.nu)]}, "
          f"토크 범위 {fixed.actuator_ctrlrange[0]} N·m  ← URDF <limit effort>에서 가져옴")
    fixed_data.qpos[0] = 0.05
    mujoco.mj_forward(fixed, fixed_data)
    print(f"접촉 개수 = {fixed_data.ncon}\n")
    verify_period("빌더 사용", fixed, fixed_data)
    # 큰 진폭: 작은 각 근사(sinθ≈θ)가 깨짐. 정확한 주기의 급수 근사 T0·(1 + θ²/16 + 11θ⁴/3072)
    i_pivot, m, d = pivot_inertia_and_com(fixed)
    t0 = 2 * np.pi * np.sqrt(i_pivot / (m * 9.81 * d))
    t_large_theory = t0 * (1 + 1.0**2 / 16 + 11 * 1.0**4 / 3072)
    t_large = measure_period(fixed, fixed_data, theta0=1.0)
    print(f"  큰 진폭(1.0 rad): 시뮬레이션 {t_large:.5f} s vs 급수 근사 이론 {t_large_theory:.5f} s"
          " → 진폭이 크면 주기가 길어짐 (비선형)")

    # ------------------------------------------------------------------ 4단계
    banner("4단계. 적분기 비교: 감쇠 0 진자를 1.0 rad에서 놓고 10초 후 총에너지 변화")
    e_scale = m * 9.81 * d * (1 - np.cos(1.0))
    print(f"(기준: 흔들림 에너지 m·g·d·(1−cos θ₀) = {e_scale:.4f} J)")
    print(f"{'integrator':14s} {'최종 ΔE [J]':>12s} {'ΔE / 기준':>10s} {'10초간 최대 변동폭 [J]':>22s}")
    for name in ("euler", "implicitfast", "implicit", "rk4"):
        drift, spread = energy_drift(fixed, fixed_data, name)
        print(f"{name:14s} {drift:12.2e} {drift / e_scale * 100:9.4f}% {spread:22.2e}")
    print("→ euler / implicit / implicitfast는 '속도에 비례하는 힘(감쇠 등)'을 다루는 방식만 다릅니다.")
    print("  이 진자엔 감쇠가 0이라 세 적분기가 완전히 같은 계산을 하므로 결과도 같습니다.")
    print("→ 마찰·감쇠가 없는 이상계에서는 RK4가 에너지를 가장 잘 보존합니다.")
    print("  로봇처럼 접촉·감쇠·강한 PD 게인이 있는 계에서는 implicitfast가 안정적이고 빨라서")
    print("  이 프로젝트의 기본값으로 씁니다 (builder.SimConfig.integrator).")


if __name__ == "__main__":
    main()
