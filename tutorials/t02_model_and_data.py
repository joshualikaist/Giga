#!/usr/bin/env python3
"""t02 — MjModel과 MjData 해부하기 (뷰어 없음, 출력만)

학습 목표
    1. qpos(위치)와 qvel(속도)의 길이가 왜 다른지 (nq ≠ nv) 이해한다.
    2. 이름으로 바디/관절에 접근하는 법을 익힌다.
    3. mj_forward와 mj_step의 차이를 확인한다.
    4. 쿼터니언 순서(MuJoCo: w,x,y,z / ROS: x,y,z,w)를 기억한다.

실행
    python tutorials/t02_model_and_data.py
"""
import mujoco
import numpy as np

from biped_sim.paths import FALLING_BOX_XML
from biped_sim.utils import quat_to_rpy, quat_wxyz_to_xyzw

JOINT_TYPE_NAMES = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}
JOINT_QPOS_SIZE = {0: 7, 1: 4, 2: 1, 3: 1}  # 관절 종류별 qpos 칸 수
JOINT_QVEL_SIZE = {0: 6, 1: 3, 2: 1, 3: 1}  # 관절 종류별 qvel 칸 수


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main():
    model = mujoco.MjModel.from_xml_path(str(FALLING_BOX_XML))
    data = mujoco.MjData(model)

    section("1. 크기 정보 (MjModel = 변하지 않는 설계도)")
    print(f"nbody={model.nbody}  njnt={model.njnt}  ngeom={model.ngeom}  nu(액추에이터)={model.nu}")
    print(f"nq(위치 변수 개수)={model.nq}   nv(속도 변수 = 자유도 개수)={model.nv}")

    section("2. 관절별 qpos / qvel 주소 — nq ≠ nv 인 이유")
    print(f"{'joint':12s} {'type':6s} {'qposadr':>7s} {'qpos칸':>6s} {'dofadr':>6s} {'qvel칸':>6s}")
    for j in range(model.njnt):
        t = int(model.jnt_type[j])
        print(f"{model.joint(j).name:12s} {JOINT_TYPE_NAMES[t]:6s} {model.jnt_qposadr[j]:7d} "
              f"{JOINT_QPOS_SIZE[t]:6d} {model.jnt_dofadr[j]:6d} {JOINT_QVEL_SIZE[t]:6d}")
    print("→ freejoint의 위치 = [x y z] + 쿼터니언[w x y z] = 7칸, 속도 = 선속도 3 + 각속도 3 = 6칸")
    print("→ 따라서 '관절 i의 값 = qpos[i]' 같은 가정은 틀립니다. 항상 jnt_qposadr / jnt_dofadr를 쓰세요.")

    section("3. 초기 상태 (MjData = 매 순간 바뀌는 상태)")
    box_q = data.joint("box_free").qpos  # 이름으로 접근. 원본 배열의 '뷰'이므로 수정하면 data도 바뀜
    print(f"box qpos = {np.round(box_q, 4)}")
    print(f"  위치 = {box_q[:3]},  쿼터니언(w,x,y,z) = {np.round(box_q[3:], 4)}")
    print(f"  → ROS 순서(x,y,z,w)로 바꾸면 {np.round(quat_wxyz_to_xyzw(box_q[3:]), 4)}")
    rpy = np.degrees(quat_to_rpy(box_q[3:]))
    print(f"  → URDF/ROS 방식 roll/pitch/yaw [deg] = {np.round(rpy, 2)}")
    print("     XML에는 euler=\"20 30 0\"이라고 썼는데 왜 (20, 30, 0)이 아닐까?")
    print("     MJCF euler 기본 순서는 'xyz'(소문자 = 회전하는 축 기준, intrinsic)이고,")
    print("     URDF rpy는 고정된 축 X→Y→Z 기준(extrinsic)이기 때문입니다.")
    seq_xyz = mujoco.MjModel.from_xml_string(
        '<mujoco><compiler eulerseq="XYZ"/><worldbody><body euler="20 30 0">'
        '<freejoint/><geom size=".1"/></body></worldbody></mujoco>')
    rpy_xyz = np.degrees(quat_to_rpy(seq_xyz.qpos0[3:]))
    print(f"     <compiler eulerseq=\"XYZ\"/>로 바꾸면 → {np.round(rpy_xyz, 2)}  (URDF rpy와 일치)")
    print("     교훈: 각도를 옮겨 적을 때는 '어느 규약의 오일러각인지' 반드시 확인할 것.")

    section("4. mj_forward vs mj_step")
    print(f"처음 data.body('box').xpos = {data.body('box').xpos}   ← 아직 아무것도 계산 안 됨 (0)")
    mujoco.mj_forward(model, data)
    print(f"mj_forward 후 xpos       = {data.body('box').xpos}   ← qpos로부터 위치 계산됨")
    data.joint("box_free").qpos[2] = 2.0  # 상자를 2 m 높이로 옮김
    print(f"qpos[z]=2.0 으로 바꾼 직후 xpos = {data.body('box').xpos}   ← 아직 옛날 값!")
    mujoco.mj_forward(model, data)
    print(f"다시 mj_forward 후 xpos   = {data.body('box').xpos}")
    print("→ mj_forward: 시간을 진행하지 않고 위치/센서/접촉 등 파생량만 다시 계산")
    print("→ mj_step   : mj_forward의 계산 + 적분으로 시간을 timestep만큼 진행")

    section("5. 질량과 관성 (컴파일러가 geom 모양에서 자동 계산)")
    for name in ("box", "ball"):
        b = model.body(name)
        print(f"{name:5s} mass={b.mass[0]:.3f} kg   principal inertia={np.round(b.inertia, 6)} kg·m²")
    print(f"box 이론값: m·(2a)²/6 = 1.0·0.2²/6 = {1.0 * 0.2**2 / 6:.6f}")

    section("6. 물리 옵션은 컴파일 후에도 바꿀 수 있다 — 지구 vs 달")
    for label, g in (("지구", -9.81), ("달", -1.62)):
        mujoco.mj_resetData(model, data)        # 상태를 XML의 초기값으로 되돌림
        model.opt.gravity[:] = (0, 0, g)
        while data.ncon == 0:                   # 처음 접촉이 생길 때까지 진행
            mujoco.mj_step(model, data)
        print(f"{label}: 중력 {g:6.2f} m/s² → 첫 접촉까지 {data.time:.3f} s")
    model.opt.gravity[:] = (0, 0, -9.81)

    section("7. 접촉(contact) 정보")
    mujoco.mj_resetData(model, data)
    mujoco.mj_step(model, data, nstep=1000)    # 2초 진행 (nstep으로 여러 스텝을 한 번에)
    print(f"t={data.time:.2f}s 접촉 개수 = {data.ncon}")
    wrench = np.zeros(6)
    force_sum = 0.0
    for i in range(data.ncon):
        c = data.contact[i]
        mujoco.mj_contactForce(model, data, i, wrench)  # 접촉 좌표계 기준 [법선, 접선1, 접선2, 토크...]
        force_sum += wrench[0]
        print(f"  {model.geom(c.geom1).name:10s} ↔ {model.geom(c.geom2).name:10s}"
              f"  깊이={c.dist * 1000:+.3f} mm   수직력={wrench[0]:7.3f} N")
    weight = sum(model.body(n).mass[0] for n in ("box", "ball")) * -model.opt.gravity[2]
    print(f"→ 수직력의 합 {force_sum:.3f} N  vs  두 물체 무게의 합 {weight:.3f} N  (정지 상태면 같아야 함)")
    print("  (MuJoCo 접촉은 '부드러운' 모델이라 미세하게 파고드는(음수 깊이) 것이 정상)")


if __name__ == "__main__":
    main()
