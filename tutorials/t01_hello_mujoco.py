#!/usr/bin/env python3
"""t01 — MuJoCo Hello World: 상자와 공 떨어뜨리기

학습 목표
    1. MJCF 파일 → MjModel(설계도) → MjData(현재 상태) 의 흐름을 이해한다.
    2. 시뮬레이션은 결국 `mj_step()`을 반복 호출하는 루프임을 확인한다.
    3. 뷰어를 띄우고 조작해 본다.

실행
    python tutorials/t01_hello_mujoco.py              # 뷰어 (창을 닫으면 종료)
    python tutorials/t01_hello_mujoco.py --headless   # 뷰어 없이 높이 변화를 표로 출력

뷰어 조작 (F1: 전체 도움말)
    마우스 왼쪽 드래그: 회전 / 오른쪽 드래그: 이동 / 휠: 확대
    Space: 일시정지   Backspace: 초기화   더블클릭: 물체 선택
    Ctrl + 오른쪽 드래그: 선택한 물체를 "손으로" 밀기

이 파일은 일부러 biped_sim 헬퍼를 쓰지 않고 MuJoCo API만 사용합니다.
"""
import argparse
import time

import mujoco
import mujoco.viewer

from biped_sim.paths import FALLING_BOX_XML


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float, default=2.0, help="헤드리스 시뮬레이션 시간 [s]")
    args = parser.parse_args()

    # ① 모델 로드: XML을 읽어 "변하지 않는 정보"(질량, 형상, 관절 구조, timestep...)를 담은 MjModel 생성
    model = mujoco.MjModel.from_xml_path(str(FALLING_BOX_XML))
    # ② 데이터 생성: 위치, 속도, 힘, 접촉 등 "매 순간 바뀌는 정보"를 담는 MjData
    #    하나의 model로 여러 data를 만들 수 있다 (예: 병렬 시뮬레이션).
    data = mujoco.MjData(model)

    print(f"모델 로드: {FALLING_BOX_XML.name}")
    print(f"  timestep = {model.opt.timestep} s  →  1초 = {round(1 / model.opt.timestep)} 스텝")
    print(f"  바디 수 = {model.nbody} (world 포함),  nq = {model.nq},  nv = {model.nv}")

    # MjData를 막 만든 직후에는 qpos만 있고 world 좌표(xpos 등)는 아직 0입니다.
    # mj_forward로 한 번 계산해 둡니다. (자세한 차이는 t02에서)
    mujoco.mj_forward(model, data)

    if args.headless:
        # ③ 시뮬레이션 루프 (화면 없이). 0.1초마다 상자/공의 높이를 출력
        print(f"\n{'time[s]':>8} {'box z[m]':>9} {'ball z[m]':>10} {'contacts':>9}")
        print_every = round(0.1 / model.opt.timestep)
        step = 0
        while data.time < args.duration:
            if step % print_every == 0:
                # data.body("이름") 으로 바디별 결과에 접근. xpos = world 좌표계 위치
                print(f"{data.time:8.2f} {data.body('box').xpos[2]:9.3f} "
                      f"{data.body('ball').xpos[2]:10.3f} {data.ncon:9d}")
            mujoco.mj_step(model, data)  # ← 시뮬레이션의 전부. timestep만큼 시간을 진행
            step += 1
        print("\n관찰 포인트: 상자는 z≈0.1(반 변 길이), 공은 z≈0.08(반지름)에서 멈춰야 합니다.")
        return

    # ③ 시뮬레이션 루프 (뷰어). launch_passive: 루프는 우리가 돌리고 뷰어는 화면만 그림
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.perf_counter()
            mujoco.mj_step(model, data)
            viewer.sync()  # 현재 data를 화면에 반영
            # 실시간 속도 맞추기: 계산이 timestep보다 빨리 끝나면 남은 시간만큼 대기
            remaining = model.opt.timestep - (time.perf_counter() - step_start)
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
