#!/usr/bin/env python3
"""기본 도형(box / cylinder / sphere)의 질량 관성 모멘트 계산기.

URDF의 <inertial> 태그는 "링크의 질량 중심(COM)에 대한" 관성 텐서를 요구합니다.
손으로 계산하면 실수가 잦기 때문에, URDF를 작성할 때 이 스크립트로 값을 뽑아 씁니다.

사용 예:
    python scripts/inertia_calc.py box 4.0 0.12 0.22 0.20      # 질량, x, y, z 길이(전체 길이)
    python scripts/inertia_calc.py cylinder 1.0 0.03 0.25       # 질량, 반지름, 길이 (축 = z)
    python scripts/inertia_calc.py sphere 0.5 0.05              # 질량, 반지름

출력은 URDF에 그대로 붙여 넣을 수 있는 <inertia .../> 한 줄입니다.

공식 (모두 COM 기준, 도형의 주축 = 좌표축일 때):
    box (x, y, z 전체 길이):  Ixx = m(y²+z²)/12,  Iyy = m(x²+z²)/12,  Izz = m(x²+y²)/12
    cylinder (반지름 r, 길이 h, 축=z):  Ixx = Iyy = m(3r²+h²)/12,  Izz = m r²/2
    sphere (반지름 r):  Ixx = Iyy = Izz = 2 m r²/5
"""
import argparse


def box_inertia(m, x, y, z):
    return (m * (y**2 + z**2) / 12.0,
            m * (x**2 + z**2) / 12.0,
            m * (x**2 + y**2) / 12.0)


def cylinder_inertia(m, r, h):
    i_xy = m * (3 * r**2 + h**2) / 12.0
    return (i_xy, i_xy, m * r**2 / 2.0)


def sphere_inertia(m, r):
    i = 2.0 * m * r**2 / 5.0
    return (i, i, i)


def to_urdf(ixx, iyy, izz):
    # 기본 도형을 축 정렬로 배치하면 곱관성(ixy, ixz, iyz)은 0 입니다.
    return (f'<inertia ixx="{ixx:.6g}" ixy="0" ixz="0" '
            f'iyy="{iyy:.6g}" iyz="0" izz="{izz:.6g}"/>')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="shape", required=True)
    b = sub.add_parser("box"); b.add_argument("mass", type=float)
    b.add_argument("x", type=float); b.add_argument("y", type=float); b.add_argument("z", type=float)
    c = sub.add_parser("cylinder"); c.add_argument("mass", type=float)
    c.add_argument("radius", type=float); c.add_argument("length", type=float)
    s = sub.add_parser("sphere"); s.add_argument("mass", type=float); s.add_argument("radius", type=float)
    a = p.parse_args()

    if a.shape == "box":
        vals = box_inertia(a.mass, a.x, a.y, a.z)
    elif a.shape == "cylinder":
        vals = cylinder_inertia(a.mass, a.radius, a.length)
    else:
        vals = sphere_inertia(a.mass, a.radius)
    print(to_urdf(*vals))


if __name__ == "__main__":
    main()
