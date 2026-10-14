"""key_teleop — 터미널에서 방향키로 /cmd_vel 보내기 (학습한 오리 조종, docs/11).

    ros2 run giga_sim_ros key_teleop          # 이 터미널을 클릭해 둔 채로 키를 누름

    ↑ / ↓              앞뒤 속도 +0.05 / −0.05 m/s
    ← / →              회전 +0.1 / −0.1 rad/s (왼쪽 / 오른쪽)
    PageUp / PageDown  옆 속도 +0.04 / −0.04 m/s (왼쪽 / 오른쪽)
    Space, Backspace   멈춤 (제자리 걸음)
    Home               기본 명령 (앞 0.15)
    r                  처음 자리로 (/policy/reset)
    Ctrl+C             끝

play_walk.py --view(README 16단계)와 같은 키. 방향키는 한글 입력 상태에서도 그대로 들어온다
(teleop_twist_keyboard는 글자 키(i, j, l…)를 쓰는데, 한글 상태면 'ㅑ'가 들어가 '멈춤'이 되고 아무 표시도 없음).
명령은 키를 누를 때만이 아니라 10 Hz로 계속 보낸다 → policy_node는 0.5초 동안 명령이 끊기면 멈추는 안전장치를 그대로 씀.
"""
import os
import select
import sys
import termios
import tty

import rclpy
from geometry_msgs.msg import Twist
from std_srvs.srv import Trigger

from .shutdown import init_with_stop_flag

# 키 → (명령 성분 0=vx 1=vy 2=wz, 바꾸는 양). 단계는 play_walk.py KEYS와 같음
STEPS = {'up': (0, +0.05), 'down': (0, -0.05), 'left': (2, +0.1), 'right': (2, -0.1),
         'pageup': (1, +0.04), 'pagedown': (1, -0.04)}
ESCAPES = {b'\x1b[A': 'up', b'\x1b[B': 'down', b'\x1b[C': 'right', b'\x1b[D': 'left',
           b'\x1b[5~': 'pageup', b'\x1b[6~': 'pagedown', b'\x1b[H': 'home', b'\x1b[1~': 'home', b'\x1bOH': 'home',
           b'\x1bOA': 'up', b'\x1bOB': 'down', b'\x1bOC': 'right', b'\x1bOD': 'left'}
DEFAULT = (0.15, 0.0, 0.0)
LIMITS = ((-0.20, 0.40), (-0.15, 0.15), (-0.5, 0.5))   # 오리 명령 정책(5차 R4)의 학습 범위 (policy_node도 정책의 범위로 자름)


def parse_keys(buf: bytes) -> list[str]:
    """터미널에서 읽은 바이트 → 키 이름 목록. 모르는 키는 'other', 한글 글자는 'hangul'."""
    keys, i = [], 0
    while i < len(buf):
        if buf[i] == 0x1b:
            for seq, name in ESCAPES.items():
                if buf.startswith(seq, i):
                    keys.append(name)
                    i += len(seq)
                    break
            else:
                keys.append('other')
                i += 1
            continue
        ch = buf[i:i + 1]
        if ch in (b' ', b'\x7f', b'\x08'):
            keys.append('stop')
        elif ch in (b'r', b'R'):
            keys.append('reset')
        elif ch == b'\x03':
            keys.append('quit')
        elif buf[i] >= 0x80:                      # UTF-8 여러 바이트 글자 (한글 등)
            n = 3 if buf[i] >= 0xe0 else 2
            keys.append('hangul' if buf[i] in (0xe3, 0xea, 0xeb, 0xec, 0xed) else 'other')
            i += n
            continue
        else:
            keys.append('other')
        i += 1
    return keys


def apply_key(cmd: list[float], key: str) -> list[float]:
    """키 하나를 명령 (vx, vy, wz)에 적용 (학습 범위 안으로)."""
    cmd = list(cmd)
    if key in STEPS:
        k, dv = STEPS[key]
        cmd[k] = round(min(max(cmd[k] + dv, LIMITS[k][0]), LIMITS[k][1]), 3)
    elif key == 'stop':
        cmd = [0.0, 0.0, 0.0]
    elif key == 'home':
        cmd = list(DEFAULT)
    return cmd


def status(cmd, note=''):
    return (f'\r\x1b[2K명령  앞뒤 {cmd[0]:+.2f} m/s   옆 {cmd[1]:+.2f} m/s   회전 {cmd[2]:+.1f} rad/s'
            + (f'   ← {note}' if note else ''))


def main(args=None):
    stop = init_with_stop_flag(args)
    node = rclpy.create_node('key_teleop')
    pub = node.create_publisher(Twist, 'cmd_vel', 10)
    reset = node.create_client(Trigger, 'policy/reset')
    if not sys.stdin.isatty():
        node.get_logger().error('터미널에서 실행하세요 (키보드 입력을 읽어야 함)')
        return
    print(__doc__.split('\n\n')[1])
    print('\n이 터미널을 클릭해 둔 채로 키를 누르세요.')
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    cmd = [0.0, 0.0, 0.0]
    sys.stdout.write(status(cmd))
    sys.stdout.flush()
    try:
        tty.setcbreak(fd)                         # 한 키씩 바로 읽기, 화면에 글자 안 찍힘 (Ctrl+C는 그대로 신호)
        while not stop.is_set() and rclpy.ok():
            ready, _, _ = select.select([fd], [], [], 0.1)
            note = ''
            if ready:
                buf = os.read(fd, 64)
                while select.select([fd], [], [], 0.01)[0]:   # 방향키는 여러 바이트로 나눠 올 수 있음
                    buf += os.read(fd, 64)
                for key in parse_keys(buf):
                    if key == 'quit':
                        stop.set()
                    elif key == 'reset':
                        note = '처음 자리로' if reset.service_is_ready() else '/policy/reset 없음 (duck.launch.py가 떠 있나요?)'
                        if reset.service_is_ready():
                            reset.call_async(Trigger.Request())
                        cmd = [0.0, 0.0, 0.0]
                    elif key == 'hangul':
                        note = '한글 입력 상태 — 방향키를 쓰거나 한/영 키로 영어로 바꾸세요'
                    elif key == 'other':
                        note = '모르는 키 (방향키, PageUp/Down, Space, Home, r)'
                    else:
                        cmd = apply_key(cmd, key)
                sys.stdout.write(status(cmd, note))
                sys.stdout.flush()
            msg = Twist()
            msg.linear.x, msg.linear.y, msg.angular.z = cmd
            pub.publish(msg)                      # 10 Hz로 계속 (키를 안 눌러도)
            rclpy.spin_once(node, timeout_sec=0.0)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        pub.publish(Twist())                      # 끝낼 때 멈춤 명령
        print()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
