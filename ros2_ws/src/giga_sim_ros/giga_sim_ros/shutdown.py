"""노드 종료 처리 — Ctrl+C(SIGINT)와 종료 요청(SIGTERM)을 안전하게 받는 방법.

신호가 오면 "멈춰 달라"는 표시(threading.Event)만 남기고, 실제 정리는 메인 루프가
다음 차례에 빠져나와서 순서대로 한다 (뷰어 → 노드 → rclpy).

왜 이렇게까지 하나? (모두 이 PC에서 실제로 재현한 문제, docs/05 §2)
  1) rclpy 기본 신호 처리: 신호 순간 rclpy가 먼저 종료돼, 타이머가 발행 중이면
     "RCLError: context is not valid" 트레이스 (SIGTERM 5회 중 2회)
  2) 신호를 KeyboardInterrupt로 바꾸는 방식: 수신 메시지를 Python 객체로 바꾸는 C++ 코드 안에서
     터지면 pybind11이 "RuntimeError: Unable to convert call argument"로 바꿔 버려 잡히지 않음 (20회 중 1회)
  → 신호 처리기에서는 예외를 던지지 않고 표시만 남긴다.
"""
import signal
import threading

import rclpy
from rclpy.signals import SignalHandlerOptions


def init_with_stop_flag(args=None) -> threading.Event:
    """rclpy.init + 종료 신호 처리기 설치. 반환된 Event가 set되면 메인 루프를 끝내면 된다."""
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    stop = threading.Event()

    def request_stop(signum, frame):
        stop.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    return stop
