"""관절 제어기.

관절 PD 제어 = 대부분의 보행 로봇 제어의 가장 아래층
    τ = Kp·(q_des − q) + Kd·(dq_des − dq) + τ_ff

    Kp [N·m/rad]   : 스프링 강성. 클수록 목표 각도를 세게 추종 (너무 크면 진동/발산)
    Kd [N·m·s/rad] : 댐퍼. 진동을 줄임 (너무 크면 굼뜨고, 시간 간격 대비 너무 크면 수치 발산)
    τ_ff           : 피드포워드 토크 (예: 중력 보상). 상위 제어기(보행, 균형)가 여기로 들어온다.

    실제 로봇에서도 이 계산은 보통 모터 드라이버/저수준 루프(1 kHz 이상)에서 돌고,
    ROS2 같은 상위 계층은 q_des(또는 τ_ff)만 보냅니다. 이 시뮬레이터도 같은 구조를 따릅니다.
"""
from __future__ import annotations

import numpy as np


class JointPDController:
    def __init__(self, kp, kd, torque_limits: np.ndarray):
        """
        kp, kd        : 스칼라 또는 관절 개수 길이의 배열
        torque_limits : (관절 개수, 2) 배열 [[min, max], ...]  — RobotInterface.torque_limits
        """
        self.torque_limits = np.asarray(torque_limits, dtype=float)
        n = len(self.torque_limits)
        self.kp = np.broadcast_to(np.asarray(kp, dtype=float), (n,)).copy()
        self.kd = np.broadcast_to(np.asarray(kd, dtype=float), (n,)).copy()

    def compute(self, q, dq, q_des, dq_des=0.0, tau_ff=0.0) -> np.ndarray:
        tau = self.kp * (q_des - q) + self.kd * (dq_des - dq) + tau_ff
        # 모터가 낼 수 있는 토크로 포화(saturation). 실제 모터도 한계 이상은 못 낸다.
        return np.clip(tau, self.torque_limits[:, 0], self.torque_limits[:, 1])
