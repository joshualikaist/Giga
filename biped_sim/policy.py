"""numpy만으로 학습한 보행 정책 돌리기 — jax·Brax 없이 (ROS2 policy_node, 나중에 실물 보드에서도 같은 방식).

Brax PPO 정책 파일(params.pkl)을 learning/export_policy.py로 .npz로 바꿔 두면, 여기서 그대로 읽어 계산한다:
    관측 정규화 (관측 − 평균) / 표준편차 → 은닉층 (선형 + silu) × 3 → 출력층 → 앞쪽 n칸(평균)에 tanh = 행동 (−1~1)
학습 때와 행동이 같은지는 tests/test_walk_mjx.py::test_numpy_policy_matches_brax가 확인한다.

    policy = NumpyPolicy.load("learning/pretrained/duck_cmd_policy.npz")
    action = policy(obs)                 # obs: spec.obs_size개, action: n개
    policy.config                        # 학습 설정 (같은 이름의 .json) → walk_spec.spec_from_config(policy.config)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def silu(x):
    return x / (1.0 + np.exp(-x))


class NumpyPolicy:
    def __init__(self, mean, std, layers, config: dict):
        self.mean, self.std = mean, std
        self.layers = layers            # [(kernel, bias), ...] 마지막이 출력층 (행동 평균 n + 표준편차 n)
        self.config = config
        self.action_size = layers[-1][1].shape[0] // 2

    @classmethod
    def load(cls, path: str | Path) -> "NumpyPolicy":
        """<이름>.npz (가중치) + <이름>.json (학습 설정, 정책 파일과 같은 곳)."""
        path = Path(path)
        w = np.load(path)
        n_layers = len([k for k in w.files if k.startswith("kernel_")])
        layers = [(w[f"kernel_{i}"], w[f"bias_{i}"]) for i in range(n_layers)]
        config = json.loads(path.with_suffix(".json").read_text())
        return cls(w["obs_mean"], w["obs_std"], layers, config)

    def __call__(self, obs) -> np.ndarray:
        x = (np.asarray(obs, dtype=np.float32) - self.mean) / self.std
        for kernel, bias in self.layers[:-1]:
            x = silu(x @ kernel + bias)
        kernel, bias = self.layers[-1]
        out = x @ kernel + bias
        return np.tanh(out[..., :self.action_size])   # 결정적 행동 = tanh(평균) (Brax tanh_normal 분포의 mode)
