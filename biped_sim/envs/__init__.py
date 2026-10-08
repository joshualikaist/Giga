"""강화학습 환경.

- balance.py  : BipedBalanceEnv (Gymnasium, CPU MuJoCo)  — .venv + bash scripts/setup_learning.sh
- walk_mjx.py : BipedWalkMjxEnv (Brax, GPU MuJoCo MJX)    — .venv-mjx (bash scripts/setup_gpu_learning.sh)

두 환경은 필요한 패키지가 서로 다른 가상환경에 있으므로, 실제로 쓸 때만 불러온다 (PEP 562 지연 import).
biped_sim 본체는 학습 패키지 없이도 동작한다.
"""

_BALANCE = {"BipedBalanceEnv", "evaluate_push_recovery", "format_results", "zero_policy"}

__all__ = sorted(_BALANCE)


def __getattr__(name):
    if name in _BALANCE:
        from . import balance
        return getattr(balance, name)
    raise AttributeError(f"module 'biped_sim.envs' has no attribute {name!r}")
