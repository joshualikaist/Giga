#!/usr/bin/env python3
"""Brax 정책 파일(params.pkl) → numpy 정책(.npz + .json) — jax 없이 돌리기 위해 (ROS2 policy_node, 실물 보드)

실행 (GPU 학습 환경에서: source scripts/activate_gpu.sh)
    python learning/export_policy.py --params learning/pretrained/duck_cmd_policy.pkl
        → learning/pretrained/duck_cmd_policy.npz (+ 같은 이름 .json이 없으면 학습 설정을 복사)
    python learning/export_policy.py --params output/learning/<실행>/params.pkl --out my_policy.npz

.npz 안: obs_mean, obs_std (관측 정규화), kernel_i, bias_i (정책 신경망 층, 0부터). 가치(critic) 신경망은 넣지 않음.
변환한 정책은 같은 관측에서 Brax 정책과 같은 행동을 내는지 확인한 뒤 저장한다 (차이 > 1e-4면 오류).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from walk_tools import get_spec, load_policy

from biped_sim.policy import NumpyPolicy


def export(params_path: Path, out_path: Path) -> Path:
    from brax.io import model as brax_model

    policy, config = load_policy(params_path)
    normalizer, policy_params = brax_model.load_params(str(params_path))[:2]
    mean, std = normalizer.mean, normalizer.std
    if isinstance(mean, dict):   # 비대칭 actor-critic: 정책은 "state"만
        mean, std = mean["state"], std["state"]
    layers = policy_params["params"]
    arrays = {"obs_mean": np.asarray(mean, np.float32), "obs_std": np.asarray(std, np.float32)}
    for i in range(len(layers)):
        arrays[f"kernel_{i}"] = np.asarray(layers[f"hidden_{i}"]["kernel"], np.float32)
        arrays[f"bias_{i}"] = np.asarray(layers[f"hidden_{i}"]["bias"], np.float32)

    # 확인: 무작위 관측 100개에서 Brax 정책과 같은 행동
    spec = get_spec(config)
    rng = np.random.default_rng(0)
    obs = (arrays["obs_mean"] + arrays["obs_std"] * rng.normal(size=(100, spec.obs_size))).astype(np.float32)
    mine = NumpyPolicy(arrays["obs_mean"], arrays["obs_std"],
                       [(arrays[f"kernel_{i}"], arrays[f"bias_{i}"]) for i in range(len(layers))], config)
    diff = max(float(np.abs(mine(o) - policy(o)).max()) for o in obs)
    if diff > 1e-4:
        raise RuntimeError(f"numpy 정책이 Brax 정책과 다름 (최대 차이 {diff:.2e})")

    np.savez(out_path, **arrays)
    config_out = out_path.with_suffix(".json")
    if not config_out.exists():
        config_out.write_text(json.dumps(config, ensure_ascii=False, indent=2))
    print(f"저장: {out_path} ({out_path.stat().st_size / 1024:.0f} KB, 층 {len(layers)}개, 관측 {spec.obs_size} → 행동 {spec.n}), "
          f"Brax 정책과 최대 차이 {diff:.1e}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--params", type=Path, required=True, help="Brax 정책 파일 (params.pkl, pretrained/*.pkl)")
    parser.add_argument("--out", type=Path, help="저장할 .npz (기본: 정책 파일 옆, 실행 폴더면 policy.npz)")
    args = parser.parse_args()
    out = args.out or (args.params.with_name("policy.npz") if args.params.name == "params.pkl"
                       else args.params.with_suffix(".npz"))
    export(args.params, out)


if __name__ == "__main__":
    main()
