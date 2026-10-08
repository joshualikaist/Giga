"""울퉁불퉁한 지형(biped_sim/terrain.py) 테스트: 높이가 MuJoCo의 실제 표면과 같은지, 실행 중 바꿔 끼우기, 출발 자리.

높이맵은 MuJoCo가 모델을 만들 때 데이터를 0~1로 다시 늘이므로, 넣는 방법이 틀리면 언덕이 2배로 높아진다 (실측).
그래서 MuJoCo의 광선(mj_ray)으로 표면 높이를 직접 재서 terrain.height_at과 비교한다.
"""
import mujoco
import numpy as np
import pytest

from biped_sim import SIMPLE_BIPED, build_robot_model
from biped_sim.terrain import KINDS, make_terrain, make_training_terrain, max_slope_deg


def _surface_error_mm(model, terrain, n=200):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    floor, geom_id, errors = model.geom("floor").id, np.zeros(1, dtype=np.int32), []
    lo = np.array(terrain.center) - [terrain.size_x - 0.1, terrain.size_y - 0.1]
    hi = np.array(terrain.center) + [terrain.size_x - 0.1, terrain.size_y - 0.1]
    for x, y in np.random.default_rng(0).uniform(lo, hi, (n, 2)):
        dist = mujoco.mj_ray(model, data, np.array([x, y, 1.0]), np.array([0, 0, -1.0]), None, 1, -1, geom_id)
        if geom_id[0] == floor:
            errors.append((1.0 - dist) - terrain.height_at(x, y))
    return 1000 * np.abs(errors).max()


@pytest.fixture(scope="module")
def hills_model():
    terrain = make_terrain("hills", seed=3, difficulty=1.0)
    model, _ = build_robot_model(SIMPLE_BIPED.urdf, SIMPLE_BIPED.sim_config(terrain=terrain))
    return model, terrain


def test_terrain_height_matches_mujoco_surface(hills_model):
    model, terrain = hills_model
    assert _surface_error_mm(model, terrain) < 1.0     # 격자 사이 보간 차이만 (실측 0.3 mm)


def test_terrain_can_be_swapped_at_runtime(hills_model):
    model, _ = hills_model
    other = make_terrain("slopes", seed=9, difficulty=0.7)
    other.apply(model)
    assert _surface_error_mm(model, other) < 1.0


@pytest.mark.parametrize("kind", KINDS)
def test_start_area_is_flat_and_difficulty_scales(kind):
    easy, hard = (make_terrain(kind, seed=1, difficulty=d) for d in (0.0, 1.0))
    assert abs(hard.height_at(0.0, 0.0)) < 1e-9 and abs(hard.height_at(0.3, 0.1)) < 1e-9   # 출발 자리 평평
    if kind != "flat":
        assert np.abs(hard.heights).max() > np.abs(easy.heights).max()
        assert max_slope_deg(hard) > max_slope_deg(easy)


def test_training_terrain_mixes_rough_and_flat_regions():
    t = make_training_terrain(seed=0, resolution=0.08)
    dx = 2 * t.size_x / (t.ncol - 1)
    gy, gx = np.gradient(t.heights, dx)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    assert 0.01 < np.abs(t.heights).max() < 0.06            # 높이 수 cm (오리 로봇 크기)
    assert np.percentile(slope, 90) > 3.0                    # 충분히 울퉁불퉁 (평균으로 섞어 상쇄되지 않음)
    assert np.mean((slope < 2) & (np.abs(t.heights) < 0.003)) > 0.1   # 평지 구역도 있음
