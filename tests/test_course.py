"""직접 만드는 험지 (biped_sim/course.py, docs/11 §3): 코스 파일·그림 → 높이맵.

블록마다 높이가 설명대로인지, 출발 자리가 평평한지, 그림의 방향(위 = 앞, 왼쪽 = 왼쪽)이 맞는지, 예시 코스가 모두 읽히는지.
"""
import numpy as np
import pytest

from biped_sim.course import COURSES_DIR, build_course, load_course

AREA = {"x": [-1.0, 4.0], "y": [-1.0, 1.0]}


def course(*features, resolution=0.02):
    return build_course({"area": AREA, "resolution": resolution, "features": list(features)})


def test_stairs_ramp_bump_heights():
    t = course({"type": "stairs", "at": [1.0, 0.0], "steps": 2, "rise": 0.01, "run": 0.3, "width": 1.0, "top": 0.4,
                "down": True})
    h = [float(t.height_at(x, 0.0)) for x in (0.95, 1.15, 1.45, 1.8, 2.15, 2.45, 2.65)]
    assert h == pytest.approx([0.0, 0.01, 0.02, 0.02, 0.01, 0.0, 0.0], abs=1e-6)   # 오르막 2칸 → 꼭대기 → 내리막
    assert float(t.height_at(1.15, 0.6)) == 0.0                                       # 폭 밖

    t = course({"type": "ramp", "at": [1.0, 0.0], "length": 1.0, "angle": 5, "width": 1.0, "top": 0.5, "down": True})
    top = np.tan(np.radians(5))
    assert float(t.height_at(1.5, 0.0)) == pytest.approx(top / 2, abs=1e-3)
    assert float(t.height_at(2.2, 0.0)) == pytest.approx(top, abs=1e-6)
    assert float(t.height_at(3.0, 0.0)) == pytest.approx(top / 2, abs=1e-3)

    t = course({"type": "bump", "at": [1.5, 0.0], "depth": 0.06, "width": 1.0, "height": 0.015}, resolution=0.01)
    assert float(t.height_at(1.5, 0.0)) == pytest.approx(0.015) and float(t.height_at(1.6, 0.0)) == 0.0


def test_yaw_rotates_feature_and_start_stays_flat():
    """yaw 90 = 왼쪽(+y)으로 뻗음. 출발 자리(반지름 0.35 m)에 블록을 놓아도 높이 0 (서 있는 자세 기준)."""
    t = course({"type": "box", "at": [0.0, 0.6], "yaw": 90, "size": [0.4, 0.2], "height": 0.02},
               {"type": "box", "at": [0.0, 0.0], "size": [0.3, 0.3], "height": 0.05})
    assert float(t.height_at(0.0, 0.75)) == pytest.approx(0.02)      # yaw 90: 앞뒤 0.4가 y 방향
    assert float(t.height_at(0.15, 0.6)) == 0.0                       # x로는 폭 0.2의 절반 0.1 밖
    assert np.allclose([t.height_at(x, y) for x, y in ((0, 0), (0.1, 0.1), (-0.2, 0.0))], 0.0)


def test_png_orientation_top_is_forward_left_is_left(tmp_path):
    """그림의 위쪽 = 앞(+x), 왼쪽 = +y. 흰색 = height."""
    from PIL import Image
    img = np.zeros((100, 60), np.uint8)
    img[:10, :10] = 255          # 그림 왼쪽 위 → 앞쪽 왼쪽
    Image.fromarray(img).save(tmp_path / "pic.png")
    t = build_course({"area": {"x": [-0.5, 2.5], "y": [-1.0, 1.0]}, "resolution": 0.02,
                      "features": [{"type": "image", "file": "pic.png", "at": [0.0, 0.0], "size": [2.0, 1.2], "height": 0.03}]},
                     base_dir=tmp_path)
    assert float(t.height_at(1.9, 0.5)) == pytest.approx(0.03)        # 앞쪽 왼쪽이 높음
    assert float(t.height_at(1.9, -0.5)) == 0.0 and float(t.height_at(0.5, 0.5)) == 0.0
    direct = load_course(tmp_path / "pic.png")                         # 그림만: 1 px = 2 cm, 아래 0.5 m에서 출발
    assert 2 * direct.size_x == pytest.approx(2.0) and 2 * direct.size_y == pytest.approx(1.2)
    assert float(direct.height_at(1.45, 0.5)) == pytest.approx(0.03)


def test_example_courses_load_and_build_models():
    """models/courses의 예시가 모두 읽히고, 출발 자리가 평평하고, 오리 학습 모델(walk_spec)에 들어감."""
    from biped_sim import OPEN_DUCK_MINI
    names = sorted(p.stem for p in COURSES_DIR.glob("*.yaml"))
    assert {"stairs", "ramp", "stones", "gravel", "playground", "drawing"} <= set(names)
    for name in names:
        t = load_course(name)
        assert float(t.height_at(0.0, 0.0)) == 0.0 and t.heights.max() <= 0.15, name
    if OPEN_DUCK_MINI.urdf.exists():
        from biped_sim.envs import walk_spec
        spec = walk_spec.spec_from_config({"robot": "open_duck_mini", "servo": "open_duck", "home_knee": 0.6})
        model, _ = walk_spec.build_mjx_model(spec, load_course("playground"))
        assert model.hfield_nrow[0] == 501 and model.hfield_ncol[0] == 501


def test_unknown_course_and_feature_errors():
    with pytest.raises(FileNotFoundError, match="stairs"):
        load_course("no_such_course")
    with pytest.raises(ValueError, match="모르는 type"):
        course({"type": "lava"})
    with pytest.raises(ValueError, match="출발 자리"):
        build_course({"area": {"x": [1.0, 3.0], "y": [-1, 1]}, "features": []})


def test_cross_slope_is_centered_and_eased():
    t = course({"type": "ramp", "at": [0.5, 0.0], "length": 0.0, "rise": 0.0, "width": 1.6, "top": 2.0, "cross_slope": 5,
                "mode": "add"})
    s = np.tan(np.radians(5))
    assert float(t.height_at(1.5, 0.0)) == pytest.approx(0.0, abs=1e-9)            # 가운데 줄은 그대로
    assert float(t.height_at(1.5, 0.5)) == pytest.approx(0.5 * s, abs=1e-3)        # 왼쪽 높게
    assert float(t.height_at(1.5, -0.5)) == pytest.approx(-0.5 * s, abs=1e-3)      # 오른쪽 낮게 (add)
    assert abs(float(t.height_at(0.55, 0.5))) < 0.25 * s                            # 들어갈 때는 거의 평평
