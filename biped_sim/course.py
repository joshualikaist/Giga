"""직접 만드는 험지 — 코스 파일(YAML: 계단·경사로·디딤돌·턱·자갈밭·그림)이나 흑백 그림(PNG)으로 지형 만들기 (docs/11 §3).

    terrain = load_course("models/courses/stairs.yaml")      # 또는 "my_drawing.png", 또는 이름만 "stairs"
    ros2 launch giga_sim_ros duck.launch.py terrain:=stairs   # ROS 시뮬레이터에서 (sim_node가 이 함수로 읽음)
    python -m biped_sim.course models/courses/stairs.yaml     # 위에서 본 그림 (코스를 고칠 때 확인용)

좌표: 오리는 (0, 0)에서 +x를 보고 출발한다. 출발 자리(반지름 0.35 m)는 늘 평평한 높이 0으로 맞춘다 (모델의 서 있는 자세가
높이 0 기준이라). 코스 파일의 위치는 모두 [m], 높이는 땅(0)에서 잰 [m]. 오리는 작아서 (다리 약 18 cm, 발 들기 약 2 cm)
계단 한 칸 ≈ 2 cm 이하, 경사 ≈ 10° 이하가 현실적인 범위다 (docs/09 §8.3~8.4).

YAML 형식 (models/courses/*.yaml에 예시)
    name: 계단
    area: {x: [-1.0, 6.0], y: [-1.5, 1.5]}   # 지형의 범위 (밖으로 나가면 떨어짐)
    resolution: 0.02                          # 격자 간격 [m] (기본 2 cm. 작을수록 모서리가 날카롭지만 무거움)
    features:                                  # 위에서부터 차례로 쌓음. 겹치면 높은 쪽 (mode: add면 더함)
      - {type: stairs, at: [1.0, 0.0], steps: 4, rise: 0.015, run: 0.3, width: 1.2, top: 0.6, down: true}

모든 블록에 공통: at [x, y] (놓는 기준점), yaw [°] (기본 0 = +x 방향으로 뻗음), mode (max | add).
    box             size [앞뒤, 옆], height                       — 상자 (at = 가운데)
    bump (턱)       depth (앞뒤 두께), width, height                — 길게 가로놓인 낮은 턱 (at = 가운데)
    stairs          steps, rise (한 칸 높이), run (한 칸 깊이), width, top (꼭대기 평평한 길이), down (내리막도)
                                                                      (at = 첫 칸 앞 모서리 가운데)
    ramp            length, rise (끝 높이) 또는 angle [°], width, top, down,
                    cross_slope [°] (옆으로 기운 길: 가운데 그대로, 왼쪽 높게·오른쪽 낮게 — 낮은 쪽까지 쓰려면 mode: add)
                                                                      (at = 오르막 시작 가운데)
    stones (디딤돌)  area [앞뒤, 옆], stone (한 변), gap (사이 간격), height, height_jitter, seed  (at = 영역 시작 가운데)
    gravel (자갈밭)  area [앞뒤, 옆], height (최대 높이), cell (자갈 크기), seed                   (at = 영역 시작 가운데)
    image (그림)     file (PNG, YAML 기준 경로), size [앞뒤, 옆] (생략하면 1 px = resolution), height (흰색 높이)
                                                                      (at = 그림 아래쪽 가운데)

PNG만 줄 때: 1 px = 2 cm, 흰색 = 3 cm, 검정 = 0. **그림의 아래 가운데에서 위쪽을 보고 출발** (MuJoCo 창 '7 위에서' 보기와
같은 방향: 그림 위 = 오리의 앞 +x, 그림 왼쪽 = +y). 회색은 중간 높이. 그림판·GIMP 어느 것이든 흑백으로 저장하면 됨.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from . import paths
from .terrain import Terrain, _grid

COURSES_DIR = paths.MODELS_DIR / "courses"
START_RADIUS = 0.35          # [m] 출발 자리 (늘 평평한 높이 0)
PNG_DEFAULTS = {"resolution": 0.02, "height": 0.03, "margin": 0.5}   # PNG만 줄 때: 1 px, 흰색 높이, 출발 앞 여유


def find_course(name: str | Path) -> Path:
    """'stairs' 같은 이름 → models/courses/stairs.yaml. 경로면 그대로 (상대 경로는 지금 폴더, 없으면 저장소 기준)."""
    p = Path(name).expanduser()
    for cand in (p, paths.REPO_ROOT / p, COURSES_DIR / p, COURSES_DIR / f"{p}.yaml", COURSES_DIR / f"{p}.png"):
        if cand.is_file():
            return cand.resolve()
    known = sorted(f.stem for f in COURSES_DIR.glob("*.yaml")) if COURSES_DIR.is_dir() else []
    raise FileNotFoundError(f"코스 '{name}'를 찾지 못했습니다. 예시 코스: {', '.join(known)} (또는 .yaml / .png 경로)")


def load_course(name: str | Path) -> Terrain:
    """코스 파일(.yaml) 또는 흑백 그림(.png) → Terrain (MuJoCo 높이맵)."""
    path = find_course(name)
    if path.suffix.lower() == ".png":
        img = _read_png(path)
        res, h, margin = PNG_DEFAULTS["resolution"], PNG_DEFAULTS["height"], PNG_DEFAULTS["margin"]
        length, width = img.shape[0] * res, img.shape[1] * res
        spec = {"name": path.stem, "resolution": res,
                "area": {"x": [-margin, length - margin], "y": [-width / 2, width / 2]},
                "features": [{"type": "image", "file": path.name, "at": [-margin, 0.0], "size": [length, width], "height": h}]}
    else:
        import yaml
        spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    return build_course(spec, base_dir=path.parent)


def build_course(spec: dict, base_dir: Path | None = None) -> Terrain:
    """코스 설명(dict, YAML과 같은 모양) → Terrain."""
    res = float(spec.get("resolution", 0.02))
    (x0, x1), (y0, y1) = spec["area"]["x"], spec["area"]["y"]
    if not (x0 < -START_RADIUS and x1 > START_RADIUS and y0 < -START_RADIUS and y1 > START_RADIUS):
        raise ValueError(f"area가 출발 자리 (0, 0) 둘레 {START_RADIUS} m를 포함해야 합니다: x {x0}~{x1}, y {y0}~{y1}")
    size_x, size_y = (x1 - x0) / 2, (y1 - y0) / 2
    center = ((x0 + x1) / 2, (y0 + y1) / 2)
    gx, gy = _grid(size_x, size_y, res)
    wx, wy = gx + center[0], gy + center[1]
    h = np.zeros_like(wx)
    for i, f in enumerate(spec.get("features") or []):
        kind = f.get("type")
        if kind not in FEATURES:
            raise ValueError(f"features[{i}]: 모르는 type '{kind}' (사용 가능: {', '.join(FEATURES)})")
        ax, ay = f.get("at", (0.0, 0.0))
        yaw = math.radians(f.get("yaw", 0.0))
        u = math.cos(yaw) * (wx - ax) + math.sin(yaw) * (wy - ay)      # 블록 기준 앞(+u)
        v = -math.sin(yaw) * (wx - ax) + math.cos(yaw) * (wy - ay)     # 블록 기준 왼쪽(+v)
        feature_h = FEATURES[kind](f, u, v, res=res, base_dir=base_dir or Path.cwd())
        h = h + feature_h if f.get("mode", "max") == "add" else np.maximum(h, feature_h)
    r = np.hypot(wx, wy)                                  # 출발 자리는 평평하게, 바깥으로 0.15 m에 걸쳐 원래 높이로
    h = h * np.clip((r - START_RADIUS) / 0.15, 0.0, 1.0)
    return Terrain(heights=h, size_x=size_x, size_y=size_y, center=center, kind=str(spec.get("name", "course")))


# ---------------------------------------------------------------------------- 블록들 (u: 앞, v: 왼쪽, 블록 기준)
def _inside(u, v, u0, u1, half_w):
    return (u >= u0) & (u < u1) & (np.abs(v) <= half_w)


def _box(f, u, v, **_):
    lu, lv = f["size"]
    return np.where((np.abs(u) <= lu / 2) & (np.abs(v) <= lv / 2), float(f["height"]), 0.0)


def _bump(f, u, v, **_):
    return np.where((np.abs(u) <= f.get("depth", 0.05) / 2) & (np.abs(v) <= f.get("width", 1.0) / 2), float(f["height"]), 0.0)


def _stairs(f, u, v, **_):
    n, rise, run = int(f["steps"]), float(f["rise"]), float(f["run"])
    half_w, top = f.get("width", 1.0) / 2, float(f.get("top", 0.6))
    h = np.zeros_like(u)
    up = _inside(u, v, 0.0, n * run, half_w)
    h = np.where(up, (np.floor(u / run) + 1) * rise, h)
    h = np.where(_inside(u, v, n * run, n * run + top, half_w), n * rise, h)
    if f.get("down", True):
        d = u - (n * run + top)                            # 내리막: 한 칸씩 내려감
        down = _inside(u, v, n * run + top, 2 * n * run + top, half_w)
        h = np.where(down, (n - np.floor(d / run) - 1) * rise, h)
    return h


def _ramp(f, u, v, **_):
    length = float(f.get("length", 0.0))
    rise = float(f["rise"]) if "rise" in f else length * math.tan(math.radians(float(f.get("angle", 0.0))))
    half_w, top, down = f.get("width", 1.0) / 2, float(f.get("top", 0.0)), bool(f.get("down", False))
    span = (2 * length if down else length) + top
    profile = np.where(u < length, u / max(length, 1e-9) * rise, rise)            # 오르막 → 꼭대기
    if down:                                                                         # → 내리막
        profile = np.where(u > length + top, rise - (u - length - top) / max(length, 1e-9) * rise, profile)
    # 옆 기울기: 가운데 줄은 그대로, 왼쪽(+v)은 높게·오른쪽은 낮게 (땅보다 낮게 파이려면 mode: add).
    # 들어가고 나올 때 0.6 m에 걸쳐 기울어짐 (턱이 안 생기게)
    cross = math.tan(math.radians(f.get("cross_slope", 0.0)))
    ease = np.clip(u / 0.6, 0, 1) * np.clip((span - u) / 0.6, 0, 1)
    return np.where(_inside(u, v, 0.0, span, half_w), profile + cross * v * ease, 0.0)


def _stones(f, u, v, **_):
    lu, lv = f["area"]
    stone, gap = float(f.get("stone", 0.15)), float(f.get("gap", 0.04))
    pitch = stone + gap
    rng = np.random.default_rng(f.get("seed", 0))
    nu, nv = int(lu // pitch), int(lv // pitch)
    tops = float(f["height"]) + rng.uniform(-1, 1, (nu, nv)) * float(f.get("height_jitter", 0.0))
    iu, iv = np.floor(u / pitch), np.floor((v + lv / 2) / pitch)
    in_stone = ((u - iu * pitch) < stone) & (((v + lv / 2) - iv * pitch) < stone)
    ok = (iu >= 0) & (iu < nu) & (iv >= 0) & (iv < nv) & in_stone
    return np.where(ok, tops[np.clip(iu, 0, nu - 1).astype(int), np.clip(iv, 0, nv - 1).astype(int)], 0.0)


def _gravel(f, u, v, res, **_):
    lu, lv = f["area"]
    cell = float(f.get("cell", 0.06))
    rng = np.random.default_rng(f.get("seed", 0))
    nu, nv = int(math.ceil(lu / cell)) + 1, int(math.ceil(lv / cell)) + 1
    bumps = rng.uniform(0, 1, (nu, nv)) ** 2 * float(f["height"])          # 낮은 자갈이 더 많게
    iu, iv = np.floor(u / cell), np.floor((v + lv / 2) / cell)
    ok = (u >= 0) & (u < lu) & (np.abs(v) <= lv / 2)
    return np.where(ok, bumps[np.clip(iu, 0, nu - 1).astype(int), np.clip(iv, 0, nv - 1).astype(int)], 0.0)


def _image(f, u, v, res, base_dir, **_):
    img = _read_png(Path(base_dir) / f["file"])
    rows, cols = img.shape
    lu, lv = f.get("size", (rows * res, cols * res))
    r = np.floor((1 - u / lu) * rows).astype(int)          # 그림 위 = 앞(+u). u = 0이 그림 아래쪽
    c = np.floor((0.5 - v / lv) * cols).astype(int)        # 그림 왼쪽 = 왼쪽(+v)
    ok = (u >= 0) & (u < lu) & (np.abs(v) < lv / 2)
    return np.where(ok, img[np.clip(r, 0, rows - 1), np.clip(c, 0, cols - 1)] * float(f.get("height", 0.03)), 0.0)


def _read_png(path: Path) -> np.ndarray:
    """흑백(또는 컬러 → 밝기) 그림 → 0(검정) ~ 1(흰색)."""
    from PIL import Image
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.float32) / 255.0


FEATURES = {"box": _box, "bump": _bump, "stairs": _stairs, "ramp": _ramp, "stones": _stones, "gravel": _gravel,
            "image": _image}


# ---------------------------------------------------------------------------- 미리 보기
def preview(terrain: Terrain, out: Path, title: str | None = None) -> Path:
    """위에서 본 코스 그림 (높이 색 + 그림자) — 출발 자리와 방향 표시. MuJoCo 창 '7 위에서'와 같은 방향 (위 = +x)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource

    from matplotlib import font_manager
    for f in font_manager.findSystemFonts():                  # 한글 글꼴 (없으면 기본 글꼴, 글자가 네모로 보임)
        if "NanumGothic" in f or "NotoSansCJK" in f:
            font_manager.fontManager.addfont(f)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False

    h = terrain.heights * 100                                 # [cm]
    img = np.rot90(h, k=1)[:, ::-1]                           # 행 y, 열 x → 위 = +x, 왼쪽 = +y
    x0, x1 = terrain.center[0] - terrain.size_x, terrain.center[0] + terrain.size_x
    y0, y1 = terrain.center[1] - terrain.size_y, terrain.center[1] + terrain.size_y
    aspect = (x1 - x0) / (y1 - y0)                             # 긴 코스도 화면에 들어오게 높이 9인치까지
    fig, ax = plt.subplots(figsize=(max(4.0, 8.0 / aspect + 2.0), 9.0) if aspect > 1.2 else (7.5, 7.5 * aspect))
    lo, hi = min(h.min(), 0.0), max(h.max(), 0.5)
    shaded = LightSource(azdeg=315, altdeg=35).shade(img, cmap=plt.cm.viridis, vert_exag=40, blend_mode="soft",
                                                      vmin=lo, vmax=hi)
    ax.imshow(shaded, extent=(y1, y0, x0, x1), origin="upper")
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(lo, hi))
    fig.colorbar(sm, ax=ax, fraction=0.04, label="높이 [cm]")
    ax.annotate("", xy=(0, 0.5), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="red", lw=2))
    ax.plot(0, 0, "o", color="red")
    ax.set_xlabel("y [m] (왼쪽 +)")
    ax.set_ylabel("x [m] (출발할 때 앞)")
    ax.set_title(title or f"{terrain.kind} (빨간 점 = 출발, 화살표 = 앞)")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def main():
    import argparse
    parser = argparse.ArgumentParser(description="코스 파일 → 위에서 본 그림 (고칠 때 확인용)")
    parser.add_argument("course", help="코스 이름(models/courses), .yaml 또는 .png 경로")
    parser.add_argument("--out", type=Path, help="저장할 그림 (기본: output/courses/<이름>.png)")
    args = parser.parse_args()
    terrain = load_course(args.course)
    out = args.out or (paths.OUTPUT_DIR / "courses" / f"{Path(args.course).stem}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    h = terrain.heights
    print(f"{terrain.kind}: {2 * terrain.size_x:.1f} × {2 * terrain.size_y:.1f} m, 격자 {terrain.ncol} × {terrain.nrow}, "
          f"높이 {h.min() * 100:.1f} ~ {h.max() * 100:.1f} cm → {preview(terrain, out)}")


if __name__ == "__main__":
    main()
