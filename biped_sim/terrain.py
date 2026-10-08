"""울퉁불퉁한 지형 — MuJoCo 높이맵(heightfield)으로 언덕, 불규칙한 경사로, 장애물 만들기.

    terrain = make_terrain("hills", seed=0, difficulty=0.5)          # 높이 격자 [m]
    model, _ = build_robot_model(urdf, SimConfig(terrain=terrain))   # 평평한 바닥 대신 이 지형
    terrain.height_at(x, y)                                          # 그 위치의 땅 높이 [m]

높이맵 = 일정한 간격(기본 4 cm)의 격자마다 높이 하나. MuJoCo는 이것을 삼각형 면으로 이어 충돌·화면에 쓴다.
⚠ MuJoCo는 모델을 만들 때 높이맵 데이터를 '가장 낮은 값 0, 가장 높은 값 1'로 다시 늘인다 (실측: 다른 비율로 넣었더니
   언덕이 2배로 높아짐). 그래서 높이를 [최저, 최고] → [0, 1]로 바꿔 넣고, 높이 범위(size z)와 바닥 위치(geom z)를 함께 정한다.
같은 격자 크기의 지형이면 실행 중에 apply()로 바꿔 끼울 수 있다 (terrain_trial.py가 에피소드마다 장애물을 바꾸는 방법).

지형의 크기는 오리 로봇(Open Duck Mini, 다리 약 0.17 m, 발 들기 약 2 cm)에 맞춘 값이 기본이다.
difficulty(보통 0~1, 한계 시험은 1 이상)로 높이·경사를 키운다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

KINDS = ("flat", "hills", "slopes", "obstacles", "mixed")


@dataclass
class Terrain:
    heights: np.ndarray          # [nrow, ncol] 땅 높이 [m]. 행 = y(−size_y → +size_y), 열 = x(−size_x → +size_x)
    size_x: float                # [m] 지형의 x 방향 반길이 (중심 기준)
    size_y: float                # [m] y 방향 반길이
    center: tuple[float, float] = (0.0, 0.0)   # 지형 중심의 world 좌표
    kind: str = "flat"

    @property
    def nrow(self) -> int:
        return self.heights.shape[0]

    @property
    def ncol(self) -> int:
        return self.heights.shape[1]

    @property
    def z_low(self) -> float:      # 가장 낮은 땅 = 높이맵 geom의 z 위치
        return float(self.heights.min())

    @property
    def z_range(self) -> float:    # 높이 범위 = hfield size z (0이면 MuJoCo가 거부하므로 최소 1 mm)
        return max(float(self.heights.max() - self.heights.min()), 0.001)

    def hfield_data(self) -> np.ndarray:
        """MuJoCo hfield_data (0~1): 높이 h ↔ (h − 최저) / 범위."""
        return ((self.heights - self.z_low) / self.z_range).astype(np.float32).ravel()

    def apply(self, model, hfield: str = "terrain", geom: str = "floor") -> None:
        """이미 만든 MuJoCo 모델의 지형을 이 지형으로 바꿔 끼운다 (격자 크기가 같아야 함).
        화면에도 반영하려면 그 뒤 viewer.update_hfield(model.hfield(hfield).id)."""
        h = model.hfield(hfield).id
        if (model.hfield_nrow[h], model.hfield_ncol[h]) != (self.nrow, self.ncol):
            raise ValueError("격자 크기가 다른 지형은 바꿔 끼울 수 없습니다")
        adr = model.hfield_adr[h]
        model.hfield_data[adr:adr + self.nrow * self.ncol] = self.hfield_data()
        model.hfield_size[h, 2] = self.z_range
        model.geom_pos[model.geom(geom).id, 2] = self.z_low

    def height_at(self, x, y):
        """world 좌표 (x, y)의 땅 높이 [m] (격자 사이는 선형 보간, 지형 밖은 가장자리 값). x, y는 배열이어도 됨."""
        col = (np.asarray(x) - self.center[0] + self.size_x) / (2 * self.size_x) * (self.ncol - 1)
        row = (np.asarray(y) - self.center[1] + self.size_y) / (2 * self.size_y) * (self.nrow - 1)
        col = np.clip(col, 0, self.ncol - 1.000001)
        row = np.clip(row, 0, self.nrow - 1.000001)
        c0, r0 = np.floor(col).astype(int), np.floor(row).astype(int)
        fc, fr = col - c0, row - r0
        h = self.heights
        return ((1 - fr) * ((1 - fc) * h[r0, c0] + fc * h[r0, c0 + 1])
                + fr * ((1 - fc) * h[r0 + 1, c0] + fc * h[r0 + 1, c0 + 1]))


def _grid(size_x, size_y, resolution):
    ncol = int(round(2 * size_x / resolution)) + 1
    nrow = int(round(2 * size_y / resolution)) + 1
    x = np.linspace(-size_x, size_x, ncol)
    y = np.linspace(-size_y, size_y, nrow)
    return np.meshgrid(x, y)   # 각각 [nrow, ncol]


def _smooth_noise(rng, shape, cells):
    """격자 칸 cells개 정도의 크기로 부드럽게 이어지는 무작위 높이 (평균 0, 최대 1)."""
    noise = rng.standard_normal(shape)
    fy, fx = np.fft.fftfreq(shape[0])[:, None], np.fft.fftfreq(shape[1])[None, :]
    smooth = np.real(np.fft.ifft2(np.fft.fft2(noise) * np.exp(-(fx ** 2 + fy ** 2) * (np.pi * cells) ** 2 / 2)))
    smooth -= smooth.mean()
    return smooth / (np.abs(smooth).max() + 1e-12)


def make_terrain(kind: str, seed: int = 0, difficulty: float = 0.5, size_x: float = 2.5, size_y: float = 1.0,
                 center: tuple[float, float] = (1.5, 0.0), resolution: float = 0.04,
                 start: tuple[float, float] | None = (0.0, 0.0), start_radius: float = 0.35) -> Terrain:
    """kind: flat / hills(산악 언덕) / slopes(불규칙한 경사로) / obstacles(장애물) / mixed(앞의 셋을 x 방향으로 이어 붙임).
    start: 이 위치 주변(start_radius)은 평평하게 (출발 자리). None이면 그대로.
    기본 크기: x −1.0 ~ 4.0 m, y −1 ~ 1 m (오리가 10초에 1.5 m 걷는 코스 + 여유)."""
    if kind not in KINDS:
        raise ValueError(f"모르는 지형 '{kind}' (사용 가능: {', '.join(KINDS)})")
    rng = np.random.default_rng(seed)
    gx, gy = _grid(size_x, size_y, resolution)
    d = max(float(difficulty), 0.0)   # 1보다 크게 하면 더 험하게 (한계 시험용)

    def hills():   # 파장 약 1 m의 부드러운 언덕, 높이 ±1~3 cm → 경사 최대 약 4~12°
        return (0.01 + 0.02 * d) * _smooth_noise(rng, gx.shape, cells=0.25 / resolution)

    def slopes():  # 0.4 m 칸마다 높이를 무작위로 정하고 그 사이를 평면으로 이음 → 칸마다 다른 기울기
        tile = 0.4
        nx, ny = int(np.ceil(2 * size_x / tile)) + 2, int(np.ceil(2 * size_y / tile)) + 2
        coarse = rng.uniform(-1, 1, (ny, nx)) * (0.008 + 0.022 * d)   # 이웃 칸 높이 차 → 경사 약 2~8°
        cx = (gx + size_x) / tile
        cy = (gy + size_y) / tile
        c0, r0 = np.floor(cx).astype(int), np.floor(cy).astype(int)
        fc, fr = cx - c0, cy - r0
        return ((1 - fr) * ((1 - fc) * coarse[r0, c0] + fc * coarse[r0, c0 + 1])
                + fr * ((1 - fc) * coarse[r0 + 1, c0] + fc * coarse[r0 + 1, c0 + 1]))

    def obstacles():  # 납작한 상자들: 높이 0.5~2 cm, 한 변 6~24 cm
        h = np.zeros_like(gx)
        n = int(round(2 * size_x * 2 * size_y * (4 + 6 * d)))       # 1 m²당 4~10개
        for _ in range(n):
            x0, y0 = rng.uniform(-size_x, size_x), rng.uniform(-size_y, size_y)
            wx, wy = rng.uniform(0.06, 0.24, 2) / 2
            top = rng.uniform(0.005, 0.005 + 0.015 * d)
            mask = (np.abs(gx - x0) < wx) & (np.abs(gy - y0) < wy)
            h[mask] = np.maximum(h[mask], top)
        return h

    if kind == "flat":
        heights = np.zeros_like(gx)
    elif kind == "hills":
        heights = hills()
    elif kind == "slopes":
        heights = slopes()
    elif kind == "obstacles":
        heights = obstacles()
    else:   # mixed: x를 세 구간으로 나눠 언덕 → 경사로 → 장애물, 경계는 부드럽게 섞음
        parts = [hills(), slopes(), obstacles()]
        edges = np.linspace(-size_x, size_x, 4)
        weights = [np.clip(1 - np.abs(gx - (edges[i] + edges[i + 1]) / 2) / ((edges[1] - edges[0]) * 0.6), 0, 1)
                   for i in range(3)]
        total = sum(weights) + 1e-9
        heights = sum(w * p for w, p in zip(weights, parts)) / total

    world_x, world_y = gx + center[0], gy + center[1]
    if start is not None:   # 출발 자리는 평평하게, 바깥으로 갈수록 원래 지형 (0.15 m에 걸쳐 섞음)
        r = np.hypot(world_x - start[0], world_y - start[1])
        blend = np.clip((r - start_radius) / 0.15, 0.0, 1.0)
        heights = heights * blend
    return Terrain(heights=heights, size_x=size_x, size_y=size_y, center=center, kind=kind)


def max_slope_deg(terrain: Terrain) -> float:
    """지형의 가장 가파른 경사 [°] (격자 이웃 사이)."""
    dx = 2 * terrain.size_x / (terrain.ncol - 1)
    gy, gx = np.gradient(terrain.heights, dx)
    return float(np.degrees(np.arctan(np.hypot(gx, gy).max())))


def make_training_terrain(seed: int = 0, half_size: float = 6.0, resolution: float = 0.04,
                          difficulty: tuple[float, float] = (0.4, 1.2)) -> Terrain:
    """학습용 넓은 지형 (기본 12 × 12 m): 약 1.5 m 크기의 구역마다 언덕·경사로·장애물·평지 중 하나가 주로 나오고,
    험한 정도도 곳마다 다르다 (시험 지형의 최대 1.0보다 조금 더 험한 1.2까지 — 시험보다 어렵게 연습).
    에피소드마다 무작위 위치에서 출발하면 (BipedWalkMjxEnv spawn_area) 여러 지형을 골고루 겪는다.
    (처음엔 네 지형을 평균으로 섞었더니 서로 상쇄돼 절반 넘게 거의 평지가 됨 → 구역마다 한 종류가 주도하게 바꿈)"""
    rng = np.random.default_rng(seed + 100)
    parts = [make_terrain(kind, seed=seed + i, difficulty=1.0, size_x=half_size, size_y=half_size,
                          center=(0.0, 0.0), resolution=resolution, start=None).heights
             for i, kind in enumerate(("hills", "slopes", "obstacles"))]
    parts.append(np.zeros_like(parts[0]))                                   # 평지 (출발·회복 연습용)
    shape = parts[0].shape
    cells = 1.5 / resolution                                                # 구역 크기 약 1.5 m
    scores = np.stack([_smooth_noise(rng, shape, cells) for _ in parts])
    scores[-1] -= 0.4                                                       # 평지는 덜 나오게
    weights = np.exp(8.0 * scores)                                          # 날카로운 softmax: 구역마다 한 종류가 주도
    weights /= weights.sum(axis=0)
    d_lo, d_hi = difficulty
    level = d_lo + (d_hi - d_lo) * (_smooth_noise(rng, shape, cells) + 1) / 2   # 곳마다 다른 험한 정도
    heights = level * np.einsum("kij,kij->ij", weights, np.stack(parts))
    return Terrain(heights=heights, size_x=half_size, size_y=half_size, center=(0.0, 0.0), kind="training")
