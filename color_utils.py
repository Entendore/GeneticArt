"""Harmonic colour palette generation, Perlin noise, and palette presets."""

from __future__ import annotations

import math
import random
from typing import Callable, List, Tuple

import numpy as np


# ── Perlin noise ──────────────────────────────────────────────────────────────

try:
    from noise import pnoise1, pnoise2
except ImportError:
    _perm = list(range(256))
    random.shuffle(_perm)
    _perm = _perm + _perm
    _grad1 = [random.uniform(-1.0, 1.0) for _ in range(512)]
    _grad2 = [(random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)) for _ in range(512)]

    def _fade(t: float) -> float:
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

    def _lerp(a: float, b: float, t: float) -> float:
        return a + t * (b - a)

    def pnoise1(x: float, octaves: int = 1, persistence: float = 0.5,
                lacunarity: float = 2.0, repeat: int = 1024, base: int = 0) -> float:
        x = x % repeat
        xi = int(math.floor(x)) & 255
        xf = x - math.floor(x)
        u = _fade(xf)
        a, b = _perm[xi], _perm[xi + 1]
        return _lerp(_grad1[a], _grad1[b], u)

    def pnoise2(x: float, y: float, octaves: int = 1, persistence: float = 0.5,
                lacunarity: float = 2.0, repeatx: int = 1024, repeaty: int = 1024,
                base: int = 0) -> float:
        xi = int(math.floor(x)) & 255
        yi = int(math.floor(y)) & 255
        xf = x - math.floor(x)
        yf = y - math.floor(y)
        u, v = _fade(xf), _fade(yf)
        aa = _perm[_perm[xi] + yi]
        ab = _perm[_perm[xi] + yi + 1]
        ba = _perm[_perm[xi + 1] + yi]
        bb = _perm[_perm[xi + 1] + yi + 1]
        def _dot(g, dx, dy): return g[0] * dx + g[1] * dy
        x1 = _lerp(_dot(_grad2[aa], xf, yf), _dot(_grad2[ba], xf - 1, yf), u)
        x2 = _lerp(_dot(_grad2[ab], xf, yf - 1), _dot(_grad2[bb], xf - 1, yf - 1), u)
        return _lerp(x1, x2, v)


# ── HSV / RGB ─────────────────────────────────────────────────────────────────

def hsv_to_rgb(h: float, s: float, v: float) -> Tuple[float, float, float]:
    if s == 0.0:
        return v, v, v
    i = int(h * 6.0)
    f = (h * 6.0) - i
    p, q, t = v * (1.0 - s), v * (1.0 - s * f), v * (1.0 - s * (1.0 - f))
    i %= 6
    return [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i]


# ── Palette preset functions ──────────────────────────────────────────────────

ColorFn = Callable[[int, int, float], np.ndarray]


def _palette_harmonic(index: int, total: int, t_shift: float = 0.0) -> np.ndarray:
    hue = ((index / max(1, total)) + t_shift) % 1.0
    sat = 0.7 + 0.3 * math.sin(2.0 * math.pi * hue * 2)
    val = 0.8 + 0.2 * math.cos(2.0 * math.pi * hue * 3)
    r, g, b = hsv_to_rgb(hue, sat, val)
    return np.clip([r, g, b], 0.0, 1.0)


def _palette_complementary(index: int, total: int, t_shift: float = 0.0) -> np.ndarray:
    t = ((index / max(1, total)) + t_shift) % 1.0
    hue = (0.0 if t < 0.5 else 0.5) + t * 0.15
    r, g, b = hsv_to_rgb(hue % 1.0, 0.8, 0.9)
    return np.clip([r, g, b], 0.0, 1.0)


def _palette_analogous(index: int, total: int, t_shift: float = 0.0) -> np.ndarray:
    t = ((index / max(1, total)) + t_shift) % 1.0
    hue = 0.55 + t * 0.15
    sat = 0.7 + 0.2 * math.sin(t * math.pi)
    val = 0.7 + 0.3 * math.cos(t * math.pi * 0.5)
    r, g, b = hsv_to_rgb(hue % 1.0, sat, val)
    return np.clip([r, g, b], 0.0, 1.0)


def _palette_triadic(index: int, total: int, t_shift: float = 0.0) -> np.ndarray:
    t = ((index / max(1, total)) + t_shift) % 1.0
    sector = int(t * 3) % 3
    hue = (sector / 3.0) + (t * 3 % 1.0) * 0.08
    r, g, b = hsv_to_rgb(hue % 1.0, 0.85, 0.9)
    return np.clip([r, g, b], 0.0, 1.0)


def _palette_monochrome(index: int, total: int, t_shift: float = 0.0) -> np.ndarray:
    t = ((index / max(1, total)) + t_shift) % 1.0
    r, g, b = hsv_to_rgb(0.58, 0.3 + 0.5 * t, 0.5 + 0.5 * (1.0 - t * 0.5))
    return np.clip([r, g, b], 0.0, 1.0)


def _palette_fire(index: int, total: int, t_shift: float = 0.0) -> np.ndarray:
    t = ((index / max(1, total)) + t_shift) % 1.0
    r = min(1.0, 0.5 + t * 0.8)
    g = max(0.0, min(1.0, t * 1.2 - 0.1))
    b = max(0.0, min(1.0, (t - 0.8) * 3.0))
    return np.clip([r, g, b], 0.0, 1.0)


def _palette_ocean(index: int, total: int, t_shift: float = 0.0) -> np.ndarray:
    t = ((index / max(1, total)) + t_shift) % 1.0
    r = max(0.0, min(1.0, 0.05 + t * 0.3))
    g = max(0.0, min(1.0, 0.15 + t * 0.55))
    b = max(0.0, min(1.0, 0.45 + t * 0.55))
    return np.clip([r, g, b], 0.0, 1.0)


def _palette_neon(index: int, total: int, t_shift: float = 0.0) -> np.ndarray:
    hue = ((index / max(1, total)) + t_shift) % 1.0
    r, g, b = hsv_to_rgb(hue, 1.0, 1.0)
    return np.clip([r, g, b], 0.0, 1.0)


def _palette_pastel(index: int, total: int, t_shift: float = 0.0) -> np.ndarray:
    hue = ((index / max(1, total)) + t_shift) % 1.0
    r, g, b = hsv_to_rgb(hue, 0.4, 0.95)
    return np.clip([r, g, b], 0.0, 1.0)


_PALETTE_MAP: dict[str, ColorFn] = {
    "Harmonic": _palette_harmonic,
    "Complementary": _palette_complementary,
    "Analogous": _palette_analogous,
    "Triadic": _palette_triadic,
    "Monochrome": _palette_monochrome,
    "Fire": _palette_fire,
    "Ocean": _palette_ocean,
    "Neon": _palette_neon,
    "Pastel": _palette_pastel,
}


def get_palette_color_fn(name: str) -> ColorFn:
    return _PALETTE_MAP.get(name, _palette_harmonic)


# ── Backward-compatible alias ─────────────────────────────────────────────────

harmonic_color = _palette_harmonic


# ── Colour blending utilities ─────────────────────────────────────────────────

def blend_colors(colors: List[np.ndarray], weights: List[float] | None = None) -> np.ndarray:
    if not colors:
        return np.array([0.0, 0.0, 0.0])
    if weights is None:
        weights = [1.0 / len(colors)] * len(colors)
    result = sum(c * w for c, w in zip(colors, weights))
    return result / max(1e-6, sum(weights))


def average_colour(individuals: list, global_time: float,
                   palette_name: str = "Harmonic") -> np.ndarray:
    color_fn = get_palette_color_fn(palette_name)
    acc, n = np.zeros(3), 0
    for ind in individuals:
        for gene in ind:
            acc += color_fn(int(gene["color_offset"] * 1000), 1000, global_time * 0.01)
            n += 1
    return acc / max(1, n)


def gradient_colors(top: np.ndarray, bottom: np.ndarray, steps: int) -> List[np.ndarray]:
    return [np.clip(bottom * (1.0 - t) + top * t, 0.0, 1.0)
            for t in (i / max(1, steps - 1) for i in range(steps))]