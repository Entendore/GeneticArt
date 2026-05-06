"""Harmonic colour palette generation and Perlin noise utilities."""

from __future__ import annotations

import math
import random
from typing import List, Tuple

import numpy as np


# ── Perlin noise ──────────────────────────────────────────────────────────────

try:
    from noise import pnoise1  # type: ignore[import-untyped]
except ImportError:
    # Pure-Python fallback
    _perm = list(range(256))
    random.shuffle(_perm)
    _perm = _perm + _perm
    _grad = [random.uniform(-1.0, 1.0) for _ in range(512)]

    def _fade(t: float) -> float:
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

    def _lerp(a: float, b: float, t: float) -> float:
        return a + t * (b - a)

    def pnoise1(
        x: float,
        octaves: int = 1,
        persistence: float = 0.5,
        lacunarity: float = 2.0,
        repeat: int = 1024,
        base: int = 0,
    ) -> float:
        """Pure-Python 1D Perlin noise fallback."""
        x = x % repeat
        xi = int(math.floor(x)) & 255
        xf = x - math.floor(x)
        u = _fade(xf)
        a = _perm[xi]
        b = _perm[xi + 1]
        return _lerp(_grad[a], _grad[b], u)


# ── Colour utilities ──────────────────────────────────────────────────────────

def hsv_to_rgb(h: float, s: float, v: float) -> Tuple[float, float, float]:
    """Convert HSV (all in [0,1]) to RGB (in [0,1])."""
    if s == 0.0:
        return v, v, v
    i = int(h * 6.0)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i %= 6
    if i == 0: return v, t, p
    if i == 1: return q, v, p
    if i == 2: return p, v, t
    if i == 3: return p, q, v
    if i == 4: return t, p, v
    return v, p, q


def harmonic_color(index: int, total: int, t_shift: float = 0.0) -> np.ndarray:
    """Generate a harmonic RGB colour from index position with time shift.

    Cycles through hues using sinusoidal interpolation for smooth
    colour transitions.

    Parameters
    ----------
    index : int
        Position in the palette.
    total : int
        Total palette slots.
    t_shift : float
        Time-based hue offset for animated colour cycling.

    Returns
    -------
    np.ndarray
        RGB values in [0, 1].
    """
    hue = ((index / max(1, total)) + t_shift) % 1.0
    saturation = 0.7 + 0.3 * math.sin(2.0 * math.pi * hue * 2)
    value = 0.8 + 0.2 * math.cos(2.0 * math.pi * hue * 3)
    r, g, b = hsv_to_rgb(hue, saturation, value)
    return np.clip([r, g, b], 0.0, 1.0)


def blend_colors(colors: List[np.ndarray], weights: List[float] | None = None) -> np.ndarray:
    """Blend multiple RGB colours with optional weights."""
    if not colors:
        return np.array([0.0, 0.0, 0.0])
    if weights is None:
        weights = [1.0 / len(colors)] * len(colors)
    result = np.zeros(3)
    total_weight = sum(weights)
    for c, w in zip(colors, weights):
        result += c * w
    return result / max(1e-6, total_weight)


def average_colour(individuals: list, global_time: float) -> np.ndarray:
    """Compute mean harmonic colour across all genes in all individuals."""
    acc = np.zeros(3)
    n = 0
    for ind in individuals:
        for gene in ind:
            acc += harmonic_color(
                int(gene["color_offset"] * 1000), 1000, global_time * 0.01
            )
            n += 1
    return acc / max(1, n)


def gradient_colors(
    top_color: np.ndarray,
    bottom_color: np.ndarray,
    steps: int
) -> List[np.ndarray]:
    """Generate a list of colours for a vertical gradient."""
    colors = []
    for i in range(steps):
        t = i / max(1, steps - 1)
        c = bottom_color * (1.0 - t) + top_color * t
        colors.append(np.clip(c, 0.0, 1.0))
    return colors