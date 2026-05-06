"""Core recursive fractal-tree logic shared by all renderers.

The actual drawing primitive is injected via *draw_line_fn* so the same
recursion works with PIL, QPainter, or any other backend.
"""

from __future__ import annotations

import random
from typing import Callable

import numpy as np

from colors import harmonic_color
from noise_utils import pnoise1

# Callback signature:
#   (x1, y1, x2, y2, colour_rgb_01, alpha_01, line_width) -> None
DrawLineFn = Callable[[float, float, float, float, np.ndarray, float, int], None]


def draw_fractal_recursive(
    x: float,
    y: float,
    length: float,
    angle: float,
    depth: int,
    branch_factor: int,
    color_offset: float,
    curvature: float,
    global_time: float,
    perlin_scale: float,
    draw_line_fn: DrawLineFn,
    rng: random.Random,
) -> None:
    """Recursively draw a single fractal tree.

    Parameters
    ----------
    x, y : float
        Root position.
    length, angle, depth, branch_factor, color_offset, curvature
        Gene-driven parameters.
    global_time, perlin_scale
        Control Perlin-noise perturbation.
    draw_line_fn
        Backend-specific function that paints one line segment.
    rng : random.Random
        Random instance (seeded for stable, fresh for shimmer).
    """
    if depth <= 0 or length < 2:
        return

    angle_mod = (
        angle
        + pnoise1(global_time + x * perlin_scale, repeat=1024) * 0.5
        + curvature
    )
    x2 = x + np.cos(angle_mod) * length
    y2 = y + np.sin(angle_mod) * length

    colour = harmonic_color(int(color_offset * 1000), 1000, global_time * 0.01)
    alpha = 0.15 + 0.85 * (depth / 6.0)
    draw_line_fn(x, y, x2, y2, colour, alpha, max(1, depth))

    for i in range(branch_factor):
        draw_fractal_recursive(
            x2, y2,
            length * rng.uniform(0.6, 0.8),
            angle_mod + rng.uniform(-0.5, 0.5),
            depth - 1,
            branch_factor,
            color_offset + i * 0.05,
            curvature,
            global_time,
            perlin_scale,
            draw_line_fn,
            rng,
        )