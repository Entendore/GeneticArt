"""Core recursive fractal-tree logic shared by all renderers.

The actual drawing primitive is injected via *draw_line_fn* so the same
recursion works with PIL, QPainter, or any other backend.
"""

from __future__ import annotations

import random
from typing import Callable

import numpy as np

from color_utils import harmonic_color, pnoise1

# Callback signature:
#   (x1, y1, x2, y2, colour_rgb_01, alpha_01, line_width) -> None
DrawLineFn = Callable[[float, float, float, float, np.ndarray, float, int], None]


# ── Configuration ─────────────────────────────────────────────────────────────

MAX_DEPTH = 8
MIN_LENGTH = 1.5
LENGTH_DECAY_MIN = 0.55
LENGTH_DECAY_MAX = 0.85
ANGLE_SPREAD = 0.6


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
        Root position in image coordinates.
    length : float
        Initial branch length in pixels.
    angle : float
        Initial branch angle in radians.
    depth : int
        Maximum recursion depth.
    branch_factor : int
        Number of child branches per node.
    color_offset : float
        Hue offset for colour variation.
    curvature : float
        Base curvature applied to branches.
    global_time : float
        Animation time parameter.
    perlin_scale : float
        Scale for Perlin noise perturbation.
    draw_line_fn : DrawLineFn
        Backend-specific line drawing callback.
    rng : random.Random
        Random instance for stochastic variation.
    """
    if depth <= 0 or length < MIN_LENGTH:
        return

    # Apply Perlin noise for organic movement
    noise_val = pnoise1(global_time + x * perlin_scale + y * perlin_scale * 0.5, repeat=1024)
    angle_mod = angle + noise_val * 0.4 + curvature

    # Calculate end point
    x2 = x + np.cos(angle_mod) * length
    y2 = y + np.sin(angle_mod) * length

    # Compute visual properties
    depth_ratio = depth / MAX_DEPTH
    colour = harmonic_color(int(color_offset * 1000), 1000, global_time * 0.01)
    alpha = 0.1 + 0.9 * depth_ratio
    line_width = max(1, int(depth * 1.2))

    draw_line_fn(x, y, x2, y2, colour, alpha, line_width)

    # Recurse into child branches
    for i in range(branch_factor):
        # Staggered angle distribution
        offset = (i - (branch_factor - 1) / 2.0) * (ANGLE_SPREAD / max(1, branch_factor - 1))
        child_angle = angle_mod + offset + rng.uniform(-0.15, 0.15)

        # Variable length decay for natural appearance
        decay = rng.uniform(LENGTH_DECAY_MIN, LENGTH_DECAY_MAX)
        child_length = length * decay

        # Slightly shift colour for each branch
        child_color = color_offset + i * 0.04

        draw_fractal_recursive(
            x2,
            y2,
            child_length,
            child_angle,
            depth - 1,
            branch_factor,
            child_color,
            curvature * 0.9,  # Reduce curvature with depth
            global_time,
            perlin_scale,
            draw_line_fn,
            rng,
        )


def draw_fractal_recursive_fast(
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
) -> None:
    """Fast variant without random perturbation (deterministic rendering)."""
    if depth <= 0 or length < MIN_LENGTH:
        return

    noise_val = pnoise1(global_time + x * perlin_scale, repeat=1024)
    angle_mod = angle + noise_val * 0.4 + curvature

    x2 = x + np.cos(angle_mod) * length
    y2 = y + np.sin(angle_mod) * length

    depth_ratio = depth / MAX_DEPTH
    colour = harmonic_color(int(color_offset * 1000), 1000, global_time * 0.01)
    alpha = 0.1 + 0.9 * depth_ratio
    line_width = max(1, int(depth * 1.2))

    draw_line_fn(x, y, x2, y2, colour, alpha, line_width)

    for i in range(branch_factor):
        offset = (i - (branch_factor - 1) / 2.0) * (ANGLE_SPREAD / max(1, branch_factor - 1))
        child_angle = angle_mod + offset
        child_length = length * 0.7  # Fixed decay
        child_color = color_offset + i * 0.04

        draw_fractal_recursive_fast(
            x2, y2, child_length, child_angle, depth - 1,
            branch_factor, child_color, curvature * 0.9,
            global_time, perlin_scale, draw_line_fn,
        )