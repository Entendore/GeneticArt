"""Core recursive fractal-tree logic with palette and particle support."""

from __future__ import annotations

import random
from typing import Callable, Optional

import numpy as np

from color_utils import pnoise1, get_palette_color_fn, ColorFn

DrawLineFn = Callable[[float, float, float, float, np.ndarray, float, int], None]
DrawParticleFn = Callable[[float, float, np.ndarray, float], None]

MAX_DEPTH = 8
MIN_LENGTH = 1.5
LENGTH_DECAY_MIN = 0.55
LENGTH_DECAY_MAX = 0.85
ANGLE_SPREAD = 0.6


def draw_fractal_recursive(
    x: float, y: float, length: float, angle: float,
    depth: int, branch_factor: int, color_offset: float,
    curvature: float, global_time: float, perlin_scale: float,
    draw_line_fn: DrawLineFn,
    rng: random.Random,
    color_fn: Optional[ColorFn] = None,
    draw_particle_fn: Optional[DrawParticleFn] = None,
) -> None:
    if depth <= 0 or length < MIN_LENGTH:
        return

    if color_fn is None:
        color_fn = get_palette_color_fn("Harmonic")

    noise_val = pnoise1(global_time + x * perlin_scale + y * perlin_scale * 0.5, repeat=1024)
    angle_mod = angle + noise_val * 0.4 + curvature

    x2 = x + np.cos(angle_mod) * length
    y2 = y + np.sin(angle_mod) * length

    depth_ratio = depth / MAX_DEPTH
    colour = color_fn(int(color_offset * 1000), 1000, global_time * 0.01)
    alpha = 0.1 + 0.9 * depth_ratio
    line_width = max(1, int(depth * 1.2))

    draw_line_fn(x, y, x2, y2, colour, alpha, line_width)

    # Leaf particles at terminal branches
    if depth == 1 and draw_particle_fn is not None:
        draw_particle_fn(x2, y2, colour, alpha)

    for i in range(branch_factor):
        offset = (i - (branch_factor - 1) / 2.0) * (ANGLE_SPREAD / max(1, branch_factor - 1))
        child_angle = angle_mod + offset + rng.uniform(-0.15, 0.15)
        decay = rng.uniform(LENGTH_DECAY_MIN, LENGTH_DECAY_MAX)
        child_length = length * decay
        child_color = color_offset + i * 0.04

        draw_fractal_recursive(
            x2, y2, child_length, child_angle, depth - 1,
            branch_factor, child_color, curvature * 0.9,
            global_time, perlin_scale, draw_line_fn, rng,
            color_fn, draw_particle_fn,
        )


def draw_fractal_recursive_fast(
    x: float, y: float, length: float, angle: float,
    depth: int, branch_factor: int, color_offset: float,
    curvature: float, global_time: float, perlin_scale: float,
    draw_line_fn: DrawLineFn,
    color_fn: Optional[ColorFn] = None,
    draw_particle_fn: Optional[DrawParticleFn] = None,
) -> None:
    if depth <= 0 or length < MIN_LENGTH:
        return

    if color_fn is None:
        color_fn = get_palette_color_fn("Harmonic")

    noise_val = pnoise1(global_time + x * perlin_scale, repeat=1024)
    angle_mod = angle + noise_val * 0.4 + curvature

    x2 = x + np.cos(angle_mod) * length
    y2 = y + np.sin(angle_mod) * length

    depth_ratio = depth / MAX_DEPTH
    colour = color_fn(int(color_offset * 1000), 1000, global_time * 0.01)
    alpha = 0.1 + 0.9 * depth_ratio
    line_width = max(1, int(depth * 1.2))

    draw_line_fn(x, y, x2, y2, colour, alpha, line_width)

    if depth == 1 and draw_particle_fn is not None:
        draw_particle_fn(x2, y2, colour, alpha)

    for i in range(branch_factor):
        offset = (i - (branch_factor - 1) / 2.0) * (ANGLE_SPREAD / max(1, branch_factor - 1))
        child_angle = angle_mod + offset
        child_length = length * 0.7
        child_color = color_offset + i * 0.04

        draw_fractal_recursive_fast(
            x2, y2, child_length, child_angle, depth - 1,
            branch_factor, child_color, curvature * 0.9,
            global_time, perlin_scale, draw_line_fn,
            color_fn, draw_particle_fn,
        )