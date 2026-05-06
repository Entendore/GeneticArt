"""PIL-based fractal renderer for file export (screenshots & frame sequences)."""

from __future__ import annotations

import math
import random
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw

from colors import harmonic_color
from fractal_base import draw_fractal_recursive
from genetics import Gene, gene_seed


def render_individual_pil(
    individual: List[Gene],
    image_size: Tuple[int, int],
    global_time: float = 0.0,
    perlin_scale: float = 0.08,
) -> Image.Image:
    """Render an individual to a PIL RGBA image (deterministic, no swirl)."""
    w, h = image_size
    img = Image.new("RGBA", image_size, (0, 0, 0, 255))
    draw = ImageDraw.Draw(img, "RGBA")

    for gene in individual:
        rng = random.Random(gene_seed(gene))

        def _line(x1, y1, x2, y2, col, alpha, lw, _draw=draw):
            r, g, b = int(col[0] * 255), int(col[1] * 255), int(col[2] * 255)
            a = int(alpha * 255)
            _draw.line((int(x1), int(y1), int(x2), int(y2)),
                       fill=(r, g, b, a), width=lw)

        draw_fractal_recursive(
            gene["x"], gene["y"], gene["length"], gene["angle"],
            gene["depth"], gene["branch_factor"], gene["color_offset"],
            gene["curvature"], global_time, perlin_scale, _line, rng,
        )
    return img


def render_gradient_pil(
    image_size: Tuple[int, int],
    color_top: np.ndarray,
    color_bottom: np.ndarray,
) -> Image.Image:
    """Create a vertical-gradient PIL image."""
    w, h = image_size
    img = Image.new("RGB", image_size)
    px = img.load()
    for y in range(h):
        t = y / max(1, h - 1)
        c = color_bottom * (1.0 - t) + color_top * t
        c = tuple(int(max(0, min(255, v * 255))) for v in c)
        for x in range(w):
            px[x, y] = c
    return img


def average_colour(individuals: List[List[Gene]],
                   global_time: float) -> np.ndarray:
    """Compute mean harmonic colour across all genes."""
    acc = np.zeros(3)
    n = 0
    for ind in individuals:
        for gene in ind:
            acc += harmonic_color(int(gene["color_offset"] * 1000), 1000,
                                  global_time * 0.01)
            n += 1
    return acc / max(1, n)