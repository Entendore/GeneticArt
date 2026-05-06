"""Fractal renderers for both PIL export and Qt live display."""

from __future__ import annotations

import math
import random
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPen

from color_utils import harmonic_color, average_colour
from fractal_base import draw_fractal_recursive, draw_fractal_recursive_fast
from genetics import Gene, Individual, gene_seed


# ── PIL Renderer (for file export) ────────────────────────────────────────────

def render_individual_pil(
    individual: Individual,
    image_size: Tuple[int, int],
    global_time: float = 0.0,
    perlin_scale: float = 0.08,
) -> Image.Image:
    """Render an individual to a PIL RGBA image (deterministic, no swirl).

    Parameters
    ----------
    individual : Individual
        List of genes defining the fractal trees.
    image_size : Tuple[int, int]
        Output image dimensions.
    global_time : float
        Animation time parameter.
    perlin_scale : float
        Perlin noise scale.

    Returns
    -------
    Image.Image
        RGBA PIL image.
    """
    w, h = image_size
    img = Image.new("RGBA", image_size, (0, 0, 0, 255))
    draw = ImageDraw.Draw(img, "RGBA")

    for gene in individual:
        rng = random.Random(gene_seed(gene))

        def _line(x1, y1, x2, y2, col, alpha, lw, _draw=draw):
            r = int(np.clip(col[0] * 255, 0, 255))
            g = int(np.clip(col[1] * 255, 0, 255))
            b = int(np.clip(col[2] * 255, 0, 255))
            a = int(np.clip(alpha * 255, 0, 255))
            _draw.line(
                (int(x1), int(y1), int(x2), int(y2)),
                fill=(r, g, b, a),
                width=lw,
            )

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


# ── Qt Renderer (for live canvas) ────────────────────────────────────────────

def draw_gradient_qp(
    painter: QPainter,
    image_size: Tuple[int, int],
    color_top: np.ndarray,
    color_bottom: np.ndarray,
) -> None:
    """Paint a vertical gradient background using QPainter."""
    w, h = image_size
    grad = QLinearGradient(0, 0, 0, h)
    ct = QColor(*(int(np.clip(c * 255, 0, 255)) for c in color_top))
    cb = QColor(*(int(np.clip(c * 255, 0, 255)) for c in color_bottom))
    grad.setColorAt(0, ct)
    grad.setColorAt(1, cb)
    painter.fillRect(0, 0, w, h, grad)


def draw_fractal_qp(
    painter: QPainter,
    gene: Gene,
    global_time: float,
    image_size: Tuple[int, int],
    alpha: float = 1.0,
    swirl: float = 0.0,
    perlin_scale: float = 0.08,
    shimmer: bool = True,
) -> None:
    """Draw one fractal tree with optional swirl transformation.

    Parameters
    ----------
    painter : QPainter
        Active Qt painter.
    gene : Gene
        Fractal gene parameters.
    global_time : float
        Animation time.
    image_size : Tuple[int, int]
        Canvas dimensions.
    alpha : float
        Global opacity multiplier.
    swirl : float
        Rotation angle for swirl effect.
    perlin_scale : float
        Perlin noise scale.
    shimmer : bool
        If True, use fresh random each frame for shimmer effect.
    """
    w, h = image_size
    cx, cy = w / 2.0, float(h)
    cos_s, sin_s = math.cos(swirl), math.sin(swirl)

    if shimmer:
        rng = random.Random()
    else:
        rng = random.Random(gene_seed(gene))

    def _line(x1, y1, x2, y2, col, a, lw):
        a = min(1.0, max(0.0, alpha * a))
        r = int(np.clip(col[0] * 255, 0, 255))
        g = int(np.clip(col[1] * 255, 0, 255))
        b = int(np.clip(col[2] * 255, 0, 255))
        a_int = int(a * 255)

        pen = QPen(QColor(r, g, b, a_int))
        pen.setWidth(lw)
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)

        # Apply swirl transformation around base center
        dx1, dy1 = x1 - cx, y1 - cy
        dx2, dy2 = x2 - cx, y2 - cy

        p1 = QPointF(
            cx + dx1 * cos_s - dy1 * sin_s,
            cy + dx1 * sin_s + dy1 * cos_s,
        )
        p2 = QPointF(
            cx + dx2 * cos_s - dy2 * sin_s,
            cy + dx2 * sin_s + dy2 * cos_s,
        )
        painter.drawLine(p1, p2)

    draw_fractal_recursive(
        gene["x"], gene["y"], gene["length"], gene["angle"],
        gene["depth"], gene["branch_factor"], gene["color_offset"],
        gene["curvature"], global_time, perlin_scale, _line, rng,
    )