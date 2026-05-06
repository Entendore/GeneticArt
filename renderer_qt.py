"""QPainter-based fractal renderer for the live canvas."""

from __future__ import annotations

import math
import random
from typing import List, Tuple

import numpy as np
from PySide6.QtCore import QPointF
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPen

from colors import harmonic_color
from fractal_base import draw_fractal_recursive
from genetics import Gene, gene_seed


def draw_gradient_qp(painter: QPainter, image_size: Tuple[int, int],
                     color_top: np.ndarray, color_bottom: np.ndarray) -> None:
    """Paint a vertical gradient background."""
    w, h = image_size
    grad = QLinearGradient(0, 0, 0, h)
    grad.setColorAt(0, QColor(*(int(c * 255) for c in color_top)))
    grad.setColorAt(1, QColor(*(int(c * 255) for c in color_bottom)))
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
    """Draw one fractal tree with optional swirl transformation."""
    w, h = image_size
    cx, cy = w / 2.0, float(h)
    cos_s, sin_s = math.cos(swirl), math.sin(swirl)
    rng = random.Random() if shimmer else random.Random(gene_seed(gene))

    def _line(x1, y1, x2, y2, col, a, lw):
        a = min(1.0, max(0.0, alpha * a))
        pen = QPen(QColor(int(col[0] * 255), int(col[1] * 255),
                          int(col[2] * 255), int(a * 255)))
        pen.setWidth(lw)
        pen.setCapStyle(Qt.RoundCap)  # type: ignore[attr-defined]
        painter.setPen(pen)
        # swirl around base
        dx1, dy1 = x1 - cx, y1 - cy
        dx2, dy2 = x2 - cx, y2 - cy
        painter.drawLine(
            QPointF(cx + dx1 * cos_s - dy1 * sin_s,
                    cy + dx1 * sin_s + dy1 * cos_s),
            QPointF(cx + dx2 * cos_s - dy2 * sin_s,
                    cy + dx2 * sin_s + dy2 * cos_s),
        )

    draw_fractal_recursive(
        gene["x"], gene["y"], gene["length"], gene["angle"],
        gene["depth"], gene["branch_factor"], gene["color_offset"],
        gene["curvature"], global_time, perlin_scale, _line, rng,
    )