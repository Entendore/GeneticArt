"""Fractal renderers for PIL export and Qt live display with palette and particle support."""

from __future__ import annotations

import math
import random
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import (
    QColor,
    QLinearGradient,
    QPainter,
    QPen,
    QRadialGradient,
    QPixmap,
    QImage,
)
from PySide6.QtWidgets import QLabel

from color_utils import get_palette_color_fn, average_colour, ColorFn
from fractal_base import draw_fractal_recursive, draw_fractal_recursive_fast
from genetics import Gene, Individual, gene_seed


# ── PIL Renderer ──────────────────────────────────────────────────────────────

def render_individual_pil(
    individual: Individual,
    image_size: Tuple[int, int],
    global_time: float = 0.0,
    perlin_scale: float = 0.08,
    palette_name: str = "Harmonic",
    show_leaves: bool = True,
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
    palette_name : str
        Name of the color palette preset to use.
    show_leaves : bool
        Whether to render leaf particles at branch tips.

    Returns
    -------
    Image.Image
        RGBA PIL image.
    """
    w, h = image_size
    img = Image.new("RGBA", image_size, (0, 0, 0, 255))
    draw = ImageDraw.Draw(img, "RGBA")
    color_fn = get_palette_color_fn(palette_name)

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

        def _particle(x, y, col, alpha, _draw=draw):
            if not show_leaves:
                return
            r = int(np.clip(col[0] * 255, 0, 255))
            g = int(np.clip(col[1] * 255, 0, 255))
            b = int(np.clip(col[2] * 255, 0, 255))
            a = int(np.clip(alpha * 200, 0, 255))
            sz = 2
            _draw.ellipse(
                (int(x - sz), int(y - sz), int(x + sz), int(y + sz)),
                fill=(r, g, b, a),
            )

        draw_fractal_recursive(
            gene["x"], gene["y"], gene["length"], gene["angle"],
            gene["depth"], gene["branch_factor"], gene["color_offset"],
            gene["curvature"], global_time, perlin_scale,
            _line, rng, color_fn, _particle,
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
    palette_name: str = "Harmonic",
    show_leaves: bool = True,
    show_glow: bool = True,
) -> None:
    """Draw one fractal tree with optional swirl transformation and particles.

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
    palette_name : str
        Name of the color palette preset to use.
    show_leaves : bool
        If True, draw leaf particles at branch tips.
    show_glow : bool
        If True, add glow effect at branch tips.
    """
    w, h = image_size
    cx, cy = w / 2.0, float(h)
    cos_s, sin_s = math.cos(swirl), math.sin(swirl)
    color_fn = get_palette_color_fn(palette_name)

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

    def _particle(x, y, col, a):
        if not show_leaves and not show_glow:
            return
        dx, dy = x - cx, y - cy
        tx = cx + dx * cos_s - dy * sin_s
        ty = cy + dx * sin_s + dy * cos_s

        r = int(np.clip(col[0] * 255, 0, 255))
        g = int(np.clip(col[1] * 255, 0, 255))
        b = int(np.clip(col[2] * 255, 0, 255))

        if show_glow:
            glow = QRadialGradient(tx, ty, 8)
            glow.setColorAt(0, QColor(r, g, b, int(a * 120)))
            glow.setColorAt(1, QColor(r, g, b, 0))
            painter.setPen(Qt.NoPen)
            painter.setBrush(glow)
            painter.drawEllipse(QPointF(tx, ty), 8, 8)

        if show_leaves:
            painter.setBrush(QColor(r, g, b, int(a * 200)))
            painter.setPen(
                QPen(
                    QColor(
                        min(255, r + 40),
                        min(255, g + 40),
                        min(255, b + 40),
                        int(a * 150),
                    ),
                    0.5,
                )
            )
            painter.drawEllipse(QPointF(tx, ty), 2.5, 2.5)

    draw_fractal_recursive(
        gene["x"], gene["y"], gene["length"], gene["angle"],
        gene["depth"], gene["branch_factor"], gene["color_offset"],
        gene["curvature"], global_time, perlin_scale,
        _line, rng, color_fn, _particle,
    )


# ── Thumbnail helper ──────────────────────────────────────────────────────────

def render_thumbnail_qpixmap(
    individual: Individual,
    size: int = 120,
    global_time: float = 0.0,
    perlin_scale: float = 0.08,
    palette_name: str = "Harmonic",
) -> QPixmap:
    """Render a small thumbnail QPixmap for gallery display."""
    img = render_individual_pil(
        individual, (size, size), global_time, perlin_scale,
        palette_name, show_leaves=True,
    )
    img_rgba = img.convert("RGBA")
    data = img_rgba.tobytes("raw", "RGBA")
    qimg = QImage(data, size, size, size * 4, QImage.Format_RGBA8888).copy()
    return QPixmap.fromImage(qimg)