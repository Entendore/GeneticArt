"""Harmonic colour palette generation."""

import math

import numpy as np


def harmonic_color(index: int, total: int, t_shift: float = 0.0) -> np.ndarray:
    """Return an RGB array in [0, 1] cycling through hues harmonically.

    Parameters
    ----------
    index : int
        Position in the palette.
    total : int
        Total palette slots.
    t_shift : float
        Time-based hue offset for animated colour cycling.
    """
    hue = (index / total + t_shift) % 1.0
    r = (math.sin(2.0 * math.pi * hue) + 1.0) / 2.0
    g = (math.sin(2.0 * math.pi * hue + 2.0 * math.pi / 3.0) + 1.0) / 2.0
    b = (math.sin(2.0 * math.pi * hue + 4.0 * math.pi / 3.0) + 1.0) / 2.0
    return np.clip([r, g, b], 0.0, 1.0)