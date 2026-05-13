"""Linear and eased interpolation of genes and individuals."""

from __future__ import annotations

import math
from typing import List

from genetics import Gene


# ── Easing functions ──────────────────────────────────────────────────────────

def ease_linear(t: float) -> float:
    return t


def ease_smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def ease_cubic(t: float) -> float:
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        return 4.0 * t * t * t
    return 1.0 - pow(-2.0 * t + 2.0, 3) / 2.0


def ease_in_out(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return -(math.cos(math.pi * t) - 1.0) / 2.0


_EASING_MAP = {
    "linear": ease_linear,
    "smoothstep": ease_smoothstep,
    "cubic": ease_cubic,
    "ease_in_out": ease_in_out,
}


def apply_easing(t: float, mode: str = "smoothstep") -> float:
    fn = _EASING_MAP.get(mode, ease_smoothstep)
    return fn(t)


# ── Gene / individual interpolation ───────────────────────────────────────────

def interpolate_gene(g1: Gene, g2: Gene, t: float) -> Gene:
    result: Gene = {}
    for key in g1:
        v1, v2 = g1[key], g2[key]
        if isinstance(v1, float):
            result[key] = v1 * (1.0 - t) + v2 * t
        else:
            result[key] = int(v1 * (1.0 - t) + v2 * t)
    return result


def interpolate_individual(
    ind1: List[Gene], ind2: List[Gene], t: float, easing: str = "smoothstep"
) -> List[Gene]:
    t_eased = apply_easing(t, easing)
    return [interpolate_gene(g1, g2, t_eased) for g1, g2 in zip(ind1, ind2)]