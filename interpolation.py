"""Linear interpolation of genes and individuals for smooth transitions."""

from __future__ import annotations

from typing import Dict, List

from genetics import Gene


def interpolate_gene(g1: Gene, g2: Gene, t: float) -> Gene:
    """Linearly blend every field of two genes by factor *t* ∈ [0, 1]."""
    result: Gene = {}
    for key in g1:
        if isinstance(g1[key], float):
            result[key] = g1[key] * (1.0 - t) + g2[key] * t
        else:
            result[key] = int(g1[key] * (1.0 - t) + g2[key] * t)
    return result


def interpolate_individual(ind1: List[Gene], ind2: List[Gene],
                           t: float) -> List[Gene]:
    """Interpolate two individuals (same-length gene lists)."""
    return [interpolate_gene(g1, g2, t) for g1, g2 in zip(ind1, ind2)]