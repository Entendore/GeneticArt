"""Perlin noise with a pure-Python fallback when the 'noise' package is absent."""

import math
import random

try:
    from noise import pnoise1  # type: ignore[import-untyped]
except ImportError:
    _perm = list(range(256))
    random.shuffle(_perm)
    _perm = _perm + _perm
    _grad = [random.uniform(-1.0, 1.0) for _ in range(512)]

    def _fade(t: float) -> float:
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

    def _lerp(a: float, b: float, t: float) -> float:
        return a + t * (b - a)

    def pnoise1(  # noqa: D401 – intentionally matches upstream API
        x: float,
        octaves: int = 1,
        persistence: float = 0.5,
        lacunarity: float = 2.0,
        repeat: int = 1024,
        base: int = 0,
    ) -> float:
        x = x % repeat
        xi = int(math.floor(x)) & 255
        xf = x - math.floor(x)
        u = _fade(xf)
        a = _perm[xi]
        b = _perm[xi + 1]
        return _lerp(_grad[a], _grad[b], u)