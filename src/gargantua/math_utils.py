from __future__ import annotations

from typing import Any


def normalize(xp: Any, v: Any, eps: float = 1e-12) -> Any:
    """Normalize vector with division-by-zero guard."""
    n = xp.linalg.norm(v)
    if float(n) <= eps:
        return v
    return v / n


def clamp_float(x: float, lo: float, hi: float) -> float:
    """Clamp a Python float to [lo, hi]."""
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x
