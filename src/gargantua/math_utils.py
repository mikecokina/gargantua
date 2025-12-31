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


def normalize_batch(xp: Any, v: Any) -> Any:
    """Normalize vector with division-by-zero guard."""
    n = xp.linalg.norm(v, axis=-1, keepdims=True)
    n = xp.maximum(n, xp.asarray(1e-12, dtype=xp.float64))
    return v / n


def cross(xp: Any, a: Any, b: Any) -> Any:
    """Cross product that works for NumPy/CuPy and array-likes."""
    if hasattr(xp, "cross"):
        return xp.cross(a, b)
    ax, ay, az = a[..., 0], a[..., 1], a[..., 2]
    bx, by, bz = b[..., 0], b[..., 1], b[..., 2]
    return xp.stack([ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx], axis=-1)

