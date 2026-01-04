from __future__ import annotations

from typing import Any


def normalize(
        xp: Any,
        v: Any,
        eps: float = 1e-12,
        axis: int | None = -1,
        keepdims: bool = True,
) -> Any:
    """Normalize vector or batch of vectors with numerical safety.

    This function is backend-agnostic and works with NumPy or CuPy.

    By default, vectors are assumed to live in the last dimension, which makes
    the operation batch-safe:
      - (3,)           single vector
      - (N, 3)         batch of vectors
      - (H, W, 3)      image-like grid of vectors

    Parameters:
        xp:
            Array module, typically numpy or cupy.
        v:
            Input vector or array of vectors.
        eps:
            Small positive value used to avoid division by zero.
        axis:
            Axis along which to compute the norm.
            - Default (-1) normalizes per-vector.
            - If None, computes a single global norm.
        keepdims:
            Whether to keep the reduced dimension when axis is not None.

    Returns:
        Normalized vector(s). If the norm is below eps, the input is returned unchanged.
    """
    if axis is None:
        n = xp.linalg.norm(v)
        if float(n) <= eps:
            return v
        return v / n

    n = xp.linalg.norm(v, axis=axis, keepdims=keepdims)
    n = xp.maximum(n, xp.asarray(eps, dtype=xp.float64))
    return v / n


def normalize_batch(xp: Any, v: Any) -> Any:
    """Normalize a batch of vectors along the last axis.

    This is a backward-compatible alias for:
        normalize(xp, v, axis=-1, keepdims=True)
    """
    return normalize(xp, v, axis=-1, keepdims=True)


def clamp(xp: Any, x: Any, lo: float, hi: float) -> Any:
    """Clamp array values to a closed interval [lo, hi].

    Works for NumPy or CuPy arrays.
    """
    return xp.clip(x, float(lo), float(hi))


def clamp_float(x: float, lo: float, hi: float) -> float:
    """Clamp a Python float to a closed interval [lo, hi]."""
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def cross(xp: Any, a: Any, b: Any) -> Any:
    """Compute cross product for vectors or batches of vectors.

    Uses xp.cross if available (NumPy or CuPy).
    Falls back to a manual implementation for array-like backends.

    Inputs are assumed to have the last dimension of size 3.
    """
    if hasattr(xp, "cross"):
        return xp.cross(a, b)

    ax, ay, az = a[..., 0], a[..., 1], a[..., 2]
    bx, by, bz = b[..., 0], b[..., 1], b[..., 2]
    return xp.stack(
        [
            ay * bz - az * by,
            az * bx - ax * bz,
            ax * by - ay * bx,
        ],
        axis=-1,
    )
