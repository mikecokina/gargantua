from __future__ import annotations

from typing import Any

import numpy as np

try:
    import cupy as cp  # type: ignore
except ImportError:  # pragma: no cover
    cp = None

ArrayModule = Any


def get_array_module(use_cuda: bool) -> ArrayModule:
    """Return numpy or cupy depending on availability and request."""
    if use_cuda and cp is not None:
        return cp
    return np


def is_cupy(xp: ArrayModule) -> bool:
    """Return True if xp is the CuPy module."""
    return cp is not None and xp is cp


def to_numpy(xp: ArrayModule, a: Any) -> np.ndarray:
    """Convert xp array to NumPy for matplotlib."""
    if is_cupy(xp):
        return cp.asnumpy(a)  # type: ignore[union-attr]
    return np.asarray(a)
