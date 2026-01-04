from __future__ import annotations

from typing import Any, Literal

import numpy as np

try:
    import cupy as cp  # type: ignore
except ImportError:  # pragma: no cover
    cp = None

ArrayModule = Any
BackendName = Literal["auto", "numpy", "cupy"]


def is_cupy(xp: ArrayModule) -> bool:
    """Return True if xp is the CuPy module."""
    return cp is not None and xp is cp


def get_array_module(backend: BackendName = "auto") -> ArrayModule:
    """Return numpy or cupy depending on availability and request.

    Supported calls:
      - get_array_module() -> "auto"
      - get_array_module("auto")  -> cupy if available else numpy
      - get_array_module("numpy") -> numpy
      - get_array_module("cupy")  -> cupy (raises if unavailable)
    """
    if backend == "numpy":
        return np

    if backend == "cupy":
        if cp is None:
            msg = "BACKEND='cupy' requested but CuPy is not available."
            raise RuntimeError(msg)
        return cp

    # backend == "auto"
    return cp if cp is not None else np


def to_numpy(xp: ArrayModule, a: Any) -> np.ndarray:
    """Convert xp array to NumPy for matplotlib."""
    if is_cupy(xp):
        return cp.asnumpy(a)  # type: ignore[union-attr]
    return np.asarray(a)
