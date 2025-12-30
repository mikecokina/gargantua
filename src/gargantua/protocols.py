from __future__ import annotations

from typing import Any, Protocol


class SDF(Protocol):
    """Pure signed distance field contract."""

    def sdf(self, p: Any) -> Any:
        """Signed distance to surface at point p."""
        ...


class Drawable2D(Protocol):
    """2D debug drawable contract."""

    def polyline(self, num: int = 600) -> Any:
        """Return a (N,2) polyline suitable for plotting."""
        ...
