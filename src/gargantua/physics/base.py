from __future__ import annotations

from typing import Any, Protocol

from gargantua.raymarch.config import RayMarchResult


class RayTracer2D(Protocol):
    """2D ray tracer interface producing a RayMarchResult."""

    def trace(self, origin: Any, direction: Any) -> RayMarchResult:
        ...
