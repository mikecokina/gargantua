from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from gargantua.raymarch.config import RayMarchResult


class RayTracer2D(Protocol):
    """2D ray tracer interface producing a RayMarchResult."""

    def trace(self, origin: Any, direction: Any) -> RayMarchResult:
        ...


class RayTracer3D(Protocol):
    """3D ray tracer interface producing a RayMarchResult."""

    def trace(self, origin: Any, direction: Any) -> RayMarchResult:
        ...
