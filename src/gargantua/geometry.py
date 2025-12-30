from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from gargantua.backend import ArrayModule, to_numpy
from gargantua.protocols import Drawable2D, SDF


@dataclass(frozen=True, slots=True)
class SphereSDF(SDF, Drawable2D):
    """N-dimensional sphere SDF (circle in 2D)."""

    xp: ArrayModule
    center: Any
    radius: float

    def sdf(self, p: Any) -> Any:
        return self.xp.linalg.norm(p - self.center) - self.radius

    def polyline(self, num: int = 600) -> Any:
        center_np = to_numpy(self.xp, self.center)
        if center_np.shape != (2,):
            raise ValueError("Polyline only valid for 2D spheres")

        theta = np.linspace(0.0, 2.0 * np.pi, num, dtype=np.float64)
        poly = center_np[None, :] + np.stack(
            [np.cos(theta), np.sin(theta)], axis=-1
        ) * self.radius
        return self.xp.asarray(poly, dtype=self.xp.float64)
