from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from gargantua.math_utils import normalize

if TYPE_CHECKING:
    from gargantua.backend import ArrayModule


@dataclass(frozen=True, slots=True)
class Camera2D:
    """Simple 2D pinhole camera that emits a fan of rays.

    Coordinate convention:
    - points are (x, z)
    - forward is a direction vector in world coordinates
    - angles are measured in radians from +x-axis toward +z axis (atan2)

    Parameters
    ----------
    position:
        Camera location (x, z).
    forward:
        Direction vector where the camera is looking (x, z).
        Does not need to be normalized.
    fov_deg:
        Full field of view in degrees.
    num_rays:
        Number of rays across the FOV.

    """

    position: Any
    forward: Any
    fov_deg: float
    num_rays: int

    def ray_directions(self, xp: ArrayModule) -> list[Any]:
        """Generate unit direction vectors spanning the camera FOV."""
        forward = normalize(xp, xp.asarray(self.forward, dtype=xp.float64))
        theta0 = float(xp.arctan2(forward[1], forward[0]))

        half_fov = np.deg2rad(self.fov_deg) * 0.5
        offsets = np.linspace(-half_fov, half_fov, self.num_rays, dtype=np.float64)

        out: list[Any] = []
        for da in offsets:
            a = theta0 + float(da)
            d = xp.asarray([np.cos(a), np.sin(a)], dtype=xp.float64)
            out.append(normalize(xp, d))
        return out

    @classmethod
    def from_look_at(
            cls,
            position: Any,
            look_at: Any,
            fov_deg: float,
            num_rays: int,
            xp: ArrayModule,
    ) -> Camera2D:
        """Define camera by a world-space target point.

        Convenience constructor: define camera by a world-space target point.

        This preserves the old behavior but makes it opt-in.
        """
        pos = xp.asarray(position, dtype=xp.float64)
        target = xp.asarray(look_at, dtype=xp.float64)
        forward = target - pos
        return cls(position=position, forward=forward, fov_deg=fov_deg, num_rays=num_rays)
