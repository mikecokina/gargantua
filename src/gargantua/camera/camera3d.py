from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from gargantua.math_utils import cross, normalize_batch

if TYPE_CHECKING:
    from gargantua.backend import ArrayModule


@dataclass(frozen=True, slots=True)
class Camera3D:
    position: Any
    forward: Any
    up: Any
    fov_x_deg: float
    fov_y_deg: float

    def basis(self, xp: ArrayModule) -> tuple[Any, Any, Any]:
        f = normalize_batch(xp, xp.asarray(self.forward, dtype=xp.float64))
        up_hint = normalize_batch(xp, xp.asarray(self.up, dtype=xp.float64))
        r = normalize_batch(xp, cross(xp, f, up_hint))
        u = normalize_batch(xp, cross(xp, r, f))
        return f, r, u

    def ray_directions_grid(self, xp: ArrayModule, width: int, height: int) -> Any:
        """Return rd0 of shape (H, W, 3), normalized.

        Uses independent horizontal and vertical FOV.
        """
        forward, right, up = self.basis(xp)

        half_x = np.deg2rad(self.fov_x_deg) * 0.5
        half_y = np.deg2rad(self.fov_y_deg) * 0.5

        tx = float(np.tan(half_x))
        ty = float(np.tan(half_y))

        xs = xp.linspace(-tx, tx, width, dtype=xp.float64)
        ys = xp.linspace(ty, -ty, height, dtype=xp.float64)  # top -> bottom

        rd = (
                forward[None, None, :]
                + xs[None, :, None] * right[None, None, :]
                + ys[:, None, None] * up[None, None, :]
        )
        return normalize_batch(xp, rd)

    @classmethod
    def from_forward(
            cls,
            position: Any,
            forward: Any,
            up: Any,
            fov_x_deg: float,
            fov_y_deg: float,
            xp: ArrayModule,
    ) -> Camera3D:
        pos = xp.asarray(position, dtype=xp.float64)
        fwd = xp.asarray(forward, dtype=xp.float64)
        return cls(position=pos, forward=fwd, up=up, fov_x_deg=fov_x_deg, fov_y_deg=fov_y_deg)

    @classmethod
    def from_look_at(
            cls,
            position: Any,
            look_at: Any,
            up: Any,
            fov_x_deg: float,
            fov_y_deg: float,
            xp: ArrayModule,
    ) -> Camera3D:
        pos = xp.asarray(position, dtype=xp.float64)
        tgt = xp.asarray(look_at, dtype=xp.float64)
        forward = tgt - pos
        return cls(position=pos, forward=forward, up=up, fov_x_deg=fov_x_deg, fov_y_deg=fov_y_deg)
