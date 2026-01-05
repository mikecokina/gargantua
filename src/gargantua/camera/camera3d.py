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

    def basis(self, xp: ArrayModule) -> tuple[Any, Any, Any]:
        f = normalize_batch(xp, xp.asarray(self.forward, dtype=xp.float64))
        up_hint = normalize_batch(xp, xp.asarray(self.up, dtype=xp.float64))

        # right-handed: right = up x forward
        r = normalize_batch(xp, cross(xp, up_hint, f))

        # recompute true up: up = forward x right
        u = normalize_batch(xp, cross(xp, f, r))
        return f, r, u

    def ray_grid(
            self,
            xp: ArrayModule,
            width: int,
            height: int,
            projection: str = "rectilinear",
    ) -> Any:
        if projection == "rectilinear":
            return self.ray_rectilinear_grid(xp, width, height)
        if projection == "equirectangular":
            return self.ray_equirectangular_grid(xp, width, height)

        msg = f"Unknown projection '{projection}'"
        raise ValueError(msg)

    def ray_rectilinear_grid(self, xp: ArrayModule, width: int, height: int) -> Any:
        """
        Rectilinear (pinhole) camera projection.

        This is a true pinhole camera with an implicit image plane at distance f
        in front of the camera, where:

            f = 1 / tan(fov_x / 2)

        The image plane spans [-1, +1] in normalized X, and the vertical extent
        is adjusted by the image aspect ratio to preserve square pixels.

        The ray direction before normalization is:

            d = f * forward + x * right + y * up
        """
        forward, right, up = self.basis(xp)

        # --- focal length (implicit pinhole camera) ---
        half_x = np.deg2rad(self.fov_x_deg) * 0.5
        focal_length = 1.0 / float(np.tan(half_x))  # f

        # --- pixel-center sampling ---
        ix = xp.arange(width, dtype=xp.float64)
        iy = xp.arange(height, dtype=xp.float64)

        # Normalized image plane coordinates in [-1, 1]
        x_ndc = ((ix + 0.5) / float(width)) * 2.0 - 1.0
        y_ndc = 1.0 - ((iy + 0.5) / float(height)) * 2.0  # top -> bottom

        # Preserve square pixels by scaling Y by aspect ratio
        aspect = float(height) / float(width)
        x_img = x_ndc
        y_img = y_ndc * aspect

        # --- pinhole ray construction ---
        rd = (
                focal_length * forward[None, None, :] +
                x_img[None, :, None] * right[None, None, :] +
                y_img[:, None, None] * up[None, None, :]
        )

        return normalize_batch(xp, rd)

    def ray_equirectangular_grid(self, xp: ArrayModule, width: int, height: int) -> Any:
        """Return rd0 of shape (H, W, 3) using Equirectangular Projection."""
        forward, right, up = self.basis(xp)

        # 1. Define the angular FOV limits in radians
        fov_x_rad = np.deg2rad(self.fov_x_deg)
        # Maintain aspect ratio for the vertical FOV
        fov_y_rad = fov_x_rad * (float(height) / float(width))

        # 2. Create 1D grids for coordinates
        ix = xp.arange(width, dtype=xp.float64)
        iy = xp.arange(height, dtype=xp.float64)

        # Normalized coordinates [-1, 1]
        u = ((ix + 0.5) / float(width)) * 2.0 - 1.0
        v = 1.0 - ((iy + 0.5) / float(height)) * 2.0

        # Longitude (horizontal) and Latitude (vertical)
        lon = u * (fov_x_rad / 2.0)
        lat = v * (fov_y_rad / 2.0)

        # 3. Reshape for broadcasting to create (H, W) shapes
        # lon becomes (1, W), lat becomes (H, 1)
        lon_2d = lon[None, :]
        lat_2d = lat[:, None]

        # 4. Compute directions in the camera's local coordinate system
        # These will now result in (H, W) arrays
        dir_local_x = xp.sin(lon_2d) * xp.cos(lat_2d)
        dir_local_y = xp.sin(lat_2d)
        dir_local_z = xp.cos(lon_2d) * xp.cos(lat_2d)

        # 5. Transform local directions to world space using the camera basis
        # Resulting shape: (H, W, 3)
        rd = (
                dir_local_z[:, :, None] * forward[None, None, :] +
                dir_local_x[:, :, None] * right[None, None, :] +
                dir_local_y[:, :, None] * up[None, None, :]
        )

        return normalize_batch(xp, rd)

    @classmethod
    def from_forward(
            cls,
            position: Any,
            forward: Any,
            up: Any,
            fov_x_deg: float,
            xp: ArrayModule,
    ) -> Camera3D:
        pos = xp.asarray(position, dtype=xp.float64)
        fwd = xp.asarray(forward, dtype=xp.float64)
        return cls(position=pos, forward=fwd, up=up, fov_x_deg=fov_x_deg)

    @classmethod
    def from_look_at(
            cls,
            position: Any,
            look_at: Any,
            up: Any,
            fov_x_deg: float,
            xp: ArrayModule,
    ) -> Camera3D:
        pos = xp.asarray(position, dtype=xp.float64)
        tgt = xp.asarray(look_at, dtype=xp.float64)
        forward = tgt - pos
        return cls(position=pos, forward=forward, up=up, fov_x_deg=fov_x_deg)
