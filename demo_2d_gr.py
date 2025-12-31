from __future__ import annotations

import numpy as np

from gargantua.backend import get_array_module, to_numpy
from gargantua.camera import Camera2D
from gargantua.geometry import SphereSDF
from gargantua.physics.geodesics import (
    SchwarzschildBlackHole2D,
    SchwarzschildConfig2D,
    SchwarzschildNullGeodesicTracer2D,
)
from gargantua.viz.plot2d import Plotter2D


def main(*, no_gravity: bool = False) -> None:
    use_cuda = False
    xp = get_array_module(use_cuda)

    # Scene drawables (optional): a "target" circle to compare visually
    obj = SphereSDF(xp=xp, center=xp.asarray([2.5, 7.0], dtype=xp.float64), radius=1.0)

    # BH in sim units: choose M as a length
    # Horizon = 2M, photon sphere = 3M.
    bh_center = xp.asarray([0.0, 3.0], dtype=xp.float64)
    cfg = SchwarzschildConfig2D(
        mass_length=0.0 if no_gravity else 0.35,  # tune this to see stronger/weaker lensing
        dphi=2e-3,
        escape_radius=60.0,
        max_steps=60_000,
    )
    bh = SchwarzschildBlackHole2D(xp=xp, center=bh_center, cfg=cfg)
    tracer = SchwarzschildNullGeodesicTracer2D(xp=xp, bh=bh, surface=obj)

    camera_pos = xp.asarray([-3.5, 1.0], dtype=xp.float64)

    # Camera looks "up" in your (x,z) 2D plane
    camera = Camera2D(
        position=camera_pos,
        forward=xp.asarray([0.0, 1.0], dtype=xp.float64),
        fov_deg=70.0,
        num_rays=35,
    )

    plotter = Plotter2D()
    plotter.ax.set_title("Gargantua - 2D Schwarzschild Null Geodesics")

    # Draw object circle
    plotter.draw_drawable(xp, obj)

    # Draw horizon circle
    horizon = bh.horizon_radius
    th = np.linspace(0.0, 2.0 * np.pi, 700)
    hxy = np.stack([np.cos(th) * horizon, np.sin(th) * horizon], axis=-1)
    hxy = hxy + to_numpy(xp, bh_center)[None, :]
    plotter.ax.plot(hxy[:, 0], hxy[:, 1], linewidth=2)

    plotter.draw_point(to_numpy(xp, bh_center), label="BH")
    plotter.draw_point(to_numpy(xp, camera_pos), label="camera")

    for d in camera.ray_directions(xp):
        result = tracer.trace(camera_pos, d)
        plotter.draw_ray(xp, result)

    plotter.show(xlim=(-7, 7), ylim=(-1, 11))


if __name__ == "__main__":
    main(no_gravity=False)
