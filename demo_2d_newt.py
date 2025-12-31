from __future__ import annotations

from gargantua.backend import get_array_module, to_numpy
from gargantua.camera import Camera2D
from gargantua.geometry import SphereSDF
from gargantua.physics.newtonian import BlackHoleBender
from gargantua.raymarch.config import RayMarchConfig
from gargantua.raymarch.marcher import RayMarcher
from gargantua.scene import Scene, SceneBounds
from gargantua.viz.plot2d import Plotter2D


def main(*, no_gravity: bool = False) -> None:
    use_cuda = True
    xp = get_array_module(use_cuda)

    obj = SphereSDF(xp=xp, center=xp.asarray([2.5, 7.0], dtype=xp.float64), radius=1.0)
    bh = BlackHoleBender(
        xp=xp,
        center=xp.asarray([0.0, 3.0], dtype=xp.float64),
        mass=0.0 if no_gravity else 1.6,
        horizon=0.28,
    )

    scene = Scene(
        surface=obj,
        bh=bh,
        bounds=SceneBounds(far_distance=30.0),
        drawables=[
            obj,
            SphereSDF(xp=xp, center=bh.center, radius=bh.horizon),
        ],
    )

    camera_pos = xp.asarray([-3.5, 1.0], dtype=xp.float64)

    # Example: "look up" (positive z)
    camera_forward = xp.asarray([1.0, 1.0], dtype=xp.float64)

    # Or: "look toward BH center" using a direction vector
    # camera_forward = bh.center - camera_pos

    camera = Camera2D(
        position=camera_pos,
        forward=camera_forward,
        fov_deg=125.0,
        num_rays=35,
    )

    marcher = RayMarcher(xp=xp, config=RayMarchConfig(), scene=scene)

    plotter = Plotter2D()
    for drawable in scene.drawables:
        plotter.draw_drawable(xp, drawable)

    plotter.draw_point(to_numpy(xp, bh.center))
    plotter.draw_point(to_numpy(xp, camera_pos), label="camera")

    for d in camera.ray_directions(xp):
        result = marcher.trace(camera_pos, d)
        plotter.draw_ray(xp, result)

    plotter.show(xlim=(-7, 7), ylim=(-1, 11))


if __name__ == "__main__":
    main(no_gravity=False)
