from __future__ import annotations

from gargantua.backend import BackendName, get_array_module, to_numpy
from gargantua.camera import Camera2D
from gargantua.geometry import SphereSDF
from gargantua.physics.newtonian import BlackHoleBender
from gargantua.raymarch.config import RayMarchConfig
from gargantua.raymarch.marcher import RayMarcher
from gargantua.scene import Scene, SceneBounds
from gargantua.viz.plot2d import Plotter2D

# ============================================================
# TOP-LEVEL PARAMETERS (extract everything tweakable here)
# ============================================================

# Run
BACKEND: BackendName = "auto"
NO_GRAVITY = False

# Scene: main object
OBJ_CENTER = [2.5, 7.0]
OBJ_RADIUS = 1.0

# Scene: BH
BH_CENTER = [0.0, 3.0]
BH_MASS = 1.6
BH_HORIZON = 0.28

# Scene bounds
FAR_DISTANCE = 30.0

# Camera
CAMERA_POS = [-3.5, 1.0]

# Option A: explicit forward vector
CAMERA_FORWARD = [1.0, 1.0]

# Option B: look at BH center (overrides CAMERA_FORWARD if enabled)
CAMERA_LOOK_AT_BH = False

# Camera ray fan
CAMERA_FOV_DEG = 125.0
CAMERA_NUM_RAYS = 35

# Plot
PLOT_XLIM = (-7, 7)
PLOT_YLIM = (-1, 11)


def main() -> None:
    xp = get_array_module(BACKEND)

    obj = SphereSDF(xp=xp, center=xp.asarray(OBJ_CENTER, dtype=xp.float64), radius=float(OBJ_RADIUS))
    bh = BlackHoleBender(
        xp=xp,
        center=xp.asarray(BH_CENTER, dtype=xp.float64),
        mass=0.0 if NO_GRAVITY else float(BH_MASS),
        horizon=float(BH_HORIZON),
    )

    scene = Scene(
        surface=obj,
        bh=bh,
        bounds=SceneBounds(far_distance=float(FAR_DISTANCE)),
        drawables=[
            obj,
            SphereSDF(xp=xp, center=bh.center, radius=bh.horizon),
        ],
    )

    camera_pos = xp.asarray(CAMERA_POS, dtype=xp.float64)

    camera_forward = bh.center - camera_pos if CAMERA_LOOK_AT_BH else xp.asarray(CAMERA_FORWARD, dtype=xp.float64)

    camera = Camera2D(
        position=camera_pos,
        forward=camera_forward,
        fov_deg=float(CAMERA_FOV_DEG),
        num_rays=int(CAMERA_NUM_RAYS),
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

    plotter.show(xlim=PLOT_XLIM, ylim=PLOT_YLIM)


if __name__ == "__main__":
    main()
