from __future__ import annotations

import numpy as np

from gargantua.backend import BackendName, get_array_module, to_numpy
from gargantua.camera import Camera2D
from gargantua.geometry import SphereSDF
from gargantua.physics.geodesics import (
    SchwarzschildBlackHole2D,
    SchwarzschildConfig2D,
    SchwarzschildNullGeodesicTracer2D,
)
from gargantua.viz.plot2d import Plotter2D

# ============================================================
# TOP-LEVEL PARAMETERS (extract everything tweakable here)
# ============================================================

# Run
BACKEND: BackendName = "auto"
NO_GRAVITY = False

# Target object (optional visual reference)
OBJ_CENTER = [2.5, 7.0]
OBJ_RADIUS = 1.0

# BH (Schwarzschild)
BH_CENTER = [0.0, 3.0]
MASS_LENGTH = 0.35  # Horizon = 2M, photon sphere = 3M
DPHI = 2e-3
ESCAPE_RADIUS = 60.0
MAX_STEPS = 60_000

# Camera
CAMERA_POS = [-3.5, 1.0]
CAMERA_FORWARD = [0.0, 1.0]  # looks "up" in (x,z) plane
CAMERA_FOV_DEG = 70.0
CAMERA_NUM_RAYS = 35

# Plot
PLOT_TITLE = "Gargantua - 2D Schwarzschild Null Geodesics"
PLOT_XLIM = (-7, 7)
PLOT_YLIM = (-1, 11)

# Horizon draw sampling
HORIZON_SAMPLES = 700


def main() -> None:
    xp = get_array_module(BACKEND)

    # Scene drawables (optional): a "target" circle to compare visually
    obj = SphereSDF(xp=xp, center=xp.asarray(OBJ_CENTER, dtype=xp.float64), radius=float(OBJ_RADIUS))

    # BH in sim units: choose M as a length
    # Horizon = 2M, photon sphere = 3M.
    bh_center = xp.asarray(BH_CENTER, dtype=xp.float64)
    cfg = SchwarzschildConfig2D(
        mass_length=0.0 if NO_GRAVITY else float(MASS_LENGTH),  # tune this to see stronger/weaker lensing
        dphi=float(DPHI),
        escape_radius=float(ESCAPE_RADIUS),
        max_steps=int(MAX_STEPS),
    )
    bh = SchwarzschildBlackHole2D(xp=xp, center=bh_center, cfg=cfg)
    tracer = SchwarzschildNullGeodesicTracer2D(xp=xp, bh=bh, surface=obj)

    camera_pos = xp.asarray(CAMERA_POS, dtype=xp.float64)

    # Camera looks "up" in your (x,z) 2D plane
    camera = Camera2D(
        position=camera_pos,
        forward=xp.asarray(CAMERA_FORWARD, dtype=xp.float64),
        fov_deg=float(CAMERA_FOV_DEG),
        num_rays=int(CAMERA_NUM_RAYS),
    )

    plotter = Plotter2D()
    plotter.ax.set_title(PLOT_TITLE)

    # Draw object circle
    plotter.draw_drawable(xp, obj)

    # Draw horizon circle
    horizon = bh.horizon_radius
    th = np.linspace(0.0, 2.0 * np.pi, int(HORIZON_SAMPLES))
    hxy = np.stack([np.cos(th) * horizon, np.sin(th) * horizon], axis=-1)
    hxy = hxy + to_numpy(xp, bh_center)[None, :]
    plotter.ax.plot(hxy[:, 0], hxy[:, 1], linewidth=2)

    plotter.draw_point(to_numpy(xp, bh_center), label="BH")
    plotter.draw_point(to_numpy(xp, camera_pos), label="camera")

    for d in camera.ray_directions(xp):
        result = tracer.trace(camera_pos, d)
        plotter.draw_ray(xp, result)

    plotter.show(xlim=PLOT_XLIM, ylim=PLOT_YLIM)


if __name__ == "__main__":
    main()
