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
# TOP-LEVEL PARAMETERS (single source of truth)
# Match demo_3d_schw.py semantics, but projected to XZ plane.
# ============================================================

# Backend / run
BACKEND: BackendName = "auto"
NO_GRAVITY = False

# Scene (from 3D, projected to XZ -> [x, z])
SPHERE_CENTER_XZ = np.array([0.0, 0.0], dtype=np.float64)  # from SPHERE_CENTER_XYZ[x,z]
SPHERE_RADIUS = 1.0

BH_CENTER_XZ = np.array([0.4, 3.0], dtype=np.float64)  # from BH_CENTER_XYZ[x,z]

# Schwarzschild physics (same meaning as 3D demo)
MASS_LENGTH = 0.35
DPHI = 2e-3
ESCAPE_RADIUS = 80.0  # should match plot far and tracer escape

# Camera (from 3D, projected to XZ -> [x, z])
CAMERA_POS_XZ = np.array([0.6, -3.0], dtype=np.float64)  # from CAMERA_R0[x,z]

# Camera forward (from 3D LOOK_FORWARD_NP, projected to XZ)
LOOK_FORWARD_XZ = np.array([-0.2, 1.0], dtype=np.float64)  # from LOOK_FORWARD_NP[x,z]

# 2D camera ray fan (use same knobs as the 3D debug fan)
FOV_DEG = 90.0  # from FOV_X_DEG
NUM_RAYS = 33  # from DEBUG_NUM_RAYS

# Plot
TITLE = "Top-down (XZ): Schwarzschild null geodesics (2D)"
XLIM = (-5, 5)
ZLIM = (-5, 10)


# ============================================================
# Helpers
# ============================================================

def _unit_np(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v)) + 1e-12
    return v / n


def main() -> None:
    xp = get_array_module(BACKEND)

    # -------- Surface (sphere) --------
    sphere_center = xp.asarray(SPHERE_CENTER_XZ, dtype=xp.float64)
    sphere_radius = float(SPHERE_RADIUS)
    surface = SphereSDF(xp=xp, center=sphere_center, radius=sphere_radius)

    # -------- BH + tracer --------
    bh_center = xp.asarray(BH_CENTER_XZ, dtype=xp.float64)
    cfg = SchwarzschildConfig2D(
        mass_length=0.0 if NO_GRAVITY else float(MASS_LENGTH),
        dphi=float(DPHI),
        escape_radius=float(ESCAPE_RADIUS),
        max_steps=60_000,  # keep high for smooth polylines
    )
    bh = SchwarzschildBlackHole2D(xp=xp, center=bh_center, cfg=cfg)
    tracer = SchwarzschildNullGeodesicTracer2D(xp=xp, bh=bh, surface=surface)

    # -------- Camera (2D) --------
    cam_pos = xp.asarray(CAMERA_POS_XZ, dtype=xp.float64)

    fwd_np = _unit_np(LOOK_FORWARD_XZ)
    cam_forward = xp.asarray(fwd_np, dtype=xp.float64)

    cam = Camera2D(
        position=cam_pos,
        forward=cam_forward,
        fov_deg=float(FOV_DEG),
        num_rays=int(NUM_RAYS),
    )

    # -------- Plot --------
    plotter = Plotter2D()
    plotter.ax.set_title(str(TITLE))

    # Draw sphere
    plotter.draw_drawable(xp, surface)

    # Draw horizon circle (radius = 2M)
    horizon = float(bh.horizon_radius)
    th = np.linspace(0.0, 2.0 * np.pi, 800)
    hxy = np.stack([np.cos(th) * horizon, np.sin(th) * horizon], axis=-1)
    hxy = hxy + to_numpy(xp, bh_center)[None, :]
    plotter.ax.plot(hxy[:, 0], hxy[:, 1], linewidth=2, label="Black hole")

    plotter.draw_point(to_numpy(xp, bh_center), label="BH")
    plotter.draw_point(to_numpy(xp, cam_pos), label="camera")

    # Trace fan
    for d in cam.ray_directions(xp):
        rr = tracer.trace(cam_pos, d)
        plotter.draw_ray(xp, rr)

    plotter.show(xlim=XLIM, ylim=ZLIM)


if __name__ == "__main__":
    main()
