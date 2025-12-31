from __future__ import annotations

import numpy as np

from gargantua.backend import get_array_module, to_numpy
from gargantua.camera.camera3d import Camera3D
from gargantua.geometry import SphereSDF
from gargantua.math_utils import normalize
from gargantua.physics.newtonian import BlackHoleBender
from gargantua.raymarch.config import RayMarchConfig
from gargantua.raymarch.marcher import ImageMarchConfig, ImageMarcher, RayMarcher
from gargantua.scene import Scene, SceneBounds
from gargantua.viz.plot3d import RenderFrame, RenderSplitPlotter


def render_colors(hit: np.ndarray, fell_in: np.ndarray) -> np.ndarray:
    h, w = hit.shape
    img = np.ones((h, w, 3), dtype=np.float32)  # white background
    img[fell_in] = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # BH capture
    img[hit] = np.array([0.15, 0.35, 1.0], dtype=np.float32)  # sphere hit
    return img


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v)) + 1e-12
    return v / n


def camera_path_orbit_forward(i: int, n: int, bh_center: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (ro, forward) in world space."""
    t = i / float(max(1, n - 1))
    ang = 2.0 * np.pi * t
    ro = np.array(
        [bh_center[0] + 3.5 * np.cos(ang), 0.6, bh_center[2] - 3.5 * np.sin(ang)],
        dtype=np.float64,
    )
    target = np.array([0.0, 0.0, 5.5], dtype=np.float64)  # aim toward the sphere
    forward = target - ro
    return ro, forward


def camera_path_slide_forward(i: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (ro, forward) in world space."""
    t = i / float(max(1, n - 1))
    ro = np.array([0.6 - 1.2 * t, 0.5, -0.8], dtype=np.float64)
    target = np.array([0.0 - 0.4 * t, 0.0, 5.5], dtype=np.float64)
    forward = target - ro
    return ro, forward


def horizontal_fan_directions(
        xp: np.ndarray,
        forward: np.ndarray,
        right: np.ndarray,
        fov_x_deg: float,
        num_rays: int,
) -> list[np.ndarray]:
    """Horizontal ray fan in the camera's (forward,right) plane.

    Returns list of (3,) directions in xp-space.
    """
    half = np.deg2rad(fov_x_deg) * 0.5
    tx = float(np.tan(half))
    xs = np.linspace(-tx, tx, num_rays, dtype=np.float64)

    out = []
    for x in xs:
        d = forward + x * right
        # noinspection PyUnresolvedReferences
        out.append(normalize(xp, xp.asarray(d, dtype=xp.float64)))
    return out


def main() -> None:
    use_cuda = True
    xp = get_array_module(use_cuda)

    # Fixed scene
    sphere_center = xp.asarray([0.0, 0.0, 5.5], dtype=xp.float64)
    sphere_radius = 1.0
    surface = SphereSDF(xp=xp, center=sphere_center, radius=sphere_radius)

    bh_center = xp.asarray([0.4, 0.0, 3.0], dtype=xp.float64)
    bh = BlackHoleBender(
        xp=xp,
        center=bh_center,
        mass=0.35,
        horizon=0.15,
        softening=5e-2,
    )

    scene = Scene(
        surface=surface,
        bh=bh,
        bounds=SceneBounds(far_distance=80.0),
        drawables=[],
    )

    # Image marcher (render)
    image_marcher = ImageMarcher(
        xp=xp,
        config=ImageMarchConfig(max_steps=220, eps=1e-3, min_step=1e-3, max_step=0.08),
        scene=scene,
    )

    # Ray marcher (debug fan in 2D split)
    ray_marcher = RayMarcher(xp=xp, config=RayMarchConfig(), scene=scene)

    # Render config
    width, height = 360, 200
    fov_horizontal_deg = 90.0
    fov_vertical_deg = 55.0
    up = xp.asarray([0.0, 1.0, 0.0], dtype=xp.float64)

    # Debug rays for top-down panel
    num_debug_rays = 41
    decimate_stride = 2
    look_line_len = 6.0  # just for visualization in the top-down panel

    frames_n = 10
    frames: list[RenderFrame] = []

    bh_center_np = to_numpy(xp, bh_center)

    for i in range(frames_n):
        # Pick ONE path:
        ro_np, forward_np = camera_path_slide_forward(i, frames_n)
        # ro_np, forward_np = camera_path_orbit_forward(i, frames_n, bh_center_np)

        ro = xp.asarray(ro_np, dtype=xp.float64)
        forward = xp.asarray(forward_np, dtype=xp.float64)

        cam = Camera3D.from_forward(
            position=ro,
            forward=forward,
            up=up,
            fov_x_deg=fov_horizontal_deg,
            fov_y_deg=fov_vertical_deg,
            xp=xp,
        )

        # ===== 1) Render image =====
        rd0 = cam.ray_directions_grid(xp, width=width, height=height)
        res_img = image_marcher.march(ro=ro, rd0=rd0)

        img = render_colors(
            hit=to_numpy(xp, res_img.hit),
            fell_in=to_numpy(xp, res_img.fell_in),
        )

        # ===== 2) Build debug ray fan for top-down XZ =====
        fwd, right, _true_up = cam.basis(xp)
        fwd_np_unit = _unit(to_numpy(xp, fwd))
        right_np = to_numpy(xp, right)

        rays_xz: list[np.ndarray] = []
        dirs = horizontal_fan_directions(
            xp=xp,
            forward=fwd_np_unit,  # use unit forward for fan construction
            right=right_np,
            fov_x_deg=fov_horizontal_deg,
            num_rays=num_debug_rays,
        )

        for d in dirs:
            res = ray_marcher.trace(ro, d)
            pts = to_numpy(xp, res.points)  # (K,3)
            if pts.shape[0] > 2 and decimate_stride > 1:
                pts = pts[::decimate_stride]
            rays_xz.append(pts[:, [0, 2]])  # (K,2) => x,z

        # Derive a "look_at point" for plotting the red line
        look_at_np = ro_np + fwd_np_unit * look_line_len

        frames.append(
            RenderFrame(
                img=img,
                cam_pos=to_numpy(xp, ro),
                look_at=look_at_np,
                rays_xz=rays_xz,
            ),
        )

    two_d_split = True
    plotter = RenderSplitPlotter(two_d_split=two_d_split)

    ani = plotter.animate(
        frames=frames,
        sphere_center_xz=np.array([0.0, 5.5], dtype=np.float64),
        sphere_radius=sphere_radius,
        bh_center_xz=np.array([bh_center_np[0], bh_center_np[2]], dtype=np.float64),
        bh_horizon=float(bh.horizon),
        xlim=(-5, 5),
        zlim=(-2, 10),
        interval_ms=120,
    )

    _ = ani  # keep reference
    plotter.show()


if __name__ == "__main__":
    main()
