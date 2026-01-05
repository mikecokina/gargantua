from __future__ import annotations

import matplotlib as mpl

mpl.use("TkAgg", force=True)

import matplotlib.pyplot as plt
import numpy as np

from gargantua.backend import BackendName, get_array_module, to_numpy
from gargantua.geometry import SphereSDF
from gargantua.physics.geodesics.schwarzschild_3d import (
    SchwarzschildBlackHole3D,
    SchwarzschildConfig3D,
    SchwarzschildGeodesicStepper3D,
)

# ============================================================
# PARAMS
# ============================================================
BACKEND: BackendName = "auto"
NO_GRAVITY = False

SPHERE_CENTER_XYZ = np.array([0.0, 0.0, 0.0], dtype=np.float64)
SPHERE_RADIUS = 1.0

BH_CENTER_XYZ = np.array([0.4, 0.0, 3.0], dtype=np.float64)
MASS_LENGTH = 0.35
DPHI = 2e-3
ESCAPE_RADIUS = 80.0

# ------------------------------------------------------------
# TWO INDEPENDENT VECTORS
#   1) STEP_RAY_RD0_NP: ray you STEP (world-space direction)
#   2) CAMERA_FORWARD_NP: camera "look" direction (world-space)
# ------------------------------------------------------------

RO_NP = np.array([0.6, 0.5, -3.0], dtype=np.float64)
STEP_RAY_RD0_NP = np.array([0.1, 0.0, 1.0], dtype=np.float64)

CAMERA_POS_NP = RO_NP.copy()
CAMERA_FORWARD_NP = np.array([-0.2, 0.0, 1.0], dtype=np.float64)
CAMERA_UP_NP = np.array([0.0, 1.0, 0.0], dtype=np.float64)

# ------------------------------------------------------------
# CAMERA-PLANE MODE
# Plot + optionally constrain the stepped ray to the 2D plane spanned by (camera_right, camera_forward)
#
# NOTE:
#   This is NOT the usual "image plane" (right + up). It's the plane (right + forward) as requested.
# ------------------------------------------------------------
CONSTRAIN_TO_CAMERA_PLANE = True

# Debug draw lengths in the camera plane plot (u,v)
CAM_ARROW_LEN = 2.0  # camera basis arrows in (u,v)
RAY_ARROW_LEN = 1.0  # ray direction arrow in (u,v)
ARROW_HEAD = 10  # points

# Colors
CAM_FWD_COLOR = "tab:blue"
CAM_RIGHT_COLOR = "tab:green"
RAY_COLOR = "tab:red"

# Object intersection draw
INTERSECTION_SAMPLES = 720
DRAW_INTERSECTION_CENTER = True

# March-like step control
EPS = 2e-3
MAX_STEP = 0.05
MAX_DS_STEP = 0.005
DS_SURFACE_FACTOR = 0.15
MAX_OUTER_STEPS = 4000

# Plot window in (u,v) coordinates (camera plane)
ULIM = (-8, 8)  # u = right
VLIM = (-2, 14)  # v = forward


def _unit_np(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v)) + 1e-12
    return v / n


def _camera_basis_np(forward_np: np.ndarray, up_np: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (fwd, right, up_ortho) as unit vectors in world space.

    Right-handed basis convention:
      right = up x forward

    If you use forward x up, 'right' points the opposite way.
    """
    fwd = _unit_np(forward_np)
    up = _unit_np(up_np)

    # If forward and up are nearly parallel, pick a fallback up
    if abs(float(np.dot(fwd, up))) > 0.999:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(fwd, up))) > 0.999:
            up = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    right = np.cross(up, fwd)
    right = _unit_np(right)
    up_ortho = np.cross(fwd, right)
    up_ortho = _unit_np(up_ortho)
    return fwd, right, up_ortho


def _project_to_cam_plane_uv(p_world: np.ndarray, cam_pos: np.ndarray, cam_right: np.ndarray, cam_fwd: np.ndarray) -> \
tuple[float, float]:
    """Project world point into 2D (u,v) where u=right, v=forward."""
    d = p_world - cam_pos
    u = float(np.dot(d, cam_right))
    v = float(np.dot(d, cam_fwd))
    return u, v


def _constrain_point_to_plane(p_world: np.ndarray, plane_point: np.ndarray, plane_n: np.ndarray) -> np.ndarray:
    """Project world point onto plane through plane_point with normal plane_n."""
    d = p_world - plane_point
    return p_world - plane_n * float(np.dot(d, plane_n))


def _constrain_dir_to_plane(rd_world: np.ndarray, plane_n: np.ndarray) -> np.ndarray:
    """Remove component along plane normal, then renormalize."""
    rd = rd_world - plane_n * float(np.dot(rd_world, plane_n))
    return _unit_np(rd)


def _sphere_plane_intersection_circle(
        *,
        sphere_center: np.ndarray,
        sphere_radius: float,
        plane_point: np.ndarray,
        plane_n_unit: np.ndarray,
) -> tuple[np.ndarray, float] | None:
    """Intersection of sphere with plane is a circle (or none / tangent).

    Returns (circle_center_world, circle_radius) or None if no intersection.
    """
    # signed distance from center to plane
    d = float(np.dot(sphere_center - plane_point, plane_n_unit))
    ad = abs(d)
    if ad > sphere_radius:
        return None
    circle_center = sphere_center - plane_n_unit * d
    circle_radius = float((sphere_radius * sphere_radius - d * d) ** 0.5)
    return circle_center, circle_radius


def _circle_points_in_plane(
        *,
        circle_center: np.ndarray,
        circle_radius: float,
        plane_u_unit: np.ndarray,
        plane_v_unit: np.ndarray,
        n_samples: int,
) -> np.ndarray:
    """Build sampled points on a circle embedded in the plane spanned by (plane_u_unit, plane_v_unit).
    """
    th = np.linspace(0.0, 2.0 * np.pi, int(n_samples), endpoint=True)
    pts = circle_center[None, :] + circle_radius * (
            np.cos(th)[:, None] * plane_u_unit[None, :] + np.sin(th)[:, None] * plane_v_unit[None, :]
    )
    return pts


def main(*, backend: BackendName = BACKEND, no_gravity: bool = NO_GRAVITY) -> None:
    xp = get_array_module(backend)

    # Surface
    sphere_center = xp.asarray(SPHERE_CENTER_XYZ, dtype=xp.float64)
    surface = SphereSDF(xp=xp, center=sphere_center, radius=float(SPHERE_RADIUS))

    # BH + stepper
    bh_center = xp.asarray(BH_CENTER_XYZ, dtype=xp.float64)
    cfg = SchwarzschildConfig3D(
        mass_length=0.0 if no_gravity else float(MASS_LENGTH),
        dphi=float(DPHI),
        escape_radius=float(ESCAPE_RADIUS),
    )
    bh_params = SchwarzschildBlackHole3D(xp=xp, center=bh_center, cfg=cfg)
    stepper = SchwarzschildGeodesicStepper3D(xp=xp, bh=bh_params)

    # Camera basis (independent from step-ray direction)
    cam_fwd_np, cam_right_np, cam_up_np = _camera_basis_np(CAMERA_FORWARD_NP, CAMERA_UP_NP)

    # Camera-plane is spanned by (cam_right, cam_fwd)
    plane_u = cam_right_np
    plane_v = cam_fwd_np
    plane_n_np = _unit_np(np.cross(plane_u, plane_v))

    # Initial ray (the one we STEP)
    ro0_np = RO_NP.copy()
    rd0_np = _unit_np(STEP_RAY_RD0_NP)

    if CONSTRAIN_TO_CAMERA_PLANE:
        ro0_np = _constrain_point_to_plane(ro0_np, CAMERA_POS_NP, plane_n_np)
        rd0_np = _constrain_dir_to_plane(rd0_np, plane_n_np)

    ro = xp.asarray(ro0_np, dtype=xp.float64)
    rd0 = xp.asarray(rd0_np, dtype=xp.float64)

    # Stepper expects batch shapes, so make 1x1 "image"
    p = xp.zeros((1, 1, 3), dtype=xp.float64)
    p[0, 0, :] = ro
    rd = xp.zeros((1, 1, 3), dtype=xp.float64)
    rd[0, 0, :] = rd0

    state = stepper.init_batch(ro=p, rd0=rd)

    # Storage in world + camera-plane UV for plotting
    pts_world = [to_numpy(xp, p[0, 0]).copy()]
    dirs_world = [to_numpy(xp, rd[0, 0]).copy()]
    u0, v0 = _project_to_cam_plane_uv(pts_world[0], CAMERA_POS_NP, cam_right_np, cam_fwd_np)
    pts_uv = [(u0, v0)]
    traveled = 0.0

    # Setup plot in camera plane coords (u,v)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_title("Step viewer (camera plane u=right, v=forward): ENTER = step, q = quit")
    ax.set_xlabel("u (right)")
    ax.set_ylabel("v (forward)")
    ax.set_xlim(*ULIM)
    ax.set_ylim(*VLIM)
    ax.set_aspect("equal", adjustable="box")

    # Camera origin is (0,0) in this plot
    ax.scatter([0.0], [0.0], s=80, marker="x", linewidths=2, label="Camera pos (origin)", zorder=6)

    # ------------------------------------------------------------
    # Draw object intersections with the camera-plane
    # ------------------------------------------------------------
    plane_point = CAMERA_POS_NP

    # Sphere intersection
    sph = _sphere_plane_intersection_circle(
        sphere_center=SPHERE_CENTER_XYZ,
        sphere_radius=float(SPHERE_RADIUS),
        plane_point=plane_point,
        plane_n_unit=plane_n_np,
    )
    if sph is not None:
        c_center, c_r = sph
        pts = _circle_points_in_plane(
            circle_center=c_center,
            circle_radius=c_r,
            plane_u_unit=plane_u,
            plane_v_unit=plane_v,
            n_samples=INTERSECTION_SAMPLES,
        )
        uv = np.array([_project_to_cam_plane_uv(pp, plane_point, cam_right_np, cam_fwd_np) for pp in pts],
                      dtype=np.float64)
        ax.plot(uv[:, 0], uv[:, 1], linewidth=2, label="Sphere ∩ plane")
        if DRAW_INTERSECTION_CENTER:
            cu, cv = _project_to_cam_plane_uv(c_center, plane_point, cam_right_np, cam_fwd_np)
            ax.scatter([cu], [cv], s=20, marker="o", zorder=4)

    # BH horizon intersection (sphere of radius 2M)
    horizon_r = float(2.0 * cfg.mass_length)
    bh = _sphere_plane_intersection_circle(
        sphere_center=BH_CENTER_XYZ,
        sphere_radius=horizon_r,
        plane_point=plane_point,
        plane_n_unit=plane_n_np,
    )
    if bh is not None and horizon_r > 0.0:
        c_center, c_r = bh
        pts = _circle_points_in_plane(
            circle_center=c_center,
            circle_radius=c_r,
            plane_u_unit=plane_u,
            plane_v_unit=plane_v,
            n_samples=INTERSECTION_SAMPLES,
        )
        uv = np.array([_project_to_cam_plane_uv(pp, plane_point, cam_right_np, cam_fwd_np) for pp in pts],
                      dtype=np.float64)
        ax.plot(uv[:, 0], uv[:, 1], linewidth=2, label="Horizon ∩ plane")
        if DRAW_INTERSECTION_CENTER:
            cu, cv = _project_to_cam_plane_uv(c_center, plane_point, cam_right_np, cam_fwd_np)
            ax.scatter([cu], [cv], s=20, marker="o", zorder=4)

    # Path
    path_line, = ax.plot([], [], linewidth=2, label="Stepped ray path")
    head_scatter = ax.scatter([], [], s=30, zorder=5)

    # Arrows as annotations
    cam_fwd_ann = None
    cam_right_ann = None
    ray_dir_ann = None
    cam_fwd_lbl = None
    cam_right_lbl = None

    info = ax.text(
        0.02, 0.98, "",
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        family="monospace",
    )

    ax.legend(loc="lower left")
    plt.ion()
    plt.show()

    def _clear_ann(a):
        if a is None:
            return
        try:
            a.remove()
        except Exception:
            pass
        return

    def redraw() -> None:
        nonlocal cam_fwd_ann, cam_right_ann, ray_dir_ann, cam_fwd_lbl, cam_right_lbl

        uv = np.array(pts_uv, dtype=np.float64)
        us = uv[:, 0]
        vs = uv[:, 1]

        path_line.set_data(us, vs)
        head_scatter.set_offsets(np.array([[us[-1], vs[-1]]], dtype=np.float64))

        # Remove old annotations/labels
        cam_fwd_ann = _clear_ann(cam_fwd_ann)
        cam_right_ann = _clear_ann(cam_right_ann)
        ray_dir_ann = _clear_ann(ray_dir_ann)
        cam_fwd_lbl = _clear_ann(cam_fwd_lbl)
        cam_right_lbl = _clear_ann(cam_right_lbl)

        # In (u,v), camera forward is +v, camera right is +u by definition
        cam_fwd_tip = (0.0, CAM_ARROW_LEN)
        cam_right_tip = (CAM_ARROW_LEN, 0.0)

        cam_fwd_ann = ax.annotate(
            "",
            xy=cam_fwd_tip, xytext=(0.0, 0.0),
            arrowprops=dict(arrowstyle="->", mutation_scale=ARROW_HEAD, lw=2, color=CAM_FWD_COLOR),
            zorder=7,
        )
        cam_right_ann = ax.annotate(
            "",
            xy=cam_right_tip, xytext=(0.0, 0.0),
            arrowprops=dict(arrowstyle="->", mutation_scale=ARROW_HEAD, lw=2, color=CAM_RIGHT_COLOR),
            zorder=7,
        )

        cam_fwd_lbl = ax.text(cam_fwd_tip[0], cam_fwd_tip[1], "  cam_fwd", fontsize=9, color=CAM_FWD_COLOR, zorder=8)
        cam_right_lbl = ax.text(cam_right_tip[0], cam_right_tip[1], "  cam_right", fontsize=9, color=CAM_RIGHT_COLOR,
                                zorder=8)

        # Ray direction arrow at current head in (u,v)
        d_now = dirs_world[-1]
        du = float(np.dot(d_now, cam_right_np))
        dv = float(np.dot(d_now, cam_fwd_np))
        dn = (du * du + dv * dv) ** 0.5
        if dn < 1e-12:
            du, dv = 0.0, 0.0
        else:
            du, dv = du / dn, dv / dn

        ray_tip = (float(us[-1]) + RAY_ARROW_LEN * du, float(vs[-1]) + RAY_ARROW_LEN * dv)

        ray_dir_ann = ax.annotate(
            "",
            xy=ray_tip, xytext=(float(us[-1]), float(vs[-1])),
            arrowprops=dict(arrowstyle="->", mutation_scale=ARROW_HEAD, lw=2, color=RAY_COLOR),
            zorder=7,
        )

        fig.canvas.draw()
        fig.canvas.flush_events()

    redraw()

    for step_i in range(int(MAX_OUTER_STEPS)):
        s = input("ENTER = step, q = quit: ").strip().lower()
        if s == "q":
            break

        # Termination checks at current point (still done in world space)
        dist_pre = float(to_numpy(xp, surface.sdf(p))[0, 0])
        dist_bh = float(to_numpy(xp, xp.linalg.norm(p - bh_center[None, None, :], axis=-1))[0, 0])

        horizon = float(2.0 * cfg.mass_length)
        if dist_pre < float(EPS):
            print("HIT sphere (pre-step).")
            break
        if dist_bh < float(horizon):
            print("FELL into horizon (pre-step).")
            break
        if traveled > float(ESCAPE_RADIUS):
            print("ESCAPED (far).")
            break

        # Choose ds like the marcher
        ds_total = min(max(dist_pre, 0.0), float(MAX_STEP))
        ds_cap_surface = float(DS_SURFACE_FACTOR) * max(dist_pre, float(EPS))
        ds_cap = min(float(MAX_DS_STEP), ds_cap_surface)
        ds = min(ds_total, ds_cap)

        # Advance one step
        ds_arr = xp.asarray([[ds]], dtype=xp.float64)
        p_new, rd_new, fell_now = stepper.advance_batch(state, ds_arr)

        # Pull to numpy for optional constraint in world space
        p_np = to_numpy(xp, p_new[0, 0]).copy()
        rd_np = to_numpy(xp, rd_new[0, 0]).copy()

        if CONSTRAIN_TO_CAMERA_PLANE:
            p_np = _constrain_point_to_plane(p_np, plane_point, plane_n_np)
            rd_np = _constrain_dir_to_plane(rd_np, plane_n_np)

            p_new = xp.asarray(p_np, dtype=xp.float64)[None, None, :]
            rd_new = xp.asarray(rd_np, dtype=xp.float64)[None, None, :]

        p = p_new
        rd = rd_new

        traveled += ds
        pts_world.append(p_np)
        dirs_world.append(rd_np)

        u, v = _project_to_cam_plane_uv(p_np, plane_point, cam_right_np, cam_fwd_np)
        pts_uv.append((u, v))

        dist_post = float(to_numpy(xp, surface.sdf(p))[0, 0])
        dist_bh_post = float(to_numpy(xp, xp.linalg.norm(p - bh_center[None, None, :], axis=-1))[0, 0])

        info.set_text(
            f"step={step_i}\n"
            f"ds={ds:.6g}  traveled={traveled:.6g}\n"
            f"sdf_pre={dist_pre:.6g}  sdf_post={dist_post:.6g}\n"
            f"rBH_pre={dist_bh:.6g}  rBH_post={dist_bh_post:.6g}\n"
            f"fell_now={bool(to_numpy(xp, fell_now)[0, 0])}\n"
            f"\n"
            f"CAM_POS     ={CAMERA_POS_NP}\n"
            f"CAM_FWD     ={cam_fwd_np}\n"
            f"CAM_RIGHT   ={cam_right_np}\n"
            f"PLANE_N     ={plane_n_np}\n"
            f"\n"
            f"Plot coords: u=dot(p-cam_pos, cam_right), v=dot(p-cam_pos, cam_fwd)\n"
            f"Ray arrow:   du=dot(rd, cam_right), dv=dot(rd, cam_fwd) (normalized)",
        )

        redraw()

        if dist_post < float(EPS):
            print("HIT sphere (post-step).")
            break
        if bool(to_numpy(xp, fell_now)[0, 0]) or dist_bh_post < float(horizon):
            print("FELL into horizon (post-step).")
            break

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main(no_gravity=NO_GRAVITY, backend=BACKEND)
