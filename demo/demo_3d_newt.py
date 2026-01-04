from __future__ import annotations

import numpy as np

from matplotlib.animation import FuncAnimation

from gargantua.backend import BackendName, get_array_module, to_numpy
from gargantua.camera.camera3d import Camera3D
from gargantua.geometry import SphereSDF
from gargantua.math_utils import normalize
from gargantua.physics.newtonian import BlackHoleBender
from gargantua.raymarch.config import RayMarchConfig
from gargantua.raymarch.marcher import ImageMarchConfig, ImageMarcherNewtonian, RayMarcher
from gargantua.scene import Scene, SceneBounds

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

import matplotlib

matplotlib.use("TkAgg", force=True)

import matplotlib.pyplot as plt  # noqa


# ============================================================
# TOP-LEVEL SCENE + CAMERA + RENDER SETTINGS (edit these)
# ============================================================
BACKEND: BackendName = "auto"
NO_GRAVITY = False
FRAMES_N = 20
USE_TQDM = True

# Scene objects
SPHERE_CENTER_NP = np.array([0.0, 0.0, 5.5], dtype=np.float64)
SPHERE_RADIUS = 1.0

BH_CENTER_NP = np.array([0.4, 0.0, 3.0], dtype=np.float64)
BH_MASS = 0.55
BH_HORIZON = 0.75
BH_SOFTENING = 5e-2

# Camera start / path
CAM_START_RO_NP = np.array([0.6, 0.0, -5.5], dtype=np.float64)
SLIDE_RO_OFFSET_PER_T = np.array([-1.2, 0.0, 0.0], dtype=np.float64)  # multiplied by t

# Where the camera looks (WORLD-SPACE direction vector)
LOOK_FORWARD_NP = np.array([0.0, 0.0, 1.0], dtype=np.float64)

# Render resolution + FOV
RENDER_WIDTH = 360
RENDER_HEIGHT = 200
FOV_HORIZONTAL_DEG = 90.0
UP_NP = np.array([0.0, 1.0, 0.0], dtype=np.float64)

# March configs (image marcher)
IMAGE_MARCH_MAX_STEPS = 220
IMAGE_MARCH_EPS = 1e-3
IMAGE_MARCH_MIN_STEP = 1e-3
IMAGE_MARCH_MAX_STEP = 0.08

# ============================================================
# TRUE 2D SLICE PLOT (single physical plane)
# ============================================================
DEBUG_NUM_RAYS_SLICE = 41
DECIMATE_STRIDE = 2
DEBUG_ARROW_LEN = 2.5  # arrow length in slice (u,v) units
INTERSECTION_SAMPLES = 720

# Plot / animation
TWO_D_SPLIT = True
RIGHT_PLOT_TITLE = (
    "2D slice plane through camera + BH center\n"
    "UV axes are slice-plane basis (not the camera image plane)"
)
ULIM = (-10, 10)
VLIM = (-4, 20)
INTERVAL_MS = 320

# 3rd plot: world XZ overview (no rays, just objects + camera motion/basis)
XZ_TITLE = "World XZ overview (objects + camera motion/basis)"
XZ_XLIM = (-20, 20)
XZ_ZLIM = (-10, 25)
XZ_ARROW_LEN = 1.5

# Scene bounds / marcher defaults
FAR_DISTANCE = 80.0
USE_ORBIT_PATH = True


def normalize_vector_np(v: np.ndarray) -> np.ndarray:
    """Normalize a numpy vector."""
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-12:
        return v
    return v / v_norm


def render_colors(hit: np.ndarray, fell_in: np.ndarray) -> np.ndarray:
    """Simple RGB coloring."""
    h, w = hit.shape
    img = np.ones((h, w, 3), dtype=np.float32)
    img[fell_in] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    img[hit] = np.array([0.15, 0.35, 1.0], dtype=np.float32)
    return img


def camera_path_orbit_ro(
        i: int,
        n_frames: int,
        bh_center_np: np.ndarray,
        cam_start_ro_np: np.ndarray,
) -> np.ndarray:
    """Orbit BH in XZ plane, starting exactly at CAM_START_RO_NP (and keeping its Y)."""
    dx = float(cam_start_ro_np[0] - bh_center_np[0])
    dz = float(cam_start_ro_np[2] - bh_center_np[2])

    r = float(np.hypot(dx, dz))
    if r < 1e-12:
        r = 1e-12

    a0 = float(np.arctan2(-dz, dx))

    t = i / float(max(1, n_frames - 1))
    a = a0 + 2.0 * np.pi * t

    return np.array(
        [
            bh_center_np[0] + r * np.cos(a),
            float(cam_start_ro_np[1]),
            bh_center_np[2] - r * np.sin(a),
        ],
        dtype=np.float64,
    )


def camera_path_slide_ro(i: int, n_frames: int) -> np.ndarray:
    """Return ro in world space."""
    t = i / float(max(1, n_frames - 1))
    return CAM_START_RO_NP + SLIDE_RO_OFFSET_PER_T * t


def _compute_fov_vertical_deg(width: int, height: int, fov_horizontal_deg: float) -> float:
    aspect = float(width) / float(height)
    return float(
        np.rad2deg(
            2.0
            * np.arctan(
                np.tan(np.deg2rad(fov_horizontal_deg) * 0.5) / aspect,
            ),
        ),
    )


def _build_slice_basis(
        *,
        ro_np: np.ndarray,
        cam_fwd_np_unit: np.ndarray,
        bh_center_np: np.ndarray,
        world_up_np_unit: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a physical 2D slice plane basis (u_hat, v_hat, n_hat)."""
    a_hat = normalize_vector_np(bh_center_np - ro_np)

    n_vec = np.cross(a_hat, cam_fwd_np_unit)

    # Degeneracy fix: if forward || a_hat, keep slice stable by using world_up as normal.
    if float(np.linalg.norm(n_vec)) < 1e-10:
        n_hat = world_up_np_unit
    else:
        n_hat = normalize_vector_np(n_vec)

    v_vec = cam_fwd_np_unit - n_hat * float(np.dot(cam_fwd_np_unit, n_hat))
    if float(np.linalg.norm(v_vec)) < 1e-10:
        v_vec = a_hat
    v_hat = normalize_vector_np(v_vec)

    u_hat = normalize_vector_np(np.cross(n_hat, v_hat))
    return u_hat, v_hat, n_hat


def _stabilize_slice_basis(
        *,
        u_hat: np.ndarray,
        v_hat: np.ndarray,
        n_hat: np.ndarray,
        world_up_unit: np.ndarray,
        prev_u_hat: np.ndarray | None,
        prev_v_hat: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prevent sign flips of the (u,v,n) basis that cause mirrored UV plots."""
    # 1) Keep n pointing to the same "up" hemisphere.
    if float(np.dot(n_hat, world_up_unit)) < 0.0:
        n_hat = -n_hat
        u_hat = -u_hat  # keep right-handed: (-u) x v = -(u x v) = new n

    # 2) Keep v continuous across frames (if it flips, UV mirrors).
    if prev_v_hat is not None:
        if float(np.dot(v_hat, prev_v_hat)) < 0.0:
            v_hat = -v_hat
            u_hat = -u_hat  # flip both keeps n the same

    # (Optional: you can also continuity-lock u, but v is the main culprit)
    return u_hat, v_hat, n_hat


def _slice_fan_dirs_xp(
        *,
        xp: np.ndarray,
        v_hat_np: np.ndarray,
        u_hat_np: np.ndarray,
        fov_x_deg: float,
        num_rays: int,
) -> list[np.ndarray]:
    """Build unit directions spanning the slice plane (v + x*u), returned as xp arrays."""
    half = np.deg2rad(float(fov_x_deg)) * 0.5
    tx = float(np.tan(half))
    xs = np.linspace(-tx, tx, int(num_rays), dtype=np.float64)

    out: list[np.ndarray] = []
    for x in xs:
        d_np = v_hat_np + x * u_hat_np
        out.append(normalize(xp, xp.asarray(d_np, dtype=xp.float64)))
    return out


def _project_uv(
        p_world: np.ndarray,
        ro_np: np.ndarray,
        u_hat: np.ndarray,
        v_hat: np.ndarray,
) -> tuple[float, float]:
    d = p_world - ro_np
    return float(np.dot(d, u_hat)), float(np.dot(d, v_hat))


def _sphere_plane_intersection_circle(
        *,
        sphere_center: np.ndarray,
        sphere_radius: float,
        plane_point: np.ndarray,
        plane_n_unit: np.ndarray,
) -> tuple[np.ndarray, float] | None:
    dist = float(np.dot(sphere_center - plane_point, plane_n_unit))
    if abs(dist) > sphere_radius:
        return None
    circle_center = sphere_center - plane_n_unit * dist
    circle_radius = float((sphere_radius * sphere_radius - dist * dist) ** 0.5)
    return circle_center, circle_radius


def _circle_points_in_plane(
        *,
        circle_center: np.ndarray,
        circle_radius: float,
        plane_u_unit: np.ndarray,
        plane_v_unit: np.ndarray,
        n_samples: int,
) -> np.ndarray:
    th = np.linspace(0.0, 2.0 * np.pi, int(n_samples), endpoint=True)
    return circle_center[None, :] + circle_radius * (
            np.cos(th)[:, None] * plane_u_unit[None, :]
            + np.sin(th)[:, None] * plane_v_unit[None, :]
    )


def build_scene(*, xp: np.ndarray, no_gravity: bool) -> tuple[Scene, np.ndarray, np.ndarray]:
    sphere_center = xp.asarray(SPHERE_CENTER_NP, dtype=xp.float64)
    surface = SphereSDF(xp=xp, center=sphere_center, radius=float(SPHERE_RADIUS))

    bh_center = xp.asarray(BH_CENTER_NP, dtype=xp.float64)
    bh = BlackHoleBender(
        xp=xp,
        center=bh_center,
        mass=0.0 if no_gravity else float(BH_MASS),
        horizon=float(BH_HORIZON),
        softening=float(BH_SOFTENING),
    )

    scene = Scene(
        surface=surface,
        bh=bh,
        bounds=SceneBounds(far_distance=float(FAR_DISTANCE)),
        drawables=[],
    )
    return scene, sphere_center, bh_center


def main() -> None:
    xp = get_array_module(BACKEND)

    scene, _sphere_center, bh_center = build_scene(xp=xp, no_gravity=NO_GRAVITY)

    image_marcher = ImageMarcherNewtonian(
        xp=xp,
        config=ImageMarchConfig(
            max_steps=int(IMAGE_MARCH_MAX_STEPS),
            eps=float(IMAGE_MARCH_EPS),
            min_step=float(IMAGE_MARCH_MIN_STEP),
            max_step=float(IMAGE_MARCH_MAX_STEP),
        ),
        scene=scene,
    )

    ray_marcher = RayMarcher(xp=xp, config=RayMarchConfig(), scene=scene)

    width, height = int(RENDER_WIDTH), int(RENDER_HEIGHT)
    fov_horizontal_deg = float(FOV_HORIZONTAL_DEG)
    fov_vertical_deg = _compute_fov_vertical_deg(width, height, fov_horizontal_deg)
    up = xp.asarray(UP_NP, dtype=xp.float64)

    frames_n = int(FRAMES_N)
    bh_center_np = to_numpy(xp, bh_center)
    world_up_np_unit = normalize_vector_np(UP_NP)
    base_forward_np = normalize_vector_np(LOOK_FORWARD_NP)

    imgs: list[np.ndarray] = []
    rays_uv_frames: list[list[np.ndarray]] = []
    slice_basis_frames: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []  # (ro, u_hat, v_hat, n_hat)

    cam_pos_xz_frames: list[tuple[float, float]] = []
    cam_fwd_xz_frames: list[tuple[float, float]] = []
    cam_right_xz_frames: list[tuple[float, float]] = []

    prev_u_hat: np.ndarray | None = None
    prev_v_hat: np.ndarray | None = None

    frame_iter = range(frames_n)
    if USE_TQDM and tqdm is not None:
        frame_iter = tqdm(frame_iter, total=frames_n, desc="Rendering frames")

    for i in frame_iter:
        if USE_ORBIT_PATH:
            ro_np = camera_path_orbit_ro(i, frames_n, bh_center_np, CAM_START_RO_NP)
            forward_np = normalize_vector_np(bh_center_np - ro_np)
        else:
            ro_np = camera_path_slide_ro(i, frames_n)
            forward_np = base_forward_np

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

        # 1) Render image
        rd0 = cam.ray_directions_grid(xp, width=width, height=height)
        res_img = image_marcher.march(ro=ro, rd0=rd0)
        imgs.append(
            render_colors(
                hit=to_numpy(xp, res_img.hit),
                fell_in=to_numpy(xp, res_img.fell_in),
            ),
        )

        # 2) TRUE 2D SLICE rays
        fwd, right, _up_ortho = cam.basis(xp)

        # --- 3rd panel (world XZ) storage: camera pos + basis projected to XZ ---
        cam_pos_xz_frames.append((float(ro_np[0]), float(ro_np[2])))

        fwd_np3 = to_numpy(xp, fwd)
        right_np3 = to_numpy(xp, right)
        fwd_xz = np.array([fwd_np3[0], fwd_np3[2]], dtype=np.float64)
        right_xz = np.array([right_np3[0], right_np3[2]], dtype=np.float64)
        fwd_xz_n = normalize_vector_np(fwd_xz)
        right_xz_n = normalize_vector_np(right_xz)
        cam_fwd_xz_frames.append((float(fwd_xz_n[0]), float(fwd_xz_n[1])))
        cam_right_xz_frames.append((float(right_xz_n[0]), float(right_xz_n[1])))

        cam_fwd_np = normalize_vector_np(to_numpy(xp, fwd))

        u_hat, v_hat, n_hat = _build_slice_basis(
            ro_np=ro_np,
            cam_fwd_np_unit=cam_fwd_np,
            bh_center_np=bh_center_np,
            world_up_np_unit=world_up_np_unit,
        )

        # ---- NEW: stabilize basis to prevent UV mirroring jumps ----
        u_hat, v_hat, n_hat = _stabilize_slice_basis(
            u_hat=u_hat,
            v_hat=v_hat,
            n_hat=n_hat,
            world_up_unit=world_up_np_unit,
            prev_u_hat=prev_u_hat,
            prev_v_hat=prev_v_hat,
        )
        prev_u_hat = u_hat.copy()
        prev_v_hat = v_hat.copy()
        # -----------------------------------------------------------

        dirs = _slice_fan_dirs_xp(
            xp=xp,
            v_hat_np=v_hat,
            u_hat_np=u_hat,
            fov_x_deg=fov_horizontal_deg,
            num_rays=int(DEBUG_NUM_RAYS_SLICE),
        )

        rays_uv: list[np.ndarray] = []
        for d in dirs:
            res = ray_marcher.trace(ro, d)
            pts = to_numpy(xp, res.points)
            if pts.shape[0] > 2 and int(DECIMATE_STRIDE) > 1:
                pts = pts[:: int(DECIMATE_STRIDE)]

            uv = np.zeros((pts.shape[0], 2), dtype=np.float64)
            for k in range(pts.shape[0]):
                uv[k, 0], uv[k, 1] = _project_uv(pts[k], ro_np, u_hat, v_hat)
            rays_uv.append(uv)

        rays_uv_frames.append(rays_uv)
        slice_basis_frames.append((ro_np, u_hat, v_hat, n_hat))

    if not TWO_D_SPLIT:
        raise RuntimeError("This demo variant expects TWO_D_SPLIT=True.")

    fig, (ax_img, ax_uv, ax_xz) = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.suptitle("Newtonian demo: image + TRUE 2D slice debug + world XZ overview")

    # Left: image
    im_artist = ax_img.imshow(imgs[0], origin="upper")
    ax_img.set_title("Rendered image")
    ax_img.set_axis_off()

    # Middle: slice UV
    ax_uv.set_title(RIGHT_PLOT_TITLE)
    ax_uv.set_xlabel("u (slice-plane axis)")
    ax_uv.set_ylabel("v (slice-plane axis)")
    ax_uv.set_xlim(*ULIM)
    ax_uv.set_ylim(*VLIM)
    ax_uv.set_aspect("equal", adjustable="box")

    ax_uv.scatter([0.0], [0.0], s=80, marker="x", linewidths=2, zorder=6)

    # Debug: actual object centers projected into UV slice coordinates
    bh_uv_pt = ax_uv.scatter([], [], s=40, marker="o", zorder=9)
    sphere_uv_pt = ax_uv.scatter([], [], s=40, marker="o", zorder=9)

    uv_v_ann = ax_uv.annotate(
        "",
        xy=(0.0, float(DEBUG_ARROW_LEN)),
        xytext=(0.0, 0.0),
        arrowprops={"arrowstyle": "->", "mutation_scale": 10, "lw": 2},
        zorder=7,
    )
    uv_u_ann = ax_uv.annotate(
        "",
        xy=(float(DEBUG_ARROW_LEN), 0.0),
        xytext=(0.0, 0.0),
        arrowprops={"arrowstyle": "->", "mutation_scale": 10, "lw": 2},
        zorder=7,
    )
    ax_uv.text(0.0, float(DEBUG_ARROW_LEN), "  +v (slice)", fontsize=9, zorder=8)
    ax_uv.text(float(DEBUG_ARROW_LEN), 0.0, "  +u (slice)", fontsize=9, zorder=8)

    sphere_line, = ax_uv.plot([], [], linewidth=2, label="Sphere ∩ slice plane")
    horizon_line, = ax_uv.plot([], [], linewidth=2, label="Horizon ∩ slice plane")

    ray_lines = [
        ax_uv.plot([], [], linewidth=1.5, alpha=0.9)[0]
        for _ in range(int(DEBUG_NUM_RAYS_SLICE))
    ]

    info = ax_uv.text(
        0.02,
        0.98,
        "",
        transform=ax_uv.transAxes,
        va="top",
        fontsize=9,
        family="monospace",
    )

    ax_uv.legend(loc="lower left")

    # Right: world XZ overview
    ax_xz.set_title(XZ_TITLE)
    ax_xz.set_xlabel("x")
    ax_xz.set_ylabel("z")
    ax_xz.set_xlim(*XZ_XLIM)
    ax_xz.set_ylim(*XZ_ZLIM)
    ax_xz.set_aspect("equal", adjustable="box")

    ax_xz.text(
        0.02,
        0.98,
        "arrows = camera forward/right projected to XZ",
        transform=ax_xz.transAxes,
        va="top",
        fontsize=9,
    )

    th = np.linspace(0.0, 2.0 * np.pi, 400, endpoint=True)

    sx = SPHERE_CENTER_NP[0] + float(SPHERE_RADIUS) * np.cos(th)
    sz = SPHERE_CENTER_NP[2] + float(SPHERE_RADIUS) * np.sin(th)
    ax_xz.plot(sx, sz, linewidth=2, label="Sphere (XZ)")

    hx = BH_CENTER_NP[0] + float(BH_HORIZON) * np.cos(th)
    hz = BH_CENTER_NP[2] + float(BH_HORIZON) * np.sin(th)
    ax_xz.plot(hx, hz, linewidth=2, label="Horizon (XZ)")

    cam_trail_line, = ax_xz.plot([], [], linewidth=1.5, alpha=0.8, label="camera trail")
    cam_point = ax_xz.scatter([], [], s=60, marker="x", linewidths=2, zorder=5, label="camera")

    cam_quiver_fwd = ax_xz.quiver(
        [0.0], [0.0], [0.0], [0.0],
        angles="xy", scale_units="xy", scale=1.0,
        width=0.005, zorder=6,
    )
    cam_quiver_right = ax_xz.quiver(
        [0.0], [0.0], [0.0], [0.0],
        angles="xy", scale_units="xy", scale=1.0,
        width=0.005, zorder=6,
    )

    ax_xz.legend(loc="lower left")

    def _update_object_intersections(
            ro_np_: np.ndarray,
            u_hat_: np.ndarray,
            v_hat_: np.ndarray,
            n_hat_: np.ndarray,
    ) -> None:
        plane_point = ro_np_

        sph = _sphere_plane_intersection_circle(
            sphere_center=SPHERE_CENTER_NP,
            sphere_radius=float(SPHERE_RADIUS),
            plane_point=plane_point,
            plane_n_unit=n_hat_,
        )
        if sph is None:
            sphere_line.set_data([], [])
        else:
            c_center, c_r = sph
            pts = _circle_points_in_plane(
                circle_center=c_center,
                circle_radius=c_r,
                plane_u_unit=u_hat_,
                plane_v_unit=v_hat_,
                n_samples=int(INTERSECTION_SAMPLES),
            )
            uv = np.zeros((pts.shape[0], 2), dtype=np.float64)
            for k in range(pts.shape[0]):
                uv[k, 0], uv[k, 1] = _project_uv(pts[k], ro_np_, u_hat_, v_hat_)
            sphere_line.set_data(uv[:, 0], uv[:, 1])

        horizon_r = float(scene.bh.horizon)
        if horizon_r <= 0.0:
            horizon_line.set_data([], [])
            return

        bh_int = _sphere_plane_intersection_circle(
            sphere_center=BH_CENTER_NP,
            sphere_radius=horizon_r,
            plane_point=plane_point,
            plane_n_unit=n_hat_,
        )
        if bh_int is None:
            horizon_line.set_data([], [])
        else:
            c_center, c_r = bh_int
            pts = _circle_points_in_plane(
                circle_center=c_center,
                circle_radius=c_r,
                plane_u_unit=u_hat_,
                plane_v_unit=v_hat_,
                n_samples=int(INTERSECTION_SAMPLES),
            )
            uv = np.zeros((pts.shape[0], 2), dtype=np.float64)
            for k in range(pts.shape[0]):
                uv[k, 0], uv[k, 1] = _project_uv(pts[k], ro_np_, u_hat_, v_hat_)
            horizon_line.set_data(uv[:, 0], uv[:, 1])

    def update(frame_i: int):
        im_artist.set_data(imgs[frame_i])

        # XZ overview
        cx, cz = cam_pos_xz_frames[frame_i]
        fx, fz = cam_fwd_xz_frames[frame_i]
        rx, rz = cam_right_xz_frames[frame_i]

        trail = cam_pos_xz_frames[: frame_i + 1]
        trail_x = [p[0] for p in trail]
        trail_z = [p[1] for p in trail]
        cam_trail_line.set_data(trail_x, trail_z)
        cam_point.set_offsets([[cx, cz]])

        cam_quiver_fwd.set_offsets([[cx, cz]])
        cam_quiver_fwd.set_UVC([fx * float(XZ_ARROW_LEN)], [fz * float(XZ_ARROW_LEN)])
        cam_quiver_right.set_offsets([[cx, cz]])
        cam_quiver_right.set_UVC([rx * float(XZ_ARROW_LEN)], [rz * float(XZ_ARROW_LEN)])

        ro_np_, u_hat_, v_hat_, n_hat_ = slice_basis_frames[frame_i]
        _update_object_intersections(ro_np_, u_hat_, v_hat_, n_hat_)

        # Project centers to UV
        bh_u, bh_v = _project_uv(BH_CENTER_NP, ro_np_, u_hat_, v_hat_)
        sp_u, sp_v = _project_uv(SPHERE_CENTER_NP, ro_np_, u_hat_, v_hat_)
        bh_uv_pt.set_offsets([[bh_u, bh_v]])
        sphere_uv_pt.set_offsets([[sp_u, sp_v]])

        rays_uv_ = rays_uv_frames[frame_i]
        for j, line in enumerate(ray_lines):
            if j >= len(rays_uv_):
                line.set_data([], [])
                continue
            uv_ = rays_uv_[j]
            line.set_data(uv_[:, 0], uv_[:, 1])

        info.set_text(
            f"frame={frame_i}/{frames_n - 1}\n"
            f"cam_pos={ro_np_}\n"
            f"u_hat={u_hat_}\n"
            f"v_hat={v_hat_}\n"
            f"n_hat={n_hat_}\n"
            "plane: through cam pos and BH center\n"
            "u=dot(p-cam_pos, u_hat)\n"
            "v=dot(p-cam_pos, v_hat)",
        )

        return [
            im_artist,
            sphere_line,
            horizon_line,
            bh_uv_pt,
            sphere_uv_pt,
            *ray_lines,
            uv_v_ann,
            uv_u_ann,
            cam_trail_line,
            cam_point,
            cam_quiver_fwd,
            cam_quiver_right,
        ]

    ani = FuncAnimation(
        fig,
        update,
        frames=frames_n,
        interval=int(INTERVAL_MS),
        blit=False,
        repeat=True,
    )
    _ = ani
    plt.show()


if __name__ == "__main__":
    main()
