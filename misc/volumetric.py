import time

import matplotlib
matplotlib.use("TkAgg")  # must be before pyplot

import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# BACKEND (NumPy or CuPy)
# ============================================================
# Options:
#   BACKEND = "auto"  -> try CuPy, fallback to NumPy
#   BACKEND = "numpy" -> force CPU
#   BACKEND = "cupy"  -> force GPU (requires cupy)
BACKEND = "auto"

cp = None
if BACKEND in ("auto", "cupy"):
    try:
        import cupy as cp  # type: ignore
    except Exception:
        cp = None

if BACKEND == "cupy" and cp is None:
    raise RuntimeError("BACKEND='cupy' requested but CuPy is not available.")

xp = cp if (cp is not None and BACKEND in ("auto", "cupy")) else np


def to_numpy(a):
    """Convert xp array to numpy for matplotlib."""
    if cp is not None and isinstance(a, cp.ndarray):
        return cp.asnumpy(a)
    return a


def host_scalar(x):
    """Safely get a Python float from numpy/cupy scalars."""
    try:
        return float(x)
    except Exception:
        if cp is not None and isinstance(x, cp.ndarray):
            return float(cp.asnumpy(x).item())
        return float(np.asarray(x).item())


# ============================================================
# GLOBAL SCENE PARAMETERS (UPPERCASE)
# ============================================================

# ---- Image / Resolution ----
IMAGE_WIDTH = 720
IMAGE_ASPECT = 16 / 9
IMAGE_HEIGHT = int(round(IMAGE_WIDTH / IMAGE_ASPECT))
IMAGE_ASPECT = IMAGE_WIDTH / IMAGE_HEIGHT  # enforce exact aspect

# ---- Light ----
LIGHT_POSITION = xp.array([-3.0, 2.2, 4.6], dtype=xp.float64)
LIGHT_INTENSITY = 40.0

# ---- Camera ----
CAMERA_POSITION = xp.array([2.6, 4.0, 0.4], dtype=xp.float64)
CAMERA_TARGET = xp.array([0.0, 0.7, 4.5], dtype=xp.float64)
CAMERA_UP_HINT = xp.array([0.0, 1.0, 0.0], dtype=xp.float64)

# ---- Discretization (Cloud sampling density) ----
# Smaller DS = more points (denser sampling), but we keep distance constant via TMAX.
RAY_DS = 0.1
RAY_TMAX = 12.0
RAY_STEPS = int(np.ceil(RAY_TMAX / RAY_DS))

SHADOW_DS = 0.1
SHADOW_TMAX = 10.0
SHADOW_STEPS = int(np.ceil(SHADOW_TMAX / SHADOW_DS))
SHADOW_EVERY = 2

# ---- Medium / Cloud density (PHYSICAL meaning) ----
# Higher values -> more extinction (darker, stronger shadows), lower values -> more transparent.
FOG_DENSITY = 0.01
CLOUD_DENSITY = 1.2
EDGE_SOFTNESS = 0.12

# Optional: adjust overall "glow strength" independent of density
EMISSION_STRENGTH = 1.0


# ============================================================
# Helpers
# ============================================================
def normalize(v, axis=-1, eps=1e-12):
    n = xp.sqrt(xp.sum(v * v, axis=axis, keepdims=True))
    return v / (n + eps)


def smoothstep(edge0, edge1, x):
    t = xp.clip((x - edge0) / (edge1 - edge0 + 1e-12), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def hash01_from_ij(h, w):
    """
    Deterministic 0..1 hash per pixel.
    Used for jitter to remove banding from discretized ray steps.
    Works on NumPy and CuPy.
    """
    yy, xx = xp.mgrid[0:h, 0:w]
    return xp.mod(xp.sin(xx * 12.9898 + yy * 78.233) * 43758.5453, 1.0)


def camera_basis(forward, up_hint):
    fwd = normalize(xp.asarray(forward, dtype=xp.float64))
    uph = normalize(xp.asarray(up_hint, dtype=xp.float64))

    if abs(host_scalar(xp.sum(fwd * uph))) > 0.999:
        uph = xp.array([0.0, 0.0, 1.0], dtype=xp.float64)

    right = normalize(xp.cross(fwd, uph))
    up = normalize(xp.cross(right, fwd))
    return fwd, right, up


def camera_rays(ro, forward, up_hint, w, h, fov_x_deg, fov_y_deg):
    fwd, right, up = camera_basis(forward, up_hint)

    half_x = xp.deg2rad(fov_x_deg) * 0.5
    half_y = xp.deg2rad(fov_y_deg) * 0.5
    tx = xp.tan(half_x)
    ty = xp.tan(half_y)

    xs = xp.linspace(-tx, tx, w, dtype=xp.float64)
    ys = xp.linspace(ty, -ty, h, dtype=xp.float64)

    rd = (
        fwd[None, None, :]
        + xs[None, :, None] * right[None, None, :]
        + ys[:, None, None] * up[None, None, :]
    )
    return normalize(rd), (fwd, right, up), (tx, ty)


def project_point_to_pixel(ro, basis, tan_half, p_world, width, height):
    fwd, right, up = basis
    tx, ty = tan_half

    v = p_world - ro
    z = host_scalar(xp.sum(v * fwd))
    if z <= 1e-8:
        return None

    x_cam = host_scalar(xp.sum(v * right))
    y_cam = host_scalar(xp.sum(v * up))

    x = x_cam / z
    y = y_cam / z

    u = (x / (host_scalar(tx) + 1e-12) + 1.0) * 0.5
    v2 = 1.0 - (y / (host_scalar(ty) + 1e-12) + 1.0) * 0.5

    px = int(u * width)
    py = int(v2 * height)

    if px < 0 or px >= width or py < 0 or py >= height:
        return None
    return px, py


# ============================================================
# Medium: exact cube
# ============================================================
def box_density(p, box_min, box_max, edge_soft=EDGE_SOFTNESS):
    dx0 = box_min[0] - p[..., 0]
    dx1 = p[..., 0] - box_max[0]
    dy0 = box_min[1] - p[..., 1]
    dy1 = p[..., 1] - box_max[1]
    dz0 = box_min[2] - p[..., 2]
    dz1 = p[..., 2] - box_max[2]

    a0 = xp.maximum(dx0, 0.0)
    a1 = xp.maximum(dx1, 0.0)
    a2 = xp.maximum(dy0, 0.0)
    a3 = xp.maximum(dy1, 0.0)
    a4 = xp.maximum(dz0, 0.0)
    a5 = xp.maximum(dz1, 0.0)

    outside = xp.maximum(
        xp.maximum(xp.maximum(a0, a1), xp.maximum(a2, a3)),
        xp.maximum(a4, a5),
    )

    return smoothstep(edge_soft, 0.0, outside)


def sigma_t(p, box_min, box_max):
    return FOG_DENSITY + CLOUD_DENSITY * box_density(p, box_min, box_max)


def emission_base(p, box_min, box_max):
    center = 0.5 * (box_min + box_max)
    r = xp.sqrt(xp.sum((p - center) * (p - center), axis=-1))
    core = 1.0 - smoothstep(0.0, 1.5, r)
    return EMISSION_STRENGTH * box_density(p, box_min, box_max) * (0.4 + 0.6 * core)


# ============================================================
# Shadow march
# ============================================================
def transmittance_to_light(
    x,
    box_min,
    box_max,
    ds_light=SHADOW_DS,
    max_steps=SHADOW_STEPS,
):
    v = LIGHT_POSITION - x
    dist = xp.sqrt(xp.sum(v * v, axis=-1))
    dirL = v / (dist[..., None] + 1e-12)

    steps = xp.minimum(max_steps, xp.ceil(dist / ds_light).astype(xp.int32))

    T = xp.ones(dist.shape, dtype=xp.float64)
    p = x.copy()

    for si in range(int(max_steps)):
        active = si < steps
        if not host_scalar(xp.any(active)):
            break
        p = p + dirL * (ds_light * active[..., None])
        st = sigma_t(p, box_min, box_max)
        T *= xp.exp(-st * ds_light * active)
        if host_scalar(xp.all(T < 1e-4)):
            break

    return T


# ============================================================
# Volumetric renderer
# ============================================================
def render(
    fov_x_deg=80.0,
    fov_y_deg=None,
    cube_center=(0.0, 0.8, 4.6),
    cube_size=2.4,
):
    width = IMAGE_WIDTH
    height = IMAGE_HEIGHT
    aspect = IMAGE_ASPECT

    if fov_y_deg is None:
        fov_y_deg = host_scalar(
            xp.rad2deg(2.0 * xp.arctan(xp.tan(xp.deg2rad(fov_x_deg) * 0.5) / aspect))
        )

    forward = CAMERA_TARGET - CAMERA_POSITION

    rd0, basis, tan_half = camera_rays(
        CAMERA_POSITION,
        forward,
        CAMERA_UP_HINT,
        width,
        height,
        fov_x_deg,
        fov_y_deg,
    )

    center = xp.array(cube_center, dtype=xp.float64)
    half = 0.5 * cube_size
    box_min = center - half
    box_max = center + half

    medium_color = xp.array([1.0, 0.75, 0.25], dtype=xp.float64)

    L = xp.zeros((height, width, 3), dtype=xp.float64)
    Tcam = xp.ones((height, width), dtype=xp.float64)

    # Per-pixel jitter in [0, RAY_DS) to break banding
    j01 = hash01_from_ij(height, width).astype(xp.float64)
    t0 = j01 * RAY_DS

    p = xp.broadcast_to(CAMERA_POSITION, (height, width, 3)).copy()
    p = p + rd0 * t0[..., None]

    Tlight_cache = xp.ones((height, width), dtype=xp.float64)

    for k in range(int(RAY_STEPS)):
        p = p + rd0 * RAY_DS

        st = sigma_t(p, box_min, box_max)

        if (k % SHADOW_EVERY) == 0:
            Tlight_cache = transmittance_to_light(
                p,
                box_min,
                box_max,
            )

        d = LIGHT_POSITION - p
        dL = xp.sqrt(xp.sum(d * d, axis=-1))
        Lin = LIGHT_INTENSITY * Tlight_cache / (dL * dL + 1e-3)

        j = emission_base(p, box_min, box_max) * Lin
        L += Tcam[..., None] * j[..., None] * RAY_DS * medium_color

        Tcam *= xp.exp(-st * RAY_DS)
        if host_scalar(xp.all(Tcam < 1e-4)):
            break

    img = xp.clip(L, 0.0, 1.0)

    px = project_point_to_pixel(
        CAMERA_POSITION, basis, tan_half, LIGHT_POSITION, width, height
    )
    if px is not None:
        lx, ly = px
        yy, xx = xp.ogrid[:height, :width]
        img[(xx - lx) ** 2 + (yy - ly) ** 2 <= 36] = 1.0

    return img.astype(xp.float32)


def main():
    t0 = time.perf_counter()
    img = render()

    if cp is not None and isinstance(img, cp.ndarray):
        cp.cuda.Device().synchronize()

    elapsed = time.perf_counter() - t0
    backend = "cupy" if (cp is not None and isinstance(img, cp.ndarray)) else "numpy"

    print(f"Render time [{backend}]: {elapsed:.3f} s")
    print(
        f"RAY_DS={RAY_DS} RAY_TMAX={RAY_TMAX} RAY_STEPS={RAY_STEPS} | "
        f"SHADOW_DS={SHADOW_DS} SHADOW_TMAX={SHADOW_TMAX} SHADOW_STEPS={SHADOW_STEPS} SHADOW_EVERY={SHADOW_EVERY} | "
        f"FOG_DENSITY={FOG_DENSITY} CLOUD_DENSITY={CLOUD_DENSITY} EDGE_SOFTNESS={EDGE_SOFTNESS}"
    )

    plt.figure(figsize=(10, 5))
    plt.axis("off")
    plt.title(f"Volumetric cube - backend: {backend} - {elapsed:.3f}s")
    plt.imshow(to_numpy(img), interpolation="nearest")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
