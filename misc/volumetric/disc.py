import time

import matplotlib

matplotlib.use("TkAgg")  # must be before pyplot

import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# BACKEND (NumPy or CuPy)
# ============================================================
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
IMAGE_ASPECT = IMAGE_WIDTH / IMAGE_HEIGHT

# ---- Light (CENTER) ----
LIGHT_POSITION = xp.array([0.0, 0.0, 0.0], dtype=xp.float64)
LIGHT_INTENSITY = 100.0
LIGHT_R2_EPS = 1e-2  # bigger epsilon reduces harsh hotspot at center

# ---- Camera (3/4 view, matches your occupancy mask camera) ----
CAMERA_POSITION = xp.array([7.5, 4.5, 7.5], dtype=xp.float64)
CAMERA_TARGET = xp.array([0.0, 0.0, 0.0], dtype=xp.float64)
CAMERA_UP_HINT = xp.array([0.0, 1.0, 0.0], dtype=xp.float64)

# ---- FOV ----
FOV_X_DEG = 75.0

# ---- Disk Geometry (XZ plane, thickness in Y) ----
DISK_INNER_RADIUS = 1.5
DISK_OUTER_RADIUS = 4.2
DISK_HALF_THICKNESS = 0.15

DISK_EDGE_SOFT_R = 0.4
DISK_EDGE_SOFT_Y = 0.12

# ---- Discretization ----
RAY_DS = 0.05
RAY_TMAX = 30.0
RAY_STEPS = int(np.ceil(RAY_TMAX / RAY_DS))

SHADOW_DS = 0.05
SHADOW_TMAX = 12.0
SHADOW_STEPS = int(np.ceil(SHADOW_TMAX / SHADOW_DS))
SHADOW_EVERY = 2

# ---- Medium / Physical density ----
FOG_DENSITY = 0.0
DISK_DENSITY = 1.0

# ---- Emission / Appearance ----
EMISSION_STRENGTH = 4.0
MEDIUM_COLOR = xp.array([1.0, 0.65, 0.20], dtype=xp.float64)

# ---- Debug: solid object overlay (occupancy mask) ----
DEBUG_SOLID_OVERLAY = False
DEBUG_SOLID_THRESHOLD = 0.01  # density threshold for "solid exists"
OPACITY_DEBUG = 0.25  # 0..1, how strong the white overlay is


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
    return normalize(rd)


# ============================================================
# Disk density and optical properties
# ============================================================
def disk_density(p):
    x = p[..., 0]
    y = p[..., 1]
    z = p[..., 2]

    r = xp.sqrt(x * x + z * z)
    ay = xp.abs(y)

    rin0 = DISK_INNER_RADIUS
    rin1 = DISK_INNER_RADIUS + DISK_EDGE_SOFT_R
    rout0 = DISK_OUTER_RADIUS - DISK_EDGE_SOFT_R
    rout1 = DISK_OUTER_RADIUS

    in_r = smoothstep(rin0, rin1, r) * (1.0 - smoothstep(rout0, rout1, r))

    h0 = DISK_HALF_THICKNESS
    h1 = DISK_HALF_THICKNESS + DISK_EDGE_SOFT_Y
    in_y = 1.0 - smoothstep(h0, h1, ay)

    return xp.clip(in_r * in_y, 0.0, 1.0)


def sigma_t(p):
    return FOG_DENSITY + DISK_DENSITY * disk_density(p)


def emission_base(p):
    dens = disk_density(p)
    x = p[..., 0]
    z = p[..., 2]
    r = xp.sqrt(x * x + z * z)

    t = xp.clip(
        (r - DISK_INNER_RADIUS) / (DISK_OUTER_RADIUS - DISK_INNER_RADIUS + 1e-12),
        0.0,
        1.0,
    )
    inner_hot = (1.0 - t) ** 2
    return EMISSION_STRENGTH * dens * (0.20 + 0.80 * inner_hot)


# ============================================================
# Shadow transmittance
# ============================================================
def transmittance_to_light(x):
    v = LIGHT_POSITION - x
    dist = xp.sqrt(xp.sum(v * v, axis=-1))
    dirL = v / (dist[..., None] + 1e-12)

    steps = xp.minimum(SHADOW_STEPS, xp.ceil(dist / SHADOW_DS).astype(xp.int32))

    T = xp.ones(dist.shape, dtype=xp.float64)
    p = x.copy()

    for si in range(int(SHADOW_STEPS)):
        active = si < steps
        if not host_scalar(xp.any(active)):
            break
        p = p + dirL * (SHADOW_DS * active[..., None])
        st = sigma_t(p)
        T *= xp.exp(-st * SHADOW_DS * active)
        if host_scalar(xp.all(T < 1e-4)):
            break

    return T


# ============================================================
# Debug overlay: true "solid exists" mask (raymarch occupancy)
# ============================================================
def solid_occupancy_mask(rd0):
    """True if the camera ray intersects any region where disk_density(p) > threshold.
    This is the same idea as your occupancy script, and is independent of lighting.
    """
    h, w, _ = rd0.shape
    p = xp.broadcast_to(CAMERA_POSITION, (h, w, 3)).copy()

    thr = max(DEBUG_SOLID_THRESHOLD, FOG_DENSITY)
    occ = xp.zeros((h, w), dtype=bool)

    for _ in range(int(RAY_STEPS)):
        p = p + rd0 * RAY_DS
        occ |= (disk_density(p) > thr)
        if host_scalar(xp.all(occ)):
            break

    return occ


# ============================================================
# Volumetric renderer
# ============================================================
def render():
    w = IMAGE_WIDTH
    h = IMAGE_HEIGHT
    aspect = IMAGE_ASPECT

    fov_y_deg = host_scalar(
        xp.rad2deg(2.0 * xp.arctan(xp.tan(xp.deg2rad(FOV_X_DEG) * 0.5) / aspect)),
    )

    forward = CAMERA_TARGET - CAMERA_POSITION
    rd0 = camera_rays(CAMERA_POSITION, forward, CAMERA_UP_HINT, w, h, FOV_X_DEG, fov_y_deg)

    L = xp.zeros((h, w, 3), dtype=xp.float64)
    Tcam = xp.ones((h, w), dtype=xp.float64)

    # jitter start
    j01 = hash01_from_ij(h, w).astype(xp.float64)
    t0 = j01 * RAY_DS

    p = xp.broadcast_to(CAMERA_POSITION, (h, w, 3)).copy()
    p = p + rd0 * t0[..., None]

    Tlight_cache = xp.ones((h, w), dtype=xp.float64)

    for k in range(int(RAY_STEPS)):
        p = p + rd0 * RAY_DS

        st = sigma_t(p)

        if (k % SHADOW_EVERY) == 0:
            Tlight_cache = transmittance_to_light(p)

        d = LIGHT_POSITION - p
        r2 = xp.sum(d * d, axis=-1) + LIGHT_R2_EPS
        Lin = LIGHT_INTENSITY * Tlight_cache / r2

        j = emission_base(p) * Lin

        L += (Tcam[..., None] * j[..., None] * RAY_DS) * MEDIUM_COLOR[None, None, :]

        Tcam *= xp.exp(-st * RAY_DS)
        if host_scalar(xp.all(Tcam < 1e-4)):
            break

    img = xp.clip(L, 0.0, 1.0).astype(xp.float32)

    # Debug: overlay the actual solid mask as semi-transparent white
    if DEBUG_SOLID_OVERLAY and OPACITY_DEBUG > 0.0:
        m = solid_occupancy_mask(rd0)
        a = float(OPACITY_DEBUG)
        white = xp.array([1.0, 1.0, 1.0], dtype=xp.float32)
        img = xp.clip(img * (1.0 - a) + (m[..., None].astype(xp.float32) * a) * white, 0.0, 1.0)

    return img


def main():
    t0 = time.perf_counter()
    img = render()

    if cp is not None and isinstance(img, cp.ndarray):
        cp.cuda.Device().synchronize()

    elapsed = time.perf_counter() - t0
    backend = "cupy" if (cp is not None and isinstance(img, cp.ndarray)) else "numpy"

    print(f"Render time [{backend}]: {elapsed:.3f} s")
    print(
        f"CAMERA_POSITION={to_numpy(CAMERA_POSITION)} CAMERA_TARGET={to_numpy(CAMERA_TARGET)} FOV_X_DEG={FOV_X_DEG} | "
        f"DISK Rin={DISK_INNER_RADIUS} Rout={DISK_OUTER_RADIUS} H={DISK_HALF_THICKNESS} | "
        f"RAY_DS={RAY_DS} RAY_TMAX={RAY_TMAX} RAY_STEPS={RAY_STEPS} | "
        f"SHADOW_DS={SHADOW_DS} SHADOW_TMAX={SHADOW_TMAX} SHADOW_STEPS={SHADOW_STEPS} SHADOW_EVERY={SHADOW_EVERY} | "
        f"DEBUG_SOLID_OVERLAY={DEBUG_SOLID_OVERLAY} OPACITY_DEBUG={OPACITY_DEBUG}",
    )

    plt.figure(figsize=(12, 6.75))
    plt.axis("off")
    plt.title(f"Volumetric disk - backend: {backend} - {elapsed:.3f}s")
    plt.imshow(to_numpy(img), interpolation="nearest")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
