"""
cube_best.py
============
Improved volumetric cube — same ideas applied here as in disk_best.py.

Improvements over the original cube.py
---------------------------------------
1. Analytical AABB ray–box intersection.
   Primary rays start marching exactly where they enter the box, not from
   the camera through metres of empty air.  Skipping empty space is by far
   the biggest performance win.

2. Pre-baked 3D fBm noise volume (NX × NY × NZ, trilinear sampler).
   Gives the cube turbulent, cloud-like density instead of a solid blob.
   Baked once on CPU, uploaded to the compute device, zero per-step cost.

3. Density-based colour LUT (2048 entries, stays on device during march).
   Dense regions → hot white/yellow.  Sparse edges → orange → dark red.
   Replaces the single hard-coded medium_color and avoids CPU↔device
   round-trips inside the hot loop.

4. Correct march initialisation.
   First sample lands at t_in + jitter, not one step beyond it.

5. Shadow rays capped at AABB exit distance.
   No point shadow-marching outside the box where density is always zero.

6. ACES filmic tonemapping + Gaussian bloom on bright core.

NumPy / CuPy compatible (set BACKEND = "cupy" for GPU).
"""

import time

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Backend
# ─────────────────────────────────────────────────────────────────────────────
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
    if cp is not None and isinstance(a, cp.ndarray):
        return cp.asnumpy(a)
    return np.asarray(a)


def host_scalar(x):
    try:
        return float(x)
    except Exception:
        return float(np.asarray(to_numpy(x)).item())


# ─────────────────────────────────────────────────────────────────────────────
# Scene parameters  (all tunable here)
# ─────────────────────────────────────────────────────────────────────────────

# ── Image ──────────────────────────────────────────────────────────────────
IMAGE_WIDTH  = 720
IMAGE_HEIGHT = int(round(IMAGE_WIDTH / (16 / 9)))

# ── Camera ─────────────────────────────────────────────────────────────────
# Same scene layout as cube.py so outputs are directly comparable.
CAMERA_POSITION = xp.array([2.6, 4.0, 0.4], dtype=xp.float64)
CAMERA_TARGET   = xp.array([0.0, 0.7, 4.5], dtype=xp.float64)
CAMERA_UP_HINT  = xp.array([0.0, 1.0, 0.0], dtype=xp.float64)
FOV_X_DEG = 80.0                              # same as cube.py

# ── Light ───────────────────────────────────────────────────────────────────
LIGHT_POSITION  = xp.array([3.0, 2.2, 4.6], dtype=xp.float64)   # same as cube.py
LIGHT_INTENSITY = 60.0    # same as cube.py
LIGHT_R2_EPS    = 1e-3    # same as cube.py — gives the same 1/r² gradient

# ── Cube ────────────────────────────────────────────────────────────────────
CUBE_CENTER    = (0.0, 0.8, 4.6)   # same as cube.py
CUBE_SIZE      = 2.4               # same as cube.py
EDGE_SOFTNESS  = 0.12              # same as cube.py

# ── Medium ───────────────────────────────────────────────────────────────────
CLOUD_DENSITY     = 1.2    # same extinction as cube.py
EMISSION_STRENGTH = 1.0    # same as cube.py — the ACES/bloom handle the HDR
T_THRESH          = 1e-4   # early-exit transmittance

# ── Colour ramp (density_norm in [0,1] → RGB) ─────────────────────────────
# Stays in the warm orange/yellow family so the dominant brightness variation
# comes from the LIGHT source (directional), not from the colour darkening.
# With NOISE_STRENGTH=0: density≈1 everywhere → all warm yellow, light creates
#   the bright-side / dark-side gradient exactly as in cube.py.
# With NOISE_STRENGTH>0: dense patches become yellower, sparse become orange.
COLOR_RAMP = np.array([
    [1.00, 1.00, 0.85, 0.35],   # dense / hot:  warm yellow
    [0.60, 1.00, 0.65, 0.15],   # medium:       orange-yellow
    [0.25, 0.90, 0.38, 0.04],   # sparse:       deep orange
    [0.00, 0.55, 0.16, 0.00],   # very sparse:  burnt orange  (not black)
], dtype=np.float64)

# ── Radial core-glow (same formula as cube.py emission_base) ──────────────
# emission × (CORE_LOW + (1-CORE_LOW) × core_factor)
# CORE_LOW=0.7 means the centre is at most 1.4× brighter than edges,
# so the directional light from Lin stays the dominant brightness signal.
CORE_LOW    = 0.7    # cube.py used 0.4; higher = subtler radial glow
CORE_RADIUS = 1.5    # same as cube.py smoothstep(0, 1.5, r)

# ── 3D noise volume (pre-baked, fast trilinear sampler) ───────────────────
NOISE_NX       = 64
NOISE_NY       = 64
NOISE_NZ       = 64
NOISE_OCTAVES  = 5
NOISE_GAIN     = 0.50
NOISE_LACUN    = 2.0
NOISE_CONTRAST = 2.0    # >1 sharpens dense filaments
NOISE_STRENGTH = 0.0    # 0 = solid uniform cube (like cube.py), 0.7-0.8 = turbulent cloud
NOISE_MARGIN   = 0.5

# ── Ray marching ─────────────────────────────────────────────────────────────
DS_FINE     = 0.04    # finer than cube.py's 0.1 — we only march inside the AABB
                       # so total work is similar despite the smaller step size
N_STEPS_MAX = 500

# ── Shadow rays ──────────────────────────────────────────────────────────────
SHADOW_DS    = 0.07   # finer than cube.py's 0.1 — capped at AABB exit
SHADOW_STEPS = 80     # enough to cross the diagonal; AABB caps it earlier
SHADOW_EVERY = 2      # same as cube.py

# ── Post-processing ───────────────────────────────────────────────────────────
TONEMAP_EXPOSURE = 1.2    # slight boost — matches cube.py brightness after ACES+gamma
BLOOM_SIGMA      = 2.5    # Gaussian bloom radius in pixels
BLOOM_STRENGTH   = 0.35   # fraction of bloom added back

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def normalize(v, axis=-1, eps=1e-12):
    n = xp.sqrt(xp.sum(v * v, axis=axis, keepdims=True))
    return v / (n + eps)


def smoothstep(e0, e1, x):
    t = xp.clip((x - e0) / (e1 - e0 + 1e-12), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def aces_tonemap(x):
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return xp.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)


def camera_rays(w, h):
    fwd = normalize(CAMERA_TARGET - CAMERA_POSITION)
    uph = normalize(CAMERA_UP_HINT)
    if abs(host_scalar(xp.dot(fwd.ravel(), uph.ravel()))) > 0.999:
        uph = xp.array([0.0, 0.0, 1.0], dtype=xp.float64)
    right = normalize(xp.cross(fwd, uph))
    up    = normalize(xp.cross(right, fwd))
    tx    = float(np.tan(np.deg2rad(FOV_X_DEG) * 0.5))
    ty    = tx * h / w
    xs = xp.linspace(-tx,  tx, w, dtype=xp.float64)
    ys = xp.linspace( ty, -ty, h, dtype=xp.float64)
    rd = (fwd[None, None, :]
          + xs[None, :, None] * right[None, None, :]
          + ys[:, None, None] * up[None, None, :])
    return normalize(rd)


def project_point_to_pixel(p_world, w, h):
    """
    Project a world-space point onto the image plane.
    Returns (px, py) pixel coordinates or None if behind / outside frame.
    Same algorithm as cube.py's project_point_to_pixel.
    """
    fwd = normalize(CAMERA_TARGET - CAMERA_POSITION)
    uph = normalize(CAMERA_UP_HINT)
    if abs(host_scalar(xp.dot(fwd.ravel(), uph.ravel()))) > 0.999:
        uph = xp.array([0.0, 0.0, 1.0], dtype=xp.float64)
    right = normalize(xp.cross(fwd, uph))
    up    = normalize(xp.cross(right, fwd))
    tx    = float(np.tan(np.deg2rad(FOV_X_DEG) * 0.5))
    ty    = tx * h / w

    v   = p_world - CAMERA_POSITION
    z   = host_scalar(xp.sum(v * fwd))
    if z <= 1e-8:
        return None
    x_c = host_scalar(xp.sum(v * right)) / z
    y_c = host_scalar(xp.sum(v * up))    / z
    u   = (x_c / tx  + 1.0) * 0.5
    v2  = 1.0 - (y_c / ty + 1.0) * 0.5
    px  = int(u * w)
    py  = int(v2 * h)
    if 0 <= px < w and 0 <= py < h:
        return px, py
    return None


# ─────────────────────────────────────────────────────────────────────────────
# AABB ray–box intersection
# ─────────────────────────────────────────────────────────────────────────────

def aabb_intersect(ro, rd, bmin, bmax):
    """
    Analytically intersect rays with an axis-aligned bounding box.
    Works for rays starting outside OR inside the box.

    ro, rd : (H, W, 3) or (3,) broadcast-compatible arrays
    bmin, bmax : (3,) world-space box corners
    Returns (t_in, t_out) : (H, W) each.
        t_in  = entry distance  (clipped to 0 if camera is inside)
        t_out = exit distance
        Hit condition: t_in < t_out and t_out > 0
    """
    # Guard against zero ray components
    safe_rd = xp.where(xp.abs(rd) < 1e-15, xp.full_like(rd, 1e-15), rd)
    inv = 1.0 / safe_rd

    t1 = (float(bmin[0]) - ro[..., 0]) * inv[..., 0]
    t2 = (float(bmax[0]) - ro[..., 0]) * inv[..., 0]
    t3 = (float(bmin[1]) - ro[..., 1]) * inv[..., 1]
    t4 = (float(bmax[1]) - ro[..., 1]) * inv[..., 1]
    t5 = (float(bmin[2]) - ro[..., 2]) * inv[..., 2]
    t6 = (float(bmax[2]) - ro[..., 2]) * inv[..., 2]

    t_in  = xp.maximum(xp.maximum(xp.minimum(t1, t2),
                                   xp.minimum(t3, t4)),
                        xp.minimum(t5, t6))
    t_out = xp.minimum(xp.minimum(xp.maximum(t1, t2),
                                   xp.maximum(t3, t4)),
                        xp.maximum(t5, t6))

    t_in = xp.clip(t_in, 0.0, None)   # camera inside box → entry at 0
    return t_in, t_out


# ─────────────────────────────────────────────────────────────────────────────
# Colour LUT (pre-baked, stays on device during march)
# ─────────────────────────────────────────────────────────────────────────────

_LUT_N    = 2048
_COLOR_LUT: "xp.ndarray | None" = None   # (LUT_N, 3)


def build_color_lut() -> None:
    global _COLOR_LUT
    t_vals = np.linspace(0.0, 1.0, _LUT_N, dtype=np.float64)
    stops  = COLOR_RAMP[:, 0]
    colors = COLOR_RAMP[:, 1:]
    n      = len(stops)
    lut    = np.zeros((_LUT_N, 3), dtype=np.float64)

    for i, t in enumerate(t_vals):
        if t >= stops[0]:
            lut[i] = colors[0]; continue
        if t <= stops[-1]:
            lut[i] = colors[-1]; continue
        for j in range(n - 1):
            t_hi, t_lo = stops[j], stops[j + 1]
            if t_lo <= t <= t_hi:
                alpha  = (t - t_lo) / (t_hi - t_lo + 1e-12)
                lut[i] = colors[j + 1] + alpha * (colors[j] - colors[j + 1])
                break

    _COLOR_LUT = xp.asarray(lut.astype(np.float64))


def color_from_density(density_norm):
    """
    Look up emission colour from normalised density (0..1).
    Fully vectorised, stays on the compute device.
    density_norm : (H, W) in [0, 1]
    Returns      : (H, W, 3)
    """
    idx_f = xp.clip(density_norm, 0.0, 1.0) * (_LUT_N - 1)
    idx0  = xp.clip(idx_f.astype(xp.int32), 0, _LUT_N - 2)
    idx1  = idx0 + 1
    frac  = (idx_f - idx0.astype(xp.float64))[..., None]
    return _COLOR_LUT[idx0] * (1.0 - frac) + _COLOR_LUT[idx1] * frac


# ─────────────────────────────────────────────────────────────────────────────
# 3D noise volume (pre-baked on CPU, uploaded once)
# ─────────────────────────────────────────────────────────────────────────────

_NOISE_VOL    = None   # (NZ, NY, NX) float32 on compute device
_NOISE_BOUNDS = None   # (x0,x1,y0,y1,z0,z1) world-space extents


def _hash3(ix, iy, iz):
    return np.mod(np.sin(ix * 127.1 + iy * 311.7 + iz * 74.7) * 43758.5453123, 1.0)


def _value3(px, py, pz):
    x0 = np.floor(px).astype(np.int32); y0 = np.floor(py).astype(np.int32)
    z0 = np.floor(pz).astype(np.int32)
    x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1
    fx, fy, fz = px - x0, py - y0, pz - z0
    u = fx*fx*(3 - 2*fx); v = fy*fy*(3 - 2*fy); w = fz*fz*(3 - 2*fz)
    c000=_hash3(x0,y0,z0); c100=_hash3(x1,y0,z0)
    c010=_hash3(x0,y1,z0); c110=_hash3(x1,y1,z0)
    c001=_hash3(x0,y0,z1); c101=_hash3(x1,y0,z1)
    c011=_hash3(x0,y1,z1); c111=_hash3(x1,y1,z1)
    x00 = c000*(1-u)+c100*u; x10 = c010*(1-u)+c110*u
    x01 = c001*(1-u)+c101*u; x11 = c011*(1-u)+c111*u
    y0v = x00*(1-v)+x10*v;   y1v = x01*(1-v)+x11*v
    return np.clip(y0v*(1-w)+y1v*w, 0.0, 1.0)


def _fbm3(px, py, pz, octaves, lacun, gain):
    amp, freq, s, norm = 0.5, 1.0, np.zeros_like(px), 0.0
    for _ in range(octaves):
        s    += amp * _value3(px*freq, py*freq, pz*freq)
        norm += amp; amp *= gain; freq *= lacun
    return np.clip(s / (norm + 1e-12), 0.0, 1.0)


def build_noise_volume() -> None:
    global _NOISE_VOL, _NOISE_BOUNDS
    t0 = time.perf_counter()

    half = CUBE_SIZE * 0.5 + NOISE_MARGIN
    cx, cy, cz = CUBE_CENTER
    x0, x1 = cx - half, cx + half
    y0, y1 = cy - half, cy + half
    z0, z1 = cz - half, cz + half
    _NOISE_BOUNDS = (x0, x1, y0, y1, z0, z1)

    # World-space grid positions
    iz, iy, ix = np.mgrid[0:NOISE_NZ, 0:NOISE_NY, 0:NOISE_NX]
    wx = x0 + (x1 - x0) * ix / (NOISE_NX - 1)
    wy = y0 + (y1 - y0) * iy / (NOISE_NY - 1)
    wz = z0 + (z1 - z0) * iz / (NOISE_NZ - 1)

    n = _fbm3(wx, wy, wz, NOISE_OCTAVES, NOISE_LACUN, NOISE_GAIN)
    n = np.power(np.clip(n, 0.0, 1.0), NOISE_CONTRAST)

    _NOISE_VOL = xp.asarray(n.astype(np.float32))
    print(f"  Noise volume {NOISE_NX}×{NOISE_NY}×{NOISE_NZ} built in "
          f"{time.perf_counter() - t0:.2f}s")


def sample_noise(p):
    """
    Trilinear sample from pre-baked 3D noise volume.
    p : (H, W, 3) world-space positions.
    Returns (H, W) in [0, 1].
    """
    x0, x1, y0, y1, z0, z1 = _NOISE_BOUNDS
    px = p[..., 0]; py = p[..., 1]; pz = p[..., 2]

    tx = (px - x0) / (x1 - x0 + 1e-12) * (NOISE_NX - 1)
    ty = (py - y0) / (y1 - y0 + 1e-12) * (NOISE_NY - 1)
    tz = (pz - z0) / (z1 - z0 + 1e-12) * (NOISE_NZ - 1)

    tx = xp.clip(tx, 0.0, NOISE_NX - 1.0001)
    ty = xp.clip(ty, 0.0, NOISE_NY - 1.0001)
    tz = xp.clip(tz, 0.0, NOISE_NZ - 1.0001)

    xi = xp.floor(tx).astype(xp.int32); xi1 = xp.clip(xi+1, 0, NOISE_NX-1)
    yi = xp.floor(ty).astype(xp.int32); yi1 = xp.clip(yi+1, 0, NOISE_NY-1)
    zi = xp.floor(tz).astype(xp.int32); zi1 = xp.clip(zi+1, 0, NOISE_NZ-1)

    fx = (tx - xi).astype(xp.float32); u = fx*fx*(3 - 2*fx)
    fy = (ty - yi).astype(xp.float32); v = fy*fy*(3 - 2*fy)
    fz = (tz - zi).astype(xp.float32); w = fz*fz*(3 - 2*fz)

    vol = _NOISE_VOL
    c000=vol[zi,yi,xi];   c100=vol[zi,yi,xi1]
    c010=vol[zi,yi1,xi];  c110=vol[zi,yi1,xi1]
    c001=vol[zi1,yi,xi];  c101=vol[zi1,yi,xi1]
    c011=vol[zi1,yi1,xi]; c111=vol[zi1,yi1,xi1]

    x00 = c000*(1-u)+c100*u; x10 = c010*(1-u)+c110*u
    x01 = c001*(1-u)+c101*u; x11 = c011*(1-u)+c111*u
    y0v = x00*(1-v)+x10*v;   y1v = x01*(1-v)+x11*v
    return xp.clip(y0v*(1-w)+y1v*w, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Medium density
# ─────────────────────────────────────────────────────────────────────────────

def _box_shape(p, bmin, bmax, soft=EDGE_SOFTNESS):
    """Smooth [0,1] density shape for the box — 0 outside, 1 deep inside."""
    a0 = xp.maximum(float(bmin[0]) - p[..., 0], 0.0)
    a1 = xp.maximum(p[..., 0] - float(bmax[0]), 0.0)
    a2 = xp.maximum(float(bmin[1]) - p[..., 1], 0.0)
    a3 = xp.maximum(p[..., 1] - float(bmax[1]), 0.0)
    a4 = xp.maximum(float(bmin[2]) - p[..., 2], 0.0)
    a5 = xp.maximum(p[..., 2] - float(bmax[2]), 0.0)
    outside = xp.maximum(xp.maximum(xp.maximum(a0, a1), xp.maximum(a2, a3)),
                          xp.maximum(a4, a5))
    return smoothstep(soft, 0.0, outside)


def box_density(p, bmin, bmax):
    """
    Combined density: box-shape envelope × turbulent noise.
    Returns (H, W) density in [0, ∞).
    """
    shape = _box_shape(p, bmin, bmax)
    n     = sample_noise(p)
    noise_mod = (1.0 - NOISE_STRENGTH) + NOISE_STRENGTH * n
    return xp.clip(shape * noise_mod, 0.0, None)


# ─────────────────────────────────────────────────────────────────────────────
# Shadow transmittance (AABB-bounded)
# ─────────────────────────────────────────────────────────────────────────────

def transmittance_to_light(x, bmin, bmax):
    """
    March a shadow ray from sample points x toward the light.
    Stops at the AABB exit so we never waste steps outside the box.
    x    : (H, W, 3)
    Returns (H, W) transmittance in [0, 1].
    """
    v    = LIGHT_POSITION - x
    dist = xp.sqrt(xp.sum(v * v, axis=-1))          # (H, W)
    dirL = v / (dist[..., None] + 1e-12)             # (H, W, 3)

    # Find where shadow ray exits the box (camera position is x, may be inside)
    _, t_box_exit = aabb_intersect(x, dirL, bmin, bmax)
    t_limit = xp.clip(xp.minimum(t_box_exit, dist), 0.0, None)

    T = xp.ones(dist.shape, dtype=xp.float64)
    p = x.copy()

    for si in range(SHADOW_STEPS):
        p   = p + dirL * SHADOW_DS
        t_s = (si + 1) * SHADOW_DS

        active = (t_s < t_limit) & (T > T_THRESH)
        if not host_scalar(xp.any(active)):
            break

        dens = box_density(p, bmin, bmax) * CLOUD_DENSITY
        T   *= xp.where(active, xp.exp(-dens * SHADOW_DS), xp.ones_like(T))

    return T


# ─────────────────────────────────────────────────────────────────────────────
# Main volumetric render
# ─────────────────────────────────────────────────────────────────────────────

def render():
    w, h = IMAGE_WIDTH, IMAGE_HEIGHT
    rd   = camera_rays(w, h)       # (H, W, 3)

    # ── Box bounds ────────────────────────────────────────────────────────
    cx, cy, cz = CUBE_CENTER
    half  = 0.5 * CUBE_SIZE
    bmin  = np.array([cx - half, cy - half, cz - half])
    bmax  = np.array([cx + half, cy + half, cz + half])
    bmin_xp = xp.asarray(bmin)
    bmax_xp = xp.asarray(bmax)

    # ── Analytical AABB intersection (replaces full T_MAX march from camera) ─
    ro_exp = xp.broadcast_to(CAMERA_POSITION, (h, w, 3))
    t_in, t_out = aabb_intersect(ro_exp, rd, bmin, bmax)

    # No hit if t_out ≤ t_in or t_out ≤ 0
    no_hit = (t_out <= t_in) | (t_out <= 0.0)
    t_out  = xp.where(no_hit, t_in, t_out)   # zero-length march for miss

    # ── Accumulation buffers ──────────────────────────────────────────────
    L    = xp.zeros((h, w, 3), dtype=xp.float64)
    Tcam = xp.ones( (h, w),    dtype=xp.float64)

    # Sub-step jitter — eliminates banding artefacts
    rng    = np.random.default_rng(7)
    jitter = xp.asarray(rng.random((h, w)), dtype=xp.float64) * DS_FINE

    # Initialise position at box entry + jitter.
    # Subtract DS_FINE so that after the first advance we land at t_in+jitter
    # (fixes the off-by-one present in all previous cube/cloud files).
    ro_copy = xp.broadcast_to(CAMERA_POSITION, (h, w, 3)).copy()
    p = ro_copy + rd * (t_in + jitter - DS_FINE)[..., None]

    Tlight_cache = xp.ones((h, w), dtype=xp.float64)

    # ── March (only inside the AABB — no wasted steps in empty air) ───────
    for k in range(N_STEPS_MAX):
        p     = p + rd * DS_FINE
        t_cur = t_in + jitter + k * DS_FINE

        active = (t_cur < t_out) & (Tcam > T_THRESH)
        if not host_scalar(xp.any(active)):
            break

        dens = box_density(p, bmin_xp, bmax_xp)   # turbulent density (or flat if NOISE_STRENGTH=0)

        # Radial core-glow — identical to cube.py's emission_base formula:
        #   emission *= (CORE_LOW + (1-CORE_LOW) * core_factor)
        # core_factor = 1 near centre, 0 at CORE_RADIUS distance from centre.
        centre   = xp.asarray([cx, cy, cz], dtype=xp.float64)
        r_centre = xp.sqrt(xp.sum((p - centre) ** 2, axis=-1))
        core_fac = 1.0 - smoothstep(0.0, CORE_RADIUS, r_centre)
        radial   = CORE_LOW + (1.0 - CORE_LOW) * core_fac   # (H, W)

        # Shadow transmittance (AABB-bounded, updated every SHADOW_EVERY steps)
        if (k % SHADOW_EVERY) == 0:
            Tlight_cache = transmittance_to_light(p, bmin_xp, bmax_xp)

        # Point-light irradiance: fall-off with distance — same formula as cube.py
        d_light = LIGHT_POSITION - p
        r2      = xp.sum(d_light * d_light, axis=-1) + LIGHT_R2_EPS
        Lin     = LIGHT_INTENSITY * Tlight_cache / r2   # (H, W)

        # Emission colour from density-based LUT (stays on device).
        # dens_norm≈1 everywhere when NOISE_STRENGTH=0 → flat warm yellow;
        # directional light gradient from Lin dominates, like cube.py.
        # When NOISE_STRENGTH>0 dense patches become yellower, sparse become orange.
        dens_norm  = xp.clip(dens, 0.0, 1.0)
        emit_color = color_from_density(dens_norm)                       # (H, W, 3)
        emit_J     = EMISSION_STRENGTH * dens * radial * Lin             # (H, W)

        # Zero out inactive pixels
        active_f = active.astype(xp.float64)
        emit_J   = emit_J   * active_f
        dens_a   = dens     * active_f

        # Volumetric radiative transfer step
        sigma  = CLOUD_DENSITY * dens_a
        L     += Tcam[..., None] * emit_color * emit_J[..., None] * DS_FINE
        Tcam  *= xp.exp(-sigma * DS_FINE)

    return L.astype(xp.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Post-processing
# ─────────────────────────────────────────────────────────────────────────────

def _gauss_blur_np(img: np.ndarray, sigma: float) -> np.ndarray:
    r   = int(3 * sigma + 0.5)
    k   = np.arange(-r, r + 1, dtype=np.float64)
    ker = np.exp(-0.5 * (k / sigma) ** 2)
    ker /= ker.sum()
    out = np.apply_along_axis(lambda row: np.convolve(row, ker, "same"), 1, img)
    out = np.apply_along_axis(lambda col: np.convolve(col, ker, "same"), 0, out)
    return out


def apply_bloom(img: np.ndarray) -> np.ndarray:
    if BLOOM_SIGMA <= 0 or BLOOM_STRENGTH <= 0:
        return img
    lum     = img[..., 0]*0.2126 + img[..., 1]*0.7152 + img[..., 2]*0.0722
    bright  = np.clip(lum - 0.45, 0.0, None)[..., np.newaxis] * img
    blurred = np.stack(
        [_gauss_blur_np(bright[..., c], BLOOM_SIGMA) for c in range(3)], axis=-1)
    return np.clip(img + BLOOM_STRENGTH * blurred, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    backend_name = "cupy" if (cp is not None and xp is cp) else "numpy"

    print("Building colour LUT …")
    build_color_lut()

    print("Building 3-D noise volume …")
    build_noise_volume()

    print(f"Rendering {IMAGE_WIDTH}×{IMAGE_HEIGHT} with [{backend_name}] …")
    t0  = time.perf_counter()
    raw = render()
    if cp is not None and isinstance(raw, cp.ndarray):
        cp.cuda.Device().synchronize()
    elapsed = time.perf_counter() - t0
    print(f"Render time [{backend_name}]: {elapsed:.1f}s")

    # ── HDR → display ─────────────────────────────────────────────────────
    img = to_numpy(raw).astype(np.float32) * TONEMAP_EXPOSURE
    img = to_numpy(aces_tonemap(xp.asarray(img))).astype(np.float32)
    img = apply_bloom(img)
    img = np.clip(img, 0.0, 1.0) ** (1.0 / 2.2)   # gamma correction

    # ── Light-position dot (same as cube.py) ──────────────────────────────
    lp = project_point_to_pixel(LIGHT_POSITION, IMAGE_WIDTH, IMAGE_HEIGHT)
    if lp is not None:
        lx, ly = lp
        yy, xx = np.mgrid[:IMAGE_HEIGHT, :IMAGE_WIDTH]
        img[(xx - lx) ** 2 + (yy - ly) ** 2 <= 36] = 1.0
        print(f"Light projected to pixel ({lx}, {ly})")
    else:
        print("Light is outside the camera frame")

    # ── Save PNG ──────────────────────────────────────────────────────────
    out_path = "cube_best.png"
    plt.imsave(out_path, img)
    print(f"Saved → {out_path}")

    # ── Display ───────────────────────────────────────────────────────────
    fig, ax = plt.subplots(
        figsize=(IMAGE_WIDTH / 96, IMAGE_HEIGHT / 96), dpi=96,
        facecolor="black",
    )
    ax.imshow(img, interpolation="lanczos")
    ax.axis("off")
    ax.set_title(
        f"Volumetric cube  [{backend_name}  {elapsed:.1f}s]"
        f"   DS={DS_FINE}  σ={CLOUD_DENSITY}"
        f"   noise={NOISE_STRENGTH}  bloom={BLOOM_STRENGTH}",
        color="white", fontsize=8,
    )
    plt.tight_layout(pad=0.2)
    plt.show()


if __name__ == "__main__":
    main()

