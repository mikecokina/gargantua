"""
disk_best.py
============
Best-quality volumetric black-hole accretion disk renderer.
Designed to be plugged into the gargantua BH ray-marcher later.

What makes this better than previous attempts
---------------------------------------------
1. Pre-baked colour LUT (2048 entries, stays on compute device during march).
   Previous versions did a full CPU↔GPU numpy round-trip inside the hot loop.
2. Pre-baked 2D polar fBm noise with logarithmic spiral twist.
   Gives realistic turbulent spiral arm structure at zero per-step cost.
3. Correct ray-march initialisation.
   Previous code sampled at t_in + jitter + DS (one step too far).
4. Per-ray closest-approach skip.
   Rays whose XZ closest-approach to the BH axis is > Rout are skipped
   entirely, eliminating hundreds of wasted steps for sky rays.
5. Flared Gaussian disk geometry: H(r) = H0*(r/Rin)^beta.
   Thin inner edge (~ISCO), realistically flared outer disk.
6. Shakura-Sunyaev temperature ramp: T_norm = (Rin/r)^0.75.
   Inner edge: blue-white.  Outer edge: deep orange-red.
7. Relativistic Doppler beaming + colour temperature shift.
   Approaching side brighter and bluer; receding side dimmer and redder.
8. Event horizon: simple dark sphere, blocks rays passing within BH_RADIUS.
9. ACES filmic tonemapping + Gaussian bloom on bright inner edge.
10. Pure black background – no stars, no distractions.

NumPy / CuPy compatible (set BACKEND = "cupy" to use the GPU).
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
IMAGE_WIDTH  = 800
IMAGE_HEIGHT = 450

# ── Camera ─────────────────────────────────────────────────────────────────
# Low angle above the equatorial plane (~10°) for the classic BH disk look.
# Shift CAMERA_POSITION[1] up to see more disk structure; down for edge-on.
CAMERA_POSITION = xp.array([0.0, 5.0, 14.0], dtype=xp.float64)
CAMERA_TARGET   = xp.array([0.0,  0.0,  0.0], dtype=xp.float64)
CAMERA_UP_HINT  = xp.array([0.0,  1.0,  0.0], dtype=xp.float64)
FOV_X_DEG = 50.0

# ── Disk geometry ───────────────────────────────────────────────────────────
DISK_INNER_R       = 1.5      # innermost stable circular orbit (ISCO)
DISK_OUTER_R       = 4.5      # outer disk boundary – smaller fits cleanly in frame
# Original (very thin, flaring):
# DISK_H0    = 0.030
# DISK_FLARE = 1.15
DISK_H0            = 0.05     # scale-height at r = Rin  – visible, not too thick
DISK_FLARE         = 1.0      # H(r) = H0*(r/Rin)^FLARE  – FLARE=1 → uniform thickness
DISK_EDGE_DR       = 0.18     # soft-edge width – slight diffusion at inner/outer boundary
# Original: DISK_DENSITY_POWER = 2.5
DISK_DENSITY_POWER = 1.8      # steeper radial gradient → visible inner/outer brightness difference
# Super-Gaussian power for vertical profile: 2 = standard Gaussian (fuzzy edges),
# 4 = sharper,  6 = very sharp flat-top disk with hard cutoff
DISK_VERTICAL_POWER = 4

# Conservative slab half-height used for slab intersection test.
# We multiply by 5.0 to capture 5σ of the outer-disk Gaussian.
_H_SLAB = DISK_H0 * (DISK_OUTER_R / DISK_INNER_R) ** DISK_FLARE * 5.0

# ── Emission / absorption ───────────────────────────────────────────────────
EMISSION_STRENGTH  = 18.0   # overall brightness scale
# Original (fuzzy/transparent):
# ABSORPTION_COEFF = 0.45
ABSORPTION_COEFF   = 3.0    # extinction – higher → more opaque → sharper defined disk surface
TEMP_EMIT_POWER    = 3.5    # emission ∝ T_norm^N  – inner edge brighter, outer dimmer

# ── Colour ramp  (T_norm=1 → hottest inner edge, T_norm=0 → cold outer rim)
# Each row: [T_norm, R, G, B].  Must be sorted descending by T_norm.
# Interstellar-style: warm white/orange throughout, no extreme blue
COLOR_RAMP = np.array([
    [1.00, 1.00, 0.95, 0.80],  # inner: warm white
    [0.70, 1.00, 0.85, 0.50],  # yellow-white
    [0.40, 1.00, 0.60, 0.15],  # orange
    [0.15, 0.90, 0.30, 0.05],  # deep orange
    [0.00, 0.40, 0.08, 0.00],  # dark red outer rim
], dtype=np.float64)

# ── Doppler beaming ─────────────────────────────────────────────────────────
# Keplerian CCW orbit in XZ plane.  Approaching side → brighter & bluer.
# Original values (asymmetric, realistic):
# V_KEPLER      = 0.52
# BEAM_POWER    = 3.5
# DOPPLER_COLOR = 0.30
V_KEPLER      = 0.0    # orbital speed at Rin (fraction of c).  0 = off → symmetric disk
BEAM_POWER    = 3.5    # I_obs ∝ D^BEAM_POWER  (inactive when V_KEPLER = 0)
DOPPLER_COLOR = 0.0    # how much Doppler shifts the colour temperature (0=brightness only)

# ── 2D polar noise texture (pre-baked once at startup) ──────────────────────
NOISE_NR        = 512    # radial samples in texture
NOISE_NPHI      = 512    # azimuthal samples (wraps at 2π)
NOISE_SCALE_R   = 1.5    # blob size in radial direction (world units)
NOISE_SCALE_PHI = 0.65   # blob size in azimuthal direction
NOISE_OCTAVES   = 6      # fBm octaves
NOISE_GAIN      = 0.50   # fBm gain per octave
NOISE_LACUN     = 2.0    # fBm lacunarity
NOISE_CONTRAST  = 2.2    # >1 sharpens clumps  (higher → more distinct filaments)
# Original: NOISE_STRENGTH = 0.82
NOISE_STRENGTH  = 0.45   # visible spiral/turbulent structure without destroying disk shape

# Logarithmic spiral winding of the noise pattern.
# Higher → tighter spiral arms (like an Sc galaxy).
SPIRAL_WIND = 3.5

# ── Ray marching ─────────────────────────────────────────────────────────────
DS_FINE      = 0.022   # step size inside disk slab (world units)
N_STEPS_MAX  = 1200    # hard cap on steps per ray  (1200*0.022 = 26.4 – covers far disk side)
T_THRESH     = 1e-4    # early-exit when transmittance < this

# ── Event horizon ────────────────────────────────────────────────────────────
BH_RADIUS = 0.82   # rays hitting this sphere see pure black

# ── Post-processing ──────────────────────────────────────────────────────────
TONEMAP_EXPOSURE = 1.6   # linear exposure before ACES tonemapping
BLOOM_SIGMA      = 3.5   # Gaussian bloom radius in pixels  (0 = off)
BLOOM_STRENGTH   = 0.40  # fraction of bloom added back

# ─────────────────────────────────────────────────────────────────────────────
# Maths helpers
# ─────────────────────────────────────────────────────────────────────────────

def normalize(v, axis=-1, eps=1e-12):
    n = xp.sqrt(xp.sum(v * v, axis=axis, keepdims=True))
    return v / (n + eps)


def smoothstep(e0, e1, x):
    t = xp.clip((x - e0) / (e1 - e0 + 1e-12), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def aces_tonemap(x):
    """ACES filmic curve  [0,∞) → [0,1]."""
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return xp.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)


def camera_rays(w, h):
    """Return (H, W, 3) normalised ray directions for a pinhole camera."""
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
    return normalize(rd)   # (H, W, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Colour LUT (pre-baked on CPU, uploaded to compute device once)
# Eliminates the expensive NumPy↔device round-trip inside the march loop.
# ─────────────────────────────────────────────────────────────────────────────

_LUT_N    = 2048
_COLOR_LUT: "xp.ndarray | None" = None   # shape (LUT_N, 3)


def build_color_lut() -> None:
    global _COLOR_LUT
    t_vals = np.linspace(0.0, 1.0, _LUT_N, dtype=np.float64)
    stops   = COLOR_RAMP[:, 0]   # descending T_norm stops
    colors  = COLOR_RAMP[:, 1:]  # RGB at each stop

    lut = np.zeros((_LUT_N, 3), dtype=np.float64)
    n   = len(stops)

    for i, t in enumerate(t_vals):
        if t >= stops[0]:
            lut[i] = colors[0]
            continue
        if t <= stops[-1]:
            lut[i] = colors[-1]
            continue
        for j in range(n - 1):
            t_hi, t_lo = stops[j], stops[j + 1]
            if t_lo <= t <= t_hi:
                alpha  = (t - t_lo) / (t_hi - t_lo + 1e-12)
                lut[i] = colors[j + 1] + alpha * (colors[j] - colors[j + 1])
                break

    _COLOR_LUT = xp.asarray(lut.astype(np.float64))


def color_ramp(t_norm):
    """
    Vectorised LUT look-up.
    t_norm : array [...] in [0, 1]
    Returns : array [..., 3]
    Stays entirely on the compute device – no Python loops, no CPU/GPU transfer.
    """
    idx_f = xp.clip(t_norm, 0.0, 1.0) * (_LUT_N - 1)
    idx0  = xp.clip(idx_f.astype(xp.int32), 0, _LUT_N - 2)
    idx1  = idx0 + 1
    frac  = (idx_f - idx0.astype(xp.float64))[..., None]
    return _COLOR_LUT[idx0] * (1.0 - frac) + _COLOR_LUT[idx1] * frac


# ─────────────────────────────────────────────────────────────────────────────
# 2D polar noise texture (pre-baked on CPU, then uploaded)
# ─────────────────────────────────────────────────────────────────────────────

_NOISE_TEX: "xp.ndarray | None" = None   # shape (NOISE_NR, NOISE_NPHI), float32


def _hash2(ix: "np.ndarray", iy: "np.ndarray") -> "np.ndarray":
    return np.mod(np.sin(ix * 127.1 + iy * 311.7) * 43758.5453123, 1.0)


def _value_noise2(px: "np.ndarray", py: "np.ndarray") -> "np.ndarray":
    x0 = np.floor(px).astype(np.int32)
    y0 = np.floor(py).astype(np.int32)
    x1, y1 = x0 + 1, y0 + 1
    fx, fy  = px - x0, py - y0
    u = fx * fx * (3.0 - 2.0 * fx)
    v = fy * fy * (3.0 - 2.0 * fy)
    c00 = _hash2(x0, y0); c10 = _hash2(x1, y0)
    c01 = _hash2(x0, y1); c11 = _hash2(x1, y1)
    return np.clip(
        (c00 * (1 - u) + c10 * u) * (1 - v)
        + (c01 * (1 - u) + c11 * u) * v,
        0.0, 1.0,
    )


def _fbm2(px, py, octaves, lacun, gain):
    amp, freq, s, norm = 0.5, 1.0, np.zeros_like(px), 0.0
    for _ in range(octaves):
        s    += amp * _value_noise2(px * freq, py * freq)
        norm += amp
        amp  *= gain
        freq *= lacun
    return np.clip(s / (norm + 1e-12), 0.0, 1.0)


def build_noise_texture() -> None:
    global _NOISE_TEX
    t0 = time.perf_counter()

    # Grid of (radial_index, phi_index)
    ir, ip = np.mgrid[0:NOISE_NR, 0:NOISE_NPHI]

    # World-space (r, phi)
    r   = DISK_INNER_R + (DISK_OUTER_R - DISK_INNER_R) * ir / (NOISE_NR - 1)
    phi = 2.0 * np.pi * ip / NOISE_NPHI

    # Logarithmic spiral twist: rotate phi by WIND * ln(r/Rin)
    phi_s = phi + SPIRAL_WIND * np.log(np.maximum(r / DISK_INNER_R, 1e-9))

    # Cartesian noise coordinates (anisotropic scale for elongated blobs)
    px = r * np.cos(phi_s) / NOISE_SCALE_R
    py = r * np.sin(phi_s) / NOISE_SCALE_PHI

    n = _fbm2(px, py, NOISE_OCTAVES, NOISE_LACUN, NOISE_GAIN)
    n = np.power(np.clip(n, 0.0, 1.0), NOISE_CONTRAST)   # sharpen clumps

    _NOISE_TEX = xp.asarray(n.astype(np.float32))
    print(f"  Noise texture {NOISE_NR}×{NOISE_NPHI} built in "
          f"{time.perf_counter() - t0:.2f}s")


def sample_noise(r, phi):
    """
    Fast bilinear look-up in the pre-baked polar texture.
    r, phi : arrays (H, W).
    Returns : (H, W) in [0, 1].
    """
    tr = (r - DISK_INNER_R) / (DISK_OUTER_R - DISK_INNER_R + 1e-12) * (NOISE_NR  - 1)
    tp = (phi % (2.0 * np.pi)) / (2.0 * np.pi) * NOISE_NPHI

    tr = xp.clip(tr, 0.0, NOISE_NR  - 1.0001)
    tp = xp.clip(tp, 0.0, NOISE_NPHI - 0.0001)

    r0 = xp.floor(tr).astype(xp.int32)
    p0 = xp.floor(tp).astype(xp.int32)
    r1 = xp.clip(r0 + 1, 0, NOISE_NR  - 1)
    p1 = (p0 + 1) % NOISE_NPHI          # wraps in phi

    fr = (tr - r0).astype(xp.float32)
    fp = (tp - p0).astype(xp.float32)
    ur = fr * fr * (3.0 - 2.0 * fr)
    up = fp * fp * (3.0 - 2.0 * fp)

    tex = _NOISE_TEX
    c00 = tex[r0, p0]; c10 = tex[r1, p0]
    c01 = tex[r0, p1]; c11 = tex[r1, p1]
    return xp.clip(
        (c00 * (1 - ur) + c10 * ur) * (1 - up)
        + (c01 * (1 - ur) + c11 * ur) * up,
        0.0, 1.0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Disk volumetric properties
# ─────────────────────────────────────────────────────────────────────────────

# Pre-compute Doppler LOS direction (camera XZ projection, constant for all steps)
def _los_xz_unit():
    cam = to_numpy(CAMERA_POSITION)
    v   = np.array([cam[0], cam[2]], dtype=np.float64)
    v  /= np.linalg.norm(v) + 1e-12
    return float(v[0]), float(v[1])   # (los_x, los_z)

_LOS_X, _LOS_Z = _los_xz_unit()


def disk_properties(p):
    """
    Compute density and spectral emission at sample points p (H, W, 3).

    density    : (H, W) float64  – local extinction / emission coefficient
    emit_color : (H, W, 3) float64  – HDR RGB emission (already scaled by
                 temperature gradient, emission strength, and Doppler beaming)
    """
    x = p[..., 0]
    y = p[..., 1]
    z = p[..., 2]

    r   = xp.sqrt(x * x + z * z)
    rc  = xp.maximum(r, DISK_INNER_R * 0.3)    # prevent division by zero
    phi = xp.arctan2(z, x)

    # ── Vertical super-Gaussian profile (flared scale-height) ────────────
    H_r   = DISK_H0 * (rc / DISK_INNER_R) ** DISK_FLARE
    # original:
    # gauss = xp.exp(-0.5 * (y / (H_r + 1e-9)) ** 2)
    # DISK_VERTICAL_POWER=2 → standard Gaussian (fuzzy), 4-6 → sharp flat-top
    gauss = xp.exp(-0.5 * (y / (H_r + 1e-9)) ** DISK_VERTICAL_POWER)

    # ── Radial envelope (power-law density × soft inner/outer masking) ────
    radial = (DISK_INNER_R / rc) ** DISK_DENSITY_POWER
    inner  = smoothstep(DISK_INNER_R,          DISK_INNER_R + DISK_EDGE_DR, r)
    outer  = 1.0 - smoothstep(DISK_OUTER_R - DISK_EDGE_DR, DISK_OUTER_R,   r)
    base   = radial * gauss * inner * outer

    # ── Spiral turbulent noise ────────────────────────────────────────────
    n         = sample_noise(rc, phi)
    noise_mod = (1.0 - NOISE_STRENGTH) + NOISE_STRENGTH * n
    density   = xp.clip(base * noise_mod, 0.0, None)

    # ── Shakura-Sunyaev temperature: T_norm = (Rin/r)^0.75 ───────────────
    T_norm = xp.clip((DISK_INNER_R / rc) ** 0.75, 0.0, 1.0)

    # ── Doppler beaming ───────────────────────────────────────────────────
    # Keplerian CCW orbit in XZ: v_dir = (−sin φ, 0, cos φ)
    # v_los = v_K · (LOS unit in XZ)
    v_K   = V_KEPLER * xp.sqrt(DISK_INNER_R / rc)
    v_los = v_K * (-xp.sin(phi) * _LOS_X + xp.cos(phi) * _LOS_Z)
    D     = xp.clip(1.0 + v_los, 0.02, 8.0)

    # Doppler beaming factor (brightness)
    beaming = D ** BEAM_POWER

    # Doppler colour shift: approaching → bluer (higher T_norm),
    # receding → redder (lower T_norm). Controlled by DOPPLER_COLOR.
    T_shifted = xp.clip(T_norm * D ** DOPPLER_COLOR, 0.0, 1.0)

    # ── Colour from LUT ───────────────────────────────────────────────────
    emit_color = color_ramp(T_shifted)   # (H, W, 3), pure device computation

    # ── Total emission = colour × temperature gradient × beaming ─────────
    strength      = EMISSION_STRENGTH * T_norm ** TEMP_EMIT_POWER
    full_emission = emit_color * (strength * beaming)[..., None]

    return density.astype(xp.float64), full_emission.astype(xp.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Event-horizon sphere
# ─────────────────────────────────────────────────────────────────────────────

def bh_hit_distance(ro_expanded, rd):
    """
    Analytical ray–sphere intersection with a sphere at the origin.
    Returns t of first positive intersection, or 1e30 if no hit.
    ro_expanded : (H, W, 3)
    rd          : (H, W, 3)
    Returns     : (H, W)
    """
    b    = xp.sum(ro_expanded * rd, axis=-1)
    c    = xp.sum(ro_expanded * ro_expanded, axis=-1) - BH_RADIUS ** 2
    disc = b * b - c
    hit  = disc >= 0.0
    sqd  = xp.sqrt(xp.maximum(disc, 0.0))
    t    = xp.where(hit, -b - sqd, xp.full_like(b, 1e30))
    return xp.where(t > 0.0, t, xp.full_like(t, 1e30))


# ─────────────────────────────────────────────────────────────────────────────
# Main volumetric render
# ─────────────────────────────────────────────────────────────────────────────

def render():
    w, h = IMAGE_WIDTH, IMAGE_HEIGHT
    rd   = camera_rays(w, h)   # (H, W, 3)

    # ── Analytical slab intersection (y = ±H_SLAB plane pair) ────────────
    cam_y = host_scalar(CAMERA_POSITION[1])
    rd_y  = rd[..., 1]   # (H, W)

    t_bot = (-_H_SLAB - cam_y) / (rd_y + 1e-15)
    t_top = ( _H_SLAB - cam_y) / (rd_y + 1e-15)
    t_in  = xp.minimum(t_bot, t_top)
    t_out = xp.maximum(t_bot, t_top)

    # Rays nearly parallel to the disk plane
    par     = xp.abs(rd_y) < 1e-5
    in_slab = par & (abs(cam_y) < _H_SLAB)
    t_in    = xp.where(in_slab,  xp.zeros_like(t_in), t_in)
    t_out   = xp.where(in_slab,  xp.full_like(t_out, DS_FINE * N_STEPS_MAX), t_out)
    t_in    = xp.where(par & ~in_slab, xp.full_like(t_in,  1e30), t_in)
    t_out   = xp.where(par & ~in_slab, xp.full_like(t_out, 1e30), t_out)

    t_in  = xp.clip(t_in,  0.0, DS_FINE * N_STEPS_MAX)
    t_out = xp.clip(t_out, 0.0, DS_FINE * N_STEPS_MAX)

    # ── Per-ray closest-approach skip ─────────────────────────────────────
    # Rays whose minimum XZ distance from the BH axis exceeds Rout can
    # never intersect the disk; skip them to save hundreds of wasted steps.
    cam_x = host_scalar(CAMERA_POSITION[0])
    cam_z = host_scalar(CAMERA_POSITION[2])
    rd_x  = rd[..., 0]
    rd_z  = rd[..., 2]

    dot_xz    = cam_x * rd_x + cam_z * rd_z
    rdxz_sq   = rd_x * rd_x + rd_z * rd_z + 1e-20
    t_ca      = xp.maximum(-dot_xz / rdxz_sq, 0.0)   # closest-approach t
    ca_x      = cam_x + t_ca * rd_x
    ca_z      = cam_z + t_ca * rd_z
    r_ca      = xp.sqrt(ca_x * ca_x + ca_z * ca_z)

    miss      = r_ca > DISK_OUTER_R + DISK_EDGE_DR * 2.0
    t_out     = xp.where(miss, t_in, t_out)   # zero-length march for misses

    # ── Clamp march against event horizon ─────────────────────────────────
    ro_exp = xp.broadcast_to(CAMERA_POSITION, (h, w, 3))
    t_bh   = bh_hit_distance(ro_exp, rd)
    t_out  = xp.minimum(t_out, t_bh)          # stop before hitting BH

    # ── Accumulation buffers ──────────────────────────────────────────────
    L    = xp.zeros((h, w, 3), dtype=xp.float64)
    Tcam = xp.ones( (h, w),    dtype=xp.float64)

    # Sub-step jitter (stochastic sampling to eliminate banding artefacts)
    rng    = np.random.default_rng(42)
    jitter = xp.asarray(rng.random((h, w)), dtype=xp.float64) * DS_FINE

    # Initial position: place point AT the slab entry (t_in + jitter).
    # We subtract DS_FINE so that after the first "advance" in the loop
    # we land exactly at t_in + jitter — avoiding the off-by-one present
    # in earlier versions.
    ro_copy = xp.broadcast_to(CAMERA_POSITION, (h, w, 3)).copy()
    p = ro_copy + rd * (t_in + jitter - DS_FINE)[..., None]

    # ── March ─────────────────────────────────────────────────────────────
    for k in range(N_STEPS_MAX):
        # Advance first, then sample (first sample lands at t_in + jitter)
        p     = p + rd * DS_FINE
        t_cur = t_in + jitter + k * DS_FINE

        # Active mask: inside slab AND transmittance above threshold
        active = (t_cur < t_out) & (Tcam > T_THRESH)

        # Early exit when no pixels are still active
        if not host_scalar(xp.any(active)):
            break

        density, emit_color = disk_properties(p)

        # Zero out inactive pixels (no branch divergence on GPU)
        active_f   = active.astype(xp.float64)
        density    = density    * active_f
        emit_color = emit_color * active_f[..., None]

        # Emission–absorption radiative transfer step
        #   dL = T_cam * j(x) * ds
        #   dT = exp(-sigma * ds)       sigma = absorption × density
        sigma  = ABSORPTION_COEFF * density
        L     += Tcam[..., None] * emit_color * density[..., None] * DS_FINE
        Tcam  *= xp.exp(-sigma * DS_FINE)

    return L.astype(xp.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Post-processing
# ─────────────────────────────────────────────────────────────────────────────

def _gauss_blur_np(img: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur (pure NumPy, no scipy required)."""
    r   = int(3 * sigma + 0.5)
    k   = np.arange(-r, r + 1, dtype=np.float64)
    ker = np.exp(-0.5 * (k / sigma) ** 2)
    ker /= ker.sum()
    out = np.apply_along_axis(lambda row: np.convolve(row, ker, "same"), 1, img)
    out = np.apply_along_axis(lambda col: np.convolve(col, ker, "same"), 0, out)
    return out


def apply_bloom(img: np.ndarray) -> np.ndarray:
    """
    img : (H, W, 3) float32 in [0, 1] after tonemapping.
    Adds a soft glow halo around bright regions (inner disk edge).
    """
    if BLOOM_SIGMA <= 0 or BLOOM_STRENGTH <= 0:
        return img
    lum     = img[..., 0] * 0.2126 + img[..., 1] * 0.7152 + img[..., 2] * 0.0722
    bright  = np.clip(lum - 0.45, 0.0, None)[..., np.newaxis] * img
    blurred = np.stack(
        [_gauss_blur_np(bright[..., c], BLOOM_SIGMA) for c in range(3)],
        axis=-1,
    )
    return np.clip(img + BLOOM_STRENGTH * blurred, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    backend_name = "cupy" if (cp is not None and xp is cp) else "numpy"

    print("Building colour LUT …")
    build_color_lut()

    print("Building spiral noise texture …")
    build_noise_texture()

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

    # ── Save to file ──────────────────────────────────────────────────────
    out_path = "disk_best.png"
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
        f"Volumetric accretion disk  [{backend_name}  {elapsed:.1f}s]"
        f"   DS={DS_FINE}  σ_abs={ABSORPTION_COEFF}"
        f"   V_K={V_KEPLER}  bloom={BLOOM_STRENGTH}",
        color="white", fontsize=8,
    )
    plt.tight_layout(pad=0.2)
    plt.show()


if __name__ == "__main__":
    main()

