from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from gargantua.math_utils import normalize, normalize_batch
from gargantua.physics.base import RayTracer2D
from gargantua.raymarch.config import RayMarchResult

if TYPE_CHECKING:
    from gargantua.backend import ArrayModule
    from gargantua.protocols import SDF


@dataclass(frozen=True, slots=True)
class SchwarzschildConfig3D:
    """3D Schwarzschild null geodesic configuration.

    Uses planar ODE in each ray's own plane:
        u'' + u = 3 M u^2, where u = 1/r and derivatives are wrt phi.

    mass_length:
        M = GM/c^2 in your simulation length units (geometric units).
        Event horizon radius r_h = 2M.
    dphi:
        Base step in phi for integrator.
    max_steps:
        Maximum number of phi-steps for tracer-style trace().
    escape_radius:
        Stop if r grows beyond this.
    radial_eps:
        Near-radial rays (tiny angular component) use a straight fallback.
    hit_eps:
        Surface hit threshold if surface is provided.
    fallback_ds:
        Fallback step size for near-radial rays in trace().
    horizon_plot_eps:
        Offset used to place last point just outside horizon for stable plotting.
    """

    mass_length: float
    dphi: float = 2e-3
    phi_scale_max: float = 20.0
    max_steps: int = 60_000
    escape_radius: float = 80.0
    radial_eps: float = 1e-8
    hit_eps: float = 1e-3
    fallback_ds: float = 0.02
    horizon_plot_eps: float = 1e-6


@dataclass(frozen=True, slots=True)
class SchwarzschildBlackHole3D:
    """Schwarzschild BH parameters in simulation units."""

    xp: ArrayModule
    center: Any
    cfg: SchwarzschildConfig3D

    @property
    def horizon_radius(self) -> float:
        return 2.0 * self.cfg.mass_length


class SchwarzschildNullGeodesicTracer3D(RayTracer2D):
    """Real GR: 3D Schwarzschild null geodesic tracer with optional SDF hit testing.

    Same interface as SchwarzschildNullGeodesicTracer2D.trace(), but embedded in 3D:
    each ray moves in the plane spanned by (r0, d0) through BH center.

    Returns RayMarchResult(hit, fell_in, escaped, termination, points).
    """

    def __init__(self, xp: ArrayModule, bh: SchwarzschildBlackHole3D, surface: SDF | None = None) -> None:
        """Initialise SchwarzschildNullGeodesicTracer3D."""
        self.xp = xp
        self.bh = bh
        self.surface = surface

    def _rk4_step(self, u: float, v: float, h: float) -> tuple[float, float]:
        """Stepper.

        RK4 for:
        u' = v
        v' = -u + 3 M u^2
        """
        m = self.bh.cfg.mass_length

        # noinspection PyUnusedLocal
        def f_u(u0: float, v0: float) -> float:  # noqa: ARG001
            return v0

        # noinspection PyUnusedLocal
        def f_v(u0: float, v0: float) -> float:  # noqa: ARG001
            return -u0 + 3.0 * m * u0 * u0

        k1_u = f_u(u, v)
        k1_v = f_v(u, v)

        u2 = u + 0.5 * h * k1_u
        v2 = v + 0.5 * h * k1_v
        k2_u = f_u(u2, v2)
        k2_v = f_v(u2, v2)

        u3 = u + 0.5 * h * k2_u
        v3 = v + 0.5 * h * k2_v
        k3_u = f_u(u3, v3)
        k3_v = f_v(u3, v3)

        u4 = u + h * k3_u
        v4 = v + h * k3_v
        k4_u = f_u(u4, v4)
        k4_v = f_v(u4, v4)

        u_next = u + (h / 6.0) * (k1_u + 2.0 * k2_u + 2.0 * k3_u + k4_u)
        v_next = v + (h / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
        return u_next, v_next

    def _check_surface_hit(self, p_world: Any) -> bool:
        if self.surface is None:
            return False
        dist = float(self.surface.sdf(p_world))
        return dist < self.bh.cfg.hit_eps

    def _make_plane_basis(self, r0: Any, d0: Any) -> tuple[Any, Any, Any]:
        """Return orthonormal (e1, e2, n) spanning ray plane."""
        xp = self.xp
        rhat = normalize(xp, r0)

        n = xp.cross(r0, d0)
        nn = float(xp.linalg.norm(n))

        if nn < 1e-10:
            up = xp.asarray([0.0, 1.0, 0.0], dtype=xp.float64)
            if abs(float(xp.dot(up, rhat))) > 0.95:
                up = xp.asarray([1.0, 0.0, 0.0], dtype=xp.float64)
            n = normalize(xp, xp.cross(rhat, up))
        else:
            n = n / xp.asarray(nn, dtype=xp.float64)

        e1 = rhat
        e2 = normalize(xp, xp.cross(n, e1))
        return e1, e2, n

    def _trace_nearly_radial(self, origin: Any, direction: Any) -> RayMarchResult:
        xp = self.xp
        cfg = self.bh.cfg
        p = xp.asarray(origin, dtype=xp.float64)
        d = normalize(xp, xp.asarray(direction, dtype=xp.float64))

        pts: list[Any] = [p.copy()]
        horizon = self.bh.horizon_radius

        for _ in range(min(cfg.max_steps, 20_000)):
            p = p + d * float(cfg.fallback_ds)
            pts.append(p.copy())

            if self._check_surface_hit(p):
                return RayMarchResult(True, False, False, "hit", xp.stack(pts))

            r = float(xp.linalg.norm(p - self.bh.center))
            if r <= horizon:
                return RayMarchResult(False, True, False, "horizon", xp.stack(pts))
            if r >= cfg.escape_radius:
                return RayMarchResult(False, False, True, "far", xp.stack(pts))

        return RayMarchResult(False, False, True, "max_steps", xp.stack(pts))

    def trace(self, origin: Any, direction: Any) -> RayMarchResult:
        xp = self.xp
        cfg = self.bh.cfg

        origin = xp.asarray(origin, dtype=xp.float64)
        direction = normalize(xp, xp.asarray(direction, dtype=xp.float64))

        r0_vec = origin - xp.asarray(self.bh.center, dtype=xp.float64)
        r0 = float(xp.linalg.norm(r0_vec))
        if r0 <= 1e-12:
            return RayMarchResult(False, True, False, "horizon", xp.zeros((0, 3), dtype=xp.float64))

        e1, e2, _n = self._make_plane_basis(r0_vec, direction)

        x0 = float(xp.dot(r0_vec, e1))
        y0 = float(xp.dot(r0_vec, e2))
        phi = float(xp.arctan2(y0, x0))

        # 2D polar basis in the ray plane
        c = float(xp.cos(phi))
        s = float(xp.sin(phi))
        rhat_2 = xp.asarray([c, s], dtype=xp.float64)
        phihat_2 = xp.asarray([-s, c], dtype=xp.float64)

        d_plane = xp.asarray([float(xp.dot(direction, e1)), float(xp.dot(direction, e2))], dtype=xp.float64)
        dr_dlambda = float(xp.dot(d_plane, rhat_2))
        r_dphi_dlambda = float(xp.dot(d_plane, phihat_2))

        if abs(r_dphi_dlambda) < cfg.radial_eps:
            return self._trace_nearly_radial(origin, direction)

        sign = 1.0 if r_dphi_dlambda > 0.0 else -1.0
        h = float(cfg.dphi) * sign

        dr_dphi = r0 * dr_dlambda / r_dphi_dlambda
        u = 1.0 / r0
        v = -(1.0 / (r0 * r0)) * dr_dphi

        horizon = self.bh.horizon_radius
        pts: list[Any] = [origin.copy()]

        for _ in range(cfg.max_steps):
            u, v = self._rk4_step(u, v, h)
            phi += h

            if u <= 0.0:
                return RayMarchResult(False, False, True, "far", xp.stack(pts))

            r = 1.0 / u

            cc = float(xp.cos(phi))
            ss = float(xp.sin(phi))
            p_world = xp.asarray(self.bh.center, dtype=xp.float64) + r * (cc * e1 + ss * e2)
            pts.append(p_world)

            if self._check_surface_hit(p_world):
                return RayMarchResult(True, False, False, "hit", xp.stack(pts))

            if r <= horizon:
                r_plot = horizon + cfg.horizon_plot_eps
                p_world_h = xp.asarray(self.bh.center, dtype=xp.float64) + r_plot * (cc * e1 + ss * e2)
                pts[-1] = p_world_h
                return RayMarchResult(False, True, False, "horizon", xp.stack(pts))

            if r >= cfg.escape_radius:
                return RayMarchResult(False, False, True, "far", xp.stack(pts))

        return RayMarchResult(False, False, True, "max_steps", xp.stack(pts))


@dataclass(frozen=True, slots=True)
class SchwarzschildGeodesicStepper3D:
    """Batch-friendly stepper for ImageMarcherSchwarzschild.

    Required by marcher:
        state = init_batch(ro, rd0)
        p_new, rd_new, fell_in_now = advance_batch(state, ds)

    This uses the same u(phi) ODE, but batched and embedded in 3D per ray plane.

    ds meaning:
      - marcher supplies ds from SDF distance (clipped)
      - we map ds -> dphi by scaling around cfg.dphi conservatively
    """

    xp: ArrayModule
    bh: SchwarzschildBlackHole3D

    def init_batch(self, ro: Any, rd0: Any) -> dict[str, Any]:
        xp = self.xp
        center = xp.asarray(self.bh.center, dtype=xp.float64)

        p0 = xp.asarray(ro, dtype=xp.float64)
        d0 = normalize_batch(xp, xp.asarray(rd0, dtype=xp.float64))

        r0_vec = p0 - center
        r0 = xp.linalg.norm(r0_vec, axis=-1)
        r0_safe = xp.maximum(r0, xp.asarray(1e-12, dtype=xp.float64))
        rhat = r0_vec / r0_safe[..., None]

        n = xp.cross(r0_vec, d0)
        nn = xp.linalg.norm(n, axis=-1, keepdims=True)

        up = xp.asarray([0.0, 1.0, 0.0], dtype=xp.float64)
        up_b = up[None, None, :] if (hasattr(p0, "ndim") and int(p0.ndim) == 3) else up  # safe-ish
        # compute fallback normal: cross(rhat, up)
        n_fb = xp.cross(rhat, up_b)
        n_fb = normalize_batch(xp, n_fb)

        use_fb = nn < xp.asarray(1e-10, dtype=xp.float64)
        n = xp.where(use_fb, n_fb, n / xp.maximum(nn, xp.asarray(1e-12, dtype=xp.float64)))

        e1 = rhat
        e2 = normalize_batch(xp, xp.cross(n, e1))

        x0 = xp.sum(r0_vec * e1, axis=-1)
        y0 = xp.sum(r0_vec * e2, axis=-1)
        phi0 = xp.arctan2(y0, x0)

        u0 = 1.0 / r0_safe

        c = xp.cos(phi0)
        s = xp.sin(phi0)
        rhat2 = xp.stack([c, s], axis=-1)
        phihat2 = xp.stack([-s, c], axis=-1)

        d_plane = xp.stack([xp.sum(d0 * e1, axis=-1), xp.sum(d0 * e2, axis=-1)], axis=-1)
        dr_dlambda = xp.sum(d_plane * rhat2, axis=-1)
        r_dphi_dlambda = xp.sum(d_plane * phihat2, axis=-1)

        radial = xp.abs(r_dphi_dlambda) < xp.asarray(self.bh.cfg.radial_eps, dtype=xp.float64)

        # dr/dphi = r * dr/dλ / (r dphi/dλ)
        denom = xp.where(radial, xp.asarray(1.0, dtype=xp.float64), r_dphi_dlambda)
        dr_dphi = r0_safe * dr_dlambda / denom
        v0 = -(1.0 / (r0_safe * r0_safe)) * dr_dphi

        sign = xp.where(r_dphi_dlambda >= 0.0, xp.asarray(1.0, dtype=xp.float64), xp.asarray(-1.0, dtype=xp.float64))

        return {
            "center": center,
            "p": p0,
            "d": d0,
            "e1": e1,
            "e2": e2,
            "phi": phi0,
            "u": u0,
            "v": v0,
            "sign": sign,
            "radial": radial,
        }

    def _rk4_step_batch(self, u: Any, v: Any, h: Any) -> tuple[Any, Any]:
        xp = self.xp
        # noinspection PyUnusedLocal
        m = xp.asarray(self.bh.cfg.mass_length, dtype=xp.float64)  # it is actually used in `f_v` method bellow

        # u0 is on 1st position but not used
        def f_u(_: Any, v0: Any) -> Any:
            return v0

        # v0 is on 2nd position but not used
        def f_v(u0: Any, _: Any) -> Any:
            return -u0 + 3.0 * m * u0 * u0

        k1_u = f_u(u, v)
        k1_v = f_v(u, v)

        u2 = u + 0.5 * h * k1_u
        v2 = v + 0.5 * h * k1_v
        k2_u = f_u(u2, v2)
        k2_v = f_v(u2, v2)

        u3 = u + 0.5 * h * k2_u
        v3 = v + 0.5 * h * k2_v
        k3_u = f_u(u3, v3)
        k3_v = f_v(u3, v3)

        u4 = u + h * k3_u
        v4 = v + h * k3_v
        k4_u = f_u(u4, v4)
        k4_v = f_v(u4, v4)

        u_next = u + (h / 6.0) * (k1_u + 2.0 * k2_u + 2.0 * k3_u + k4_u)
        v_next = v + (h / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
        return u_next, v_next

    def advance_batch(self, state: dict[str, Any], ds: Any) -> tuple[Any, Any, Any]:
        xp = self.xp
        cfg = self.bh.cfg

        center = state["center"]
        p_old = state["p"]
        d_old = state["d"]

        e1 = state["e1"]
        e2 = state["e2"]
        phi = state["phi"]
        u = state["u"]
        v = state["v"]
        sign = state["sign"]
        radial = state["radial"]

        ds = xp.asarray(ds, dtype=xp.float64)

        # Map ds to dphi conservatively
        base = xp.asarray(cfg.dphi, dtype=xp.float64)
        scale = xp.clip(ds / base, 0.0, float(cfg.phi_scale_max))  # was 0.25 lower bound
        h = base * scale * sign

        # Geodesic update (non-radial)
        u_new, v_new = self._rk4_step_batch(u, v, h)
        phi_new = phi + h

        u_new = xp.maximum(u_new, xp.asarray(1e-12, dtype=xp.float64))
        r = 1.0 / u_new

        cc = xp.cos(phi_new)
        ss = xp.sin(phi_new)
        p_geo = center + r[..., None] * (cc[..., None] * e1 + ss[..., None] * e2)

        # Tangent direction from dp/dphi = r' e_r + r e_phi
        dr_dphi = -(v_new / (u_new * u_new))
        er = cc[..., None] * e1 + ss[..., None] * e2
        ephi = (-ss)[..., None] * e1 + cc[..., None] * e2
        dp_dphi = dr_dphi[..., None] * er + r[..., None] * ephi
        d_geo = normalize_batch(xp, dp_dphi)

        # Radial fallback: straight line step
        p_lin = p_old + d_old * ds[..., None]
        d_lin = d_old

        m3 = radial[..., None]
        p_new = xp.where(m3, p_lin, p_geo)
        d_new = xp.where(m3, d_lin, d_geo)

        # Update state
        state["p"] = p_new
        state["d"] = d_new
        state["phi"] = xp.where(radial, phi, phi_new)
        state["u"] = xp.where(radial, u, u_new)
        state["v"] = xp.where(radial, v, v_new)

        # Fell into horizon?
        r_now = xp.linalg.norm(p_new - center, axis=-1)
        fell_in = r_now <= xp.asarray(self.bh.horizon_radius, dtype=xp.float64)

        return p_new, d_new, fell_in
