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
    """Configuration for 3D Schwarzschild null-geodesic tracing (per-ray planar ODE).

    Each ray is integrated in its own 2D plane that passes through:
      - the BH center
      - the ray origin
      - the initial ray direction

    Within that plane, the null geodesic in Schwarzschild spacetime is approximated by
    the standard planar ODE in terms of u = 1/r, with derivatives with respect to phi:

        u'' + u = 3 M u^2

    where:
      - r is the areal radius from the BH center
      - M is the mass-length (GM/c^2) expressed in your simulation length units

    Fields:
      mass_length:
        M = GM/c^2 in your simulation length units (geometric units).
        Event horizon radius is r_h = 2M.
      dphi:
        Base step in phi used by the RK4 integrator.

        For the batched stepper (advance_batch):
          - ds is a world-space step suggestion from the marcher (typically an SDF distance).
          - advance_batch maps ds to an angular step |dphi| approximately as ds / r, where r is
            the current radius estimate for that ray. This keeps world-space displacement roughly
            proportional to ds.
      phi_scale_max:
        Maximum multiplier applied when mapping marcher-supplied ds into an angular step.
        Used by the batched stepper to cap |dphi| <= dphi * phi_scale_max for stability.
      max_steps:
        Maximum number of integration steps for trace() style integration.
      escape_radius:
        Terminate as escaped if r grows beyond this value.
      radial_eps:
        Rays with tiny angular component (|r * dphi/dlambda| small) are treated as nearly
        radial and use a straight-line fallback in trace().
      hit_eps:
        Surface hit threshold (in SDF distance units) when surface is provided.
      fallback_ds:
        Step length for the near-radial straight-line fallback in trace().
      horizon_plot_eps:
        Small offset used to place the final point just outside the horizon for stable plotting.
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
    """Schwarzschild black hole parameters in simulation units.

    This object holds:
      - xp backend (NumPy or CuPy)
      - BH center in world coordinates
      - SchwarzschildConfig3D configuration
    """

    xp: ArrayModule
    center: Any
    cfg: SchwarzschildConfig3D

    @property
    def horizon_radius(self) -> float:
        """Event horizon radius r_h = 2M in simulation units."""
        return 2.0 * self.cfg.mass_length


class SchwarzschildNullGeodesicTracer3D(RayTracer2D):
    """3D Schwarzschild null geodesic tracer with optional SDF hit testing.

    This is a single-ray tracer that integrates the planar u(phi) ODE per ray, then
    embeds the resulting planar curve back into 3D using a per-ray orthonormal basis.

    Termination conditions:
      - hit: surface SDF distance < cfg.hit_eps (if surface is provided)
      - horizon: r <= 2M
      - far: r >= cfg.escape_radius or u <= 0
      - max_steps: loop limit reached

    Returns:
      RayMarchResult(hit, fell_in, escaped, termination, points)
    """

    def __init__(self, xp: ArrayModule, bh: SchwarzschildBlackHole3D, surface: SDF | None = None) -> None:
        """Initialise the tracer.

        Args:
            xp: Backend module (NumPy or CuPy).
            bh: Schwarzschild black hole descriptor.
            surface: Optional signed distance field for hit testing.
        """
        self.xp = xp
        self.bh = bh
        self.surface = surface

    def _rk4_step(self, u: float, v: float, h: float) -> tuple[float, float]:
        """Advance (u, v) by one RK4 step with step size h in phi.

        Integrates the first-order system corresponding to:

            u'' + u = 3 M u^2

        with:
            u' = v
            v' = -u + 3 M u^2

        Args:
            u: Current u = 1/r.
            v: Current v = du/dphi.
            h: Step in phi (can be negative).

        Returns:
            (u_next, v_next)
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
        """Return True if a surface is provided and p_world is within hit_eps of it."""
        if self.surface is None:
            return False
        dist = float(self.surface.sdf(p_world))
        return dist < self.bh.cfg.hit_eps

    def _make_plane_basis(self, r0: Any, d0: Any) -> tuple[Any, Any, Any]:
        """Build an orthonormal basis (e1, e2, n) for the ray's motion plane.

        The plane is defined by:
          - r0: vector from BH center to ray origin
          - d0: initial ray direction

        Construction:
          - e1 aligns with the initial radial direction (unit r0)
          - n is the plane normal, proportional to cross(r0, d0)
          - e2 completes the right-handed basis: e2 = normalize(cross(n, e1))

        For nearly radial rays where cross(r0, d0) is tiny, a stable fallback normal is
        built using a world-up vector.
        """
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
        """Fallback tracer for nearly radial rays.

        If the ray's angular component is too small (|r * dphi/dlambda| < radial_eps),
        the u(phi) formulation becomes ill-conditioned. In that case, this routine
        advances the ray in a straight line with step size cfg.fallback_ds and checks:
          - SDF surface hit (if provided)
          - horizon crossing (r <= 2M)
          - escape (r >= cfg.escape_radius)

        Returns:
            RayMarchResult with a polyline of visited points in world space.
        """
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
        """Trace a single ray in 3D using a per-ray planar Schwarzschild ODE.

        Steps:
          1. Convert origin and direction to xp arrays and normalize direction.
          2. Build the ray plane basis (e1, e2) around BH center.
          3. Compute initial polar angle phi in that plane.
          4. Convert the initial 3D direction into planar polar components to infer v0.
          5. Integrate (u, v) forward in phi using RK4 and embed points back to 3D.

        Notes:
          - If the initial angular component is too small, uses _trace_nearly_radial().
          - Points are recorded in world coordinates.

        Args:
            origin: Ray origin in world coordinates.
            direction: Initial ray direction in world coordinates.

        Returns:
            RayMarchResult(hit, fell_in, escaped, termination, points)
        """
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
    """Batch-friendly 3D Schwarzschild geodesic stepper for image marching.

    Intended consumer:
      - ImageMarcherSchwarzschild (or equivalent)

    Required interface:
      - state = init_batch(ro, rd0)
      - p_new, rd_new, fell_in_now = advance_batch(state, ds)

    Approach:
      - Each ray gets its own plane basis (e1, e2) through BH center.
      - Dynamics are integrated in that plane using the same u(phi) ODE:
            u'' + u = 3 M u^2
      - The marcher provides ds (usually from an SDF distance estimate in world-space units).
        advance_batch maps ds into an angular step dphi using the current radius estimate r so
        that the world-space displacement stays roughly proportional to ds:
            |dphi| ~= ds / r
        and caps |dphi| <= cfg.dphi * cfg.phi_scale_max for stability.

    Notes:
      - Near-radial rays are detected and advanced by straight-line stepping.
      - The state dict is mutated in-place by advance_batch().
    """

    xp: ArrayModule
    bh: SchwarzschildBlackHole3D

    def init_batch(self, ro: Any, rd0: Any) -> dict[str, Any]:
        """Initialize batched geodesic state for a set of rays.

        Args:
            ro: Ray origins, shape (..., 3).
            rd0: Initial ray directions, shape (..., 3).

        Returns:
            A state dict containing per-ray plane basis, (phi, u, v), and flags used by advance_batch().
        """
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

        sign = xp.where(
            r_dphi_dlambda >= 0.0,
            xp.asarray(1.0, dtype=xp.float64),
            xp.asarray(-1.0, dtype=xp.float64),
        )

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
        """Vectorized RK4 step for the batched (u, v) system.

        Integrates the first-order form of:
            u'' + u = 3 M u^2

        with:
            u' = v
            v' = -u + 3 M u^2

        Args:
            u: Batched u = 1/r, shape (...,).
            v: Batched v = du/dphi, shape (...,).
            h: Batched step in phi, shape (...,). Can be negative per-ray.

        Returns:
            (u_next, v_next) with the same shape as inputs.
        """
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
        """Advance a batch of rays by a marcher-supplied distance ds.

        The marcher provides ds (typically derived from an SDF distance estimate, in world-space units).
        For non-radial rays we map ds into an angular step dphi in a radius-aware way:

            r_est = 1 / max(u, tiny)
            |dphi_target| ~= ds / r_est
            |dphi| = min(|dphi_target|, cfg.dphi * cfg.phi_scale_max)
            h = |dphi| * sign

        This keeps the world-space displacement per update (approximately r * |dphi|) roughly proportional
        to ds while maintaining a conservative cap on the angular step.

        Then:
          - Non-radial rays: integrate u(phi) via RK4, update phi, embed back into 3D.
          - Radial rays: take a straight-line step p += d * ds.

        State mutation:
          Updates "p", "d", "phi", "u", "v" in-place.

        Args:
            state: Mutable per-ray state returned by init_batch().
            ds: Step distances, shape (...,).

        Returns:
            (p_new, d_new, fell_in_now)
              - p_new: New positions, shape (..., 3)
              - d_new: New directions, shape (..., 3)
              - fell_in_now: Boolean mask, shape (...,), True where r <= horizon radius
        """
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

        # Map ds to dphi conservatively (radius-aware so world displacement ~= ds)
        u_safe = xp.maximum(u, xp.asarray(1e-12, dtype=xp.float64))
        r_est = 1.0 / u_safe

        h_target = ds / xp.maximum(r_est, xp.asarray(1e-12, dtype=xp.float64))

        h_cap = xp.asarray(float(cfg.dphi) * float(cfg.phi_scale_max), dtype=xp.float64)
        h_mag = xp.minimum(h_target, h_cap)

        h = h_mag * sign

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
