from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from gargantua.math_utils import normalize
from gargantua.physics.base import RayTracer2D
from gargantua.raymarch.config import RayMarchResult

if TYPE_CHECKING:
    from gargantua.backend import ArrayModule
    from gargantua.protocols import SDF


@dataclass(frozen=True, slots=True)
class SchwarzschildConfig2D:
    """Configuration for 2D Schwarzschild null geodesic tracing.

    mass_length:
        M = GM/c^2 expressed in your simulation units (length).
        Event horizon radius is r_h = 2M.
    dphi:
        Step in polar angle for the integrator.
    max_steps:
        Maximum integration steps.
    escape_radius:
        Stop tracing if r exceeds this.
    radial_eps:
        If the ray is nearly radial (too little angular momentum), switch to fallback.
    hit_eps:
        SDF hit threshold when surface is provided.
    """

    mass_length: float
    dphi: float = 2e-3
    max_steps: int = 60_000
    escape_radius: float = 80.0
    radial_eps: float = 1e-8
    hit_eps: float = 1e-3


@dataclass(frozen=True, slots=True)
class SchwarzschildBlackHole2D:
    """Schwarzschild BH parameters in simulation units."""

    xp: ArrayModule
    center: Any
    cfg: SchwarzschildConfig2D

    @property
    def horizon_radius(self) -> float:
        return 2.0 * self.cfg.mass_length


class SchwarzschildNullGeodesicTracer2D(RayTracer2D):
    """Real GR: 2D Schwarzschild null geodesic tracer with optional SDF hit testing.

    This integrates the standard planar null-geodesic equation:

        u'' + u = 3 M u^2,   where u = 1/r

    Notes:
    - This produces the *path* in Schwarzschild coordinates (r, phi) and maps
      it back to Cartesian points for plotting.
    - Optional SDF hit testing checks `surface.sdf(p_world)` at every step.
    - No "sphere marching" here - we integrate geodesics, then test geometry.

    Termination:
    - "hit": surface hit (if surface is provided)
    - "horizon": captured by BH (r <= 2M)
    - "far": escaped (r >= escape_radius)
    - "max_steps": integrator step limit

    """

    HORIZON_PLOT_EPS: float = 1e-6

    def __init__(
            self,
            xp: ArrayModule,
            bh: SchwarzschildBlackHole2D,
            surface: SDF | None = None,
    ) -> None:
        self.xp = xp
        self.bh = bh
        self.surface = surface

    def _rk4_step(self, u: float, v: float, h: float) -> tuple[float, float]:
        """RK4 for:
        u' = v
        v' = -u + 3 M u^2
        """
        m = self.bh.cfg.mass_length

        # noinspection PyUnusedLocal
        def f_u(u0: float, v0: float) -> float:
            return v0

        # noinspection PyUnusedLocal
        def f_v(u0: float, v0: float) -> float:
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

    def _trace_nearly_radial(self, origin: Any, direction: Any) -> RayMarchResult:
        """Fallback for near-radial rays.

        u(phi) is ill-conditioned when angular momentum is near zero, so we step
        approximately along the initial direction and apply only horizon/escape/hit tests.

        This keeps the tracer stable without pretending to resolve near-radial lensing.
        """
        xp = self.xp
        cfg = self.bh.cfg

        p = xp.asarray(origin, dtype=xp.float64)
        d = normalize(xp, xp.asarray(direction, dtype=xp.float64))

        pts: list[Any] = [p.copy()]
        ds = 0.02
        horizon = self.bh.horizon_radius

        for _ in range(min(cfg.max_steps, 20_000)):
            p = p + d * ds
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

        # Work in BH-centered coordinates for polar math.
        p0 = origin - xp.asarray(self.bh.center, dtype=xp.float64)
        r0 = float(xp.linalg.norm(p0))

        if r0 <= 1e-12:
            return RayMarchResult(False, True, False, "horizon", xp.zeros((0, 2), dtype=xp.float64))

        phi0 = float(xp.arctan2(p0[1], p0[0]))

        # Local polar basis at phi:
        # rhat = (cos phi, sin phi)
        # phihat = (-sin phi, cos phi)
        c = float(xp.cos(phi0))
        s = float(xp.sin(phi0))
        rhat = xp.asarray([c, s], dtype=xp.float64)
        phihat = xp.asarray([-s, c], dtype=xp.float64)

        # Decompose initial direction in local polar basis.
        dr_dlambda = float(xp.dot(direction, rhat))
        r_dphi_dlambda = float(xp.dot(direction, phihat))  # equals r * dphi/dlambda

        # Near-radial rays have tiny angular momentum -> fallback.
        if abs(r_dphi_dlambda) < cfg.radial_eps:
            return self._trace_nearly_radial(origin, direction)

        # Integrate phi in the same orientation as the ray's angular motion.
        sign = 1.0 if r_dphi_dlambda > 0.0 else -1.0
        h = float(cfg.dphi) * sign

        # Convert initial condition to u and du/dphi.
        # dr/dphi = (dr/dlambda) / (dphi/dlambda) and dphi/dlambda = (r_dphi_dlambda / r)
        dr_dphi = r0 * dr_dlambda / r_dphi_dlambda

        u0 = 1.0 / r0
        v0 = -(1.0 / (r0 * r0)) * dr_dphi  # du/dphi

        horizon = self.bh.horizon_radius
        pts: list[Any] = [origin.copy()]

        u, v = u0, v0

        for _ in range(cfg.max_steps):
            u, v = self._rk4_step(u, v, h)
            phi0 += h

            if u <= 0.0:
                return RayMarchResult(False, False, True, "far", xp.stack(pts))

            r = 1.0 / u

            # Convert back to world coordinates.
            x = r * float(xp.cos(phi0))
            y = r * float(xp.sin(phi0))
            p_world = xp.asarray([x, y], dtype=xp.float64) + self.bh.center

            pts.append(p_world)

            if self._check_surface_hit(p_world):
                return RayMarchResult(True, False, False, "hit", xp.stack(pts))

            if r <= horizon:
                # Replace last point with something just outside the horizon for stable plotting.
                r_plot = horizon + self.HORIZON_PLOT_EPS
                xh = r_plot * float(xp.cos(phi0))
                yh = r_plot * float(xp.sin(phi0))
                pts[-1] = xp.asarray([xh, yh], dtype=xp.float64) + self.bh.center
                return RayMarchResult(False, True, False, "horizon", xp.stack(pts))

            if r >= cfg.escape_radius:
                return RayMarchResult(False, False, True, "far", xp.stack(pts))

        return RayMarchResult(False, False, True, "max_steps", xp.stack(pts))
