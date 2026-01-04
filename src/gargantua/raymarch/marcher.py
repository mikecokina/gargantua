from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from gargantua.math_utils import clamp_float, normalize
from gargantua.physics.base import RayTracer2D
from gargantua.raymarch.config import RayMarchConfig, RayMarchResult

if TYPE_CHECKING:
    from gargantua.backend import ArrayModule
    from gargantua.physics.newtonian import BlackHoleBender
    from gargantua.scene import Scene


class RayMarcher(RayTracer2D):
    """Dimension-agnostic ray marcher."""

    HORIZON_EPS: float = 1e-6

    def __init__(self, xp: ArrayModule, config: RayMarchConfig, scene: Scene) -> None:
        """Initialise the marcher."""
        self.xp = xp
        self.cfg = config
        self.scene = scene

    def _step_scale_near_bh(self, p: Any, bh: BlackHoleBender) -> float:
        dist = float(self.xp.linalg.norm(p - bh.center))
        if dist >= self.cfg.slow_radius:
            return 1.0
        t = max(dist / self.cfg.slow_radius, 0.0)
        return self.cfg.slow_factor + (1.0 - self.cfg.slow_factor) * t

    def _ray_sphere_intersection_distance(
            self,
            origin: Any,
            direction: Any,
            center: Any,
            radius: float,
    ) -> float | None:
        oc = origin - center
        a = float(self.xp.dot(direction, direction))
        b = 2.0 * float(self.xp.dot(oc, direction))
        c = float(self.xp.dot(oc, oc) - radius * radius)

        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            return None

        s = float(self.xp.sqrt(disc))
        t0 = (-b - s) / (2.0 * a)
        t1 = (-b + s) / (2.0 * a)

        candidates = [t for t in (t0, t1) if t >= 0.0]
        if not candidates:
            return None
        return min(candidates)

    def trace(self, origin: Any, direction: Any) -> RayMarchResult:
        """Trace using the scene provided at construction time."""
        scene = self.scene
        xp = self.xp

        p = xp.asarray(origin, dtype=xp.float64).copy()
        d = normalize(xp, xp.asarray(direction, dtype=xp.float64))

        points: list[Any] = [p.copy()]
        traveled = 0.0

        for _ in range(self.cfg.max_steps):
            dist = float(scene.surface.sdf(p))
            if dist < self.cfg.eps:
                return RayMarchResult(
                    hit_object=True,
                    fell_in_bh=False,
                    escaped_scene=False,
                    termination="hit",
                    points=xp.stack(points),
                )

            ds = clamp_float(dist, self.cfg.min_step, self.cfg.max_step)
            ds *= self._step_scale_near_bh(p, scene.bh)

            t_h = self._ray_sphere_intersection_distance(p, d, scene.bh.center, scene.bh.horizon)
            if t_h is not None and t_h <= ds:
                p = p + d * max(t_h - self.HORIZON_EPS, 0.0)
                points.append(p.copy())
                return RayMarchResult(
                    hit_object=False,
                    fell_in_bh=True,
                    escaped_scene=False,
                    termination="horizon",
                    points=xp.stack(points),
                )

            p = p + d * ds
            traveled += ds
            points.append(p.copy())

            if traveled > scene.bounds.far_distance:
                return RayMarchResult(
                    hit_object=False,
                    fell_in_bh=False,
                    escaped_scene=True,
                    termination="far",
                    points=xp.stack(points),
                )

            d = scene.bh.bend(d, p, ds)

        return RayMarchResult(
            hit_object=False,
            fell_in_bh=False,
            escaped_scene=True,
            termination="max_steps",
            points=xp.stack(points),
        )


@dataclass(frozen=True, slots=True)
class ImageMarchConfig:
    max_steps: int = 220
    eps: float = 1e-3
    min_step: float = 1e-3
    max_step: float = 0.08

    # Hard cap for a single GR step (substepping happens if ds > max_ds_step)
    # Set to something smaller than max_step to avoid tunneling.
    max_ds_step: float = 0.02

    # Adaptive cap near surfaces (fraction of dist_pre used as an extra ds cap)
    # 0.25 is a good start; 0.1 is more robust but slower.
    ds_surface_factor: float = 0.25


@dataclass(frozen=True, slots=True)
class ImageMarchResult:
    hit: Any
    fell_in: Any
    traveled: Any


class ImageMarcherNewtonian:
    def __init__(self, xp: ArrayModule, config: ImageMarchConfig, scene: Any) -> None:
        """Initialise the marcher."""
        self.xp = xp
        self.config = config
        self.scene = scene

    def march(self, ro: Any, rd0: Any) -> ImageMarchResult:
        """March.

        ro: (3,)
        rd0: (H,W,3)
        """
        xp = self.xp
        cfg = self.config

        height, width, _ = rd0.shape

        p = xp.broadcast_to(ro[None, None, :], (height, width, 3)).astype(xp.float64).copy()
        rd = rd0.astype(xp.float64).copy()

        hit = xp.zeros((height, width), dtype=bool)
        fell_in = xp.zeros((height, width), dtype=bool)
        traveled = xp.zeros((height, width), dtype=xp.float64)

        # NEW: cap for a single advance (substepping happens if ds_total > max_ds_step)
        max_ds_step = float(getattr(cfg, "max_ds_step", cfg.max_step))
        max_ds_step = max(max_ds_step, 1e-12)

        for _ in range(int(cfg.max_steps)):
            active = (~hit) & (~fell_in) & (traveled < float(self.scene.bounds.far_distance))
            if not bool(xp.any(active)):
                break

            # ------------------------------------------------------------
            # Immediate termination tests at current position
            # ------------------------------------------------------------

            # Object hit (SDF)
            d_obj0 = self.scene.surface.sdf(p)  # (H,W)
            hit = hit | (active & (d_obj0 < float(cfg.eps)))

            # BH capture (SDF of horizon sphere)
            # horizon_sdf(p) = |p-center| - horizon_radius
            dist_bh0 = xp.linalg.norm(p - self.scene.bh.center[None, None, :], axis=-1)
            h_sdf0 = dist_bh0 - float(self.scene.bh.horizon)
            fell_in = fell_in | (active & (h_sdf0 <= 0.0))

            active = (~hit) & (~fell_in) & (traveled < float(self.scene.bounds.far_distance))
            if not bool(xp.any(active)):
                break

            # ------------------------------------------------------------
            # Step size selection (SDF-driven) + substepping for robustness
            # ------------------------------------------------------------

            # Base desired step from object SDF (like before).
            # For Newtonian it is OK to keep min_step, but if you want maximum robustness
            # near surfaces, you can set min_step=0 in ImageMarchConfig for this marcher too.
            ds_total = xp.clip(d_obj0, float(cfg.min_step), float(cfg.max_step))
            ds_total = ds_total * active.astype(xp.float64)

            if float(xp.max(ds_total)) <= 0.0:
                break

            # Batch-wide number of sub-steps.
            n_sub = int(xp.ceil(xp.max(ds_total) / xp.asarray(max_ds_step, dtype=xp.float64)))
            n_sub = max(n_sub, 1)

            remaining = ds_total

            for _sub in range(n_sub):
                active_sub = (
                    (~hit)
                    & (~fell_in)
                    & (traveled < float(self.scene.bounds.far_distance))
                    & (remaining > 0.0)
                )
                if not bool(xp.any(active_sub)):
                    break

                # Pre-substep SDFs
                d_obj_pre = self.scene.surface.sdf(p)

                dist_bh_pre = xp.linalg.norm(p - self.scene.bh.center[None, None, :], axis=-1)
                h_sdf_pre = dist_bh_pre - float(self.scene.bh.horizon)

                # Sub-step size is capped by max_ds_step, by remaining,
                # AND adaptively by distance to the object to reduce grazing overshoot.
                k = float(getattr(cfg, "ds_surface_factor", 0.25))  # try 0.25, then 0.1 if needed

                ds_cap_surface = k * xp.maximum(
                    d_obj_pre,
                    xp.asarray(float(cfg.eps), dtype=xp.float64),
                )

                ds_cap = xp.minimum(
                    xp.asarray(max_ds_step, dtype=xp.float64),
                    ds_cap_surface,
                )

                ds = xp.minimum(remaining, ds_cap)
                ds = ds * active_sub.astype(xp.float64)

                # Advance (straight-line step in Newtonian)
                p_new = p + rd * ds[..., None]

                # Post-substep SDFs
                d_obj_post = self.scene.surface.sdf(p_new)

                dist_bh_post = xp.linalg.norm(p_new - self.scene.bh.center[None, None, :], axis=-1)
                h_sdf_post = dist_bh_post - float(self.scene.bh.horizon)

                # --------------------------------------------------------
                # Termination logic (SDF-only)
                # --------------------------------------------------------

                # Object hit tests
                hit_post = d_obj_post < float(cfg.eps)
                # Optional crossing test for signed SDFs:
                crossed_obj = (d_obj_pre > 0.0) & (d_obj_post <= 0.0)
                hit = hit | (active_sub & (hit_post | crossed_obj))

                # Horizon capture tests (sphere-horizon SDF, but still SDF-only)
                fell_post = h_sdf_post <= 0.0
                crossed_h = (h_sdf_pre > 0.0) & (h_sdf_post <= 0.0)
                fell_in = fell_in | (active_sub & (fell_post | crossed_h))

                # Apply the move only for rays still active after terminations
                still = (~hit) & (~fell_in) & (traveled < float(self.scene.bounds.far_distance))
                m3 = still[..., None]
                p = xp.where(m3, p_new, p)

                traveled = traveled + ds
                remaining = remaining - ds

                # Newtonian bending (direction update)
                # Only bend for rays that are still active (avoid NaNs after capture)
                rd = xp.where(m3, self.scene.bh.bend_batch(rd, p, ds), rd)

        return ImageMarchResult(hit=hit, fell_in=fell_in, traveled=traveled)


class ImageMarcherSchwarzschild:
    """Batch image marcher for Schwarzschild-style curved rays.

    This marcher does not assume straight segments. It delegates position and direction
    updates to a geodesic stepper stored in scene.bh.

    Required scene interface:
      - scene.surface.sdf(p) -> (H,W) float distances
      - scene.bounds.far_distance (float)
      - scene.bh.init_batch(ro, rd0) -> state (any, typically dict of arrays)
      - scene.bh.advance_batch(state, ds) -> (p_new, rd_new, fell_in_now)

    ds is interpreted by the stepper (it may map ds to dphi, dλ, etc).

    Why max_ds_step / substepping:
      - The original logic only checked the SDF at p (dist0) and after one big GR step (dist1).
      - With curved rays, it's easy to "enter and exit" the object in a single step:
          dist0 > 0, you go through the sphere, and land outside again with dist1 > 0.
        That misses both:
          hit_post (dist1 < eps)  and  crossed (dist0 > 0 and dist1 <= 0)
      - Result: a few pixels incorrectly reach the background (white spot / holes).
      - Substepping caps the per-update ds so we sample the SDF often enough to detect hits.
    """

    def __init__(self, xp: ArrayModule, config: ImageMarchConfig, scene: Any) -> None:
        """Initialise the marcher."""
        self.xp = xp
        self.config = config
        self.scene = scene

    def march(self, ro: Any, rd0: Any) -> ImageMarchResult:
        xp = self.xp
        cfg = self.config

        rd = normalize(xp, rd0.astype(xp.float64))

        # Broadcast camera origin to per-pixel positions.
        p = xp.zeros_like(rd)
        p[..., 0] = ro[0]
        p[..., 1] = ro[1]
        p[..., 2] = ro[2]

        traveled = xp.zeros(rd.shape[:-1], dtype=xp.float64)

        hit = xp.zeros(rd.shape[:-1], dtype=bool)
        fell_in = xp.zeros(rd.shape[:-1], dtype=bool)

        state = self.scene.bh.init_batch(ro=p, rd0=rd)

        # NEW: cap for a single geodesic advance.
        # If ds_total is larger than this, we split it into multiple sub-steps.
        # Start with something like 0.005 - 0.02 depending on quality vs speed.
        max_ds_step = float(getattr(cfg, "max_ds_step", cfg.max_step))
        max_ds_step = max(max_ds_step, 1e-12)

        for _ in range(cfg.max_steps):
            active = (~hit) & (~fell_in) & (traveled < float(self.scene.bounds.far_distance))
            if not bool(xp.any(active)):
                break

            # Distance at the *start* of this outer iteration.
            dist0 = self.scene.surface.sdf(p)

            hit_now = dist0 < float(cfg.eps)
            hit = hit | (active & hit_now)

            active = (~hit) & (~fell_in) & (traveled < float(self.scene.bounds.far_distance))
            if not bool(xp.any(active)):
                break

            # IMPORTANT for GR: don't force a positive min_step near the surface,
            # it causes tunneling (skipping over the hit band).
            #
            # This is our "desired" step for the iteration (before substepping).
            ds_total = xp.clip(dist0, 0.0, float(cfg.max_step))
            ds_total = ds_total * active.astype(xp.float64)

            # If nothing left to do, we can stop.
            if float(xp.max(ds_total)) <= 0.0:
                break

            # NEW: choose how many sub-steps to take for the worst-case ray in the batch.
            # We do a batch-wide n_sub for simplicity (fast + predictable).
            n_sub = int(xp.ceil(xp.max(ds_total) / xp.asarray(max_ds_step, dtype=xp.float64)))
            n_sub = max(n_sub, 1)

            remaining = ds_total

            for _sub in range(n_sub):
                # Only sub-step rays that are still active and still have remaining distance to advance.
                active_sub = (
                    (~hit)
                    & (~fell_in)
                    & (traveled < float(self.scene.bounds.far_distance))
                    & (remaining > 0.0)
                )
                if not bool(xp.any(active_sub)):
                    break

                # Pre-substep SDF. This matters: crossing detection must be per sub-step.
                dist_pre = self.scene.surface.sdf(p)

                # Sub-step size is capped by max_ds_step and also by remaining.
                # Sub-step size is capped by max_ds_step, by remaining,
                # AND adaptively by distance to the object to reduce grazing overshoot.
                #
                # The factor k controls how conservative we are near the surface:
                #   smaller k -> smaller steps near the object -> less overshoot (slower).
                k = float(getattr(cfg, "ds_surface_factor", 0.25))  # try 0.25, then 0.1 if needed

                ds_cap_surface = k * xp.maximum(
                    dist_pre,
                    xp.asarray(float(cfg.eps), dtype=xp.float64),
                )

                ds_cap = xp.minimum(
                    xp.asarray(max_ds_step, dtype=xp.float64),
                    ds_cap_surface,
                )

                ds = xp.minimum(remaining, ds_cap)
                ds = ds * active_sub.astype(xp.float64)

                p_new, rd_new, fell_now = self.scene.bh.advance_batch(state, ds)

                m3 = active_sub[..., None]
                p = xp.where(m3, p_new, p)
                rd = xp.where(m3, rd_new, rd)

                # Post-substep hit test to avoid tunneling.
                dist_post = self.scene.surface.sdf(p)
                hit_post = dist_post < float(cfg.eps)

                # For signed SDFs (sphere), crossing into the object means we hit it even if we skipped eps.
                # Doing this per sub-step prevents "enter + exit in one big step" misses.
                crossed = (dist_pre > 0.0) & (dist_post <= 0.0)

                hit = hit | (active_sub & (hit_post | crossed))

                # Only mark fell_in for rays that did not just hit the object.
                fell_in = fell_in | (active_sub & (~hit_post) & (~crossed) & fell_now)

                traveled = traveled + ds
                remaining = remaining - ds

        return ImageMarchResult(hit=hit, fell_in=fell_in, traveled=traveled)

