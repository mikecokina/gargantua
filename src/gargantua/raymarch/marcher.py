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


@dataclass(frozen=True, slots=True)
class ImageMarchResult:
    hit: Any
    fell_in: Any
    traveled: Any


class ImageMarcher:
    def __init__(self, xp: ArrayModule, config: ImageMarchConfig, scene: Any) -> None:
        """Initialise the marcher."""
        self.xp = xp
        self.config = config
        self.scene = scene

    def _segment_hits_sphere(self, p: Any, d: Any, ds: Any, center: Any, radius: float) -> Any:
        """Check whether the segment from p to p + d*ds intersects sphere(center, radius).

        p: (H,W,3)
        d: (H,W,3) normalized
        ds: (H,W)
        returns: (H,W) bool
        """
        xp = self.xp

        oc = p - center[None, None, :]  # (H,W,3)
        a = xp.sum(d * d, axis=-1)  # ~1
        b = 2.0 * xp.sum(oc * d, axis=-1)
        c = xp.sum(oc * oc, axis=-1) - radius * radius

        disc = b * b - 4.0 * a * c
        hit = disc >= 0.0
        if not bool(xp.any(hit)):
            return hit

        s = xp.sqrt(xp.maximum(disc, 0.0))
        t0 = (-b - s) / (2.0 * a)
        t1 = (-b + s) / (2.0 * a)

        # segment is [0, ds]
        in0 = (t0 >= 0.0) & (t0 <= ds)
        in1 = (t1 >= 0.0) & (t1 <= ds)
        return hit & (in0 | in1)

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

        for _ in range(int(cfg.max_steps)):
            active = (~hit) & (~fell_in) & (traveled < float(self.scene.bounds.far_distance))
            if not bool(xp.any(active)):
                break

            # BH capture
            dist_bh = xp.linalg.norm(p - self.scene.bh.center[None, None, :], axis=-1)
            fell_in = fell_in | (active & (dist_bh < float(self.scene.bh.horizon)))

            active = (~hit) & (~fell_in) & (traveled < float(self.scene.bounds.far_distance))
            if not bool(xp.any(active)):
                break

            d_obj = self.scene.surface.sdf(p)  # (H,W)
            hit = hit | (active & (d_obj < float(cfg.eps)))

            active = (~hit) & (~fell_in) & (traveled < float(self.scene.bounds.far_distance))
            if not bool(xp.any(active)):
                break

            ds = xp.clip(d_obj, float(cfg.min_step), float(cfg.max_step))
            ds = ds * active.astype(xp.float64)

            # NEW: horizon intersection on the segment we are about to take
            hits_horizon = self._segment_hits_sphere(p, rd, ds, self.scene.bh.center, float(self.scene.bh.horizon))
            fell_in = fell_in | (active & hits_horizon)

            # recompute active after horizon
            active = (~hit) & (~fell_in) & (traveled < float(self.scene.bounds.far_distance))
            ds = ds * active.astype(xp.float64)

            p = p + rd * ds[..., None]
            traveled = traveled + ds

            rd = self.scene.bh.bend_batch(rd, p, ds)

        return ImageMarchResult(hit=hit, fell_in=fell_in, traveled=traveled)
