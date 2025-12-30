from __future__ import annotations

from typing import Any

from gargantua.backend import ArrayModule
from gargantua.math_utils import clamp_float, normalize
from gargantua.physics.base import RayTracer2D
from gargantua.physics.newtonian import BlackHoleBender
from gargantua.raymarch.config import RayMarchConfig, RayMarchResult
from gargantua.scene import Scene


class RayMarcher(RayTracer2D):
    """Dimension-agnostic ray marcher."""

    HORIZON_EPS: float = 1e-6

    def __init__(self, xp: ArrayModule, config: RayMarchConfig, scene: Scene) -> None:
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
