from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from gargantua.math_utils import normalize

if TYPE_CHECKING:
    from gargantua.backend import ArrayModule


@dataclass(frozen=True, slots=True)
class BlackHoleBender:
    """Newton-like ray bender.

    d_new = normalize(d - (r / |r|^3) * (ds * mass))
    """

    xp: ArrayModule
    center: Any
    mass: float
    horizon: float
    softening: float = 1e-2

    def bend(self, direction: Any, position: Any, ds: float) -> Any:
        r = position - self.center
        r2 = self.xp.dot(r, r) + self.softening * self.softening
        r_len = self.xp.sqrt(r2)
        r_over_r3 = r / (r_len ** 3)
        d_new = direction - r_over_r3 * (ds * self.mass)
        return normalize(self.xp, d_new)

    def bend_batch(self, direction: Any, position: Any, ds: Any) -> Any:
        """Bends batch.

        Vectorized bend:
        - direction: (..., D)
        - position:  (..., D)
        - ds:        (...,) or (..., 1)
        """
        r = position - self.center
        r2 = self.xp.sum(r * r, axis=-1) + self.softening * self.softening
        r_len = self.xp.sqrt(r2)
        r_over_r3 = r / (r_len[..., None] ** 3)

        d_new = direction - r_over_r3 * (ds[..., None] * self.mass)

        # vectorized normalize
        n = self.xp.linalg.norm(d_new, axis=-1, keepdims=True)
        n = self.xp.maximum(n, self.xp.asarray(1e-12, dtype=self.xp.float64))
        return d_new / n
