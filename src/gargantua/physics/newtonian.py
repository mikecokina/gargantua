from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gargantua.backend import ArrayModule
from gargantua.math_utils import normalize


@dataclass(frozen=True, slots=True)
class BlackHoleBender:
    """
    Newton-like ray bender.

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
