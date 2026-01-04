from __future__ import annotations

from dataclasses import dataclass
from typing import List

from gargantua.physics.geodesics.schwarzschild_3d import SchwarzschildGeodesicStepper3D
from gargantua.physics.newtonian import BlackHoleBender
from gargantua.protocols import Drawable2D, SDF


@dataclass(frozen=True, slots=True)
class SceneBounds:
    """Global scene limits."""

    far_distance: float


@dataclass(frozen=True, slots=True)
class Scene:
    """Complete scene definition."""

    surface: SDF
    bh: BlackHoleBender | SchwarzschildGeodesicStepper3D
    bounds: SceneBounds
    drawables: List[Drawable2D]
