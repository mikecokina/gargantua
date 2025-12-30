from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class RayMarchConfig:
    max_steps: int = 160
    eps: float = 1e-3
    min_step: float = 1e-3
    max_step: float = 0.25
    slow_radius: float = 1.2
    slow_factor: float = 0.2


Termination = Literal["hit", "horizon", "far", "max_steps"]


@dataclass(frozen=True, slots=True)
class RayMarchResult:
    hit_object: bool
    fell_in_bh: bool
    escaped_scene: bool
    termination: Termination
    points: Any
