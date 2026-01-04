from __future__ import annotations

import os
from typing import TYPE_CHECKING

import matplotlib as mpl

# IMPORTANT: set backend before importing pyplot
# - If running inside PyCharm scientific mode, their custom backend can break.
# - Prefer a stable GUI backend if available; fallback to Agg.
_BACKEND = os.environ.get("GARGANTUA_MPL_BACKEND", "").strip()
if _BACKEND:
    mpl.use(_BACKEND, force=True)
else:
    # Try stable interactive backends first; fallback to Agg.
    for candidate in ("TkAgg", "QtAgg", "Agg"):
        # noinspection PyBroadException
        try:
            mpl.use(candidate, force=True)
            break
        except Exception:  # pragma: no cover  # noqa: BLE001, S112
            continue

import matplotlib.pyplot as plt  # noqa: E402

from gargantua.backend import ArrayModule, to_numpy  # noqa: E402

if TYPE_CHECKING:
    import numpy as np

    from gargantua.protocols import Drawable2D
    from gargantua.raymarch.config import RayMarchResult


class Plotter2D:
    """Matplotlib plotter for the 2D demo."""

    def __init__(self) -> None:
        """Initialize the plotter."""
        _, ax = plt.subplots(figsize=(8, 8))
        self.ax = ax
        ax.set_aspect("equal", "box")
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.set_title("Gargantua - 2D Ray Marching")

    def draw_drawable(self, xp: ArrayModule, drawable: Drawable2D, linewidth: float = 2.0) -> None:
        pts = to_numpy(xp, drawable.polyline())
        self.ax.plot(pts[:, 0], pts[:, 1], linewidth=linewidth)

    def draw_point(self, p: np.ndarray, label: str | None = None) -> None:
        self.ax.scatter([p[0]], [p[1]], s=80)
        if label:
            self.ax.text(float(p[0] + 0.1), float(p[1] + 0.1), label)

    def draw_ray(self, xp: ArrayModule, result: RayMarchResult) -> None:
        pts = to_numpy(xp, result.points)
        self.ax.plot(pts[:, 0], pts[:, 1], linewidth=1)

        if pts.shape[0] == 0:
            return

        end = pts[-1]

        # Endpoint markers based on termination reason
        if result.termination == "hit":
            self.ax.scatter([end[0]], [end[1]], marker="x", s=70)
            return

        if result.termination == "horizon":
            self.ax.scatter([end[0]], [end[1]], marker="o", s=40)
            return

        if result.termination == "far":
            self.ax.scatter([end[0]], [end[1]], marker=".", s=25)
            return

        if result.termination == "max_steps":
            self.ax.scatter([end[0]], [end[1]], marker="s", s=25)
            return

        # Fallback (should not happen)
        self.ax.scatter([end[0]], [end[1]], marker=".", s=20)

    def show(self, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)
        plt.tight_layout()
        plt.show()

    def save(self, path: str, xlim: tuple[float, float], ylim: tuple[float, float], dpi: int = 150) -> None:
        """Save figure to disk (useful if running headless with Agg)."""
        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)
        plt.tight_layout()
        plt.savefig(path, dpi=dpi)
