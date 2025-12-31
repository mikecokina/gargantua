from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import matplotlib as mpl
import numpy as np
from matplotlib.animation import FuncAnimation

# IMPORTANT: set backend before importing pyplot
_BACKEND = os.environ.get("GARGANTUA_MPL_BACKEND", "").strip()
if _BACKEND:
    mpl.use(_BACKEND, force=True)
else:
    for candidate in ("TkAgg", "QtAgg", "Agg"):
        # noinspection PyBroadException
        try:
            mpl.use(candidate, force=True)
            break
        except Exception:  # pragma: no cover  # noqa: BLE001, S112
            continue

import matplotlib.pyplot as plt  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True, slots=True)
class RenderFrame:
    """Single render frame for animation.

    Attributes
    ----------
    img:
        Rendered RGB image (H, W, 3), numpy float32/float64 in [0, 1].
    cam_pos:
        Camera position in world coordinates (3,).
    look_at:
        Look-at point in world coordinates (3,).
    rays_xz:
        Ray polylines in top-down space: list of (K, 2) arrays with columns [x, z].
        This is the fan-of-rays debug view.

    """

    img: np.ndarray
    cam_pos: np.ndarray
    look_at: np.ndarray
    rays_xz: list[np.ndarray]


class RenderSplitPlotter:
    """Plot rendered view + optional 2D top-down XZ debug view with ray fan."""

    def __init__(self, *, two_d_split: bool = True) -> None:
        """Initialize the plot."""
        self.two_d_split = two_d_split

        if two_d_split:
            self.fig, (self.ax_img, self.ax_top) = plt.subplots(1, 2, figsize=(12, 5))
        else:
            self.fig, self.ax_img = plt.subplots(1, 1, figsize=(8, 5))
            self.ax_top = None

        self.ax_img.axis("off")
        self.ax_img.set_title("Rendered view")

        # Artists (initialized in animate)
        self.im: Any = None
        self.cam_dot: Any = None
        self.look_line: Any = None
        self.ray_lines: list[Any] = []

    def _init_topdown_static(
            self,
            sphere_center_xz: np.ndarray,
            sphere_radius: float,
            bh_center_xz: np.ndarray,
            bh_horizon: float,
            xlim: tuple[float, float],
            zlim: tuple[float, float],
            title: str,
    ) -> None:
        if self.ax_top is None:
            return

        ax = self.ax_top
        ax.cla()
        ax.set_aspect("equal", "box")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("z")

        th = np.linspace(0.0, 2.0 * np.pi, 256)

        # Sphere outline (XZ)
        ax.plot(
            sphere_center_xz[0] + sphere_radius * np.cos(th),
            sphere_center_xz[1] + sphere_radius * np.sin(th),
            linewidth=2,
            label="Sphere",
        )

        # BH horizon outline (XZ)
        ax.plot(
            bh_center_xz[0] + bh_horizon * np.cos(th),
            bh_center_xz[1] + bh_horizon * np.sin(th),
            linewidth=2,
            label="Black hole",
        )

        ax.set_xlim(*xlim)
        ax.set_ylim(*zlim)
        ax.legend()

    def _init_ray_lines(self, rays_xz: Sequence[np.ndarray]) -> None:
        """Create/replace ray polyline artists for top-down view."""
        if self.ax_top is None:
            return

        # Remove any existing ray lines
        for ln in self.ray_lines:
            # noinspection PyBroadException
            try:  # noqa: SIM105
                ln.remove()
            except Exception:  # pragma: no cover  # noqa: BLE001, S110
                pass
        self.ray_lines = []

        for ray in rays_xz:
            if ray is None or len(ray) < 2:
                (ln,) = self.ax_top.plot([], [], linewidth=1.0)
            else:
                (ln,) = self.ax_top.plot(ray[:, 0], ray[:, 1], linewidth=1.0)
            self.ray_lines.append(ln)

    def animate(  # noqa: C901
            self,
            frames: Sequence[RenderFrame],
            sphere_center_xz: np.ndarray,
            sphere_radius: float,
            bh_center_xz: np.ndarray,
            bh_horizon: float,
            xlim: tuple[float, float] = (-6.0, 6.0),
            zlim: tuple[float, float] = (-3.0, 12.0),
            interval_ms: int = 120,
            topdown_title: str = "Top-down (XZ)",
    ) -> FuncAnimation:
        if not frames:
            msg = "frames is empty"
            raise ValueError(msg)

        # Init image
        self.im = self.ax_img.imshow(frames[0].img)
        self.ax_img.set_title("Rendered view")

        # Init topdown (optional)
        if self.two_d_split and self.ax_top is not None:
            self._init_topdown_static(
                sphere_center_xz=sphere_center_xz,
                sphere_radius=float(sphere_radius),
                bh_center_xz=bh_center_xz,
                bh_horizon=float(bh_horizon),
                xlim=xlim,
                zlim=zlim,
                title=topdown_title,
            )

            f0 = frames[0]

            # Camera dot and look line
            self.cam_dot = self.ax_top.scatter(f0.cam_pos[0], f0.cam_pos[2], s=60)
            (self.look_line,) = self.ax_top.plot(
                [f0.cam_pos[0], f0.look_at[0]],
                [f0.cam_pos[2], f0.look_at[2]],
            )

            self._init_ray_lines(f0.rays_xz or [])

        def _update(i: int) -> list[Any]:
            f = frames[i]
            artists: list[Any] = []

            # Update image
            if self.im is not None:
                self.im.set_data(f.img)
                artists.append(self.im)

            # Update topdown
            if self.two_d_split and self.ax_top is not None:
                if self.cam_dot is not None:
                    self.cam_dot.set_offsets([[f.cam_pos[0], f.cam_pos[2]]])
                    artists.append(self.cam_dot)

                if self.look_line is not None:
                    self.look_line.set_data(
                        [f.cam_pos[0], f.look_at[0]],
                        [f.cam_pos[2], f.look_at[2]],
                    )
                    artists.append(self.look_line)

                rays = f.rays_xz or []
                if len(rays) != len(self.ray_lines):
                    self._init_ray_lines(rays)

                for ln, ray in zip(self.ray_lines, rays, strict=False):
                    if ray is None or len(ray) < 2:
                        ln.set_data([], [])
                    else:
                        ln.set_data(ray[:, 0], ray[:, 1])
                    artists.append(ln)

            return artists

        return FuncAnimation(
            self.fig,
            _update,
            frames=len(frames),
            interval=interval_ms,
            repeat=True,
            blit=False,
        )

    @staticmethod
    def show() -> None:
        plt.tight_layout()
        plt.show()
