"""Not used since library is not stable!"""
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

    Backwards-compatible:
      - If *_2d fields are None, the plotter uses the XZ hardcoded mode with cam_pos/look_at/rays_xz.

    Camera-plane (or any custom 2D plane) mode:
      - Provide cam_2d, look_2d, rays_2d, sphere_center_2d, bh_center_2d.
      - The plotter then draws in that 2D coordinate system.
    """

    img: np.ndarray

    # World-space (used by default XZ mode)
    cam_pos: np.ndarray
    look_at: np.ndarray
    rays_xz: list[np.ndarray]

    # Optional: custom 2D coords (u,v) for the right panel
    cam_2d: np.ndarray | None = None  # (2,)
    look_2d: np.ndarray | None = None  # (2,)
    rays_2d: list[np.ndarray] | None = None  # list of (K,2)

    sphere_center_2d: np.ndarray | None = None  # (2,)
    bh_center_2d: np.ndarray | None = None  # (2,)


class RenderSplitPlotter:
    """Plot rendered view + optional 2D debug view (default XZ, or custom 2D plane)."""

    def __init__(self, *, two_d_split: bool = True) -> None:
        self.two_d_split = two_d_split

        if two_d_split:
            self.fig, (self.ax_img, self.ax_top) = plt.subplots(1, 2, figsize=(12, 5))
        else:
            self.fig, self.ax_img = plt.subplots(1, 1, figsize=(8, 5))
            self.ax_top = None

        self.ax_img.axis("off")
        self.ax_img.set_aspect("equal", adjustable="box")
        self.ax_img.set_title("Rendered view")

        # Artists (initialized in animate)
        self.im: Any = None
        self.cam_dot: Any = None
        self.look_line: Any = None
        self.ray_lines: list[Any] = []

        # Optional dynamic circle outlines for custom 2D plane mode
        self.sphere_outline: Any = None
        self.bh_outline: Any = None

    @staticmethod
    def _circle_xy(center_2d: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
        th = np.linspace(0.0, 2.0 * np.pi, 256)
        x = center_2d[0] + radius * np.cos(th)
        y = center_2d[1] + radius * np.sin(th)
        return x, y

    def _init_topdown_static_xz(
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

        # reset custom outlines
        self.sphere_outline = None
        self.bh_outline = None

    def _init_topdown_custom_2d(
            self,
            xlim: tuple[float, float],
            ylim: tuple[float, float],
            title: str,
            xlabel: str,
            ylabel: str,
    ) -> None:
        if self.ax_top is None:
            return
        ax = self.ax_top
        ax.cla()
        ax.set_aspect("equal", "box")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        # Create empty outlines that we will update per frame
        (self.sphere_outline,) = ax.plot([], [], linewidth=2, label="Sphere")
        (self.bh_outline,) = ax.plot([], [], linewidth=2, label="Black hole")
        ax.legend()

    def _init_ray_lines(self, rays_2d: "Sequence[np.ndarray]") -> None:
        if self.ax_top is None:
            return

        for ln in self.ray_lines:
            # noinspection PyBroadException
            try:  # noqa: SIM105
                ln.remove()
            except Exception:  # pragma: no cover
                pass
        self.ray_lines = []

        for ray in rays_2d:
            if ray is None or len(ray) < 2:
                (ln,) = self.ax_top.plot([], [], linewidth=1.0)
            else:
                (ln,) = self.ax_top.plot(ray[:, 0], ray[:, 1], linewidth=1.0)
            self.ray_lines.append(ln)

    def animate(  # noqa: C901
            self,
            frames: "Sequence[RenderFrame]",
            sphere_center_xz: np.ndarray,
            sphere_radius: float,
            bh_center_xz: np.ndarray,
            bh_horizon: float,
            xlim: tuple[float, float] = (-6.0, 6.0),
            zlim: tuple[float, float] = (-3.0, 12.0),
            interval_ms: int = 120,
            topdown_title: str = "Top-down (XZ)",
            # New: labels for custom 2D plane mode (ignored in XZ mode)
            topdown_xlabel: str = "x",
            topdown_ylabel: str = "z",
    ) -> FuncAnimation:
        if not frames:
            msg = "frames is empty"
            raise ValueError(msg)

        # Init image
        self.im = self.ax_img.imshow(frames[0].img, aspect="equal")
        self.ax_img.set_title("Rendered view")

        # Decide mode from first frame
        f0 = frames[0]
        custom_2d = (
                f0.cam_2d is not None
                and f0.look_2d is not None
                and f0.rays_2d is not None
                and f0.sphere_center_2d is not None
                and f0.bh_center_2d is not None
        )

        if self.two_d_split and self.ax_top is not None:
            if custom_2d:
                self._init_topdown_custom_2d(
                    xlim=xlim,
                    ylim=zlim,  # reuse args, but they are now (u,v)
                    title=topdown_title,
                    xlabel=topdown_xlabel,
                    ylabel=topdown_ylabel,
                )

                # Camera dot and look line in custom 2D
                self.cam_dot = self.ax_top.scatter(f0.cam_2d[0], f0.cam_2d[1], s=60)
                (self.look_line,) = self.ax_top.plot(
                    [f0.cam_2d[0], f0.look_2d[0]],
                    [f0.cam_2d[1], f0.look_2d[1]],
                )
                self._init_ray_lines(f0.rays_2d or [])
            else:
                self._init_topdown_static_xz(
                    sphere_center_xz=sphere_center_xz,
                    sphere_radius=float(sphere_radius),
                    bh_center_xz=bh_center_xz,
                    bh_horizon=float(bh_horizon),
                    xlim=xlim,
                    zlim=zlim,
                    title=topdown_title,
                )

                # Camera dot and look line in XZ
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

            # Update right panel
            if self.two_d_split and self.ax_top is not None:
                if custom_2d:
                    # Update outlines
                    if self.sphere_outline is not None and f.sphere_center_2d is not None:
                        sx, sy = self._circle_xy(f.sphere_center_2d, float(sphere_radius))
                        self.sphere_outline.set_data(sx, sy)
                        artists.append(self.sphere_outline)

                    if self.bh_outline is not None and f.bh_center_2d is not None:
                        bx, by = self._circle_xy(f.bh_center_2d, float(bh_horizon))
                        self.bh_outline.set_data(bx, by)
                        artists.append(self.bh_outline)

                    # Camera dot and look line
                    if self.cam_dot is not None and f.cam_2d is not None:
                        self.cam_dot.set_offsets([[f.cam_2d[0], f.cam_2d[1]]])
                        artists.append(self.cam_dot)

                    if self.look_line is not None and f.cam_2d is not None and f.look_2d is not None:
                        self.look_line.set_data(
                            [f.cam_2d[0], f.look_2d[0]],
                            [f.cam_2d[1], f.look_2d[1]],
                        )
                        artists.append(self.look_line)

                    rays = f.rays_2d or []
                    if len(rays) != len(self.ray_lines):
                        self._init_ray_lines(rays)

                    for ln, ray in zip(self.ray_lines, rays, strict=False):
                        if ray is None or len(ray) < 2:
                            ln.set_data([], [])
                        else:
                            ln.set_data(ray[:, 0], ray[:, 1])
                        artists.append(ln)

                else:
                    # XZ mode (existing behavior)
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
