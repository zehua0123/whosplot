import numpy as np
from typing import Optional, Tuple, Dict, Any
from matplotlib.patches import FancyArrowPatch, ArrowStyle
import matplotlib.patheffects as pe


class Primitives2D:
    """
    Pretty 2D primitives (vector, wall) for 2D plots.

    Notes on conventions
    --------------------
    - Colors are NOT taken from any global/system colormap. Pass them explicitly.
      * Vector:   `arrow_color`
      * Wall & hatches (and normal arrow by default): `wall_color`
    - For `draw_wall2d`, the provided `normal` is treated as pointing to the
      **exterior** side. Diagonal hatches are drawn on that side to indicate
      outside vs. inside.
    """

    def __init__(self, host):
        self.h = host  # external Run instance
        # Access to axes/plt through host is kept to remain compatible with Run.

    # ---- internal helpers ----
    def _ax(self, fig_num: int):
        return self.h.axs[
            int(fig_num // self.h._Run__cols), int(fig_num % self.h._Run__cols)
        ]

    def _scale(self, fig_num: int, base: float = 1.0) -> float:
        try:
            return float(self.h._Run__scale_per_fig[fig_num]) * base
        except Exception:
            return base

    # ---- public API ----
    def draw_vector2d(
        self,
        fig_num: int,
        start: Tuple[float, float],
        end: Tuple[float, float],
        label: Optional[str] = None,
        place: float = 0.6,
        label_offset_pts: float = 6.0,
        arrow_color: str = "black",
        linewidth: float = 2.0,
        mutation_scale: float = 16.0,
        arrow_kwargs: Optional[Dict[str, Any]] = None,
        text_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Draw a 2D vector with a TikZ-style (stealth-like) arrow head and an optional label.

        Parameters
        ----------
        fig_num : int
            Subplot index (row-major, same convention as Run.two_d_subplots).
        start, end : (float, float)
            Start and end coordinates.
        label : str, optional
            Text to display along the vector. If None, defaults to the components
            "(dx, dy)" rounded to 2 decimals.
        place : float in [0, 1]
            Relative position along the arrow where the label is placed.
        label_offset_pts : float
            Offset from the arrow in screen points.
        arrow_color : str
            Color of the arrow (no system palette).
        linewidth : float
            Arrow shaft line width in points.
        head_length_pts, head_width_pts : float
            Arrow head size (points), styled like TikZ 'Stealth' (sleek triangle).
        tail_width : float
            Relative shaft width for ArrowStyle.Simple (in axis coordinates).
        arrow_kwargs, text_kwargs : dict
            Additional style overrides.

        Returns
        -------
        dict with keys 'arrow', 'text' (if label is not None).
        """
        ax = self._ax(fig_num)
        p0 = np.asarray(start, dtype=float)
        p1 = np.asarray(end, dtype=float)
        v = p1 - p0
        L = float(np.hypot(v[0], v[1]))
        if L == 0.0:
            raise ValueError("Zero-length vector: 'start' and 'end' cannot coincide.")

        style = ArrowStyle("->")

        ak = {
            "arrowstyle": style,
            "linewidth": linewidth * self._scale(fig_num, 1.0),
            "color": arrow_color,
            "shrinkA": 0.0,
            "shrinkB": 0.0,
            "joinstyle": "round",
            "capstyle": "round",
            "mutation_scale": mutation_scale,
        }
        if arrow_kwargs:
            ak.update(arrow_kwargs)

        arr = FancyArrowPatch(
            posA=(float(p0[0]), float(p0[1])),
            posB=(float(p1[0]), float(p1[1])),
            **ak,
        )
        # Light stroke for contrast, but keep it subtle
        arr.set_path_effects(
            [
                pe.Stroke(linewidth=arr.get_linewidth() + 0.8, foreground="white"),
                pe.Normal(),
            ]
        )
        ax.add_patch(arr)

        # Label (no white bbox)
        dx, dy = v[0], v[1]
        if label is None:
            label = f"({dx:.2f}, {dy:.2f})"
        label_pos = p0 + np.clip(place, 0.0, 1.0) * v
        angle_deg = float(np.degrees(np.arctan2(dy, dx)))

        tk = {
            "fontsize": 12 * self._scale(fig_num, 1.0),
            "rotation": angle_deg,
            "rotation_mode": "anchor",
            "ha": "center",
            "va": "center",
            # No bbox to avoid white frame
            "color": arrow_color,
        }
        if text_kwargs:
            tk.update(text_kwargs)

        txt = ax.annotate(
            label,
            xy=(label_pos[0], label_pos[1]),
            xytext=(0, label_offset_pts),
            textcoords="offset points",
            **tk,
        )

        # Axis niceties
        try:
            ax.set_aspect("equal", adjustable="datalim")
        except Exception:
            pass
        ax.margins(0.05, 0.05)

        return {"arrow": arr, "text": txt}

    def draw_wall2d(
        self,
        fig_num: int,
        start: Tuple[float, float],
        end: Tuple[float, float],
        normal: Tuple[float, float],
        wall_color: str = "black",
        linewidth: float = 3.0,
        show_normal: bool = True,
        normal_length_frac: float = 0.15,
        normal_kwargs: Optional[Dict[str, Any]] = None,
        hatch: bool = True,
        hatch_spacing_frac: float = 0.12,
        hatch_length_frac: float = 0.10,
        hatch_offset_frac: float = 0.02,
        hatch_angle_deg: float = 45.0,
        wall_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Draw a wall segment and diagonal hatches on the EXTERIOR side (the side
        pointed by `normal`) to indicate outside vs. inside.

        Parameters
        ----------
        fig_num : int
            Subplot index.
        start, end : (float, float)
            Endpoints of the wall segment.
        normal : (float, float)
            Wall outward normal (treated as exterior side).
        wall_color : str
            Color for wall line, hatches, and normal arrow (unless overridden).
        linewidth : float
            Wall line width (points).
        show_normal : bool
            If True, draw a small normal arrow at the midpoint (same color).
        normal_length_frac : float
            Normal arrow length as a fraction of wall length.
        hatch : bool
            If True, draw diagonal hatch ticks on the exterior side.
        hatch_spacing_frac : float
            Spacing between hatch centers as a fraction of wall length.
        hatch_length_frac : float
            Length of each hatch tick as a fraction of wall length.
        hatch_offset_frac : float
            Offset of hatch centers away from the wall (along normal), as a fraction of wall length.
        hatch_angle_deg : float
            Angle of hatch ticks relative to the wall tangent (degrees).
        wall_kwargs, normal_kwargs : dict
            Style overrides.

        Returns
        -------
        dict with keys 'line', optional 'normal_arrow', and 'hatch_lines' (list).
        """
        ax = self._ax(fig_num)
        p0 = np.asarray(start, dtype=float)
        p1 = np.asarray(end, dtype=float)
        t = p1 - p0
        Lt = float(np.hypot(t[0], t[1]))
        if Lt == 0.0:
            raise ValueError("Wall needs a non-degenerate segment: 'start' != 'end'.")

        n = np.asarray(normal, dtype=float)
        Ln = float(np.hypot(n[0], n[1]))
        if Ln == 0.0:
            raise ValueError("Normal vector cannot be zero.")
        n_hat = n / Ln
        t_hat = t / Lt

        # Wall line
        wk = {
            "linewidth": linewidth * self._scale(fig_num, 1.0),
            "solid_capstyle": "round",
            "alpha": 1.0,
            "color": wall_color,
        }
        if wall_kwargs:
            wk.update(wall_kwargs)

        (line_obj,) = ax.plot([p0[0], p1[0]], [p0[1], p1[1]], **wk)
        line_obj.set_path_effects(
            [
                pe.Stroke(linewidth=line_obj.get_linewidth() + 0.8, foreground="white"),
                pe.Normal(),
            ]
        )

        # Optional normal arrow at midpoint
        arr_obj = None
        mid = 0.5 * (p0 + p1)
        if show_normal:
            Lnorm = normal_length_frac * Lt
            tip = mid + n_hat * Lnorm
            nk = {
                "arrowstyle": ArrowStyle("->"),
                "linewidth": (linewidth * 0.8) * self._scale(fig_num, 1.0),
                "color": wall_color,
                "shrinkA": 0.0,
                "shrinkB": 0.0,
                "joinstyle": "round",
                "capstyle": "round",
                "mutation_scale": 16.0 * self._scale(fig_num, 1.0),
            }
            if normal_kwargs:
                nk.update(normal_kwargs)

            arr_obj = FancyArrowPatch(
                posA=(float(mid[0]), float(mid[1])),
                posB=(float(tip[0]), float(tip[1])),
                **nk,
            )
            arr_obj.set_path_effects(
                [
                    pe.Stroke(
                        linewidth=arr_obj.get_linewidth() + 0.6, foreground="white"
                    ),
                    pe.Normal(),
                ]
            )
            ax.add_patch(arr_obj)

        # Diagonal hatches on the exterior (normal) side
        hatch_objs = []
        if hatch:
            spacing = max(
                1, int(np.floor(1.0 / max(1e-6, hatch_spacing_frac)))
            )  # guard, not used directly
            step = hatch_spacing_frac * Lt
            seg_len = hatch_length_frac * Lt
            offset = hatch_offset_frac * Lt

            # hatch direction relative to tangent
            theta = np.deg2rad(hatch_angle_deg)
            u = np.cos(theta) * t_hat + np.sin(theta) * n_hat  # unit along hatch tick

            # place hatches along [margin, Lt - margin]
            margin = 0.08 * Lt
            s = margin
            while s < Lt - margin:
                center = p0 + s * t_hat + offset * n_hat
                a = center - 0.5 * seg_len * u
                b = center + 0.5 * seg_len * u
                (hl,) = ax.plot(
                    [a[0], b[0]],
                    [a[1], b[1]],
                    color=wall_color,
                    linewidth=(linewidth * 0.8) * self._scale(fig_num, 1.0),
                    solid_capstyle="round",
                )
                hatch_objs.append(hl)
                s += step

        # Axis niceties
        try:
            ax.set_aspect("equal", adjustable="datalim")
        except Exception:
            pass
        ax.margins(0.05, 0.05)

        out = {"line": line_obj}
        if arr_obj is not None:
            out["normal_arrow"] = arr_obj
        if hatch_objs:
            out["hatch_lines"] = hatch_objs
        return out
