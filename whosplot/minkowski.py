import numpy as np
from typing import Dict, Optional, List, Tuple


class Minkowski:
    """
    2D superellipse sampling, affine transforms, convex polygon preprocessing,
    and a linear-time Minkowski sum (thus allowing Minkowski difference via A ⊕ (-B)).

    This class *references* a host object (typically a `Run` instance) so it can
    use the global Matplotlib state and the project's global colormap. Nothing
    is copied; we access the host's `fig`, `axs`, `plt`, and color configuration.
    """

    def __init__(self, host):
        """
        Parameters
        ----------
        host : object
            External runner/host that owns the figure/axes grid, layout and
            global styling (e.g., a `Run` instance). We rely on the host's
            colormap to keep colors consistent across the whole framework.
        """
        self.h = host  # external Run instance

    def _get_cmap(self, fig_index: int):
        """
        Fetch the global colormap from the host for a given subplot index.

        Behavior
        --------
        - If the host stores a list/tuple of colormaps, we cycle by index.
        - If the host stores a single colormap, we return it as-is.
        - If anything goes wrong, fall back to Matplotlib's 'tab10'.
        """
        try:
            cm = self.h._Run__color_map
            if isinstance(cm, (list, tuple)):
                return cm[int(fig_index) % len(cm)]
            return cm
        except Exception:
            # Graceful fallback
            return self.h.plt.get_cmap("tab10")

    def _cm_color(
        self, fig_index: int, k: int = 0, n: int = 1
    ) -> Optional[Tuple[float, float, float, float]]:
        cmap = self._get_cmap(fig_index)
        try:
            colors = self.h.plt.get_cmap(cmap).colors
            if colors is not None and len(colors) > 0:
                L = max(1, len(colors) - 1)
                idx = int(k * L / max(1, n - 1))
                idx = 0 if idx < 0 else (L if idx > L else idx)
                return colors[idx]
            t = 0.0 if n <= 1 else float(k) / float(n - 1)
            return cmap(float(np.clip(t, 0.0, 1.0)))
        except Exception:
            return None

    # -------------------------
    # Core geometry helpers
    # -------------------------
    def _spow(self, x, p):
        """Signed power: sign(x) * |x|**p (useful for superellipse exponents)."""
        return np.sign(x) * (np.abs(x) ** p)

    def _superellipse2d_points(
        self,
        a: float = 1.0,
        b: float = 1.0,
        p_x: float = 1.0,
        p_y: float = 1.0,
        num: int = 720,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample a 2D superellipse: |x/a|^(2/p_x) + |y/b|^(2/p_y) = 1 (re-param in cosine/sine).

        Parameters
        ----------
        a, b : float
            Radii along x and y axes.
        p_x, p_y : float
            Superellipse exponents in x/y.
        num : int
            Number of samples on [0, 2π].

        Returns
        -------
        (x, y) : tuple of ndarray
            Arrays of length `num` with the sampled boundary points.
        """
        t = np.linspace(0.0, 2.0 * np.pi, int(num), endpoint=True)
        x = a * self._spow(np.cos(t), p_x)
        y = b * self._spow(np.sin(t), p_y)
        return x, y

    def _transform2d(
        self, x, y, center=(0.0, 0.0), angle: float = 0.0, degrees: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply a rotation (around origin) and then translate to `center`.

        Parameters
        ----------
        x, y : array-like
            Coordinate arrays to transform (same length).
        center : (float, float)
            Translation applied after rotation.
        angle : float
            Rotation angle (degrees by default).
        degrees : bool
            If True, `angle` is treated as degrees; otherwise radians.

        Returns
        -------
        (X, Y) : tuple of ndarray
            Transformed coordinates.
        """
        theta = np.deg2rad(angle) if degrees else angle
        c, s = np.cos(theta), np.sin(theta)
        cx, cy = center
        X = c * x - s * y + cx
        Y = s * x + c * y + cy
        return X, Y

    def _polygon_area(self, P: np.ndarray) -> float:
        """Signed polygon area (positive for CCW)."""
        x = P[:, 0]
        y = P[:, 1]
        return 0.5 * float(
            (x[:-1] * y[1:] - x[1:] * y[:-1]).sum() + (x[-1] * y[0] - x[0] * y[-1])
        )

    def _ensure_ccw(self, P: np.ndarray) -> np.ndarray:
        """Ensure polygon vertices are CCW; reverse if necessary."""
        if self._polygon_area(P) < 0.0:
            return P[::-1].copy()
        return P

    def _start_min_xy(self, P: np.ndarray) -> int:
        """Index of the lexicographically smallest vertex (x, then y)."""
        return int(np.lexsort((P[:, 1], P[:, 0]))[0])

    def _roll_to_start(self, P: np.ndarray, idx: int) -> np.ndarray:
        """Rotate vertex list so that P[idx] becomes the first vertex."""
        return np.roll(P, -int(idx), axis=0)

    def _edge_vectors(self, P: np.ndarray) -> np.ndarray:
        """Edge vectors for a (closed) polygon vertex list (wraps last->first)."""
        return np.vstack((P[(np.arange(len(P)) + 1) % len(P)] - P[np.arange(len(P))]))

    def _minkowski_sum_convex(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Linear-time Minkowski sum for convex polygons: C = A ⊕ B.

        Assumptions
        -----------
        - A and B are convex polygons, vertices ordered CCW.
        - The input may be closed (first==last) or open; we normalize it.

        Returns
        -------
        C : (M, 2) ndarray
            CCW vertex list of the Minkowski sum (no duplicated first/last).
        """

        def _uniq_close(Q):
            # Drop the duplicated last point if the polygon is explicitly closed
            if len(Q) > 1 and np.allclose(Q[0], Q[-1]):
                return Q[:-1]
            return Q

        A = _uniq_close(np.asarray(A, dtype=float))
        B = _uniq_close(np.asarray(B, dtype=float))

        A = self._ensure_ccw(A)
        B = self._ensure_ccw(B)

        ia = self._start_min_xy(A)
        ib = self._start_min_xy(B)
        A0 = self._roll_to_start(A, ia)
        B0 = self._roll_to_start(B, ib)

        eA = self._edge_vectors(A0)
        eB = self._edge_vectors(B0)
        na, nb = len(eA), len(eB)

        s = A0[0] + B0[0]
        C = [s.copy()]

        i = j = 0
        eps = 1e-15
        while i < na or j < nb:
            if i < na and j < nb:
                # Compare polar angles via cross product sign
                cross = eA[i, 0] * eB[j, 1] - eA[i, 1] * eB[j, 0]
                if cross > eps:
                    s = s + eA[i]
                    i += 1
                elif cross < -eps:
                    s = s + eB[j]
                    j += 1
                else:
                    s = s + eA[i] + eB[j]
                    i += 1
                    j += 1
            elif i < na:
                s = s + eA[i]
                i += 1
            else:
                s = s + eB[j]
                j += 1
            C.append(s.copy())

        C = np.array(C, dtype=float)
        if len(C) > 1 and np.allclose(C[0], C[-1]):
            C = C[:-1]
        return C

    def _poly_from_geom(self, g: Dict) -> np.ndarray:
        """
        Normalize a geometry spec into a (N, 2) CCW vertex array.

        Accepted formats
        ----------------
        - Explicit points: g['points'] -> shape (N, 2)
        - Superellipse parameters: a, b, p_x/px, p_y/py, center, angle, num

        Returns
        -------
        P : (N, 2) ndarray
            CCW vertex list representing the boundary.
        """
        if "points" in g and g["points"] is not None:
            P = np.asarray(g["points"], dtype=float)
            if P.ndim != 2 or P.shape[1] != 2:
                raise ValueError("geom['points'] must have shape (N, 2)")
            return self._ensure_ccw(P)

        a = float(g.get("a", 1.0))
        b = float(g.get("b", 1.0))
        p_x = float(g.get("p_x", g.get("px", 1.0)))
        p_y = float(g.get("p_y", g.get("py", 1.0)))
        center = g.get("center", (0.0, 0.0))
        angle = float(g.get("angle", 0.0))
        num = int(g.get("num", 720))

        x, y = self._superellipse2d_points(a, b, p_x, p_y, num)
        X, Y = self._transform2d(x, y, center=center, angle=angle, degrees=True)
        P = np.column_stack([X, Y])
        if np.allclose(P[0], P[-1]):
            P = P[:-1]
        return self._ensure_ccw(P)

    # -------------------------
    # Public plotting API (uses global colormap)
    # -------------------------
    def plot_superellipse2d(
        self,
        fig_num: int,
        a: float,
        b: float,
        p_x: float = 1.0,
        p_y: float = 1.0,
        center=(0.0, 0.0),
        angle: float = 0.0,
        num: int = 720,
        closed: bool = True,
        **plot_kwargs,
    ):
        """
        Draw a single 2D superellipse on the subplot given by `fig_num`.
        Colors default to the host's global colormap unless explicitly provided.
        """
        ax = self.h.axs[
            int(fig_num // self.h._Run__cols), int(fig_num % self.h._Run__cols)
        ]
        x, y = self._superellipse2d_points(a, b, p_x, p_y, num)
        X, Y = self._transform2d(x, y, center=center, angle=angle, degrees=True)
        if closed:
            X = np.r_[X, X[:1]]
            Y = np.r_[Y, Y[:1]]

        # Default color from global colormap (unless user overrides)
        if "color" not in plot_kwargs and "c" not in plot_kwargs:
            color = self._cm_color(fig_num, k=0, n=1)
            if color is not None:
                plot_kwargs.setdefault("color", color)

        (line,) = ax.plot(X, Y, **(plot_kwargs or {}))
        try:
            ax.set_aspect("equal", adjustable="datalim")
        except Exception:
            pass
        ax.margins(0.05, 0.05)
        return line

    def plot_superellipses2d(
        self, geoms: List[Dict], equal_aspect: bool = True, margins: float = 0.05
    ):
        artists_ = []
        for fig_num in range(self.h._Run__figure_number):
            ax = self.h.axs[
                int(fig_num // self.h._Run__cols), int(fig_num % self.h._Run__cols)
            ]
            artists_.append([])
            total = len(geoms)

            n_for_cmap = total

            for j, g in enumerate(geoms):
                a = float(g.get("a", 1.0))
                b = float(g.get("b", 1.0))
                p_x = float(g.get("p_x", g.get("px", 1.0)))
                p_y = float(g.get("p_y", g.get("py", 1.0)))
                center = g.get("center", (0.0, 0.0))
                angle = float(g.get("angle", 0.0))
                num = int(g.get("num", 720))

                draw_kwargs = dict(g.get("draw_kwargs", {}) or {})
                for key in ("color", "c", "facecolor", "edgecolor"):
                    draw_kwargs.pop(key, None)

                color = self._cm_color(fig_num, k=j, n=n_for_cmap)
                if color is not None:
                    draw_kwargs["color"] = color

                artists_[fig_num].append(
                    self.plot_superellipse2d(
                        fig_num,
                        a,
                        b,
                        p_x,
                        p_y,
                        center=center,
                        angle=angle,
                        num=num,
                        **draw_kwargs,
                    )
                )

            if equal_aspect:
                try:
                    ax.set_aspect("equal", adjustable="datalim")
                except Exception:
                    pass
            if margins is not None:
                ax.margins(margins, margins)

            if self.h._Run__figure_number > 1:
                try:
                    self.h._Run__figure_serial(
                        fig_num, use_tex=self.h.plt.rcParams["text.usetex"]
                    )
                except Exception:
                    pass

        return artists_

    def plot_minkowski_difference2d(
        self,
        geom_a: List[Dict],
        geom_b: List[Dict],
        draw_inputs: bool = True,
        inputs_kwargs: Optional[dict] = None,
        result_kwargs: Optional[dict] = None,
        fill: bool = False,
        closed: bool = True,
        equal_aspect: bool = True,
        margins: Optional[float] = 0.05,
    ) -> List[np.ndarray]:
        C_ = []

        for fig_num in range(self.h._Run__figure_number):
            ax = self.h.axs[
                int(fig_num // self.h._Run__cols), int(fig_num % self.h._Run__cols)
            ]
            A = self._poly_from_geom(geom_a[fig_num])
            B = self._poly_from_geom(geom_b[fig_num])
            C = self._minkowski_sum_convex(A, -B)
            cmp = self.h.plt.get_cmap("viridis")
            color_C = cmp(0.15)
            C_.append(C)

            color_A = self._cm_color(fig_num, k=0, n=3)
            color_B = self._cm_color(fig_num, k=1, n=3)

            inputs_kwargs = dict(inputs_kwargs or {})
            if draw_inputs:
                ka = dict({"linewidth": 1.0, "alpha": 1.0})
                kb = dict({"linewidth": 1.0, "alpha": 1.0})
                ka.update(inputs_kwargs.get("A", {}) or {})
                kb.update(inputs_kwargs.get("B", {}) or {})
                for d in (ka, kb):
                    for key in ("color", "c", "facecolor", "edgecolor"):
                        d.pop(key, None)
                if color_A is not None:
                    ka["color"] = color_A
                if color_B is not None:
                    kb["color"] = color_B

                a_plot = A if not closed else np.vstack([A, A[:1]])
                b_plot = B if not closed else np.vstack([B, B[:1]])
                ax.plot(a_plot[:, 0], a_plot[:, 1], **ka)
                ax.plot(b_plot[:, 0], b_plot[:, 1], **kb)

            rk = {"linewidth": 2.0, "alpha": 0.2}
            if result_kwargs:
                rk.update(result_kwargs)
            for key in ("color", "c", "facecolor", "edgecolor"):
                rk.pop(key, None)

            Cplot = C if not closed else np.vstack([C, C[:1]])
            if fill:
                ax.fill(
                    Cplot[:, 0],
                    Cplot[:, 1],
                    facecolor=color_C,
                    edgecolor=color_C,
                    **rk,
                )
            else:
                ax.plot(
                    Cplot[:, 0],
                    Cplot[:, 1],
                    color=color_C,
                    **rk,
                )

            if equal_aspect:
                try:
                    ax.set_aspect("equal", adjustable="datalim")
                except Exception:
                    pass
            if margins is not None:
                ax.margins(margins, margins)

            if self.h._Run__figure_number > 1:
                try:
                    self.h._Run__figure_serial(
                        fig_num, use_tex=self.h.plt.rcParams["text.usetex"]
                    )
                except Exception:
                    pass

        return C_

    def plot_minkowski_sum2d(
        self,
        geom_a: List[Dict],
        geom_b: List[Dict],
        draw_inputs: bool = True,
        inputs_kwargs: Optional[dict] = None,
        result_kwargs: Optional[dict] = None,
        fill: bool = False,
        closed: bool = True,
        equal_aspect: bool = True,
        margins: Optional[float] = 0.05,
    ) -> List[np.ndarray]:
        """
        Plot the Minkowski sum A ⊕ B for two convex inputs (polygons or superellipses).
        Mirrors the style and API of `plot_minkowski_difference2d`.

        Returns
        -------
        C_ : list of (M_i, 2) ndarray
            One CCW vertex list per subplot (no duplicated first/last).
        """
        C_ = []

        for fig_num in range(self.h._Run__figure_number):
            ax = self.h.axs[
                int(fig_num // self.h._Run__cols), int(fig_num % self.h._Run__cols)
            ]

            # Normalize inputs to CCW vertex arrays
            A = self._poly_from_geom(geom_a[fig_num])
            B = self._poly_from_geom(geom_b[fig_num])

            # Core: Minkowski sum
            C = self._minkowski_sum_convex(A, B)
            C_.append(C)

            # Colors from global colormap (A, B, result)
            color_A = self._cm_color(fig_num, k=0, n=3)
            color_B = self._cm_color(fig_num, k=1, n=3)
            cmp = self.h.plt.get_cmap("viridis")
            color_C = cmp(0.15)

            # Optionally draw inputs
            if draw_inputs:
                ka = {"linewidth": 1.0, "alpha": 1.0}
                kb = {"linewidth": 1.0, "alpha": 1.0}
                inp = dict(inputs_kwargs or {})
                ka.update((inp.get("A") or {}))
                kb.update((inp.get("B") or {}))
                for d in (ka, kb):
                    for key in ("color", "c", "facecolor", "edgecolor"):
                        d.pop(key, None)
                if color_A is not None:
                    ka["color"] = color_A
                if color_B is not None:
                    kb["color"] = color_B

                a_plot = A if not closed else np.vstack([A, A[:1]])
                b_plot = B if not closed else np.vstack([B, B[:1]])
                ax.plot(a_plot[:, 0], a_plot[:, 1], **ka)
                ax.plot(b_plot[:, 0], b_plot[:, 1], **kb)

            # Result styling
            rk = {"linewidth": 2.0, "alpha": 0.2}
            if result_kwargs:
                rk.update(result_kwargs)
            for key in ("color", "c", "facecolor", "edgecolor"):
                rk.pop(key, None)

            Cplot = C if not closed else np.vstack([C, C[:1]])
            if fill:
                ax.fill(
                    Cplot[:, 0],
                    Cplot[:, 1],
                    facecolor=color_C,
                    edgecolor=color_C,
                    **rk,
                )
            else:
                ax.plot(
                    Cplot[:, 0],
                    Cplot[:, 1],
                    color=color_C,
                    **rk,
                )

            if equal_aspect:
                try:
                    ax.set_aspect("equal", adjustable="datalim")
                except Exception:
                    pass
            if margins is not None:
                ax.margins(margins, margins)

            if self.h._Run__figure_number > 1:
                try:
                    self.h._Run__figure_serial(
                        fig_num, use_tex=self.h.plt.rcParams["text.usetex"]
                    )
                except Exception:
                    pass

        return C_
