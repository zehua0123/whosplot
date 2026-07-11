import numpy as np


class SuperQuadrics:
    """
    Render 3D superquadrics (superellipsoid / supertoroid / hyperboloids / superparaboloid).
    The class keeps a reference to a *host* object (expected to be a `Run` instance)
    so it can access Matplotlib handles like `plt`, `fig`, `axs`, colormaps, layout
    parameters, and private helpers on the host when needed.
    """

    def __init__(self, host):
        """
        Parameters
        ----------
        host : object
            The external runner/host (e.g., an instance of `Run`) that owns the
            Matplotlib figure/axes grid and styling state. We do not copy state;
            we only *reference* it so drawing stays consistent with the host.
        """
        self.h = host  # external Run instance

    # ---- Mathematical building blocks ----
    def _spow(self, x, p):
        """
        Signed power: sign(x) * |x|**p. Useful for superquadric exponents where
        even/odd symmetries must be preserved without branching.
        """
        return np.sign(x) * (np.abs(x) ** p)

    def _superellipse(self, angle, exponent):
        """Return a well-distributed parameterization of a superellipse.

        ``sign(cos(t))*abs(cos(t))**exponent`` is the usual parameterization,
        but it becomes extremely non-uniform when ``exponent`` is smaller than
        one.  In that regime, the radial form below describes the same implicit
        superellipse while using the polar angle directly.  This keeps
        neighboring vertices at similar distances and avoids long, thin
        surface patches.  The usual form is retained for exponents greater
        than or equal to one so a finite grid still resolves their axis-aligned
        cusps accurately.
        """
        exponent = float(exponent)
        if not np.isfinite(exponent) or exponent <= 0.0:
            raise ValueError("Superquadric exponents must be finite and positive")

        angle = np.asarray(angle, dtype=float)
        cosine, sine = np.cos(angle), np.sin(angle)
        zero_tol = 8.0 * np.finfo(float).eps
        cosine = np.where(np.abs(cosine) <= zero_tol, 0.0, cosine)
        sine = np.where(np.abs(sine) <= zero_tol, 0.0, sine)

        if exponent >= 1.0:
            return self._spow(cosine, exponent), self._spow(sine, exponent)

        abs_cosine, abs_sine = np.abs(cosine), np.abs(sine)
        scale = np.maximum(abs_cosine, abs_sine)
        power = 2.0 / exponent

        # Factoring out ``scale`` avoids underflow for very small exponents.
        # cos(angle) and sin(angle) cannot both be zero, so scale is positive.
        if np.isinf(power):
            radial_scale = np.ones_like(scale)
        else:
            radial_scale = (
                (abs_cosine / scale) ** power
                + (abs_sine / scale) ** power
            ) ** (-1.0 / power)
        radius = radial_scale / scale
        return radius * cosine, radius * sine

    def _superellipsoid(self, p):
        """
        Parametric superellipsoid surface.

        Parameters
        ----------
        p : dict
            Expected keys:
            - a1, a2, a3 : float
                Scale along X, Y, Z axes.
            - e1, e2 : float
                Exponents controlling "squareness" in elevation (e1) and azimuth (e2).
            - nu, nv : int
                Resolution along elevation (u) and azimuth (v).

        Returns
        -------
        (X, Y, Z) : tuple of ndarray
            Meshgrids of shape (nu, nv) representing the surface.
        """
        a1, a2, a3 = float(p.get("a1")), float(p.get("a2")), float(p.get("a3"))
        e1, e2 = float(p.get("e1")), float(p.get("e2"))
        nu, nv = int(p.get("nu")), int(p.get("nv"))
        u = np.linspace(-0.5 * np.pi, 0.5 * np.pi, nu)
        v = np.linspace(-np.pi, np.pi, nv)
        U, V = np.meshgrid(u, v, indexing="ij")
        Cu, Su = self._superellipse(U, e1)
        Cv, Sv = self._superellipse(V, e2)
        X = a1 * Cu * Cv
        Y = a2 * Cu * Sv
        Z = a3 * Su
        return X, Y, Z

    def _supertoroid(self, p):
        """
        Parametric supertoroid surface (a superquadric torus).

        Parameters
        ----------
        p : dict
            Expected keys:
            - a_major, a_minor, a3 : float
                Major radius, minor radius, and vertical scaling.
            - e1, e2 : float
                Exponents controlling cross-section shapes (azimuth/elevation-like).
            - nu, nv : int
                Resolution along the two parameters.

        Returns
        -------
        (X, Y, Z) : tuple of ndarray
            Meshgrids of shape (nu, nv) representing the surface.
        """
        aM, am, a3 = (
            float(p.get("a_major")),
            float(p.get("a_minor")),
            float(p.get("a3")),
        )
        e1, e2 = float(p.get("e1")), float(p.get("e2"))
        nu, nv = int(p.get("nu")), int(p.get("nv"))
        u = np.linspace(-np.pi, np.pi, nu)
        v = np.linspace(-np.pi, np.pi, nv)
        U, V = np.meshgrid(u, v, indexing="ij")
        Cu, Su = self._superellipse(U, e2)
        Cv, Sv = self._superellipse(V, e1)
        R = aM + am * Cv
        X = R * Cu
        Y = R * Su
        Z = a3 * Sv
        return X, Y, Z

    def _sh1(self, p):
        """
        Hyperboloid of one sheet (superquadric form).

        Parameters
        ----------
        p : dict
            Expected keys:
            - a1, a2, a3 : float
                Scaling along X, Y, Z.
            - e1, e2 : float
                Exponents controlling radial and angular shaping.
            - u_extent : float
                Half-extent of the parameter u (the surface is sampled on [-u_extent, +u_extent]).
            - nu, nv : int
                Resolution in u, v.

        Returns
        -------
        (X, Y, Z) : tuple of ndarray
            Meshgrids of shape (nu, nv) for the surface.
        """
        a1, a2, a3 = float(p.get("a1")), float(p.get("a2")), float(p.get("a3"))
        e1, e2 = float(p.get("e1")), float(p.get("e2"))
        uext = float(p.get("u_extent"))
        nu, nv = int(p.get("nu")), int(p.get("nv"))
        u = np.linspace(-uext, uext, nu)
        v = np.linspace(-np.pi, np.pi, nv)
        U, V = np.meshgrid(u, v, indexing="ij")
        CH = (np.cosh(U)) ** e1
        SH = self._spow(np.sinh(U), e1)
        Cv, Sv = self._superellipse(V, e2)
        X = a1 * CH * Cv
        Y = a2 * CH * Sv
        Z = a3 * SH
        return X, Y, Z

    def _sh2(self, p):
        """
        Hyperboloid of two sheets (superquadric form).

        Parameters
        ----------
        p : dict
            Expected keys:
            - a1, a2, a3 : float
                Scaling along X, Y, Z.
            - e1, e2 : float
                Exponents controlling radial and angular shaping.
            - u_min, u_max : float
                Range of the parameter u (two sheets live in disjoint u intervals).
            - nu, nv : int
                Resolution in u, v.

        Returns
        -------
        ((X, Y, Z_plus), (X, Y, Z_minus)) : tuple of tuples of ndarray
            Two surfaces (the two sheets). Each entry has shape (nu, nv).
        """
        a1, a2, a3 = float(p.get("a1")), float(p.get("a2")), float(p.get("a3"))
        e1, e2 = float(p.get("e1")), float(p.get("e2"))
        nu, nv = int(p.get("nu")), int(p.get("nv"))
        u_min, u_max = float(p.get("u_min")), float(p.get("u_max"))
        u = np.linspace(u_min, u_max, nu)
        v = np.linspace(-np.pi, np.pi, nv)
        U, V = np.meshgrid(u, v, indexing="ij")
        SH = (np.sinh(U)) ** e1
        CH = (np.cosh(U)) ** e1
        Cv, Sv = self._superellipse(V, e2)
        X = a1 * SH * Cv
        Y = a2 * SH * Sv
        Zp = a3 * CH
        Zm = -a3 * CH
        return (X, Y, Zp), (X, Y, Zm)

    def _superparaboloid(self, p):
        """
        Superparaboloid surface.

        Parameters
        ----------
        p : dict
            Expected keys:
            - a1, a2, a3 : float
                Scaling along X, Y, Z.
            - e1, e2 : float
                Exponents controlling the vertical and angular shaping.
            - nu, nv : int
                Resolution in u (radial-like, 0..1) and v (angle, -pi..pi).

        Returns
        -------
        (X, Y, Z) : tuple of ndarray
            Meshgrids of shape (nu, nv) for the surface.
        """
        a1, a2, a3 = float(p.get("a1")), float(p.get("a2")), float(p.get("a3"))
        e1, e2 = float(p.get("e1")), float(p.get("e2"))
        nu, nv = int(p.get("nu")), int(p.get("nv"))
        u = np.linspace(0.0, 1.0, nu)
        v = np.linspace(-np.pi, np.pi, nv)
        U, V = np.meshgrid(u, v, indexing="ij")
        R = U
        Zshape = U ** (2.0 / e1)
        Cv, Sv = self._superellipse(V, e2)
        X = a1 * R * Cv
        Y = a2 * R * Sv
        Z = a3 * Zshape
        return X, Y, Z

    def _ax(self, i: int):
        """
        Convenience accessor for a subplot axis by linear index.

        Parameters
        ----------
        i : int
            Linear subplot index (row-major).

        Returns
        -------
        matplotlib.axes._subplots.Axes3DSubplot
            The 3D axis at index i.
        """
        return self.h.axs[int(i // self.h._Run__cols), int(i % self.h._Run__cols)]

    # ---- Public rendering API ----
    def draw_superquadrics(self, items):
        """
        Draw multiple superquadrics on a 3D grid of subplots.

        Parameters
        ----------
        items : list[dict or str]
            A list of shape specifications. Each item may be a string
            (an alias for `shape`) or a dict with parameters. Supported
            shapes and required keys:
              - "superellipsoid" / "se":
                    a1, a2, a3, e1, e2, nu, nv
              - "supertoroid" / "st" / "ring":
                    a_major, a_minor, a3, e1, e2, nu, nv
              - "hyperboloid_one_sheet" / "sh1":
                    a1, a2, a3, e1, e2, u_extent, nu, nv
              - "hyperboloid_two_sheets" / "sh2":
                    a1, a2, a3, e1, e2, u_min, u_max, nu, nv
              - "superparaboloid" / "sp":
                    a1, a2, a3, e1, e2, nu, nv

            Optional keys for any item:
              - view: tuple[float, float]
                    (elev, azim) camera angles in degrees.

        Notes
        -----
        - This method *recreates* the host's 3D subplot grid to ensure
          the figure is configured for 3D rendering.
        - Each surface uses the host-provided colormap `self.h._Run__color_map[i]`.
        - If the host tracks figure numbering, we call its private method
          `_Run__figure_serial` to preserve the original annotation behavior.

        Returns
        -------
        None
        """
        # Recreate a 3D subplot grid, keeping consistent with the original behavior
        self.h.fig, self.h.axs = self.h.plt.subplots(
            self.h._Run__rows,
            self.h._Run__cols,
            figsize=(
                self.h._Run__width * self.h._Run__cols,
                self.h._Run__height * self.h._Run__rows,
            ),
            subplot_kw={"projection": "3d"},
            squeeze=False,
        )

        # Normalize the items list into dictionaries with a lowercase "shape" key
        def _norm_item(it):
            if isinstance(it, str):
                return {"shape": it.strip().lower()}
            d = dict(it)
            d["shape"] = d.get("shape", d.get("kind", "")).strip().lower()
            return d

        items = [_norm_item(x) for x in items]

        shape_map = {
            "superellipsoid": self._superellipsoid,
            "se": self._superellipsoid,
            "supertoroid": self._supertoroid,
            "st": self._supertoroid,
            "ring": self._supertoroid,
            "hyperboloid_one_sheet": self._sh1,
            "sh1": self._sh1,
            "hyperboloid_two_sheets": self._sh2,
            "sh2": self._sh2,
            "superparaboloid": self._superparaboloid,
            "sp": self._superparaboloid,
        }

        for i, spec in enumerate(items):
            shape = spec.get("shape", "superellipsoid")
            builder = shape_map.get(shape)
            if builder is None:
                raise ValueError(f"Unknown superquadric shape: {shape}")

            ax = self._ax(i)
            ax.set_axis_off()

            # Optional camera view (elevation, azimuth)
            view = spec.get("view")
            if view is not None:
                try:
                    elev, azim = view
                    ax.view_init(elev=elev, azim=azim)
                except Exception:
                    # Ignore invalid view specification without interrupting the drawing
                    pass

            res = builder(spec)
            if shape in ("hyperboloid_two_sheets", "sh2"):
                (X, Y, Zp), (X2, Y2, Zm) = res
                ax.plot_surface(
                    X,
                    Y,
                    Zp,
                    rstride=1,
                    cstride=1,
                    linewidth=0,
                    edgecolor="none",
                    antialiased=False,
                    shade=True,
                    cmap=self.h._Run__color_map[i],
                )
                ax.plot_surface(
                    X2,
                    Y2,
                    Zm,
                    rstride=1,
                    cstride=1,
                    linewidth=0,
                    edgecolor="none",
                    antialiased=False,
                    shade=True,
                    cmap=self.h._Run__color_map[i],
                )
                try:
                    ax.set_box_aspect(
                        (np.ptp(X), np.ptp(Y), max(np.ptp(Zp), np.ptp(Zm)))
                    )
                except Exception:
                    pass
            else:
                X, Y, Z = res
                ax.plot_surface(
                    X,
                    Y,
                    Z,
                    rstride=1,
                    cstride=1,
                    linewidth=0,
                    edgecolor="none",
                    antialiased=False,
                    shade=True,
                    cmap=self.h._Run__color_map[i],
                )
                try:
                    ax.set_box_aspect((np.ptp(X), np.ptp(Y), np.ptp(Z)))
                except Exception:
                    pass

            # Preserve the host's figure numbering/serial annotation (if enabled)
            if self.h._Run__figure_number > 1:
                try:
                    self.h._Run__figure_serial(
                        i, use_tex=self.h.plt.rcParams["text.usetex"]
                    )
                except Exception:
                    pass
