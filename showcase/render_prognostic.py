"""The prognostic-solver video: the control intrusion carried by the jet for a week.

    python showcase/render_prognostic.py DATA_DIR OUT.mp4 [--stills DIR] [--only SCENE] [--steps N]

DATA_DIR/prognostic holds meta.npz and frame_####.npz from capture_prognostic.py.
"""
import argparse
import glob
import os
import re
import numpy as np
import pyvista as pv
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from scipy.ndimage import gaussian_filter1d
import videokit as vk
from videokit import ACCENT, ACCENT2, INK, MUTED, PANEL, EDGE
import thesis_figures

ZS = 120             # vertical exaggeration of the 3D view
X_MAX = 20000.0      # km of the domain shown on the map, as in the thesis figure
WINDOW = 3200.0      # half-width in x of the 3D view [km]
TROPOPAUSE = -1.5    # PVU: the dynamical tropopause (negative: Southern Hemisphere)
DEPTH = 1.0          # km below the undisturbed tropopause that counts as intruded
WIND_MAX = 9.0       # m/s, top of the surface wind colour scale (as in the thesis)


class Run:
    """The captured frames, interpolated in time so the video can play them at any rate."""
    def __init__(self, data_dir):
        folder = os.path.join(data_dir, "prognostic")
        self.meta = dict(np.load(os.path.join(folder, "meta.npz")))
        self.paths = sorted(glob.glob(os.path.join(folder, "frame_*.npz")))
        self._cache = {}
        series = [np.load(p) for p in self.paths]
        self.hours = np.array([float(s["t"]) / 3600 for s in series])
        self.max_wind = np.array([float(s["max_wind"]) for s in series])
        self.z_dyn = np.array([float(s["z_dyn"]) for s in series]) / 1e3
        self.x, self.y, self.z = (self.meta[k] / 1e3 for k in ("x3", "y3", "z3"))
        self.sx, self.sy = self.meta["surf_x"] / 1e3, self.meta["surf_y"] / 1e3
        # The background PV depends on y and z only, and is a steady solution, so the far
        # field of the first frame is the background for the whole run
        far = (self.x > 0.5 * self.x[-1]) & (self.x < 0.8 * self.x[-1])
        self.background = self.frame(0)["epv"][far].mean(axis=0)
        strat = self.background <= TROPOPAUSE
        self.background_tropopause = self.z[np.argmax(strat, axis=1)]
        self.track = self._track()

    def frame(self, i):
        if i not in self._cache:
            if len(self._cache) > 6:
                self._cache.pop(next(iter(self._cache)))
            with np.load(self.paths[i]) as f:
                self._cache[i] = {k: f[k] for k in ("epv", "u_surf", "v_surf")}
        return self._cache[i]

    def at(self, hour):
        """Fields at `hour`, linearly interpolated between the two nearest frames."""
        hour = float(np.clip(hour, self.hours[0], self.hours[-1]))
        i = min(int(np.searchsorted(self.hours, hour, side="right") - 1), len(self.hours) - 2)
        w = (hour - self.hours[i]) / (self.hours[i + 1] - self.hours[i])
        a, b = self.frame(i), self.frame(i + 1)
        return {k: (1 - w) * a[k] + w * b[k] for k in a}

    def anomaly(self, epv):
        return epv - self.background[None]

    def intruded(self, epv, y, z):
        """Negative exactly where the air is stratospheric (beyond the -1.5 PVU tropopause)
        although it sits at least DEPTH below the undisturbed tropopause: the air that has
        intruded into the troposphere. epv is on the grid with axes (x, y, z) [km].

        The undisturbed tropopause is the lowest level at which the background crosses -1.5
        PVU. Taking the first crossing from below steps over the small pockets of positive
        (non-physical) background PV inside the jet. A PV anomaly threshold instead would
        catch the small relative changes high in the stratosphere, where PV is large."""
        trop = np.interp(y, self.y, self.background_tropopause)
        below = z[None, None, :] - (trop[None, :, None] - DEPTH)
        return np.maximum(epv - TROPOPAUSE, below)

    def _track(self):
        """Where the intrusion is: the most negative column-integrated PV anomaly, searched
        near where it last was so the downstream wave train can't steal the track."""
        xa, ya = self.meta["anomaly_x"] / 1e3, self.meta["anomaly_y"] / 1e3
        points = []
        for i in range(len(self.paths)):
            column = np.minimum(self.anomaly(self.frame(i)["epv"]), 0).sum(axis=2)
            near = (np.abs(self.x - xa)[:, None] < 800) & (np.abs(self.y - ya)[None, :] < 800)
            ix, iy = np.unravel_index(np.argmin(np.where(near, column, np.inf)), column.shape)
            xa, ya = self.x[ix], self.y[iy]
            points.append((xa, ya))
        return gaussian_filter1d(np.array(points, dtype=float), sigma=2, axis=0, mode="nearest")

    def centre(self, hour):
        return np.array([np.interp(hour, self.hours, self.track[:, j]) for j in range(2)])


class Stage:
    """The 3D view: the intrusion and the surface winds beneath it."""
    def __init__(self, run, size):
        self.run = run
        self.p = vk.plotter(size)
        self.p.enable_depth_peeling(number_of_peels=8)

    def shot(self, hour, azimuth, elevation, distance, pan=0.0, centre=None, focus_z=6.0):
        run, p = self.run, self.p
        cx, cy = run.centre(hour) if centre is None else centre
        fields = run.at(hour)
        p.clear_actors()

        ix = np.abs(run.x - cx) <= WINDOW + 100
        axes = (run.meta["x3"][ix], run.meta["y3"], run.meta["z3"])
        fine, epv = vk.refine(axes, fields["epv"][ix], (2, 2, 2), order=1)
        g = vk.grid(*fine, ZS, intruded=run.intruded(epv, fine[1] / 1e3, fine[2] / 1e3))
        blob = g.contour([0.0], scalars="intruded")
        if blob.n_points:
            blob = blob.smooth_taubin(n_iter=20, pass_band=0.1)
            blob["height"] = blob.points[:, 2] / ZS
            p.add_mesh(blob, scalars="height", cmap="magma", clim=(6, 13), smooth_shading=True,
                       specular=0.5, specular_power=25, show_scalar_bar=False, culling="front")

        sx = np.abs(run.sx - cx) <= WINDOW
        X, Y = np.meshgrid(run.sx[sx], run.sy, indexing="ij")
        u, v = fields["u_surf"][sx], fields["v_surf"][sx]
        ground = pv.StructuredGrid(X, Y, np.zeros_like(X))
        ground["speed"] = np.hypot(u, v).ravel(order="F")
        p.add_mesh(ground, scalars="speed", cmap="viridis", clim=(0, WIND_MAX), show_scalar_bar=False)
        s = 4
        cloud = pv.PolyData(np.c_[X[::s, ::s].ravel(), Y[::s, ::s].ravel(),
                                  np.full(X[::s, ::s].size, 8.0)])
        cloud["vec"] = np.c_[u[::s, ::s].ravel(), v[::s, ::s].ravel(), np.zeros(cloud.n_points)]
        cloud["mag"] = np.hypot(u[::s, ::s], v[::s, ::s]).ravel()
        p.add_mesh(cloud.glyph(orient="vec", scale="mag", factor=55), color="white", opacity=0.85)

        vk.orbit(p, (cx, cy, focus_z * ZS), distance, azimuth, elevation, pan=pan)
        p.render()  # screenshot alone reuses the last frame after a camera move
        return p.screenshot(return_img=True)


def scenes(run, figs, steps=None):
    stage = Stage(run, (1920, 680))
    title_stage = Stage(run, (1920, 1080))
    keep = run.sx <= X_MAX + 1
    T = run.hours[-1]
    meta = run.meta
    unknowns = (float(meta["dg_dofs"]) + float(meta["cg_dofs"])) / 1e6

    def minimap(fig, rect, hour, window=True):
        fields = run.at(hour)
        u, v = fields["u_surf"][keep], fields["v_surf"][keep]
        sx = run.sx[keep]
        ax = fig.add_axes(rect)
        ax.imshow(np.hypot(u, v).T, origin="lower", extent=(sx[0], sx[-1], run.sy[0], run.sy[-1]),
                  cmap="viridis", vmin=0, vmax=WIND_MAX, aspect="auto", interpolation="bilinear")
        qx, qy = slice(5, None, 11), slice(6, None, 11)
        Xq, Yq = np.meshgrid(sx[qx], run.sy[qy], indexing="ij")
        ax.quiver(Xq, Yq, u[qx, qy], v[qx, qy], color="white", alpha=0.7, angles="xy",
                  scale_units="xy", scale=1 / 110, width=0.0016, headwidth=4)
        past = run.hours <= hour
        ax.plot(run.track[past, 0], run.track[past, 1], color=ACCENT2, lw=2, ls="--", alpha=0.95)
        cx, cy = run.centre(hour)
        ax.plot([cx], [cy], "o", color=ACCENT2, ms=8, mec="white", mew=1.5)
        if window:
            ax.axvspan(cx - WINDOW, cx + WINDOW, color="white", alpha=0.06)
        ax.set_xlim(0, X_MAX)
        ax.set_ylim(run.sy[0], run.sy[-1])
        ax.set_xlabel("x  (west → east)  [km]")
        ax.set_ylabel("y [km]")
        ax.tick_params(labelsize=11)
        for spine in ax.spines.values():
            spine.set_color(EDGE)
        return ax

    def sparkline(fig, rect, values, hour, label, unit, colour, fmt):
        ax = fig.add_axes(rect)
        shown = run.hours <= hour
        ax.plot(run.hours, values, color=EDGE, lw=1.5)
        ax.plot(run.hours[shown], values[shown], color=colour, lw=2.4)
        now = np.interp(hour, run.hours, values)
        ax.plot([hour], [now], "o", color=colour, ms=6)
        ax.set_xlim(0, T)
        lo, hi = values.min(), values.max()
        span = max(hi - lo, 1e-6)
        ax.set_ylim(lo - 0.1 * span, hi + 0.1 * span)
        ax.set_xticks(np.arange(0, T + 1, 48))
        ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.tick_params(labelsize=11)
        for spine in ax.spines.values():
            spine.set_color(EDGE)
        fig.text(rect[0], rect[1] + rect[3] + 0.012, label, fontsize=14, color=MUTED)
        fig.text(rect[0] + rect[2], rect[1] + rect[3] + 0.012, fmt.format(now) + " " + unit,
                 fontsize=16, color=colour, ha="right", fontweight="semibold")

    def title(fig, t):
        img = title_stage.shot(0.0, -95 + 5 * t, 24 + t, 3900, pan=900)
        vk.image(fig, img, [0, 0, 1, 1])
        vk.left_shade(fig, power=1.0)
        vk.title_block(fig, t, "Letting the\natmosphere move",
                       "A discontinuous Galerkin solver that carries a\n"
                       "stratospheric intrusion through a week of weather")

    def loop(fig, t):
        vk.section_heading(fig, "How it works", "PV moves with the winds it creates", t)
        vk.text(fig, 0.05, 0.69, r"$\dfrac{\partial q}{\partial t} \;-\; \dfrac{\partial \psi}{\partial y}\dfrac{\partial q}{\partial x}"
                                 r" \;+\; \dfrac{\partial \psi}{\partial x}\dfrac{\partial q}{\partial y} \;=\; 0$",
                t, 0.3, fontsize=34)
        nodes = [((0.25, 0.47), "PV  q", "the state we step forward"),
                 ((0.62, 0.47), "invert  →  ψ", "3D elliptic solve (previous video)"),
                 ((0.62, 0.17), "winds  u = −ψ$_y$,  v = ψ$_x$", "geostrophic balance"),
                 ((0.25, 0.17), "advect q", "upwind DG + limiter + SSPRK3")]
        active = int((t - 1.2) / 0.7) % 4 if t > 1.2 else -1
        for i, ((x, y), head, sub) in enumerate(nodes):
            a = vk.ramp(t, 0.6 + 0.25 * i)
            if a <= 0:
                continue
            colour = ACCENT if i == active else EDGE
            fig.text(x, y + 0.02, head, ha="center", va="center", fontsize=24, fontweight="semibold",
                     alpha=a, bbox=dict(boxstyle="round,pad=0.9,rounding_size=0.6", fc=PANEL,
                                        ec=colour, lw=2.2, alpha=a))
            fig.text(x, y - 0.065, sub, ha="center", fontsize=15, color=MUTED, alpha=a)
        a = vk.ramp(t, 1.6)
        for i, ((x0, y0), (x1, y1)) in enumerate([((0.32, 0.49), (0.52, 0.49)), ((0.62, 0.40), (0.62, 0.26)),
                                                   ((0.49, 0.19), (0.33, 0.19)), ((0.25, 0.26), (0.25, 0.40))]):
            fig.patches.append(FancyArrowPatch((x0, y0), (x1, y1), transform=fig.transFigure,
                                               arrowstyle="-|>,head_length=10,head_width=6", lw=2.4,
                                               color=ACCENT if i == active else MUTED, alpha=a))
        if steps:
            vk.text(fig, 0.79, 0.37, f"3 stages per step\n× {steps:,} steps\n= {3 * steps:,} inversions",
                    t, 2.4, fontsize=21, color=ACCENT2, linespacing=1.5, va="center",
                    fontweight="semibold")

    evo_len = 27.0
    lead, tail = 1.0, 1.5

    def evolve(fig, t, duration=evo_len):
        hour = T * vk.clamp01((t - lead) / (duration - lead - tail))
        u = hour / T
        img = stage.shot(hour, -100 + 16 * u, 24 - 2 * u, 3400, focus_z=7.5)
        vk.image(fig, img, [0, 0.355, 1, 680 / 1080])
        vk.text(fig, 0.035, 0.935, f"t = {hour:5.1f} h", None, fontsize=40, fontweight="bold",
                family="DejaVu Sans Mono", va="top")
        vk.text(fig, 0.035, 0.865, "intrusion: stratospheric air below the normal tropopause, coloured by height\n"
                                   "ground: surface wind (0–9 m/s)", None, fontsize=14,
                color=MUTED, va="top", linespacing=1.4)
        minimap(fig, [0.05, 0.075, 0.62, 0.25], hour)
        fig.text(0.05, 0.335, "The whole 20,000 km domain, with the intrusion's track", fontsize=15,
                 color=MUTED)
        sparkline(fig, [0.73, 0.215, 0.23, 0.085], run.max_wind, hour, "max surface wind", "m/s",
                  ACCENT, "{:.1f}")
        sparkline(fig, [0.73, 0.075, 0.23, 0.085], run.z_dyn, hour, "lowest tropopause", "km",
                  ACCENT2, "{:.1f}")

    def insight(fig, t):
        vk.section_heading(fig, "What the solver shows", "East with the jet, and toward the pole", t)
        minimap(fig, [0.05, 0.47, 0.58, 0.29], T, window=False)
        lines = [
            ("The jet carries the intrusion east, and its\nPV gradient acts like the planetary β-effect,\n"
             "pulling the cyclone poleward (south).", 0.5),
            ("A wave train of new circulations spins up\ndownstream as the week goes on.", 1.6),
        ]
        for i, (s, t0) in enumerate(lines):
            vk.text(fig, 0.66, 0.75 - i * 0.17, s, t, t0, fontsize=20, linespacing=1.4, va="top")
        img = figs["evolution_last"]
        a = vk.ramp(t, 2.4)
        if a > 0:
            h, w = img.shape[:2]
            height = 0.3
            width = height * w / h * vk.H / vk.W
            vk.card(fig, [0.05, 0.06, width + 0.012, height + 0.02], color="white", edge="white", alpha=a)
            vk.image(fig, img, [0.056, 0.07, width, height], alpha=a)
            x = 0.08 + width
            vk.text(fig, x, 0.33, "The same run in the thesis, at t = 168 h:", t, 2.6, fontsize=17,
                    va="top")
            vk.text(fig, x, 0.285, "5×10$^7$ unknowns, 36 hours on a 40-core HPC node", t, 2.7,
                    fontsize=16, color=MUTED, va="top")
            vk.text(fig, x, 0.2, "This animation:", t, 2.9, fontsize=17, va="top")
            vk.text(fig, x, 0.155, f"{unknowns:.1f}×10$^6$ unknowns (p = {int(meta['p'])}), 4 CPU cores",
                    t, 3.0, fontsize=16, color=MUTED, va="top")

    def numerics(fig, t):
        vk.section_heading(fig, "Under the hood", "A verified, bound-preserving scheme", t)
        items = [("Upwind discontinuous Galerkin", "conserves PV cell by cell"),
                 ("Zhang–Shu limiter", "keeps PV within its initial bounds"),
                 ("SSPRK3 time stepping", "strong-stability-preserving, 3rd order"),
                 ("Verified by manufactured solutions", "5th order in space, 3rd order in time"),
                 ("Stable to Courant number 2.7", "run at 1.3 for safety"),
                 ("Every stage re-inverts the PV", "matrix-free p-multigrid each time")]
        for i, (head, sub) in enumerate(items):
            col, row = i % 2, i // 2
            x, y = 0.05 + col * 0.46, 0.62 - row * 0.18
            a = vk.ramp(t, 0.4 + 0.25 * i)
            if a <= 0:
                continue
            vk.card(fig, [x, y - 0.05, 0.43, 0.14], alpha=a)
            fig.text(x + 0.02, y + 0.035, head, fontsize=22, fontweight="semibold", alpha=a)
            fig.text(x + 0.02, y - 0.015, sub, fontsize=17, color=MUTED, alpha=a)

    def end(fig, t):
        vk.end_card(fig, t)

    return [vk.Scene(5.0, title, fade_in=0.6), vk.Scene(7.0, loop), vk.Scene(evo_len, evolve),
            vk.Scene(8.0, insight), vk.Scene(6.5, numerics), vk.Scene(5.5, end, fade_out=0.8)]


def last_panel(img):
    """The bottom (t = 168 h) panel of the thesis' stacked evolution figure, without its
    colour bar."""
    ink = (np.abs(img.astype(int) - 255).sum(-1) > 30)
    rows = np.where(ink[:, : int(img.shape[1] * 0.75)].any(1))[0]
    gaps = np.where(np.diff(rows) > 8)[0]
    start = rows[gaps[-2] + 1] if len(gaps) >= 2 else rows[0]
    panel = img[start:, : int(img.shape[1] * 0.78)]
    return thesis_figures.trim(panel)


def steps_from_log(data_dir):
    path = os.path.join(data_dir, "prognostic.log")
    if os.path.exists(path):
        found = re.findall(r"done: (\d+) steps", open(path).read())
        if found:
            return int(found[-1])
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("out")
    parser.add_argument("--stills", help="Directory to save README stills into")
    parser.add_argument("--only", type=int, help="Render one scene only (preview)")
    parser.add_argument("--steps", type=int, help="Time steps the run took (read from prognostic.log if left out)")
    parser.add_argument("--pdf", default=os.path.join(os.path.dirname(__file__), "..", "docs", "thesis.pdf"))
    args = parser.parse_args()

    vk.setup_style()
    run = Run(args.data_dir)
    figs = thesis_figures.load(args.pdf, ["evolution"])
    figs["evolution_last"] = last_panel(figs["evolution"])
    stills = {}
    if args.stills:
        os.makedirs(args.stills, exist_ok=True)
        stills = {os.path.join(args.stills, "prognostic_title.png"): (0, 4.0),
                  os.path.join(args.stills, "prognostic_evolution.png"): (2, 20.0)}
    steps = args.steps or steps_from_log(args.data_dir)
    vk.render(scenes(run, figs, steps), args.out, stills=stills, only=args.only)


if __name__ == "__main__":
    main()
