# AI-generated (Claude Code) for the README and videos only. Not part of the thesis code:
# the project's own solvers and experiments are the files in the repository root.
"""The diagnostic-solver video: from potential vorticity to the full atmosphere, with the
p-multigrid conjugate gradient solver racing a textbook one on the way.

    python showcase/render_diagnostic.py DATA_DIR OUT.mp4 [--stills DIR] [--only SCENE]

DATA_DIR holds diagnostic.npz from capture_diagnostic.py; figures from the thesis PDF are read
from docs/thesis.pdf.
"""
import argparse
import os
import numpy as np
import pyvista as pv
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
import videokit as vk
from videokit import ACCENT, ACCENT2, GOOD, INK, MUTED, PANEL, EDGE
import thesis_figures

ZS = 120          # vertical exaggeration of the 3D views
HALF = 1500       # half-width of the 3D views [km]


def load(data_dir):
    d = dict(np.load(os.path.join(data_dir, "diagnostic.npz")))
    d["xc"] = (d["x3"][0] + d["x3"][-1]) / 2e3
    d["yc"] = (d["y3"][0] + d["y3"][-1]) / 2e3
    return d


# --- 3D stages ----------------------------------------------------------------------------

class Stage:
    """A PyVista scene, built once and re-shot from a moving camera."""
    def __init__(self, d, size, layers):
        self.d, self.size = d, size
        self.p = vk.plotter(size)
        self.p.enable_depth_peeling(number_of_peels=10)
        for mesh, kw in layers:
            self.p.add_mesh(mesh, **kw)

    def shot(self, azimuth, elevation, distance, focus_z, pan=0.0):
        vk.orbit(self.p, (self.d["xc"], self.d["yc"], focus_z * ZS), distance, azimuth, elevation,
                 pan=pan)
        self.p.render()  # screenshot alone reuses the last frame after a camera move
        return self.p.screenshot(return_img=True)


def meshes(d):
    xc, yc = d["xc"], d["yc"]
    # The 3D window, upsampled: the intrusion is only a few samples across
    ix = np.abs(d["x3"] / 1e3 - xc) <= HALF + 70
    iy = np.abs(d["y3"] / 1e3 - yc) <= HALF + 70
    iz = d["z3"] <= 20.5e3
    window = np.ix_(ix, iy, iz)
    axes = (d["x3"][ix], d["y3"][iy], d["z3"][iz])
    factors = (3, 3, 2)
    fine_axes, epv = vk.refine(axes, d["epv_volume"][window], factors)
    _, low = vk.refine(axes, d["induced_p_volume"][window], factors)
    box = vk.grid(*fine_axes, ZS, epv=epv, low=low)
    trop = box.contour([-1.5], scalars="epv").smooth_taubin(n_iter=30, pass_band=0.1)
    trop["height"] = trop.points[:, 2] / ZS

    ground = pv.Plane(center=(xc, yc, 0), direction=(0, 0, 1), i_size=2 * HALF, j_size=2 * HALF,
                      i_resolution=12, j_resolution=12)

    sx, sy = d["surf_x"] / 1e3, d["surf_y"] / 1e3
    mx = (sx >= xc - HALF) & (sx <= xc + HALF)
    my = (sy >= yc - HALF) & (sy <= yc + HALF)
    X, Y = np.meshgrid(sx[mx], sy[my], indexing="ij")
    surface = pv.StructuredGrid(X, Y, np.zeros_like(X))
    surface["p"] = d["final_p_surf"][np.ix_(mx, my)].ravel(order="F")
    s = 3
    U, V = d["final_u_surf"][np.ix_(mx, my)][::s, ::s], d["final_v_surf"][np.ix_(mx, my)][::s, ::s]
    cloud = pv.PolyData(np.c_[X[::s, ::s].ravel(), Y[::s, ::s].ravel(), np.full(U.size, 5.0)])
    cloud["vec"] = np.c_[U.ravel(), V.ravel(), np.zeros(U.size)]
    cloud["mag"] = np.hypot(U.ravel(), V.ravel())
    arrows = cloud.glyph(orient="vec", scale="mag", factor=60)

    lows = [box.contour([level], scalars="low").smooth_taubin(n_iter=30, pass_band=0.1)
            for level in LOW_LEVELS]
    return dict(trop=trop, ground=ground, surface=surface, arrows=arrows, lows=lows)


LOW_LEVELS = (-1.0, -2.5, -4.0)
LOW_STYLE = (("#5ec8ff", 0.16), ("#3b8fe8", 0.32), ("#2a56c6", 0.97))
TROP_STYLE = dict(scalars="height", cmap="magma", clim=(6.5, 13), smooth_shading=True,
                  specular=0.5, specular_power=25, show_scalar_bar=False)


def input_stage(d, m, size):
    return Stage(d, size, [
        (m["ground"], dict(color="#141c31")),
        (m["ground"].extract_all_edges(), dict(color="#26314d")),
        (m["trop"], TROP_STYLE),
    ])


def output_stage(d, m, size, tropopause=False):
    layers = [(m["surface"], dict(scalars="p", cmap="Blues_r", clim=(-4.6, 0), show_scalar_bar=False)),
              (m["arrows"], dict(color="white"))]
    if tropopause:
        layers.append((m["trop"], dict(TROP_STYLE, opacity=0.3)))
    for mesh, (colour, opacity) in zip(m["lows"], LOW_STYLE):
        style = dict(color=colour, opacity=opacity, smooth_shading=True, specular=0.3)
        if opacity > 0.9:
            # The contour's inner faces z-fight its outer ones at the silhouette; the winding
            # makes the inner ones front-facing, so cull those
            style["culling"] = "front"
        layers.append((mesh, style))
    return Stage(d, size, layers)


# --- 2D helpers -------------------------------------------------------------------------

def section_axes(fig, rect, title, colour=INK):
    ax = fig.add_axes(rect)
    ax.set_title(title, loc="left", color=colour, fontsize=17, pad=10)
    ax.set_xlabel("y  (south → north)  [km]")
    ax.set_ylabel("height [km]")
    for spine in ax.spines.values():
        spine.set_color(EDGE)
    return ax


def show_section(ax, d, field, norm, cmap):
    y = d["xs"] / 1e3 - d["xc"] + d["yc"]
    z = d["zs"] / 1e3
    top = z <= 20
    return ax.imshow(field[:, top].T, origin="lower", aspect="auto", cmap=cmap, norm=norm,
                     extent=(y[0], y[-1], z[0], z[top][-1]), interpolation="bilinear")


def colorbar(fig, mappable, rect, label, ticks=None):
    cax = fig.add_axes(rect)
    cb = fig.colorbar(mappable, cax=cax, orientation="horizontal", ticks=ticks)
    cb.set_label(label, color=MUTED, fontsize=13)
    cb.outline.set_edgecolor(EDGE)
    cax.tick_params(labelsize=11)
    return cb


def fmt_sci(x):
    if x == 0:
        return "0"
    e = int(np.floor(np.log10(abs(x))))
    m = x / 10 ** e
    return f"{m:.1f}×10$^{{{e}}}$" if abs(m - 1) > 0.05 else f"10$^{{{e}}}$"


# --- scenes -----------------------------------------------------------------------------

def scenes(d, figs):
    m = meshes(d)
    hero = output_stage(d, m, (vk.W, vk.H), tropopause=True)
    inp = input_stage(d, m, (1180, 1000))
    out = output_stage(d, m, (1180, 1000))
    dofs_m = d["dofs"] / 1e6
    ranks = int(d.get("ranks", 4))

    def title(fig, t):
        img = hero.shot(-75 + 6 * t, 12 + 1.5 * t, 4300, 5.5, pan=1100)
        vk.image(fig, img, [0, 0, 1, 1])
        vk.left_shade(fig, power=1.0)
        vk.title_block(fig, t, "Inverting the\natmosphere",
                       "A fifth-order finite element solver that turns\n"
                       "potential vorticity into wind, pressure and temperature")

    def the_input(fig, t, duration=8.5):
        u = t / duration
        img = inp.shot(-70 + 45 * vk.ease(u), -2 + 16 * vk.ease(u), 3900 + 200 * u, 10.5)
        vk.image(fig, img, [0.0, 0.03, 0.6, 0.94])
        x = 0.62
        vk.section_heading(fig, "The input", "A stratospheric intrusion", t, x=x, y=0.80)
        lines = [
            ("Potential vorticity (PV) is the only field\nthe solver is given.", 0.5),
            (f"High-PV stratospheric air plunges from\n12.5 km down to {d['z_dyn'] / 1e3:.1f} km.", 1.6),
            ("Intrusions like this feed cut-off lows:\nabout 20% of South Africa's flash floods.", 2.7),
            ("How much of it reaches the ground?", 3.8),
        ]
        for i, (s, t0) in enumerate(lines):
            vk.text(fig, x, 0.64 - i * 0.12, s, t, t0, fontsize=21, linespacing=1.35,
                    color=INK if i < 3 else ACCENT, fontweight="medium" if i == 3 else "normal")
        if t > 1.0:
            colorbar(fig, ScalarMappable(norm=Normalize(6.5, 13), cmap="magma"),
                     [x, 0.13, 0.28, 0.022], "tropopause height (−1.5 PVU surface) [km]")

    pmg_t, pmg_e = d["pmg_times"], d["pmg_errors"]
    has_jacobi = "jacobi_times" in d
    jac_t = None
    if has_jacobi:
        jac_its, jac_e = d["jacobi_kept_its"], d["jacobi_errors"]
        jac_t = d["jacobi_times"][jac_its]
    T_race = pmg_t[-1] * 1.04
    race_start, race_len = 1.6, 11.5
    p_norm = TwoSlopeNorm(0, -6, 6)

    def race(fig, t, duration=16.0):
        clock = T_race * vk.clamp01((t - race_start) / race_len)
        vk.section_heading(fig, "The solver at work", f"{dofs_m:.1f} million unknowns, {ranks} CPU cores", t)
        cols = [("This work: matrix-free CG + p-multigrid", ACCENT, "pmg", pmg_t),
                ("Textbook: CG + Jacobi preconditioner", ACCENT2, "jacobi", jac_t)]
        for c, (label, colour, key, times) in enumerate(cols):
            if times is None:
                continue
            x0 = 0.05 + c * 0.47
            k = int(np.searchsorted(times, clock, side="right") - 1)
            ax = section_axes(fig, [x0, 0.47, 0.42, 0.32], label, colour)
            im = show_section(ax, d, d[f"{key}_p_yz"][k, 0], p_norm, "RdBu_r")
            its = k if key == "pmg" else int(jac_its[k])
            err = pmg_e[k] if key == "pmg" else jac_e[k]
            ax.text(0.985, 0.94, f"iteration {its}", transform=ax.transAxes, ha="right", va="top",
                    fontsize=15, color=INK, fontweight="semibold",
                    bbox=dict(boxstyle="round,pad=0.3", fc=PANEL, ec=colour, lw=1.2, alpha=0.9))
            done = key == "pmg" and clock >= times[-1]
            status = "converged ✓" if done else f"error {fmt_sci(err)}" if its else "starting from zero"
            ax.text(0.015, 0.94, status, transform=ax.transAxes, ha="left", va="top", fontsize=15,
                    color=GOOD if done else INK,
                    bbox=dict(boxstyle="round,pad=0.3", fc=PANEL, ec=EDGE, lw=1.0, alpha=0.9))
        colorbar(fig, im, [0.05, 0.375, 0.25, 0.016], "pressure anomaly p* [hPa], north–south section")

        ax = fig.add_axes([0.42, 0.08, 0.53, 0.24])
        ax.set_yscale("log")
        ax.set_xlim(0, T_race)
        ax.set_ylim(1e-9, 2)
        ax.set_xlabel("solver time [s]")
        ax.set_ylabel("relative error")
        ax.grid(True, which="major", color=EDGE, lw=0.8)
        for spine in ax.spines.values():
            spine.set_color(EDGE)
        # Errors are measured against the converged p-multigrid solution, so its own last
        # iterate (the reference itself) is marked as converged rather than plotted as zero
        for times, errors, colour in [(pmg_t[:-1], pmg_e[:-1], ACCENT)] + (
                [(jac_t, jac_e, ACCENT2)] if has_jacobi else []):
            shown = times <= clock
            ax.plot(times[shown], errors[shown], color=colour, lw=2.6, marker="o", ms=5, zorder=3)
        if clock >= pmg_t[-1]:
            ax.axvline(pmg_t[-1], color=GOOD, lw=2)
            ax.text(pmg_t[-1], 0.05, "converged ", color=GOOD, ha="right", fontsize=13,
                    fontweight="semibold")
        ax.axvline(clock, color=MUTED, lw=1, ls=":")
        vk.text(fig, 0.05, 0.295, f"{clock:4.1f} s", None, fontsize=46, fontweight="bold",
                family="DejaVu Sans Mono", va="top")
        vk.text(fig, 0.05, 0.215, "solver time on the same cores", None, fontsize=15, color=MUTED, va="top")
        if has_jacobi and t > race_start + race_len + 0.3:
            k = int(np.searchsorted(jac_t, T_race, side="right") - 1)
            total_its = len(d["jacobi_residuals"]) - 1
            vk.text(fig, 0.05, 0.145, f"p-multigrid: {len(pmg_t) - 1} iterations to 10$^{{-9}}$",
                    t, race_start + race_len + 0.3, fontsize=17, color=ACCENT, fontweight="semibold")
            vk.text(fig, 0.05, 0.105, f"Jacobi: {int(jac_its[k])} iterations, still {jac_e[k] * 100:.0f}% off",
                    t, race_start + race_len + 0.7, fontsize=17, color=ACCENT2, fontweight="semibold")
            slower = d["jacobi_times"][-1] / pmg_t[-1]
            vk.text(fig, 0.05, 0.065, f"It needs {total_its} iterations and {d['jacobi_times'][-1]:.0f} s"
                                     f" to converge: {slower:.0f}× slower", t, race_start + race_len + 1.1,
                    fontsize=15, color=MUTED)

    n_pmg = len(pmg_t)
    err_norm = LogNorm(1e-8, 10)
    residual = d["pmg_residuals"] / d["pmg_residuals"][0]
    rtol = 1e-9

    def errors(fig, t, duration=8.0):
        # The error field is against the converged solution, so the last iterate (which is
        # that solution) is left out: iterations 0 .. n - 2
        per = (duration - 1.2) / (n_pmg - 1)
        k = min(n_pmg - 2, int(max(t - 0.4, 0) / per))
        vk.section_heading(fig, "Under the hood", "Every iteration removes most of the error", t)
        ax = section_axes(fig, [0.05, 0.2, 0.5, 0.56], f"|error in p*| after iteration {k}")
        e = np.maximum(np.abs(d["pmg_err_p_yz"][k, 0]), 1e-12)
        im = show_section(ax, d, e, err_norm, "inferno")
        colorbar(fig, im, [0.05, 0.08, 0.3, 0.02], "pressure error [hPa], log scale",
                 ticks=[1e-8, 1e-6, 1e-4, 1e-2, 1])
        ax = fig.add_axes([0.62, 0.2, 0.33, 0.56])
        ax.set_yscale("log")
        ax.set_ylim(1e-11, 3)
        ax.set_xlim(-0.6, n_pmg - 0.4)
        ax.set_xlabel("CG iteration")
        ax.set_title("residual, relative to the first", loc="left", fontsize=17, pad=10)
        ax.grid(True, axis="y", color=EDGE, lw=0.8)
        for spine in ax.spines.values():
            spine.set_color(EDGE)
        colours = [ACCENT if i == k else "#2b4a6b" if i < k else "#1a2235" for i in range(n_pmg)]
        ax.bar(np.arange(n_pmg), residual - 1e-12, color=colours, width=0.7, bottom=1e-12)
        ax.axhline(rtol, color=GOOD, lw=1.5, ls="--")
        ax.text(n_pmg - 0.5, rtol * 2, "stopping tolerance", color=GOOD, ha="right", fontsize=12)
        vk.text(fig, 0.62, 0.11, "p-multigrid: smooth on p = 4, 3, 2, solve exactly on p = 1",
                t, 0.8, fontsize=16, color=MUTED)

    def the_output(fig, t, duration=9.0):
        u = t / duration
        img = out.shot(-95 + 70 * vk.ease(u), 26 - 14 * vk.ease(u), 4300 - 300 * u, 4.5)
        vk.image(fig, img, [0.0, 0.03, 0.6, 0.94])
        x = 0.62
        vk.section_heading(fig, "The output", "The low reaches the ground", t, x=x, y=0.80)
        lines = [
            ("Wind, pressure and temperature, recovered\neverywhere from PV alone.", 0.5),
            (f"At the surface: a {abs(d['min_surf_p']):.1f} hPa low with\n"
             f"{d['max_surf_wind']:.1f} m/s cyclonic winds.", 1.6),
            ("A derivative-free optimiser found intrusions\nwith about 10× this surface response.", 2.7),
        ]
        for i, (s, t0) in enumerate(lines):
            vk.text(fig, x, 0.64 - i * 0.12, s, t, t0, fontsize=21, linespacing=1.35)
        vk.chip_row(fig, [f"{level:g} hPa".replace("-", "−") for level in LOW_LEVELS], 0.2, t, 3.6,
                    x=x, colors=[colour for colour, _ in LOW_STYLE], stagger=0.2)
        vk.text(fig, x, 0.25, "pressure anomaly induced by the intrusion", t, 3.5, fontsize=14,
                color=MUTED)

    def verified(fig, t):
        vk.section_heading(fig, "Built for speed, checked for accuracy", "Fast, scalable, verified", t)
        for i, (img, caption, sub) in enumerate([
                (figs["scaling"], "Solve time grows linearly with problem size",
                 "matrix-free beats assembled at every size (UCT HPC, 40 cores)"),
                (figs["convergence"], "Error falls as Δz$^{p+1}$: fifth order at p = 4",
                 "method of manufactured solutions")]):
            x0 = 0.05 + i * 0.47
            reveal = vk.ease((t - 0.4 - 0.5 * i) / 1.2)
            if reveal <= 0:
                continue
            vk.card(fig, [x0, 0.3, 0.43, 0.5], color="white", edge="white", alpha=reveal)
            h, w = img.shape[:2]
            cols = max(1, int(w * reveal))
            vk.image(fig, img[:, :cols], [x0 + 0.01, 0.315, 0.41 * cols / w, 0.47])
            vk.text(fig, x0, 0.25, caption, t, 0.6 + 0.5 * i, fontsize=20, fontweight="semibold")
            vk.text(fig, x0, 0.215, sub, t, 0.8 + 0.5 * i, fontsize=15, color=MUTED)
        vk.chip_row(fig, ["40+ solver configurations benchmarked", "< 3 s per atmospheric state (HPC)",
                          "1000-state parameter sweeps", "MPI-parallel"], 0.1, t, 2.2,
                    center=0.5, stagger=0.25, gap=0.02)

    def end(fig, t):
        vk.end_card(fig, t)

    return [vk.Scene(5.0, title, fade_in=0.6), vk.Scene(8.5, the_input), vk.Scene(16.0, race),
            vk.Scene(8.0, errors), vk.Scene(9.0, the_output), vk.Scene(8.0, verified),
            vk.Scene(5.5, end, fade_out=0.8)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("out")
    parser.add_argument("--stills", help="Directory to save README stills into")
    parser.add_argument("--only", type=int, help="Render one scene only (preview)")
    parser.add_argument("--pdf", default=os.path.join(os.path.dirname(__file__), "..", "docs", "thesis.pdf"))
    args = parser.parse_args()

    vk.setup_style()
    d = load(args.data_dir)
    figs = thesis_figures.load(args.pdf)
    stills = {}
    if args.stills:
        os.makedirs(args.stills, exist_ok=True)
        stills = {os.path.join(args.stills, "diagnostic_title.png"): (0, 4.0),
                  os.path.join(args.stills, "diagnostic_race.png"): (2, 15.0),
                  os.path.join(args.stills, "diagnostic_output.png"): (4, 8.0)}
    vk.render(scenes(d, figs), args.out, stills=stills, only=args.only)


if __name__ == "__main__":
    main()
