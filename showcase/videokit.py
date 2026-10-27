# AI-generated (Claude Code) for the README and videos only. Not part of the thesis code:
# the project's own solvers and experiments are the files in the repository root.
"""Shared look and plumbing for the showcase videos and README assets.

Frames are drawn with matplotlib (2D panels and text) and PyVista (3D), and piped straight
into ffmpeg. Runs on the host, from the .npz files the capture scripts write - no Firedrake.
"""
import os
import subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyBboxPatch

W, H = 1920, 1080
FPS = 30

BG = "#0b1020"
PANEL = "#141c31"
EDGE = "#26314d"
INK = "#eef2fa"
MUTED = "#8e9ab5"
ACCENT = "#5ec8ff"   # this work
ACCENT2 = "#ffb454"  # comparison / highlight
GOOD = "#7ee3a1"
BG_RGB = np.array(to_rgb(BG)) * 255

REPO_URL = "github.com/TrueCP6/PV-Inversion"


def setup_style():
    for path in ("/usr/share/fonts/opentype/inter", "/usr/share/fonts/truetype/inter"):
        if os.path.isdir(path):
            for f in os.listdir(path):
                if f.endswith((".otf", ".ttf")):
                    font_manager.fontManager.addfont(os.path.join(path, f))
    names = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams.update({
        "font.family": "Inter" if "Inter" in names else "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "text.color": INK, "axes.labelcolor": MUTED, "axes.edgecolor": EDGE,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.facecolor": PANEL,
        "figure.facecolor": BG, "savefig.facecolor": BG,
        "axes.titlecolor": INK, "axes.titleweight": "semibold",
        "font.size": 15, "axes.titlesize": 17, "axes.labelsize": 14,
        "xtick.labelsize": 12, "ytick.labelsize": 12,
    })


# --- timing helpers ---------------------------------------------------------------------

def clamp01(x):
    return float(np.clip(x, 0.0, 1.0))


def ease(x):
    """Smooth in and out."""
    x = clamp01(x)
    return x * x * (3 - 2 * x)


def ease_out(x):
    x = clamp01(x)
    return 1 - (1 - x) ** 3


def ramp(t, t0, duration=0.6):
    """0 before t0, easing up to 1 over `duration`."""
    return ease_out((t - t0) / duration)


# --- drawing helpers --------------------------------------------------------------------

def text(fig, x, y, s, t=None, t0=0.0, rise=0.012, duration=0.6, alpha=1.0, **kw):
    """Figure text that fades and rises into place from time t0."""
    a = 1.0 if t is None else ramp(t, t0, duration)
    if a <= 0:
        return None
    kw.setdefault("color", INK)
    return fig.text(x, y - (1 - a) * rise, s, alpha=a * alpha, transform=fig.transFigure, **kw)


def chip(fig, x, y, label, t=None, t0=0.0, color=ACCENT, size=15, pad=0.45, ha="left"):
    """A rounded label, like a tag."""
    a = 1.0 if t is None else ramp(t, t0, 0.5)
    if a <= 0:
        return
    return fig.text(x, y, label, ha=ha, va="center", fontsize=size, color=color, alpha=a,
                    fontweight="medium",
                    bbox=dict(boxstyle=f"round,pad={pad},rounding_size=0.8", fc=PANEL,
                              ec=color, lw=1.4, alpha=a))


def chip_row(fig, labels, y, t=None, t0=0.0, stagger=0.12, x=None, center=None, gap=0.014,
             size=15, colors=None, pad=0.45):
    """Chips in a row with even gaps, measured from their rendered widths. Starts at x, or is
    centred on `center`."""
    renderer = fig.canvas.get_renderer()
    pad_frac = pad * size * fig.dpi / 72 / W
    widths = []
    for label in labels:
        probe = fig.text(0, 0, label, fontsize=size, fontweight="medium")
        widths.append(probe.get_window_extent(renderer).width / W + 2 * pad_frac)
        probe.remove()
    total = sum(widths) + gap * (len(labels) - 1)
    left = x if x is not None else (0.5 if center is None else center) - total / 2
    for i, (label, width) in enumerate(zip(labels, widths)):
        chip(fig, left + pad_frac, y, label, t, t0 + stagger * i,
             color=(colors[i] if colors else ACCENT), size=size, pad=pad)
        left += width + gap


def card(fig, rect, alpha=1.0, color=PANEL, edge=EDGE):
    """A rounded panel behind content, in figure coordinates."""
    x, y, w, h = rect
    fig.patches.append(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.012",
                                      transform=fig.transFigure, fc=color, ec=edge, lw=1.2,
                                      alpha=alpha, zorder=-1))


def image(fig, img, rect, alpha=1.0, zorder=0, aspect="equal"):
    ax = fig.add_axes(rect, zorder=zorder)
    ax.imshow(img, alpha=alpha, interpolation="lanczos", aspect=aspect)
    ax.set_axis_off()
    return ax


def left_shade(fig, strength=0.96, power=1.3):
    """Darken the left of the frame so text reads over a full-bleed render."""
    shade = np.linspace(1, 0, 256)[None, :, None] ** power
    overlay = np.concatenate([np.broadcast_to(BG_RGB / 255, (1, 256, 3)),
                              np.clip(shade * strength, 0, 1)], axis=-1)
    image(fig, overlay, [0, 0, 1, 1], zorder=1, aspect="auto")


def section_heading(fig, kicker, title, t, x=0.05, y=0.9):
    text(fig, x, y + 0.045, kicker.upper(), t, 0.0, fontsize=15, color=ACCENT,
         fontweight="semibold")
    text(fig, x, y, title, t, 0.1, fontsize=36, fontweight="bold")


AUTHOR = "Robert Eliott"
STACK = ["Python", "Firedrake", "PETSc", "MPI", "NumPy / SciPy", "SLURM HPC"]


def title_block(fig, t, title, subtitle, x=0.06):
    """The opening card's text, over a render shaded by left_shade."""
    text(fig, x, 0.71, "HONOURS THESIS  ·  UNIVERSITY OF CAPE TOWN", t, 0.2, fontsize=17,
         color=ACCENT, fontweight="semibold", va="top")
    text(fig, x, 0.665, title, t, 0.4, fontsize=72, fontweight="bold", linespacing=1.05, va="top")
    text(fig, x, 0.44, subtitle, t, 0.8, fontsize=22, color=MUTED, linespacing=1.45, va="top")
    text(fig, x, 0.315, AUTHOR, t, 1.2, fontsize=22, fontweight="semibold", va="top")


def end_card(fig, t):
    text(fig, 0.5, 0.74, AUTHOR, t, 0.1, fontsize=60, fontweight="bold", ha="center", va="top")
    text(fig, 0.5, 0.62, "Honours thesis · Department of Mathematics and Applied Mathematics\n"
                         "University of Cape Town · supervised by Dr. Stefaan Conradie",
         t, 0.4, fontsize=19, color=MUTED, ha="center", va="top", linespacing=1.5)
    chip_row(fig, STACK, 0.44, t, 0.8, stagger=0.1, center=0.5, size=17, gap=0.016)
    text(fig, 0.5, 0.31, "Code and full thesis", t, 1.6, fontsize=18, color=MUTED, ha="center", va="top")
    text(fig, 0.5, 0.265, REPO_URL, t, 1.7, fontsize=30, fontweight="semibold", color=ACCENT,
         ha="center", va="top")


def frame_rgb(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3]


def autocrop(img, tol=8):
    """Trim uniform margins (e.g. the white page around a PDF figure)."""
    diff = np.abs(img[..., :3].astype(int) - img[0, 0, :3].astype(int)).sum(-1) > tol
    rows, cols = np.where(diff.any(1))[0], np.where(diff.any(0))[0]
    return img[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


# --- scenes and output ------------------------------------------------------------------

class Scene:
    def __init__(self, duration, draw, fade_in=0.35, fade_out=0.35):
        self.duration, self.draw = duration, draw
        self.fade_in, self.fade_out = fade_in, fade_out


def preview(scenes, moments, path_pattern):
    """Save single frames, {(scene index, seconds): ...}, without encoding any video."""
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    for index, t in moments:
        fig.clf()
        fig.patches.clear()
        scenes[index].draw(fig, t)
        plt.imsave(path_pattern.format(index=index, t=t), frame_rgb(fig))
    plt.close(fig)


class Video:
    """Raw RGB frames into ffmpeg, as H.264 that LinkedIn and every browser play."""
    def __init__(self, path, fps=FPS, size=(W, H)):
        self.path = path
        self.proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{size[0]}x{size[1]}", "-r", str(fps), "-i", "-",
             "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", path],
            stdin=subprocess.PIPE)

    def write(self, rgb):
        self.proc.stdin.write(np.ascontiguousarray(rgb, dtype=np.uint8).tobytes())

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()


def render(scenes, path, stills=None, only=None):
    """Draw every scene frame by frame into `path`.

    stills: {name: (scene index, seconds into it)} frames also saved as PNG.
    only: index of a single scene to render (for previews).
    """
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    video = Video(path)
    stills = stills or {}
    for index, scene in enumerate(scenes):
        if only is not None and index != only:
            continue
        n = int(round(scene.duration * FPS))
        wanted = {round(s * FPS): name for name, (i, s) in stills.items() if i == index}
        for k in range(n):
            t = k / FPS
            fig.clf()
            fig.patches.clear()
            scene.draw(fig, t)
            rgb = frame_rgb(fig).astype(np.float32)
            a = min(1.0, (t + 1 / FPS) / scene.fade_in if scene.fade_in else 1.0,
                    (scene.duration - t) / scene.fade_out if scene.fade_out else 1.0)
            if a < 1:
                rgb = rgb * a + BG_RGB * (1 - a)
            video.write(rgb)
            if k in wanted:
                plt.imsave(wanted[k], frame_rgb(fig))
        print(f"scene {index}: {n} frames", flush=True)
    video.close()
    plt.close(fig)


# --- 3D ---------------------------------------------------------------------------------

def plotter(size):
    import pyvista as pv
    p = pv.Plotter(off_screen=True, window_size=size, lighting="three lights")
    p.set_background(BG)
    p.enable_anti_aliasing("ssaa")
    return p


def grid(x, y, z, z_scale, **fields):
    """A uniform grid in km, with height exaggerated by z_scale, carrying `fields`."""
    import pyvista as pv
    g = pv.ImageData(dimensions=(x.size, y.size, z.size),
                     spacing=((x[1] - x[0]) / 1e3, (y[1] - y[0]) / 1e3, (z[1] - z[0]) / 1e3 * z_scale),
                     origin=(x[0] / 1e3, y[0] / 1e3, z[0] / 1e3 * z_scale))
    for name, values in fields.items():
        g.point_data[name] = np.asarray(values, dtype=np.float32).ravel(order="F")
    return g


def refine(axes, values, factors, order=3):
    """Cubic upsampling of gridded samples of a smooth field, so isosurfaces of a feature
    only a few samples wide come out smooth instead of stair-stepped."""
    from scipy.ndimage import zoom
    fine = zoom(np.asarray(values, dtype=np.float32), factors, order=order, mode="nearest")
    return [np.linspace(a[0], a[-1], n) for a, n in zip(axes, fine.shape)], fine


def orbit(p, focus, distance, azimuth, elevation, view_up=(0, 0, 1), pan=0.0):
    """Place the camera on a sphere about `focus` (angles in degrees). `pan` slides the view
    sideways so the subject sits right of centre (positive) or left of it (negative)."""
    az, el = np.radians(azimuth), np.radians(elevation)
    focus = np.array(focus, dtype=float)
    pos = focus + distance * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    right = np.cross(focus - pos, view_up)
    shift = -pan * right / np.linalg.norm(right)
    p.camera_position = [tuple(pos + shift), tuple(focus + shift), view_up]
