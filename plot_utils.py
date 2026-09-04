"""Shared styling and helpers for the thesis figures.

Deliberately free of Firedrake imports, so the plotting entry points run
wherever LaTeX is installed rather than only where the solver is.
"""

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np

# Two single-column widths of the LaTeX document
FIGURE_SIZE = (3.15 * 2, 4.0)

# A single-column width, for roughly square figures placed two to a row.
SQUARE_HALF_FIGURE_SIZE = (FIGURE_SIZE[0] / 2, FIGURE_SIZE[0] / 2)

def apply_style():
    """Global parameters for academic styling, matching the LaTeX document."""
    plt.rcParams.update({
        "text.usetex": True,  # Use LaTeX to render text
        "font.family": "serif",  # Use serif fonts
        "font.serif": ["Computer Modern"],  # Match default LaTeX font
        "text.latex.preamble": r"\usepackage{siunitx}",
        "font.size": 11,  # Match typical LaTeX document font size
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 1.0,  # Thicker axes frames
        "xtick.direction": "in",  # Inward facing ticks
        "ytick.direction": "in"
    })

def qualitative_colours(n):
    """
    n visually distinguishable RGBA colours from a 20-colour qualitative map,
    reordered so the 10 distinct hues are used before their lighter tab20
    pairing repeats one - keeps neighbouring categories (e.g. varied
    parameters) as distinguishable as possible. Categorical, not ordinal, so
    a sequential map like viridis would be misleading here.
    """
    tab20 = plt.cm.tab20.colors
    order = list(range(0, 20, 2)) + list(range(1, 20, 2))
    return [tab20[order[i % len(order)]] for i in range(n)]

def log_log_slope(x, y): # todo switch to max gradient
    """Least-squares gradient of log(y) against log(x), or nan if there is nothing to fit."""
    if len(x) < 2:
        return float('nan')

    slope, _ = np.polyfit(np.log(x), np.log(y), 1)
    return slope

def finish_figure(output_path, legend_kwargs=None, legend=True):
    """Add the shared grid (and, by default, a legend) to the current figure,
    then write it out.

    legend_kwargs is forwarded to plt.legend(), e.g. to move a many-entry
    legend outside the axes rather than overlapping the plotted lines. Pass
    legend=False for figures that label their lines directly instead (see
    label_lines_at_end).
    """
    plt.grid(True, which='both', linestyle=':', alpha=0.5)
    if legend:
        plt.legend(**(legend_kwargs or {}))
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def _declutter_isotonic(values, min_gap):
    """
    Nudge sorted `values` apart so consecutive entries are at least min_gap
    apart, minimizing total squared displacement from their original
    positions (isotonic regression via pool-adjacent-violators).
    """
    shifted = [v - i * min_gap for i, v in enumerate(values)]
    blocks = []  # [sum, count] per pooled block; block averages non-decreasing
    for v in shifted:
        blocks.append([v, 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            s2, c2 = blocks.pop()
            s1, c1 = blocks.pop()
            blocks.append([s1 + s2, c1 + c2])
    expanded = [s / c for s, c in blocks for _ in range(c)]
    return [expanded[i] + i * min_gap for i in range(len(values))]

def label_lines_at_end(ax, entries, fontsize=7):
    """
    Annotate each line directly at its right-hand endpoint instead of using a
    legend - draws a short coloured tick at the endpoint and the label text
    just beyond the axes, in place of a legend box. Labels are vertically
    decluttered so overlapping line endpoints don't produce overlapping
    text, with a tick linking each label back to the endpoint it belongs to.

    entries: iterable of (y_end, colour, text), one per line. The x-axis is
    assumed to run over [0, 1], with every line ending at x=1.
    """
    order = sorted(range(len(entries)), key=lambda i: entries[i][0])
    y_ends = [entries[i][0] for i in order]

    # Data-y distance one line of label text needs, so decluttered labels
    # don't overlap - derived from the current data-to-pixel scale, so it
    # holds regardless of the axes' y-range or size.
    disp_per_data = abs(ax.transData.transform((0, 1))[1] - ax.transData.transform((0, 0))[1])
    min_gap = (fontsize * 1.4) * (ax.figure.dpi / 72.0) / disp_per_data if disp_per_data else 0.0

    label_ys = _declutter_isotonic(y_ends, min_gap) if min_gap else y_ends

    # x in axes-fraction, y in data coordinates, so labels stay pinned just
    # outside the right edge of the plot regardless of final layout.
    transform = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    for rank, idx in enumerate(order):
        y_end, colour, text = entries[idx]
        y_label = label_ys[rank]
        ax.plot([1.0, 1.04], [y_end, y_label], transform=transform, color=colour,
                linewidth=0.8, clip_on=False)
        ax.text(1.05, y_label, text, transform=transform, color=colour,
                fontsize=fontsize, va='center', ha='left', clip_on=False)

    if min_gap:
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(min(ymin, label_ys[0] - min_gap), max(ymax, label_ys[-1] + min_gap))
