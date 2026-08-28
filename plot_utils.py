"""Shared styling and helpers for the thesis figures.

Deliberately free of Firedrake imports, so the plotting entry points run
wherever LaTeX is installed rather than only where the solver is.
"""

import matplotlib.pyplot as plt
import numpy as np

# Two single-column widths of the LaTeX document
FIGURE_SIZE = (3.15 * 2, 4.0)

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

def log_log_slope(x, y): # todo switch to max gradient
    """Least-squares gradient of log(y) against log(x), or nan if there is nothing to fit."""
    if len(x) < 2:
        return float('nan')

    slope, _ = np.polyfit(np.log(x), np.log(y), 1)
    return slope

def finish_figure(output_path):
    """Add the shared grid and legend to the current figure, then write it out."""
    plt.grid(True, which='both', linestyle=':', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
