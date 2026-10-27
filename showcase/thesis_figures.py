# AI-generated (Claude Code) for the README and videos only. Not part of the thesis code:
# the project's own solvers and experiments are the files in the repository root.
"""Figures cut straight out of the compiled thesis, so the videos and README show the
published results (HPC timings, convergence studies) rather than re-creations of them."""
import numpy as np
import pymupdf

# Figure number -> page of docs/thesis.pdf it is on
FIGURES = {"scaling": (9, 22), "convergence": (10, 25), "evolution": (30, 49),
           "epv": (7, 16), "random_sample": (23, 38)}


def crop(pdf, number, page_number, dpi=220):
    """The figure above caption `number` on `page_number`, as an RGB array, margins trimmed."""
    page = pdf[page_number - 1]
    caption = page.search_for(f"Figure {number}:")[0]
    # The figure is the vector artwork between the caption and whatever sits above it
    above = [r.y1 for other in range(1, 60) if other != number
             for r in page.search_for(f"Figure {other}:") if r.y1 < caption.y0]
    top = max([40.0] + [y + 25 for y in above] +
              [b[3] for b in page.get_text("blocks") if b[3] < caption.y0 - 5 and len(b[4]) > 200])
    art = [r for r in page.cluster_drawings() if r.y0 >= top - 5 and r.y1 <= caption.y0 + 2]
    box = art[0]
    for r in art[1:]:
        box |= r
    clip = pymupdf.Rect(box.x0 - 42, box.y0 - 16, box.x1 + 48, min(box.y1 + 30, caption.y0 - 1))
    pix = page.get_pixmap(dpi=dpi, clip=clip)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[..., :3]
    return trim(img)


def trim(img, pad=12, tol=10):
    ink = (np.abs(img.astype(int) - 255).sum(-1) > tol)
    rows, cols = np.where(ink.any(1))[0], np.where(ink.any(0))[0]
    r0, r1 = max(rows[0] - pad, 0), min(rows[-1] + pad + 1, img.shape[0])
    c0, c1 = max(cols[0] - pad, 0), min(cols[-1] + pad + 1, img.shape[1])
    return np.ascontiguousarray(img[r0:r1, c0:c1])


def load(pdf_path, names=None):
    pdf = pymupdf.open(pdf_path)
    return {name: crop(pdf, *FIGURES[name]) for name in (names or FIGURES)}
