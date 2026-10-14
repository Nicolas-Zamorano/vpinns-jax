"""
Option 1 (criss-cross family), n = 2: the family before and after refining one cell
patch by its nested children.

  (a) n = 2: 13 patches (4 corners, 4 mid-side, 1 interior node, 4 cells), S = 1, N_ov = 3
  (b) the cell patch centred at (1/4,1/4) is removed and replaced by its 5 children:
      one diamond of radius s/2 at the same centre (alpha = 1) and the 4 squares of the
      sub-cells (alpha = 1/2). Neighbours are untouched.
  (c) how many patches cover each point after the refinement (3 or 4)

Usage:  python plot_refinement.py [--outdir figures]
Output: refinement_n2.{pdf,png}
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon

INK, DARK, CENTRE, FILL = "#52514E", "#1f1e1c", "#EB6834", "#cde2fb"
NEW = "#2A78D6"
plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.edgecolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "mathtext.fontset": "cm"})

S = 0.5                                       # grid spacing for n = 2
M = (0.25, 0.25)                              # centre of the refined cell
NODES = [(i * S, j * S) for i in range(3) for j in range(3)]
CELLS = [((i + .5) * S, (j + .5) * S) for i in range(2) for j in range(2)]
KIDS_CELLS = [(M[0] + a * S / 4, M[1] + b * S / 4) for a in (-1, 1) for b in (-1, 1)]


# ----------------------------------------------------------------------------- weights
def diamond(X, Y, c, r):
    return np.clip(1 - (abs(X - c[0]) + abs(Y - c[1])) / r, 0, None)


def square(X, Y, c, h):
    return np.clip(1 - np.maximum(abs(X - c[0]), abs(Y - c[1])) / h, 0, None)


def refined_family(X, Y):
    """Weights of the refined family and their coefficients alpha (A5)."""
    w = [(diamond(X, Y, c, S), 1.0) for c in NODES]
    w += [(square(X, Y, c, S / 2), 1.0) for c in CELLS if c != M]
    w += [(diamond(X, Y, M, S / 2), 1.0)]
    w += [(square(X, Y, c, S / 4), 0.5) for c in KIDS_CELLS]
    return w


# ----------------------------------------------------------------------------- drawing
def criss_cross(ax, x0, y0, h, **kw):
    """Edges and both diagonals of the cell [x0,x0+h]x[y0,y0+h]."""
    for xs, ys in [((x0, x0 + h), (y0, y0)), ((x0, x0 + h), (y0 + h, y0 + h)),
                   ((x0, x0), (y0, y0 + h)), ((x0 + h, x0 + h), (y0, y0 + h)),
                   ((x0, x0 + h), (y0, y0 + h)), ((x0, x0 + h), (y0 + h, y0))]:
        ax.plot(xs, ys, **kw)


def frame(ax, title):
    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color=DARK, lw=2.2, zorder=5)
    ax.set_xlim(-0.06, 1.06)
    ax.set_ylim(-0.06, 1.06)
    ax.set_aspect("equal")
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(length=2, labelsize=8)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, pad=8)


def node_markers(ax, pts, **kw):
    ax.plot(*np.array(pts).T, ls="", clip_on=False, zorder=7, **kw)


def panel_before(ax):
    for i in range(2):
        for j in range(2):
            criss_cross(ax, i * S, j * S, S, color=INK, lw=0.7, alpha=0.6)
    ax.add_patch(Polygon([(0, 0), (S, 0), (S, S), (0, S)], color=FILL, zorder=0))
    corners = [p for p in NODES if p[0] in (0, 1) and p[1] in (0, 1)]
    midside = [p for p in NODES if p not in corners and p != (0.5, 0.5)]
    node_markers(ax, corners, marker="o", ms=8, mfc=CENTRE, mec="white")
    node_markers(ax, midside, marker="D", ms=6.5, mfc=CENTRE, mec="white")
    node_markers(ax, [(0.5, 0.5)], marker="o", ms=8, mfc="white", mec=CENTRE, mew=1.6)
    node_markers(ax, CELLS, marker="s", ms=7, mfc="white", mec=INK, mew=1.4)
    frame(ax, "(a) $n=2$: 13 patches, $S\\equiv1$, $N_{ov}=3$")


def panel_after(ax):
    for i in range(2):
        for j in range(2):
            if (i, j) != (0, 0):
                criss_cross(ax, i * S, j * S, S, color=INK, lw=0.7, alpha=0.6)
    for i in range(2):                                     # fine criss-cross inside the cell
        for j in range(2):
            criss_cross(ax, i * S / 2, j * S / 2, S / 2, color=NEW, lw=0.8, alpha=0.9)
    ax.add_patch(Polygon([(M[0], M[1] - S / 2), (M[0] + S / 2, M[1]), (M[0], M[1] + S / 2),
                          (M[0] - S / 2, M[1])], color=FILL, zorder=0))
    corners = [p for p in NODES if p[0] in (0, 1) and p[1] in (0, 1)]
    midside = [p for p in NODES if p not in corners and p != (0.5, 0.5)]
    node_markers(ax, corners, marker="o", ms=8, mfc=CENTRE, mec="white")
    node_markers(ax, midside, marker="D", ms=6.5, mfc=CENTRE, mec="white")
    node_markers(ax, [(0.5, 0.5)], marker="o", ms=8, mfc="white", mec=CENTRE, mew=1.6)
    node_markers(ax, [c for c in CELLS if c != M], marker="s", ms=7, mfc="white", mec=INK, mew=1.4)
    node_markers(ax, [M], marker="o", ms=7, mfc="white", mec=NEW, mew=1.6)       # new diamond
    node_markers(ax, KIDS_CELLS, marker="s", ms=5, mfc="white", mec=NEW, mew=1.3)
    frame(ax, "(b) refine the cell at $(1/4,1/4)$: 17 patches")


def panel_cover(ax, fig):
    t = np.linspace(0, 1, 801)
    X, Y = np.meshgrid(t, t)
    w = refined_family(X, Y)
    cover = (np.array([f for f, _ in w]) > 0).sum(0)
    S_unit = sum(f for f, _ in w)
    S_alpha = sum(a * f for f, a in w)
    assert np.allclose(S_alpha, 1.0) and cover.max() == 4
    cmap = ListedColormap(["#ffffff", "#cde2fb", "#86b6ef", "#1c5cab"])
    ax.imshow(cover, origin="lower", extent=(0, 1, 0, 1), cmap=cmap, vmin=0.5, vmax=4.5,
              interpolation="nearest")
    for i in range(2):
        for j in range(2):
            if (i, j) != (0, 0):
                criss_cross(ax, i * S, j * S, S, color="white", lw=0.6)
    for i in range(2):
        for j in range(2):
            criss_cross(ax, i * S / 2, j * S / 2, S / 2, color="white", lw=0.6)
    frame(ax, "(c) patches covering each point")
    ax.legend(handles=[Patch(color="#86b6ef", label="3 patches"),
                       Patch(color="#1c5cab", label="4 patches")],
              loc="upper right", fontsize=8, frameon=True, framealpha=0.95)
    ax.text(0.0, -0.17, f"peak-1 weights: $S\\in[{S_unit.min():.2f},{S_unit.max():.2f}]$;  "
            r"$\sum_i\alpha_i\varphi_i\equiv1$", transform=ax.transData, fontsize=8.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="figures")
    out = Path(ap.parse_args().outdir)
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 4.9))
    panel_before(axs[0])
    panel_after(axs[1])
    panel_cover(axs[2], fig)
    handles = [
        Line2D([], [], ls="", marker="o", ms=8, mfc=CENTRE, mec="white", label="corner"),
        Line2D([], [], ls="", marker="D", ms=6.5, mfc=CENTRE, mec="white", label="mid-side"),
        Line2D([], [], ls="", marker="o", ms=8, mfc="white", mec=CENTRE, mew=1.6, label="interior node"),
        Line2D([], [], ls="", marker="s", ms=7, mfc="white", mec=INK, mew=1.4, label="cell"),
        Line2D([], [], ls="", marker="o", ms=7, mfc="white", mec=NEW, mew=1.6, label=r"new diamond (radius $s/2$), $\alpha=1$"),
        Line2D([], [], ls="", marker="s", ms=5, mfc="white", mec=NEW, mew=1.3, label=r"new cells (side $s/2$), $\alpha=1/2$"),
        Patch(color=FILL, label="removed patch (a) / new diamond (b)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8.5)
    fig.subplots_adjust(left=0.03, right=0.99, top=0.90, bottom=0.2, wspace=0.12)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"refinement_n2.{ext}", dpi=200)
    print("wrote", out / "refinement_n2.pdf", "and .png")


if __name__ == "__main__":
    main()
