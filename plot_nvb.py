"""
Option 1 (criss-cross family), n = 2: local refinement by newest-vertex bisection (NVB).

The patches are the vertex stars of a conforming triangulation and the weights are the
hat functions, so S = sum_i phi_i = 1 holds exactly after every refinement (alpha = 1)
and every point lies in exactly 3 patches.

  (a) n = 2 criss-cross mesh; the 4 triangles of the cell (0,1/2)^2 are marked
  (b) after one bisection: the cell edges are halved; conformity forces the two
      neighbouring triangles across x = 1/2 and y = 1/2 to be bisected as well
  (c) after a second bisection of the cell's triangles: the cell now carries the
      criss-cross mesh of size s/2. Vertices are coloured by the size of their patch
      (coarse, half size, or a new transition shape)
  (d), (e) the weights of two of the three transition patches
  (f) every point lies in exactly 3 patches

Usage:  python plot_nvb.py [--outdir figures]
Output: nvb_refinement_n2.{pdf,png}
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
from scipy.spatial import ConvexHull

INK, DARK, CENTRE, NEW, FILL = "#52514E", "#1f1e1c", "#EB6834", "#2A78D6", "#cde2fb"
WEIGHT_CMAP = LinearSegmentedColormap.from_list(
    "weight", ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.edgecolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "mathtext.fontset": "cm"})


# ----------------------------------------------------------------------------- mesh + NVB
class Mesh:
    """Triangles stored as (newest vertex, v1, v2); the refinement edge is (v1, v2)."""

    def __init__(self, n: int):
        self.P: list[tuple[float, float]] = []
        self.index: dict[tuple[float, float], int] = {}
        self.T: list[tuple[int, int, int]] = []
        s = 1.0 / n
        for i in range(n):
            for j in range(n):
                a, b = self.vertex((i * s, j * s)), self.vertex(((i + 1) * s, j * s))
                c, d = self.vertex(((i + 1) * s, (j + 1) * s)), self.vertex((i * s, (j + 1) * s))
                m = self.vertex(((i + .5) * s, (j + .5) * s))
                # criss-cross: right angle at the cell centre, refinement edge = cell edge
                self.T += [(m, a, b), (m, b, c), (m, c, d), (m, d, a)]

    def vertex(self, p):
        p = (round(p[0], 12), round(p[1], 12))
        if p not in self.index:
            self.index[p] = len(self.P)
            self.P.append(p)
        return self.index[p]

    def neighbour(self, k):
        """Triangle sharing the refinement edge of triangle k (None on the boundary)."""
        e = {self.T[k][1], self.T[k][2]}
        for j, t in enumerate(self.T):
            if j != k and e <= set(t):
                return j
        return None

    def bisect(self, k):
        """Bisect triangle k, recursively refining its neighbour first if needed."""
        while True:
            j = self.neighbour(k)
            if j is None or {self.T[j][1], self.T[j][2]} == {self.T[k][1], self.T[k][2]}:
                break
            self.bisect(j)                       # neighbour has another refinement edge
            k = self._find(self.T[k]) if isinstance(k, tuple) else k
        nv, v1, v2 = self.T[k]
        p = self.vertex(tuple((np.array(self.P[v1]) + np.array(self.P[v2])) / 2))
        pair = [k] + ([j] if j is not None else [])
        new = []
        for idx in pair:
            a, b1, b2 = self.T[idx]
            new += [(p, a, b1), (p, b2, a)]
        for idx in sorted(pair, reverse=True):
            self.T.pop(idx)
        self.T += new

    def refine(self, inside, levels):
        """Bisect, `levels` times, every triangle whose centroid satisfies `inside`."""
        for _ in range(levels):
            marked = [t for t in self.T if inside(np.mean([self.P[v] for v in t], axis=0))]
            for t in marked:
                if t in self.T:
                    self.bisect(self.T.index(t))

    def triangulation(self):
        P = np.array(self.P)
        return mtri.Triangulation(P[:, 0], P[:, 1], np.array(self.T))


# ----------------------------------------------------------------------------- patches
def star(mesh, v):
    return [t for t in mesh.T if v in t]


def star_polygon(mesh, v):
    pts = np.array([mesh.P[w] for t in star(mesh, v) for w in t])
    hull = ConvexHull(pts)                      # every star in this mesh is convex
    return pts[hull.vertices]


def classify(mesh, v, s):
    """Configuration of the patch of vertex v, read off the shape of its star:
    kind (corner / mid-side / node / cell) and size (coarse, half or transition)."""
    c = np.array(mesh.P[v])
    on_side = sum([abs(c[0]) < 1e-12, abs(c[0] - 1) < 1e-12, abs(c[1]) < 1e-12, abs(c[1] - 1) < 1e-12])
    hull = star_polygon(mesh, v) - c               # extreme points only (collinear ones dropped)
    r = np.abs(hull).max()                         # reach of the patch from its centre
    if on_side:
        kind = "corner" if on_side == 2 else "mid-side"
        standard = len(hull) == 3
    elif len(hull) == 4 and np.allclose(np.abs(hull[:, 0]), np.abs(hull[:, 1])):
        kind, standard, r = "cell", True, 2 * r    # square of half-width r: grid size 2r
    else:
        kind = "node"
        standard = len(hull) == 4 and np.all(np.isclose(hull, 0).any(axis=1))
    if not standard:
        return kind, "transition"
    return kind, "coarse" if np.isclose(r, s) else ("fine" if np.isclose(r, s / 2) else "other")


def hats(mesh, X, Y):
    tri = mesh.triangulation()
    out = []
    for v in range(len(mesh.P)):
        z = np.zeros(len(mesh.P))
        z[v] = 1
        out.append(np.asarray(mtri.LinearTriInterpolator(tri, z)(X, Y).filled(0)))
    return np.array(out)


# ----------------------------------------------------------------------------- drawing
MARK = {"corner": "o", "mid-side": "D", "node": "o", "cell": "s"}


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
    ax.set_title(title, pad=8, fontsize=10)


def draw_mesh(ax, mesh, highlight=None):
    tri = mesh.triangulation()
    if highlight is not None:
        for t in mesh.T:
            if highlight(np.mean([mesh.P[v] for v in t], axis=0)):
                ax.add_patch(Polygon([mesh.P[v] for v in t], color=FILL, zorder=0))
    ax.triplot(tri, color=INK, lw=0.7, alpha=0.7)


def draw_vertices(ax, mesh, s, colour_by_size=True):
    for v, p in enumerate(mesh.P):
        kind, size = classify(mesh, v, s)
        col = {"coarse": CENTRE, "fine": NEW, "transition": "#7a3fbf"}[size] if colour_by_size else CENTRE
        filled = kind in ("corner", "mid-side")
        ax.plot(*p, MARK[kind], ms=6.5 if kind != "mid-side" else 5.5, clip_on=False, zorder=7,
                mfc=col if filled else "white", mec="white" if filled else col, mew=1.4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="figures")
    out = Path(ap.parse_args().outdir)
    n, s = 2, 0.5
    in_cell = lambda c: c[0] < 0.5 and c[1] < 0.5

    m0 = Mesh(n)
    m1 = Mesh(n)
    m1.refine(in_cell, 1)
    m2 = Mesh(n)
    m2.refine(in_cell, 2)

    # checks: partition of unity and exactly 3 patches per point
    # sample points off every mesh line (different shifts in x and y)
    X, Y = np.meshgrid((np.arange(600) + .5 + .13) / 600, (np.arange(600) + .5 - .21) / 600)
    H = hats(m2, X, Y)
    assert np.allclose(H.sum(0), 1.0), "hats do not sum to 1"
    cover = (H > 1e-12).sum(0)
    kinds = [classify(m2, v, s) for v in range(len(m2.P))]
    print(f"refined mesh: {len(m2.T)} triangles, {len(m2.P)} patches, S = 1, N_ov = {cover.max()}")
    for k in sorted(set(kinds)):
        print(f"   {k[0]:9s} {k[1]:10s}: {kinds.count(k)}")

    fig = plt.figure(figsize=(12.5, 9.4))
    gs = fig.add_gridspec(2, 3, wspace=0.12, hspace=0.22)
    ax = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]

    draw_mesh(ax[0], m0, highlight=in_cell)
    draw_vertices(ax[0], m0, s, colour_by_size=False)
    frame(ax[0], f"(a) $n=2$: {len(m0.T)} triangles, {len(m0.P)} patches")

    draw_mesh(ax[1], m1, highlight=in_cell)
    for t in m1.T:                                    # closure: bisected neighbours
        c = np.mean([m1.P[v] for v in t], axis=0)
        if not in_cell(c) and t not in m0.T:
            ax[1].add_patch(Polygon([m1.P[v] for v in t], color="#f6d3c2", zorder=0))
    draw_vertices(ax[1], m1, s, colour_by_size=False)
    frame(ax[1], "(b) one bisection (+ closure, orange)")

    draw_mesh(ax[2], m2, highlight=in_cell)
    draw_vertices(ax[2], m2, s)
    frame(ax[2], f"(c) two bisections: {len(m2.T)} triangles, {len(m2.P)} patches")

    tri = m2.triangulation()
    for k, (centre, title) in enumerate([((0.5, 0.0), "(d) transition mid-side patch at $(1/2,0)$"),
                                         ((0.5, 0.5), "(e) transition interior node at $(1/2,1/2)$")]):
        a_ = ax[3 + k]
        v = m2.index[centre]
        z = np.zeros(len(m2.P))
        z[v] = 1
        a_.tricontourf(tri, z, levels=np.linspace(0, 1, 11), cmap=WEIGHT_CMAP)
        a_.triplot(tri, color=INK, lw=0.5, alpha=0.5)
        a_.add_patch(Polygon(star_polygon(m2, v), closed=True, fill=False, ec="#7a3fbf", lw=1.4, zorder=4))
        a_.plot(*centre, "D" if centre[1] == 0 else "o", ms=6.5, mfc="#7a3fbf", mec="white",
                zorder=7, clip_on=False)
        frame(a_, title)

    cmap3 = LinearSegmentedColormap.from_list("c3", ["#86b6ef", "#86b6ef"])
    ax[5].imshow(cover, origin="lower", extent=(0, 1, 0, 1), cmap=cmap3, interpolation="nearest")
    ax[5].triplot(tri, color="white", lw=0.6)
    assert cover.min() == cover.max() == 3
    frame(ax[5], "(f) patches covering each point: 3 everywhere")

    handles = [
        Line2D([], [], ls="", marker="o", ms=6.5, mfc=CENTRE, mec="white", label="corner"),
        Line2D([], [], ls="", marker="D", ms=5.5, mfc=CENTRE, mec="white", label="mid-side"),
        Line2D([], [], ls="", marker="o", ms=6.5, mfc="white", mec=CENTRE, mew=1.4, label="interior node (diamond)"),
        Line2D([], [], ls="", marker="s", ms=6.5, mfc="white", mec=CENTRE, mew=1.4, label="cell (square)"),
        Patch(color=CENTRE, label="coarse size"),
        Patch(color=NEW, label="half size"),
        Patch(color="#7a3fbf", label="transition shape"),
        Patch(color=FILL, label="refined triangles"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8.5)
    fig.subplots_adjust(left=0.03, right=0.99, top=0.95, bottom=0.10)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"nvb_refinement_n2.{ext}", dpi=200)
    print("wrote", out / "nvb_refinement_n2.pdf", "and .png")


if __name__ == "__main__":
    main()
