"""
Vertex family of ``l^inf`` cones centred at the grid points of the unit square.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

from vpinns_jax.test_spaces.meshfree_space import (
    MeshfreePatchFamily,
    create_meshfree_patch_family,
)
from vpinns_jax.test_spaces.poincare_constants import (
    compute_linf_poincare_constants,
    round_up,
)

# Cells around a grid point by their lower-left corner, counterclockwise from
# the upper right.
_CELL_OFFSETS = ((0, 0), (-1, 0), (-1, -1), (0, -1))
# Corners of a cell relative to its lower-left corner, counterclockwise.
_CELL_CORNERS = ((0, 0), (1, 0), (1, 1), (0, 1))
# Grid points whose square meets a grid point's square, the point itself first.
_NEIGHBOUR_OFFSETS = (
    (0, 0),
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
)
# Configurations of the patches by the number of boundary lines through their
# centre: interior, side, corner.
_CONFIGURATIONS = ("interior", "side", "corner")
# Decimals kept when the computed constants c_2 h / s are rounded up.
_CONSTANT_DECIMALS = 3


def linf_poincare_constants(refinement: int = 32) -> np.ndarray:
    """
    Constants ``c_2`` of (W2) of the interior, side, and corner patches.

    Parameters
    ----------
    refinement : int
        Number of mesh cells per grid spacing of the eigenvalue problems.

    Returns
    -------
    constants : np.ndarray
        ``c_2`` of each configuration, shape ``(3,)``.

    Notes
    -----
    ``c_2 h / s`` is computed with ``compute_linf_poincare_constants`` and
    rounded up in the third decimal: the P1 eigenvalue problems approximate
    the constants from below, and the rounded values bound them for the
    default refinement. The diameters are ``h = 2 sqrt(2) s``, ``sqrt(5) s``
    and ``sqrt(2) s``.
    """
    constants = [
        compute_linf_poincare_constants(configuration, refinement)
        for configuration in _CONFIGURATIONS
    ]
    return np.array(
        [
            round_up(constant.weighted, _CONSTANT_DECIMALS) / constant.diameter
            for constant in constants
        ]
    )


def _linf_weight(
    x: jax.Array, centre: jax.Array, kind: jax.Array, *, spacing: float
) -> jax.Array:
    """
    Raw weight of an ``l^inf`` cone at one point.

    Parameters
    ----------
    x : jax.Array
        Point, shape ``(2,)``.
    centre : jax.Array
        Patch centre, shape ``(2,)``.
    kind : jax.Array
        Unused; every patch has the same kind, shape ``()``.
    spacing : float
        Grid spacing ``s``.

    Returns
    -------
    weight : jax.Array
        ``max(0, 1 - |x - c|_inf / s)``, shape ``()``.
    """
    return jnp.maximum(1.0 - jnp.max(jnp.abs(x - centre), axis=-1) / spacing, 0.0)


def _grid_point_index(
    i: np.ndarray, j: np.ndarray, cells_per_side: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Patch index of the grid point ``(i, j)`` and whether it exists.

    Parameters
    ----------
    i : np.ndarray
        Column of the grid point, any shape.
    j : np.ndarray
        Row of the grid point, same shape.
    cells_per_side : int
        Number of cells per side ``n``.

    Returns
    -------
    index : np.ndarray
        ``j (n + 1) + i``, or ``0`` outside the grid, same shape.
    valid : np.ndarray
        ``True`` inside the grid, same shape.
    """
    valid = (i >= 0) & (i <= cells_per_side) & (j >= 0) & (j <= cells_per_side)
    return np.where(valid, j * (cells_per_side + 1) + i, 0), valid


def create_linf_vertex_family(
    cells_per_side: int, *, poincare_refinement: int = 32
) -> MeshfreePatchFamily:
    """
    Family of ``l^inf`` cones centred at the grid points of the unit square.

    Parameters
    ----------
    cells_per_side : int
        Number of grid cells per side ``n``; the spacing is ``s = 1 / n``.
    poincare_refinement : int
        Number of mesh cells per grid spacing of the eigenvalue problems that
        give the constants ``c_2`` of (W2).

    Returns
    -------
    family : MeshfreePatchFamily
        One patch per grid point ``(i s, j s)``, with index ``j (n + 1) + i``.

    Notes
    -----
    The patch of a grid point ``z`` is the square ``|x - z|_inf < s`` cut to
    the unit square, with weight ``1 - |x - z|_inf / s``: a full square
    inside, a rectangle on a side, and a quarter square at a corner. The
    weights do not sum to one, ``1 <= S <= 2``, so the family is meant to be
    Shepard-normalised; the boundary patches are the ``4 n`` grid points on
    the boundary.

    Every weight has its kinks on the cell edges and the cell diagonals. The
    local triangles of a patch cut each of its cells along both diagonals,
    sixteen for an interior grid point, so ``psi_i = phi_i / S`` is smooth on
    each of them, and the triangles of two patches agree where the patches
    overlap.

    The constants ``c_2`` of (W2) are computed numerically on the three
    reference configurations (``linf_poincare_constants``); they are scale
    invariant, hence the same for every ``n``. With the default refinement,
    ``c_2 h = 0.470 s`` inside and on a side and ``0.361 s`` at a corner,
    against the a priori bound ``2 sqrt(2) s / pi = 0.900 s`` of a concave
    weight on the square of side ``2 s``.
    """
    spacing = 1.0 / cells_per_side
    grid_j, grid_i = np.divmod(np.arange((cells_per_side + 1) ** 2), cells_per_side + 1)
    grid_points = spacing * np.stack((grid_i, grid_j), axis=-1).astype(float)
    # Boundary lines through each grid point: 0 inside, 1 on a side, 2 at a corner.
    boundary_sides = ((grid_i == 0) | (grid_i == cells_per_side)).astype(int) + (
        (grid_j == 0) | (grid_j == cells_per_side)
    )

    triangles = []
    triangle_mask = []
    for a, b in _CELL_OFFSETS:
        cell_i = grid_i + a
        cell_j = grid_j + b
        valid = (
            (cell_i >= 0)
            & (cell_i < cells_per_side)
            & (cell_j >= 0)
            & (cell_j < cells_per_side)
        )
        lower_left = spacing * np.stack((cell_i, cell_j), axis=-1).astype(float)
        centre = lower_left + 0.5 * spacing
        corners = [lower_left + spacing * np.array(c) for c in _CELL_CORNERS]
        triangles += [
            np.stack((centre, corners[k], corners[(k + 1) % 4]), axis=-2)
            for k in range(4)
        ]
        triangle_mask += [valid] * 4

    neighbours = [
        _grid_point_index(grid_i + a, grid_j + b, cells_per_side)
        for a, b in _NEIGHBOUR_OFFSETS
    ]
    return create_meshfree_patch_family(
        functools.partial(_linf_weight, spacing=spacing),
        centres=grid_points,
        kinds=np.zeros(grid_i.shape, dtype=int),
        local_triangles=np.stack(triangles, axis=-3),
        triangle_mask=np.stack(triangle_mask, axis=-1),
        neighbours=np.stack([index for index, _ in neighbours], axis=-1),
        neighbour_mask=np.stack([valid for _, valid in neighbours], axis=-1),
        boundary_patch_mask=boundary_sides > 0,
        poincare_constants=linf_poincare_constants(poincare_refinement)[boundary_sides],
    )
