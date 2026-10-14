"""
Criss-cross patch family: diamonds at the grid points and squares at the cell
centres of the unit square.
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

_GRID_POINT = 0
_CELL_CENTRE = 1
# Quadrants around a grid point, counterclockwise from the upper right.
_QUADRANTS = ((1, 1), (-1, 1), (-1, -1), (1, -1))
# Corners of a cell relative to its centre, counterclockwise from the lower left.
_CELL_CORNERS = ((-1, -1), (1, -1), (1, 1), (-1, 1))


def _crisscross_weight(
    x: jax.Array, centre: jax.Array, kind: jax.Array, *, spacing: float
) -> jax.Array:
    """
    Raw weight of a criss-cross patch at one point.

    Parameters
    ----------
    x : jax.Array
        Point, shape ``(2,)``.
    centre : jax.Array
        Patch centre, shape ``(2,)``.
    kind : jax.Array
        ``0`` for a grid point, ``1`` for a cell centre, shape ``()``.
    spacing : float
        Grid spacing ``s``.

    Returns
    -------
    weight : jax.Array
        ``max(0, 1 - |x - c|_1 / s)`` for a grid point and
        ``max(0, 1 - 2 |x - c|_inf / s)`` for a cell centre, shape ``()``.
    """
    offsets = jnp.abs(x - centre)
    grid_point_weight = 1.0 - jnp.sum(offsets, axis=-1) / spacing
    cell_centre_weight = 1.0 - 2.0 * jnp.max(offsets, axis=-1) / spacing
    return jnp.maximum(
        jnp.where(kind == _CELL_CENTRE, cell_centre_weight, grid_point_weight), 0.0
    )


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


def _cell_centre_index(
    i: np.ndarray, j: np.ndarray, cells_per_side: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Patch index of the centre of the cell ``(i, j)`` and whether it exists.

    Parameters
    ----------
    i : np.ndarray
        Column of the cell, any shape.
    j : np.ndarray
        Row of the cell, same shape.
    cells_per_side : int
        Number of cells per side ``n``.

    Returns
    -------
    index : np.ndarray
        ``(n + 1)^2 + j n + i``, or ``0`` outside the grid, same shape.
    valid : np.ndarray
        ``True`` inside the grid, same shape.
    """
    valid = (i >= 0) & (i < cells_per_side) & (j >= 0) & (j < cells_per_side)
    index = (cells_per_side + 1) ** 2 + j * cells_per_side + i
    return np.where(valid, index, 0), valid


def create_crisscross_family(cells_per_side: int) -> MeshfreePatchFamily:
    """
    Criss-cross patch family of the unit square.

    Parameters
    ----------
    cells_per_side : int
        Number of grid cells per side ``n``; the spacing is ``s = 1 / n``.

    Returns
    -------
    family : MeshfreePatchFamily
        The ``(n + 1)^2`` grid-point patches, the point ``(i s, j s)`` with
        index ``j (n + 1) + i``, followed by the ``n^2`` cell-centre patches,
        the centre ``((i + 1/2) s, (j + 1/2) s)`` with index
        ``(n + 1)^2 + j n + i``.

    Notes
    -----
    The patch of a grid point ``z`` is the diamond ``|x - z|_1 < s`` cut to
    the unit square, with weight ``1 - |x - z|_1 / s``: a full diamond
    inside, half of it on a side, and a quarter at a corner. The patch of a
    cell centre ``m`` is its cell, with weight ``1 - 2 |x - m|_inf / s``. The
    weights sum to one, so the family needs no normalisation, and the
    boundary patches are the ``4 n`` grid points on the boundary.

    The local triangles of a patch are the triangles with a vertex at its
    centre obtained by cutting every cell along both diagonals: eight for an
    interior grid point, four for a cell or a point on a side, and two at a
    corner. Every weight is affine on each of them.
    """
    spacing = 1.0 / cells_per_side
    grid_j, grid_i = np.divmod(np.arange((cells_per_side + 1) ** 2), cells_per_side + 1)
    cell_j, cell_i = np.divmod(np.arange(cells_per_side**2), cells_per_side)
    grid_points = spacing * np.stack((grid_i, grid_j), axis=-1).astype(float)
    cell_centres = spacing * (np.stack((cell_i, cell_j), axis=-1) + 0.5)

    # Two triangles per quadrant, between the point, a neighbouring grid
    # point, and the centre of the quadrant's cell.
    grid_triangles = []
    grid_triangle_mask = []
    grid_cells = []
    for x_sign, y_sign in _QUADRANTS:
        cell = _cell_centre_index(
            grid_i - (x_sign < 0), grid_j - (y_sign < 0), cells_per_side
        )
        grid_cells.append(cell)
        centre = grid_points + 0.5 * spacing * np.array([x_sign, y_sign])
        for step in ((x_sign, 0), (0, y_sign)):
            grid_triangles.append(
                np.stack(
                    (grid_points, grid_points + spacing * np.array(step), centre),
                    axis=-2,
                )
            )
            grid_triangle_mask.append(cell[1])

    corners = [cell_centres + 0.5 * spacing * np.array(c) for c in _CELL_CORNERS]
    cell_triangles = [
        np.stack((cell_centres, corners[k], corners[(k + 1) % 4]), axis=-2)
        for k in range(4)
    ]
    cell_triangle_mask = np.ones(cell_i.shape + (4,), dtype=bool)

    grid_neighbours = [
        _grid_point_index(grid_i + a, grid_j + b, cells_per_side)
        for a, b in ((0, 0), (1, 0), (0, 1), (-1, 0), (0, -1))
    ] + grid_cells
    cell_neighbours = [_cell_centre_index(cell_i, cell_j, cells_per_side)] + [
        _grid_point_index(cell_i + a, cell_j + b, cells_per_side)
        for a, b in ((0, 0), (1, 0), (1, 1), (0, 1))
    ]
    padding = len(grid_neighbours) - len(cell_neighbours)
    cell_neighbours += [(np.zeros_like(cell_i), np.zeros_like(cell_i, dtype=bool))] * (
        padding
    )

    return create_meshfree_patch_family(
        functools.partial(_crisscross_weight, spacing=spacing),
        centres=np.concatenate((grid_points, cell_centres), axis=-2),
        kinds=np.concatenate(
            (np.full(grid_i.shape, _GRID_POINT), np.full(cell_i.shape, _CELL_CENTRE)),
            axis=-1,
        ),
        local_triangles=np.concatenate(
            (
                np.stack(grid_triangles, axis=-3),
                np.stack(cell_triangles * 2, axis=-3),
            ),
            axis=-4,
        ),
        triangle_mask=np.concatenate(
            (
                np.stack(grid_triangle_mask, axis=-1),
                np.concatenate((cell_triangle_mask, ~cell_triangle_mask), axis=-1),
            ),
            axis=-2,
        ),
        neighbours=np.concatenate(
            (
                np.stack([index for index, _ in grid_neighbours], axis=-1),
                np.stack([index for index, _ in cell_neighbours], axis=-1),
            ),
            axis=-2,
        ),
        neighbour_mask=np.concatenate(
            (
                np.stack([valid for _, valid in grid_neighbours], axis=-1),
                np.stack([valid for _, valid in cell_neighbours], axis=-1),
            ),
            axis=-2,
        ),
        boundary_patch_mask=np.concatenate(
            (
                (grid_i == 0)
                | (grid_i == cells_per_side)
                | (grid_j == 0)
                | (grid_j == cells_per_side),
                np.zeros(cell_i.shape, dtype=bool),
            ),
            axis=-1,
        ),
    )
