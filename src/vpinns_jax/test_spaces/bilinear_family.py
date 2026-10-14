"""
Bilinear patch family: tensor-product hats on the squares around the grid
points of the unit square.
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

# Cells around a grid point by their lower-left corner, counterclockwise from
# the upper right.
_CELL_OFFSETS = ((0, 0), (-1, 0), (-1, -1), (0, -1))
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


def _bilinear_weight(
    x: jax.Array, centre: jax.Array, kind: jax.Array, *, spacing: float
) -> jax.Array:
    """
    Raw weight of a bilinear patch at one point.

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
        ``max(0, 1 - |x_1 - c_1| / s) max(0, 1 - |x_2 - c_2| / s)``, shape
        ``()``.
    """
    return jnp.prod(jnp.maximum(1.0 - jnp.abs(x - centre) / spacing, 0.0), axis=-1)


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


def create_bilinear_family(cells_per_side: int) -> MeshfreePatchFamily:
    """
    Bilinear patch family of the unit square.

    Parameters
    ----------
    cells_per_side : int
        Number of grid cells per side ``n``; the spacing is ``s = 1 / n``.

    Returns
    -------
    family : MeshfreePatchFamily
        One patch per grid point ``(i s, j s)``, with index ``j (n + 1) + i``.

    Notes
    -----
    The patch of a grid point ``z`` is the square ``|x - z|_inf < s`` cut to
    the unit square, with the bilinear weight
    ``(1 - |x_1 - z_1| / s)(1 - |x_2 - z_2| / s)``: a full square inside, a
    rectangle on a side, and a quarter square at a corner. The weights sum to
    one, so the family needs no normalisation, and the boundary patches are
    the ``4 n`` grid points on the boundary.

    Every weight is a polynomial on each grid cell. The local triangles of a
    patch split each of its cells along the diagonal through its lower-left
    corner, the same diagonal for every patch, so the triangles of two
    patches agree where the patches overlap.
    """
    spacing = 1.0 / cells_per_side
    grid_j, grid_i = np.divmod(np.arange((cells_per_side + 1) ** 2), cells_per_side + 1)
    grid_points = spacing * np.stack((grid_i, grid_j), axis=-1).astype(float)

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
        lower_right = lower_left + np.array([spacing, 0.0])
        upper_right = lower_left + np.array([spacing, spacing])
        upper_left = lower_left + np.array([0.0, spacing])
        triangles += [
            np.stack((lower_left, lower_right, upper_right), axis=-2),
            np.stack((lower_left, upper_right, upper_left), axis=-2),
        ]
        triangle_mask += [valid, valid]

    neighbours = [
        _grid_point_index(grid_i + a, grid_j + b, cells_per_side)
        for a, b in _NEIGHBOUR_OFFSETS
    ]
    return create_meshfree_patch_family(
        functools.partial(_bilinear_weight, spacing=spacing),
        centres=grid_points,
        kinds=np.zeros(grid_i.shape, dtype=int),
        local_triangles=np.stack(triangles, axis=-3),
        triangle_mask=np.stack(triangle_mask, axis=-1),
        neighbours=np.stack([index for index, _ in neighbours], axis=-1),
        neighbour_mask=np.stack([valid for _, valid in neighbours], axis=-1),
        boundary_patch_mask=(grid_i == 0)
        | (grid_i == cells_per_side)
        | (grid_j == 0)
        | (grid_j == cells_per_side),
    )
