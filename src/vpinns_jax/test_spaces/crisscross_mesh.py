"""
Criss-cross triangulations of rectangles.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax_fem.mesh import TriangleMesh, create_triangle_mesh_from_arrays

# Corners of a cell relative to its lower-left corner, counterclockwise.
_CELL_CORNERS = ((0, 0), (1, 0), (1, 1), (0, 1))


def _on_line(axis: int, value: float) -> Callable[[jax.Array], jax.Array]:
    """
    Boundary predicate of the facets on a coordinate line.

    Parameters
    ----------
    axis : int
        Coordinate fixed on the line.
    value : float
        Value of that coordinate.

    Returns
    -------
    predicate : Callable[[jax.Array], jax.Array]
        Maps facet midpoints, shape ``(N, 1, 2)``, to ``True`` for those on
        the line, shape ``(N,)``.
    """

    def predicate(midpoints: jax.Array) -> jax.Array:
        """
        Whether each facet midpoint lies on the line.

        Parameters
        ----------
        midpoints : jax.Array
            Facet midpoints, shape ``(N, 1, 2)``.

        Returns
        -------
        on_line : jax.Array
            Shape ``(N,)``.
        """
        return jnp.isclose(midpoints[..., 0, axis], value)

    return predicate


def create_crisscross_mesh(
    lower: tuple[float, float],
    upper: tuple[float, float],
    cells: tuple[int, int],
) -> TriangleMesh:
    """
    Criss-cross triangulation of a rectangle.

    Parameters
    ----------
    lower : tuple[float, float]
        Lower-left corner of the rectangle.
    upper : tuple[float, float]
        Upper-right corner of the rectangle.
    cells : tuple[int, int]
        Numbers of grid cells along ``x`` and ``y``, each at least one.

    Returns
    -------
    mesh : TriangleMesh
        Every grid cell cut into four triangles by its two diagonals. The
        grid points come first, the point ``(i, j)`` with index
        ``j (n_x + 1) + i``, then the cell centres, the centre of the cell
        ``(i, j)`` with index ``(n_x + 1) (n_y + 1) + j n_x + i``. The
        boundary facets are tagged ``"bottom"``, ``"right"``, ``"top"`` and
        ``"left"``, as by ``create_gmsh_unit_square_mesh``.

    Raises
    ------
    ValueError
        If a number of cells is not positive or the rectangle is empty.

    Notes
    -----
    On the unit square with ``n`` cells per side, this is the triangulation
    ``T_s`` of the paper, ``s = 1 / n``: every weight of the ``l^inf`` and
    criss-cross families is affine on each of its triangles, and so is the
    covering function ``S``. With ``n r`` cells per side it refines ``T_s``
    for every integer ``r``.
    """
    cells_x, cells_y = cells
    if cells_x < 1 or cells_y < 1:
        raise ValueError(f"cells must be positive, got {cells}.")
    lower_corner = np.asarray(lower, dtype=float)
    upper_corner = np.asarray(upper, dtype=float)
    if not np.all(upper_corner > lower_corner):
        raise ValueError(f"upper {upper} must exceed lower {lower} in each coordinate.")
    spacing = (upper_corner - lower_corner) / np.array([cells_x, cells_y])

    grid_j, grid_i = np.divmod(np.arange((cells_x + 1) * (cells_y + 1)), cells_x + 1)
    grid_points = lower_corner + spacing * np.stack((grid_i, grid_j), axis=-1)
    cell_j, cell_i = np.divmod(np.arange(cells_x * cells_y), cells_x)
    cell_centres = lower_corner + spacing * (np.stack((cell_i, cell_j), axis=-1) + 0.5)

    centre_indices = (cells_x + 1) * (cells_y + 1) + np.arange(cells_x * cells_y)
    corners = [(cell_j + b) * (cells_x + 1) + cell_i + a for a, b in _CELL_CORNERS]
    triangles = np.stack(
        [
            np.stack((centre_indices, corners[k], corners[(k + 1) % 4]), axis=-1)
            for k in range(4)
        ],
        axis=-2,
    )
    return create_triangle_mesh_from_arrays(
        np.concatenate((grid_points, cell_centres), axis=-2),
        np.reshape(triangles, (-1, 3)),
        boundary_data={
            "bottom": _on_line(1, float(lower_corner[1])),
            "right": _on_line(0, float(upper_corner[0])),
            "top": _on_line(1, float(upper_corner[1])),
            "left": _on_line(0, float(lower_corner[0])),
        },
    )


def create_crisscross_unit_square_mesh(cells_per_side: int) -> TriangleMesh:
    """
    Criss-cross triangulation of the unit square.

    Parameters
    ----------
    cells_per_side : int
        Number of grid cells per side, at least one.

    Returns
    -------
    mesh : TriangleMesh
        ``create_crisscross_mesh((0, 0), (1, 1), (n, n))``.
    """
    return create_crisscross_mesh(
        (0.0, 0.0), (1.0, 1.0), (cells_per_side, cells_per_side)
    )
