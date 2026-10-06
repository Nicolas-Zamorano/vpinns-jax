"""
Patch families: raw weight functions represented on an integration mesh.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from jax_fem.mesh import TriangleMesh, create_triangle_mesh_from_arrays
from numpy.typing import ArrayLike

_WEIGHT_TOLERANCE = 1e-12


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class PatchFamily:
    """
    Patches and their raw weight functions on an integration mesh.

    Attributes
    ----------
    mesh : TriangleMesh
        Integration mesh; every raw weight is affine on each of its cells.
    cells_to_patches : jax.Array
        Patches whose weight does not vanish on each cell, padded with ``0``,
        shape ``(K, M)``.
    cell_patch_mask : jax.Array
        ``True`` for real entries of ``cells_to_patches``, shape ``(K, M)``.
    local_weights : jax.Array
        Value of each listed weight at the three vertices of the cell, zero
        at padding, shape ``(K, 1, M, 3)``.
    patch_diameters : jax.Array
        Diameter ``h_i`` of each patch, shape ``(N_patch,)``.
    boundary_patch_mask : jax.Array
        ``True`` for the boundary patches ``I_Gamma``, shape ``(N_patch,)``.
    interior_patches : jax.Array
        The interior patches ``I_Omega`` in increasing order, shape
        ``(N_interior,)``.
    covering_bounds : tuple[float, float]
        ``(c_U2, c_U1)``: minimum and maximum of ``S = sum_i phi_i`` over the
        closed domain.

    Notes
    -----
    A patch family is a finite set of open patches ``P_i`` with raw weights
    ``phi_i``, continuous on the closed domain, with ``0 <= phi_i <= 1`` and
    support ``closure(P_i)``. The integration mesh refines the fan
    triangulation of every patch, so each ``phi_i`` is a P1 function on it,
    stored cell by cell through its nodal values. Then ``S`` is P1 as well,
    its extrema over the closed domain lie at vertices, and
    ``covering_bounds`` are the exact covering constants.

    The boundary patches ``I_Gamma`` are those whose weight does not vanish
    identically on the boundary, which for a P1 weight is decided at the
    boundary vertices. Every patch enters ``S``, but only the interior
    patches ``I_Omega`` define test functions in ``H^1_0``. The whole boundary
    is a Dirichlet boundary.
    """

    mesh: TriangleMesh
    cells_to_patches: jax.Array
    cell_patch_mask: jax.Array
    local_weights: jax.Array
    patch_diameters: jax.Array
    boundary_patch_mask: jax.Array
    interior_patches: jax.Array
    covering_bounds: tuple[float, float]

    def __post_init__(self) -> None:
        """
        Validate the array shapes.

        Raises
        ------
        ValueError
            If the arrays are inconsistent with each other or with the mesh.
        """
        number_of_cells = self.mesh.cells_to_vertices.shape[-2]
        if self.cells_to_patches.ndim != 2 or (
            self.cells_to_patches.shape[-2] != number_of_cells
        ):
            raise ValueError(
                f"cells_to_patches must have shape ({number_of_cells}, M), got "
                f"{self.cells_to_patches.shape}."
            )
        maximum_patches = self.cells_to_patches.shape[-1]
        if self.cell_patch_mask.shape != self.cells_to_patches.shape:
            raise ValueError(
                "cell_patch_mask must have the shape of cells_to_patches, got "
                f"{self.cell_patch_mask.shape}."
            )
        expected_weights = (number_of_cells, 1, maximum_patches, 3)
        if self.local_weights.shape != expected_weights:
            raise ValueError(
                f"local_weights must have shape {expected_weights}, got "
                f"{self.local_weights.shape}."
            )
        if self.patch_diameters.ndim != 1 or (
            self.boundary_patch_mask.shape != self.patch_diameters.shape
        ):
            raise ValueError(
                "patch_diameters and boundary_patch_mask must have shape "
                f"(N_patch,), got {self.patch_diameters.shape} and "
                f"{self.boundary_patch_mask.shape}."
            )
        if not 0.0 < self.covering_bounds[0] <= self.covering_bounds[1]:
            raise ValueError(
                "covering_bounds must satisfy 0 < c_U2 <= c_U1, got "
                f"{self.covering_bounds}."
            )

    @property
    def number_of_patches(self) -> int:
        """
        Number of patches.

        Returns
        -------
        number_of_patches : int
            ``N_patch``, interior and boundary patches together.
        """
        return self.patch_diameters.shape[-1]


def _create_patch_family(
    mesh: TriangleMesh,
    cells_to_patches: np.ndarray,
    cell_patch_mask: np.ndarray,
    local_weights: np.ndarray,
    number_of_patches: int,
) -> PatchFamily:
    """
    Validate raw weights on an integration mesh and derive the patch data.

    Parameters
    ----------
    mesh : TriangleMesh
        Integration mesh.
    cells_to_patches : np.ndarray
        Patches listed on each cell, shape ``(K, M)``; entries outside the
        mask are ignored.
    cell_patch_mask : np.ndarray
        ``True`` for real entries, shape ``(K, M)``.
    local_weights : np.ndarray
        Weight values at the cell vertices, shape ``(K, M, 3)``.
    number_of_patches : int
        Number of patches ``N_patch``.

    Returns
    -------
    family : PatchFamily
        The validated family.

    Raises
    ------
    ValueError
        If a weight leaves ``[0, 1]``, is listed twice on a cell, is
        discontinuous, is missing from a cell where it does not vanish, has
        empty support, or if ``S`` vanishes somewhere on the closed domain.
    """
    cell_patch_mask = np.asarray(cell_patch_mask, dtype=bool)
    cells_to_patches = np.where(cell_patch_mask, cells_to_patches, 0)
    local_weights = np.where(
        np.expand_dims(cell_patch_mask, axis=-1), local_weights, 0.0
    )
    if np.any(local_weights < 0.0) or np.any(local_weights > 1.0):
        raise ValueError("Raw weights must take values in [0, 1].")

    cells_to_vertices = np.asarray(mesh.cells_to_vertices)
    number_of_vertices = mesh.vertex_coordinates.shape[-3]
    entry_cells, entry_slots = np.nonzero(cell_patch_mask)
    entry_patches = cells_to_patches[entry_cells, entry_slots]
    listed_pairs = entry_cells * number_of_patches + entry_patches
    if np.unique(listed_pairs).shape[-1] != listed_pairs.shape[-1]:
        raise ValueError("A patch is listed more than once on a cell.")

    # One (patch, vertex) key per listed weight and cell vertex.
    keys = np.ravel(
        np.expand_dims(entry_patches, axis=-1) * number_of_vertices
        + cells_to_vertices[entry_cells]
    )
    values = np.ravel(local_weights[entry_cells, entry_slots])
    unique_keys, key_indices = np.unique(keys, return_inverse=True)
    nodal_minimum = np.full(unique_keys.shape, np.inf)
    nodal_maximum = np.full(unique_keys.shape, -np.inf)
    np.minimum.at(nodal_minimum, key_indices, values)
    np.maximum.at(nodal_maximum, key_indices, values)
    if np.any(nodal_maximum - nodal_minimum > _WEIGHT_TOLERANCE):
        raise ValueError("Raw weights must be continuous: cells disagree at a vertex.")
    nodal_patches = unique_keys // number_of_vertices
    nodal_vertices = unique_keys % number_of_vertices
    positive = nodal_maximum > _WEIGHT_TOLERANCE

    # Every cell around a vertex where a weight is positive must list it.
    stars = np.asarray(mesh.vertices_to_cells)[nodal_vertices[positive]]
    star_mask = np.asarray(mesh.vertex_patch_mask)[nodal_vertices[positive]]
    required_pairs = (
        stars * number_of_patches + np.expand_dims(nodal_patches[positive], axis=-1)
    )[star_mask]
    if not np.all(np.isin(required_pairs, listed_pairs)):
        raise ValueError(
            "Every cell where a raw weight does not vanish must list that weight."
        )
    supported = np.zeros(number_of_patches, dtype=bool)
    supported[nodal_patches[positive]] = True
    if not np.all(supported):
        raise ValueError(
            f"{np.count_nonzero(~supported)} patches have identically zero weights."
        )

    weight_sums = np.zeros(number_of_vertices)
    np.add.at(weight_sums, nodal_vertices, nodal_maximum)
    uncovered = weight_sums <= _WEIGHT_TOLERANCE
    if np.any(uncovered):
        raise ValueError(
            f"S = sum_i phi_i vanishes at {np.count_nonzero(uncovered)} vertices; "
            "the family does not cover the closed domain (a family without "
            "boundary patches never does)."
        )

    boundary_vertices = np.zeros(number_of_vertices, dtype=bool)
    boundary_vertices[
        np.asarray(mesh.facets_to_vertices)[np.asarray(mesh.boundary_facets)]
    ] = True
    boundary_patch_mask = np.zeros(number_of_patches, dtype=bool)
    boundary_patch_mask[nodal_patches[positive & boundary_vertices[nodal_vertices]]] = (
        True
    )

    support_entries = np.max(local_weights[entry_cells, entry_slots], axis=-1) > (
        _WEIGHT_TOLERANCE
    )
    patch_diameters = _patch_diameters(
        entry_patches[support_entries],
        cells_to_vertices[entry_cells[support_entries]],
        np.asarray(mesh.vertex_coordinates)[..., 0, :],
        number_of_patches,
    )

    return PatchFamily(
        mesh=mesh,
        cells_to_patches=jnp.asarray(cells_to_patches, dtype=jnp.int32),
        cell_patch_mask=jnp.asarray(cell_patch_mask),
        local_weights=jnp.expand_dims(jnp.asarray(local_weights), axis=-3),
        patch_diameters=jnp.asarray(patch_diameters),
        boundary_patch_mask=jnp.asarray(boundary_patch_mask),
        interior_patches=jnp.flatnonzero(jnp.asarray(~boundary_patch_mask)),
        covering_bounds=(float(np.min(weight_sums)), float(np.max(weight_sums))),
    )


def _patch_diameters(
    patches: np.ndarray,
    patch_cell_vertices: np.ndarray,
    vertex_coordinates: np.ndarray,
    number_of_patches: int,
) -> np.ndarray:
    """
    Largest distance between two vertices of the support of each patch.

    Parameters
    ----------
    patches : np.ndarray
        Patch of each support cell entry, shape ``(E,)``.
    patch_cell_vertices : np.ndarray
        Vertices of the cell of each entry, shape ``(E, 3)``.
    vertex_coordinates : np.ndarray
        Vertex coordinates, shape ``(N_vertices, d)``.
    number_of_patches : int
        Number of patches.

    Returns
    -------
    diameters : np.ndarray
        Patch diameters, shape ``(N_patch,)``.
    """
    number_of_vertices = vertex_coordinates.shape[-2]
    keys = np.unique(
        np.ravel(
            np.expand_dims(patches, axis=-1) * number_of_vertices + patch_cell_vertices
        )
    )
    key_patches = keys // number_of_vertices
    key_vertices = keys % number_of_vertices
    boundaries = np.searchsorted(key_patches, np.arange(1, number_of_patches))
    diameters = np.zeros(number_of_patches)
    for patch, vertices in enumerate(np.split(key_vertices, boundaries)):
        points = vertex_coordinates[vertices]
        differences = np.expand_dims(points, axis=-2) - np.expand_dims(points, axis=-3)
        diameters[patch] = np.sqrt(np.max(np.sum(differences**2, axis=-1)))
    return diameters


def create_vertex_star_family(mesh: TriangleMesh) -> PatchFamily:
    """
    Family of vertex stars with the nodal P1 shape functions as raw weights.

    Parameters
    ----------
    mesh : TriangleMesh
        Mesh whose vertex stars are the patches; it is also the integration
        mesh.

    Returns
    -------
    family : PatchFamily
        One patch per vertex, with ``phi_z = lambda_z``.

    Notes
    -----
    The shape functions of all vertices, including the boundary ones, form a
    partition of unity, so ``S = 1`` and the normalisation is the identity.
    The interior patches then span the P1 space with homogeneous Dirichlet
    values, and the patch test space coincides with the finite element test
    space of the same mesh.
    """
    number_of_cells = mesh.cells_to_vertices.shape[-2]
    return _create_patch_family(
        mesh,
        cells_to_patches=np.asarray(mesh.cells_to_vertices),
        cell_patch_mask=np.ones((number_of_cells, 3), dtype=bool),
        local_weights=np.broadcast_to(np.eye(3), (number_of_cells, 3, 3)),
        number_of_patches=mesh.vertex_coordinates.shape[-3],
    )


def create_linf_grid_family(
    cell_mask: ArrayLike,
    *,
    spacing: float,
    origin: tuple[float, float] = (0.0, 0.0),
) -> PatchFamily:
    """
    Family of ``l^inf`` cones centred at every grid point of a union of squares.

    Parameters
    ----------
    cell_mask : ArrayLike
        Grid squares that make up the domain, shape ``(n_y, n_x)``; entry
        ``[j, i]`` is the square ``origin + spacing * ([i, i + 1] x [j, j + 1])``.
    spacing : float
        Grid spacing ``s``.
    origin : tuple[float, float]
        Lower-left corner of the grid.

    Returns
    -------
    family : PatchFamily
        One patch per grid point of the domain, interior and boundary alike.

    Notes
    -----
    The raw weight of the grid point ``c`` is
    ``phi_c(x) = max(0, 1 - |x - c|_inf / s)`` restricted to the domain, the
    fan hat of the square patch ``|x - c|_inf < s``: a full square for an
    interior centre, a half square for a centre on a side, and the part of
    the square inside the domain at a corner, including re-entrant corners.
    The integration mesh splits every grid square into four triangles along
    both diagonals; on it ``phi_c`` is the P1 function with value ``1`` at
    ``c``, ``1/2`` at the centres of the adjacent squares, and ``0`` at the
    other vertices. On the unit square, ``1 <= S <= 2`` on the closed domain.
    """
    cell_mask = np.asarray(cell_mask, dtype=bool)
    number_of_columns = cell_mask.shape[-1]
    rows, columns = np.nonzero(cell_mask)
    # Corners of each square counterclockwise from the lower left, as (row, column).
    corner_offsets = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
    corner_rows = np.expand_dims(rows, axis=-1) + corner_offsets[:, 0]
    corner_columns = np.expand_dims(columns, axis=-1) + corner_offsets[:, 1]
    corner_keys = corner_rows * (number_of_columns + 1) + corner_columns
    grid_keys = np.unique(corner_keys)
    corners = np.searchsorted(grid_keys, corner_keys)

    origin_array = np.asarray(origin, dtype=float)
    grid_points = origin_array + spacing * np.stack(
        (grid_keys % (number_of_columns + 1), grid_keys // (number_of_columns + 1)),
        axis=-1,
    )
    square_centres = origin_array + spacing * np.stack(
        (columns + 0.5, rows + 0.5), axis=-1
    )
    number_of_grid_points = grid_keys.shape[-1]
    centre_vertices = number_of_grid_points + np.arange(rows.shape[-1])

    cells = []
    weights = []
    for side in range(4):
        following = (side + 1) % 4
        cells.append(
            np.stack(
                (centre_vertices, corners[:, side], corners[:, following]), axis=-1
            )
        )
        side_weights = np.zeros((4, 3))
        side_weights[:, 0] = 0.5
        side_weights[side, 1] = 1.0
        side_weights[following, 2] = 1.0
        weights.append(np.broadcast_to(side_weights, (rows.shape[-1], 4, 3)))

    mesh = create_triangle_mesh_from_arrays(
        np.concatenate((grid_points, square_centres), axis=-2),
        np.concatenate(cells, axis=-2),
    )
    return _create_patch_family(
        mesh,
        cells_to_patches=np.concatenate([corners] * 4, axis=-2),
        cell_patch_mask=np.ones((4 * rows.shape[-1], 4), dtype=bool),
        local_weights=np.concatenate(weights, axis=-3),
        number_of_patches=number_of_grid_points,
    )
