"""Mesh-free star-shaped patch test space ``V_P = span{varphi_i}`` (math reference Section 2.2).

**Scoping decision.** The math reference's patch construction is fully
abstract: any star-shaped polygon ``P_i`` with kernel point ``c_Pi``,
triangulated by connecting ``c_Pi`` to its vertices, carrying a reference hat
function transported by an affine map ``M_i``. This module realizes one
concrete, standard instance of that construction rather than the fully
abstract one: **vertex-star patches** of the underlying triangle mesh. For
interior vertex ``i``, ``P_i`` is the union of cells incident to vertex
``i``, its sub-triangulation ``T_i`` *is* those incident cells (already a fan
around ``i``, i.e. already star-shaped with kernel point ``i`` itself), and
its hat function ``varphi_i`` is exactly the restriction of the P1 nodal
basis function at vertex ``i`` to its support -- so no separate reference
patch / affine map machinery is built: the vertex-star cover is evaluated
directly through jax-fem's own P1 ``CellBasis``, gathered per patch.

This is a faithful, standard choice of star-shaped cover (it satisfies every
structural requirement in Section 2.2: covers ``Omega``, two-sided overlap
bound, per-patch sub-triangulation) and is what makes an MF-VPINN
*mesh-free* in the sense that matters operationally: residual/loss
evaluation per patch is a small local gather-and-reduce over that patch's
incident cells, never a global sparse assembly. It is less general than the
paper's arbitrary-star-shaped-polygon construction (whose patches need not
come from any single background triangulation at all); extending to that
generality is future work.

Patch topology (which cells/local-DOF a vertex touches) is built once,
eagerly, in plain NumPy -- consistent with how jax-fem itself builds mesh
topology (``mesh.gmsh``, ``mesh.amr``) -- then converted to fixed-shape,
padded JAX arrays for fast, ``vmap``/``jit``-friendly per-patch quadrature
gathers in the hot training loop.
"""

from __future__ import annotations

import dataclasses
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax_fem.element import LagrangeTriangleP1
from jax_fem.mesh import TriangleMesh
from jax_fem.reference_cell import ReferenceTriangle
from jax_fem.space import (
    CellBasis,
    FiniteElementSpace,
    create_cell_basis,
    create_finite_element_space,
    find_boundary_dofs,
)

__all__ = [
    "PatchTestSpace",
    "create_patch_test_space",
    "PatchQuadrature",
    "gather_patch_quadrature",
]


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class PatchTestSpace:
    """Vertex-star patch test space ``V_P = span{varphi_i}_{i in I_P}``.

    Attributes
    ----------
    mesh, space, basis :
        The underlying ``TriangleMesh`` and its P1 ``FiniteElementSpace``/
        ``CellBasis`` (``space``'s DOF ``i`` <-> mesh vertex ``i``).
    interior_vertices : jax.Array
        ``I_P``: global vertex/DOF indices of the patch centers, shape
        ``(N_patch,)``, ascending.
    number_of_test_functions : int
        ``N_patch = |I_P|``.
    patch_cells, patch_local_dof, patch_mask : jax.Array
        Padded incidence tables, shape ``(N_patch, max_valence)``:
        ``patch_cells[i, k]`` is the ``k``-th cell incident to patch ``i``'s
        center vertex, ``patch_local_dof[i, k]`` is which local P1 DOF (0,
        1, or 2) that vertex is within that cell, and ``patch_mask[i, k]``
        is ``True`` for valid (non-padding) slots.
    patch_diameter : jax.Array
        ``h_i = diam(P_i)``, shape ``(N_patch,)``: exact max pairwise
        distance among ``P_i``'s vertex set.
    cover_constant_lower, cover_constant_upper : float
        ``c_{U,2} <= sum_i varphi_i(x) <= c_{U,1}``, estimated numerically as
        the min/max of ``sum_{i in I_P} varphi_i`` over every cell-quadrature
        point of ``basis`` (Section 2.2) -- a numerical certificate over the
        quadrature sample, not a proven analytic bound.
    """

    mesh: TriangleMesh
    space: FiniteElementSpace
    basis: CellBasis
    interior_vertices: jax.Array
    number_of_test_functions: int
    patch_cells: jax.Array
    patch_local_dof: jax.Array
    patch_mask: jax.Array
    patch_diameter: jax.Array
    cover_constant_lower: float
    cover_constant_upper: float


def _build_vertex_incidences(
    cells_to_vertices: np.ndarray, interior_vertices: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Padded ``(patch_cells, patch_local_dof, patch_mask)`` incidence tables."""
    vertex_to_incidences: dict[int, list[tuple[int, int]]] = {}
    for cell_index, cell_vertices in enumerate(cells_to_vertices):
        for local_dof, vertex in enumerate(cell_vertices):
            vertex_to_incidences.setdefault(int(vertex), []).append((cell_index, local_dof))

    incidences = [vertex_to_incidences[int(v)] for v in interior_vertices]
    max_valence = max((len(inc) for inc in incidences), default=0)
    n_patch = len(interior_vertices)

    patch_cells = np.full((n_patch, max_valence), -1, dtype=np.int64)
    patch_local_dof = np.full((n_patch, max_valence), -1, dtype=np.int64)
    patch_mask = np.zeros((n_patch, max_valence), dtype=bool)
    for row, inc in enumerate(incidences):
        for col, (cell_index, local_dof) in enumerate(inc):
            patch_cells[row, col] = cell_index
            patch_local_dof[row, col] = local_dof
            patch_mask[row, col] = True

    return patch_cells, patch_local_dof, patch_mask


def _patch_diameters(
    cells_to_vertices: np.ndarray,
    vertex_coordinates: np.ndarray,
    interior_vertices: np.ndarray,
    patch_cells: np.ndarray,
    patch_mask: np.ndarray,
) -> np.ndarray:
    """Exact ``h_i = diam(P_i)`` for every patch, via its vertex set's max pairwise distance."""
    diameters = np.zeros(len(interior_vertices))
    for row in range(len(interior_vertices)):
        cells = patch_cells[row, patch_mask[row]]
        patch_vertex_ids = np.unique(cells_to_vertices[cells])
        coords = vertex_coordinates[patch_vertex_ids]
        if coords.shape[0] > 1:
            diffs = coords[:, None, :] - coords[None, :, :]
            diameters[row] = np.sqrt((diffs**2).sum(-1)).max()
    return diameters


def _cover_constants(
    basis: CellBasis, cells_to_dofs: jax.Array, interior_mask: jax.Array
) -> tuple[float, float]:
    """Numerically estimated ``(c_{U,2}, c_{U,1})`` over every cell-quadrature point."""
    is_interior_local = interior_mask[cells_to_dofs]  # (K, 3)
    reference_values = basis.values[0, :, :, 0]  # (Q, 3), cell-independent (P1 "identity")
    contributions = jnp.einsum(
        "ql,kl->kq", reference_values, is_interior_local.astype(reference_values.dtype)
    )  # (K, Q): sum_{i in I_P} varphi_i at each cell/quadrature point
    return float(jnp.min(contributions)), float(jnp.max(contributions))


def create_patch_test_space(
    mesh: TriangleMesh,
    *,
    space: FiniteElementSpace | None = None,
    dirichlet_boundary_tags: tuple[int, ...] | None = None,
    quadrature_degree: int = 4,
) -> PatchTestSpace:
    """Build the vertex-star patch test space ``V_P`` on ``mesh``.

    Patch centers ``I_P`` are every mesh vertex *not* on a Dirichlet
    boundary tag (``V_P subset H^1_0``, matching the FE test space's own
    interior restriction).

    Parameters
    ----------
    mesh : TriangleMesh
    space : FiniteElementSpace | None
        An existing P1 ``FiniteElementSpace`` on ``mesh`` to reuse, rather
        than building a fresh one. Pass ``fe_test_space.space`` here when
        combining a patch test space with an *interpolated* trial built on
        an ``FETestSpace`` for the same mesh
        (``losses.residual.interpolated_trial_patch_residual`` requires the
        ``FiniteElementFunction`` and the patch space's basis to share the
        exact same ``FiniteElementSpace`` instance -- jax-fem's
        ``evaluate_finite_element_function`` checks object identity, not
        structural equality). Defaults to building a fresh P1 space.
    dirichlet_boundary_tags : tuple[int, ...] | None
        Boundary tags excluded from ``I_P``. Defaults to every tag the mesh
        has (``mesh.boundary_tag_names.values()``), i.e. the whole boundary.
    quadrature_degree : int
        Cell quadrature exactness degree for patch integration.

    Returns
    -------
    PatchTestSpace
    """
    if space is None:
        element = LagrangeTriangleP1(ReferenceTriangle())
        space = create_finite_element_space(mesh, element)
    elif space.element.number_of_local_dofs != 3:
        raise ValueError(
            "create_patch_test_space requires a P1 (3-local-dof) "
            "FiniteElementSpace: vertex-star patches use the P1 nodal basis "
            f"as their hat function, got an element with "
            f"{space.element.number_of_local_dofs} local DOFs."
        )
    quadrature = space.element.reference_cell.create_quadrature(quadrature_degree)
    basis = create_cell_basis(space, quadrature)

    tags = (
        dirichlet_boundary_tags
        if dirichlet_boundary_tags is not None
        else tuple(mesh.boundary_tag_names.values())
    )
    boundary_dofs = find_boundary_dofs(space, tags)
    is_boundary = (
        jnp.zeros((space.number_of_dofs,), dtype=bool).at[boundary_dofs].set(True)
    )
    interior_vertices = jnp.nonzero(~is_boundary)[0]

    cells_to_vertices_np = np.asarray(mesh.cells_to_vertices)
    interior_vertices_np = np.asarray(interior_vertices)
    patch_cells_np, patch_local_dof_np, patch_mask_np = _build_vertex_incidences(
        cells_to_vertices_np, interior_vertices_np
    )
    diameters_np = _patch_diameters(
        cells_to_vertices_np,
        np.asarray(mesh.vertex_coordinates[:, 0, :]),
        interior_vertices_np,
        patch_cells_np,
        patch_mask_np,
    )

    cover_lower, cover_upper = _cover_constants(basis, space.cells_to_dofs, ~is_boundary)

    return PatchTestSpace(
        mesh=mesh,
        space=space,
        basis=basis,
        interior_vertices=interior_vertices,
        number_of_test_functions=int(interior_vertices.shape[0]),
        patch_cells=jnp.asarray(patch_cells_np, dtype=jnp.int32),
        patch_local_dof=jnp.asarray(patch_local_dof_np, dtype=jnp.int32),
        patch_mask=jnp.asarray(patch_mask_np),
        patch_diameter=jnp.asarray(diameters_np),
        cover_constant_lower=cover_lower,
        cover_constant_upper=cover_upper,
    )


class PatchQuadrature(NamedTuple):
    """Per-patch quadrature data, gathered from ``PatchTestSpace.basis``.

    Every array's first two axes are ``(N_patch, max_valence)``; padding
    slots (``~mask``) carry a zero ``physical_weights`` entry, so summing
    over ``(max_valence, Q)`` without further masking is safe wherever a
    finite (possibly garbage-at-padding) integrand is multiplied against
    ``physical_weights`` -- exactly the ``integrate_cellwise`` convention
    used elsewhere in jax-fem.

    Attributes
    ----------
    physical_points : jax.Array
        Shape ``(N_patch, max_valence, Q, 1, 2)``.
    physical_weights : jax.Array
        Shape ``(N_patch, max_valence, Q, 1, 1)``, zero at padding.
    hat_values : jax.Array
        ``varphi_i`` at each point, shape ``(N_patch, max_valence, Q, 1, 1)``.
    hat_gradients : jax.Array
        ``grad varphi_i``, constant per incident cell (P1), shape
        ``(N_patch, max_valence, 1, 1, 2)``.
    mask : jax.Array
        Shape ``(N_patch, max_valence)``.
    """

    physical_points: jax.Array
    physical_weights: jax.Array
    hat_values: jax.Array
    hat_gradients: jax.Array
    mask: jax.Array


def gather_patch_quadrature(patch_space: PatchTestSpace) -> PatchQuadrature:
    """Gather ``patch_space.basis`` onto every patch's incident-cell quadrature points."""
    basis = patch_space.basis
    mask = patch_space.patch_mask
    safe_cell = jnp.where(mask, patch_space.patch_cells, 0)
    safe_local_dof = jnp.where(mask, patch_space.patch_local_dof, 0)

    physical_points = basis.physical_points[safe_cell]  # (N_patch, max_valence, Q, 1, 2)
    physical_weights = basis.physical_weights[safe_cell] * mask[:, :, None, None, None]

    reference_values = basis.values[0]  # (Q, 3, 1), cell-independent
    hat_values = reference_values[:, safe_local_dof, :]  # (Q, N_patch, max_valence, 1)
    hat_values = jnp.moveaxis(hat_values, 0, 2)  # (N_patch, max_valence, Q, 1)
    hat_values = hat_values[..., None, :] * mask[:, :, None, None, None]

    cell_gradients = basis.gradients[:, 0, :, :]  # (K, 3, 2), constant over Q for P1
    hat_gradients = cell_gradients[safe_cell, safe_local_dof, :]  # (N_patch, max_valence, 2)
    hat_gradients = hat_gradients[:, :, None, None, :] * mask[:, :, None, None, None]

    return PatchQuadrature(
        physical_points=physical_points,
        physical_weights=physical_weights,
        hat_values=hat_values,
        hat_gradients=hat_gradients,
        mask=mask,
    )
