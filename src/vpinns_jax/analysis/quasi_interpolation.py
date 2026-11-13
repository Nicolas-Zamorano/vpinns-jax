"""
Operator norm of the quasi-interpolant of a patch family.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import scipy.spatial
from jax_fem.assembly import assemble_matrix, integrate_cellwise
from jax_fem.element import LagrangeTriangleP1
from jax_fem.forms import elliptic_energy_bilinear_form
from jax_fem.problem import EllipticProblem, evaluate_elliptic_coefficients
from jax_fem.reference_cell import ReferenceTriangle
from jax_fem.space import create_cell_basis, create_finite_element_space

from vpinns_jax.losses import restrict_to_free
from vpinns_jax.test_spaces.crisscross_mesh import create_crisscross_unit_square_mesh
from vpinns_jax.test_spaces.meshfree_space import (
    MeshfreePatchFamily,
    create_meshfree_patch_test_space,
)
from vpinns_jax.test_spaces.poincare_constants import (
    assemble_weighted_p1_matrices,
    to_scipy_matrix,
)

# Number of right-hand sides solved at once with the fine stiffness matrix.
_SOLVE_CHUNK = 64


def _nodal_weights(
    family: MeshfreePatchFamily, patches: np.ndarray, nodes: np.ndarray
) -> scipy.sparse.csr_array:
    """
    Raw weights of some patches at the nodes of a mesh.

    Parameters
    ----------
    family : MeshfreePatchFamily
        Patch family.
    patches : np.ndarray
        Indices of the patches, shape ``(N_p,)``.
    nodes : np.ndarray
        Node coordinates, shape ``(N_nodes, 2)``.

    Returns
    -------
    weights : scipy.sparse.csr_array
        ``phi_i`` of the ``i``-th listed patch at node ``a`` in entry
        ``(a, i)``, shape ``(N_nodes, N_p)``.

    Notes
    -----
    A patch contains its centre in its closure, so it lies in the disk of
    radius ``h_i`` around it; the weight is evaluated at the nodes of that
    disk only.
    """
    centres = np.asarray(family.centres)[patches]
    radii = np.asarray(family.patch_diameters)[patches]
    neighbourhoods = scipy.spatial.cKDTree(nodes).query_ball_point(centres, radii)
    columns = np.repeat(np.arange(patches.size), [len(n) for n in neighbourhoods])
    rows = np.concatenate([np.asarray(n, dtype=int) for n in neighbourhoods])
    values = jnp.vectorize(family.weight, signature="(d),(d),()->()")(
        jnp.asarray(nodes[rows]),
        jnp.take(family.centres, jnp.asarray(patches[columns]), axis=-2),
        jnp.take(family.kinds, jnp.asarray(patches[columns]), axis=-1),
    )
    return scipy.sparse.csr_array(
        (np.asarray(values), (rows, columns)), shape=(nodes.shape[0], patches.size)
    )


def compute_quasi_interpolant_norm(
    family: MeshfreePatchFamily,
    problem: EllipticProblem,
    cells_per_side: int,
    *,
    normalise: bool,
    refinement: int = 8,
    quadrature_degree: int = 8,
) -> float:
    """
    Discrete operator norm of the quasi-interpolant of a patch family.

    Parameters
    ----------
    family : MeshfreePatchFamily
        Patch family of the unit square whose raw weights are affine on the
        triangles of the criss-cross mesh with ``cells_per_side`` cells per
        side, such as the ``l^inf`` and the criss-cross families.
    problem : EllipticProblem
        Problem providing the coefficients of the energy norm ``|v|_A``.
    cells_per_side : int
        Number of grid cells per side ``n`` of the family.
    normalise : bool
        Interpolate onto the normalised weights ``psi_i = phi_i / S``, the
        operator ``Pi_psi``, or onto the raw weights, the operator ``Pi_P``
        of a family whose weights sum to one.
    refinement : int
        Number of cells of the fine mesh per grid cell.
    quadrature_degree : int
        Exactness degree of the rule on the local triangles of the Gram
        matrix of the test functions.

    Returns
    -------
    norm : float
        ``||Pi||_h = sup |Pi v|_A / |v|_A`` over the continuous piecewise
        linear functions on the criss-cross mesh with ``n r`` cells per side
        that vanish on the boundary.

    Notes
    -----
    The quasi-interpolant is ``Pi v = sum_i v_i psi_i`` over the interior
    patches, with the weighted means ``v_i = (phi_i, v) / (phi_i, 1)``. The
    fine mesh refines the triangulation on which the raw weights are
    affine, so each ``phi_i`` is a fine P1 function and ``(phi_i, v) =
    Phi_i^T M v`` with its nodal values ``Phi_i`` and the fine mass matrix
    ``M``; the weighted means are exact. With the matrix ``W`` of the means,
    the energy Gram matrix ``G = L L^T`` of the interior test functions,
    and the fine stiffness matrix ``K`` on the free nodes,
    ``|Pi v|_A^2 = (W v)^T G (W v)``, so ``||Pi||_h^2`` is the largest
    eigenvalue of ``L^T W K^(-1) W^T L``. The supremum over a subspace of
    ``H^1_0`` bounds ``||Pi||`` from below and increases under refinement.
    """
    mesh = create_crisscross_unit_square_mesh(cells_per_side * refinement)
    element = LagrangeTriangleP1(reference_cell=ReferenceTriangle())
    basis = create_cell_basis(
        create_finite_element_space(mesh, element),
        element.reference_cell.create_quadrature(4),
    )
    space = basis.space
    coefficients = evaluate_elliptic_coefficients(problem, basis.physical_points)
    stiffness = to_scipy_matrix(
        assemble_matrix(
            integrate_cellwise(
                elliptic_energy_bilinear_form(basis, basis, coefficients),
                basis.physical_weights,
            ),
            space,
            space,
        )
    )
    _, mass = assemble_weighted_p1_matrices(
        basis, jnp.ones_like(basis.physical_weights)
    )

    nodes = np.reshape(np.asarray(space.dof_coordinates), (-1, 2))
    boundary = np.any(np.isclose(nodes, 0.0) | np.isclose(nodes, 1.0), axis=-1)
    free = np.flatnonzero(~boundary)
    interior = np.asarray(family.interior_patches)
    weighted_masses = (mass @ _nodal_weights(family, interior, nodes)).T
    means = scipy.sparse.csr_array(
        weighted_masses / np.expand_dims(weighted_masses.sum(axis=-1), axis=-1)
    )[:, free]

    test_space = create_meshfree_patch_test_space(
        family, problem, quadrature_degree=quadrature_degree, normalise=normalise
    )
    gram = np.asarray(restrict_to_free(test_space.gram, family.interior_patches))
    factor = np.linalg.cholesky(0.5 * (gram + gram.T))

    solver = scipy.sparse.linalg.splu(scipy.sparse.csc_array(stiffness[free][:, free]))
    transposed_means = scipy.sparse.csc_array(means.T)
    projected = np.empty((interior.size, interior.size))
    for start in range(0, interior.size, _SOLVE_CHUNK):
        stop = min(start + _SOLVE_CHUNK, interior.size)
        solutions = solver.solve(transposed_means[:, start:stop].toarray())
        projected[:, start:stop] = means @ solutions
    reduced = factor.T @ (0.5 * (projected + projected.T)) @ factor
    largest = scipy.linalg.eigvalsh(reduced, subset_by_index=[interior.size - 1] * 2)
    return float(np.sqrt(largest[0]))
