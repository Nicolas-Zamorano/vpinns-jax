"""
Shepard-normalised patch test spaces.
"""

from __future__ import annotations

import dataclasses
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental import sparse as jax_sparse
from jax_fem.assembly import GlobalMatrix, integrate_cellwise
from jax_fem.element import LagrangeTriangleP1
from jax_fem.forms import elliptic_energy_bilinear_form
from jax_fem.problem import (
    EllipticCoefficientValues,
    EllipticProblem,
    evaluate_elliptic_coefficients,
)
from jax_fem.reference_cell import ReferenceTriangle
from jax_fem.space import CellBasis, create_cell_basis, create_finite_element_space

from vpinns_jax.test_spaces.patch_family import PatchFamily


class PatchFunctions(NamedTuple):
    """
    Patch functions evaluated at the quadrature points of the integration mesh.

    Attributes
    ----------
    values : jax.Array
        Values of the functions listed on each cell, zero at padding, shape
        ``(K, Q, M, 1)``.
    gradients : jax.Array
        Their gradients, zero at padding, shape ``(K, Q, M, d)``, or
        ``(K, 1, M, d)`` when constant on each cell.

    Notes
    -----
    The layout is that of an evaluated basis with ``M`` local functions, so
    the ``jax_fem`` forms, which read only ``values`` and ``gradients``,
    accept it as a test or trial basis.
    """

    values: jax.Array
    gradients: jax.Array


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class PatchTestSpace:
    """
    Patch test space with its energy Gram matrix.

    Attributes
    ----------
    family : PatchFamily
        Patches, raw weights, and integration mesh.
    basis : CellBasis
        P1 basis of the integration mesh; its quadrature is used for every
        patch integral.
    coefficients : EllipticCoefficientValues
        Coefficients at ``basis.physical_points``.
    test_functions : PatchFunctions
        Test functions ``psi_i``, by default the normalised weights.
    raw_weights : PatchFunctions
        Raw weights ``phi_i``.
    gram : GlobalMatrix
        Energy Gram matrix ``(psi_i, psi_j)_A`` of all patches, shape
        ``(N_patch, N_patch)``.

    Notes
    -----
    With ``S = sum_i phi_i``, the normalised weights ``psi_i = phi_i / S``
    form a partition of unity, ``sum_i psi_i = 1`` and
    ``sum_i grad psi_i = 0``. The test space is the span of the ``psi_i`` of
    the interior patches, a subspace of ``H^1_0``; residual vectors and the
    Gram matrix are assembled over all patches and restricted to
    ``family.interior_patches`` by the caller. The squared dual norm of the
    residual over this space, ``r^T P^{-1} r`` with ``P`` the restricted Gram
    matrix, is efficient with constant one, and together with the residual
    terms ``h_i^2 ||phi_i^(1/2) rho(w)||^2`` it bounds the energy error from
    above with constants independent of the mesh size; without normalisation
    the bound degenerates under refinement.
    """

    family: PatchFamily
    basis: CellBasis
    coefficients: EllipticCoefficientValues
    test_functions: PatchFunctions
    raw_weights: PatchFunctions
    gram: GlobalMatrix

    def __post_init__(self) -> None:
        """
        Validate the array shapes.

        Raises
        ------
        ValueError
            If the patch functions or the Gram matrix do not match the family
            and the basis.
        """
        number_of_cells, number_of_points = self.basis.physical_points.shape[-4:-2]
        maximum_patches = self.family.cells_to_patches.shape[-1]
        expected_values = (number_of_cells, number_of_points, maximum_patches, 1)
        for name, functions in (
            ("test_functions", self.test_functions),
            ("raw_weights", self.raw_weights),
        ):
            if functions.values.shape != expected_values:
                raise ValueError(
                    f"{name}.values must have shape {expected_values}, got "
                    f"{functions.values.shape}."
                )
        number_of_patches = self.family.number_of_patches
        if self.gram.shape != (number_of_patches, number_of_patches):
            raise ValueError(
                f"gram must have shape {(number_of_patches, number_of_patches)}, "
                f"got {self.gram.shape}."
            )


def assemble_patch_vector(local_vectors: jax.Array, family: PatchFamily) -> jax.Array:
    """
    Assemble local vectors over the patches listed on each cell.

    Parameters
    ----------
    local_vectors : jax.Array
        Integrated local vectors, shape ``(..., K, 1, M, 1)``.
    family : PatchFamily
        Family whose ``cells_to_patches`` maps local to global indices.

    Returns
    -------
    global_vector : jax.Array
        Assembled vector, shape ``(..., N_patch, 1)``.

    Notes
    -----
    This is ``jax_fem.assembly.assemble_vector`` with ``cells_to_patches`` as
    the DOF map. Padding entries add their zero values to patch ``0``.
    """
    global_vector = jnp.zeros(
        local_vectors.shape[:-4] + (family.number_of_patches, 1),
        dtype=local_vectors.dtype,
    )
    return global_vector.at[..., family.cells_to_patches, :].add(
        jnp.squeeze(local_vectors, axis=-3)
    )


def assemble_patch_matrix(
    local_matrices: jax.Array, family: PatchFamily
) -> GlobalMatrix:
    """
    Assemble local matrices over the patches listed on each cell.

    Parameters
    ----------
    local_matrices : jax.Array
        Integrated local matrices, shape ``(K, 1, M, M)``.
    family : PatchFamily
        Family whose ``cells_to_patches`` maps local to global indices.

    Returns
    -------
    global_matrix : GlobalMatrix
        Assembled sparse matrix with uncompacted duplicates, shape
        ``(N_patch, N_patch)``.

    Notes
    -----
    This is ``jax_fem.assembly.assemble_matrix`` with ``cells_to_patches`` as
    the DOF map. Padding entries add zeros at patch ``0``.
    """
    rows = jnp.broadcast_to(
        jnp.expand_dims(family.cells_to_patches, axis=(-3, -1)), local_matrices.shape
    )
    columns = jnp.broadcast_to(
        jnp.expand_dims(family.cells_to_patches, axis=(-3, -2)), local_matrices.shape
    )
    indices = jnp.stack((jnp.ravel(rows), jnp.ravel(columns)), axis=-1)
    return jax_sparse.BCOO(
        (jnp.ravel(local_matrices), indices),
        shape=(family.number_of_patches, family.number_of_patches),
    )


def create_patch_test_space(
    family: PatchFamily,
    problem: EllipticProblem,
    *,
    quadrature_degree: int = 5,
    normalise: bool = True,
) -> PatchTestSpace:
    """
    Evaluate the patch functions and assemble their energy Gram matrix.

    Parameters
    ----------
    family : PatchFamily
        Patches, raw weights, and integration mesh.
    problem : EllipticProblem
        Problem providing the coefficients.
    quadrature_degree : int
        Exactness degree of the cell quadrature on the integration mesh.
    normalise : bool
        Use the normalised weights ``psi_i = phi_i / S`` as test functions;
        if ``False``, use the raw weights.

    Returns
    -------
    patch_test_space : PatchTestSpace
        The test space.

    Notes
    -----
    The raw weights are P1 on every cell, so their values and gradients come
    from the P1 basis. The normalised weights follow from the quotient rule,
    ``grad psi_i = grad phi_i / S - phi_i grad S / S^2``, and are rational on
    each cell, smooth since every kink lies on a cell edge; the quadrature is
    exact for the raw weights only. The raw weights as test functions do not
    give a uniformly reliable estimate and are meant for comparisons.
    """
    element = LagrangeTriangleP1(reference_cell=ReferenceTriangle())
    space = create_finite_element_space(family.mesh, element)
    basis = create_cell_basis(
        space, element.reference_cell.create_quadrature(quadrature_degree)
    )
    raw_weights = PatchFunctions(
        values=family.local_weights @ basis.values,
        gradients=family.local_weights @ basis.gradients,
    )
    if normalise:
        sums = jnp.sum(raw_weights.values, axis=-2, keepdims=True)
        sum_gradients = jnp.sum(raw_weights.gradients, axis=-2, keepdims=True)
        test_functions = PatchFunctions(
            values=raw_weights.values / sums,
            gradients=raw_weights.gradients / sums
            - raw_weights.values * sum_gradients / sums**2,
        )
    else:
        test_functions = raw_weights

    coefficients = evaluate_elliptic_coefficients(problem, basis.physical_points)
    gram = assemble_patch_matrix(
        integrate_cellwise(
            elliptic_energy_bilinear_form(test_functions, test_functions, coefficients),
            basis.physical_weights,
        ),
        family,
    )
    return PatchTestSpace(
        family=family,
        basis=basis,
        coefficients=coefficients,
        test_functions=test_functions,
        raw_weights=raw_weights,
        gram=gram,
    )
