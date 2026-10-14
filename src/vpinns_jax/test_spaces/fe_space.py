"""
Conforming finite element test spaces.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
from jax_fem.assembly import (
    GlobalMatrix,
    assemble_matrix,
    assemble_vector,
    integrate_cellwise,
)
from jax_fem.constraints import evaluate_dirichlet_dof_values
from jax_fem.forms import (
    elliptic_bilinear_form,
    elliptic_energy_bilinear_form,
    elliptic_linear_form,
)
from jax_fem.problem import EllipticCoefficientValues, EllipticProblem
from jax_fem.space import CellBasis


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class FETestSpace:
    """
    Conforming finite element test space with its assembled operators.

    Attributes
    ----------
    basis : CellBasis
        Evaluated basis of the space; its quadrature is used for every
        residual.
    coefficients : EllipticCoefficientValues
        Coefficients at ``basis.physical_points``.
    dirichlet_dofs : jax.Array
        Dirichlet DOF indices in increasing order, shape ``(N_dirichlet,)``.
    dirichlet_values : jax.Array
        Prescribed Dirichlet values, shape ``(N_dirichlet, 1)``.
    free_dofs : jax.Array
        Remaining DOF indices in increasing order, shape ``(N_free,)``.
    stiffness : GlobalMatrix
        Matrix of the elliptic bilinear form, shape ``(dofs_test, dofs_test)``.
    load : jax.Array
        Vector of the elliptic linear form, shape ``(dofs_test, 1)``.
    gram : GlobalMatrix
        Matrix of the energy inner product, shape ``(dofs_test, dofs_test)``.

    Notes
    -----
    The test functions are the basis functions of the free DOFs, a basis of
    the discrete subspace of ``H^1_0``. Residual vectors are assembled over
    all DOFs and restricted to ``free_dofs`` by the caller.

    On a discrete fracture network the basis is embedded and the diffusion is
    the per-cell fracture transmissivity. Trace vertices are shared, so a
    finite element trial is continuous across traces and may kink there, as
    the exact head does. A direct network trial that is continuously
    differentiable inside each fracture cannot: its conormal fluxes from the
    two halves of a fracture cancel at an X-crossing, so it exchanges no flux
    between fractures there. The interpolated trial avoids this on a fixed
    mesh, since the network only has to match nodal values.
    """

    basis: CellBasis
    coefficients: EllipticCoefficientValues
    dirichlet_dofs: jax.Array
    dirichlet_values: jax.Array
    free_dofs: jax.Array
    stiffness: GlobalMatrix
    load: jax.Array
    gram: GlobalMatrix

    def __post_init__(self) -> None:
        """
        Validate the array shapes.

        Raises
        ------
        ValueError
            If an operator does not match the number of DOFs, or the Dirichlet
            and free DOFs do not partition the DOFs.
        """
        number_of_dofs = self.basis.space.number_of_dofs
        square = (number_of_dofs, number_of_dofs)
        if self.stiffness.shape != square or self.gram.shape != square:
            raise ValueError(
                f"stiffness and gram must have shape {square}, got "
                f"{self.stiffness.shape} and {self.gram.shape}."
            )
        if self.load.shape != (number_of_dofs, 1):
            raise ValueError(
                f"load must have shape {(number_of_dofs, 1)}, got {self.load.shape}."
            )
        number_of_dirichlet_dofs = self.dirichlet_dofs.shape[-1]
        if self.dirichlet_values.shape != (number_of_dirichlet_dofs, 1):
            raise ValueError(
                "dirichlet_values must have shape "
                f"{(number_of_dirichlet_dofs, 1)}, got {self.dirichlet_values.shape}."
            )
        if number_of_dirichlet_dofs + self.free_dofs.shape[-1] != number_of_dofs:
            raise ValueError(
                "dirichlet_dofs and free_dofs must partition the "
                f"{number_of_dofs} DOFs."
            )


def create_fe_test_space(
    basis: CellBasis,
    coefficients: EllipticCoefficientValues,
    problem: EllipticProblem,
) -> FETestSpace:
    """
    Assemble the operators of a finite element test space.

    Parameters
    ----------
    basis : CellBasis
        Evaluated basis of the space, planar or embedded.
    coefficients : EllipticCoefficientValues
        Coefficients at ``basis.physical_points``.
    problem : EllipticProblem
        Problem providing the Dirichlet conditions.

    Returns
    -------
    fe_test_space : FETestSpace
        The test space with its stiffness matrix, load vector, energy Gram
        matrix, and DOF classification.

    Notes
    -----
    The basis and coefficients are built by the caller, so the same
    constructor serves planar bases (``create_cell_basis``), embedded bases
    (``create_embedded_cell_basis``), and coefficients whose diffusion was
    replaced, e.g. by a per-cell fracture transmissivity. The DOF
    classification has a data-dependent size, so this function is eager.
    """
    space = basis.space
    weights = basis.physical_weights
    stiffness = assemble_matrix(
        integrate_cellwise(elliptic_bilinear_form(basis, basis, coefficients), weights),
        space,
        space,
    )
    load = assemble_vector(
        integrate_cellwise(elliptic_linear_form(basis, coefficients), weights), space
    )
    gram = assemble_matrix(
        integrate_cellwise(
            elliptic_energy_bilinear_form(basis, basis, coefficients), weights
        ),
        space,
        space,
    )
    dirichlet_dofs, dirichlet_values = evaluate_dirichlet_dof_values(problem, space)
    is_dirichlet = (
        jnp.zeros((space.number_of_dofs,), dtype=bool).at[dirichlet_dofs].set(True)
    )
    return FETestSpace(
        basis=basis,
        coefficients=coefficients,
        dirichlet_dofs=dirichlet_dofs,
        dirichlet_values=dirichlet_values,
        free_dofs=jnp.flatnonzero(~is_dirichlet),
        stiffness=stiffness,
        load=load,
        gram=gram,
    )
