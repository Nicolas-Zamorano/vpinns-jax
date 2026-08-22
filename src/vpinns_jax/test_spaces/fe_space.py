"""Finite-element test space ``V_h``, a thin bundle over jax-fem's own FE machinery.

Wraps ``jax_fem``'s ``FiniteElementSpace``/``CellBasis`` and pre-assembles
the global stiffness matrix and load vector (``elliptic_bilinear_form`` /
``elliptic_linear_form``), so ``losses.residual``'s interpolated-trial fast
path (``r = stiffness @ w_dofs - load``) and direct-trial path (needs
``basis``/``coefficients``) both have what they need without recomputing it.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
from jax_fem.assembly import GlobalMatrix, assemble_bilinear_form, assemble_linear_form
from jax_fem.constraints import evaluate_dirichlet_dof_values
from jax_fem.element import FiniteElement, LagrangeTriangleP1
from jax_fem.forms import elliptic_bilinear_form, elliptic_linear_form
from jax_fem.mesh import TriangleMesh
from jax_fem.problem import EllipticCoefficientValues, evaluate_elliptic_coefficients
from jax_fem.reference_cell import ReferenceTriangle
from jax_fem.space import (
    CellBasis,
    FiniteElementSpace,
    create_cell_basis,
    create_finite_element_space,
)

from vpinns_jax.pde.base import EllipticPDE

__all__ = ["FETestSpace", "create_fe_test_space"]


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class FETestSpace:
    """FE test space bundle for one ``(mesh, element, problem, quadrature)`` combination.

    Attributes
    ----------
    mesh, element, space, basis :
        The underlying jax-fem mesh/element/space/quadrature-evaluated basis.
    coefficients : EllipticCoefficientValues
        Elliptic coefficients evaluated at ``basis.physical_points``.
    free_dofs, dirichlet_dofs, dirichlet_values :
        DOF classification; ``I_test = free_dofs`` is the interior test
        space (``V_h subset H^1_0``).
    stiffness, load :
        Assembled global operators. The interpolated-trial residual is
        ``r = stiffness @ w_dofs - load`` (``losses.residual``).
    number_of_test_functions : int
        ``len(free_dofs)``.
    """

    mesh: TriangleMesh
    element: FiniteElement
    space: FiniteElementSpace
    basis: CellBasis
    coefficients: EllipticCoefficientValues
    free_dofs: jax.Array
    dirichlet_dofs: jax.Array
    dirichlet_values: jax.Array
    stiffness: GlobalMatrix
    load: jax.Array
    number_of_test_functions: int


def create_fe_test_space(
    mesh: TriangleMesh,
    problem: EllipticPDE,
    *,
    element: FiniteElement | None = None,
    quadrature_degree: int = 4,
) -> FETestSpace:
    """Build the FE test space for ``problem`` on ``mesh``.

    Parameters
    ----------
    mesh : TriangleMesh
    problem : EllipticPDE
    element : FiniteElement | None
        Defaults to ``LagrangeTriangleP1``.
    quadrature_degree : int
        Cell quadrature exactness degree.

    Returns
    -------
    FETestSpace
    """
    element = element if element is not None else LagrangeTriangleP1(ReferenceTriangle())
    space = create_finite_element_space(mesh, element)
    quadrature = element.reference_cell.create_quadrature(quadrature_degree)
    basis = create_cell_basis(space, quadrature)
    coefficients = evaluate_elliptic_coefficients(problem, basis.physical_points)

    stiffness = assemble_bilinear_form(elliptic_bilinear_form, basis, coefficients)
    load = assemble_linear_form(elliptic_linear_form, basis, coefficients)

    dirichlet_dofs, dirichlet_values = evaluate_dirichlet_dof_values(problem, space)
    is_dirichlet = (
        jnp.zeros((space.number_of_dofs,), dtype=bool).at[dirichlet_dofs].set(True)
    )
    free_dofs = jnp.nonzero(~is_dirichlet)[0]

    return FETestSpace(
        mesh=mesh,
        element=element,
        space=space,
        basis=basis,
        coefficients=coefficients,
        free_dofs=free_dofs,
        dirichlet_dofs=dirichlet_dofs,
        dirichlet_values=dirichlet_values,
        stiffness=stiffness,
        load=load,
        number_of_test_functions=int(free_dofs.shape[0]),
    )
