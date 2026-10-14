"""Shared fixtures: a small mesh, problems, and FE test spaces."""

import jax.numpy as jnp
import jax.random as jr
import pytest
from jax_fem.element import LagrangeTriangleP1
from jax_fem.mesh import TriangleMesh, create_gmsh_unit_square_mesh
from jax_fem.problem import (
    EllipticProblem,
    create_poisson_sin_sin_problem,
    create_simple_elliptic_problem,
    evaluate_elliptic_coefficients,
)
from jax_fem.reference_cell import ReferenceTriangle
from jax_fem.space import CellBasis, create_cell_basis, create_finite_element_space

from vpinns_jax.test_spaces import FETestSpace, create_fe_test_space


def pytest_configure() -> None:
    import matplotlib

    matplotlib.use("Agg")


@pytest.fixture
def prng_key() -> jnp.ndarray:
    return jr.PRNGKey(0)


@pytest.fixture(scope="session")
def unit_square_mesh() -> TriangleMesh:
    return create_gmsh_unit_square_mesh(4)


@pytest.fixture(scope="session")
def p1_basis(unit_square_mesh: TriangleMesh) -> CellBasis:
    element = LagrangeTriangleP1(reference_cell=ReferenceTriangle())
    space = create_finite_element_space(unit_square_mesh, element)
    return create_cell_basis(space, element.reference_cell.create_quadrature(4))


@pytest.fixture(scope="session")
def sin_sin_problem(unit_square_mesh: TriangleMesh) -> EllipticProblem:
    return create_poisson_sin_sin_problem(unit_square_mesh)


@pytest.fixture(scope="session")
def advection_problem(unit_square_mesh: TriangleMesh) -> EllipticProblem:
    return create_simple_elliptic_problem(unit_square_mesh)


def _fe_test_space(basis: CellBasis, problem: EllipticProblem) -> FETestSpace:
    coefficients = evaluate_elliptic_coefficients(problem, basis.physical_points)
    return create_fe_test_space(basis, coefficients, problem)


@pytest.fixture(scope="session")
def poisson_fe(p1_basis: CellBasis, sin_sin_problem: EllipticProblem) -> FETestSpace:
    return _fe_test_space(p1_basis, sin_sin_problem)


@pytest.fixture(scope="session")
def advection_fe(
    p1_basis: CellBasis, advection_problem: EllipticProblem
) -> FETestSpace:
    return _fe_test_space(p1_basis, advection_problem)
