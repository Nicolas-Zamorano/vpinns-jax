"""Tests for the interpolated trial and the FE test space on a fracture network."""

import jax.numpy as jnp
import pytest
from jax_fem.constraints import condense_dirichlet_system, expand_condensed_solution
from jax_fem.diagnostics import (
    compute_h1_seminorm_error,
    energy_error_functional,
    h1_seminorm_error_functional,
)
from jax_fem.element import LagrangeTriangleP1
from jax_fem.function import FiniteElementFunction
from jax_fem.mesh import create_dfn_two_fractures_mesh
from jax_fem.problem import (
    create_dfn_two_fractures_problem,
    evaluate_elliptic_coefficients,
    evaluate_fracture_transmissivity,
)
from jax_fem.reference_cell import ReferenceTriangle
from jax_fem.space import create_embedded_cell_basis, create_finite_element_space

from vpinns_jax.analysis import (
    compute_solution_norm,
    plot_network_solution,
)
from vpinns_jax.losses import (
    compute_interpolated_fe_residual,
    restrict_to_free,
    riesz_loss,
)
from vpinns_jax.models import (
    MultiLayerPerceptron,
    evaluate_network_values,
    interpolate_network,
)
from vpinns_jax.test_spaces import FETestSpace, create_fe_test_space


@pytest.fixture(scope="module")
def dfn():
    mesh = create_dfn_two_fractures_mesh(4)
    problem = create_dfn_two_fractures_problem(mesh)
    element = LagrangeTriangleP1(reference_cell=ReferenceTriangle())
    space = create_finite_element_space(mesh, element)
    basis = create_embedded_cell_basis(
        space, element.reference_cell.create_quadrature(4)
    )
    coefficients = evaluate_elliptic_coefficients(problem, basis.physical_points)
    transmissivity = evaluate_fracture_transmissivity(mesh, jnp.ones(2))
    fe = create_fe_test_space(
        basis, coefficients._replace(diffusion=transmissivity), problem
    )
    return mesh, problem, fe, create_fe_test_space(basis, coefficients, problem)


def _galerkin_dof_values(fe: FETestSpace) -> jnp.ndarray:
    system = condense_dirichlet_system(
        fe.stiffness, fe.load, fe.dirichlet_dofs, fe.dirichlet_values
    )
    free_values = jnp.linalg.solve(system.matrix.todense(), system.vector)
    return expand_condensed_solution(free_values, system)


def test_transmissivity_gives_the_operators_of_the_isotropic_problem(dfn) -> None:
    """With unit transmissivity, k (I - n n^T) acts like I on tangential gradients."""
    _, _, fe, isotropic_fe = dfn
    assert jnp.allclose(fe.stiffness.todense(), isotropic_fe.stiffness.todense())
    assert jnp.allclose(fe.gram.todense(), isotropic_fe.gram.todense())


def test_interpolate_network_on_a_fracture_network(dfn, prng_key) -> None:
    mesh, _, fe, _ = dfn
    space = fe.basis.space
    network = MultiLayerPerceptron(input_dimension=3, width=8, depth=2, key=prng_key)
    interpolant = interpolate_network(network, fe)
    network_values = jnp.squeeze(
        evaluate_network_values(network, space.dof_coordinates), axis=-2
    )

    # One DOF per vertex: the trace vertices are shared by both fractures.
    assert space.number_of_dofs == mesh.vertex_coordinates.shape[0]
    assert interpolant.dof_values.shape == (space.number_of_dofs, 1)
    assert jnp.allclose(interpolant.dof_values[fe.dirichlet_dofs], fe.dirichlet_values)
    assert jnp.allclose(
        interpolant.dof_values[fe.free_dofs], network_values[fe.free_dofs]
    )


def test_galerkin_solution_has_zero_free_residual(dfn) -> None:
    _, _, fe, _ = dfn
    residual = compute_interpolated_fe_residual(fe, _galerkin_dof_values(fe))
    assert jnp.allclose(jnp.take(residual, fe.free_dofs, axis=-2), 0.0, atol=1e-12)


def test_riesz_loss_is_the_energy_distance_to_galerkin(dfn, prng_key) -> None:
    _, _, fe, _ = dfn
    network = MultiLayerPerceptron(input_dimension=3, width=8, depth=2, key=prng_key)
    dof_values = interpolate_network(network, fe).dof_values
    difference = dof_values - _galerkin_dof_values(fe)
    factor = jnp.linalg.cholesky(restrict_to_free(fe.gram, fe.free_dofs))
    residual = jnp.take(
        compute_interpolated_fe_residual(fe, dof_values), fe.free_dofs, axis=-2
    )

    assert jnp.allclose(
        riesz_loss(residual, factor), jnp.sum(difference * (fe.stiffness @ difference))
    )


def test_solution_norm_on_embedded_cells(dfn) -> None:
    _, problem, fe, _ = dfn
    basis = fe.basis
    zero = FiniteElementFunction(
        space=basis.space, dof_values=jnp.zeros((basis.space.number_of_dofs, 1))
    )
    assert jnp.allclose(
        compute_solution_norm(basis, problem, h1_seminorm_error_functional),
        compute_h1_seminorm_error(zero, basis, problem),
    )
    assert compute_solution_norm(basis, problem, energy_error_functional) > 0.0


def test_field_plots_reject_fracture_networks(dfn) -> None:
    mesh, problem, _, _ = dfn

    def zero(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros((), dtype=x.dtype)

    with pytest.raises(NotImplementedError, match="planar"):
        plot_network_solution(zero, mesh, problem)
