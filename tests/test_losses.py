"""Tests for the residual vectors and the Euclidean, Riesz, and boundary losses."""

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest
from jax_fem.constraints import condense_dirichlet_system, expand_condensed_solution
from jax_fem.mesh import create_gmsh_unit_square_mesh

from vpinns_jax.losses import (
    boundary_residual_loss,
    compute_fe_residual,
    compute_interpolated_fe_residual,
    diagonal_riesz_loss,
    euclidean_loss,
    restrict_to_free,
    riesz_loss,
)
from vpinns_jax.models import (
    ConstrainedAnsatz,
    ExactDistanceFunction,
    MultiLayerPerceptron,
    evaluate_network,
    evaluate_network_values,
    interpolate_network,
)
from vpinns_jax.test_spaces import FETestSpace


def affine_function(x: jnp.ndarray) -> jnp.ndarray:
    return 1.0 + 2.0 * x[0] - 0.5 * x[1]


def sin_sin_solution(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.sin(jnp.pi * x[0]) * jnp.sin(jnp.pi * x[1])


def _nodal_values(function, fe: FETestSpace) -> jnp.ndarray:
    values = evaluate_network_values(function, fe.basis.space.dof_coordinates)
    return jnp.squeeze(values, axis=-2)


def _galerkin_solution(fe: FETestSpace) -> jnp.ndarray:
    system = condense_dirichlet_system(
        fe.stiffness, fe.load, fe.dirichlet_dofs, fe.dirichlet_values
    )
    free_values = jnp.linalg.solve(system.matrix.todense(), system.vector)
    return expand_condensed_solution(free_values, system)


def _free(vector: jnp.ndarray, fe: FETestSpace) -> jnp.ndarray:
    return jnp.take(vector, fe.free_dofs, axis=-2)


@pytest.mark.parametrize("fe_name", ["poisson_fe", "advection_fe"])
def test_direct_residual_of_a_p1_function_equals_its_interpolated_residual(
    fe_name, request
) -> None:
    """An affine function is its own P1 interpolant, so both residuals agree."""
    fe = request.getfixturevalue(fe_name)
    direct = compute_fe_residual(
        fe, evaluate_network(affine_function, fe.basis.physical_points)
    )
    interpolated = compute_interpolated_fe_residual(
        fe, _nodal_values(affine_function, fe)
    )

    assert direct.shape == (fe.basis.space.number_of_dofs, 1)
    assert jnp.allclose(direct, interpolated, atol=1e-12)


@pytest.mark.parametrize("fe_name", ["poisson_fe", "advection_fe"])
def test_galerkin_solution_has_zero_free_residual(fe_name, request) -> None:
    fe = request.getfixturevalue(fe_name)
    residual = compute_interpolated_fe_residual(fe, _galerkin_solution(fe))
    assert jnp.allclose(_free(residual, fe), 0.0, atol=1e-12)


def test_exact_solution_residual_is_much_smaller_than_zero_trial(poisson_fe) -> None:
    def zero(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros((), dtype=x.dtype)

    points = poisson_fe.basis.physical_points
    exact = _free(
        compute_fe_residual(poisson_fe, evaluate_network(sin_sin_solution, points)),
        poisson_fe,
    )
    zero_trial = _free(
        compute_fe_residual(poisson_fe, evaluate_network(zero, points)), poisson_fe
    )
    assert euclidean_loss(exact) < 1e-3 * euclidean_loss(zero_trial)


def test_riesz_losses_match_dense_formulas(advection_fe) -> None:
    gram = restrict_to_free(advection_fe.gram, advection_fe.free_dofs)
    residual = jr.normal(jr.PRNGKey(2), (advection_fe.free_dofs.shape[-1], 1))

    assert jnp.allclose(
        riesz_loss(residual, jnp.linalg.cholesky(gram)),
        jnp.sum(residual * jnp.linalg.solve(gram, residual)),
    )
    assert jnp.allclose(
        diagonal_riesz_loss(residual, jnp.diagonal(gram)),
        jnp.sum(jnp.ravel(residual) ** 2 / jnp.diagonal(gram)),
    )
    assert jnp.allclose(gram, gram.T)


def test_riesz_loss_of_interpolated_trial_is_energy_distance_to_galerkin(
    poisson_fe, prng_key
) -> None:
    """For a symmetric problem, r^T G^-1 r = |grad(w - u_h)|^2 when w = u_h on Gamma."""
    interpolant = interpolate_network(MultiLayerPerceptron(key=prng_key), poisson_fe)
    error = interpolant.dof_values - _galerkin_solution(poisson_fe)
    factor = jnp.linalg.cholesky(
        restrict_to_free(poisson_fe.gram, poisson_fe.free_dofs)
    )
    residual = compute_interpolated_fe_residual(poisson_fe, interpolant.dof_values)

    assert jnp.allclose(
        riesz_loss(_free(residual, poisson_fe), factor),
        jnp.sum(error * (poisson_fe.stiffness @ error)),
    )


def test_losses_inside_jit_match_eager(poisson_fe, prng_key) -> None:
    factor = jnp.linalg.cholesky(
        restrict_to_free(poisson_fe.gram, poisson_fe.free_dofs)
    )
    distance = ExactDistanceFunction("unit_square")
    network = MultiLayerPerceptron(width=8, depth=2, key=prng_key)

    def direct_loss(model: MultiLayerPerceptron) -> jnp.ndarray:
        trial = evaluate_network(
            ConstrainedAnsatz(model, distance), poisson_fe.basis.physical_points
        )
        residual = _free(compute_fe_residual(poisson_fe, trial), poisson_fe)
        return riesz_loss(residual, factor) + euclidean_loss(residual)

    def interpolated_loss(model: MultiLayerPerceptron) -> jnp.ndarray:
        dof_values = interpolate_network(model, poisson_fe).dof_values
        residual = compute_interpolated_fe_residual(poisson_fe, dof_values)
        return riesz_loss(_free(residual, poisson_fe), factor)

    for loss_fn in (direct_loss, interpolated_loss):
        assert jnp.allclose(eqx.filter_jit(loss_fn)(network), loss_fn(network))


def test_boundary_residual_loss_integrates_over_the_boundary(unit_square_mesh) -> None:
    def zero_data(points: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros(points.shape[:-1] + (1,), dtype=points.dtype)

    def one(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.ones((), dtype=x.dtype)

    bottom = unit_square_mesh.boundary_tag_names["bottom"]

    assert jnp.allclose(boundary_residual_loss(one, unit_square_mesh, zero_data), 4.0)
    assert jnp.allclose(
        boundary_residual_loss(
            one, unit_square_mesh, zero_data, boundary_tags=(bottom,)
        ),
        1.0,
    )
    assert jnp.allclose(
        boundary_residual_loss(sin_sin_solution, unit_square_mesh, zero_data),
        0.0,
        atol=1e-28,
    )


def test_boundary_residual_loss_inside_jit_matches_eager() -> None:
    mesh = create_gmsh_unit_square_mesh(2)
    left = mesh.boundary_tag_names["left"]

    def data(points: jnp.ndarray) -> jnp.ndarray:
        return points[..., 1:2]

    def loss(scale: jnp.ndarray) -> jnp.ndarray:
        return boundary_residual_loss(
            lambda x: scale * x[0], mesh, data, boundary_tags=(left,)
        )

    assert jnp.allclose(eqx.filter_jit(loss)(jnp.asarray(2.0)), loss(jnp.asarray(2.0)))
    assert jnp.allclose(loss(jnp.asarray(2.0)), 1.0 / 3.0)
