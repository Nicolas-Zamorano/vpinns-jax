"""Tests for the network trial functions and the strong Dirichlet imposition."""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from vpinns_jax.models import (
    ConstrainedAnsatz,
    ExactDistanceFunction,
    FourierFeatureMLP,
    MultiLayerPerceptron,
    TransfiniteLift,
    WachspressLift,
    evaluate_network,
    evaluate_network_flux_divergence,
    evaluate_network_values,
    interpolate_network,
)

_UNIT_SQUARE_BOUNDARY_POINTS = [[0.0, 0.5], [1.0, 0.5], [0.5, 0.0], [0.5, 1.0]]


def _unit_square_boundary_points(points_per_side: int = 9) -> jnp.ndarray:
    """Points on the four sides of the unit square, corners included."""
    t = jnp.linspace(0.0, 1.0, points_per_side)
    zero, one = jnp.zeros_like(t), jnp.ones_like(t)
    sides = [(zero, t), (one, t), (t, zero), (t, one)]
    return jnp.concatenate([jnp.stack(side, axis=-1) for side in sides], axis=-2)


def _blended_function(points: jnp.ndarray) -> jnp.ndarray:
    """``sin(3x) + y cos(2x) + x exp(y)``, a sum of terms affine in x or y."""
    x, y = points[..., [0]], points[..., [1]]
    return jnp.sin(3.0 * x) + y * jnp.cos(2.0 * x) + x * jnp.exp(y)


def _blended_gradient(points: jnp.ndarray) -> jnp.ndarray:
    """Gradient of ``_blended_function``, shape ``(*batch, 1, 2)``."""
    x, y = points[..., 0], points[..., 1]
    grad_x = 3.0 * jnp.cos(3.0 * x) - 2.0 * y * jnp.sin(2.0 * x) + jnp.exp(y)
    grad_y = jnp.cos(2.0 * x) + x * jnp.exp(y)
    return jnp.stack((grad_x, grad_y), axis=-1)


@pytest.mark.parametrize("network_type", [MultiLayerPerceptron, FourierFeatureMLP])
def test_networks_return_scalars(network_type, prng_key) -> None:
    network = network_type(key=prng_key)
    assert network(jnp.array([0.3, 0.4])).shape == ()


def test_fourier_frequencies_are_frozen_unless_trainable(prng_key) -> None:
    network = FourierFeatureMLP(key=prng_key, number_of_features=4)
    gradient = eqx.filter_grad(lambda model: model(jnp.array([0.3, 0.4])))(network)
    assert jnp.all(gradient.frequency_matrix == 0.0)


def test_unit_square_distance_vanishes_on_boundary_and_is_positive_inside() -> None:
    distance = ExactDistanceFunction("unit_square")
    for point in _UNIT_SQUARE_BOUNDARY_POINTS:
        assert distance(jnp.array(point)) == 0.0
    assert distance(jnp.array([0.5, 0.5])) > 0.0


def test_l_shape_distance_vanishes_on_boundary_and_is_positive_inside() -> None:
    distance = ExactDistanceFunction("l_shape")
    for point in [[0.0, 0.0], [1.0, 0.0], [-1.0, -1.0], [0.5, 0.0], [0.0, -0.5]]:
        assert jnp.allclose(distance(jnp.array(point)), 0.0, atol=1e-14)
    assert jnp.allclose(distance(jnp.array([-0.5, 0.5])), 0.5)


def test_circle_distance_vanishes_on_boundary_and_is_positive_inside() -> None:
    distance = ExactDistanceFunction("circle")
    assert jnp.allclose(distance(jnp.array([0.6, 0.8])), 0.0, atol=1e-14)
    assert distance(jnp.array([0.0, 0.0])) == 1.0


def test_unknown_geometry_is_rejected() -> None:
    with pytest.raises(ValueError, match="geometry"):
        ExactDistanceFunction("triangle")


def test_constrained_ansatz_satisfies_homogeneous_condition(prng_key) -> None:
    trial = ConstrainedAnsatz(
        MultiLayerPerceptron(key=prng_key), ExactDistanceFunction("unit_square")
    )
    for point in _UNIT_SQUARE_BOUNDARY_POINTS:
        assert trial(jnp.array(point)) == 0.0


def test_constrained_ansatz_matches_lift_on_boundary(prng_key) -> None:
    def lift(x: jnp.ndarray) -> jnp.ndarray:
        return x[0] + 2.0 * x[1]

    trial = ConstrainedAnsatz(
        MultiLayerPerceptron(key=prng_key), ExactDistanceFunction("unit_square"), lift
    )
    for point in _UNIT_SQUARE_BOUNDARY_POINTS:
        assert jnp.allclose(trial(jnp.array(point)), lift(jnp.array(point)))


@pytest.mark.parametrize("lift_type", [TransfiniteLift, WachspressLift])
def test_lift_matches_dirichlet_data_on_boundary(lift_type, advection_problem) -> None:
    (condition,) = advection_problem.dirichlet_conditions
    points = jnp.expand_dims(_unit_square_boundary_points(), axis=-2)
    lifted = evaluate_network_values(lift_type(condition.value), points)
    assert jnp.allclose(lifted, condition.value(points), rtol=0.0, atol=1e-13)


def _reference_wachspress_lift(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Transfinite interpolant with Wachspress coordinates, written as published."""
    x_verts = np.array([0.0, 1.0, 1.0, 0.0])
    y_verts = np.array([0.0, 0.0, 1.0, 1.0])
    lambda_wsp = np.stack([(1.0 - x) * (1.0 - y), x * (1.0 - y), x * y, (1.0 - x) * y])

    def boundary_function(px: np.ndarray, py: np.ndarray) -> np.ndarray:
        points = np.expand_dims(np.stack([px, py], axis=-1), axis=-2)
        return np.asarray(_blended_function(jnp.asarray(points)))[..., 0, 0]

    g = np.zeros_like(x)
    for i in range(4):
        ip1, im1 = (i + 1) % 4, (i - 1) % 4
        x_ip1 = (1.0 - lambda_wsp[ip1]) * x_verts[i] + lambda_wsp[ip1] * x_verts[ip1]
        y_ip1 = (1.0 - lambda_wsp[ip1]) * y_verts[i] + lambda_wsp[ip1] * y_verts[ip1]
        x_im1 = (1.0 - lambda_wsp[im1]) * x_verts[i] + lambda_wsp[im1] * x_verts[im1]
        y_im1 = (1.0 - lambda_wsp[im1]) * y_verts[i] + lambda_wsp[im1] * y_verts[im1]
        vertex = boundary_function(
            np.full_like(x, x_verts[i]), np.full_like(x, y_verts[i])
        )
        g += lambda_wsp[i] * (
            boundary_function(x_ip1, y_ip1) + boundary_function(x_im1, y_im1) - vertex
        )
    return g


def test_wachspress_lift_matches_the_published_interpolant() -> None:
    points = jnp.array([[[0.3, 0.4]], [[0.9, 0.1]], [[0.5, 0.5]], [[0.05, 0.75]]])
    lifted = evaluate_network_values(WachspressLift(_blended_function), points)
    expected = _reference_wachspress_lift(
        np.asarray(points[:, 0, 0]), np.asarray(points[:, 0, 1])
    )
    assert jnp.allclose(lifted[:, 0, 0], expected, rtol=0.0, atol=1e-13)


def test_transfinite_lift_reproduces_blended_functions() -> None:
    lift = TransfiniteLift(_blended_function)
    points = jnp.array([[[0.3, 0.4]], [[0.9, 0.1]], [[0.5, 0.5]], [[0.05, 0.75]]])
    evaluation = evaluate_network(lift, points)
    assert jnp.allclose(evaluation.values, _blended_function(points), atol=1e-13)
    assert jnp.allclose(evaluation.gradients, _blended_gradient(points), atol=1e-12)


def test_transfinite_lift_flux_divergence_matches_laplacian() -> None:
    def identity_diffusion(points: jnp.ndarray) -> jnp.ndarray:
        return jnp.broadcast_to(jnp.eye(2), points.shape[:-2] + (2, 2))

    # Laplacian of sin(3x) + y cos(2x) + x exp(y).
    def laplacian(points: jnp.ndarray) -> jnp.ndarray:
        x, y = points[..., [0]], points[..., [1]]
        return -9.0 * jnp.sin(3.0 * x) - 4.0 * y * jnp.cos(2.0 * x) + x * jnp.exp(y)

    points = jnp.array([[[0.3, 0.4]], [[0.9, 0.1]], [[0.05, 0.75]]])
    flux_divergence = evaluate_network_flux_divergence(
        TransfiniteLift(_blended_function), identity_diffusion, points
    )
    assert jnp.allclose(flux_divergence, laplacian(points), atol=1e-11)


@pytest.mark.parametrize("lift_type", [TransfiniteLift, WachspressLift])
def test_constrained_ansatz_with_lift_satisfies_dirichlet_data(
    lift_type, advection_problem, prng_key
) -> None:
    (condition,) = advection_problem.dirichlet_conditions
    trial = ConstrainedAnsatz(
        MultiLayerPerceptron(key=prng_key),
        ExactDistanceFunction("unit_square"),
        lift_type(condition.value),
    )
    points = jnp.expand_dims(_unit_square_boundary_points(), axis=-2)
    assert jnp.allclose(
        evaluate_network_values(trial, points),
        condition.value(points),
        rtol=0.0,
        atol=1e-13,
    )


def test_interpolate_network_imposes_dirichlet_values(poisson_fe, prng_key) -> None:
    network = MultiLayerPerceptron(key=prng_key)
    interpolant = interpolate_network(network, poisson_fe)
    space = poisson_fe.basis.space
    network_values = jnp.squeeze(
        evaluate_network_values(network, space.dof_coordinates), axis=-2
    )

    assert interpolant.space is space
    assert interpolant.dof_values.shape == (space.number_of_dofs, 1)
    assert jnp.array_equal(
        interpolant.dof_values[poisson_fe.dirichlet_dofs], poisson_fe.dirichlet_values
    )
    assert jnp.array_equal(
        interpolant.dof_values[poisson_fe.free_dofs],
        network_values[poisson_fe.free_dofs],
    )


def test_interpolate_network_inside_jit_matches_eager(poisson_fe, prng_key) -> None:
    network = MultiLayerPerceptron(key=prng_key)

    def dof_values(model: MultiLayerPerceptron) -> jnp.ndarray:
        return interpolate_network(model, poisson_fe).dof_values

    assert jnp.allclose(eqx.filter_jit(dof_values)(network), dof_values(network))
