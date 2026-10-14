"""Tests for the network trial functions and the strong Dirichlet imposition."""

import equinox as eqx
import jax.numpy as jnp
import pytest

from vpinns_jax.models import (
    ConstrainedAnsatz,
    ExactDistanceFunction,
    FourierFeatureMLP,
    MultiLayerPerceptron,
    evaluate_network_values,
    interpolate_network,
)

_UNIT_SQUARE_BOUNDARY_POINTS = [[0.0, 0.5], [1.0, 0.5], [0.5, 0.0], [0.5, 1.0]]


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
