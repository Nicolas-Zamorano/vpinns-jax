"""tests/test_ansatz.py (implementation plan naming): MLP outputs, exact distance
function B(x)=0 on the boundary, ConstrainedAnsatz, and Pi_h u_theta interpolation.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr

from vpinns_jax.models import (
    ConstrainedAnsatz,
    ExactDistanceFunction,
    FourierFeatureMLP,
    MultiLayerPerceptron,
    interpolated_trial_dofs,
    precompute_interpolation_data,
)
from vpinns_jax.test_spaces import create_fe_test_space


def test_mlp_output_is_scalar(prng_key):
    model = MultiLayerPerceptron(key=prng_key)
    output = model(jnp.array([0.3, 0.4]))
    assert output.shape == ()


def test_fourier_feature_mlp_output_is_scalar(prng_key):
    model = FourierFeatureMLP(key=prng_key, num_fourier_features=8)
    output = model(jnp.array([0.3, 0.4]))
    assert output.shape == ()


def test_unit_square_distance_vanishes_on_boundary_and_is_positive_inside():
    distance = ExactDistanceFunction("unit_square")
    for point in [[0.0, 0.5], [1.0, 0.5], [0.5, 0.0], [0.5, 1.0]]:
        assert jnp.allclose(distance(jnp.array(point)), 0.0)
    assert distance(jnp.array([0.5, 0.5])) > 0.0


def test_l_shape_distance_vanishes_on_boundary_and_is_positive_inside():
    distance = ExactDistanceFunction("l_shape")
    # Vertices and the re-entrant corner itself are exactly on Gamma.
    for point in [[0.0, 0.0], [1.0, 0.0], [-1.0, -1.0], [0.5, 0.0]]:
        assert jnp.allclose(distance(jnp.array(point)), 0.0, atol=1e-10)
    assert distance(jnp.array([-0.5, 0.5])) > 0.0


def test_circle_distance_vanishes_on_boundary_and_is_positive_inside():
    distance = ExactDistanceFunction("circle")
    assert jnp.allclose(distance(jnp.array([1.0, 0.0])), 0.0, atol=1e-10)
    assert distance(jnp.array([0.0, 0.0])) > 0.0


def test_constrained_ansatz_satisfies_boundary_condition_exactly(prng_key):
    network = MultiLayerPerceptron(key=prng_key)
    distance = ExactDistanceFunction("unit_square")
    constrained = ConstrainedAnsatz(network, distance)
    for point in [[0.0, 0.5], [1.0, 0.5], [0.5, 0.0], [0.5, 1.0]]:
        assert jnp.allclose(constrained(jnp.array(point)), 0.0, atol=1e-12)


def test_constrained_ansatz_nonhomogeneous_lift_matches_g_on_boundary(prng_key):
    network = MultiLayerPerceptron(key=prng_key)
    distance = ExactDistanceFunction("unit_square")
    lift = lambda x: x[0] + 2.0 * x[1]  # noqa: E731
    constrained = ConstrainedAnsatz(network, distance, lift)
    point = jnp.array([1.0, 0.3])
    assert jnp.allclose(constrained(point), lift(point), atol=1e-12)


def test_interpolated_trial_dofs_matches_exact_dirichlet_data(unit_square_mesh, sin_sin_problem, prng_key):
    network = MultiLayerPerceptron(key=prng_key)
    fe = create_fe_test_space(unit_square_mesh, sin_sin_problem)
    data = precompute_interpolation_data(fe.space, sin_sin_problem)
    dofs = interpolated_trial_dofs(network, data)

    assert dofs.shape == (fe.space.number_of_dofs, 1)
    # Boundary DOFs must equal the (here, zero) Dirichlet data exactly.
    assert jnp.allclose(dofs[data.dirichlet_dofs, 0], data.dirichlet_values[:, 0])
    # Interior DOFs must equal the network's own value at that coordinate.
    interior_coordinates = data.dof_coordinates[data.free_dofs]
    expected_interior = jnp.array([network(x) for x in interior_coordinates])
    assert jnp.allclose(dofs[data.free_dofs, 0], expected_interior)
