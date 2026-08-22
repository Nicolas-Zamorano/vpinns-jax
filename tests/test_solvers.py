"""tests/test_adaptive_loop.py (implementation plan naming): trainer convergence and the
SOLVE -> ESTIMATE -> MARK -> REFINE cycle end to end.
"""

from __future__ import annotations

import jax.numpy as jnp

from vpinns_jax.models import ConstrainedAnsatz, ExactDistanceFunction, MultiLayerPerceptron
from vpinns_jax.pde import create_poisson_singular_problem
from vpinns_jax.solvers import run_adaptive_loop, train_adam


def test_train_adam_decreases_a_toy_quadratic_loss(prng_key):
    import equinox as eqx

    class Point(eqx.Module):
        value: jnp.ndarray

    def loss_fn(point: Point) -> jnp.ndarray:
        target = jnp.array([1.0, -2.0, 0.5])
        return jnp.sum((point.value - target) ** 2)

    point = Point(value=jnp.zeros((3,)))
    initial_loss = float(loss_fn(point))
    trained, history = train_adam(loss_fn, point, num_steps=200, learning_rate=0.1)
    assert history.loss.shape == (200,)
    assert float(loss_fn(trained)) < initial_loss * 1e-3


def test_train_adam_reduces_direct_patch_euclidean_loss(unit_square_mesh, sin_sin_problem, prng_key):
    from vpinns_jax.losses import direct_trial_patch_residual, euclidean_loss
    from vpinns_jax.test_spaces import create_patch_test_space

    patches = create_patch_test_space(unit_square_mesh)
    distance = ExactDistanceFunction("unit_square")
    network = MultiLayerPerceptron(key=prng_key, width=16, depth=2)

    def loss_fn(trial_model):
        trial = ConstrainedAnsatz(trial_model, distance)
        residual = direct_trial_patch_residual(trial, patches, sin_sin_problem)
        return euclidean_loss(residual)

    initial_loss = float(loss_fn(network))
    trained, history = train_adam(loss_fn, network, num_steps=100, learning_rate=3e-3)
    assert float(loss_fn(trained)) < initial_loss


def test_adaptive_loop_grows_the_mesh_and_returns_one_result_per_level(prng_key):
    from jax_fem.mesh import create_gmsh_unit_square_mesh

    mesh = create_gmsh_unit_square_mesh(4)
    network = MultiLayerPerceptron(key=prng_key, width=16, depth=2)
    distance = ExactDistanceFunction("unit_square")

    results = run_adaptive_loop(
        mesh,
        create_poisson_singular_problem,
        network,
        distance,
        num_levels=2,
        adam_steps=50,
        adam_learning_rate=3e-3,
        dorfler_theta=0.5,
        max_marked_fraction=0.3,
    )

    assert len(results) == 2
    assert results[1].mesh.cells_to_vertices.shape[0] > results[0].mesh.cells_to_vertices.shape[0]
    assert results[0].number_of_marked_cells > 0
