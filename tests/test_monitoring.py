"""Tests for the training monitor and the solution norm."""

import jax.numpy as jnp
from jax_fem.diagnostics import energy_error_functional, l2_error_functional

from vpinns_jax.analysis import (
    TrainingRecord,
    compute_network_error,
    compute_solution_norm,
    create_training_monitor,
)
from vpinns_jax.losses import (
    compute_fe_residual,
    euclidean_loss,
    restrict_to_free,
    riesz_loss,
)
from vpinns_jax.models import (
    ConstrainedAnsatz,
    ExactDistanceFunction,
    MultiLayerPerceptron,
    evaluate_network,
)


def test_solution_norms_of_the_sin_sin_problem(p1_basis, sin_sin_problem) -> None:
    energy_norm = compute_solution_norm(
        p1_basis, sin_sin_problem, energy_error_functional
    )
    l2_norm = compute_solution_norm(p1_basis, sin_sin_problem, l2_error_functional)

    assert jnp.allclose(energy_norm, jnp.pi / jnp.sqrt(2.0), rtol=1e-2)
    assert jnp.allclose(l2_norm, 0.5, rtol=1e-2)


def test_monitor_records_consistent_quantities(
    poisson_fe, sin_sin_problem, prng_key
) -> None:
    basis = poisson_fe.basis
    factor = jnp.linalg.cholesky(
        restrict_to_free(poisson_fe.gram, poisson_fe.free_dofs)
    )
    distance = ExactDistanceFunction("unit_square")

    def loss_terms(model):
        trial = evaluate_network(
            ConstrainedAnsatz(model, distance), basis.physical_points
        )
        residual = jnp.take(
            compute_fe_residual(poisson_fe, trial), poisson_fe.free_dofs, axis=-2
        )
        return {
            "riesz": riesz_loss(residual, factor),
            "euclidean": euclidean_loss(residual),
        }

    def energy_error(model):
        return compute_network_error(
            ConstrainedAnsatz(model, distance),
            basis,
            sin_sin_problem,
            energy_error_functional,
        )

    solution_norm = compute_solution_norm(
        basis, sin_sin_problem, energy_error_functional
    )
    monitor = create_training_monitor(loss_terms, energy_error, solution_norm)
    network = MultiLayerPerceptron(width=8, depth=2, key=prng_key)
    record = monitor(network)
    terms = loss_terms(network)

    assert isinstance(record, TrainingRecord)
    assert jnp.allclose(record.loss, terms["riesz"] + terms["euclidean"])
    assert jnp.allclose(record.energy_error, energy_error(network))
    assert jnp.allclose(record.relative_loss, jnp.sqrt(record.loss) / solution_norm)
    assert jnp.allclose(
        record.relative_energy_error, record.energy_error / solution_norm
    )
    assert jnp.allclose(
        record.robustness, record.relative_loss / record.relative_energy_error
    )
