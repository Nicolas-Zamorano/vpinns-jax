"""tests/test_convergence.py: exact error evaluators, efficiency index, and the
uniform vs. adaptive convergence study runner.
"""

from __future__ import annotations

import jax.numpy as jnp

from vpinns_jax.analysis import (
    direct_trial_energy_error,
    direct_trial_h1_seminorm_error,
    direct_trial_relative_l2_error,
    efficiency_index,
    run_convergence_study,
)
from vpinns_jax.models import ExactDistanceFunction, MultiLayerPerceptron
from vpinns_jax.pde import create_poisson_sin_sin_problem
from vpinns_jax.test_spaces import create_fe_test_space


def _exact_sin_sin(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.sin(jnp.pi * x[0]) * jnp.sin(jnp.pi * x[1])


def test_direct_trial_errors_are_near_zero_for_the_exact_solution(
    unit_square_mesh, sin_sin_problem
):
    fe = create_fe_test_space(unit_square_mesh, sin_sin_problem)
    assert direct_trial_relative_l2_error(_exact_sin_sin, fe.basis, sin_sin_problem) < 1e-6
    assert direct_trial_h1_seminorm_error(_exact_sin_sin, fe.basis, sin_sin_problem) < 1e-6
    assert direct_trial_energy_error(_exact_sin_sin, fe.basis, sin_sin_problem) < 1e-6


def test_direct_trial_errors_are_large_for_the_zero_function(unit_square_mesh, sin_sin_problem):
    fe = create_fe_test_space(unit_square_mesh, sin_sin_problem)
    zero = lambda x: jnp.zeros(())  # noqa: E731
    assert direct_trial_relative_l2_error(zero, fe.basis, sin_sin_problem) > 0.5


def test_efficiency_index_formula():
    assert jnp.allclose(efficiency_index(2.0, 4.0), 0.5)


def test_convergence_study_returns_one_record_per_level_with_positive_fields(prng_key):
    from jax_fem.mesh import create_gmsh_unit_square_mesh

    mesh = create_gmsh_unit_square_mesh(4)
    distance = ExactDistanceFunction("unit_square")

    def model_factory(key):
        return MultiLayerPerceptron(key=key, width=8, depth=2)

    for adaptive in (False, True):
        records = run_convergence_study(
            mesh,
            create_poisson_sin_sin_problem,
            model_factory,
            distance,
            num_levels=2,
            adaptive=adaptive,
            key=prng_key,
            adam_steps=30,
            adam_learning_rate=3e-3,
        )
        assert len(records) == 2
        assert records[1].number_of_dofs >= records[0].number_of_dofs
        for record in records:
            assert record.relative_l2_error > 0.0
            assert record.energy_error > 0.0
            assert record.global_estimator >= 0.0
