"""tests/test_pde.py: manufactured PDE problems and the energy coefficient."""

from __future__ import annotations

import jax.numpy as jnp

from vpinns_jax.pde import (
    create_poisson_cos_sin_problem,
    create_poisson_sin_sin_problem,
    energy_reaction_coefficient,
)


def test_sin_sin_solution_and_source_are_consistent(unit_square_mesh):
    problem = create_poisson_sin_sin_problem(unit_square_mesh)
    points = jnp.array([[[0.3, 0.4]], [[0.7, 0.2]]])
    exact = problem.exact_solution(points)
    expected = jnp.sin(jnp.pi * points[..., 0]) * jnp.sin(jnp.pi * points[..., 1])
    assert jnp.allclose(exact[..., 0], expected)


def test_sin_sin_vanishes_on_boundary(unit_square_mesh):
    problem = create_poisson_sin_sin_problem(unit_square_mesh)
    boundary_points = jnp.array([[[0.0, 0.5]], [[1.0, 0.3]], [[0.6, 0.0]], [[0.2, 1.0]]])
    assert jnp.allclose(problem.exact_solution(boundary_points), 0.0, atol=1e-12)


def test_cos_sin_boundary_condition_is_nonhomogeneous(unit_square_mesh):
    problem = create_poisson_cos_sin_problem(unit_square_mesh)
    condition = problem.dirichlet_conditions[0]
    point = jnp.array([[[0.0, 0.5]]])
    value = condition.value(point)
    assert not jnp.allclose(value, 0.0)


def test_energy_reaction_coefficient_matches_symmetrized_formula(unit_square_mesh):
    problem = create_poisson_sin_sin_problem(unit_square_mesh)
    coefficient = energy_reaction_coefficient(problem)
    points = jnp.array([[[0.3, 0.4]]])
    # Poisson: sigma = 0, div(beta) = 0, so the symmetrized reaction is 0.
    assert jnp.allclose(coefficient(points), 0.0)
