"""Tests for the error norms of network trial functions."""

import jax.numpy as jnp
import pytest
from jax_fem.diagnostics import (
    compute_energy_error,
    compute_h1_seminorm_error,
    compute_l2_error,
    energy_error_functional,
    h1_seminorm_error_functional,
    l2_error_functional,
)
from jax_fem.function import FiniteElementFunction

from vpinns_jax.analysis import compute_network_error
from vpinns_jax.models import evaluate_network_values


def affine_function(x: jnp.ndarray) -> jnp.ndarray:
    return 1.0 + 2.0 * x[0] - 0.5 * x[1]


def sin_sin_solution(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.sin(jnp.pi * x[0]) * jnp.sin(jnp.pi * x[1])


_FUNCTIONALS_AND_ERRORS = [
    (l2_error_functional, compute_l2_error),
    (h1_seminorm_error_functional, compute_h1_seminorm_error),
    (energy_error_functional, compute_energy_error),
]


@pytest.mark.parametrize(("functional", "fe_error"), _FUNCTIONALS_AND_ERRORS)
def test_network_error_matches_fe_error_of_its_interpolant(
    functional, fe_error, advection_fe, advection_problem
) -> None:
    """An affine function equals its P1 interpolant, so both errors agree."""
    basis = advection_fe.basis
    nodal_values = evaluate_network_values(affine_function, basis.space.dof_coordinates)
    interpolant = FiniteElementFunction(
        space=basis.space, dof_values=jnp.squeeze(nodal_values, axis=-2)
    )

    assert jnp.allclose(
        compute_network_error(affine_function, basis, advection_problem, functional),
        fe_error(interpolant, basis, advection_problem),
    )
    assert jnp.allclose(
        compute_network_error(
            affine_function, basis, advection_problem, functional, reduce=False
        ),
        fe_error(interpolant, basis, advection_problem, reduce=False),
    )


@pytest.mark.parametrize("functional", [pair[0] for pair in _FUNCTIONALS_AND_ERRORS])
def test_exact_solution_has_zero_error(functional, p1_basis, sin_sin_problem) -> None:
    error = compute_network_error(
        sin_sin_solution, p1_basis, sin_sin_problem, functional
    )
    assert error < 1e-12


def test_cell_errors_square_sum_to_global_error(p1_basis, sin_sin_problem) -> None:
    cell_errors = compute_network_error(
        affine_function, p1_basis, sin_sin_problem, l2_error_functional, reduce=False
    )
    global_error = compute_network_error(
        affine_function, p1_basis, sin_sin_problem, l2_error_functional
    )

    assert cell_errors.shape == (p1_basis.physical_points.shape[0], 1, 1, 1)
    assert jnp.allclose(jnp.sum(cell_errors**2), global_error**2)
