"""tests/test_estimators.py: jump-term reduction, local indicators, and Doerfler marking."""

from __future__ import annotations

import jax.numpy as jnp

from vpinns_jax.estimators import (
    cap_marked_fraction,
    direct_trial_patch_residual_indicators,
    global_patch_estimator,
    mark_by_dorfler_bulk_criterion,
    patch_indicators_to_cell_indicators,
    patch_loss_indicators,
)
from vpinns_jax.losses import direct_trial_patch_residual, patch_self_energy
from vpinns_jax.test_spaces import create_patch_test_space


def _exact_sin_sin(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.sin(jnp.pi * x[0]) * jnp.sin(jnp.pi * x[1])


def _zero(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.zeros(())


def test_direct_trial_residual_indicator_near_zero_for_exact_solution(
    unit_square_mesh, sin_sin_problem
):
    """r_i(w) = f + Laplacian(w) is analytically zero for the exact solution
    (jump term already dropped -- Section 4.3 -- since the trial is direct/smooth)."""
    patches = create_patch_test_space(unit_square_mesh)
    eta_r_exact = direct_trial_patch_residual_indicators(
        _exact_sin_sin, patches, sin_sin_problem
    )
    eta_r_zero = direct_trial_patch_residual_indicators(_zero, patches, sin_sin_problem)
    assert jnp.all(eta_r_exact < 1e-8)
    assert jnp.all(eta_r_zero > eta_r_exact)


def test_patch_loss_indicators_square_sum_to_riesz_patch_loss(unit_square_mesh, sin_sin_problem):
    from vpinns_jax.losses import riesz_loss_patch

    patches = create_patch_test_space(unit_square_mesh)
    residual = direct_trial_patch_residual(_exact_sin_sin, patches, sin_sin_problem)
    self_energy = patch_self_energy(patches, sin_sin_problem)
    eta_l = patch_loss_indicators(residual, self_energy)
    assert jnp.allclose(jnp.sum(eta_l**2), riesz_loss_patch(residual, self_energy))


def test_global_patch_estimator_combines_loss_and_residual_indicators():
    eta_l = jnp.array([1.0, 2.0, 3.0])
    eta_r = jnp.array([0.5, 0.5, 0.5])
    expected = jnp.sqrt(jnp.sum(eta_l**2 + eta_r**2))
    assert jnp.allclose(global_patch_estimator(eta_l, eta_r), expected)


def test_dorfler_marking_marks_the_largest_indicators_first():
    indicators = jnp.array([1.0, 5.0, 2.0, 0.1])
    marked = mark_by_dorfler_bulk_criterion(indicators, theta=0.6)
    # 5.0^2 = 25, total = 1+25+4+0.01=30.01; 25/30.01 ~ 0.833 >= 0.6 alone.
    assert bool(marked[1])
    assert int(jnp.sum(marked)) >= 1


def test_dorfler_theta_one_marks_everything():
    indicators = jnp.array([1.0, 5.0, 2.0, 0.1])
    marked = mark_by_dorfler_bulk_criterion(indicators, theta=1.0)
    assert bool(jnp.all(marked))


def test_cap_marked_fraction_respects_the_cap():
    indicators = jnp.arange(1.0, 11.0)
    marked = jnp.ones((10,), dtype=bool)
    capped = cap_marked_fraction(marked, indicators, max_fraction=0.3)
    assert int(jnp.sum(capped)) == 3
    # The kept entries must be the 3 largest-indicator ones.
    assert bool(jnp.all(indicators[capped] >= 8.0))


def test_patch_indicators_to_cell_indicators_shape_and_nonnegativity(unit_square_mesh, sin_sin_problem):
    patches = create_patch_test_space(unit_square_mesh)
    residual = direct_trial_patch_residual(_exact_sin_sin, patches, sin_sin_problem)
    self_energy = patch_self_energy(patches, sin_sin_problem)
    indicators = patch_loss_indicators(residual, self_energy)
    cell_indicators = patch_indicators_to_cell_indicators(patches, indicators)
    assert cell_indicators.shape == (unit_square_mesh.cells_to_vertices.shape[0],)
    assert jnp.all(cell_indicators >= 0.0)
