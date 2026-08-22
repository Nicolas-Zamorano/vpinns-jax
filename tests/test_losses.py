"""tests/test_losses.py: weak residual, Euclidean loss, Riesz loss, and the
single-DOF patch scalar Gram collapse identity.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax_fem.function import FiniteElementFunction

from vpinns_jax.losses import (
    assemble_energy_gram_matrix,
    direct_trial_fe_residual,
    direct_trial_patch_residual,
    euclidean_loss,
    interpolated_trial_fe_residual,
    interpolated_trial_patch_residual,
    patch_self_energy,
    restrict_to_free_dofs,
    riesz_loss_global,
    riesz_loss_patch,
)
from vpinns_jax.models import (
    interpolated_trial_dofs,
    precompute_interpolation_data,
)
from vpinns_jax.test_spaces import create_fe_test_space, create_patch_test_space


def _exact_sin_sin(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.sin(jnp.pi * x[0]) * jnp.sin(jnp.pi * x[1])


def _zero(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.zeros(())


def test_direct_trial_residual_is_much_smaller_for_exact_solution_than_zero(
    unit_square_mesh, sin_sin_problem
):
    fe = create_fe_test_space(unit_square_mesh, sin_sin_problem)
    r_exact = direct_trial_fe_residual(_exact_sin_sin, fe)[fe.free_dofs, 0]
    r_zero = direct_trial_fe_residual(_zero, fe)[fe.free_dofs, 0]
    assert euclidean_loss(r_exact) < 1e-3 * euclidean_loss(r_zero)


def test_direct_trial_patch_residual_is_much_smaller_for_exact_solution_than_zero(
    unit_square_mesh, sin_sin_problem
):
    patches = create_patch_test_space(unit_square_mesh)
    r_exact = direct_trial_patch_residual(_exact_sin_sin, patches, sin_sin_problem)
    r_zero = direct_trial_patch_residual(_zero, patches, sin_sin_problem)
    assert euclidean_loss(r_exact) < 1e-3 * euclidean_loss(r_zero)


def test_interpolated_trial_fe_residual_shape(unit_square_mesh, sin_sin_problem, prng_key):
    from vpinns_jax.models import MultiLayerPerceptron

    fe = create_fe_test_space(unit_square_mesh, sin_sin_problem)
    network = MultiLayerPerceptron(key=prng_key)
    data = precompute_interpolation_data(fe.space, sin_sin_problem)
    w_dofs = interpolated_trial_dofs(network, data)
    residual = interpolated_trial_fe_residual(w_dofs, fe)
    assert residual.shape == (fe.space.number_of_dofs, 1)


def test_interpolated_trial_patch_residual_matches_fe_residual_at_shared_dofs(
    unit_square_mesh, sin_sin_problem, prng_key
):
    """Both paths (global-FE-scatter vs. patch-local-gather) must agree exactly on
    the same interpolated (piecewise-linear) trial function: both compute the
    same discrete weak residual r_h(w, phi_i), just via different code paths."""
    from vpinns_jax.models import MultiLayerPerceptron

    fe = create_fe_test_space(unit_square_mesh, sin_sin_problem)
    patches = create_patch_test_space(unit_square_mesh, space=fe.space)
    network = MultiLayerPerceptron(key=prng_key)
    data = precompute_interpolation_data(fe.space, sin_sin_problem)
    w_dofs = interpolated_trial_dofs(network, data)
    solution = FiniteElementFunction(space=fe.space, dof_values=w_dofs)

    r_fe = interpolated_trial_fe_residual(w_dofs, fe)[patches.interior_vertices, 0]
    r_patch = interpolated_trial_patch_residual(solution, patches, sin_sin_problem)
    assert jnp.allclose(r_fe, r_patch, atol=1e-8)


def test_energy_gram_matrix_is_symmetric_positive_definite(unit_square_mesh, sin_sin_problem):
    fe = create_fe_test_space(unit_square_mesh, sin_sin_problem)
    gram = assemble_energy_gram_matrix(fe, sin_sin_problem)
    gram_free = restrict_to_free_dofs(gram, fe.free_dofs)
    assert jnp.allclose(gram_free, gram_free.T)
    eigenvalues = jnp.linalg.eigvalsh(gram_free)
    assert jnp.all(eigenvalues > 0.0)


def test_riesz_loss_global_is_much_smaller_for_exact_solution_than_zero(
    unit_square_mesh, sin_sin_problem
):
    fe = create_fe_test_space(unit_square_mesh, sin_sin_problem)
    gram_free = restrict_to_free_dofs(
        assemble_energy_gram_matrix(fe, sin_sin_problem), fe.free_dofs
    )
    r_exact = direct_trial_fe_residual(_exact_sin_sin, fe)[fe.free_dofs, 0]
    r_zero = direct_trial_fe_residual(_zero, fe)[fe.free_dofs, 0]
    assert riesz_loss_global(r_exact, gram_free) < 1e-3 * riesz_loss_global(r_zero, gram_free)


def test_single_dof_patch_collapse_identity(unit_square_mesh, sin_sin_problem):
    """L_i(w)^2 / (varphi_i, varphi_i)_A summed over patches == riesz_loss_patch
    (math reference Section 2.3's single-function-per-patch identity)."""
    patches = create_patch_test_space(unit_square_mesh)
    residual = direct_trial_patch_residual(_exact_sin_sin, patches, sin_sin_problem)
    self_energy = patch_self_energy(patches, sin_sin_problem)
    assert jnp.all(self_energy > 0.0)

    total = riesz_loss_patch(residual, self_energy)
    per_patch_sum = jnp.sum(residual**2 / self_energy)
    assert jnp.allclose(total, per_patch_sum)
