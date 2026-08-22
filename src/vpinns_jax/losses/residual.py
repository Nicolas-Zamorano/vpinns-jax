"""Discrete weak residual ``r_h(w_theta, v) = a_h(w_theta, v) - l_h(v)`` (math reference Section 1).

Four evaluators, one per (trial representation, test space) combination
(math reference Section 2.1/2.2's composability): direct/interpolated trial
against an FE (``test_spaces.fe_space``) or mesh-free patch
(``test_spaces.patch_space``) test space. All four share one pointwise
integrand formula, ``_residual_pointwise``, matching ``jax_fem.forms
.elliptic``'s bilinear/linear forms combined into one residual:

    r_h(w, v) = int (grad v)^T mu grad(w) + v beta.grad(w) + sigma v w - f v

- **Direct** trial (``w = u_theta``, e.g. a raw network or a
  ``models.boundary.ConstrainedAnsatz``): ``w``'s value/gradient at
  quadrature points come from ``jax.value_and_grad``, vmapped.
- **Interpolated** trial (``w = Pi_h u_theta``): ``w`` is already a P1 FE
  function, so the FE-test-space residual collapses to a plain sparse
  matrix-vector product (``stiffness @ w_dofs - load``, precomputed by
  ``test_spaces.fe_space``); the patch-test-space residual gathers
  jax-fem's own ``evaluate_finite_element_function``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax_fem.assembly import integrate_cellwise
from jax_fem.function import FiniteElementFunction, evaluate_finite_element_function
from jax_fem.problem import EllipticCoefficientValues, evaluate_elliptic_coefficients

from vpinns_jax.pde.base import EllipticPDE
from vpinns_jax.test_spaces.fe_space import FETestSpace
from vpinns_jax.test_spaces.patch_space import PatchTestSpace, gather_patch_quadrature

__all__ = [
    "evaluate_trial_at_points",
    "direct_trial_fe_residual",
    "interpolated_trial_fe_residual",
    "direct_trial_patch_residual",
    "interpolated_trial_patch_residual",
]


def evaluate_trial_at_points(network, points: jax.Array) -> tuple[jax.Array, jax.Array]:
    """``(values, gradients)`` of a direct trial ``network`` at physical points.

    Parameters
    ----------
    network : Callable[[jax.Array], jax.Array]
        A single-point scalar evaluator ``x: (d,) -> u_theta(x): ()``
        (``models.ansatz``/``models.boundary.ConstrainedAnsatz``'s
        convention).
    points : jax.Array
        Shape ``(*batch, 1, d)``.

    Returns
    -------
    values : jax.Array
        Shape ``(*batch, 1, 1)``.
    gradients : jax.Array
        Shape ``(*batch, 1, d)``.
    """
    flat_points = points.reshape(-1, points.shape[-1])
    flat_values, flat_gradients = jax.vmap(jax.value_and_grad(network))(flat_points)
    values = flat_values.reshape(*points.shape[:-1], 1)
    gradients = flat_gradients.reshape(*points.shape[:-2], 1, points.shape[-1])
    return values, gradients


def _residual_pointwise(
    test_values: jax.Array,
    test_gradients: jax.Array,
    trial_values: jax.Array,
    trial_gradients: jax.Array,
    coefficients: EllipticCoefficientValues,
) -> jax.Array:
    """``r_h(w, v)`` pointwise integrand, broadcasting ``test_*`` against ``trial_*``."""
    diffusion_term = test_gradients @ coefficients.diffusion @ jnp.swapaxes(
        trial_gradients, -1, -2
    )
    advection_term = test_values * (
        trial_gradients @ jnp.swapaxes(coefficients.advection, -1, -2)
    )
    reaction_term = coefficients.reaction * test_values * trial_values
    load_term = coefficients.source * test_values
    return diffusion_term + advection_term + reaction_term - load_term


def _scatter_to_dofs(
    local_vectors: jax.Array, cells_to_dofs: jax.Array, number_of_dofs: int
) -> jax.Array:
    """``(K, N_phi, 1) -> (N_dofs, 1)`` scatter-add (mirrors ``assemble_linear_form``)."""
    flat_indices = cells_to_dofs.reshape(-1)
    flat_values = local_vectors.reshape(-1)
    global_vector = jnp.zeros((number_of_dofs,), dtype=flat_values.dtype)
    global_vector = global_vector.at[flat_indices].add(flat_values)
    return global_vector[:, None]


def direct_trial_fe_residual(network, fe_test_space: FETestSpace) -> jax.Array:
    """``r_h(u_theta, psi_i))_i`` for every FE test/trial DOF ``i``, direct trial.

    Returns
    -------
    jax.Array
        Shape ``(N_dofs, 1)``; restrict to ``fe_test_space.free_dofs`` for
        the interior test space ``V_h subset H^1_0``.
    """
    basis = fe_test_space.basis
    trial_values, trial_gradients = evaluate_trial_at_points(network, basis.physical_points)
    pointwise = _residual_pointwise(
        basis.values, basis.gradients, trial_values, trial_gradients, fe_test_space.coefficients
    )
    local_vectors = integrate_cellwise(pointwise, basis.physical_weights)
    return _scatter_to_dofs(
        local_vectors, fe_test_space.space.cells_to_dofs, fe_test_space.space.number_of_dofs
    )


def interpolated_trial_fe_residual(w_dofs: jax.Array, fe_test_space: FETestSpace) -> jax.Array:
    """``r_h(Pi_h u_theta, psi_i))_i``: ``stiffness @ w_dofs - load``, shape ``(N_dofs, 1)``."""
    return fe_test_space.stiffness @ w_dofs - fe_test_space.load


def direct_trial_patch_residual(
    network, patch_space: PatchTestSpace, problem: EllipticPDE
) -> jax.Array:
    """``r_h(u_theta, varphi_i))_i`` for every patch ``i``, direct trial.

    Returns
    -------
    jax.Array
        Shape ``(N_patch,)``.
    """
    patch_quadrature = gather_patch_quadrature(patch_space)
    batch_shape = patch_quadrature.physical_points.shape[:-2]
    trial_values, trial_gradients = evaluate_trial_at_points(
        network, patch_quadrature.physical_points
    )
    coefficients = evaluate_elliptic_coefficients(problem, patch_quadrature.physical_points)
    pointwise = _residual_pointwise(
        patch_quadrature.hat_values,
        patch_quadrature.hat_gradients,
        trial_values,
        trial_gradients,
        coefficients,
    )
    weighted = patch_quadrature.physical_weights * pointwise
    return jnp.sum(weighted.reshape(batch_shape[0], -1), axis=-1)


def interpolated_trial_patch_residual(
    solution: FiniteElementFunction, patch_space: PatchTestSpace, problem: EllipticPDE
) -> jax.Array:
    """``r_h(Pi_h u_theta, varphi_i))_i`` for every patch ``i``, interpolated trial.

    Returns
    -------
    jax.Array
        Shape ``(N_patch,)``.
    """
    evaluation = evaluate_finite_element_function(solution, patch_space.basis)
    mask = patch_space.patch_mask
    safe_cell = jnp.where(mask, patch_space.patch_cells, 0)
    trial_values = evaluation.values[safe_cell]
    trial_gradients = jnp.broadcast_to(
        evaluation.gradients[safe_cell], trial_values.shape[:-1] + (2,)
    )

    patch_quadrature = gather_patch_quadrature(patch_space)
    batch_shape = patch_quadrature.physical_points.shape[:-2]
    coefficients = evaluate_elliptic_coefficients(problem, patch_quadrature.physical_points)
    pointwise = _residual_pointwise(
        patch_quadrature.hat_values,
        patch_quadrature.hat_gradients,
        trial_values,
        trial_gradients,
        coefficients,
    )
    weighted = patch_quadrature.physical_weights * pointwise
    return jnp.sum(weighted.reshape(batch_shape[0], -1), axis=-1)
