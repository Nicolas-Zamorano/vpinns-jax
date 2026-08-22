"""True-solution error evaluators and the estimator efficiency index.

For an **interpolated** trial (a P1 ``FiniteElementFunction``), jax-fem's
own ``diagnostics`` error functionals apply directly and are re-exported
here unchanged. For a **direct** trial (a raw callable, not a member of any
fixed FE space), this module adds quadrature-based analogues built on
``losses.residual.evaluate_trial_at_points`` and any ``CellBasis`` (e.g.
``test_spaces.fe_space.FETestSpace.basis``, reused purely as a convenient,
already-built quadrature grid -- the direct trial need not actually belong
to that FE space).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax_fem.assembly import integrate_cellwise
from jax_fem.diagnostics import (
    compute_energy_error,
    compute_h1_seminorm_error,
    compute_l2_error,
    compute_relative_l2_error,
)
from jax_fem.space import CellBasis

from vpinns_jax.losses.residual import evaluate_trial_at_points
from vpinns_jax.pde.base import EllipticPDE, energy_reaction_coefficient

__all__ = [
    "compute_l2_error",
    "compute_relative_l2_error",
    "compute_h1_seminorm_error",
    "compute_energy_error",
    "direct_trial_l2_error",
    "direct_trial_relative_l2_error",
    "direct_trial_h1_seminorm_error",
    "direct_trial_energy_error",
    "efficiency_index",
]


def direct_trial_l2_error(network, basis: CellBasis, problem: EllipticPDE) -> jax.Array:
    """``||u - w||_{L2(Omega)}``, absolute, for a direct trial (quadrature-based)."""
    values, _ = evaluate_trial_at_points(network, basis.physical_points)
    exact = problem.exact_solution(basis.physical_points)
    difference_squared = (values - exact) ** 2
    return jnp.sqrt(jnp.sum(integrate_cellwise(difference_squared, basis.physical_weights)))


def direct_trial_relative_l2_error(
    network, basis: CellBasis, problem: EllipticPDE
) -> jax.Array:
    """``||u - w||_{L2} / ||u||_{L2}``, for a direct trial."""
    absolute = direct_trial_l2_error(network, basis, problem)
    exact = problem.exact_solution(basis.physical_points)
    exact_norm = jnp.sqrt(jnp.sum(integrate_cellwise(exact * exact, basis.physical_weights)))
    return absolute / exact_norm


def direct_trial_h1_seminorm_error(
    network, basis: CellBasis, problem: EllipticPDE
) -> jax.Array:
    """``|u - w|_{H1} = ||grad(u - w)||_{L2}``, for a direct trial."""
    _, gradients = evaluate_trial_at_points(network, basis.physical_points)
    exact_gradient = problem.exact_gradient(basis.physical_points)
    difference = gradients - exact_gradient
    difference_squared = difference @ jnp.swapaxes(difference, -1, -2)
    return jnp.sqrt(jnp.sum(integrate_cellwise(difference_squared, basis.physical_weights)))


def direct_trial_energy_error(network, basis: CellBasis, problem: EllipticPDE) -> jax.Array:
    """``||u - w||_A``, for a direct trial (math reference Section 1's energy norm)."""
    values, gradients = evaluate_trial_at_points(network, basis.physical_points)
    exact_values = problem.exact_solution(basis.physical_points)
    exact_gradient = problem.exact_gradient(basis.physical_points)
    value_difference = values - exact_values
    gradient_difference = gradients - exact_gradient

    diffusion = problem.diffusion(basis.physical_points)
    diffusion_term = (
        gradient_difference @ diffusion @ jnp.swapaxes(gradient_difference, -1, -2)
    )
    symmetrized_reaction = energy_reaction_coefficient(problem)(basis.physical_points)
    reaction_term = symmetrized_reaction * value_difference * value_difference

    return jnp.sqrt(
        jnp.sum(integrate_cellwise(diffusion_term + reaction_term, basis.physical_weights))
    )


def efficiency_index(estimator, energy_error) -> jax.Array:
    r"""``Eff = eta / ||u - w||_A`` (implementation plan Section 7).

    Reliability/efficiency of the a posteriori estimator keep this ratio
    bounded away from ``0``/``infinity`` (ideally close to ``1``).
    """
    return jnp.asarray(estimator) / jnp.asarray(energy_error)
