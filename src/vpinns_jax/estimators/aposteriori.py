"""Patch-local a posteriori error estimator (math reference Section 4.2/4.3).

Implements the Poisson-case patch-local estimator the math reference derives
in full (``mu = I``, ``beta = 0``, ``sigma = 0``): loss indicators
``eta_{L,i}``, element-residual indicators ``eta_{R,i}`` with the **jump-term
reduction** (Section 4.3: dropped for a direct trial, included for an
interpolated trial), and their combination into the global estimator.

**Scoping decisions**, both documented at the point they're made below:

1. The residual/jump indicators are derived here for constant isotropic
   diffusion (``mu`` locally constant, so ``div(mu grad w) = mu * trace(Hess
   w)`` with no ``grad(mu).grad(w)`` term) and zero advection/reaction --
   exactly the Poisson case the math reference works out (Section 4.2), and
   this library's primary showcase. A fully general ADR strong-form residual
   would need a coefficient gradient (``grad(mu)``) this library's
   ``ScalarCoefficient``/``TensorCoefficient`` protocol does not expose.
2. The interpolated-trial jump indicator sums the *classical* per-cell,
   ``h_E``-scaled flux-jump term (the same quantity jax-fem's own
   ``compute_residual_error_estimator`` uses) over each patch's incident
   cells, rather than the paper's exact ``varphi_i``-weighted, ``h_E^perp``
   facet integral (Section 4.2). Same asymptotic scaling, simpler to
   compute; the shape-regularity constant ``c_2(P_i)`` this simplification
   would need is folded into the reported indicator's overall scale instead
   of applied per-facet.

For an FE test space, jax-fem's own ``diagnostics.compute_residual_error_estimator``
already *is* this estimator for a Galerkin FE solution (Section 4.1's
general, mesh-based row); it applies unchanged to an interpolated VPINN
trial (also a P1 ``FiniteElementFunction``) and is re-exported here for
convenience rather than reimplemented.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax_fem.assembly import integrate_cellwise
from jax_fem.diagnostics import compute_residual_error_estimator
from jax_fem.function import FiniteElementFunction
from jax_fem.reference_cell import ReferenceInterval
from jax_fem.space import create_facet_basis

from vpinns_jax.pde.base import EllipticPDE
from vpinns_jax.test_spaces.patch_space import PatchTestSpace, gather_patch_quadrature

__all__ = [
    "compute_residual_error_estimator",
    "patch_loss_indicators",
    "direct_trial_patch_residual_indicators",
    "interpolated_trial_patch_residual_indicators",
    "global_patch_estimator",
    "patch_indicators_to_cell_indicators",
]

#: Shape-regularity constant c_2(P_i) <= 1/pi on convex patches (math reference Section 4.2).
_SHAPE_REGULARITY_ELEMENT = 1.0 / math.pi


def patch_loss_indicators(residual_patch: jax.Array, self_energy: jax.Array) -> jax.Array:
    r"""``eta_{L,i}(w) = C_h^i |r_h(w, varphi_i)|``, ``C_h^i = (varphi_i, varphi_i)_A^{-1/2}``.

    Squares to exactly the Riesz-loss patch term
    (``losses.riesz.riesz_loss_patch``): ``sum_i eta_{L,i}^2 = L_Riesz(w)``,
    the natural global loss estimator under the Riesz loss (Section 4.1).
    """
    return jnp.abs(residual_patch) / jnp.sqrt(self_energy)


def direct_trial_strong_residual(network, points: jax.Array, source) -> jax.Array:
    r"""``r_i(w) = f + Laplacian(w)`` (Poisson strong residual), for a direct trial.

    Parameters
    ----------
    network : Callable[[jax.Array], jax.Array]
        Single-point scalar trial evaluator.
    points : jax.Array
        Shape ``(*batch, 1, 2)``.
    source : ScalarCoefficient
        ``f(x)``.

    Returns
    -------
    jax.Array
        Shape ``(*batch, 1, 1)``.
    """

    def laplacian_at(x: jax.Array) -> jax.Array:
        return jnp.trace(jax.hessian(network)(x))

    flat_points = points.reshape(-1, points.shape[-1])
    flat_laplacian = jax.vmap(laplacian_at)(flat_points)
    laplacian = flat_laplacian.reshape(*points.shape[:-1], 1)
    return source(points) + laplacian


def direct_trial_patch_residual_indicators(
    network, patch_space: PatchTestSpace, problem: EllipticPDE
) -> jax.Array:
    r"""``eta_{R,i}(w)`` for a direct trial, jump term dropped (Section 4.3).

    .. math::

        \eta_{R,i}(w) = h_i \, c_2(P_i) \, \|\varphi_i^{1/2} r_i(w)\|_{P_i}

    Valid for direct trials with smooth coefficients across a patch's
    internal facets (Section 4.3): ``u_theta`` is smooth, so the normal
    derivative does not jump there and the flux-jump term vanishes exactly.

    Returns
    -------
    jax.Array
        Shape ``(N_patch,)``.
    """
    patch_quadrature = gather_patch_quadrature(patch_space)
    strong_residual = direct_trial_strong_residual(
        network, patch_quadrature.physical_points, problem.source
    )
    weighted_squared = (
        patch_quadrature.physical_weights
        * patch_quadrature.hat_values
        * strong_residual**2
    )
    number_of_patches = patch_quadrature.physical_points.shape[0]
    integral = jnp.sum(weighted_squared.reshape(number_of_patches, -1), axis=-1)
    return patch_space.patch_diameter * _SHAPE_REGULARITY_ELEMENT * jnp.sqrt(integral)


def _cellwise_facet_jump_term(
    solution: FiniteElementFunction, patch_space: PatchTestSpace, problem: EllipticPDE,
    facet_quadrature_degree: int = 3,
) -> jax.Array:
    """``0.5 h_E ||[A grad(w).n]||^2_{L2(E)}`` per cell, mirroring
    ``jax_fem.diagnostics.estimator``'s own (private) facet-jump term."""
    facet_quadrature = ReferenceInterval().create_quadrature(facet_quadrature_degree)
    facet_basis = create_facet_basis(patch_space.basis, facet_quadrature)
    space = facet_basis.space

    local_plus = solution.dof_values[space.cells_to_dofs[facet_basis.cells_plus]]
    local_minus = solution.dof_values[space.cells_to_dofs[facet_basis.cells_minus]]
    grad_plus = local_plus[:, None, :, :].mT @ facet_basis.gradients_plus
    grad_minus = local_minus[:, None, :, :].mT @ facet_basis.gradients_minus

    diffusion = problem.diffusion(facet_basis.physical_points)
    diffusion_transpose = jnp.swapaxes(diffusion, -1, -2)
    flux_jump = (grad_plus - grad_minus) @ diffusion_transpose
    normal_jump = flux_jump @ facet_basis.normals.mT

    jump_squared_integral = integrate_cellwise(
        normal_jump * normal_jump, facet_basis.physical_weights
    )[:, 0, 0]
    facet_lengths = facet_basis.facet_lengths[:, 0, 0, 0]
    contribution = 0.5 * facet_lengths * jump_squared_integral

    number_of_cells = patch_space.space.cells_to_dofs.shape[0]
    segment_ids = jnp.concatenate((facet_basis.cells_plus, facet_basis.cells_minus))
    contributions = jnp.concatenate((contribution, contribution))
    return jax.ops.segment_sum(contributions, segment_ids, num_segments=number_of_cells)


def interpolated_trial_patch_residual_indicators(
    solution: FiniteElementFunction,
    patch_space: PatchTestSpace,
    problem: EllipticPDE,
    *,
    facet_quadrature_degree: int = 3,
) -> jax.Array:
    r"""``eta_{R,i}(w)`` for an interpolated trial, jump term included (Section 4.3).

    Element-residual part as ``direct_trial_patch_residual_indicators``
    (evaluated on ``solution``'s piecewise-linear values/gradients), plus a
    patch-aggregated flux-jump part (see module docstring's scoping note 2)
    combined as ``sqrt(element_part^2 + jump_part^2)``.

    Returns
    -------
    jax.Array
        Shape ``(N_patch,)``.
    """
    # P1: div(mu grad(w)) == 0 pointwise (piecewise-constant gradient), so the
    # element residual is just the source term evaluated pointwise -- no need
    # to evaluate `solution` itself for this part.
    patch_quadrature = gather_patch_quadrature(patch_space)
    strong_residual = problem.source(patch_quadrature.physical_points)
    weighted_squared = (
        patch_quadrature.physical_weights
        * patch_quadrature.hat_values
        * strong_residual**2
    )
    number_of_patches = patch_quadrature.physical_points.shape[0]
    element_integral = jnp.sum(weighted_squared.reshape(number_of_patches, -1), axis=-1)
    element_part = patch_space.patch_diameter * _SHAPE_REGULARITY_ELEMENT * jnp.sqrt(
        element_integral
    )

    cell_jump_sq = _cellwise_facet_jump_term(
        solution, patch_space, problem, facet_quadrature_degree
    )
    safe_cell = jnp.where(patch_space.patch_mask, patch_space.patch_cells, 0)
    gathered = cell_jump_sq[safe_cell] * patch_space.patch_mask
    jump_part = patch_space.patch_diameter * jnp.sqrt(jnp.sum(gathered, axis=-1))

    return jnp.sqrt(element_part**2 + jump_part**2)


def global_patch_estimator(
    loss_indicators: jax.Array, residual_indicators: jax.Array
) -> jax.Array:
    r"""``eta = sqrt(sum_i (eta_{L,i}^2 + eta_{R,i}^2))`` (Section 4.2/4.4)."""
    return jnp.sqrt(jnp.sum(loss_indicators**2 + residual_indicators**2))


def patch_indicators_to_cell_indicators(
    patch_space: PatchTestSpace, patch_indicators: jax.Array
) -> jax.Array:
    """Map per-patch indicators onto per-cell indicators for LEB refinement.

    LEB (``jax_fem.mesh.refine_mesh_longest_edge_bisection``) refines
    *cells*, not patches, so the adaptive loop needs cell-level marks. Each
    cell gets the *maximum* indicator among the patches touching it (a cell
    is marked as troubled as its worst attached patch) -- a monotone,
    non-inflating aggregation, unlike summing (which would double-count a
    cell shared by several patches).

    Returns
    -------
    jax.Array
        Shape ``(K,)``.
    """
    number_of_cells = patch_space.space.cells_to_dofs.shape[0]
    safe_cell = jnp.where(patch_space.patch_mask, patch_space.patch_cells, 0)
    broadcast_indicators = jnp.broadcast_to(
        patch_indicators[:, None], patch_space.patch_mask.shape
    )
    flat_cells = safe_cell.reshape(-1)
    flat_indicators = jnp.where(
        patch_space.patch_mask.reshape(-1), broadcast_indicators.reshape(-1), -jnp.inf
    )
    cell_indicators = jax.ops.segment_max(
        flat_indicators, flat_cells, num_segments=number_of_cells
    )
    return jnp.where(jnp.isfinite(cell_indicators), cell_indicators, 0.0)
